import os
import re
import cv2
import argparse
import numpy as np
from typing import List, Tuple

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic_path", type=str, required=True, help="Path to dynamic folder (contains test_vis)")
    parser.add_argument("--static_path",  type=str, required=True, help="Path to static folder (contains test_vis)")
    parser.add_argument("--output_path",  type=str, required=True, help="Output folder for videos")
    parser.add_argument("--image_width",  type=int, default=800, help="single tile width (default 800)")
    parser.add_argument("--image_height", type=int, default=600, help="single tile height (default 600)")
    parser.add_argument("--fps",          type=int, default=10,   help="video FPS")
    parser.add_argument("--codec",        type=str, default="mp4v", help="fourcc codec (e.g., mp4v, avc1, H264)")
    parser.add_argument("--indices",      type=str, default="",
                        help='Frame indices to include, e.g. "0,5-20,100-120". Leave empty = all.')
    parser.add_argument("--cols",         type=str, default="",
                        help='Optional: which column indices to export, e.g. "0,1,4,5". Leave empty = all detected cols.')
    parser.add_argument("--overwrite",    action="store_true", help="Overwrite existing mp4 files if present")
    return parser.parse_args()

def natural_key(s: str):
    # "frame_2.png" < "frame_10.png"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def parse_indices(spec: str) -> List[int]:
    """Parse '0,5-20,100-120' into list of ints (dedup + sorted)."""
    if not spec:
        return []  # empty means: no restriction
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            if a <= b:
                out.update(range(a, b + 1))
            else:
                out.update(range(b, a + 1))
        else:
            out.add(int(part))
    return sorted(out)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def open_writers(base_out: str, n_cols: int, w: int, h: int, fps: int, fourcc: str, overwrite: bool, col_filter: List[int]):
    ensure_dir(base_out)
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writers = {}
    selected_cols = range(n_cols) if not col_filter else col_filter
    for c in selected_cols:
        out_file = os.path.join(base_out, f"col_{c}.mp4")

        if os.path.exists(out_file) and not overwrite:
            raise FileExistsError(f"Output exists: {out_file} (use --overwrite to replace)")
        writers[c] = cv2.VideoWriter(out_file, fourcc_code, fps, (w, h))
    return writers

def close_writers(writers: dict):
    for w in writers.values():
        w.release()

def main():
    opt = arg_parser()

    dyn_dir = os.path.join(opt.dynamic_path, "test_vis")
    sta_dir = os.path.join(opt.static_path,  "test_vis")
    #out_dir = opt.output_path
    out_dir = os.path.join(opt.output_path, opt.indices)
    ensure_dir(out_dir)

    # List and align by filename
    dyn_files = sorted([f for f in os.listdir(dyn_dir) if not f.startswith(".")], key=natural_key)
    sta_files = sorted([f for f in os.listdir(sta_dir) if not f.startswith(".")], key=natural_key)
    #assert len(dyn_files) == len(sta_files), "dynamic/static frame counts differ."

    # If indices are given, subselect
    idx_list = parse_indices(opt.indices)
    if idx_list:
        # Filter to given indices (skip oob safely)
        pairs = [(i, dyn_files[i], sta_files[i]) for i in idx_list if 0 <= i < len(dyn_files)]
    else:
        pairs = list(enumerate(zip(dyn_files, sta_files)))
        pairs = [(i, a, b) for i, (a, b) in pairs]

    if not pairs:
        print("No frames selected. Check --indices.")
        return

    # Peek first static frame to infer number of columns if user didn't pass --cols
    first_sta = cv2.imread(os.path.join(sta_dir, pairs[0][2]))
    if first_sta is None:
        raise RuntimeError("Failed to read first static frame.")
    H, W, _ = first_sta.shape
    n_cols = W // opt.image_width
    assert opt.image_height == H, f"Given image_height={opt.image_height} but frame H={H}."
    assert n_cols >= 1, "Detected zero columns; check image_width."

    col_filter = [int(x) for x in opt.cols.split(",")] if opt.cols.strip() else []

    # Prepare per-column writers
    writers = open_writers(
        base_out=out_dir,
        n_cols=n_cols,
        w=opt.image_width,
        h=opt.image_height,
        fps=opt.fps,
        fourcc=opt.codec,
        overwrite=opt.overwrite,
        col_filter=col_filter
    )

    # Process frames
    for idx, dyn_name, sta_name in pairs:
        dyn_path = os.path.join(dyn_dir, dyn_name)
        sta_path = os.path.join(sta_dir, sta_name)

        dynamic_fig = cv2.imread(dyn_path, 0)            # gray
        static_fig  = cv2.imread(sta_path, cv2.IMREAD_COLOR)  # color

        if dynamic_fig is None or static_fig is None:
            print(f"[WARN] skip idx {idx}: read fail: {dyn_name}, {sta_name}")
            continue

        # --- apply your original overlay to the static frame ---
        w = opt.image_width
        # dynamic masks (GT: col 4, Pred: col 5)
        if n_cols >= 5:
            dynamic_gt = dynamic_fig[:, 4*w:5*w].copy()
            dynamic_gt[dynamic_gt > 0] = 1
        else:
            dynamic_gt = None

        if n_cols >= 6:
            dynamic_pred = dynamic_fig[:, 5*w:].copy()
            dynamic_pred[dynamic_pred > 0] = 1
        else:
            dynamic_pred = None

        # extract static GT/Pred regions
        if dynamic_gt is not None:
            static_gt = static_fig[:, 4*w:5*w].copy()
            static_gt[dynamic_gt == 1] = np.array([255, 255, 255], dtype=np.uint8)
            static_fig[:, 4*w:5*w] = static_gt
        if dynamic_pred is not None:
            static_pred = static_fig[:, 5*w:].copy()
            static_pred[dynamic_pred == 1] = np.array([255, 255, 255], dtype=np.uint8)
            static_fig[:, 5*w:] = static_pred

        # --- split the (already overlaid) static figure by columns and write to per-col videos ---
        selected_cols = writers.keys()
        for c in selected_cols:
            x0 = c * w
            x1 = (c + 1) * w if c < n_cols - 1 else static_fig.shape[1]
            tile = static_fig[:, x0:x1]
            # enforce exact width (should be exact if stitched), else resize safely
            if tile.shape[1] != w:
                tile = cv2.resize(tile, (w, opt.image_height), interpolation=cv2.INTER_AREA)
            writers[c].write(tile)

    close_writers(writers)
    print(f"Done. Videos saved to: {out_dir}")

if __name__ == "__main__":
    main()
