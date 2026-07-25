import torch
import torch.nn as nn

from einops import rearrange

import torch.nn.functional as F
import torch


def improved_triplet_loss(expert_outputs, fused_feature, mask, margin=1.0, neg_weight=1.0):
    """
    改进版三元组损失:
    - 每个 expert 输出接近 fused_feature
    - 每个 expert 与其他 expert 尽可能远离（取最相似的负样本 - hard negative）

    Args:
        expert_outputs: (B, C, H, W, num_experts)
        fused_feature:  (B, C, H, W)
        mask:           (B, num_experts)  有效 expert 标志
        margin: float, 默认 1.0
        neg_weight: float, 负样本项的权重
    Returns:
        loss: 标量
    """
    B, C, H, W, num_experts = expert_outputs.shape
    D = C * H * W

    # flatten
    expert_flat = expert_outputs.view(B, D, num_experts)  # (B, D, num_experts)
    #fused_flat = fused_feature.view(B, D, 1).expand(-1, -1, num_experts)  # (B, D, num_experts)
    fused_flat = fused_feature.reshape(B, D, 1).expand(-1, -1, num_experts)
    # 有效 expert mask
    mask = mask.float()  # (B, num_experts)
    valid_expert_mask = mask > 0  # (B, num_experts)

    # 正样本距离: expert 与 fused_feature
    d_pos = torch.mean((expert_flat - fused_flat) ** 2, dim=1)  # (B, num_experts)

    # 负样本距离矩阵: expert_i 和 expert_j
    expert_i = expert_flat.unsqueeze(3)  # (B, D, num_experts, 1)
    expert_j = expert_flat.unsqueeze(2)  # (B, D, 1, num_experts)
    diff_ij = expert_i - expert_j  # (B, D, num_experts, num_experts)
    d_neg_matrix = torch.mean(diff_ij ** 2, dim=1)  # (B, num_experts, num_experts)

    # 去掉对角线（i==j）
    eye = torch.eye(num_experts, device=expert_outputs.device).bool()  # (num_experts, num_experts)
    d_neg_matrix.masked_fill_(eye.unsqueeze(0), float('inf'))  # 对角线无效

    # 取每个 expert 最相似的“负样本”（hard negative）
    d_neg_hard, _ = torch.min(d_neg_matrix, dim=2)  # (B, num_experts)

    # 有效 expert 的 mask
    valid_mask = valid_expert_mask.float()  # (B, num_experts)

    # triplet term: max(0, d_pos - d_neg + margin)
    triplet_term = F.relu(d_pos - d_neg_hard + margin)  # (B, num_experts)

    # 只考虑有效 expert
    triplet_term = triplet_term * valid_mask

    # 平均
    num_valid = torch.sum(valid_mask) + 1e-6
    loss_pos = torch.sum(d_pos * valid_mask) / num_valid
    loss_neg = torch.sum(triplet_term) / num_valid

    total_loss = loss_pos + neg_weight * loss_neg

    return total_loss

def triplet_loss(expert_outputs, f, mask, margin=1.0):
    """
    计算三元损失，以确保专家输出的多样性，同时保持与输入特征的相似性。

    参数：
        expert_outputs (torch.Tensor): 专家输出，形状为 (B, 128, 32, 32, num_experts)
        f (torch.Tensor): attention 特征，形状为 (B, 128, 32, 32)
        mask (torch.Tensor): 掩码，形状为 (B, num_experts)，指示有效的 CAV
        margin (float): 三元损失的边距，默认为 1.0

    返回：
        torch.Tensor: 表示总三元损失的标量张量
    """
    B, C, H, W, num_experts = expert_outputs.shape
    D = C * H * W  # 展平后的特征维度

    # 展平张量以便计算距离
    expert_flat = expert_outputs.contiguous().view(B, D, num_experts)  # (B, D, num_experts)
    #f_flat = f.contiguous().view(B, num_experts, D).permute(0, 2, 1)  # (B, D, num_experts)
    f_flat = f.contiguous().view(B, 1, D).permute(0, 2, 1)
    f_flat = f_flat.repeat(1,1,num_experts)
    # 计算正距离：expert_i(x) 和 f_i 之间的均方误差
    d_pos = torch.mean((expert_flat - f_flat)**2, dim=1)  # (B, num_experts)

    # 计算负距离：expert_i(x) 和 expert_j(x) 之间的均方误差
    diff = expert_flat[:, :, :, None] - expert_flat[:, :, None, :]  # (B, D, num_experts, num_experts)
    d_neg = torch.mean(diff**2, dim=1)  # (B, num_experts, num_experts)

    # 计算三元损失项
    triplet_terms = d_pos[:, :, None] - d_neg + margin  # (B, num_experts, num_experts)
    triplet_terms = torch.clamp(triplet_terms, min=0)  # 取 max(0, ...)

    # 创建有效三元组的掩码：j != i 且 mask[b, j] == 1
    eye = torch.eye(num_experts, device=mask.device).bool()  # (num_experts, num_experts)
    mask_valid = mask[:, None, :] * (~eye[None, :, :]).float()  # (B, num_experts, num_experts)

    # 计算有效三元组的总三元损失
    total_triplet_loss = torch.sum(triplet_terms * mask_valid)

    # 对有效三元组数量取平均值
    num_triplets = torch.sum(mask_valid) + 1e-6  # 避免除以零
    loss = total_triplet_loss / num_triplets

    return loss



class GrokTriLoss(nn.Module):
    def __init__(self, args):
        super(GrokTriLoss, self).__init__()

        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        self.d_coe = args['d_coe']
        self.s_coe = args['s_coe']
        self.target = args['target']

        self.loss_func_static = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_dynamic = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.d_weights]).cuda())

        self.loss_dict = {}


    def forward(self, output_dict, gt_dict):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """

        static_pred = output_dict['static_seg']
        dynamic_pred = output_dict['dynamic_seg']

        expert_loss = torch.tensor(0, device=static_pred.device)

        expert_out = output_dict['expert']
        cav_feature = output_dict['cav_feature']
        fused_feature = output_dict['swap_fusion']
        mask = output_dict['mask']
        # expert_loss = triplet_loss(expert_out, cav_feature, mask, margin=1.0)
        #expert_loss = triplet_loss(expert_out, fused_feature, mask, margin=1.0)
        expert_loss = improved_triplet_loss(expert_out, fused_feature, mask, margin=0.5, neg_weight=1.0)
        static_loss = torch.tensor(0, device=static_pred.device)
        dynamic_loss = torch.tensor(0, device=dynamic_pred.device)

        # during training, we only need to compute the ego vehicle's gt loss
        static_gt = gt_dict['gt_static']
        dynamic_gt = gt_dict['gt_dynamic']
        static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
        dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

        if self.target == 'dynamic':
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)

        elif self.target == 'static':
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        else:
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss

        total_loss = total_loss + 0.5 * expert_loss

        self.loss_dict.update({'total_loss': total_loss,
                               'static_loss': static_loss,
                               'dynamic_loss': dynamic_loss,
                               'expert_loss': expert_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']
        expert_loss = self.loss_dict['expert_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                  " || Dynamic Loss: %.4f || expert Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), static_loss.item(), dynamic_loss.item(), expert_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                                 " || Dynamic Loss: %.4f || expert Loss: %.4f" % (
                                     epoch, batch_id + 1, batch_len,
                                     total_loss.item(), static_loss.item(), dynamic_loss.item(), expert_loss.item()))

        writer.add_scalar('Static_loss', static_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('Expert_loss', expert_loss.item(),
                          epoch * batch_len + batch_id)




