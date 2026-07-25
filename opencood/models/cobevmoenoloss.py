"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix


class KernelGenerator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=128, in_channels=128, kernel_size=3):
        super(KernelGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        # 使用转置卷积将潜在空间上采样到卷积核形状
        self.deconv = nn.ConvTranspose2d(latent_dim, out_channels * in_channels, kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, latent):
        # 输入 latent: (B, latent_dim)
        # 扩展为 4D 张量以适应转置卷积: (B, latent_dim, 1, 1)
        latent = latent.unsqueeze(-1).unsqueeze(-1)
        # 通过转置卷积生成卷积核: (B, out_channels * in_channels, kernel_size, kernel_size)
        kernel = self.deconv(latent)
        # 重塑为目标卷积核形状: (B, out_channels, in_channels, kernel_size, kernel_size)
        kernel = kernel.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return kernel

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MoE(nn.Module):
    def __init__(self, kernel_size, num_experts, batch_size):
        super(MoE, self).__init__()
        self.nums = num_experts
        self.batch_size = batch_size
        self.gate = nn.Linear(128, self.nums)

    def forward(self, x, kernel_f, mask):
        batch_s = x.size(0)  # 获取批量大小 B
        kernel_f = [k.view(batch_s, 128, 128, 3, 3) for k in kernel_f]
        # 定义 single_sample_conv 函数，处理单个样本的卷积
        def single_sample_conv(x_single, weights):
            # x_single: (128, H, W)
            # weights: (num_experts, 128, 128, 3, 3)
            outputs = []
            for i in range(weights.shape[0]):
                weight = weights[i]  # (128, 128, 3, 3)
                output = F.conv2d(x_single.unsqueeze(0), weight, padding=1)
                outputs.append(output.squeeze(0))
            return torch.stack(outputs, dim=0)  # (num_experts, 128, H, W)

        # 用 for 循环替代 vmap，处理每个样本
        expert_outputs_list = []
        for b in range(batch_s):
            x_single = x[b]  # 提取单个样本，形状 (128, H, W)
            # 为当前样本收集所有专家的卷积核
            weights_b = torch.stack([kernel_f[i][b] for i in range(self.nums)], dim=0)  # (num_experts, 128, 128, 3, 3)
            expert_out = single_sample_conv(x_single, weights_b)  # (num_experts, 128, H, W)
            expert_outputs_list.append(expert_out)
        expert_outputs = torch.stack(expert_outputs_list, dim=0)  # (B, num_experts, 128, H, W)
        expert_outputs = rearrange(expert_outputs, 'b l c h w  -> b c h w l')
        # 门控网络部分保持不变
        gate_input = x.mean(dim=[2, 3])  # (B, 128)
        gate_output = self.gate(gate_input)  # (B, num_experts)
        mask = mask.float()
        gate_output = gate_output * mask + (1 - mask) * (-1e9)
        gate_output = torch.softmax(gate_output, dim=-1)  # (B, num_experts)

        # 加权求和
        x_m = torch.sum(gate_output.view(batch_s, 1, 1, 1, -1) * expert_outputs, dim=-1)  # (B, 128, H, W)

        return x_m


class CoMoE(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, max_cavs):
        super(CoMoE, self).__init__()
        # 使用单个 MLP，输出维度为卷积核大小
        #self.mlp1 = MLP(input_size=input_size, hidden_dim=hidden_dim, output_dim=128 * 128 * 3 * 3)
        #self.mlplist = nn.ModuleList([self.mlp1 for _ in range(max_cavs)])
        self.mlplist = nn.ModuleList([MLP(input_size, hidden_dim, output_dim) for _ in range(max_cavs)])
        self.moe = MoE(kernel_size=3, num_experts=max_cavs, batch_size=2)
        self.kernel_generator = KernelGenerator()

    def forward(self, x, f, mask):
        # 对输入特征 f 进行池化
        f_pooled = F.avg_pool3d(f, kernel_size=8, stride=8)  # (B, max_cavs, 128, 4, 4)
        split_f = torch.unbind(f_pooled, dim=1)  # List of (B, 128, 4, 4)
        latent_f = [mlp(fu) for mlp, fu in zip(self.mlplist, split_f)]  # List of (B, 128*128*3*3)
        kernel_f = [self.kernel_generator(latent) for latent in latent_f ]
        # MoE 前向传播
        x_m = self.moe(x, kernel_f, mask)
        x = x + x_m  # 残差连接
        return x



class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, max_cavs):
        super(FusionModule, self).__init__()
        self.max_cavs = max_cavs
        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, (max_cavs+1)*in_channels, 1, 1)
            nn.Conv2d((max_cavs+1)*in_channels, (max_cavs+1)*in_channels//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d((max_cavs+1)*in_channels//4, max_cavs+1, kernel_size=1),
            nn.Sigmoid()
        )
        # 1x1 卷积融合
        self.conv = nn.Conv2d((max_cavs+1)*in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, max_cavs+1, C, H, W)
        B, num_agents, C, H, W = x.shape
        # 重塑为 (B, (max_cavs+1)*C, H, W) 以进行通道注意力
        x_reshaped = x.view(B, num_agents*C, H, W)
        # 计算注意力权重: (B, max_cavs+1, 1, 1)
        attn_weights = self.se(x_reshaped)
        # 重塑权重为 (B, max_cavs+1, 1, 1, 1)
        attn_weights = attn_weights.view(B, num_agents, 1, 1, 1)
        # 应用注意力权重
        x_weighted = x * attn_weights  # (B, max_cavs+1, C, H, W)
        # 重塑加权特征为卷积输入
        x_weighted_reshaped = x_weighted.view(B, num_agents*C, H, W)  # (B, (max_cavs+1)*C, H, W)
        # 1x1 卷积融合
        x_fused = self.conv(x_weighted_reshaped)  # (B, out_channels, H, W)
        x_fused = self.bn(x_fused)
        x_fused = self.relu(x_fused)
        return x_fused

class CoBEVMoE1(nn.Module):
    def __init__(self, config):
        super(CoBEVMoE1, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(config['fax_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

        self.comoe=CoMoE(input_size=256, hidden_dim=512, output_dim=128, max_cavs=self.max_cav)
        self.fusion_module = FusionModule(in_channels=128, out_channels=128, max_cavs=self.max_cav)


    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.fax(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)

        # compressor
        if self.compression:
            x = self.naive_compressor(x)

        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x = self.sttf(x, transformation_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        x = rearrange(x, 'b l h w c -> b l c h w')

        fsin = x#1108
        x = self.fusion_net(x, com_mask)
        x = self.comoe(x,fsin, mask)
        x = x.unsqueeze(1)

        # 使用可学习融合模块
        x = torch.cat([fsin, x], dim=1)  # (B, max_cavs+1, 128, H, W)
        x = self.fusion_module(x)  # (B, 128, H, W)
        x = x.unsqueeze(1)  # (B, 1, 128, H, W)


        # x = torch.mean(x, dim=1)
        # x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)
        return output_dict