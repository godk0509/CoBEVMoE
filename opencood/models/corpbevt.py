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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()  # 展平输入数据
        self.fc1 = nn.Linear(input_size, hidden_dim)  # 输入到隐藏层
        self.relu = nn.ReLU()  # ReLU激活
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出

    def forward(self, x):
        x = self.flatten(x)  # 将输入展平
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.relu(x)  # 激活
        x = self.fc2(x)  # 第二个全连接层到输出
        return x

class MoE(nn.Module):
    def __init__(self, kernel_size, num_experts):
        super(MoE, self).__init__()
        #custom_kernel = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
        #self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.core_conv = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3, padding=1, bias=False)
        #self.layer_norm = nn.LayerNorm([3, 3])
        # 用 custom_kernel 初始化卷积层的权重
        self.nums=num_experts
        self.experts = nn.ModuleList([nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False) for _ in range(self.nums)])

        # 门控网络，输出为每个专家的权重
        self.gate = nn.Linear(128, self.nums)

    def forward(self, x, f):
        """
        i = 0
        for expert in self.experts:
            k = f[i].reshape(1, 1, 3, 3)
            with torch.no_grad():  # 禁用梯度计算以避免影响权重更新
                expert.weight.copy_(k)
            i += 1

        1:128channel add another conv to learn
        2:kernel normalize
        3:batches
        """
        for k, expert in zip(f, self.experts):
            #k = k.reshape(1, 1, 3, 3)
            k = k.reshape(1, 2, 9)
            #k = k.repeat(1, 2, 1)
            k_1 =self.core_conv(k)
            k_1 = k_1.reshape(1,128,3,3)


            #k_1 = self.layer_norm(k_1)#0324

            #expanded_k = k.repeat(128, 128, 1, 1)
            with torch.no_grad():  # 禁用梯度计算以避免影响权重更新
                expert.weight.copy_(k_1)

        #for idx, conv in enumerate(self.experts):
        #    print(f"Conv Layer {idx} weights:\n{conv.weight}\n")
        '''
                for j in range(5):
            print(self.experts[j].weight)
            print("hhh")
        for i in range(self.nums):
            k = f[i].reshape(1, 1, 3, 3)
            with torch.no_grad():  # 禁用梯度计算以避免影响权重更新
                self.experts[i].weight.copy_(k)
            print(self.experts[i].weight)
        
        i=0
        for k, expert in zip(f, self.experts):

            k = k.reshape(1, 1, 3, 3)
            with torch.no_grad():  # 禁用梯度计算以避免影响权重更新
                expert.weight.copy_(k)
            print(self.experts[i].weight)
            i=i+1   
                
        '''


        batch_s, channels, _, _ = x.size()

        # 计算门控网络的输出
        gate_input = x.mean(dim=[2, 3])

        gate_output = torch.softmax(self.gate(gate_input), dim=-1)

        # 对每个专家的输出加权求和
        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=-1)  # 维度 [2, 1, 32, 256, 256, num_experts]

        x = torch.sum(gate_output.reshape(batch_s, 1, 1, 1, -1) * expert_outputs,
                      dim=-1)  # 维度 [2, 1, 32, 256, 256]#0828

        #return x
        return x ,expert_outputs

class CoMoE(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, max_cavs):
        super(CoMoE, self).__init__()
        self.mlp1 = MLP(input_size=input_size, hidden_dim=hidden_dim, output_dim=output_dim)
        self.mlplist = nn.ModuleList([self.mlp1 for _ in range(max_cavs)])
        self.moe = MoE(kernel_size=3, num_experts=max_cavs)


    def forward(self, x, f):
        f_pooled = F.avg_pool3d(f, kernel_size=8, stride=8)
        split_f = torch.unbind(f_pooled, dim=1)
        kernel_f = ([mlp(fu) for mlp,fu in zip(self.mlplist,split_f)])
        x_m = self.moe(x, kernel_f)
        x_m ,experts = self.moe(x, kernel_f)
        x = x + x_m
        #return x
        return x ,experts ,x_m

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


class CorpBEVT(nn.Module):
    def __init__(self, config):
        super(CorpBEVT, self).__init__()
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

        self.comoe=CoMoE(input_size=256, hidden_dim=512, output_dim=9, max_cavs=self.max_cav)
        """
        #MLP
        input_shape = (2, 16, 4, 4)  # 输入的形状
        input_size = input_shape[1] * input_shape[2] * input_shape[3]  # 计算展平后的输入大小
        hidden_dim = 512  # 隐藏层的维度，可以根据需求调整
        output_dim = 3 * 3  # 输出维度 (3, 3)，所以展平为9
        self.mlp = MLP(input_size=input_size, hidden_dim=hidden_dim, output_dim=output_dim)
        """


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

        """
        x_pooled = F.avg_pool3d(x, kernel_size=8, stride=8)
        split_x = torch.unbind(x_pooled, dim=1)

        y = mlp(split_x)
        AvgPool3d

        custom_kernel = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)

        # 用 custom_kernel 初始化卷积层的权重
        with torch.no_grad():  # 禁用梯度计算以避免影响权重更新
            conv.weight.copy_(custom_kernel)
        """
        fsin = x#1108
        x = self.fusion_net(x, com_mask)
        fcol =x
        #x = self.comoe(x,fsin)
        #x, experts_out, moe_out= self.comoe(x,fsin)#1108
        x = x.unsqueeze(1)

        #x = torch.cat([fsin, x], dim=1)#contact 0511

        fcol = fcol.unsqueeze(1) #0219 attention fusion
        x = torch.cat([fsin, fcol], dim=1) #0219

        x = torch.mean(x, dim=1)
        x = x.unsqueeze(1)

        # dynamic head4444444444
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)
        #output_dict._dict_['expert'] = experts_out
        #output_dict._dict_['moe'] = moe_out
        #moe_dict = {'expert': experts_out, 'moe': fcol }
        #output_dict.update(moe_dict)
        return output_dict