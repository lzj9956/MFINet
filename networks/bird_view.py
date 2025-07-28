import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.vision_transformer import Block, Attention

from . import backbone
from networks.transformer import TransBlock
from utils.util import ChannelAttention, SpatialAttention
from utils.util import DeformConv2d, Depth_wise_separable_conv
from networks.patch import reverse_patches
import pdb


class Merge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(Merge, self).__init__()
        cin = cin_low + cin_high
        self.merge_layer = nn.Sequential(
                    backbone.conv3x3(cin, cin // 2, stride=1, dilation=1),
                    nn.BatchNorm2d(cin // 2),
                    backbone.act_layer,
                    
                    backbone.conv3x3(cin // 2, cout, stride=1, dilation=1),
                    nn.BatchNorm2d(cout),
                    backbone.act_layer
                )
        self.scale_factor = scale_factor
    
    def forward(self, x_low, x_high):
        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        x_merge = torch.cat((x_low, x_high_up), dim=1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)
        x_out = self.merge_layer(x_merge)
        return x_out


class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, size):
        super(AttMerge, self).__init__()
        self.size = size
        self.cout = cout

        self.att_layer = nn.Sequential(
            backbone.conv3x3(2 * cout, cout // 2, stride=1, dilation=1),
            nn.BatchNorm2d(cout // 2),
            backbone.act_layer,
            backbone.conv3x3(cout // 2, 2, stride=1, dilation=1, bias=True)
        )

        self.conv_high = nn.Sequential(
            backbone.conv3x3(cin_high, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )

        self.conv_low = nn.Sequential(
            backbone.conv3x3(cin_low, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        batch_size = x_low.shape[0]
        H = x_low.shape[2]
        W = x_low.shape[3]

        # x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x_high_up = F.upsample(x_high, size=self.size, mode='bilinear', align_corners=False)

        x_merge = torch.stack((self.conv_low(x_low), self.conv_high(x_high_up)), dim=1) #(BS, 2, channels, H, W)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        # attention fusion
        ca_map = self.att_layer(x_merge.view(batch_size, 2*self.cout, H, W))
        ca_map = ca_map.view(batch_size, 2, 1, H, W)
        ca_map = F.softmax(ca_map, dim=1)

        x_out = (x_merge * ca_map).sum(dim=1) #(BS, channels, H, W)
        return x_out

# class Multi_branch(nn.Module):
#     def __init__(self, dim, factor=[64, 64, 64], qkv_bias=False):
#         super().__init__()
#         self.dim = dim
#         self.factor = factor  # 确保 sum(factor) == dim 以保持输出通道一致
#
#         # Channel & Spatial Attention
#         self.ca = ChannelAttention(in_channel=dim)
#         self.sa = SpatialAttention(k=3)
#
#         # Branch 1: Deformable Convolution
#         self.dconv = DeformConv2d(dim, factor[0], kernel_size=3, padding=1)
#         self.dconv_norm = nn.BatchNorm2d(factor[0])
#
#         # Branch 2: Depthwise Separable Convolution
#         self.dwconv = Depth_wise_separable_conv(dim, factor[1])
#         self.dwconv_norm = nn.BatchNorm2d(factor[1])
#
#         # Branch 3: Self-Attention (requires [B, L, C] input)
#         self.attn = Attention(dim, qkv_bias=qkv_bias)
#         self.attn_proj = nn.Linear(dim, factor[2])
#         self.attn_norm = nn.LayerNorm(factor[2])
#
#         # Final projection to match input dim
#         self.out_proj = nn.Conv2d(sum(factor), dim, kernel_size=1)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         # Residual path
#         residual = self.ca(x)  # [B, C, H, W]
#         residual = self.sa(residual)  # [B, C, H, W]
#
#         # Branch 1: Deformable Convolution
#         b1 = self.dconv(x)  # [B, factor[0], H, W]
#         b1 = self.dconv_norm(b1)
#
#         # Branch 2: Depthwise Separable Convolution
#         b2 = self.dwconv(x)  # [B, factor[1], H, W]
#         b2 = self.dwconv_norm(b2)
#
#         # Branch 3: Self-Attention
#         x_flat = rearrange(x, 'b c h w -> b (h w) c')  # [B, H*W, C]
#         b3 = self.attn(x_flat)  # [B, H*W, C]
#         b3 = self.attn_proj(b3)  # [B, H*W, factor[2]]
#         b3 = self.attn_norm(b3)
#         b3 = rearrange(b3, 'b (h w) c -> b c h w', h=H, w=W)  # [B, factor[2], H, W]
#
#         # Concatenate branches
#         out = torch.cat([b1, b2, b3], dim=1)  # [B, sum(factor), H, W]
#
#         # Project to original dim
#         out = self.out_proj(out)  # [B, C, H, W]
#
#         # Add residual
#         out = out + residual
#         return out
#
#
# class Patch_down_sample(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.down_conv = nn.Conv2d(
#             in_channels=dim,
#             out_channels=dim,
#             kernel_size=3,
#             stride=2,
#             padding=1
#         )
#
#     def forward(self, x):
#         """
#         输入: x [B, C, H, W]
#         输出: [B, C, H//2, W//2]（空间分辨率减半，通道数不变）
#         """
#         return self.down_conv(x)

class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5).clone()
        # num_dilation,3,B,C//num_dilation,H,W

        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2).clone()
        # num_dilation, B, H, W, C//num_dilation

        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation

        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C).clone()
        # x = x.permute(1, 2, 3, 0, 4)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BEVNet(nn.Module):
    def __init__(self, base_block, context_layers, layers, use_att):
        super(BEVNet, self).__init__()
        #encoder
        self.header = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1], layers[0], stride=2, dilation=1, use_att=use_att)
        self.res1 = self._make_layer(eval('backbone.{}'.format(base_block)), 64, context_layers[2], layers[1], stride=2, dilation=1, use_att=use_att)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), 128, context_layers[3], layers[2], stride=2, dilation=1, use_att=use_att)
        self.MultiDilatelocalAttention0 = MultiDilatelocalAttention(dim=32, num_heads=8, kernel_size=3, dilation=[1, 2])
        self.MultiDilatelocalAttention1 = MultiDilatelocalAttention(dim=64, num_heads=8, kernel_size=3, dilation=[1, 2])
        self.MultiDilatelocalAttention2 = MultiDilatelocalAttention(dim=128, num_heads=8, kernel_size=3, dilation=[1, 2])

        #decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, size=(150, 150))
        # x_high_up = F.interpolate(x_high, size=(150, 150), mode='bilinear', align_corners=True)
        
        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, size=(300, 300))

        # fusion_channels0 = fusion_channels1 // 2 + context_layers[0]
        # self.up0 = AttMerge(context_layers[0], fusion_channels1 // 2, fusion_channels0 // 2, scale_factor=2)

        self.out_channels = fusion_channels1 // 2

        self.transformer1 = TransBlock(dim=1152)
        # self.transformer1 = TransBlock(dim=512)

        dim = 128
        depth = [1, 5, 3]
        factor = [[256, 256, 256],
                  [192, 192, 384],
                  [48, 48, 672]]

        # self.block = nn.Sequential(
        #     self.build_block(dim, depth=depth[0], factor=factor[0]),
        #     Patch_down_sample(dim),
        #     self.build_block(dim, depth=depth[1], factor=factor[1]),
        #     # Patch_down_sample(dim),
        #     # self.build_block(dim, depth=depth[2], factor=factor[2]),
        #     # Patch_down_sample(dim),
        # )

    # def build_block(self, dim, depth, factor):
    #     return nn.Sequential(*[Multi_branch(dim, factor) for _ in range(depth)])

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        #pdb.set_trace()
        #encoder
        x0 = self.header(x)  # B C H W

        #att_1
        x0_att = x0.permute(0, 2, 3, 1)  #输入B H W C
        x0_att = self.MultiDilatelocalAttention0(x0_att)
        x1_att1 = x0_att.permute(0, 3, 1, 2)

        # Concatenate branches
        out1 = torch.cat([x0, x1_att1], dim=1)  # [B, sum(factor), H, W]

        x1 = self.res1(out1)
        # att_2
        x1_att = x1.permute(0, 2, 3, 1)  # 输入B H W C
        x1_att = self.MultiDilatelocalAttention1(x1_att)
        x2_att2 = x1_att.permute(0, 3, 1, 2)

        # Concatenate branches
        out2 = torch.cat([x1, x2_att2], dim=1)  # [B, sum(factor), H, W]

        x2 = self.res2(out2)
        # att_3
        x2_att = x2.permute(0, 2, 3, 1)  # 输入B H W C
        x2_att = self.MultiDilatelocalAttention2(x2_att)
        x3_att3 = x2_att.permute(0, 3, 1, 2)

        # Concatenate branches
        # out3 = torch.cat([x2, x3_att3], dim=1)  # [B, sum(factor), H, W]

        # x2 = self.block(x2)

        #decoder
        x_merge1 = self.up2(x1, x3_att3)
        x_merge0 = self.up1(x0, x_merge1)
        # x_merge = self.up0(x, x_merge0)
        return x_merge0
