import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2


class DilateAttention(nn.Module):
    "实现 Dilate Attention 机制"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, return_attn=False):
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)
        x = (attn_drop @ v).transpose(1, 2).reshape(B, H, W, d)
        if return_attn:
            return x, attn
        return x


class MultiDilatelocalAttention(nn.Module):
    "多膨胀率局部注意力模块"

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
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} 必须是 dilation 数量{self.num_dilation} 的倍数！"

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
            for i in range(self.num_dilation)
        ])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, H, W, C = x.shape
        x_permuted = x.permute(0, 3, 1, 2)  # B, C, H, W

        qkv = self.qkv(x_permuted).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        x = x_permuted.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)

        attn_maps = []
        for i in range(self.num_dilation):
            if return_attn:
                x[i], attn = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2], return_attn=True)
                attn_maps.append(attn)
            else:
                x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])

        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn_maps
        return x


def visualize_attention_map(attn_map, H, W, head=0, token_idx=0):
    """
    """
    attn = attn_map[0, head, token_idx, 0].detach().cpu().numpy()
    kernel_size = int(np.sqrt(len(attn)))
    attn = attn.reshape(kernel_size, kernel_size)
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.title(f"Head {head}, Token {token_idx}")
    plt.colorbar()
    plt.show()



def overlay_attention_on_image(attn_map, raw_img, H, W, token_idx, head=0, kernel_size=3, alpha=0.5):
    """
    """
    attn = attn_map[0, head, token_idx, 0].detach().cpu().numpy()
    attn = attn.reshape(kernel_size, kernel_size)
    token_y = token_idx // W
    token_x = token_idx % W
    pad = kernel_size // 2

    # 放回原图大小的空图
    heatmap = np.zeros((H, W))
    for i in range(kernel_size):
        for j in range(kernel_size):
            y = token_y + i - pad
            x = token_x + j - pad
            if 0 <= y < H and 0 <= x < W:
                heatmap[y, x] = attn[i, j]

    # 归一化
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap = cv2.resize(heatmap, raw_img.size)

    # 转换为 RGB 图像
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 原图转 numpy
    raw_np = np.array(raw_img)

    # 混合叠加
    overlay = np.uint8((1 - alpha) * raw_np + alpha * heatmap_color)

    # 显示结果
    plt.imshow(overlay)
    plt.title(f"Overlay Attention (Head {head}, Token {token_idx})")
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    B, H, W, C = 1, 16, 16, 64
    x = torch.rand([B, H, W, C]).cuda()
    model = MultiDilatelocalAttention(dim=C).cuda()

    output, attn_maps = model(x, return_attn=True)

    print("输出 shape:", output.shape)  # 应为 [1, 16, 16, 64]

    visualize_attention_map(attn_maps[0], H, W, head=0, token_idx=(H * W) // 2)
