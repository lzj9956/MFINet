import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# === 1. 读取图像 ===
img_path = "./img.png"  # 替换为上传图路径
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 默认是 BGR，转为 RGB 便于可视化

# === 2. 转为 Tensor 并归一化 ===
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1, 3, H, W]

# === 3. 下采样到 GFIM 输入分辨率（如 1/8）===
H, W = img.shape[:2]
target_size = (H // 8, W // 8)
downsampled = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)

# === 4. 可视化原图与稠密图 ===
downsampled_np = downsampled.squeeze().permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Semantic Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Dense Low-Resolution Feature Map")
plt.imshow(downsampled_np)
plt.axis('off')

plt.tight_layout()
plt.show()
