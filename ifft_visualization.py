import numpy as np
import matplotlib.pyplot as plt

# Step 1: 创建一个模拟完整的k-space
np.random.seed(0)  # 保证结果可复现
image_size = 128

# 随机生成复数k-space数据（实部+虚部）
k_space_full = np.random.randn(image_size, image_size) + 1j * np.random.randn(image_size, image_size)

# Step 2: 完整IFFT恢复成图像
image_full = np.fft.ifft2(k_space_full)
image_full = np.abs(image_full)

# Step 3: 制作欠采样mask（简单版：只保留一半行）
mask = np.zeros((image_size, image_size))
mask[::2, :] = 1  # 每隔一行保留

# 应用mask，生成欠采样k-space
k_space_under = k_space_full * mask

# Step 4: 欠采样IFFT恢复图像
image_under = np.fft.ifft2(k_space_under)
image_under = np.abs(image_under)

# Step 5: 可视化对比
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(np.log(1 + np.abs(k_space_full)), cmap='gray')
axs[0, 0].set_title('Full k-space')

axs[0, 1].imshow(np.log(1 + np.abs(k_space_under)), cmap='gray')
axs[0, 1].set_title('Under-sampled k-space')

axs[1, 0].imshow(image_full, cmap='gray')
axs[1, 0].set_title('Reconstructed Image (Full)')

axs[1, 1].imshow(image_under, cmap='gray')
axs[1, 1].set_title('Reconstructed Image (Under-sampled)')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
