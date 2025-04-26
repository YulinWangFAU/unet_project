import torch
import numpy as np
import matplotlib.pyplot as plt
from model.simple_unet import SimpleUNet
from sampling_mask import generate_line_mask

def apply_ifft(k_space):
    image = np.fft.ifft2(k_space)
    image = np.abs(image)
    return image

# Step 1: 模拟完整 k-space 数据（复数）
shape = (128, 128)
k_space = np.random.randn(*shape) + 1j * np.random.randn(*shape)

# Step 2: 应用 line 欠采样 mask
mask = generate_line_mask(shape, acceleration=2)
k_space_under = k_space * mask

# Step 3: IFFT 得到欠采样图像，作为网络输入
image_input = apply_ifft(k_space_under)
input_tensor = torch.tensor(image_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

# Step 4: 模型构建并前向传播
model = SimpleUNet()
output = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)

# Step 5: 可视化输入/输出图像
output_image = output.detach().squeeze().numpy()

plt.subplot(1, 2, 1)
plt.imshow(image_input, cmap='gray')
plt.title("Input (Under-sampled)")

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title("Output (U-Net Result)")

plt.tight_layout()
plt.show()
