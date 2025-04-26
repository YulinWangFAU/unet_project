import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model.simple_unet import SimpleUNet
from sampling_mask import generate_line_mask

def apply_ifft(k_space):
    image = np.fft.ifft2(k_space)
    image = np.abs(image)
    return image

# ✅ 加入图像归一化函数
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

# ===================== Step 1: 模拟数据 =====================
shape = (128, 128)
k_space_full = np.random.randn(*shape) + 1j * np.random.randn(*shape)

# Target: 还原图像
target_image = normalize(apply_ifft(k_space_full))
target_tensor = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Input: 欠采样图像
mask = generate_line_mask(shape, acceleration=4)
k_space_under = k_space_full * mask
input_image = normalize(apply_ifft(k_space_under))
input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ===================== Step 2: 定义模型和训练工具 =====================
model = SimpleUNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===================== Step 3: 训练 =====================
n_epochs = 50
losses = []

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")

# 输出的值范围检查（调试用）
print("Output stats:", output.min().item(), output.max().item(), output.mean().item())

# ===================== Step 4: 可视化 =====================
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(input_tensor.squeeze().numpy(), cmap='gray')
plt.title("Input (Under-sampled)")

plt.subplot(1, 3, 2)
plt.imshow(output.detach().squeeze().numpy(), cmap='gray')
plt.title("Output (Predicted)")

plt.subplot(1, 3, 3)
plt.imshow(target_tensor.squeeze().numpy(), cmap='gray')
plt.title("Target (Full)")

plt.tight_layout()
plt.show()

# Loss 曲线
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
