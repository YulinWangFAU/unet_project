### 模型加载 + 重建预测可视化
import torch
import matplotlib.pyplot as plt
from model.simple_unet import SimpleUNet
from dataset_loader import SyntheticMRIDataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("unet_trained.pt", map_location=device))
model.eval()
print("✅ Model loaded.")

# 构造测试数据（这里重新生成新的图像）
test_dataset = SyntheticMRIDataset(num_samples=1, shape=(128, 128), acceleration=4)
input_tensor, target_tensor = test_dataset[0]

# 添加 batch 维度并推理
input_tensor = input_tensor.unsqueeze(0).to(device)
with torch.no_grad():
    output = model(input_tensor)

# 可视化
input_image = input_tensor.cpu().squeeze().numpy()
output_image = output.cpu().squeeze().numpy()
target_image = target_tensor.squeeze().numpy()

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Input (Under-sampled)")

plt.subplot(1, 3, 2)
plt.imshow(output_image, cmap='gray')
plt.title("Output (Predicted)")

plt.subplot(1, 3, 3)
plt.imshow(target_image, cmap='gray')
plt.title("Target (Full)")

plt.tight_layout()
plt.show()
