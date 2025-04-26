### 多样本训练并保存模型
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.simple_unet import SimpleUNet
from dataset_loader import SyntheticMRIDataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = SyntheticMRIDataset(num_samples=200, shape=(128, 128), acceleration=4)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 定义模型和训练组件
model = SimpleUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 20
losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0

    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # 每 5 个 epoch 可视化 1 个样本
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            sample_input, sample_target = dataset[0]
            sample_input = sample_input.unsqueeze(0).to(device)
            pred = model(sample_input).cpu().squeeze().numpy()

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_input.cpu().squeeze().numpy(), cmap='gray')
        plt.title("Input (Under-sampled)")

        plt.subplot(1, 3, 2)
        plt.imshow(pred, cmap='gray')
        plt.title("Output (Predicted)")

        plt.subplot(1, 3, 3)
        plt.imshow(sample_target.squeeze().numpy(), cmap='gray')
        plt.title("Target (Full)")

        plt.tight_layout()
        plt.show()

# 训练完成后保存模型
torch.save(model.state_dict(), "unet_trained.pt")
print("✅ Model saved to unet_trained.pt")

# 画 Loss 曲线
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
