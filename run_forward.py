import torch
from model.simple_unet import SimpleUNet

model = SimpleUNet()
x = torch.randn(1, 1, 128, 128)  # Batch size 1, 1 channel, 128x128 image
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
