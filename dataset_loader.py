### 构造训练/测试数据集
import numpy as np
import torch
from torch.utils.data import Dataset
from sampling_mask import generate_line_mask

def apply_ifft(k_space):
    image = np.fft.ifft2(k_space)
    return np.abs(image)

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

class SyntheticMRIDataset(Dataset):
    def __init__(self, num_samples=100, shape=(128, 128), acceleration=4):
        self.num_samples = num_samples
        self.shape = shape
        self.acceleration = acceleration
        self.data = []

        for _ in range(num_samples):
            # 1. Generate full k-space (complex)
            k_space_full = np.random.randn(*shape) + 1j * np.random.randn(*shape)

            # 2. Get full image
            target = apply_ifft(k_space_full)
            target = normalize(target)

            # 3. Undersample
            mask = generate_line_mask(shape, acceleration)
            k_space_under = k_space_full * mask
            input_img = apply_ifft(k_space_under)
            input_img = normalize(input_img)

            # 4. Convert to torch tensor [1, H, W]
            input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

            self.data.append((input_tensor, target_tensor))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
