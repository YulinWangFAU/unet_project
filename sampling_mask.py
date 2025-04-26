### 生成欠采样 Mask
import numpy as np
import matplotlib.pyplot as plt

def generate_line_mask(shape, acceleration=2):
    """生成每隔 acceleration 行保留一行的 mask"""
    mask = np.zeros(shape)
    mask[::acceleration, :] = 1
    return mask

def generate_center_mask(shape, percent=0.2):
    """保留图像中心区域的 mask（模拟 MRI 中心密集采样）"""
    mask = np.zeros(shape)
    center_x, center_y = shape[0] // 2, shape[1] // 2
    width_x, width_y = int(shape[0] * percent / 2), int(shape[1] * percent / 2)
    mask[center_x - width_x:center_x + width_x, center_y - width_y:center_y + width_y] = 1
    return mask

def generate_random_mask(shape, keep_ratio=0.2):
    """随机保留 keep_ratio 比例的 mask"""
    mask = np.random.rand(*shape)
    mask = (mask < keep_ratio).astype(float)
    return mask

def visualize_masks(shape=(128, 128)):
    """显示三种 mask 的效果图"""
    line = generate_line_mask(shape)
    center = generate_center_mask(shape)
    rand = generate_random_mask(shape)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(line, cmap='gray')
    axs[0].set_title("Line Mask")
    axs[1].imshow(center, cmap='gray')
    axs[1].set_title("Center Mask")
    axs[2].imshow(rand, cmap='gray')
    axs[2].set_title("Random Mask")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_masks()
