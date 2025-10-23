import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#module load data/scikit-learn
#module load vis/matplotlib
#module load bio/Seaborn/0.13.2-gfbf-2023b
#module load ai/PyTorch/2.3.0-foss-2023b


def load_image(image_path):
    """Load and preprocess an image without torchvision"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC -> CHW
    return image_tensor.unsqueeze(0)  # [1, 3, 224, 224]


def demonstrate_convolution():
    # Create or load image
    try:
        image_tensor = load_image('lossy_524.png')
    except:
        # Create sample image if no file exists
        image_tensor = torch.rand(1, 3, 224, 224)

    print(f"Input shape: {image_tensor.shape}")

    # Define kernels
    kernels = {
        'Edge Detection': torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32),
        'Blur': torch.tensor([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ], dtype=torch.float32),
        'Sharpen': torch.tensor([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=torch.float32)
    }

    for kernel_name, kernel_tensor in kernels.items():
        print(f"\nApplying {kernel_name}...")

        # OPTION 1: With groups=3 (your current approach)
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        print(f"Weight shape with groups=3: {conv.weight.shape}")  # [3, 1, 3, 3]

        with torch.no_grad():
            # With groups=3, each output channel only connects to ONE input channel
            for out_ch in range(3):  # Only loop over output channels
                conv.weight[out_ch, 0] = kernel_tensor  # Only one input channel per output

        # Apply convolution
        with torch.no_grad():
            output = conv(image_tensor)

        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Visualize
        input_img = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
        output_img = output.squeeze(0).permute(1, 2, 0).numpy()
        output_img = np.clip(output_img, 0, 1)  # Clip to valid range

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(output_img)
        plt.title(kernel_name)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'conv_{kernel_name}.jpg')
        plt.show()

# Run the demonstration
demonstrate_convolution()
