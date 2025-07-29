import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class RGBDImageDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, rgb_anom_dir, depth_anom_dir, transform_size=(128, 128)):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.rgb_anom_dir = rgb_anom_dir
        self.depth_anom_dir = depth_anom_dir
        self.rgb_images = self._get_images(rgb_dir)
        self.depth_images = self._get_images(depth_dir)
        self.rgb_anom_images = self._get_images(rgb_anom_dir)
        self.depth_anom_images = self._get_images(depth_anom_dir)
        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.ToTensor()
        ])
        self.data = self._pair_images()

    def _get_images(self, directory):
        return sorted([f for f in os.listdir(directory) if f.endswith('.png')])

    def _pair_images(self):
        paired_data = []
        for rgb_image in self.rgb_images:
            depth_image = rgb_image  # Depth image has the same name as RGB image
            if depth_image in self.depth_images:
                paired_data.append((rgb_image, depth_image, 0))  # Label 0 for normal
        for rgb_image in self.rgb_anom_images:
            depth_image = rgb_image  # Depth image has the same name as RGB image
            if depth_image in self.depth_anom_images:
                paired_data.append((rgb_image, depth_image, 1))  # Label 1 for anomaly
        return paired_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_image, depth_image, label = self.data[idx]
        rgb_path = os.path.join(self.rgb_dir, rgb_image)
        depth_path = os.path.join(self.depth_dir, depth_image)
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')  # Assuming depth is grayscale
        rgb = self.transform(rgb)
        depth = self.transform(depth)
        combined = torch.cat((rgb, depth), dim=0)  # Combine RGB and depth
        return combined, label
