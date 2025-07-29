# mvtec_loader_3d.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import tifffile


class MVTec3DDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.rgb_paths = []
        self.depth_paths = []
        self.labels = []

        base_dir = os.path.join(root_dir, phase)
        for cls in os.listdir(base_dir):
            rgb_dir = os.path.join(base_dir, cls, "rgb")
            depth_dir = os.path.join(base_dir, cls, "xyz")
            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
                continue

            label = 0 if cls == "good" else 1

            for fname in os.listdir(rgb_dir):
                if fname.endswith(('.png', '.jpg')):
                    rgb_path = os.path.join(rgb_dir, fname)
                    depth_path = os.path.join(depth_dir, os.path.splitext(fname)[0] + ".tiff")
                    if os.path.exists(depth_path):
                        self.rgb_paths.append(rgb_path)
                        self.depth_paths.append(depth_path)
                        self.labels.append(label)

        print(f"✅ Loaded {len(self.rgb_paths)} samples from {root_dir}/{phase}")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        rgb = self.transform(rgb)

        try:
            depth_array = tifffile.imread(self.depth_paths[idx])
            if depth_array.ndim == 2:
                depth_array = np.stack([depth_array] * 3, axis=0)
            elif depth_array.ndim == 3 and depth_array.shape[2] == 3:
                depth_array = depth_array.transpose(2, 0, 1)
            depth_tensor = torch.from_numpy(depth_array).float() / 255.0
            depth_tensor = transforms.Resize((256, 256))(depth_tensor)
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi đọc depth image tại {self.depth_paths[idx]}: {e}")

        image = torch.cat([rgb, depth_tensor], dim=0)  # (6, H, W)
        label = self.labels[idx]
        return image, label
