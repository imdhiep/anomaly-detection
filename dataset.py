import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class RGBDImageDataset(Dataset):
    def __init__(self, rgb_normal_dir, depth_normal_dir,
                 rgb_anom_dir=None, depth_anom_dir=None,
                 transform_size=(128,128)):
        self.transform_rgb = T.Compose([T.Resize(transform_size), T.ToTensor()])
        self.transform_depth = T.Compose([T.Resize(transform_size), T.Grayscale(1), T.ToTensor()])
        self.samples = []
        # normal
        for fname in sorted(os.listdir(rgb_normal_dir)):
            if fname.lower().endswith(('.png', '.jpg')):
                depthname = fname if fname.lower().endswith('.png') else fname.rsplit('.',1)[0] + '.png'
                if os.path.exists(os.path.join(depth_normal_dir, depthname)):
                    self.samples.append((os.path.join(rgb_normal_dir, fname),
                                          os.path.join(depth_normal_dir, depthname),
                                          0))
        # anomaly
        if rgb_anom_dir and depth_anom_dir:
            for fname in sorted(os.listdir(rgb_anom_dir)):
                if fname.lower().endswith(('.png', '.jpg')):
                    depthname = fname if fname.lower().endswith('.png') else fname.rsplit('.',1)[0] + '.png'
                    if os.path.exists(os.path.join(depth_anom_dir, depthname)):
                        self.samples.append((os.path.join(rgb_anom_dir, fname),
                                              os.path.join(depth_anom_dir, depthname),
                                              1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        rgb_t = self.transform_rgb(rgb)
        depth_t = self.transform_depth(depth)
        inp = torch.cat((rgb_t, depth_t), dim=0)
        return inp, label
