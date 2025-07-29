import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class RGBDImageDataset(Dataset):
    def __init__(self,
                 rgb_dir_normal, depth_dir_normal,
                 rgb_dir_anomaly=None, depth_dir_anomaly=None,
                 transform_size=(128,128)):
        self.samples = []
        # load normal
        for fn in sorted(os.listdir(rgb_dir_normal)):
            if fn.lower().endswith(('.jpg','.png')):
                dfn = os.path.splitext(fn)[0] + '.png'
                rgb_path = os.path.join(rgb_dir_normal, fn)
                depth_path = os.path.join(depth_dir_normal, dfn)
                if os.path.exists(depth_path):
                    self.samples.append((rgb_path, depth_path, 0))
        if rgb_dir_anomaly and depth_dir_anomaly:
            for fn in sorted(os.listdir(rgb_dir_anomaly)):
                if fn.lower().endswith(('.jpg','.png')):
                    dfn = os.path.splitext(fn)[0] + '.png'
                    rgb_path = os.path.join(rgb_dir_anomaly, fn)
                    depth_path = os.path.join(depth_dir_anomaly, dfn)
                    if os.path.exists(depth_path):
                        self.samples.append((rgb_path, depth_path, 1))
        self.transform_rgb = T.Compose([T.Resize(transform_size), T.ToTensor()])
        self.transform_depth = T.Compose([T.Resize(transform_size), T.Grayscale(1), T.ToTensor()])

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
