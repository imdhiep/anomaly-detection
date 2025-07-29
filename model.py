import torch
import torch.nn as nn
import torch.nn.functional as F

class MRN(nn.Module):
    """Multi‑modality Reconstruction Network"""
    def __init__(self):
        super().__init__()
        # Ví dụ: 4-channel input (RGB + depth)
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1); self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        x = self.act(self.conv1(x))
        feat = self.act(self.conv2(x))
        out = self.up1(feat)
        recon = self.up2(out)
        return recon, feat

class MSN(nn.Module):
    """Segmentation network to predict anomaly map from features"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 1, 1)
    def forward(self, feat):
        return torch.sigmoid(self.conv(feat))

class EasyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mrn = MRN()
        self.msn = MSN()
    def forward(self, x):
        recon, feat = self.mrn(x)
        amap = self.msn(feat)
        return recon, amap
