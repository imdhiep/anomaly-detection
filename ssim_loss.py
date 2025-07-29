import torch
import torch.nn.functional as F
from pytorch_msssim import ssim  # cài thư viện pytorch-msssim

def ssim_loss(x, y):
    """
    Tính Structural Similarity loss:
    loss = 1 - SSIM(x, y)
    """
    return 1.0 - ssim(x, y, data_range=1.0, size_average=True)
