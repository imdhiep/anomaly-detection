import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder using only 1x1 convolutions"""
    
    def __init__(self, input_channels: int, latent_dim: int = 128):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoder: progressively reduce channels to latent dimension
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, input_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, latent_dim, kernel_size=1)
        )
        
        # Decoder: progressively increase channels back to input dimension
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, input_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, input_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, input_channels, kernel_size=1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to feature space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both latent and reconstructed features"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed


class AnomalyScorer(nn.Module):
    """Compute anomaly scores and generate segmentation masks"""
    
    def __init__(self, input_size: Tuple[int, int] = (256, 256)):
        super(AnomalyScorer, self).__init__()
        self.input_size = input_size
    
    def compute_anomaly_map(self, original: torch.Tensor, 
                           reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute L2 distance between original and reconstructed features"""
        # Compute L2 distance per pixel
        diff = original - reconstructed
        anomaly_map = torch.norm(diff, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
        
        return anomaly_map
    
    def upsample_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Upsample anomaly map to input image size"""
        upsampled = F.interpolate(
            anomaly_map, size=self.input_size,
            mode='bilinear', align_corners=False
        )
        return upsampled
    
    def create_segmentation_mask(self, anomaly_map: torch.Tensor, 
                                threshold: float = 0.5) -> torch.Tensor:
        """Create binary segmentation mask from anomaly map"""
        # Normalize anomaly map to [0, 1]
        normalized_map = self.normalize_anomaly_map(anomaly_map)
        
        # Apply threshold
        mask = (normalized_map > threshold).float()
        return mask
    
    def normalize_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Normalize anomaly map to [0, 1] range"""
        batch_size = anomaly_map.size(0)
        normalized = torch.zeros_like(anomaly_map)
        
        for i in range(batch_size):
            single_map = anomaly_map[i]
            min_val = single_map.min()
            max_val = single_map.max()
            
            if max_val > min_val:
                normalized[i] = (single_map - min_val) / (max_val - min_val)
            else:
                normalized[i] = single_map
        
        return normalized
    
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor,
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Complete forward pass for anomaly detection"""
        # Compute anomaly map
        anomaly_map = self.compute_anomaly_map(original, reconstructed)
        
        # Upsample to input size
        upsampled_map = self.upsample_anomaly_map(anomaly_map)
        
        # Create segmentation mask
        segmentation_mask = self.create_segmentation_mask(upsampled_map, threshold)
        
        return anomaly_map, upsampled_map, segmentation_mask
