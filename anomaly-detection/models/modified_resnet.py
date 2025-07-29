import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple


class ModifiedResNet50(nn.Module):
    """Modified ResNet-50 to accept RGB-D (4-channel) input and extract multi-scale features"""
    
    def __init__(self, input_channels: int = 4, pretrained: bool = True):
        super(ModifiedResNet50, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer for 4-channel input
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            input_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize weights for the modified conv1
        self._init_conv1_weights(original_conv1, input_channels)
        
        # Copy other layers from ResNet
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Freeze all parameters except conv1
        self.freeze_backbone()
    
    def _init_conv1_weights(self, original_conv1: nn.Conv2d, input_channels: int):
        """Initialize weights for the modified conv1 layer"""
        with torch.no_grad():
            if input_channels == 3:
                # If 3 channels, just copy original weights
                self.conv1.weight.copy_(original_conv1.weight)
            elif input_channels == 4:
                # Copy RGB weights
                self.conv1.weight[:, :3, :, :].copy_(original_conv1.weight)
                # Initialize depth channel with average of RGB channels
                avg_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                self.conv1.weight[:, 3:4, :, :].copy_(avg_weights)
            else:
                # For other channel numbers, use Xavier initialization
                nn.init.xavier_uniform_(self.conv1.weight)
    
    def freeze_backbone(self):
        """Freeze all ResNet parameters for feature extraction"""
        for param in self.parameters():
            param.requires_grad = False
        
        # Keep conv1 trainable if needed
        for param in self.conv1.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multi-scale feature maps"""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features at different scales
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        
        return {
            "layer1": layer1_out,
            "layer2": layer2_out,
            "layer3": layer3_out,
            "layer4": layer4_out
        }


class FeatureAligner(nn.Module):
    """Align and aggregate multi-scale feature maps"""
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64), 
                 reduce_channels: bool = True, reduction_factor: int = 4):
        super(FeatureAligner, self).__init__()
        self.target_size = target_size
        self.reduce_channels = reduce_channels
        
        # Channel reduction layers if needed
        if reduce_channels:
            self.channel_reducers = nn.ModuleDict({
                'layer1': nn.Conv2d(256, 256 // reduction_factor, kernel_size=1),
                'layer2': nn.Conv2d(512, 512 // reduction_factor, kernel_size=1),
                'layer3': nn.Conv2d(1024, 1024 // reduction_factor, kernel_size=1),
                'layer4': nn.Conv2d(2048, 2048 // reduction_factor, kernel_size=1)
            })
            
            # Calculate total channels after reduction
            self.total_channels = (256 + 512 + 1024 + 2048) // reduction_factor
        else:
            self.total_channels = 256 + 512 + 1024 + 2048
        
        # Smoothing layer
        self.smooth_conv = nn.Conv2d(self.total_channels, self.total_channels, 
                                   kernel_size=3, padding=1, groups=self.total_channels)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Align and concatenate multi-scale features"""
        aligned_features = []
        
        for layer_name, feature_map in features.items():
            # Resize to target size
            resized = F.interpolate(
                feature_map, size=self.target_size, 
                mode='bilinear', align_corners=False
            )
            
            # Reduce channels if specified
            if self.reduce_channels and layer_name in self.channel_reducers:
                resized = self.channel_reducers[layer_name](resized)
            
            aligned_features.append(resized)
        
        # Concatenate along channel dimension
        concatenated = torch.cat(aligned_features, dim=1)
        
        # Apply smoothing
        smoothed = self.smooth_conv(concatenated)
        
        return smoothed
