import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class BasicBlock3D(nn.Module):
    """Basic 3D residual block for 3D-ResNet."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck3D(nn.Module):
    """Bottleneck 3D residual block for deeper 3D-ResNet."""
    
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D ResNet for feature extraction."""
    
    def __init__(self, block, num_blocks: List[int], in_channels: int = 1, 
                 num_classes: int = 1000, extract_features: bool = True):
        super(ResNet3D, self).__init__()
        self.in_planes = 64
        self.extract_features = extract_features
        self.feature_maps = {}
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Register hooks for feature extraction
        if self.extract_features:
            self.layer1.register_forward_hook(self._get_activation('layer1'))
            self.layer2.register_forward_hook(self._get_activation('layer2'))
            self.layer3.register_forward_hook(self._get_activation('layer3'))
            self.layer4.register_forward_hook(self._get_activation('layer4'))
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str):
        def hook(model, input, output):
            self.feature_maps[name] = output
        return hook
    
    def forward(self, x):
        self.feature_maps = {}  # Clear previous feature maps
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Return extracted feature maps."""
        return self.feature_maps


class Simple3DCNN(nn.Module):
    """Simple 3D CNN for feature extraction."""
    
    def __init__(self, in_channels: int = 1, feature_dims: List[int] = [32, 64, 128, 256]):
        super(Simple3DCNN, self).__init__()
        self.feature_maps = {}
        
        layers = []
        prev_dim = in_channels
        
        for i, dim in enumerate(feature_dims):
            conv_block = nn.Sequential(
                nn.Conv3d(prev_dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(dim),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
            layers.append(conv_block)
            prev_dim = dim
            
            # Register hook for feature extraction
            conv_block.register_forward_hook(self._get_activation(f'layer{i+1}'))
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(feature_dims[-1], 1)
    
    def _get_activation(self, name: str):
        def hook(model, input, output):
            self.feature_maps[name] = output
        return hook
    
    def forward(self, x):
        self.feature_maps = {}  # Clear previous feature maps
        
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Return extracted feature maps."""
        return self.feature_maps


class FeatureExtractor3D(nn.Module):
    """Wrapper for 3D CNN feature extraction."""
    
    def __init__(self, model_type: str = 'simple', **kwargs):
        super(FeatureExtractor3D, self).__init__()
        
        if model_type == 'simple':
            self.backbone = Simple3DCNN(**kwargs)
        elif model_type == 'resnet18':
            self.backbone = ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)
        elif model_type == 'resnet34':
            self.backbone = ResNet3D(BasicBlock3D, [3, 4, 6, 3], **kwargs)
        elif model_type == 'resnet50':
            self.backbone = ResNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor, 
                        layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input tensor
            layers: List of layer names to extract features from
        
        Returns:
            Dictionary of layer names to feature tensors
        """
        # Forward pass to populate feature maps
        _ = self.backbone(x)
        
        feature_maps = self.backbone.get_feature_maps()
        
        if layers is not None:
            feature_maps = {k: v for k, v in feature_maps.items() if k in layers}
        
        return feature_maps
    
    def get_feature_dimensions(self, input_shape: Tuple[int, int, int, int]) -> Dict[str, Tuple]:
        """Get the dimensions of feature maps for given input shape."""
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape[1:])  # Remove batch dimension
            feature_maps = self.extract_features(dummy_input)
            return {k: v.shape for k, v in feature_maps.items()}


def resnet3d_18(in_channels: int = 1, **kwargs):
    """Construct a 3D ResNet-18 model."""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels=in_channels, **kwargs)


def resnet3d_34(in_channels: int = 1, **kwargs):
    """Construct a 3D ResNet-34 model."""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels=in_channels, **kwargs)


def resnet3d_50(in_channels: int = 1, **kwargs):
    """Construct a 3D ResNet-50 model."""
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels=in_channels, **kwargs)
