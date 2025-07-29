import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .modified_resnet import ModifiedResNet50, FeatureAligner
from .autoencoder import ConvolutionalAutoencoder, AnomalyScorer


class AnomalyDetectionModel(nn.Module):
    """Complete anomaly detection model combining ResNet features and CAE"""
    
    def __init__(self, 
                 input_channels: int = 4,
                 feature_layers: list = ["layer1", "layer2", "layer3", "layer4"],
                 feature_size: Tuple[int, int] = (64, 64),
                 latent_dim: int = 128,
                 input_size: Tuple[int, int] = (256, 256),
                 reduce_channels: bool = True,
                 reduction_factor: int = 4):
        super(AnomalyDetectionModel, self).__init__()
        
        self.feature_layers = feature_layers
        self.input_size = input_size
        
        # Feature extractor (modified ResNet-50)
        self.feature_extractor = ModifiedResNet50(
            input_channels=input_channels, 
            pretrained=True
        )
        
        # Feature aligner
        self.feature_aligner = FeatureAligner(
            target_size=feature_size,
            reduce_channels=reduce_channels,
            reduction_factor=reduction_factor
        )
        
        # Convolutional Autoencoder
        self.autoencoder = ConvolutionalAutoencoder(
            input_channels=self.feature_aligner.total_channels,
            latent_dim=latent_dim
        )
        
        # Anomaly scorer
        self.anomaly_scorer = AnomalyScorer(input_size=input_size)
        
        # Loss function
        self.reconstruction_loss = nn.MSELoss()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and align multi-scale features"""
        # Get multi-scale features from ResNet
        features = self.feature_extractor(x)
        
        # Filter features based on selected layers
        selected_features = {
            layer: features[layer] 
            for layer in self.feature_layers 
            if layer in features
        }
        
        # Align and aggregate features
        aligned_features = self.feature_aligner(selected_features)
        
        return aligned_features
    
    def forward(self, x: torch.Tensor, 
                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Complete forward pass for training and inference"""
        # Extract features
        features = self.extract_features(x)
        
        # Autoencoder forward pass
        latent, reconstructed = self.autoencoder(features)
        
        # Compute anomaly scores
        anomaly_map, upsampled_map, segmentation_mask = self.anomaly_scorer(
            features, reconstructed, threshold
        )
        
        return {
            "features": features,
            "latent": latent,
            "reconstructed": reconstructed,
            "anomaly_map": anomaly_map,
            "upsampled_anomaly_map": upsampled_map,
            "segmentation_mask": segmentation_mask
        }
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for training"""
        features = self.extract_features(x)
        latent, reconstructed = self.autoencoder(features)
        loss = self.reconstruction_loss(reconstructed, features)
        return loss
    
    def predict_anomaly(self, x: torch.Tensor, 
                       threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly map and segmentation mask"""
        self.eval()
        with torch.no_grad():
            results = self.forward(x, threshold)
            return results["upsampled_anomaly_map"], results["segmentation_mask"]
    
    def freeze_feature_extractor(self):
        """Freeze ResNet feature extractor for training only CAE"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in self.feature_aligner.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


class AnomalyDetectionConfig:
    """Configuration class for the anomaly detection model"""
    
    def __init__(self):
        # Model architecture
        self.input_channels = 4  # RGB-D
        self.feature_layers = ["layer1", "layer2", "layer3", "layer4"]
        self.feature_size = (64, 64)
        self.latent_dim = 128
        self.input_size = (256, 256)
        self.reduce_channels = True
        self.reduction_factor = 4
        
        # Training parameters
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.num_epochs = 700
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Evaluation parameters
        self.anomaly_threshold = 0.5
        
        # Data parameters
        self.train_normal_only = True
        
    def get_rgb_only_config(self):
        """Get config for RGB-only input"""
        config = AnomalyDetectionConfig()
        config.input_channels = 3
        return config
    
    def get_reduced_layers_config(self, layers: list):
        """Get config with specific feature layers"""
        config = AnomalyDetectionConfig()
        config.feature_layers = layers
        return config
