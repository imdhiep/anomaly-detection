import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
from datetime import datetime

from data_loader import load_and_normalize_volume, load_dataset
from feature_extractor import FeatureExtractor3D, Simple3DCNN
from feature_embedding import FeatureEmbedding, PatchSampler
from autoencoder import Autoencoder, VariationalAutoencoder, DenoisingAutoencoder
from training import train_autoencoder, FeatureVectorDataset
from inference import inference_pipeline, extract_test_features, compute_reconstruction_errors
from thresholding import compute_threshold_from_normal_scores, apply_threshold
from evaluation import comprehensive_evaluation, compute_roc_auc, compute_precision_recall


class AnomalyDetectionPipeline:
    """
    Modular PyTorch pipeline for 3D anomaly detection using autoencoders.
    """
    
    def __init__(self,
                 feature_extractor_config: Dict,
                 embedding_config: Dict,
                 autoencoder_config: Dict,
                 patch_config: Dict,
                 training_config: Optional[Dict] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the anomaly detection pipeline.
        
        Args:
            feature_extractor_config: Configuration for 3D CNN feature extractor
            embedding_config: Configuration for feature embedding
            autoencoder_config: Configuration for autoencoder
            patch_config: Configuration for patch extraction
            training_config: Configuration for training
            device: Device to run computations on
        """
        self.device = device
        self.training_config = training_config or {}
        
        # Initialize feature extractor
        self.feature_extractor = self._build_feature_extractor(feature_extractor_config)
        
        # Initialize patch sampler
        self.patch_sampler = PatchSampler(**patch_config)
        
        # Initialize feature embedding
        self.embedding_module = FeatureEmbedding(**embedding_config)
        
        # Initialize autoencoder
        self.autoencoder = self._build_autoencoder(autoencoder_config)
        
        # Pipeline state
        self.is_fitted = False
        self.threshold = None
        self.training_history = None
        
        # Configuration storage
        self.config = {
            'feature_extractor': feature_extractor_config,
            'embedding': embedding_config,
            'autoencoder': autoencoder_config,
            'patch': patch_config,
            'training': training_config
        }
    
    def _build_feature_extractor(self, config: Dict) -> nn.Module:
        """Build feature extractor from configuration."""
        extractor_type = config.get('type', 'simple')
        extractor_args = config.get('args', {})
        
        return FeatureExtractor3D(model_type=extractor_type, **extractor_args)
    
    def _build_autoencoder(self, config: Dict) -> nn.Module:
        """Build autoencoder from configuration."""
        ae_type = config.get('type', 'standard')
        ae_args = config.get('args', {})
        
        if ae_type == 'standard':
            return Autoencoder(**ae_args)
        elif ae_type == 'variational':
            return VariationalAutoencoder(**ae_args)
        elif ae_type == 'denoising':
            return DenoisingAutoencoder(**ae_args)
        else:
            raise ValueError(f"Unknown autoencoder type: {ae_type}")
    
    def _extract_features_from_volumes(self, volumes: List[torch.Tensor]) -> torch.Tensor:
        """Extract features from a list of volumes."""
        all_features = []
        
        for volume in volumes:
            # Extract features from current volume
            features, _, _ = extract_test_features(
                volume,
                self.feature_extractor,
                self.embedding_module,
                self.patch_sampler,
                layer_names=['layer1', 'layer2'],  # Default layers
                device=self.device
            )
            
            all_features.append(features)
        
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.empty(0, self.embedding_module.embed_dim)
    
    def fit(self,
            train_volumes: List[torch.Tensor],
            val_volumes: Optional[List[torch.Tensor]] = None,
            threshold_method: str = 'percentile',
            threshold_percentile: float = 95.0) -> 'AnomalyDetectionPipeline':
        """
        Fit the anomaly detection pipeline on normal training data.
        
        Args:
            train_volumes: List of normal training volumes
            val_volumes: List of normal validation volumes (optional)
            threshold_method: Method for computing anomaly threshold
            threshold_percentile: Percentile for threshold computation
        
        Returns:
            Self (for method chaining)
        """
        print("Extracting features from training volumes...")
        train_features = self._extract_features_from_volumes(train_volumes)
        
        val_features = None
        if val_volumes is not None:
            print("Extracting features from validation volumes...")
            val_features = self._extract_features_from_volumes(val_volumes)
        
        if train_features.size(0) == 0:
            raise ValueError("No features extracted from training volumes")
        
        print(f"Training autoencoder on {train_features.size(0)} feature vectors...")
        
        # Train autoencoder
        self.training_history = train_autoencoder(
            self.autoencoder,
            train_features,
            val_features,
            device=self.device,
            **self.training_config
        )
        
        # Compute threshold using training data reconstruction errors
        print("Computing anomaly threshold...")
        with torch.no_grad():
            train_errors = compute_reconstruction_errors(
                train_features,
                self.autoencoder,
                device=self.device
            )
        
        self.threshold = compute_threshold_from_normal_scores(
            train_errors,
            method=threshold_method,
            percentile=threshold_percentile
        )
        
        self.is_fitted = True
        print(f"Pipeline fitted successfully. Threshold: {self.threshold:.6f}")
        
        return self
    
    def predict(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly mask for a single volume.
        
        Args:
            volume: Input volume tensor (C, D, H, W)
        
        Returns:
            Binary anomaly mask
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        score_map, _ = self.score_map(volume)
        binary_mask = apply_threshold(score_map, self.threshold)
        
        return binary_mask
    
    def score_map(self, volume: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Generate anomaly score map for a single volume.
        
        Args:
            volume: Input volume tensor (C, D, H, W)
        
        Returns:
            Tuple of (anomaly_score_map, metadata)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before scoring")
        
        score_map, metadata = inference_pipeline(
            volume,
            self.feature_extractor,
            self.embedding_module,
            self.patch_sampler,
            self.autoencoder,
            layer_names=['layer1', 'layer2'],  # Default layers
            patch_size=self.patch_sampler.patch_size,
            device=self.device
        )
        
        return score_map, metadata
    
    def evaluate(self,
                 test_volumes: List[torch.Tensor],
                 ground_truth_masks: List[torch.Tensor],
                 custom_threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data with ground truth.
        
        Args:
            test_volumes: List of test volumes
            ground_truth_masks: List of ground truth binary masks
            custom_threshold: Custom threshold (uses fitted threshold if None)
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        if len(test_volumes) != len(ground_truth_masks):
            raise ValueError("Number of test volumes must match number of ground truth masks")
        
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        
        print(f"Evaluating on {len(test_volumes)} test volumes...")
        
        # Generate score maps for all test volumes
        score_maps = []
        for i, volume in enumerate(test_volumes):
            print(f"Processing volume {i+1}/{len(test_volumes)}")
            score_map, _ = self.score_map(volume)
            score_maps.append(score_map)
        
        # Compute comprehensive evaluation metrics
        metrics = comprehensive_evaluation(
            ground_truth_masks,
            score_maps,
            threshold
        )
        
        return metrics
    
    def batch_predict(self, volumes: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Predict anomaly masks for multiple volumes.
        
        Args:
            volumes: List of input volumes
        
        Returns:
            List of binary anomaly masks
        """
        predictions = []
        for volume in volumes:
            prediction = self.predict(volume)
            predictions.append(prediction)
        
        return predictions
    
    def batch_score_maps(self, volumes: List[torch.Tensor]) -> List[Tuple[torch.Tensor, Dict]]:
        """
        Generate anomaly score maps for multiple volumes.
        
        Args:
            volumes: List of input volumes
        
        Returns:
            List of (score_map, metadata) tuples
        """
        results = []
        for volume in volumes:
            score_map, metadata = self.score_map(volume)
            results.append((score_map, metadata))
        
        return results
    
    def set_threshold(self, threshold: float):
        """Set custom threshold value."""
        self.threshold = threshold
    
    def get_threshold(self) -> Optional[float]:
        """Get current threshold value."""
        return self.threshold
    
    def save(self, save_path: Union[str, Path]):
        """
        Save the complete pipeline to disk.
        
        Args:
            save_path: Path to save the pipeline
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model states
        torch.save({
            'feature_extractor_state': self.feature_extractor.state_dict(),
            'autoencoder_state': self.autoencoder.state_dict(),
            'threshold': self.threshold,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'training_history': self.training_history
        }, save_path / 'pipeline_state.pth')
        
        # Save configuration as JSON
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Pipeline saved to {save_path}")
    
    @classmethod
    def load(cls, save_path: Union[str, Path], device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'AnomalyDetectionPipeline':
        """
        Load a pipeline from disk.
        
        Args:
            save_path: Path to load the pipeline from
            device: Device to load the pipeline on
        
        Returns:
            Loaded pipeline instance
        """
        save_path = Path(save_path)
        
        # Load state
        state = torch.load(save_path / 'pipeline_state.pth', map_location=device)
        config = state['config']
        
        # Create pipeline instance
        pipeline = cls(
            feature_extractor_config=config['feature_extractor'],
            embedding_config=config['embedding'],
            autoencoder_config=config['autoencoder'],
            patch_config=config['patch'],
            training_config=config['training'],
            device=device
        )
        
        # Load model states
        pipeline.feature_extractor.load_state_dict(state['feature_extractor_state'])
        pipeline.autoencoder.load_state_dict(state['autoencoder_state'])
        
        # Restore pipeline state
        pipeline.threshold = state['threshold']
        pipeline.is_fitted = state['is_fitted']
        pipeline.training_history = state.get('training_history')
        
        print(f"Pipeline loaded from {save_path}")
        
        return pipeline
    
    def get_config(self) -> Dict:
        """Get pipeline configuration."""
        return self.config.copy()
    
    def get_training_history(self) -> Optional[Dict]:
        """Get training history."""
        return self.training_history
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        threshold_str = f", threshold={self.threshold:.6f}" if self.threshold is not None else ""
        return f"AnomalyDetectionPipeline(status={status}{threshold_str})"


# Convenience functions for creating common pipeline configurations

def create_simple_pipeline(input_dim: int = 64,
                          patch_size: Tuple[int, int, int] = (8, 8, 8),
                          hidden_dims: List[int] = [128, 64],
                          latent_dim: int = 32) -> AnomalyDetectionPipeline:
    """Create a simple pipeline with default configurations."""
    
    feature_extractor_config = {
        'type': 'simple',
        'args': {
            'in_channels': 1,
            'feature_dims': [32, 64, 128]
        }
    }
    
    embedding_config = {
        'input_channels': 128,  # From last layer of simple CNN
        'patch_size': patch_size,
        'embedding_method': 'flatten'
    }
    
    autoencoder_config = {
        'type': 'standard',
        'args': {
            'input_dim': 128 * patch_size[0] * patch_size[1] * patch_size[2],
            'hidden_dims': hidden_dims,
            'latent_dim': latent_dim
        }
    }
    
    patch_config = {
        'patch_size': patch_size,
        'stride': tuple(s // 2 for s in patch_size),
        'max_patches': 1000,
        'sampling_strategy': 'uniform'
    }
    
    training_config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'early_stopping_patience': 10
    }
    
    return AnomalyDetectionPipeline(
        feature_extractor_config=feature_extractor_config,
        embedding_config=embedding_config,
        autoencoder_config=autoencoder_config,
        patch_config=patch_config,
        training_config=training_config
    )


def create_resnet_pipeline(input_dim: int = 64,
                          patch_size: Tuple[int, int, int] = (4, 4, 4),
                          autoencoder_type: str = 'standard') -> AnomalyDetectionPipeline:
    """Create a pipeline with ResNet-based feature extractor."""
    
    feature_extractor_config = {
        'type': 'resnet18',
        'args': {
            'in_channels': 1,
            'extract_features': True
        }
    }
    
    embedding_config = {
        'input_channels': 512,  # From ResNet-18 final layer
        'patch_size': patch_size,
        'embedding_method': 'pool',
        'pooling_type': 'adaptive'
    }
    
    autoencoder_config = {
        'type': autoencoder_type,
        'args': {
            'input_dim': 512,  # After adaptive pooling
            'hidden_dims': [256, 128, 64],
            'latent_dim': 32
        }
    }
    
    patch_config = {
        'patch_size': patch_size,
        'stride': tuple(s // 2 for s in patch_size),
        'max_patches': 2000,
        'sampling_strategy': 'random'
    }
    
    training_config = {
        'batch_size': 64,
        'num_epochs': 150,
        'learning_rate': 1e-3,
        'scheduler_type': 'cosine',
        'early_stopping_patience': 15
    }
    
    return AnomalyDetectionPipeline(
        feature_extractor_config=feature_extractor_config,
        embedding_config=embedding_config,
        autoencoder_config=autoencoder_config,
        patch_config=patch_config,
        training_config=training_config
    )
