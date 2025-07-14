import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from feature_embedding import FeatureEmbedding, PatchSampler, MultiScaleFeatureEmbedding
from patch_extraction import extract_patches_with_coordinates, reconstruct_from_patches
import scipy.ndimage as ndimage


def extract_test_features(
    volume: torch.Tensor,
    feature_extractor,
    embedding_module: FeatureEmbedding,
    patch_sampler: PatchSampler,
    layer_names: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Extract features from test volume for anomaly detection.
    
    Args:
        volume: Input volume tensor (C, D, H, W)
        feature_extractor: Pre-trained feature extractor
        embedding_module: Feature embedding module
        patch_sampler: Patch sampling module
        layer_names: Names of layers to extract features from
        device: Device to run inference on
    
    Returns:
        Tuple of (feature_vectors, coordinates, metadata)
    """
    feature_extractor.eval()
    embedding_module.eval()
    
    volume = volume.to(device)
    feature_extractor.to(device)
    embedding_module.to(device)
    
    all_features = []
    all_coordinates = []
    metadata = {'layer_shapes': {}, 'original_shape': volume.shape}
    
    with torch.no_grad():
        # Extract feature maps from specified layers
        if hasattr(feature_extractor, 'extract_features'):
            feature_maps = feature_extractor.extract_features(volume.unsqueeze(0), layer_names)
        else:
            # For models without extract_features method
            _ = feature_extractor(volume.unsqueeze(0))
            feature_maps = feature_extractor.get_feature_maps()
        
        for layer_name in layer_names:
            if layer_name not in feature_maps:
                continue
            
            feature_map = feature_maps[layer_name].squeeze(0)  # Remove batch dimension
            metadata['layer_shapes'][layer_name] = feature_map.shape
            
            # Sample patches from feature map
            patches, coordinates = patch_sampler(feature_map)
            
            # Embed patches
            embedded_features = embedding_module(patches)
            
            all_features.append(embedded_features)
            all_coordinates.append(coordinates)
    
    # Concatenate features from all layers
    if all_features:
        combined_features = torch.cat(all_features, dim=0)
        combined_coordinates = torch.cat(all_coordinates, dim=0)
    else:
        combined_features = torch.empty(0, embedding_module.embed_dim)
        combined_coordinates = torch.empty(0, 3)
    
    return combined_features, combined_coordinates, metadata


def compute_reconstruction_errors(
    features: torch.Tensor,
    autoencoder,
    reduction: str = 'none',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Compute reconstruction errors using trained autoencoder.
    
    Args:
        features: Feature vectors (N, feature_dim)
        autoencoder: Trained autoencoder model
        reduction: 'none', 'mean', or 'sum'
        device: Device to run inference on
    
    Returns:
        Reconstruction errors
    """
    autoencoder.eval()
    autoencoder.to(device)
    features = features.to(device)
    
    with torch.no_grad():
        if hasattr(autoencoder, 'get_reconstruction_error'):
            # Use built-in method if available
            errors = autoencoder.get_reconstruction_error(features, reduction=reduction)
        else:
            # Compute manually
            reconstructed = autoencoder(features)
            
            if reduction == 'none':
                errors = F.mse_loss(reconstructed, features, reduction='none').sum(dim=1)
            elif reduction == 'mean':
                errors = F.mse_loss(reconstructed, features, reduction='mean')
            elif reduction == 'sum':
                errors = F.mse_loss(reconstructed, features, reduction='sum')
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
    
    return errors


def create_anomaly_score_map(
    reconstruction_errors: torch.Tensor,
    coordinates: torch.Tensor,
    original_shape: Tuple[int, int, int, int],
    patch_size: Tuple[int, int, int],
    aggregation: str = 'mean',
    smoothing_sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Create 3D anomaly score map from reconstruction errors and patch coordinates.
    
    Args:
        reconstruction_errors: Per-patch reconstruction errors (N,)
        coordinates: Patch coordinates (N, 3)
        original_shape: Original volume shape (C, D, H, W)
        patch_size: Size of patches (patch_d, patch_h, patch_w)
        aggregation: How to handle overlapping regions ('mean', 'max', 'min')
        smoothing_sigma: Standard deviation for Gaussian smoothing
    
    Returns:
        3D anomaly score map
    """
    C, D, H, W = original_shape
    patch_d, patch_h, patch_w = patch_size
    
    # Initialize score map and count map
    score_map = torch.zeros((D, H, W), dtype=torch.float32)
    count_map = torch.zeros((D, H, W), dtype=torch.float32)
    
    # Place reconstruction errors back into the volume
    for i, (d, h, w) in enumerate(coordinates):
        d, h, w = int(d), int(h), int(w)
        error = reconstruction_errors[i].item()
        
        # Define patch region
        d_end = min(d + patch_d, D)
        h_end = min(h + patch_h, H)
        w_end = min(w + patch_w, W)
        
        if aggregation == 'mean':
            score_map[d:d_end, h:h_end, w:w_end] += error
            count_map[d:d_end, h:h_end, w:w_end] += 1.0
        elif aggregation == 'max':
            current_scores = score_map[d:d_end, h:h_end, w:w_end]
            score_map[d:d_end, h:h_end, w:w_end] = torch.maximum(current_scores, 
                                                                torch.full_like(current_scores, error))
        elif aggregation == 'min':
            current_scores = score_map[d:d_end, h:h_end, w:w_end]
            mask = count_map[d:d_end, h:h_end, w:w_end] == 0
            score_map[d:d_end, h:h_end, w:w_end] = torch.where(
                mask, 
                torch.full_like(current_scores, error),
                torch.minimum(current_scores, torch.full_like(current_scores, error))
            )
            count_map[d:d_end, h:h_end, w:w_end] += 1.0
    
    # Handle aggregation for mean
    if aggregation == 'mean':
        # Avoid division by zero
        count_map = torch.clamp(count_map, min=1.0)
        score_map = score_map / count_map
    
    # Apply smoothing if requested
    if smoothing_sigma is not None and smoothing_sigma > 0:
        score_map_np = score_map.numpy()
        score_map_smoothed = ndimage.gaussian_filter(score_map_np, sigma=smoothing_sigma)
        score_map = torch.from_numpy(score_map_smoothed)
    
    return score_map


def inference_pipeline(
    volume: torch.Tensor,
    feature_extractor,
    embedding_module: FeatureEmbedding,
    patch_sampler: PatchSampler,
    autoencoder,
    layer_names: List[str],
    patch_size: Tuple[int, int, int],
    aggregation: str = 'mean',
    smoothing_sigma: Optional[float] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, Dict]:
    """
    Complete inference pipeline for anomaly detection.
    
    Args:
        volume: Input volume tensor (C, D, H, W)
        feature_extractor: Pre-trained feature extractor
        embedding_module: Feature embedding module
        patch_sampler: Patch sampling module
        autoencoder: Trained autoencoder
        layer_names: Names of layers to extract features from
        patch_size: Size of patches
        aggregation: Aggregation method for overlapping patches
        smoothing_sigma: Smoothing parameter for score map
        device: Device to run inference on
    
    Returns:
        Tuple of (anomaly_score_map, metadata)
    """
    # Extract features
    features, coordinates, metadata = extract_test_features(
        volume, feature_extractor, embedding_module, patch_sampler, layer_names, device
    )
    
    if features.size(0) == 0:
        # No features extracted, return zero score map
        score_map = torch.zeros(volume.shape[1:])
        return score_map, metadata
    
    # Compute reconstruction errors
    reconstruction_errors = compute_reconstruction_errors(features, autoencoder, device=device)
    
    # Create anomaly score map
    score_map = create_anomaly_score_map(
        reconstruction_errors,
        coordinates,
        volume.shape,
        patch_size,
        aggregation=aggregation,
        smoothing_sigma=smoothing_sigma
    )
    
    # Add statistics to metadata
    metadata.update({
        'num_patches': features.size(0),
        'mean_error': reconstruction_errors.mean().item(),
        'std_error': reconstruction_errors.std().item(),
        'min_error': reconstruction_errors.min().item(),
        'max_error': reconstruction_errors.max().item(),
        'score_map_stats': {
            'mean': score_map.mean().item(),
            'std': score_map.std().item(),
            'min': score_map.min().item(),
            'max': score_map.max().item()
        }
    })
    
    return score_map, metadata


def batch_inference(
    volumes: List[torch.Tensor],
    feature_extractor,
    embedding_module: FeatureEmbedding,
    patch_sampler: PatchSampler,
    autoencoder,
    layer_names: List[str],
    patch_size: Tuple[int, int, int],
    **kwargs
) -> List[Tuple[torch.Tensor, Dict]]:
    """
    Run inference on multiple volumes.
    
    Args:
        volumes: List of input volumes
        feature_extractor: Pre-trained feature extractor
        embedding_module: Feature embedding module
        patch_sampler: Patch sampling module
        autoencoder: Trained autoencoder
        layer_names: Names of layers to extract features from
        patch_size: Size of patches
        **kwargs: Additional arguments for inference_pipeline
    
    Returns:
        List of (anomaly_score_map, metadata) tuples
    """
    results = []
    
    for i, volume in enumerate(volumes):
        print(f"Processing volume {i+1}/{len(volumes)}")
        score_map, metadata = inference_pipeline(
            volume, feature_extractor, embedding_module, patch_sampler,
            autoencoder, layer_names, patch_size, **kwargs
        )
        results.append((score_map, metadata))
    
    return results


def ensemble_inference(
    volume: torch.Tensor,
    models: List[Dict],
    aggregation_method: str = 'mean'
) -> torch.Tensor:
    """
    Run ensemble inference using multiple trained models.
    
    Args:
        volume: Input volume tensor
        models: List of model dictionaries containing required components
        aggregation_method: How to combine predictions ('mean', 'max', 'voting')
    
    Returns:
        Ensemble anomaly score map
    """
    score_maps = []
    
    for model_dict in models:
        score_map, _ = inference_pipeline(volume, **model_dict)
        score_maps.append(score_map)
    
    # Combine score maps
    if aggregation_method == 'mean':
        ensemble_scores = torch.stack(score_maps).mean(dim=0)
    elif aggregation_method == 'max':
        ensemble_scores = torch.stack(score_maps).max(dim=0)[0]
    elif aggregation_method == 'voting':
        # Binary voting - each model votes for anomaly/normal
        binary_maps = [score_map > score_map.median() for score_map in score_maps]
        ensemble_scores = torch.stack(binary_maps).float().mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return ensemble_scores
