import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from typing import List, Tuple, Optional, Union
from pathlib import Path
import SimpleITK as sitk


def load_and_normalize_volume(
    file_path: Union[str, Path],
    target_shape: Optional[Tuple[int, int, int]] = None,
    normalize_method: str = "z_score",
    clip_percentile: Tuple[float, float] = (1.0, 99.0),
    add_channel_dim: bool = True
) -> torch.Tensor:
    """
    Load a 3D image dataset (NIfTI, NumPy volumes) and return normalized volumes.
    
    Args:
        file_path: Path to the image file
        target_shape: Target shape for resizing (D, H, W). If None, no resizing
        normalize_method: "z_score", "min_max", or "percentile"
        clip_percentile: Percentile values for clipping outliers
        add_channel_dim: Whether to add channel dimension for shape (C, D, H, W)
    
    Returns:
        Normalized volume tensor with shape (C, D, H, W) or (D, H, W)
    """
    file_path = Path(file_path)
    
    # Load image based on file extension
    if file_path.suffix.lower() in ['.nii', '.nii.gz']:
        # Load NIfTI file
        img = nib.load(str(file_path))
        volume = img.get_fdata()
    elif file_path.suffix.lower() in ['.npy', '.npz']:
        # Load NumPy file
        volume = np.load(str(file_path))
        if file_path.suffix.lower() == '.npz':
            # Assume the volume is stored with key 'volume' or take the first key
            keys = list(volume.keys())
            volume = volume[keys[0]] if 'volume' not in keys else volume['volume']
    else:
        # Try using SimpleITK for other formats
        img = sitk.ReadImage(str(file_path))
        volume = sitk.GetArrayFromImage(img)
    
    # Convert to float32
    volume = volume.astype(np.float32)
    
    # Clip outliers
    if clip_percentile is not None:
        low_p, high_p = np.percentile(volume, clip_percentile)
        volume = np.clip(volume, low_p, high_p)
    
    # Normalize
    if normalize_method == "z_score":
        mean = np.mean(volume)
        std = np.std(volume)
        volume = (volume - mean) / (std + 1e-8)
    elif normalize_method == "min_max":
        min_val = np.min(volume)
        max_val = np.max(volume)
        volume = (volume - min_val) / (max_val - min_val + 1e-8)
    elif normalize_method == "percentile":
        p1, p99 = np.percentile(volume, [1, 99])
        volume = (volume - p1) / (p99 - p1 + 1e-8)
    
    # Convert to tensor
    volume_tensor = torch.from_numpy(volume)
    
    # Resize if target shape is specified
    if target_shape is not None:
        # Add batch and channel dimensions for interpolation
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        volume_tensor = F.interpolate(
            volume_tensor, 
            size=target_shape, 
            mode='trilinear', 
            align_corners=False
        )
        volume_tensor = volume_tensor.squeeze(0).squeeze(0)  # (D, H, W)
    
    # Add channel dimension if requested
    if add_channel_dim:
        volume_tensor = volume_tensor.unsqueeze(0)  # (1, D, H, W)
    
    return volume_tensor


def augment_volume(
    volume: torch.Tensor,
    rotation_range: float = 10.0,
    flip_probability: float = 0.5,
    noise_std: float = 0.01,
    brightness_range: float = 0.1
) -> torch.Tensor:
    """
    Apply optional augmentations to 3D volume.
    
    Args:
        volume: Input volume tensor (C, D, H, W)
        rotation_range: Range for random rotation in degrees
        flip_probability: Probability of random flipping
        noise_std: Standard deviation for Gaussian noise
        brightness_range: Range for brightness adjustment
    
    Returns:
        Augmented volume tensor
    """
    augmented = volume.clone()
    
    # Random flip
    if torch.rand(1) < flip_probability:
        augmented = torch.flip(augmented, dims=[np.random.choice([1, 2, 3])])
    
    # Add Gaussian noise
    if noise_std > 0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise
    
    # Brightness adjustment
    if brightness_range > 0:
        brightness_factor = 1.0 + (torch.rand(1) - 0.5) * 2 * brightness_range
        augmented = augmented * brightness_factor
    
    return augmented


def load_dataset(
    data_paths: List[Union[str, Path]],
    target_shape: Optional[Tuple[int, int, int]] = None,
    normalize_method: str = "z_score",
    apply_augmentation: bool = False,
    **augment_kwargs
) -> List[torch.Tensor]:
    """
    Load multiple volumes and return a list of normalized tensors.
    
    Args:
        data_paths: List of paths to image files
        target_shape: Target shape for resizing
        normalize_method: Normalization method
        apply_augmentation: Whether to apply augmentation
        **augment_kwargs: Additional arguments for augmentation
    
    Returns:
        List of normalized volume tensors
    """
    volumes = []
    
    for path in data_paths:
        volume = load_and_normalize_volume(
            path, 
            target_shape=target_shape,
            normalize_method=normalize_method
        )
        
        if apply_augmentation:
            volume = augment_volume(volume, **augment_kwargs)
        
        volumes.append(volume)
    
    return volumes
