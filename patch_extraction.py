import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union
import itertools
from typing import Optional, Union, Tuple


def extract_patches_3d(
    volume: torch.Tensor,
    patch_sizes: List[Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: str = 'valid'
) -> List[torch.Tensor]:
    """
    Extract overlapping 3D patches from a 4D volume tensor.
    
    Args:
        volume: Input volume tensor of shape (C, D, H, W)
        patch_sizes: List of patch sizes as (patch_d, patch_h, patch_w)
        stride: Stride for patch extraction. If int, same stride for all dimensions.
               If None, uses patch_size // 2 for overlap
        padding: 'valid' or 'same'. Whether to pad the volume
    
    Returns:
        List of flattened patch tensors, one for each patch size
    """
    if volume.dim() != 4:
        raise ValueError(f"Expected 4D tensor (C, D, H, W), got {volume.dim()}D")
    
    C, D, H, W = volume.shape
    all_patches = []
    
    for patch_size in patch_sizes:
        patch_d, patch_h, patch_w = patch_size
        
        # Determine stride
        if stride is None:
            stride_d, stride_h, stride_w = patch_d // 2, patch_h // 2, patch_w // 2
        elif isinstance(stride, int):
            stride_d = stride_h = stride_w = stride
        else:
            stride_d, stride_h, stride_w = stride
        
        # Apply padding if needed
        if padding == 'same':
            pad_d = patch_d // 2
            pad_h = patch_h // 2
            pad_w = patch_w // 2
            volume_padded = F.pad(volume, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
        else:
            volume_padded = volume
        
        # Extract patches using unfold
        patches = volume_padded.unfold(1, patch_d, stride_d)\
                              .unfold(2, patch_h, stride_h)\
                              .unfold(3, patch_w, stride_w)
        
        # Reshape to (C, num_patches_d, num_patches_h, num_patches_w, patch_d, patch_h, patch_w)
        # Then flatten each patch
        C, n_d, n_h, n_w, p_d, p_h, p_w = patches.shape
        
        # Reshape to (num_patches, C * patch_d * patch_h * patch_w)
        patches_flat = patches.permute(1, 2, 3, 0, 4, 5, 6)\
                             .contiguous()\
                             .view(-1, C * p_d * p_h * p_w)
        
        all_patches.append(patches_flat)
    
    return all_patches


def extract_patches_with_coordinates(
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    return_coordinates: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract patches and optionally return their coordinates in the original volume.
    
    Args:
        volume: Input volume tensor of shape (C, D, H, W)
        patch_size: Patch size as (patch_d, patch_h, patch_w)
        stride: Stride for patch extraction
        return_coordinates: Whether to return patch coordinates
    
    Returns:
        If return_coordinates is False: patches tensor
        If return_coordinates is True: (patches, coordinates) tuple
    """
    if volume.dim() != 4:
        raise ValueError(f"Expected 4D tensor (C, D, H, W), got {volume.dim()}D")
    
    C, D, H, W = volume.shape
    patch_d, patch_h, patch_w = patch_size
    
    # Determine stride
    if stride is None:
        stride_d, stride_h, stride_w = patch_d // 2, patch_h // 2, patch_w // 2
    elif isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
    
    patches = []
    coordinates = []
    
    # Extract patches manually to keep track of coordinates
    for d in range(0, D - patch_d + 1, stride_d):
        for h in range(0, H - patch_h + 1, stride_h):
            for w in range(0, W - patch_w + 1, stride_w):
                patch = volume[:, d:d+patch_d, h:h+patch_h, w:w+patch_w]
                patches.append(patch.flatten())
                
                if return_coordinates:
                    coordinates.append([d, h, w])
    
    patches_tensor = torch.stack(patches)  # (num_patches, C * patch_d * patch_h * patch_w)
    
    if return_coordinates:
        coordinates_tensor = torch.tensor(coordinates, dtype=torch.long)
        return patches_tensor, coordinates_tensor
    else:
        return patches_tensor


def reconstruct_from_patches(
    patches: torch.Tensor,
    coordinates: torch.Tensor,
    original_shape: Tuple[int, int, int, int],
    patch_size: Tuple[int, int, int],
    aggregation: str = 'mean'
) -> torch.Tensor:
    """
    Reconstruct a volume from patches and their coordinates.
    
    Args:
        patches: Patch tensor of shape (num_patches, C * patch_d * patch_h * patch_w)
        coordinates: Coordinates tensor of shape (num_patches, 3)
        original_shape: Original volume shape (C, D, H, W)
        patch_size: Patch size (patch_d, patch_h, patch_w)
        aggregation: How to handle overlapping regions ('mean', 'sum')
    
    Returns:
        Reconstructed volume tensor
    """
    C, D, H, W = original_shape
    patch_d, patch_h, patch_w = patch_size
    
    # Initialize reconstruction volume and count volume for averaging
    reconstruction = torch.zeros(original_shape, dtype=patches.dtype, device=patches.device)
    count_volume = torch.zeros(original_shape, dtype=torch.float, device=patches.device)
    
    # Reshape patches back to patch shape
    patches_reshaped = patches.view(-1, C, patch_d, patch_h, patch_w)
    
    # Place patches back into the volume
    for i, (d, h, w) in enumerate(coordinates):
        reconstruction[:, d:d+patch_d, h:h+patch_h, w:w+patch_w] += patches_reshaped[i]
        count_volume[:, d:d+patch_d, h:h+patch_h, w:w+patch_w] += 1.0
    
    # Handle aggregation
    if aggregation == 'mean':
        # Avoid division by zero
        count_volume = torch.clamp(count_volume, min=1.0)
        reconstruction = reconstruction / count_volume
    
    return reconstruction


def create_patch_grid(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None
) -> List[Tuple[slice, slice, slice]]:
    """
    Create a grid of patch slices for systematic patch extraction.
    
    Args:
        volume_shape: Shape of the volume (D, H, W)
        patch_size: Patch size (patch_d, patch_h, patch_w)
        stride: Stride for patch extraction
    
    Returns:
        List of slice tuples for patch extraction
    """
    D, H, W = volume_shape
    patch_d, patch_h, patch_w = patch_size
    
    # Determine stride
    if stride is None:
        stride_d, stride_h, stride_w = patch_d, patch_h, patch_w  # Non-overlapping
    elif isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
    
    patch_slices = []
    
    for d in range(0, D - patch_d + 1, stride_d):
        for h in range(0, H - patch_h + 1, stride_h):
            for w in range(0, W - patch_w + 1, stride_w):
                patch_slice = (
                    slice(d, d + patch_d),
                    slice(h, h + patch_h),
                    slice(w, w + patch_w)
                )
                patch_slices.append(patch_slice)
    
    return patch_slices
