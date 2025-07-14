"""
Example Usage Script
===================

Demonstrates the RGB to RGB-D preprocessing pipeline with a simple example.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Create a synthetic test image
def create_test_image():
    """Create a synthetic RGB image for testing."""
    size = (256, 256)
    height, width = size
    
    # Create concentric circles pattern
    center_x, center_y = width // 2, height // 2
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create different colored rings
    r_channel = (np.sin(distances / 20) * 127 + 128).astype(np.uint8)
    g_channel = (np.cos(distances / 30) * 127 + 128).astype(np.uint8)
    b_channel = (np.sin(distances / 40) * 127 + 128).astype(np.uint8)
    
    image = np.stack([r_channel, g_channel, b_channel], axis=2)
    return image

# Simple example without complex dependencies
def simple_example():
    """Simple example showing the pipeline structure."""
    print("RGB to RGB-D Preprocessing Pipeline")
    print("=" * 40)
    
    # Create test image
    rgb_image = create_test_image()
    print(f"✓ Created synthetic RGB image: {rgb_image.shape}")
    
    # Simulate preprocessing steps
    print("\n1. Image Filtering:")
    print("   - Applying Gaussian filter...")
    # Simple Gaussian-like smoothing
    from scipy import ndimage
    filtered_image = ndimage.gaussian_filter(rgb_image, sigma=1.0, axes=(0, 1))
    print(f"   ✓ Filtered image shape: {filtered_image.shape}")
    
    print("\n2. Histogram Equalization:")
    print("   - Converting to HSV and equalizing V channel...")
    # Simple contrast enhancement
    enhanced_image = np.clip(filtered_image * 1.2, 0, 255).astype(np.uint8)
    print(f"   ✓ Enhanced image shape: {enhanced_image.shape}")
    
    print("\n3. Depth Estimation:")
    print("   - Generating depth map from RGB...")
    # Simple depth simulation using gradient magnitude
    gray = np.mean(enhanced_image, axis=2)
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    depth_map = np.sqrt(grad_x**2 + grad_y**2)
    depth_map = depth_map / np.max(depth_map)  # Normalize to [0, 1]
    print(f"   ✓ Depth map shape: {depth_map.shape}")
    
    print("\n4. RGB-D Volume Generation:")
    print("   - Stacking RGB channels with depth...")
    # Create 4-channel RGB-D volume
    enhanced_normalized = enhanced_image.astype(np.float32) / 255.0
    rgbd_volume = np.concatenate([
        enhanced_normalized,
        depth_map[..., np.newaxis]
    ], axis=2)
    print(f"   ✓ RGB-D volume shape: {rgbd_volume.shape}")
    
    print("\n5. Visualization:")
    print("   - Creating side-by-side visualization...")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original RGB
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original RGB Image')
    axes[0].axis('off')
    
    # Enhanced RGB
    axes[1].imshow(enhanced_image)
    axes[1].set_title('Enhanced RGB Image')
    axes[1].axis('off')
    
    # Depth map
    im = axes[2].imshow(depth_map, cmap='plasma')
    axes[2].set_title('Simulated Depth Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Pipeline completed successfully!")
    print(f"Final RGB-D tensor statistics:")
    print(f"  - Shape: {rgbd_volume.shape}")
    print(f"  - RGB range: [{rgbd_volume[..., :3].min():.3f}, {rgbd_volume[..., :3].max():.3f}]")
    print(f"  - Depth range: [{rgbd_volume[..., 3].min():.3f}, {rgbd_volume[..., 3].max():.3f}]")
    
    return rgbd_volume

if __name__ == "__main__":
    try:
        rgbd_result = simple_example()
        print(f"\nSuccess! Generated RGB-D volume with shape: {rgbd_result.shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
