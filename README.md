# 2D to 3D Preprocessing Pipeline

A comprehensive preprocessing pipeline for converting 2D RGB images to 4-channel RGB-D tensors using depth estimation and image enhancement techniques.

## Features

✅ **Image Filtering**
- Gaussian blur for noise reduction
- Median filtering for salt-and-pepper noise
- Bilateral filtering for edge-preserving smoothing

✅ **Histogram Equalization**
- YUV color space equalization (luminance channel)
- HSV color space equalization (value channel)
- LAB color space equalization (lightness channel)
- Adaptive histogram equalization (CLAHE)

✅ **Depth Estimation**
- MiDaS small/large models
- DPT Hybrid model
- Fallback simple depth estimation
- Post-processing and hole filling

✅ **RGB-D Volume Generation**
- 4-channel tensor output (R, G, B, D)
- Configurable output formats
- PyTorch tensor conversion
- Batch processing support

✅ **Visualization**
- Side-by-side RGB and depth display
- Depth map colorization
- 3D point cloud visualization
- Batch comparison tools

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU acceleration (optional):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from preprocess import RGBToRGBDProcessor
import numpy as np

# Initialize processor
processor = RGBToRGBDProcessor(
    filter_type="gaussian",
    equalization_method="hsv",
    depth_model="midas_small"
)

# Process single image
rgbd_volume = processor.process_image("input.jpg")
print(f"RGB-D shape: {rgbd_volume.shape}")  # (H, W, 4)

# Convert to PyTorch tensor
tensor = processor.to_torch_tensor(rgbd_volume)
print(f"Tensor shape: {tensor.shape}")  # (4, H, W)
```

### Command Line Interface

```bash
# Basic conversion
python cli.py input.jpg output.npz

# Batch processing
python cli.py input_folder/ output_folder/ --batch

# Custom settings
python cli.py input.jpg output.npz \
    --filter gaussian \
    --equalization hsv \
    --depth-model midas_large \
    --visualize

# Use presets
python cli.py input.jpg output.npz --preset high_quality

# List available presets
python cli.py --list-presets
```

### Advanced Usage

```python
# Custom filter parameters
filter_params = {
    'kernel_size': (7, 7),
    'sigma_x': 2.0
}

# Custom equalization parameters
equalization_params = {
    'adaptive': True,
    'clip_limit': 2.5,
    'tile_grid_size': (8, 8)
}

# Process with custom parameters
result = processor.process_image(
    "input.jpg",
    filter_params=filter_params,
    equalization_params=equalization_params,
    return_intermediate=True,
    visualize=True
)

# Access intermediate results
original_rgb = result['original_rgb']
enhanced_rgb = result['enhanced_rgb']
depth_map = result['depth_map']
rgbd_volume = result['rgbd_volume']
```

## Pipeline Overview

The preprocessing pipeline consists of four main stages:

### 1. Image Loading and Resizing
- Load RGB images from various formats (.jpg, .png, .bmp, .tiff)
- Optional resizing to target dimensions
- Input validation and format conversion

### 2. Image Enhancement
- **Filtering**: Apply denoising filters to reduce noise
  - Gaussian blur: `cv2.GaussianBlur()`
  - Median filter: `cv2.medianBlur()`
  - Bilateral filter: `cv2.bilateralFilter()`
  
- **Histogram Equalization**: Enhance contrast in different color spaces
  - YUV: Equalize luminance (Y) channel
  - HSV: Equalize value (V) channel
  - LAB: Equalize lightness (L) channel
  - CLAHE: Contrast Limited Adaptive Histogram Equalization

### 3. Depth Estimation
- **Model-based**: Use pretrained depth estimation models
  - MiDaS Small: Fast inference, good quality
  - MiDaS Large: Higher quality, slower inference
  - DPT Hybrid: State-of-the-art quality
  
- **Post-processing**: Clean up depth maps
  - Gaussian smoothing
  - Hole filling using inpainting
  - Edge-preserving filtering

### 4. RGB-D Volume Creation
- Stack enhanced RGB channels with depth map
- Normalize values to [0, 1] range
- Support for different tensor formats
- Conversion to PyTorch tensors

## Configuration Options

### Filter Types
- `gaussian`: Gaussian blur filter
- `median`: Median filter for noise removal
- `bilateral`: Edge-preserving bilateral filter

### Equalization Methods
- `yuv`: YUV color space (luminance channel)
- `hsv`: HSV color space (value channel)
- `lab`: LAB color space (lightness channel)

### Depth Models
- `midas_small`: Fast MiDaS model
- `midas_large`: High-quality MiDaS model
- `dpt_hybrid`: DPT hybrid model (best quality)

### Presets
- `high_quality`: Best quality settings
- `fast_processing`: Optimized for speed
- `noise_reduction`: Focus on noise removal
- `contrast_enhancement`: Enhanced contrast processing

## API Reference

### RGBToRGBDProcessor

Main processor class for RGB to RGB-D conversion.

```python
class RGBToRGBDProcessor:
    def __init__(
        self,
        filter_type: str = "gaussian",
        equalization_method: str = "hsv", 
        depth_model: str = "midas_small",
        device: Optional[str] = None,
        output_size: Optional[Tuple[int, int]] = None,
        normalize_output: bool = True
    )
```

#### Methods

- `process_image(image_input, **kwargs)`: Process single image
- `process_batch(image_batch, **kwargs)`: Process batch of images
- `to_torch_tensor(rgbd_volume)`: Convert to PyTorch tensor
- `save_rgbd_volume(rgbd_volume, save_path)`: Save RGB-D data

### ImageFilter

Image filtering utilities.

```python
class ImageFilter:
    def filter_image(image, filter_type, **kwargs)
    def apply_gaussian_filter(image, kernel_size, sigma_x)
    def apply_median_filter(image, kernel_size)
    def apply_bilateral_filter(image, d, sigma_color, sigma_space)
```

### HistogramEqualizer

Histogram equalization utilities.

```python
class HistogramEqualizer:
    def equalize_image(image, color_space, adaptive=False, **kwargs)
    def equalize_yuv(image, equalize_y=True)
    def equalize_hsv(image, equalize_v=True) 
    def equalize_lab(image, equalize_l=True)
```

### DepthEstimator

Depth estimation utilities.

```python
class DepthEstimator:
    def __init__(model_type="midas_small", device=None)
    def estimate_depth(image)
    def batch_estimate_depth(images)
    def estimate_depth_with_confidence(image)
```

### RGBDVisualizer

Visualization utilities.

```python
class RGBDVisualizer:
    def visualize_side_by_side(rgb_image, depth_map)
    def colorize_depth(depth_map, colormap="plasma")
    def create_depth_histogram(depth_map)
    def visualize_3d_pointcloud(rgb_image, depth_map)
```

## Examples

### Example 1: Basic Processing

```python
from preprocess import RGBToRGBDProcessor

# Initialize with default settings
processor = RGBToRGBDProcessor()

# Process image
rgbd = processor.process_image("input.jpg", visualize=True)

# Save result
processor.save_rgbd_volume(rgbd, "output.npz")
```

### Example 2: Batch Processing

```python
import numpy as np
from pathlib import Path

# Load batch of images
image_paths = list(Path("images/").glob("*.jpg"))

# Process batch
rgbd_batch = processor.process_batch(image_paths)
print(f"Processed {rgbd_batch.shape[0]} images")

# Convert to PyTorch tensors
tensors = [processor.to_torch_tensor(rgbd) for rgbd in rgbd_batch]
```

### Example 3: Custom Configuration

```python
# High-quality processing
processor = RGBToRGBDProcessor(
    filter_type="bilateral",
    equalization_method="lab", 
    depth_model="dpt_hybrid",
    output_size=(512, 512)
)

# Custom parameters
filter_params = {'d': 9, 'sigma_color': 75, 'sigma_space': 75}
eq_params = {'adaptive': True, 'clip_limit': 3.0}

rgbd = processor.process_image(
    "input.jpg",
    filter_params=filter_params,
    equalization_params=eq_params
)
```

### Example 4: Visualization

```python
from preprocess import RGBDVisualizer

visualizer = RGBDVisualizer()

# Side-by-side comparison
visualizer.visualize_side_by_side(rgb_image, depth_map)

# Depth histogram
visualizer.create_depth_histogram(depth_map)

# 3D point cloud
visualizer.visualize_3d_pointcloud(rgb_image, depth_map)
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA-capable GPU for faster depth estimation
2. **Batch Processing**: Process multiple images together for efficiency
3. **Model Selection**: Use `midas_small` for speed, `dpt_hybrid` for quality
4. **Output Size**: Resize to smaller dimensions for faster processing
5. **Memory Management**: Process large batches in chunks to avoid OOM

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or output dimensions
   - Use `midas_small` instead of larger models

2. **Slow Processing**
   - Enable GPU acceleration
   - Use faster filter types (gaussian vs bilateral)
   - Reduce output resolution

3. **Poor Depth Quality**
   - Try different depth models
   - Apply post-processing smoothing
   - Enhance input image contrast

### Error Messages

- `"Import error for MiDaS"`: Install missing dependencies
- `"CUDA not available"`: Install PyTorch with CUDA support
- `"File not found"`: Check input file paths

## License

This project is part of a final year project and is provided for educational purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex
@misc{rgb_to_rgbd_pipeline,
  title={2D to 3D Preprocessing Pipeline for RGB-D Generation},
  author={Final Year Project},
  year={2025}
}
```
# anomaly-detection
