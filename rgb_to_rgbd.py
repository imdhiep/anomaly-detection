"""
RGB to RGB-D Processor
======================

Main processing pipeline for converting 2D RGB images to 4-channel RGB-D tensors.
Combines filtering, histogram equalization, and depth estimation.
"""

import numpy as np
import torch
import cv2
from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path
import os

from .filters import ImageFilter, FilterType
from .histogram_equalization import HistogramEqualizer, ColorSpace
from .depth_estimation import DepthEstimator, DepthPostProcessor
from ..utils.visualization import RGBDVisualizer


class RGBToRGBDProcessor:
    """
    Complete pipeline for converting RGB images to RGB-D volumes.
    
    Integrates image filtering, histogram equalization, depth estimation,
    and tensor generation for deep learning applications.
    """
    
    def __init__(
        self,
        filter_type: Union[FilterType, str] = FilterType.GAUSSIAN,
        equalization_method: Union[ColorSpace, str] = ColorSpace.HSV,
        depth_model: str = "midas_small",
        device: Optional[str] = None,
        output_size: Optional[Tuple[int, int]] = None,
        normalize_output: bool = True
    ):
        """
        Initialize the RGB to RGB-D processor.
        
        Args:
            filter_type: Type of image filter to apply
            equalization_method: Color space for histogram equalization
            depth_model: Depth estimation model to use
            device: Device for PyTorch operations
            output_size: Target output size (H, W) for resizing
            normalize_output: Whether to normalize output to [0, 1]
        """
        self.filter_type = filter_type
        self.equalization_method = equalization_method
        self.depth_model = depth_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_size = output_size
        self.normalize_output = normalize_output
        
        # Initialize processing components
        self.image_filter = ImageFilter()
        self.hist_equalizer = HistogramEqualizer()
        self.depth_estimator = DepthEstimator(
            model_type=depth_model,
            device=self.device
        )
        self.depth_postprocessor = DepthPostProcessor()
        self.visualizer = RGBDVisualizer()
        
        print(f"Initialized RGB-D Processor:")
        print(f"  - Filter: {filter_type}")
        print(f"  - Equalization: {equalization_method}")
        print(f"  - Depth Model: {depth_model}")
        print(f"  - Device: {self.device}")
        print(f"  - Output Size: {output_size}")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            RGB image as numpy array (H, W, 3) with values in [0, 255]
        """
        image_path = str(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image using OpenCV (BGR format)
        image_bgr = cv2.imread(image_path)
        
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    def preprocess_rgb(
        self,
        rgb_image: np.ndarray,
        filter_params: Optional[Dict[str, Any]] = None,
        equalization_params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply filtering and histogram equalization to RGB image.
        
        Args:
            rgb_image: Input RGB image (H, W, 3) with values in [0, 255]
            filter_params: Parameters for image filtering
            equalization_params: Parameters for histogram equalization
            
        Returns:
            Enhanced RGB image (H, W, 3) with values in [0, 255]
        """
        # Apply image filtering
        filter_params = filter_params or {}
        filtered_image = self.image_filter.filter_image(
            rgb_image, 
            self.filter_type, 
            **filter_params
        )
        
        # Apply histogram equalization
        equalization_params = equalization_params or {}
        enhanced_image = self.hist_equalizer.equalize_image(
            filtered_image,
            self.equalization_method,
            **equalization_params
        )
        
        return enhanced_image
    
    def generate_depth(
        self,
        rgb_image: np.ndarray,
        post_process: bool = True,
        smooth_depth: bool = True
    ) -> np.ndarray:
        """
        Generate depth map from RGB image.
        
        Args:
            rgb_image: Enhanced RGB image (H, W, 3) with values in [0, 255]
            post_process: Whether to apply post-processing to depth
            smooth_depth: Whether to smooth the depth map
            
        Returns:
            Depth map (H, W) with values in [0, 1]
        """
        # Estimate depth
        depth_map = self.depth_estimator.estimate_depth(rgb_image)
        
        if post_process:
            # Apply post-processing
            if smooth_depth:
                depth_map = self.depth_postprocessor.smooth_depth(
                    depth_map, 
                    kernel_size=5, 
                    sigma=1.0
                )
            
            # Fill small holes
            depth_map = self.depth_postprocessor.fill_holes(depth_map)
        
        return depth_map
    
    def create_rgbd_volume(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        format: str = "channels_last"  # "channels_last" or "channels_first"
    ) -> np.ndarray:
        """
        Create 4-channel RGB-D volume.
        
        Args:
            rgb_image: RGB image (H, W, 3) with values in [0, 255] or [0, 1]
            depth_map: Depth map (H, W) with values in [0, 1]
            format: Output format ("channels_last" or "channels_first")
            
        Returns:
            RGB-D volume as numpy array
        """
        # Normalize RGB image to [0, 1] if needed
        if self.normalize_output and rgb_image.max() > 1.0:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.astype(np.float32)
        
        # Ensure depth map is the same size as RGB
        if depth_map.shape != rgb_normalized.shape[:2]:
            depth_map = cv2.resize(
                depth_map, 
                (rgb_normalized.shape[1], rgb_normalized.shape[0])
            )
        
        # Stack RGB and depth
        if format == "channels_last":
            # Shape: (H, W, 4)
            rgbd_volume = np.concatenate([
                rgb_normalized, 
                depth_map[..., np.newaxis]
            ], axis=2)
        else:  # channels_first
            # Shape: (4, H, W)
            rgbd_volume = np.concatenate([
                rgb_normalized.transpose(2, 0, 1),
                depth_map[np.newaxis, ...]
            ], axis=0)
        
        return rgbd_volume
    
    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        filter_params: Optional[Dict[str, Any]] = None,
        equalization_params: Optional[Dict[str, Any]] = None,
        return_intermediate: bool = False,
        visualize: bool = False,
        save_visualization: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Complete pipeline to process a single RGB image to RGB-D.
        
        Args:
            image_input: Input image (path or numpy array)
            filter_params: Parameters for image filtering
            equalization_params: Parameters for histogram equalization
            return_intermediate: Whether to return intermediate results
            visualize: Whether to create visualization
            save_visualization: Path to save visualization
            
        Returns:
            RGB-D volume or dictionary with intermediate results
        """
        # Load image if path is provided
        if isinstance(image_input, (str, Path)):
            rgb_image = self.load_image(image_input)
        else:
            rgb_image = image_input.copy()
        
        # Resize if output size is specified
        if self.output_size is not None:
            rgb_image = cv2.resize(rgb_image, self.output_size)
        
        # Step 1: Preprocess RGB image
        enhanced_rgb = self.preprocess_rgb(
            rgb_image, 
            filter_params, 
            equalization_params
        )
        
        # Step 2: Generate depth map
        depth_map = self.generate_depth(enhanced_rgb)
        
        # Step 3: Create RGB-D volume
        rgbd_volume = self.create_rgbd_volume(enhanced_rgb, depth_map)
        
        # Visualization
        if visualize or save_visualization:
            self.visualizer.visualize_side_by_side(
                enhanced_rgb,
                depth_map,
                titles=['Enhanced RGB', 'Estimated Depth'],
                save_path=save_visualization,
                show=visualize
            )
        
        if return_intermediate:
            return {
                'original_rgb': rgb_image,
                'enhanced_rgb': enhanced_rgb,
                'depth_map': depth_map,
                'rgbd_volume': rgbd_volume
            }
        
        return rgbd_volume
    
    def process_batch(
        self,
        image_batch: Union[List[Union[str, np.ndarray]], np.ndarray],
        filter_params: Optional[Dict[str, Any]] = None,
        equalization_params: Optional[Dict[str, Any]] = None,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Process a batch of RGB images to RGB-D volumes.
        
        Args:
            image_batch: Batch of images (list of paths/arrays or numpy array)
            filter_params: Parameters for image filtering
            equalization_params: Parameters for histogram equalization
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Batch of RGB-D volumes or list of intermediate results
        """
        results = []
        
        # Handle different input formats
        if isinstance(image_batch, np.ndarray):
            # Numpy array batch (N, H, W, 3)
            for i in range(image_batch.shape[0]):
                result = self.process_image(
                    image_batch[i],
                    filter_params,
                    equalization_params,
                    return_intermediate
                )
                results.append(result)
        else:
            # List of paths or arrays
            for image_input in image_batch:
                result = self.process_image(
                    image_input,
                    filter_params,
                    equalization_params,
                    return_intermediate
                )
                results.append(result)
        
        if return_intermediate:
            return results
        
        # Stack RGB-D volumes
        rgbd_batch = np.stack([result for result in results], axis=0)
        return rgbd_batch
    
    def to_torch_tensor(
        self,
        rgbd_volume: np.ndarray,
        channels_first: bool = True
    ) -> torch.Tensor:
        """
        Convert RGB-D volume to PyTorch tensor.
        
        Args:
            rgbd_volume: RGB-D volume as numpy array
            channels_first: Whether to use channels-first format
            
        Returns:
            PyTorch tensor
        """
        tensor = torch.from_numpy(rgbd_volume).float()
        
        if channels_first and tensor.dim() == 3:
            # Convert from (H, W, 4) to (4, H, W)
            if tensor.shape[-1] == 4:
                tensor = tensor.permute(2, 0, 1)
        elif not channels_first and tensor.dim() == 3:
            # Convert from (4, H, W) to (H, W, 4)
            if tensor.shape[0] == 4:
                tensor = tensor.permute(1, 2, 0)
        
        return tensor.to(self.device)
    
    def save_rgbd_volume(
        self,
        rgbd_volume: np.ndarray,
        save_path: Union[str, Path],
        format: str = "npz"
    ):
        """
        Save RGB-D volume to file.
        
        Args:
            rgbd_volume: RGB-D volume to save
            save_path: Path to save the volume
            format: Save format ("npz", "npy", or "pt")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "npz":
            np.savez_compressed(save_path, rgbd_volume=rgbd_volume)
        elif format == "npy":
            np.save(save_path, rgbd_volume)
        elif format == "pt":
            tensor = self.to_torch_tensor(rgbd_volume)
            torch.save(tensor, save_path)
        else:
            raise ValueError(f"Unsupported save format: {format}")
        
        print(f"Saved RGB-D volume to: {save_path}")


def create_processor_presets():
    """Create common processor preset configurations."""
    return {
        'high_quality': {
            'filter_type': FilterType.BILATERAL,
            'equalization_method': ColorSpace.LAB,
            'depth_model': 'midas_large',
            'normalize_output': True
        },
        'fast_processing': {
            'filter_type': FilterType.GAUSSIAN,
            'equalization_method': ColorSpace.HSV,
            'depth_model': 'midas_small',
            'normalize_output': True
        },
        'noise_reduction': {
            'filter_type': FilterType.MEDIAN,
            'equalization_method': ColorSpace.YUV,
            'depth_model': 'midas_small',
            'normalize_output': True
        },
        'contrast_enhancement': {
            'filter_type': FilterType.GAUSSIAN,
            'equalization_method': ColorSpace.LAB,
            'depth_model': 'dpt_hybrid',
            'normalize_output': True
        }
    }
