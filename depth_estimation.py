"""
Depth Estimation Module
=======================

Provides depth estimation capabilities using pretrained models like MiDaS.
Supports various depth estimation models and depth map processing utilities.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Optional, Union, Tuple, Dict, Any
import urllib.request
import os
from pathlib import Path


class DepthEstimator:
    """
    Depth estimation using pretrained models.
    
    Supports MiDaS and other depth estimation models for generating
    depth maps from RGB images.
    """
    
    def __init__(
        self, 
        model_type: str = "midas_small",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the depth estimator.
        
        Args:
            model_type: Type of model to use ('midas_small', 'midas_large', 'dpt_hybrid')
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir or str(Path.home() / '.cache' / 'depth_models')
        
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified depth estimation model."""
        try:
            # Try to load MiDaS model from torch hub
            if self.model_type == "midas_small":
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            elif self.model_type == "midas_large":
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform
            elif self.model_type == "dpt_hybrid":
                self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
                self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback simple depth estimation...")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback depth estimation method."""
        class SimpleLaplaceDepth:
            def __init__(self, device):
                self.device = device
            
            def __call__(self, x):
                # Simple depth estimation using Laplacian
                # Convert to grayscale if needed
                if x.dim() == 4 and x.shape[1] == 3:
                    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
                else:
                    gray = x.squeeze(1) if x.dim() == 4 else x
                
                # Apply Laplacian filter
                laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                                   [-1,  8, -1],
                                                   [-1, -1, -1]]]]).float().to(self.device)
                
                depth = F.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=1)
                depth = torch.abs(depth)
                
                # Normalize and invert (closer objects should have higher values)
                depth = 1.0 - torch.sigmoid(depth * 0.1)
                
                return depth.squeeze(1)
            
            def to(self, device):
                self.device = device
                return self
            
            def eval(self):
                return self
        
        self.model = SimpleLaplaceDepth(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single RGB image.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3) with values in [0, 255]
            
        Returns:
            Depth map as numpy array (H, W) with values in [0, 1]
        """
        original_height, original_width = image.shape[:2]
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply preprocessing transform
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Convert to numpy and resize to original dimensions
        if isinstance(prediction, torch.Tensor):
            depth_map = prediction.squeeze().cpu().numpy()
        else:
            depth_map = prediction
        
        # Resize to original dimensions
        depth_map = cv2.resize(depth_map, (original_width, original_height))
        
        # Normalize to [0, 1]
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to [0, 1] range.
        
        Args:
            depth_map: Raw depth map
            
        Returns:
            Normalized depth map with values in [0, 1]
        """
        # Handle potential NaN or infinite values
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0, 1]
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)
        
        return depth_map
    
    def batch_estimate_depth(self, images: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a batch of RGB images.
        
        Args:
            images: Batch of RGB images as numpy array (N, H, W, 3)
            
        Returns:
            Batch of depth maps as numpy array (N, H, W)
        """
        depth_maps = []
        for image in images:
            depth_map = self.estimate_depth(image)
            depth_maps.append(depth_map)
        
        return np.stack(depth_maps, axis=0)
    
    def estimate_depth_with_confidence(
        self, 
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate depth with confidence map (if supported by model).
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            return_confidence: Whether to return confidence map
            
        Returns:
            Depth map or tuple of (depth_map, confidence_map)
        """
        depth_map = self.estimate_depth(image)
        
        if return_confidence:
            # Simple confidence estimation based on gradient magnitude
            grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Invert gradient (low gradient = high confidence)
            confidence_map = 1.0 - np.clip(gradient_magnitude / np.max(gradient_magnitude), 0, 1)
            
            return depth_map, confidence_map
        
        return depth_map


class DepthPostProcessor:
    """Post-processing utilities for depth maps."""
    
    @staticmethod
    def smooth_depth(
        depth_map: np.ndarray, 
        kernel_size: int = 5, 
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Smooth depth map using Gaussian filter.
        
        Args:
            depth_map: Input depth map (H, W)
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Smoothed depth map
        """
        return cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), sigma)
    
    @staticmethod
    def edge_preserving_smooth(
        depth_map: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Apply edge-preserving smoothing to depth map.
        
        Args:
            depth_map: Input depth map (H, W)
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            Edge-preserving smoothed depth map
        """
        # Convert to 8-bit for bilateral filter
        depth_8bit = (depth_map * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(depth_8bit, d, sigma_color, sigma_space)
        return smoothed.astype(np.float32) / 255.0
    
    @staticmethod
    def fill_holes(depth_map: np.ndarray, max_hole_size: int = 100) -> np.ndarray:
        """
        Fill small holes in depth map using inpainting.
        
        Args:
            depth_map: Input depth map (H, W)
            max_hole_size: Maximum hole size to fill
            
        Returns:
            Depth map with filled holes
        """
        # Create mask for holes (very small depth values)
        mask = (depth_map < 0.01).astype(np.uint8)
        
        # Remove large holes (keep only small ones)
        kernel = np.ones((3, 3), np.uint8)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Count connected components and filter by size
        num_labels, labels = cv2.connectedComponents(opened_mask)
        final_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            component_mask = (labels == i)
            if np.sum(component_mask) <= max_hole_size:
                final_mask[component_mask] = 1
        
        # Inpaint the holes
        depth_8bit = (depth_map * 255).astype(np.uint8)
        inpainted = cv2.inpaint(depth_8bit, final_mask, 3, cv2.INPAINT_TELEA)
        
        return inpainted.astype(np.float32) / 255.0
