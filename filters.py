"""
Image Filtering Module
====================

Provides various image denoising and enhancement filters for RGB images.
Supports Gaussian blur, median filtering, and bilateral filtering.
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional
from enum import Enum


class FilterType(Enum):
    """Enumeration of available filter types."""
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


class ImageFilter:
    """
    Image filtering utilities for denoising and enhancement.
    
    Supports multiple filtering methods including Gaussian blur,
    median filtering, and bilateral filtering for edge preservation.
    """
    
    def __init__(self):
        """Initialize the ImageFilter."""
        pass
    
    def apply_gaussian_filter(
        self, 
        image: np.ndarray, 
        kernel_size: Union[int, Tuple[int, int]] = (5, 5),
        sigma_x: float = 1.0,
        sigma_y: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Gaussian blur filter to denoise the image.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            kernel_size: Size of the Gaussian kernel (width, height)
            sigma_x: Standard deviation in X direction
            sigma_y: Standard deviation in Y direction (if None, uses sigma_x)
            
        Returns:
            Filtered image as numpy array
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        # Ensure odd kernel size
        kernel_size = tuple(k if k % 2 == 1 else k + 1 for k in kernel_size)
        
        if sigma_y is None:
            sigma_y = sigma_x
            
        return cv2.GaussianBlur(image, kernel_size, sigma_x, sigmaY=sigma_y)
    
    def apply_median_filter(
        self, 
        image: np.ndarray, 
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Apply median filter to remove salt-and-pepper noise.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            kernel_size: Size of the median filter kernel (must be odd)
            
        Returns:
            Filtered image as numpy array
        """
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.medianBlur(image, kernel_size)
    
    def apply_bilateral_filter(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image as numpy array
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def filter_image(
        self,
        image: np.ndarray,
        filter_type: Union[FilterType, str] = FilterType.GAUSSIAN,
        **kwargs
    ) -> np.ndarray:
        """
        Apply the specified filter to the image.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            filter_type: Type of filter to apply
            **kwargs: Additional parameters for the specific filter
            
        Returns:
            Filtered image as numpy array
        """
        if isinstance(filter_type, str):
            filter_type = FilterType(filter_type.lower())
        
        if filter_type == FilterType.GAUSSIAN:
            return self.apply_gaussian_filter(image, **kwargs)
        elif filter_type == FilterType.MEDIAN:
            return self.apply_median_filter(image, **kwargs)
        elif filter_type == FilterType.BILATERAL:
            return self.apply_bilateral_filter(image, **kwargs)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
    
    def batch_filter(
        self,
        images: np.ndarray,
        filter_type: Union[FilterType, str] = FilterType.GAUSSIAN,
        **kwargs
    ) -> np.ndarray:
        """
        Apply filtering to a batch of images.
        
        Args:
            images: Batch of RGB images as numpy array (N, H, W, 3)
            filter_type: Type of filter to apply
            **kwargs: Additional parameters for the specific filter
            
        Returns:
            Batch of filtered images as numpy array (N, H, W, 3)
        """
        filtered_images = []
        for image in images:
            filtered_image = self.filter_image(image, filter_type, **kwargs)
            filtered_images.append(filtered_image)
        
        return np.stack(filtered_images, axis=0)


def create_filter_presets():
    """Create common filter preset configurations."""
    return {
        'light_gaussian': {
            'filter_type': FilterType.GAUSSIAN,
            'kernel_size': (3, 3),
            'sigma_x': 0.5
        },
        'medium_gaussian': {
            'filter_type': FilterType.GAUSSIAN,
            'kernel_size': (5, 5),
            'sigma_x': 1.0
        },
        'strong_gaussian': {
            'filter_type': FilterType.GAUSSIAN,
            'kernel_size': (7, 7),
            'sigma_x': 2.0
        },
        'light_median': {
            'filter_type': FilterType.MEDIAN,
            'kernel_size': 3
        },
        'medium_median': {
            'filter_type': FilterType.MEDIAN,
            'kernel_size': 5
        },
        'strong_median': {
            'filter_type': FilterType.MEDIAN,
            'kernel_size': 7
        },
        'edge_preserving': {
            'filter_type': FilterType.BILATERAL,
            'd': 9,
            'sigma_color': 75,
            'sigma_space': 75
        }
    }
