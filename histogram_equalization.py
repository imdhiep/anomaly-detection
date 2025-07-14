"""
Histogram Equalization Module
============================

Provides histogram equalization techniques for image contrast enhancement.
Supports YUV and HSV color spaces with selective channel equalization.
"""

import cv2
import numpy as np
from typing import Union, Literal
from enum import Enum


class ColorSpace(Enum):
    """Enumeration of supported color spaces."""
    YUV = "yuv"
    HSV = "hsv"
    LAB = "lab"


class HistogramEqualizer:
    """
    Histogram equalization utilities for contrast enhancement.
    
    Supports multiple color spaces and selective channel equalization
    to improve image contrast while preserving color information.
    """
    
    def __init__(self):
        """Initialize the HistogramEqualizer."""
        pass
    
    def equalize_yuv(self, image: np.ndarray, equalize_y: bool = True) -> np.ndarray:
        """
        Apply histogram equalization in YUV color space.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            equalize_y: Whether to equalize the Y (luminance) channel
            
        Returns:
            Contrast-enhanced RGB image
        """
        # Convert RGB to YUV
        yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        if equalize_y:
            # Apply histogram equalization to Y channel
            yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        
        return enhanced_image
    
    def equalize_hsv(self, image: np.ndarray, equalize_v: bool = True) -> np.ndarray:
        """
        Apply histogram equalization in HSV color space.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            equalize_v: Whether to equalize the V (value) channel
            
        Returns:
            Contrast-enhanced RGB image
        """
        # Convert RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        if equalize_v:
            # Apply histogram equalization to V channel
            hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        return enhanced_image
    
    def equalize_lab(self, image: np.ndarray, equalize_l: bool = True) -> np.ndarray:
        """
        Apply histogram equalization in LAB color space.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            equalize_l: Whether to equalize the L (lightness) channel
            
        Returns:
            Contrast-enhanced RGB image
        """
        # Convert RGB to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        if equalize_l:
            # Apply histogram equalization to L channel
            lab_image[:, :, 0] = cv2.equalizeHist(lab_image[:, :, 0])
        
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
        
        return enhanced_image
    
    def adaptive_equalize_hsv(
        self, 
        image: np.ndarray, 
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) in HSV.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of the grid for histogram equalization
            
        Returns:
            Contrast-enhanced RGB image
        """
        # Convert RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE to V channel
        hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])
        
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        return enhanced_image
    
    def adaptive_equalize_lab(
        self, 
        image: np.ndarray, 
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8)
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of the grid for histogram equalization
            
        Returns:
            Contrast-enhanced RGB image
        """
        # Convert RGB to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE to L channel
        lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
        
        # Convert back to RGB
        enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
        
        return enhanced_image
    
    def equalize_image(
        self,
        image: np.ndarray,
        color_space: Union[ColorSpace, str] = ColorSpace.HSV,
        adaptive: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Apply histogram equalization using the specified color space.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            color_space: Color space to use for equalization
            adaptive: Whether to use adaptive (CLAHE) equalization
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Contrast-enhanced RGB image
        """
        if isinstance(color_space, str):
            color_space = ColorSpace(color_space.lower())
        
        if adaptive:
            if color_space == ColorSpace.HSV:
                return self.adaptive_equalize_hsv(image, **kwargs)
            elif color_space == ColorSpace.LAB:
                return self.adaptive_equalize_lab(image, **kwargs)
            else:
                raise ValueError(f"Adaptive equalization not supported for {color_space}")
        else:
            if color_space == ColorSpace.YUV:
                return self.equalize_yuv(image, **kwargs)
            elif color_space == ColorSpace.HSV:
                return self.equalize_hsv(image, **kwargs)
            elif color_space == ColorSpace.LAB:
                return self.equalize_lab(image, **kwargs)
            else:
                raise ValueError(f"Unsupported color space: {color_space}")
    
    def batch_equalize(
        self,
        images: np.ndarray,
        color_space: Union[ColorSpace, str] = ColorSpace.HSV,
        adaptive: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Apply histogram equalization to a batch of images.
        
        Args:
            images: Batch of RGB images as numpy array (N, H, W, 3)
            color_space: Color space to use for equalization
            adaptive: Whether to use adaptive (CLAHE) equalization
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Batch of contrast-enhanced RGB images
        """
        enhanced_images = []
        for image in images:
            enhanced_image = self.equalize_image(image, color_space, adaptive, **kwargs)
            enhanced_images.append(enhanced_image)
        
        return np.stack(enhanced_images, axis=0)


def create_equalization_presets():
    """Create common histogram equalization preset configurations."""
    return {
        'yuv_standard': {
            'color_space': ColorSpace.YUV,
            'equalize_y': True
        },
        'hsv_standard': {
            'color_space': ColorSpace.HSV,
            'equalize_v': True
        },
        'lab_standard': {
            'color_space': ColorSpace.LAB,
            'equalize_l': True
        },
        'hsv_adaptive_light': {
            'color_space': ColorSpace.HSV,
            'adaptive': True,
            'clip_limit': 1.5,
            'tile_grid_size': (8, 8)
        },
        'hsv_adaptive_medium': {
            'color_space': ColorSpace.HSV,
            'adaptive': True,
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        },
        'hsv_adaptive_strong': {
            'color_space': ColorSpace.HSV,
            'adaptive': True,
            'clip_limit': 3.0,
            'tile_grid_size': (4, 4)
        },
        'lab_adaptive_medium': {
            'color_space': ColorSpace.LAB,
            'adaptive': True,
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        }
    }
