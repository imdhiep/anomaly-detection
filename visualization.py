"""
Visualization Module
===================

Provides visualization utilities for RGB-D data including side-by-side
comparisons, depth map colorization, and 3D point cloud visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple, Union, List
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


class RGBDVisualizer:
    """
    Visualization utilities for RGB-D data.
    
    Provides methods for displaying RGB images, depth maps, and
    combined visualizations for analysis and debugging.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def visualize_side_by_side(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Display RGB image and depth map side by side.
        
        Args:
            rgb_image: RGB image (H, W, 3) with values in [0, 1] or [0, 255]
            depth_map: Depth map (H, W) with values in [0, 1]
            titles: List of titles for subplots
            save_path: Path to save the figure
            show: Whether to display the figure
            
        Returns:
            Figure object if not showing, None otherwise
        """
        if titles is None:
            titles = ['RGB Image', 'Depth Map']
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Normalize RGB image if needed
        if rgb_image.max() > 1.0:
            rgb_display = rgb_image / 255.0
        else:
            rgb_display = rgb_image
        
        # Display RGB image
        axes[0].imshow(rgb_display)
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        
        # Display depth map with colormap
        depth_colored = self.colorize_depth(depth_map)
        im = axes[1].imshow(depth_colored)
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        
        # Add colorbar for depth
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def colorize_depth(
        self,
        depth_map: np.ndarray,
        colormap: str = 'plasma',
        invert: bool = False
    ) -> np.ndarray:
        """
        Apply colormap to depth map for visualization.
        
        Args:
            depth_map: Depth map (H, W) with values in [0, 1]
            colormap: Matplotlib colormap name
            invert: Whether to invert the depth values
            
        Returns:
            Colored depth map (H, W, 3)
        """
        if invert:
            depth_map = 1.0 - depth_map
        
        # Apply colormap
        colormap_fn = cm.get_cmap(colormap)
        colored_depth = colormap_fn(depth_map)
        
        # Remove alpha channel if present
        if colored_depth.shape[-1] == 4:
            colored_depth = colored_depth[..., :3]
        
        return colored_depth
    
    def visualize_rgbd_volume(
        self,
        rgbd_volume: np.ndarray,
        slice_indices: Optional[List[int]] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Visualize RGB-D volume by showing RGB and depth channels.
        
        Args:
            rgbd_volume: RGB-D volume (4, H, W) or (H, W, 4)
            slice_indices: Indices to visualize (default: [0, 1, 2, 3])
            titles: Titles for each channel
            save_path: Path to save the figure
            show: Whether to display the figure
            
        Returns:
            Figure object if not showing, None otherwise
        """
        # Ensure correct format (H, W, 4)
        if rgbd_volume.shape[0] == 4:
            rgbd_volume = rgbd_volume.transpose(1, 2, 0)
        
        if slice_indices is None:
            slice_indices = [0, 1, 2, 3]
        
        if titles is None:
            titles = ['Red Channel', 'Green Channel', 'Blue Channel', 'Depth Channel']
        
        n_channels = len(slice_indices)
        fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
        
        if n_channels == 1:
            axes = [axes]
        
        for i, (idx, title) in enumerate(zip(slice_indices, titles)):
            channel = rgbd_volume[..., idx]
            
            if idx < 3:  # RGB channels
                axes[i].imshow(channel, cmap='gray', vmin=0, vmax=1)
            else:  # Depth channel
                colored_depth = self.colorize_depth(channel)
                im = axes[i].imshow(colored_depth)
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def create_depth_histogram(
        self,
        depth_map: np.ndarray,
        bins: int = 50,
        title: str = "Depth Distribution",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create histogram of depth values.
        
        Args:
            depth_map: Depth map (H, W)
            bins: Number of histogram bins
            title: Plot title
            save_path: Path to save the figure
            show: Whether to display the figure
            
        Returns:
            Figure object if not showing, None otherwise
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Flatten depth map and remove zero values
        depth_values = depth_map.flatten()
        depth_values = depth_values[depth_values > 0]
        
        ax.hist(depth_values, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Depth Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_depth = np.mean(depth_values)
        median_depth = np.median(depth_values)
        ax.axvline(mean_depth, color='red', linestyle='--', label=f'Mean: {mean_depth:.3f}')
        ax.axvline(median_depth, color='orange', linestyle='--', label=f'Median: {median_depth:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def visualize_3d_pointcloud(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        downsample_factor: int = 10,
        max_depth: float = 1.0,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create 3D point cloud visualization from RGB-D data.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_map: Depth map (H, W)
            downsample_factor: Factor to downsample points for performance
            max_depth: Maximum depth value to include
            save_path: Path to save the figure
            show: Whether to display the figure
            
        Returns:
            Figure object if not showing, None otherwise
        """
        height, width = depth_map.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height:downsample_factor, 0:width:downsample_factor]
        
        # Get corresponding depth and color values
        depth_values = depth_map[::downsample_factor, ::downsample_factor]
        rgb_values = rgb_image[::downsample_factor, ::downsample_factor]
        
        # Filter by depth threshold
        valid_mask = (depth_values > 0) & (depth_values <= max_depth)
        
        x_3d = x_coords[valid_mask]
        y_3d = y_coords[valid_mask]
        z_3d = depth_values[valid_mask] * 100  # Scale for visualization
        colors = rgb_values[valid_mask]
        
        # Normalize colors if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x_3d, y_3d, z_3d, c=colors, s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')
        ax.set_title('3D Point Cloud from RGB-D Data')
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            return None
        else:
            return fig
    
    def compare_depth_methods(
        self,
        rgb_image: np.ndarray,
        depth_maps: List[np.ndarray],
        method_names: List[str],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Compare multiple depth estimation methods side by side.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_maps: List of depth maps to compare
            method_names: Names of the depth estimation methods
            save_path: Path to save the figure
            show: Whether to display the figure
            
        Returns:
            Figure object if not showing, None otherwise
        """
        n_methods = len(depth_maps)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        
        # Normalize RGB image if needed
        if rgb_image.max() > 1.0:
            rgb_display = rgb_image / 255.0
        else:
            rgb_display = rgb_image
        
        # Show original RGB image
        axes[0, 0].imshow(rgb_display)
        axes[0, 0].set_title('Original RGB')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')  # Empty space below RGB image
        
        # Show depth maps
        for i, (depth_map, method_name) in enumerate(zip(depth_maps, method_names)):
            col_idx = i + 1
            
            # Colorized depth map
            colored_depth = self.colorize_depth(depth_map)
            im = axes[0, col_idx].imshow(colored_depth)
            axes[0, col_idx].set_title(f'{method_name} - Depth')
            axes[0, col_idx].axis('off')
            
            # Depth histogram
            depth_values = depth_map.flatten()
            depth_values = depth_values[depth_values > 0]
            axes[1, col_idx].hist(depth_values, bins=30, alpha=0.7)
            axes[1, col_idx].set_title(f'{method_name} - Distribution')
            axes[1, col_idx].set_xlabel('Depth Value')
            axes[1, col_idx].set_ylabel('Frequency')
            axes[1, col_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            return None
        else:
            return fig
