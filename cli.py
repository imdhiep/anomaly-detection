#!/usr/bin/env python3
"""
Command Line Interface for RGB to RGB-D Preprocessing
=====================================================

Provides a command-line interface for converting 2D RGB images to 4-channel RGB-D tensors.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.rgb_to_rgbd import RGBToRGBDProcessor, create_processor_presets
from core.filters import FilterType, create_filter_presets
from core.histogram_equalization import ColorSpace, create_equalization_presets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert 2D RGB images to 4-channel RGB-D tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py input.jpg output.npz
  
  # Batch processing
  python cli.py input_folder/ output_folder/ --batch
  
  # Custom settings
  python cli.py input.jpg output.npz --filter gaussian --equalization hsv --depth-model midas_large
  
  # Use presets
  python cli.py input.jpg output.npz --preset high_quality
  
  # Enable visualization
  python cli.py input.jpg output.npz --visualize --save-viz result_viz.png
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        'input',
        help='Input image file or directory for batch processing'
    )
    parser.add_argument(
        'output',
        help='Output file or directory for RGB-D data'
    )
    
    # Processing options
    parser.add_argument(
        '--filter',
        choices=['gaussian', 'median', 'bilateral'],
        default='gaussian',
        help='Type of image filter to apply (default: gaussian)'
    )
    parser.add_argument(
        '--equalization',
        choices=['yuv', 'hsv', 'lab'],
        default='hsv',
        help='Color space for histogram equalization (default: hsv)'
    )
    parser.add_argument(
        '--depth-model',
        choices=['midas_small', 'midas_large', 'dpt_hybrid'],
        default='midas_small',
        help='Depth estimation model (default: midas_small)'
    )
    
    # Filter parameters
    parser.add_argument(
        '--gaussian-kernel',
        type=int,
        default=5,
        help='Gaussian filter kernel size (default: 5)'
    )
    parser.add_argument(
        '--gaussian-sigma',
        type=float,
        default=1.0,
        help='Gaussian filter sigma (default: 1.0)'
    )
    parser.add_argument(
        '--median-kernel',
        type=int,
        default=5,
        help='Median filter kernel size (default: 5)'
    )
    
    # Equalization parameters
    parser.add_argument(
        '--adaptive-equalization',
        action='store_true',
        help='Use adaptive histogram equalization (CLAHE)'
    )
    parser.add_argument(
        '--clip-limit',
        type=float,
        default=2.0,
        help='CLAHE clip limit (default: 2.0)'
    )
    
    # Output options
    parser.add_argument(
        '--output-format',
        choices=['npz', 'npy', 'pt'],
        default='npz',
        help='Output file format (default: npz)'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize output to specific dimensions'
    )
    parser.add_argument(
        '--channels-first',
        action='store_true',
        help='Use channels-first format (4, H, W) instead of (H, W, 4)'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        help='File extensions to process in batch mode'
    )
    
    # Visualization
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of results'
    )
    parser.add_argument(
        '--save-viz',
        help='Save visualization to file'
    )
    
    # Presets
    parser.add_argument(
        '--preset',
        choices=['high_quality', 'fast_processing', 'noise_reduction', 'contrast_enhancement'],
        help='Use predefined processing preset'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets and exit'
    )
    
    # Performance
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='Device to use for processing (auto-detect if not specified)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def list_presets():
    """List available processing presets."""
    print("Available Processing Presets:")
    print("=" * 40)
    
    presets = create_processor_presets()
    for name, config in presets.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nFilter Presets:")
    print("=" * 20)
    filter_presets = create_filter_presets()
    for name, config in filter_presets.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nEqualization Presets:")
    print("=" * 25)
    eq_presets = create_equalization_presets()
    for name, config in eq_presets.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")


def get_image_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Get list of image files in directory."""
    image_files = []
    for ext in extensions:
        pattern = f"*{ext}"
        image_files.extend(directory.glob(pattern))
        # Also check uppercase extensions
        pattern = f"*{ext.upper()}"
        image_files.extend(directory.glob(pattern))
    
    return sorted(image_files)


def process_single_image(args, processor: RGBToRGBDProcessor):
    """Process a single image."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare processing parameters
    filter_params = {}
    if args.filter == 'gaussian':
        filter_params = {
            'kernel_size': args.gaussian_kernel,
            'sigma_x': args.gaussian_sigma
        }
    elif args.filter == 'median':
        filter_params = {
            'kernel_size': args.median_kernel
        }
    
    equalization_params = {}
    if args.adaptive_equalization:
        equalization_params = {
            'adaptive': True,
            'clip_limit': args.clip_limit
        }
    
    if args.verbose:
        print(f"Processing: {input_path}")
        print(f"Filter: {args.filter} with params: {filter_params}")
        print(f"Equalization: {args.equalization} with params: {equalization_params}")
    
    try:
        # Process image
        result = processor.process_image(
            str(input_path),
            filter_params=filter_params,
            equalization_params=equalization_params,
            return_intermediate=False,
            visualize=args.visualize,
            save_visualization=args.save_viz
        )
        
        # Adjust format if needed
        if args.channels_first and result.shape[-1] == 4:
            result = result.transpose(2, 0, 1)
        elif not args.channels_first and result.shape[0] == 4:
            result = result.transpose(1, 2, 0)
        
        # Save result
        if args.output_format == 'npz':
            np.savez_compressed(output_path, rgbd_volume=result)
        elif args.output_format == 'npy':
            np.save(output_path, result)
        elif args.output_format == 'pt':
            import torch
            tensor = torch.from_numpy(result).float()
            torch.save(tensor, output_path)
        
        if args.verbose:
            print(f"Saved RGB-D volume: {output_path}")
            print(f"Output shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_batch(args, processor: RGBToRGBDProcessor):
    """Process a batch of images."""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = get_image_files(input_dir, args.extensions)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return False
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files):
        if args.verbose:
            print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Create output filename
        output_filename = image_path.stem + f".{args.output_format}"
        output_path = output_dir / output_filename
        
        # Update args for this file
        args_copy = argparse.Namespace(**vars(args))
        args_copy.input = str(image_path)
        args_copy.output = str(output_path)
        args_copy.visualize = False  # Disable visualization for batch
        
        if process_single_image(args_copy, processor):
            success_count += 1
        
        if not args.verbose:
            # Show progress
            progress = (i + 1) / len(image_files) * 100
            print(f"\rProgress: {progress:.1f}% ({i+1}/{len(image_files)})", end='')
    
    if not args.verbose:
        print()  # New line after progress
    
    print(f"\nProcessing complete: {success_count}/{len(image_files)} images processed successfully")
    return success_count > 0


def main():
    """Main function."""
    args = parse_args()
    
    # List presets and exit if requested
    if args.list_presets:
        list_presets()
        return
    
    # Apply preset if specified
    if args.preset:
        presets = create_processor_presets()
        if args.preset in presets:
            preset_config = presets[args.preset]
            
            # Override args with preset values
            if 'filter_type' in preset_config:
                args.filter = preset_config['filter_type'].value
            if 'equalization_method' in preset_config:
                args.equalization = preset_config['equalization_method'].value
            if 'depth_model' in preset_config:
                args.depth_model = preset_config['depth_model']
            
            if args.verbose:
                print(f"Using preset: {args.preset}")
        else:
            print(f"Error: Unknown preset: {args.preset}")
            return
    
    # Prepare output size
    output_size = None
    if args.output_size:
        output_size = tuple(args.output_size)
    
    # Initialize processor
    try:
        processor = RGBToRGBDProcessor(
            filter_type=FilterType(args.filter),
            equalization_method=ColorSpace(args.equalization),
            depth_model=args.depth_model,
            device=args.device,
            output_size=output_size,
            normalize_output=True
        )
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return
    
    # Process images
    if args.batch:
        success = process_batch(args, processor)
    else:
        success = process_single_image(args, processor)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
