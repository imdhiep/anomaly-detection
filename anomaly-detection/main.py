#!/usr/bin/env python3
"""
Anomaly Detection with Modified ResNet-50 and Convolutional Autoencoder

This script implements a complete anomaly detection system that:
1. Uses a modified ResNet-50 to accept RGB-D (4-channel) input
2. Extracts multi-scale features from intermediate layers
3. Trains a convolutional autoencoder using only 1x1 convolutions
4. Generates anomaly maps and segmentation masks
5. Evaluates performance with ROC-AUC and PRO-AUC metrics

Usage:
    python main.py --mode train --config rgbd_full
    python main.py --mode evaluate --config rgbd_full --checkpoint checkpoints/best_model.pth
    python main.py --mode ablation --data_dir data/
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import AnomalyDetectionModel
from train import AnomalyTrainer, AnomalyDataset, create_data_transforms
from evaluate import AnomalyEvaluator, AnomalyTestDataset, compare_configurations
from utils import (
    ExperimentConfig, Logger, setup_experiment_directory,
    save_experiment_summary, load_experiment_config, 
    create_ablation_configs, get_device_info
)


def setup_data_loaders(config: ExperimentConfig) -> tuple:
    """Setup training, validation, and test data loaders"""
    
    # Create transforms
    transform = create_data_transforms(config.model.input_size)
    
    # Training dataset (normal images only)
    train_dataset = AnomalyDataset(
        data_dir=config.data.train_normal_dir,
        transform=transform,
        simulate_depth=config.data.simulate_depth
    )
    
    # Validation dataset (normal images only)
    val_dataset = AnomalyDataset(
        data_dir=config.data.val_normal_dir,
        transform=transform,
        simulate_depth=config.data.simulate_depth
    )
    
    # Test dataset (normal + anomalous images)
    test_dataset = AnomalyTestDataset(
        data_dir=config.data.test_dir,
        transform=transform,
        simulate_depth=config.data.simulate_depth
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_model(config: ExperimentConfig, device: torch.device) -> AnomalyDetectionModel:
    """Create and initialize the anomaly detection model"""
    
    model = AnomalyDetectionModel(
        input_channels=config.model.input_channels,
        feature_layers=config.model.feature_layers,
        feature_size=config.model.feature_size,
        latent_dim=config.model.latent_dim,
        input_size=config.model.input_size,
        reduce_channels=config.model.reduce_channels,
        reduction_factor=config.model.reduction_factor
    ).to(device)
    
    return model


def train_model(config: ExperimentConfig, experiment_dir: str) -> str:
    """Train the anomaly detection model"""
    
    # Setup logging
    logger = Logger(os.path.join(experiment_dir, "logs"), config.experiment_name)
    logger.log_config(config)
    
    # Setup device
    device = torch.device(config.training.device)
    logger.logger.info(f"Using device: {device}")
    
    # Log device info
    device_info = get_device_info()
    logger.logger.info(f"Device info: {device_info}")
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_data_loaders(config)
    logger.logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config, device)
    logger.logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create trainer
    # Convert config to the expected format for trainer
    from utils.config import TrainingConfig
    trainer_config = type('Config', (), {})()
    for attr, value in config.training.__dict__.items():
        setattr(trainer_config, attr, value)
    for attr, value in config.model.__dict__.items():
        setattr(trainer_config, attr, value)
    
    trainer = AnomalyTrainer(trainer_config)
    trainer.model = model
    trainer.device = device
    
    # Train model
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    trainer.train(train_loader, val_loader, checkpoints_dir)
    
    # Save final config
    config.save_config(os.path.join(experiment_dir, "configs", "final_config.yaml"))
    
    best_checkpoint = os.path.join(checkpoints_dir, "best_model.pth")
    return best_checkpoint


def evaluate_model(config: ExperimentConfig, checkpoint_path: str, 
                  experiment_dir: str) -> dict:
    """Evaluate the trained model"""
    
    # Setup logging
    logger = Logger(os.path.join(experiment_dir, "logs"), config.experiment_name)
    
    # Setup device
    device = torch.device(config.training.device)
    
    # Setup test data loader
    _, _, test_loader = setup_data_loaders(config)
    logger.logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = AnomalyEvaluator(model, device)
    
    # Evaluate model
    vis_dir = os.path.join(experiment_dir, "visualizations")
    metrics = evaluator.evaluate_model(
        test_loader,
        save_visualizations=config.evaluation.save_visualizations,
        vis_save_dir=vis_dir
    )
    
    # Log results
    logger.log_metrics(metrics, "Evaluation - ")
    
    # Save results
    save_experiment_summary(config, metrics, experiment_dir)
    
    return metrics


def run_ablation_study(base_data_dir: str, base_experiment_dir: str = "experiments"):
    """Run ablation study with different configurations"""
    
    print("Running ablation study...")
    print("=" * 50)
    
    # Get all ablation configurations
    ablation_configs = create_ablation_configs()
    
    results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"\nRunning experiment: {config_name}")
        print("-" * 30)
        
        # Update data directories
        config.data.train_normal_dir = os.path.join(base_data_dir, "train", "normal")
        config.data.val_normal_dir = os.path.join(base_data_dir, "val", "normal") 
        config.data.test_dir = os.path.join(base_data_dir, "test")
        
        # Setup experiment directory
        experiment_dir = setup_experiment_directory(
            config_name, base_experiment_dir
        )
        
        try:
            # Train model
            checkpoint_path = train_model(config, experiment_dir)
            
            # Evaluate model
            metrics = evaluate_model(config, checkpoint_path, experiment_dir)
            results[config_name] = metrics
            
        except Exception as e:
            print(f"Error in experiment {config_name}: {str(e)}")
            continue
    
    # Print comparison results
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Image ROC-AUC':<15} {'Pixel ROC-AUC':<15} {'PRO-AUC':<15}")
    print("-" * 80)
    
    for config_name, metrics in results.items():
        img_auc = metrics.get('Image_ROC_AUC', 0)
        pixel_auc = metrics.get('Pixel_ROC_AUC', 0)
        pro_auc = metrics.get('PRO_AUC', 0)
        print(f"{config_name:<20} {img_auc:<15.4f} {pixel_auc:<15.4f} {pro_auc:<15.4f}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Anomaly Detection with ResNet-50 + CAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train RGB-D model
    python main.py --mode train --config rgbd_full --data_dir data/
    
    # Train RGB-only model
    python main.py --mode train --config rgb_baseline --data_dir data/
    
    # Evaluate trained model
    python main.py --mode evaluate --config rgbd_full --checkpoint checkpoints/best_model.pth
    
    # Run complete ablation study
    python main.py --mode ablation --data_dir data/
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["train", "evaluate", "ablation"],
        required=True,
        help="Mode to run: train, evaluate, or ablation"
    )
    
    parser.add_argument(
        "--config",
        default="rgbd_full",
        help="Configuration name (rgbd_full, rgb_baseline, rgbd_reduced)"
    )
    
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Base directory containing train/val/test subdirectories"
    )
    
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint (required for evaluate mode)"
    )
    
    parser.add_argument(
        "--experiment_dir",
        default="experiments",
        help="Base directory for experiment outputs"
    )
    
    args = parser.parse_args()
    
    if args.mode == "ablation":
        # Run ablation study
        run_ablation_study(args.data_dir, args.experiment_dir)
    
    else:
        # Load configuration
        try:
            config = load_experiment_config(args.config)
        except ValueError:
            print(f"Unknown config: {args.config}")
            print(f"Available configs: {list(load_experiment_config.__globals__['EXPERIMENT_CONFIGS'].keys())}")
            return
        
        # Update data directories
        config.data.train_normal_dir = os.path.join(args.data_dir, "train", "normal")
        config.data.val_normal_dir = os.path.join(args.data_dir, "val", "normal")
        config.data.test_dir = os.path.join(args.data_dir, "test")
        
        # Setup experiment directory
        experiment_dir = setup_experiment_directory(
            config.experiment_name, args.experiment_dir
        )
        
        if args.mode == "train":
            print(f"Training model with config: {args.config}")
            checkpoint_path = train_model(config, experiment_dir)
            print(f"Training completed. Best model saved at: {checkpoint_path}")
            
        elif args.mode == "evaluate":
            if not args.checkpoint:
                print("Error: --checkpoint is required for evaluate mode")
                return
            
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint file not found: {args.checkpoint}")
                return
            
            print(f"Evaluating model: {args.checkpoint}")
            metrics = evaluate_model(config, args.checkpoint, experiment_dir)
            
            print("\nEvaluation Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()