import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import logging
from typing import Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import AnomalyDetectionModel, AnomalyDetectionConfig


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection training (normal images only)"""
    
    def __init__(self, data_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 simulate_depth: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.simulate_depth = simulate_depth
        
        # Get list of image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend([
                f for f in os.listdir(data_dir) 
                if f.lower().endswith(ext)
            ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # For now, create dummy RGB-D data
        # In practice, you would load actual RGB-D data
        rgb_image = torch.randn(3, 256, 256)  # Placeholder
        
        if self.simulate_depth:
            # Simulate depth channel (in practice, use real depth data)
            depth_channel = torch.mean(rgb_image, dim=0, keepdim=True) * 0.5
            rgbd_image = torch.cat([rgb_image, depth_channel], dim=0)
        else:
            rgbd_image = rgb_image
        
        if self.transform:
            rgbd_image = self.transform(rgbd_image)
        
        return rgbd_image


class AnomalyTrainer:
    """Trainer class for the anomaly detection model"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = AnomalyDetectionModel(
            input_channels=config.input_channels,
            feature_layers=config.feature_layers,
            feature_size=config.feature_size,
            latent_dim=config.latent_dim,
            input_size=config.input_size,
            reduce_channels=config.reduce_channels,
            reduction_factor=config.reduction_factor
        ).to(self.device)
        
        # Freeze feature extractor (only train CAE)
        self.model.freeze_feature_extractor()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Tracking variables
        self.train_losses = []
        self.best_loss = float('inf')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_data = batch_data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and compute loss
            loss = self.model.compute_loss(batch_data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update tracking
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/num_batches:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                loss = self.model.compute_loss(batch_data)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None,
              save_dir: str = "checkpoints"):
        """Complete training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader)
                self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                current_loss = val_loss
            else:
                self.logger.info(f"Train Loss: {train_loss:.6f}")
                current_loss = train_loss
            
            # Save best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(os.path.join(save_dir, "best_model.pth"))
                self.logger.info(f"New best model saved with loss: {current_loss:.6f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(checkpoint_path)
        
        self.logger.info("Training completed!")
        self.plot_training_curves(save_dir)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_curve.png'))
        plt.close()


def create_data_transforms(input_size: Tuple[int, int] = (256, 256)):
    """Create data transforms"""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], 
                           std=[0.229, 0.224, 0.225, 0.25])  # RGB-D normalization
    ])
    return transform


def main():
    """Main training function"""
    # Configuration
    config = AnomalyDetectionConfig()
    
    # Create transforms
    transform = create_data_transforms(config.input_size)
    
    # Create datasets (you'll need to modify these paths)
    train_dataset = AnomalyDataset(
        data_dir="data/train/normal",  # Update this path
        transform=transform,
        simulate_depth=True
    )
    
    val_dataset = AnomalyDataset(
        data_dir="data/val/normal",  # Update this path
        transform=transform,
        simulate_depth=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = AnomalyTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
