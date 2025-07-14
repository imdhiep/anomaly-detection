import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Optional, Callable, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


class FeatureVectorDataset(Dataset):
    """Dataset for feature vectors."""
    
    def __init__(self, feature_vectors: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            feature_vectors: Tensor of shape (N, feature_dim)
        """
        self.feature_vectors = feature_vectors
    
    def __len__(self):
        return len(self.feature_vectors)
    
    def __getitem__(self, idx):
        return self.feature_vectors[idx]


def train_autoencoder(
    autoencoder: nn.Module,
    train_features: torch.Tensor,
    val_features: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    scheduler_type: str = 'step',
    scheduler_params: Optional[Dict] = None,
    early_stopping_patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: Optional[str] = None,
    plot_losses: bool = True
) -> Dict[str, List[float]]:
    """
    Training loop for autoencoder on normal samples only.
    
    Args:
        autoencoder: Autoencoder model
        train_features: Training feature vectors (N, feature_dim)
        val_features: Validation feature vectors (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        scheduler_type: Type of learning rate scheduler ('step', 'cosine', 'plateau')
        scheduler_params: Parameters for scheduler
        early_stopping_patience: Patience for early stopping
        device: Device to train on
        save_path: Path to save best model
        plot_losses: Whether to plot training curves
    
    Returns:
        Dictionary containing training history
    """
    # Move model to device
    autoencoder.to(device)
    train_features = train_features.to(device)
    if val_features is not None:
        val_features = val_features.to(device)
    
    # Create datasets and dataloaders
    train_dataset = FeatureVectorDataset(train_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_features is not None:
        val_dataset = FeatureVectorDataset(val_features)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize scheduler
    if scheduler_params is None:
        scheduler_params = {}
    
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, **scheduler_params)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **scheduler_params)
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        autoencoder.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_features in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_features = batch_features.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(autoencoder, 'loss_function'):
                # For VAE
                reconstructed, mu, logvar = autoencoder(batch_features)
                loss, recon_loss, kl_loss = autoencoder.loss_function(
                    batch_features, reconstructed, mu, logvar
                )
            else:
                # For standard autoencoder
                reconstructed = autoencoder(batch_features)
                loss = nn.MSELoss()(reconstructed, batch_features)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_features is not None:
            autoencoder.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_features in val_loader:
                    batch_features = batch_features.to(device)
                    
                    if hasattr(autoencoder, 'loss_function'):
                        # For VAE
                        reconstructed, mu, logvar = autoencoder(batch_features)
                        loss, _, _ = autoencoder.loss_function(
                            batch_features, reconstructed, mu, logvar
                        )
                    else:
                        # For standard autoencoder
                        reconstructed = autoencoder(batch_features)
                        loss = nn.MSELoss()(reconstructed, batch_features)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            history['val_loss'].append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = autoencoder.state_dict().copy()
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Update scheduler
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Update scheduler
            if scheduler is not None and scheduler_type != 'plateau':
                scheduler.step()
        
        # Record learning rate
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping
        if val_features is not None and patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model if validation was used
    if val_features is not None and best_model_state is not None:
        autoencoder.load_state_dict(best_model_state)
        print(f'Loaded best model with validation loss: {best_val_loss:.6f}')
    
    # Save model
    if save_path is not None:
        torch.save({
            'model_state_dict': autoencoder.state_dict(),
            'model_config': {
                'input_dim': autoencoder.input_dim,
                'hidden_dims': autoencoder.hidden_dims,
                'latent_dim': autoencoder.latent_dim,
            },
            'training_history': history
        }, save_path)
        print(f'Model saved to {save_path}')
    
    # Plot training curves
    if plot_losses:
        plot_training_curves(history)
    
    return history


def plot_training_curves(history: Dict[str, List[float]]):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Training Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Learning rate curve
    axes[1].plot(history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def load_trained_autoencoder(model_path: str, autoencoder_class) -> nn.Module:
    """Load a trained autoencoder from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with saved configuration
    model_config = checkpoint['model_config']
    autoencoder = autoencoder_class(**model_config)
    
    # Load weights
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    
    return autoencoder


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_begin(self, epoch: int, model: nn.Module):
        pass
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_loss: float, val_loss: float = None):
        pass
    
    def on_batch_begin(self, batch_idx: int, model: nn.Module):
        pass
    
    def on_batch_end(self, batch_idx: int, model: nn.Module, loss: float):
        pass


class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints during training."""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = float('inf') if 'loss' in monitor else float('-inf')
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_loss: float, val_loss: float = None):
        if self.monitor == 'train_loss':
            current_score = train_loss
        elif self.monitor == 'val_loss' and val_loss is not None:
            current_score = val_loss
        else:
            return
        
        if not self.save_best_only or current_score < self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), self.filepath)
            print(f'Model checkpoint saved to {self.filepath}')


class LossLogger(TrainingCallback):
    """Log losses during training."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.losses = []
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_loss: float, val_loss: float = None):
        log_entry = f'Epoch {epoch}: Train Loss = {train_loss:.6f}'
        if val_loss is not None:
            log_entry += f', Val Loss = {val_loss:.6f}'
        
        self.losses.append(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
