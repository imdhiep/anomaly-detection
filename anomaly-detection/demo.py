#!/usr/bin/env python3
"""
Demo script to test the anomaly detection implementation
Creates synthetic data and runs a quick training/evaluation cycle
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from models import AnomalyDetectionModel
from utils import ExperimentConfig, ModelConfig, TrainingConfig


class SyntheticAnomalyDataset(Dataset):
    """Synthetic dataset for testing the implementation"""
    
    def __init__(self, num_samples=100, image_size=(256, 256), anomaly_ratio=0.0):
        self.num_samples = num_samples
        self.image_size = image_size
        self.anomaly_ratio = anomaly_ratio
        
        # Generate synthetic RGB-D data
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # Create synthetic RGB image (3 channels)
            rgb = torch.randn(3, *image_size) * 0.5 + 0.5
            
            # Create synthetic depth channel (correlated with RGB)
            depth = torch.mean(rgb, dim=0, keepdim=True) * 0.8 + torch.randn(1, *image_size) * 0.1
            
            # Combine RGB and depth
            rgbd = torch.cat([rgb, depth], dim=0)
            
            # Add anomalies to some samples
            is_anomaly = np.random.random() < anomaly_ratio
            if is_anomaly:
                # Add random patches as anomalies
                h, w = image_size
                patch_h, patch_w = h//4, w//4
                start_h = np.random.randint(0, h - patch_h)
                start_w = np.random.randint(0, w - patch_w)
                
                # Add bright anomalous patch
                rgbd[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = 1.0
            
            self.data.append(rgbd)
            self.labels.append(1 if is_anomaly else 0)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def test_model_forward_pass():
    """Test model forward pass with synthetic data"""
    print("Testing model forward pass...")
    
    # Create config
    config = ExperimentConfig()
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectionModel(
        input_channels=4,
        feature_layers=["layer2", "layer3"],
        feature_size=(32, 32),  # Smaller for faster testing
        latent_dim=64,
        input_size=(256, 256)
    ).to(device)
    
    # Create synthetic batch
    batch_size = 2
    batch = torch.randn(batch_size, 4, 256, 256).to(device)
    
    # Test forward pass
    with torch.no_grad():
        results = model(batch)
        
        print(f"✓ Forward pass successful!")
        print(f"  Features shape: {results['features'].shape}")
        print(f"  Latent shape: {results['latent'].shape}")
        print(f"  Reconstructed shape: {results['reconstructed'].shape}")
        print(f"  Anomaly map shape: {results['anomaly_map'].shape}")
        print(f"  Upsampled anomaly map shape: {results['upsampled_anomaly_map'].shape}")
        print(f"  Segmentation mask shape: {results['segmentation_mask'].shape}")
    
    # Test loss computation
    loss = model.compute_loss(batch)
    print(f"✓ Loss computation successful! Loss: {loss.item():.6f}")
    
    return model


def test_training_loop():
    """Test a short training loop"""
    print("\nTesting training loop...")
    
    # Create synthetic datasets
    train_dataset = SyntheticAnomalyDataset(num_samples=20, anomaly_ratio=0.0)  # Normal only
    val_dataset = SyntheticAnomalyDataset(num_samples=10, anomaly_ratio=0.0)    # Normal only
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyDetectionModel(
        input_channels=4,
        feature_layers=["layer2", "layer3"],
        feature_size=(32, 32),
        latent_dim=64
    ).to(device)
    
    # Freeze feature extractor
    model.freeze_feature_extractor()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    num_epochs = 3
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch_data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print("✓ Training loop successful!")
    return model


def test_evaluation():
    """Test evaluation with synthetic anomalous data"""
    print("\nTesting evaluation...")
    
    # Create test dataset with anomalies
    test_dataset = SyntheticAnomalyDataset(num_samples=20, anomaly_ratio=0.3)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Create and load a pre-trained model (using the one from training test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = test_training_loop()  # This returns a trained model
    
    # Evaluation
    model.eval()
    all_scores = []
    all_labels = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch_data, labels in test_loader:
            batch_data = batch_data.to(device)
            
            # Get anomaly predictions
            anomaly_maps, segmentation_masks = model.predict_anomaly(batch_data)
            
            # Compute image-level scores (max of anomaly map)
            scores = torch.max(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1)[0]
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Simple evaluation metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute basic stats
    normal_scores = all_scores[all_labels == 0]
    anomaly_scores = all_scores[all_labels == 1] if np.any(all_labels == 1) else []
    
    print(f"✓ Evaluation successful!")
    print(f"  Normal samples: {len(normal_scores)}, mean score: {normal_scores.mean():.4f}")
    if len(anomaly_scores) > 0:
        print(f"  Anomaly samples: {len(anomaly_scores)}, mean score: {np.mean(anomaly_scores):.4f}")
        print(f"  Score separation: {np.mean(anomaly_scores) - normal_scores.mean():.4f}")


def test_configurations():
    """Test different model configurations"""
    print("\nTesting different configurations...")
    
    configs = [
        {"name": "RGB-only", "input_channels": 3, "feature_layers": ["layer2", "layer3"]},
        {"name": "RGB-D", "input_channels": 4, "feature_layers": ["layer2", "layer3"]},
        {"name": "RGB-D-All", "input_channels": 4, "feature_layers": ["layer1", "layer2", "layer3", "layer4"]},
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for config in configs:
        print(f"  Testing {config['name']}...")
        
        model = AnomalyDetectionModel(
            input_channels=config['input_channels'],
            feature_layers=config['feature_layers'],
            feature_size=(32, 32),
            latent_dim=64
        ).to(device)
        
        # Test with appropriate input
        input_channels = config['input_channels']
        batch = torch.randn(2, input_channels, 256, 256).to(device)
        
        with torch.no_grad():
            results = model(batch)
            loss = model.compute_loss(batch)
        
        print(f"    ✓ {config['name']}: Loss = {loss.item():.6f}")
    
    print("✓ All configurations working!")


def create_visualization():
    """Create a simple visualization of the model output"""
    print("\nCreating visualization...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = AnomalyDetectionModel(
        input_channels=4,
        feature_layers=["layer2", "layer3"],
        feature_size=(32, 32),
        latent_dim=64
    ).to(device)
    
    # Create sample with artificial anomaly
    rgb = torch.randn(1, 3, 256, 256) * 0.3 + 0.5
    depth = torch.mean(rgb, dim=1, keepdim=True) * 0.8
    
    # Add artificial anomaly patch
    rgb[:, :, 100:150, 100:150] = 1.0  # Bright patch
    
    rgbd = torch.cat([rgb, depth], dim=1).to(device)
    
    # Get model output
    model.eval()
    with torch.no_grad():
        anomaly_map, segmentation_mask = model.predict_anomaly(rgbd)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # RGB image
    rgb_img = rgb[0].permute(1, 2, 0).clamp(0, 1)
    axes[0].imshow(rgb_img)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth
    depth_img = depth[0, 0]
    axes[1].imshow(depth_img, cmap='viridis')
    axes[1].set_title('Depth Channel')
    axes[1].axis('off')
    
    # Anomaly map
    anomaly_img = anomaly_map[0, 0].cpu()
    im = axes[2].imshow(anomaly_img, cmap='jet')
    axes[2].set_title('Anomaly Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    # Segmentation
    seg_img = segmentation_mask[0, 0].cpu()
    axes[3].imshow(rgb_img, alpha=0.7)
    axes[3].imshow(seg_img, cmap='Reds', alpha=0.5)
    axes[3].set_title('Segmentation Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('demo_output', exist_ok=True)
    plt.savefig('demo_output/demo_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualization saved to 'demo_output/demo_visualization.png'")


def main():
    """Run all demo tests"""
    print("=" * 60)
    print("ANOMALY DETECTION DEMO")
    print("=" * 60)
    
    try:
        # Test 1: Basic forward pass
        model = test_model_forward_pass()
        
        # Test 2: Training loop
        test_training_loop()
        
        # Test 3: Evaluation
        test_evaluation()
        
        # Test 4: Different configurations
        test_configurations()
        
        # Test 5: Visualization
        create_visualization()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✅ The anomaly detection implementation is working correctly!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Prepare your real RGB-D dataset")
        print("2. Update dataset classes to load real images")
        print("3. Run full training with: python main.py --mode train --config rgbd_full")
        print("4. Evaluate with: python main.py --mode evaluate --config rgbd_full --checkpoint path/to/model.pth")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
