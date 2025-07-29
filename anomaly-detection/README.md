# Anomaly Detection with Modified ResNet-50 and Convolutional Autoencoder

This repository implements a state-of-the-art anomaly detection system that combines:
- **Modified ResNet-50** for RGB-D (4-channel) feature extraction
- **Multi-scale feature aggregation** from intermediate ResNet layers
- **Convolutional Autoencoder (CAE)** using only 1×1 convolutions
- **Anomaly scoring and segmentation** with comprehensive evaluation metrics

## 🏗️ Architecture Overview

### Step 1: Modified ResNet-50 for RGB-D Input
- Load pretrained ResNet-50 from torchvision
- Modify first conv layer: `Conv2d(4, 64, kernel_size=7, stride=2, padding=3)`
- Initialize depth channel with average of RGB channels
- Freeze backbone parameters (feature extractor only)

### Step 2: Multi-Scale Feature Extraction
Extract features from intermediate ResNet layers:
- `layer1`: 256 channels
- `layer2`: 512 channels  
- `layer3`: 1024 channels
- `layer4`: 2048 channels

### Step 3: Feature Alignment and Aggregation
- Resize all feature maps to same spatial resolution (64×64)
- Apply 1×1 convolution for channel reduction
- Concatenate aligned features along channel dimension
- Apply smoothing convolution to reduce noise

### Step 4: Convolutional Autoencoder (CAE)
Architecture using only 1×1 convolutions:
```
Input → Conv1×1 → ReLU → Conv1×1 → ReLU → Conv1×1 (latent) → 
        Conv1×1 → ReLU → Conv1×1 → ReLU → Conv1×1 (output)
```
- Encoder: Compress to latent dimension (default: 128)
- Decoder: Reconstruct original feature representation
- Loss: MSE between original and reconstructed features

### Step 5: Anomaly Detection and Segmentation
- Compute L2 distance between original and reconstructed features
- Generate 2D anomaly heatmap
- Upsample to input image size via bilinear interpolation
- Apply threshold for binary segmentation mask

## 📁 Project Structure

```
anomaly-detection/
├── models/
│   ├── __init__.py
│   ├── modified_resnet.py      # Modified ResNet-50 + Feature Aligner
│   ├── autoencoder.py          # Convolutional Autoencoder + Anomaly Scorer
│   └── anomaly_model.py        # Complete Anomaly Detection Model
├── utils/
│   ├── __init__.py
│   └── config.py               # Configuration management
├── main.py                     # Main script for training/evaluation
├── train.py                    # Training utilities and dataset classes
├── evaluate.py                 # Evaluation utilities and metrics
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd anomaly-detection

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Organize your data in the following structure:
```
data/
├── train/
│   └── normal/          # Normal training images (RGB-D)
├── val/
│   └── normal/          # Normal validation images (RGB-D)
└── test/
    ├── normal/          # Normal test images
    └── anomaly/         # Anomalous test images
```

**Note**: Currently, the code simulates depth channels. For real RGB-D data, modify the dataset classes in `train.py` and `evaluate.py` to load actual depth information.

### Training

```bash
# Train RGB-D model with all feature layers
python main.py --mode train --config rgbd_full --data_dir data/

# Train RGB-only baseline
python main.py --mode train --config rgb_baseline --data_dir data/

# Train with reduced feature layers (layer2 + layer3)
python main.py --mode train --config rgbd_reduced --data_dir data/
```

### Evaluation

```bash
# Evaluate trained model
python main.py --mode evaluate --config rgbd_full --checkpoint experiments/rgbd_full/checkpoints/best_model.pth

# Evaluate with custom data directory
python main.py --mode evaluate --config rgbd_full --checkpoint path/to/model.pth --data_dir custom_data/
```

### Ablation Study

```bash
# Run complete ablation study comparing all configurations
python main.py --mode ablation --data_dir data/
```

## ⚙️ Configuration

The system supports multiple configurations for ablation studies:

### Predefined Configurations

1. **RGB-D Full** (`rgbd_full`)
   - 4-channel input (RGB + Depth)
   - All feature layers (layer1-4)
   - Full channel dimensions

2. **RGB Baseline** (`rgb_baseline`)  
   - 3-channel input (RGB only)
   - All feature layers (layer1-4)

3. **RGB-D Reduced** (`rgbd_reduced`)
   - 4-channel input (RGB + Depth)
   - Reduced feature layers (layer2-3)

### Custom Configuration

Create custom configurations by modifying `utils/config.py`:

```python
from utils import ExperimentConfig, ModelConfig

# Custom model configuration
model_config = ModelConfig(
    input_channels=4,
    feature_layers=["layer2", "layer3", "layer4"],
    latent_dim=256,
    reduce_channels=True,
    reduction_factor=2
)

config = ExperimentConfig(
    model_config=model_config,
    experiment_name="custom_experiment"
)
```

## 📊 Evaluation Metrics

The system computes comprehensive evaluation metrics:

### Image-Level Metrics
- **ROC-AUC**: Area under ROC curve for image-level anomaly detection
- **Anomaly Score**: Maximum value in the anomaly map per image

### Pixel-Level Metrics  
- **Pixel ROC-AUC**: Pixel-wise anomaly detection performance
- **PRO-AUC**: Per-Region Overlap AUC for segmentation quality

### Visualization Outputs
- Original RGB images
- Depth channels (if available)
- Anomaly heatmaps
- Binary segmentation masks
- Overlay visualizations

## 🔬 Experimental Results

The ablation study compares:

| Configuration | Input Type | Feature Layers | Image ROC-AUC | Pixel ROC-AUC | PRO-AUC |
|---------------|------------|----------------|---------------|---------------|---------|
| RGB Baseline  | RGB (3ch)  | layer1-4       | TBD           | TBD           | TBD     |
| RGB-D Full    | RGB-D (4ch)| layer1-4       | TBD           | TBD           | TBD     |
| RGB-D Reduced | RGB-D (4ch)| layer2-3       | TBD           | TBD           | TBD     |

*Results will be populated after running experiments*

## 🛠️ Advanced Features

### Reflection Padding
Enable reflection padding to reduce boundary artifacts:
```python
config.data.use_reflection_padding = True
```

### Experiment Logging
- Automatic experiment directory structure
- Comprehensive logging with timestamps
- Configuration saving and loading
- Training curve visualization
- Device information tracking

### Checkpointing
- Automatic best model saving
- Regular checkpoint intervals
- Resume training from checkpoints
- Model state and optimizer state preservation

## 📈 Training Details

### Default Hyperparameters
- **Batch Size**: 4
- **Learning Rate**: 1e-4  
- **Optimizer**: Adam
- **Epochs**: 700
- **Image Size**: 256×256
- **Feature Map Size**: 64×64
- **Latent Dimension**: 128

### Training Strategy
1. **Freeze ResNet backbone**: Only train the convolutional autoencoder
2. **Normal data only**: Train exclusively on anomaly-free images
3. **MSE Loss**: Reconstruction loss between original and reconstructed features
4. **Early stopping**: Monitor validation loss with patience
5. **Gradient clipping**: Prevent gradient explosion

## 🔧 Customization

### Adding New Feature Extractors
Modify `models/modified_resnet.py` to support other backbones:
```python
class ModifiedBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        # Implement custom backbone
        pass
```

### Custom Loss Functions
Extend `models/autoencoder.py` with additional loss terms:
```python
def custom_loss(self, original, reconstructed):
    mse_loss = F.mse_loss(original, reconstructed)
    # Add perceptual loss, SSIM, etc.
    return total_loss
```

### New Evaluation Metrics
Add metrics in `evaluate.py`:
```python
def compute_custom_metrics(self, predictions, targets):
    # Implement custom evaluation metrics
    return metrics
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Reduce feature map resolution
   - Enable gradient checkpointing

2. **Slow Training**
   - Increase number of workers in DataLoader
   - Use mixed precision training
   - Reduce image resolution

3. **Poor Performance**
   - Check data quality and normalization
   - Experiment with different feature layer combinations
   - Adjust learning rate and batch size
   - Ensure sufficient training data

### Performance Optimization

```python
# Enable mixed precision
config.training.use_mixed_precision = True

# Optimize DataLoader
config.data.num_workers = 8
config.data.pin_memory = True
config.data.prefetch_factor = 2
```

## 📚 References

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Anomaly Detection Survey](https://arxiv.org/abs/2001.05254)
- [Industrial Anomaly Detection](https://arxiv.org/abs/2007.02506)

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

For questions or issues, please open an issue in the repository.
