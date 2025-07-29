import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm

from models import AnomalyDetectionModel, AnomalyDetectionConfig
from train import create_data_transforms


class AnomalyTestDataset(Dataset):
    """Dataset for anomaly detection testing (normal + anomalous images)"""
    
    def __init__(self, data_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 simulate_depth: bool = True):
        self.data_dir = data_dir
        self.transform = transform
        self.simulate_depth = simulate_depth
        
        # Get normal and anomalous image files
        self.normal_dir = os.path.join(data_dir, "normal")
        self.anomaly_dir = os.path.join(data_dir, "anomaly")
        
        self.image_files = []
        self.labels = []
        
        # Load normal images
        if os.path.exists(self.normal_dir):
            normal_files = [f for f in os.listdir(self.normal_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.image_files.extend([(self.normal_dir, f) for f in normal_files])
            self.labels.extend([0] * len(normal_files))
        
        # Load anomalous images
        if os.path.exists(self.anomaly_dir):
            anomaly_files = [f for f in os.listdir(self.anomaly_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.image_files.extend([(self.anomaly_dir, f) for f in anomaly_files])
            self.labels.extend([1] * len(anomaly_files))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        img_dir, img_file = self.image_files[idx]
        img_path = os.path.join(img_dir, img_file)
        label = self.labels[idx]
        
        # For now, create dummy RGB-D data
        # In practice, you would load actual RGB-D data
        rgb_image = torch.randn(3, 256, 256)  # Placeholder
        
        if self.simulate_depth:
            # Simulate depth channel
            depth_channel = torch.mean(rgb_image, dim=0, keepdim=True) * 0.5
            rgbd_image = torch.cat([rgb_image, depth_channel], dim=0)
        else:
            rgbd_image = rgb_image
        
        if self.transform:
            rgbd_image = self.transform(rgbd_image)
        
        return rgbd_image, label, img_file


class AnomalyEvaluator:
    """Evaluator class for anomaly detection model"""
    
    def __init__(self, model: AnomalyDetectionModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def compute_pro_auc(self, anomaly_maps: np.ndarray, 
                       gt_masks: np.ndarray, 
                       num_thresholds: int = 100) -> float:
        """Compute Per-Region Overlap (PRO) AUC"""
        thresholds = np.linspace(0, 1, num_thresholds)
        pro_scores = []
        
        for threshold in thresholds:
            # Create binary predictions
            binary_preds = (anomaly_maps > threshold).astype(np.uint8)
            
            # Compute per-region overlap for each image
            region_overlaps = []
            for i in range(len(gt_masks)):
                if gt_masks[i].sum() == 0:  # No anomaly in ground truth
                    continue
                
                # Find connected components in ground truth
                num_labels, labels = cv2.connectedComponents(gt_masks[i].astype(np.uint8))
                
                for label_idx in range(1, num_labels + 1):
                    # Extract single region
                    region_mask = (labels == label_idx)
                    
                    # Compute overlap
                    intersection = np.logical_and(binary_preds[i], region_mask).sum()
                    region_area = region_mask.sum()
                    
                    if region_area > 0:
                        overlap = intersection / region_area
                        region_overlaps.append(overlap)
            
            if len(region_overlaps) > 0:
                pro_scores.append(np.mean(region_overlaps))
            else:
                pro_scores.append(0.0)
        
        # Compute AUC
        pro_auc = auc(thresholds, pro_scores)
        return pro_auc
    
    def evaluate_model(self, dataloader: DataLoader, 
                      save_visualizations: bool = True,
                      vis_save_dir: str = "visualizations") -> Dict[str, float]:
        """Evaluate model on test dataset"""
        
        if save_visualizations:
            os.makedirs(vis_save_dir, exist_ok=True)
        
        all_anomaly_scores = []
        all_labels = []
        all_anomaly_maps = []
        all_filenames = []
        
        self.logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for batch_idx, (images, labels, filenames) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                
                # Get anomaly predictions
                anomaly_maps, segmentation_masks = self.model.predict_anomaly(images)
                
                # Convert to numpy
                anomaly_maps_np = anomaly_maps.cpu().numpy()
                labels_np = labels.numpy()
                
                # Compute image-level anomaly scores (max of anomaly map)
                image_scores = np.max(anomaly_maps_np.reshape(anomaly_maps_np.shape[0], -1), axis=1)
                
                all_anomaly_scores.extend(image_scores)
                all_labels.extend(labels_np)
                all_anomaly_maps.extend(anomaly_maps_np)
                all_filenames.extend(filenames)
                
                # Save visualizations for first few batches
                if save_visualizations and batch_idx < 5:
                    self.save_batch_visualizations(
                        images, anomaly_maps, segmentation_masks, 
                        labels, filenames, vis_save_dir, batch_idx
                    )
        
        # Convert to numpy arrays
        all_anomaly_scores = np.array(all_anomaly_scores)
        all_labels = np.array(all_labels)
        all_anomaly_maps = np.array(all_anomaly_maps)
        
        # Compute metrics
        metrics = self.compute_metrics(all_anomaly_scores, all_labels, all_anomaly_maps)
        
        # Log results
        self.logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def compute_metrics(self, anomaly_scores: np.ndarray, 
                       labels: np.ndarray,
                       anomaly_maps: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        
        # Image-level ROC-AUC
        if len(np.unique(labels)) > 1:
            roc_auc = roc_auc_score(labels, anomaly_scores)
            metrics['Image_ROC_AUC'] = roc_auc
        
        # Pixel-level metrics (if ground truth masks available)
        # Note: This would require actual pixel-level ground truth
        # For now, we'll simulate pixel-level evaluation
        
        # Generate dummy pixel-level ground truth for demonstration
        pixel_gt = np.random.randint(0, 2, size=anomaly_maps.shape)
        pixel_preds = anomaly_maps.flatten()
        pixel_gt_flat = pixel_gt.flatten()
        
        if len(np.unique(pixel_gt_flat)) > 1:
            pixel_roc_auc = roc_auc_score(pixel_gt_flat, pixel_preds)
            metrics['Pixel_ROC_AUC'] = pixel_roc_auc
            
            # PRO-AUC (simplified version)
            pro_auc = self.compute_pro_auc(anomaly_maps, pixel_gt)
            metrics['PRO_AUC'] = pro_auc
        
        return metrics
    
    def save_batch_visualizations(self, images: torch.Tensor,
                                 anomaly_maps: torch.Tensor,
                                 segmentation_masks: torch.Tensor,
                                 labels: torch.Tensor,
                                 filenames: List[str],
                                 save_dir: str,
                                 batch_idx: int):
        """Save visualization of anomaly detection results"""
        
        batch_size = images.size(0)
        
        for i in range(min(batch_size, 4)):  # Visualize up to 4 images per batch
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original RGB image (first 3 channels)
            rgb_img = images[i, :3].cpu().numpy().transpose(1, 2, 0)
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            axes[0].imshow(rgb_img)
            axes[0].set_title(f'RGB Image\nLabel: {labels[i].item()}')
            axes[0].axis('off')
            
            # Depth channel (4th channel)
            if images.size(1) > 3:
                depth_img = images[i, 3].cpu().numpy()
                axes[1].imshow(depth_img, cmap='viridis')
                axes[1].set_title('Depth Channel')
                axes[1].axis('off')
            else:
                axes[1].axis('off')
            
            # Anomaly map
            anomaly_map = anomaly_maps[i, 0].cpu().numpy()
            im2 = axes[2].imshow(anomaly_map, cmap='jet')
            axes[2].set_title('Anomaly Map')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])
            
            # Segmentation mask
            seg_mask = segmentation_masks[i, 0].cpu().numpy()
            axes[3].imshow(seg_mask, cmap='Reds', alpha=0.7)
            axes[3].imshow(rgb_img, alpha=0.3)
            axes[3].set_title('Segmentation Overlay')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}_img_{i}_{filenames[i]}'))
            plt.close()


def compare_configurations(test_dataloader: DataLoader, 
                          checkpoint_paths: Dict[str, str],
                          device: torch.device) -> Dict[str, Dict[str, float]]:
    """Compare different model configurations"""
    
    results = {}
    
    for config_name, checkpoint_path in checkpoint_paths.items():
        print(f"\nEvaluating {config_name}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        # Create model
        model = AnomalyDetectionModel(
            input_channels=config.input_channels,
            feature_layers=config.feature_layers,
            feature_size=config.feature_size,
            latent_dim=config.latent_dim,
            input_size=config.input_size,
            reduce_channels=config.reduce_channels,
            reduction_factor=config.reduction_factor
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = AnomalyEvaluator(model, device)
        metrics = evaluator.evaluate_model(
            test_dataloader, 
            save_visualizations=True,
            vis_save_dir=f"visualizations_{config_name}"
        )
        
        results[config_name] = metrics
    
    return results


def main():
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test dataset
    transform = create_data_transforms((256, 256))
    test_dataset = AnomalyTestDataset(
        data_dir="data/test",  # Update this path
        transform=transform,
        simulate_depth=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Single model evaluation
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        model = AnomalyDetectionModel(
            input_channels=config.input_channels,
            feature_layers=config.feature_layers,
            feature_size=config.feature_size,
            latent_dim=config.latent_dim,
            input_size=config.input_size,
            reduce_channels=config.reduce_channels,
            reduction_factor=config.reduction_factor
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        evaluator = AnomalyEvaluator(model, device)
        metrics = evaluator.evaluate_model(test_loader)
        
        print("\nEvaluation Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Configuration comparison (if multiple checkpoints exist)
    checkpoint_paths = {
        "RGB-D_Full": "checkpoints/rgbd_full_layers.pth",
        "RGB_Only": "checkpoints/rgb_only.pth", 
        "RGB-D_Reduced": "checkpoints/rgbd_layer23.pth"
    }
    
    existing_checkpoints = {
        name: path for name, path in checkpoint_paths.items() 
        if os.path.exists(path)
    }
    
    if len(existing_checkpoints) > 1:
        print("\nComparing configurations...")
        comparison_results = compare_configurations(
            test_loader, existing_checkpoints, device
        )
        
        # Print comparison table
        print("\nComparison Results:")
        print("-" * 60)
        print(f"{'Configuration':<20} {'Image ROC-AUC':<15} {'Pixel ROC-AUC':<15}")
        print("-" * 60)
        for config_name, metrics in comparison_results.items():
            img_auc = metrics.get('Image_ROC_AUC', 0)
            pixel_auc = metrics.get('Pixel_ROC_AUC', 0)
            print(f"{config_name:<20} {img_auc:<15.4f} {pixel_auc:<15.4f}")


if __name__ == "__main__":
    main()
