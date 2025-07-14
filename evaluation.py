import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from scipy import ndimage


def compute_roc_auc(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    pos_label: int = 1
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC-AUC score and curve.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_scores: Anomaly scores
        pos_label: Label considered as positive
    
    Returns:
        Tuple of (auc_score, fpr, tpr, thresholds)
    """
    y_true_np = y_true.cpu().numpy().flatten()
    y_scores_np = y_scores.cpu().numpy().flatten()
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true_np) | np.isnan(y_scores_np))
    y_true_clean = y_true_np[valid_mask]
    y_scores_clean = y_scores_np[valid_mask]
    
    if len(np.unique(y_true_clean)) < 2:
        print("Warning: Only one class present in y_true. AUC score is not defined.")
        return float('nan'), np.array([]), np.array([]), np.array([])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_clean, y_scores_clean, pos_label=pos_label)
    
    # Compute AUC
    auc_score = roc_auc_score(y_true_clean, y_scores_clean)
    
    return auc_score, fpr, tpr, thresholds


def compute_precision_recall(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    pos_label: int = 1
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall AUC and curve.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Anomaly scores
        pos_label: Label considered as positive
    
    Returns:
        Tuple of (pr_auc, precision, recall, thresholds)
    """
    y_true_np = y_true.cpu().numpy().flatten()
    y_scores_np = y_scores.cpu().numpy().flatten()
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true_np) | np.isnan(y_scores_np))
    y_true_clean = y_true_np[valid_mask]
    y_scores_clean = y_scores_np[valid_mask]
    
    if len(np.unique(y_true_clean)) < 2:
        print("Warning: Only one class present in y_true. PR-AUC score is not defined.")
        return float('nan'), np.array([]), np.array([]), np.array([])
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true_clean, y_scores_clean, pos_label=pos_label)
    
    # Compute PR-AUC
    pr_auc = auc(recall, precision)
    
    return pr_auc, precision, recall, thresholds


def compute_pixel_wise_iou(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: Optional[float] = None
) -> float:
    """
    Compute pixel-wise Intersection over Union (IoU).
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask or scores
        threshold: Threshold for converting scores to binary (if y_pred is not binary)
    
    Returns:
        IoU score
    """
    # Convert to binary if needed
    if threshold is not None:
        y_pred_binary = (y_pred > threshold).float()
    else:
        y_pred_binary = y_pred
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Compute intersection and union
    intersection = (y_true_flat * y_pred_flat).sum()
    union = y_true_flat.sum() + y_pred_flat.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def compute_dice_coefficient(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: Optional[float] = None,
    smooth: float = 1e-8
) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask or scores
        threshold: Threshold for converting scores to binary
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    # Convert to binary if needed
    if threshold is not None:
        y_pred_binary = (y_pred > threshold).float()
    else:
        y_pred_binary = y_pred
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    intersection = (y_true_flat * y_pred_flat).sum()
    dice = (2.0 * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)
    
    return dice.item()


def compute_volume_level_metrics(
    ground_truth_masks: List[torch.Tensor],
    predicted_masks: List[torch.Tensor],
    score_maps: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute volume-level anomaly detection metrics.
    
    Args:
        ground_truth_masks: List of ground truth binary masks
        predicted_masks: List of predicted binary masks
        score_maps: List of anomaly score maps
    
    Returns:
        Dictionary of computed metrics
    """
    volume_labels = []
    volume_scores = []
    
    for gt_mask, pred_mask, score_map in zip(ground_truth_masks, predicted_masks, score_maps):
        # Volume-level label (1 if any anomaly present, 0 otherwise)
        volume_label = 1 if gt_mask.sum() > 0 else 0
        volume_labels.append(volume_label)
        
        # Volume-level score (e.g., maximum anomaly score)
        volume_score = score_map.max().item()
        volume_scores.append(volume_score)
    
    volume_labels = np.array(volume_labels)
    volume_scores = np.array(volume_scores)
    
    # Compute metrics
    if len(np.unique(volume_labels)) > 1:
        volume_auc = roc_auc_score(volume_labels, volume_scores)
    else:
        volume_auc = float('nan')
    
    # Convert scores to binary predictions using median threshold
    threshold = np.median(volume_scores)
    volume_predictions = (volume_scores > threshold).astype(int)
    
    metrics = {
        'volume_auc': volume_auc,
        'volume_accuracy': (volume_labels == volume_predictions).mean(),
        'volume_precision': precision_score(volume_labels, volume_predictions, zero_division=0),
        'volume_recall': recall_score(volume_labels, volume_predictions, zero_division=0),
        'volume_f1': f1_score(volume_labels, volume_predictions, zero_division=0)
    }
    
    return metrics


def comprehensive_evaluation(
    ground_truth_masks: List[torch.Tensor],
    score_maps: List[torch.Tensor],
    threshold: float,
    compute_volume_metrics: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        ground_truth_masks: List of ground truth binary masks
        score_maps: List of anomaly score maps
        threshold: Threshold for converting scores to binary predictions
        compute_volume_metrics: Whether to compute volume-level metrics
    
    Returns:
        Dictionary of evaluation metrics
    """
    all_metrics = {}
    
    # Pixel-level metrics
    all_gt = torch.cat([mask.flatten() for mask in ground_truth_masks])
    all_scores = torch.cat([score.flatten() for score in score_maps])
    all_pred = (all_scores > threshold).float()
    
    # ROC-AUC
    if len(torch.unique(all_gt)) > 1:
        pixel_auc, _, _, _ = compute_roc_auc(all_gt, all_scores)
        pr_auc, _, _, _ = compute_precision_recall(all_gt, all_scores)
    else:
        pixel_auc = float('nan')
        pr_auc = float('nan')
    
    # Pixel-wise metrics
    pixel_iou = compute_pixel_wise_iou(all_gt, all_pred)
    pixel_dice = compute_dice_coefficient(all_gt, all_pred)
    
    # Classification metrics
    gt_np = all_gt.cpu().numpy()
    pred_np = all_pred.cpu().numpy()
    
    pixel_precision = precision_score(gt_np, pred_np, zero_division=0)
    pixel_recall = recall_score(gt_np, pred_np, zero_division=0)
    pixel_f1 = f1_score(gt_np, pred_np, zero_division=0)
    pixel_accuracy = (gt_np == pred_np).mean()
    
    all_metrics.update({
        'pixel_auc': pixel_auc,
        'pixel_pr_auc': pr_auc,
        'pixel_iou': pixel_iou,
        'pixel_dice': pixel_dice,
        'pixel_precision': pixel_precision,
        'pixel_recall': pixel_recall,
        'pixel_f1': pixel_f1,
        'pixel_accuracy': pixel_accuracy
    })
    
    # Volume-level metrics
    if compute_volume_metrics:
        predicted_masks = [(score > threshold).float() for score in score_maps]
        volume_metrics = compute_volume_level_metrics(ground_truth_masks, predicted_masks, score_maps)
        all_metrics.update(volume_metrics)
    
    # Per-volume metrics
    per_volume_ious = []
    per_volume_dices = []
    
    for gt_mask, score_map in zip(ground_truth_masks, score_maps):
        pred_mask = (score_map > threshold).float()
        iou = compute_pixel_wise_iou(gt_mask, pred_mask)
        dice = compute_dice_coefficient(gt_mask, pred_mask)
        per_volume_ious.append(iou)
        per_volume_dices.append(dice)
    
    all_metrics.update({
        'mean_volume_iou': np.mean(per_volume_ious),
        'std_volume_iou': np.std(per_volume_ious),
        'mean_volume_dice': np.mean(per_volume_dices),
        'std_volume_dice': np.std(per_volume_dices)
    })
    
    return all_metrics


def plot_roc_curves(
    results: List[Tuple[str, torch.Tensor, torch.Tensor]],
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multiple methods.
    
    Args:
        results: List of (method_name, y_true, y_scores) tuples
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    for method_name, y_true, y_scores in results:
        auc_score, fpr, tpr, _ = compute_roc_auc(y_true, y_scores)
        if not np.isnan(auc_score):
            plt.plot(fpr, tpr, label=f'{method_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curves(
    results: List[Tuple[str, torch.Tensor, torch.Tensor]],
    save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curves for multiple methods.
    
    Args:
        results: List of (method_name, y_true, y_scores) tuples
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    for method_name, y_true, y_scores in results:
        pr_auc, precision, recall, _ = compute_precision_recall(y_true, y_scores)
        if not np.isnan(pr_auc):
            plt.plot(recall, precision, label=f'{method_name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_score_distributions(
    normal_scores: torch.Tensor,
    anomaly_scores: torch.Tensor,
    threshold: Optional[float] = None,
    bins: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot distributions of normal and anomaly scores.
    
    Args:
        normal_scores: Scores from normal samples
        anomaly_scores: Scores from anomalous samples
        threshold: Threshold value to visualize
        bins: Number of histogram bins
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    normal_np = normal_scores.cpu().numpy().flatten()
    anomaly_np = anomaly_scores.cpu().numpy().flatten()
    
    plt.hist(normal_np, bins=bins, alpha=0.7, label='Normal', density=True)
    plt.hist(anomaly_np, bins=bins, alpha=0.7, label='Anomaly', density=True)
    
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_optimal_threshold(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold based on a specific metric.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Anomaly scores
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
    
    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    y_true_np = y_true.cpu().numpy().flatten()
    y_scores_np = y_scores.cpu().numpy().flatten()
    
    # Get thresholds from ROC curve
    _, _, thresholds = roc_curve(y_true_np, y_scores_np)
    
    best_threshold = thresholds[0]
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_scores_np > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true_np, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true_np, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true_np, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = (y_true_np == y_pred).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score
