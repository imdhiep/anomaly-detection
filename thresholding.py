import torch
import numpy as np
from typing import Tuple, Optional, List, Union
from scipy import stats
import matplotlib.pyplot as plt


def compute_threshold_from_normal_scores(
    normal_scores: torch.Tensor,
    method: str = 'percentile',
    percentile: float = 95.0,
    sigma_multiplier: float = 3.0,
    contamination_rate: Optional[float] = None
) -> float:
    """
    Compute anomaly threshold from validation set of normal scores.
    
    Args:
        normal_scores: Tensor of reconstruction scores from normal samples
        method: Thresholding method ('percentile', 'sigma', 'mad', 'iqr', 'contamination')
        percentile: Percentile for threshold (e.g., 95th percentile)
        sigma_multiplier: Number of standard deviations for sigma method
        contamination_rate: Expected contamination rate for contamination method
    
    Returns:
        Computed threshold value
    """
    normal_scores_np = normal_scores.cpu().numpy().flatten()
    
    if method == 'percentile':
        threshold = np.percentile(normal_scores_np, percentile)
    
    elif method == 'sigma':
        mean_score = np.mean(normal_scores_np)
        std_score = np.std(normal_scores_np)
        threshold = mean_score + sigma_multiplier * std_score
    
    elif method == 'mad':
        # Median Absolute Deviation
        median_score = np.median(normal_scores_np)
        mad = np.median(np.abs(normal_scores_np - median_score))
        threshold = median_score + sigma_multiplier * 1.4826 * mad  # 1.4826 makes MAD consistent with std
    
    elif method == 'iqr':
        # Interquartile Range
        q75, q25 = np.percentile(normal_scores_np, [75, 25])
        iqr = q75 - q25
        threshold = q75 + 1.5 * iqr  # Standard outlier detection
    
    elif method == 'contamination':
        if contamination_rate is None:
            raise ValueError("contamination_rate must be specified for contamination method")
        threshold_percentile = (1 - contamination_rate) * 100
        threshold = np.percentile(normal_scores_np, threshold_percentile)
    
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    return float(threshold)


def apply_threshold(
    score_map: torch.Tensor,
    threshold: float,
    binary_output: bool = True
) -> torch.Tensor:
    """
    Apply threshold to anomaly score map to generate binary anomaly mask.
    
    Args:
        score_map: 3D anomaly score map
        threshold: Threshold value
        binary_output: Whether to return binary mask or thresholded scores
    
    Returns:
        Binary anomaly mask or thresholded scores
    """
    if binary_output:
        return (score_map > threshold).float()
    else:
        return torch.clamp(score_map - threshold, min=0)


def adaptive_threshold(
    score_map: torch.Tensor,
    method: str = 'otsu',
    bins: int = 256
) -> Tuple[torch.Tensor, float]:
    """
    Apply adaptive thresholding to score map.
    
    Args:
        score_map: 3D anomaly score map
        method: Thresholding method ('otsu', 'triangle', 'yen')
        bins: Number of bins for histogram computation
    
    Returns:
        Tuple of (binary_mask, threshold_value)
    """
    scores_flat = score_map.flatten().numpy()
    
    if method == 'otsu':
        threshold = _otsu_threshold(scores_flat, bins)
    elif method == 'triangle':
        threshold = _triangle_threshold(scores_flat, bins)
    elif method == 'yen':
        threshold = _yen_threshold(scores_flat, bins)
    else:
        raise ValueError(f"Unknown adaptive method: {method}")
    
    binary_mask = apply_threshold(score_map, threshold, binary_output=True)
    
    return binary_mask, threshold


def _otsu_threshold(scores: np.ndarray, bins: int) -> float:
    """Compute Otsu's threshold."""
    hist, bin_edges = np.histogram(scores, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(float) / hist.sum()
    
    # Compute cumulative sums
    w0 = np.cumsum(hist)
    w1 = 1 - w0
    
    # Compute cumulative means
    mu0 = np.cumsum(hist * bin_centers) / w0
    mu1 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / w1
    
    # Compute between-class variance
    variance_between = w0 * w1 * (mu0 - mu1) ** 2
    
    # Find threshold that maximizes between-class variance
    idx = np.argmax(variance_between)
    threshold = bin_centers[idx]
    
    return threshold


def _triangle_threshold(scores: np.ndarray, bins: int) -> float:
    """Compute triangle threshold."""
    hist, bin_edges = np.histogram(scores, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find the peak
    peak_idx = np.argmax(hist)
    
    # Find the point on the tail with maximum distance from the line
    # connecting peak to end
    if peak_idx == len(hist) - 1:
        # Peak is at the end, use the other direction
        end_idx = 0
        search_range = range(peak_idx)
    else:
        end_idx = len(hist) - 1
        search_range = range(peak_idx + 1, len(hist))
    
    max_distance = 0
    threshold_idx = peak_idx
    
    for i in search_range:
        # Distance from point to line
        distance = _point_to_line_distance(
            i, hist[i],
            peak_idx, hist[peak_idx],
            end_idx, hist[end_idx]
        )
        
        if distance > max_distance:
            max_distance = distance
            threshold_idx = i
    
    return bin_centers[threshold_idx]


def _yen_threshold(scores: np.ndarray, bins: int) -> float:
    """Compute Yen's threshold."""
    hist, bin_edges = np.histogram(scores, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(float)
    hist = hist / hist.sum()
    
    # Compute cumulative histograms
    P1 = np.cumsum(hist)
    P2 = np.cumsum(hist[::-1])[::-1]
    
    # Compute entropies
    def entropy(p):
        p = p[p > 0]  # Remove zeros
        return -np.sum(p * np.log(p))
    
    max_criterion = -np.inf
    threshold_idx = 0
    
    for i in range(1, len(hist) - 1):
        if P1[i] > 0 and P2[i] > 0:
            # Compute criterion
            criterion = np.log(P1[i] * P2[i]) + entropy(hist[:i+1] / P1[i]) + entropy(hist[i+1:] / P2[i])
            
            if criterion > max_criterion:
                max_criterion = criterion
                threshold_idx = i
    
    return bin_centers[threshold_idx]


def _point_to_line_distance(px: int, py: float, x1: int, y1: float, x2: int, y2: float) -> float:
    """Compute distance from point to line."""
    if x1 == x2:
        return abs(px - x1)
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    return abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)


def multi_threshold_analysis(
    score_map: torch.Tensor,
    threshold_range: Tuple[float, float],
    num_thresholds: int = 50,
    return_stats: bool = True
) -> dict:
    """
    Analyze anomaly detection performance across multiple thresholds.
    
    Args:
        score_map: 3D anomaly score map
        threshold_range: Range of thresholds to test
        num_thresholds: Number of thresholds to test
        return_stats: Whether to compute statistics for each threshold
    
    Returns:
        Dictionary containing threshold analysis results
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    results = {
        'thresholds': thresholds,
        'anomaly_volumes': [],
        'max_component_sizes': [],
        'num_components': []
    }
    
    if return_stats:
        results.update({
            'mean_scores': [],
            'std_scores': [],
            'coverage_ratios': []
        })
    
    for threshold in thresholds:
        binary_mask = apply_threshold(score_map, threshold)
        
        # Basic statistics
        anomaly_volume = binary_mask.sum().item()
        results['anomaly_volumes'].append(anomaly_volume)
        
        # Connected component analysis
        binary_np = binary_mask.numpy().astype(bool)
        labeled_array, num_features = ndimage.label(binary_np)
        
        if num_features > 0:
            component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
            max_component_size = max(component_sizes)
        else:
            max_component_size = 0
        
        results['num_components'].append(num_features)
        results['max_component_sizes'].append(max_component_size)
        
        if return_stats:
            # Additional statistics
            anomaly_scores = score_map[binary_mask.bool()]
            if len(anomaly_scores) > 0:
                results['mean_scores'].append(anomaly_scores.mean().item())
                results['std_scores'].append(anomaly_scores.std().item())
            else:
                results['mean_scores'].append(0.0)
                results['std_scores'].append(0.0)
            
            coverage_ratio = anomaly_volume / score_map.numel()
            results['coverage_ratios'].append(coverage_ratio)
    
    return results


def morphological_postprocessing(
    binary_mask: torch.Tensor,
    min_component_size: int = 10,
    erosion_iterations: int = 1,
    dilation_iterations: int = 2
) -> torch.Tensor:
    """
    Apply morphological post-processing to binary anomaly mask.
    
    Args:
        binary_mask: Binary anomaly mask
        min_component_size: Minimum size for connected components
        erosion_iterations: Number of erosion iterations
        dilation_iterations: Number of dilation iterations
    
    Returns:
        Post-processed binary mask
    """
    from scipy import ndimage
    
    mask_np = binary_mask.numpy().astype(bool)
    
    # Remove small components
    if min_component_size > 1:
        labeled_array, num_features = ndimage.label(mask_np)
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_array == i)
            if component_size < min_component_size:
                mask_np[labeled_array == i] = False
    
    # Morphological operations
    if erosion_iterations > 0:
        mask_np = ndimage.binary_erosion(mask_np, iterations=erosion_iterations)
    
    if dilation_iterations > 0:
        mask_np = ndimage.binary_dilation(mask_np, iterations=dilation_iterations)
    
    return torch.from_numpy(mask_np.astype(np.float32))


def visualize_threshold_analysis(results: dict, save_path: Optional[str] = None):
    """Visualize threshold analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    thresholds = results['thresholds']
    
    # Anomaly volume vs threshold
    axes[0, 0].plot(thresholds, results['anomaly_volumes'])
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Anomaly Volume')
    axes[0, 0].set_title('Anomaly Volume vs Threshold')
    axes[0, 0].grid(True)
    
    # Number of components vs threshold
    axes[0, 1].plot(thresholds, results['num_components'])
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Number of Components')
    axes[0, 1].set_title('Components vs Threshold')
    axes[0, 1].grid(True)
    
    # Max component size vs threshold
    axes[1, 0].plot(thresholds, results['max_component_sizes'])
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Max Component Size')
    axes[1, 0].set_title('Max Component Size vs Threshold')
    axes[1, 0].grid(True)
    
    # Coverage ratio (if available)
    if 'coverage_ratios' in results:
        axes[1, 1].plot(thresholds, results['coverage_ratios'])
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Coverage Ratio')
        axes[1, 1].set_title('Coverage Ratio vs Threshold')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
