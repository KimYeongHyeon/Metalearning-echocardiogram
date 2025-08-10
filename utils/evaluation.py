"""
Evaluation metrics and utilities for landmark detection models.
This module provides functionality to evaluate model predictions against ground truth landmarks,
including distance error calculations and spatial angular similarity measurements.
"""

import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional


# ------------------- Coordinate Error Metrics -------------------
def _calculate_rmse_per_coordinate(labels: np.ndarray, preds: np.ndarray, batch_size: int) -> np.ndarray:
    error = labels - preds
    dist_sq = np.mean(error.reshape(-1, 2) ** 2, axis=1)
    dist_sq_by_coord = np.mean(dist_sq.reshape(batch_size, -1), axis=0)
    return np.sqrt(dist_sq_by_coord)

def distance_error(sample: Dict, preds: Union[torch.Tensor, np.ndarray], order: str = 'xyxy') -> np.ndarray:
    if order != 'xyxy':
        raise NotImplementedError("Only 'xyxy' coordinate format is currently supported.")

    labels_data = sample.get('label')
    if labels_data is None:
        raise KeyError("Sample dictionary must contain 'label' key.")

    processed_labels = labels_data.detach().cpu().numpy() if isinstance(labels_data, torch.Tensor) else np.asarray(labels_data)
    processed_preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)

    if processed_labels.shape != processed_preds.shape:
        raise ValueError(f"Shape mismatch: {processed_labels.shape} vs {processed_preds.shape}")

    if processed_labels.size == 0:
        return np.array([])

    batch_size = processed_labels.shape[0]
    if batch_size == 0:
        return np.array([])

    return _calculate_rmse_per_coordinate(processed_labels, processed_preds, batch_size)


# def distance_error(sample: Dict, preds: Union[torch.Tensor, np.ndarray], order: str = 'xyxy') -> np.ndarray:
#     """Root mean squared distance error per coordinate type."""
#     if order != 'xyxy':
#         raise NotImplementedError("Only 'xyxy' coordinate format is currently supported.")
    
#     labels = sample.get('label')
#     if labels is None:
#         raise KeyError("Sample dictionary must contain 'label' key.")
    
#     # Convert to NumPy
#     labels = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
#     preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    
#     if labels.shape != preds.shape:
#         raise ValueError(f"Shape mismatch: {labels.shape} vs {preds.shape}")
    
#     batch_size = labels.shape[0]
#     error = labels - preds

#     # Mean squared error per (x, y) pair
#     dist_sq = np.mean(error.reshape(-1, 2) ** 2, axis=1)
#     dist_sq_by_coord = np.mean(dist_sq.reshape(batch_size, -1), axis=0)
    
#     return np.sqrt(dist_sq_by_coord)

# ------------------- Spatial Angular Similarity Metrics -------------------

def calculate_spatial_angular_similarity(
    line1: List[Tuple[float, float]], 
    line2: List[Tuple[float, float]], 
    alpha: float = 0.5, 
    beta: float = 0.1
) -> float:
    """Calculate spatial angular similarity between two line segments.
    
    Args:
        line1: First line segment in format [(x1, y1), (x2, y2)]
        line2: Second line segment in format [(x1, y1), (x2, y2)]
        alpha: Direction similarity weight (0-1)
        beta: Distance decay parameter
        
    Returns:
        Similarity score between 0-100
    """
    # Calculate direction vectors
    v1 = np.array([line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]])
    v2 = np.array([line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]])
    
    # Normalize direction vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0  # Zero-length lines have zero similarity
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate cosine similarity
    cos_theta = np.abs(np.dot(v1, v2))
    
    # Calculate minimum distance between lines
    d = minimum_distance_between_lines(line1, line2)
    
    # Calculate distance similarity
    distance_similarity = math.exp(-beta * float(d))

    
    # Calculate spatial angular similarity
    similarity = 100 * (alpha * cos_theta + (1 - alpha) * distance_similarity)
    
    return similarity


def minimum_distance_between_lines(
    line1: List[Tuple[float, float]], 
    line2: List[Tuple[float, float]]
) -> float:
    """Calculate minimum distance between two line segments.
    
    Args:
        line1: First line segment in format [(x1, y1), (x2, y2)]
        line2: Second line segment in format [(x1, y1), (x2, y2)]
        
    Returns:
        Minimum distance between the lines
    """
    # Extract line endpoints
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    p3, p4 = np.array(line2[0]), np.array(line2[1])
    
    # Direction vectors
    v1 = p2 - p1
    v2 = p4 - p3
    
    # Handle parallel lines
    cross_product = np.cross(v1, v2)
    if abs(cross_product) < 1e-10:
        # Calculate distance between points and lines
        d1 = point_to_line_distance(p1, p3, p4)
        d2 = point_to_line_distance(p2, p3, p4)
        d3 = point_to_line_distance(p3, p1, p2)
        d4 = point_to_line_distance(p4, p1, p2)
        return min(d1, d2, d3, d4)
    
    # Calculate minimum distance between non-parallel lines
    # normal = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    distance = np.dot(p3 - p1, normal)
    return float(np.linalg.norm(distance))


def point_to_line_distance(
    point: np.ndarray, 
    line_start: np.ndarray, 
    line_end: np.ndarray
) -> float:
    """Calculate distance from a point to a line segment.
    
    Args:
        point: Point coordinates
        line_start: Line segment start point
        line_end: Line segment end point
        
    Returns:
        Minimum distance from point to line segment
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    
    # Find projection parameter
    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0.0, min(1.0, t))  # Clamp to line segment
        
    # Calculate nearest point on line
    nearest = line_start + t * line_vec
    
    # Calculate distance
    dist = np.linalg.norm(point - nearest)
    return dist


def evaluate_structural_accuracy(
    true_landmarks: np.ndarray, 
    pred_landmarks: np.ndarray, 
    alpha: float = 0.5, 
    beta: float = 0.1
) -> float:
    """Evaluate structural accuracy between true and predicted landmarks.
    
    Args:
        true_landmarks: Ground truth landmark coordinates [[x1, y1], [x2, y2], ...]
        pred_landmarks: Predicted landmark coordinates [[x1, y1], [x2, y2], ...]
        alpha: Direction weight parameter
        beta: Distance decay parameter
    
    Returns:
        Mean spatial angular similarity score
    """
    # Create line segments by connecting landmarks
    true_lines = []
    pred_lines = []
    
    # Connect landmarks to form a shape
    for i in range(len(true_landmarks)-1): # 3개의 선만 비교
        next_idx = (i + 1) % len(true_landmarks)
        true_lines.append([true_landmarks[i], true_landmarks[next_idx]])
        pred_lines.append([pred_landmarks[i], pred_landmarks[next_idx]])
    
    # Calculate similarity for each line pair
    similarities = []
    
    for true_line, pred_line in zip(true_lines, pred_lines):
        sim = calculate_spatial_angular_similarity(true_line, pred_line, alpha, beta)
        similarities.append(sim)
    
    # Return mean similarity
    return np.mean(similarities)


def tensor_to_numpy_landmarks(landmarks_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor landmarks to numpy array format.
    
    Args:
        landmarks_tensor: Landmark coordinates as tensor or array
        
    Returns:
        Landmark coordinates as numpy array
    """
    if isinstance(landmarks_tensor, torch.Tensor):
        return landmarks_tensor.detach().cpu().numpy()
    return landmarks_tensor


def batch_spatial_angular_similarity(
    sample: Dict, 
    predictions: torch.Tensor, 
    alpha: float = 0.5, 
    beta: float = 0.1
) -> float:
    """Calculate spatial angular similarity for a batch of samples.
    
    Args:
        sample: Dictionary containing 'label' with ground truth landmarks
        predictions: Model predictions
        alpha: Direction weight parameter
        beta: Distance decay parameter
        
    Returns:
        Mean spatial angular similarity across the batch
    """
    true_landmarks = tensor_to_numpy_landmarks(sample['label'])
    pred_landmarks = tensor_to_numpy_landmarks(predictions)
    
    batch_size = true_landmarks.shape[0]
    sas_scores = []

    for i in range(batch_size):
        true_batch = true_landmarks[i]
        pred_batch = pred_landmarks[i]
        
        # Reshape landmarks to 2D coordinates if needed
        if len(true_batch.shape) > 2:
            true_batch = true_batch.reshape(-1, 2)
        if len(pred_batch.shape) > 2:
            pred_batch = pred_batch.reshape(-1, 2)
        
        sas = evaluate_structural_accuracy(true_batch, pred_batch, alpha, beta)
        sas_scores.append(sas)
    
    return np.mean(sas_scores)

# ------------------ Angle -------------------
def get_angle(line1, line2):
    # Get directional vectors
    d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    # Compute dot product
    p = d1[0] * d2[0] + d1[1] * d2[1]
    # Compute norms
    n1 = math.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = math.sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    # Compute angle

    # ang = math.acos(min(p / (n1 * n2), 1))
    ang = math.acos(max(-1.0, min(p / (n1 * n2), 1.0)))

    if math.isnan(ang):
        return 180
    # Convert to degrees if you want
    ang2 = math.degrees(ang)
    return ang2

# ------------------- Utility Classes -------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics with new value.
        
        Args:
            val: Current value
            n: Number of instances represented by this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ------------------- Model Testing Function -------------------

def test_model(
    batch: Dict, 
    model: torch.nn.Module, 
    loss_fn: callable, 
    optimizer: torch.optim.Optimizer, 
    config: Dict, 
    visualize: bool = True
) -> Tuple[AverageMeter, float]:
    """Test and adapt model on a batch of data.
    
    Args:
        batch: Dictionary containing 'data' and 'label'
        model: Neural network model
        loss_fn: Loss function
        optimizer: Optimizer
        config: Configuration dictionary
        visualize: Whether to visualize predictions
        
    Returns:
        Tuple of (loss_meter, mean_distance_error)
    """
    losses = AverageMeter()
    mean_distance_error = AverageMeter()    

    # Prepare data
    data, labels = batch['data'], batch['label']
    adaptation_data = data.to(config['device'])
    adaptation_labels = labels.to(config['device'])

    # Generate heatmap labels (assuming render_gaussian_dot_f is imported)
    from .heatmaps import render_gaussian_dot_f, heatmap2coor
    
    adaptation_labels_heatmap = render_gaussian_dot_f(
        adaptation_labels.flip(dims=[2]),  # Convert xy to yx
        torch.tensor([config['std'], config['std']], dtype=torch.float32).to(config['device']),
        [config['height'], config['width']],
    ).to(torch.float)
    
    # Add background channel
    background = 1 - adaptation_labels_heatmap.sum(dim=1).unsqueeze(1).clip(0, 1)
    adaptation_labels_heatmap = torch.concat((adaptation_labels_heatmap, background), 1)

    sample = {
        'data': adaptation_data, 
        'label': adaptation_labels
    }

    # Adaptation loop
    for step in range(config['adaptation_steps'] * 2):
        optimizer.zero_grad()

        # Forward pass
        predictions = model(adaptation_data)
        
        # Calculate loss
        adaptation_error = loss_fn(
            predictions, 
            adaptation_labels_heatmap, 
            target_weight=torch.tensor([[1, 1, 1, 1, 0.01]]).to(config['device'])
        )
        
        print(f"Step {step}: Loss = {adaptation_error.item():.4f}")
        
        # Backward pass and optimization
        adaptation_error.backward()
        optimizer.step()
        
        # Calculate metrics
        metric = distance_error(sample, heatmap2coor(predictions[:, :4, ...]))

        # Update statistics
        losses.update(adaptation_error.item(), adaptation_data.size(0))
        mean_distance_error.update(
            np.mean(metric), 
            adaptation_data.size(0)
        )
    
    # Visualize predictions if requested
    if visualize:
        _visualize_predictions(predictions[0], adaptation_labels_heatmap[0])
    
    return losses, mean_distance_error.avg


def _visualize_predictions(predictions: torch.Tensor, ground_truth: torch.Tensor):
    """Visualize model predictions against ground truth.
    
    Args:
        predictions: Model predictions (single sample)
        ground_truth: Ground truth heatmaps (single sample)
    """
    # Visualize predictions
    fig, ax = plt.subplots(1, 5, figsize=(15, 10))
    for i, prediction in enumerate(predictions):
        ax[i].imshow(prediction.detach().cpu().numpy())
    plt.show() 
    
    # Visualize ground truth
    fig, ax = plt.subplots(1, 5, figsize=(15, 10))
    for i, gt in enumerate(ground_truth):
        ax[i].imshow(gt.detach().cpu().numpy())
    plt.show()


def add_sas_to_evaluation(config: Dict):
    """Add spatial angular similarity to model evaluation.
    
    This is a placeholder function that shows how to incorporate SAS metrics
    into the evaluation pipeline.
    
    Example usage:
    ```python
    with torch.no_grad():
        for sample in tqdm(test_loader):
            sample['data'] = sample['data'].to(config['device'])
            sample['label'] = sample['label'].to(config['device'])
            
            predictions = model(sample['data'])
            
            # Calculate MDE
            mde = distance_error(sample, predictions)
            mean_distance_error.update(np.mean(mde), sample['data'].size(0))
            
            # Calculate SAS
            sas = batch_spatial_angular_similarity(sample, predictions)
            spatial_similarity.update(sas, sample['data'].size(0))
    
    config['trace_func'](f"Mean Distance Error: {mean_distance_error.avg:.4f}")
    config['trace_func'](f"Spatial Angular Similarity: {spatial_similarity.avg:.2f}/100")
    ```
    """
    pass  # Implementation depends on evaluation loop