import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss

def calculate_classification_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Calculates standard classification accuracy."""
    preds = np.argmax(logits, axis=1)
    return accuracy_score(targets, preds)

def calculate_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int, is_ood: bool = False) -> float:
    """
    Calculates the Total Brier Score (BS_Total).
    
    Args:
        probs: Predicted class probabilities (N x K).
        labels: True integer labels (N).
        num_classes: Number of classes (K).
        is_ood: If True, calculates Brier Score against a uniform target 
                (representing ideal OOD uncertainty).
    """
    if len(labels) == 0:
        return np.nan
    
    if is_ood:
        # For OOD data, the "target" is the Uniform Distribution (P_target = 1/K for all classes).
        # This measures how close the model's prediction is to total uncertainty.
        y_true_one_hot = np.full((probs.shape[0], num_classes), 1.0 / num_classes)
    else:
        # For ID data, the target is the True One-Hot label.
        y_true_one_hot = np.zeros((labels.size, num_classes))
        y_true_one_hot[np.arange(labels.size), labels] = 1

    # Total Brier Score formula: BS = 1/N * Sum [ (P_ij - Y_ij)^2 ]
    squared_error = np.sum((probs - y_true_one_hot)**2, axis=1)
    bs_total = np.mean(squared_error)
    
    return bs_total

def calculate_ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    """
    Calculates AUROC and AUPR for OOD detection using uncertainty scores.
    Args:
        id_scores: Uncertainty scores for In-Distribution data.
        ood_scores: Uncertainty scores for Out-of-Distribution data.
    Returns: (AUROC, AUPR)
    """
    # 1. Concatenate scores (OOD scores are the 'positive' class)
    scores = np.concatenate([id_scores, ood_scores])
    
    # 2. Create labels (0 for ID, 1 for OOD)
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    
    # AUROC requires score distribution of positive/negative classes
    auroc = roc_auc_score(labels, scores)
    
    # AUPR requires the same scores/labels
    aupr = average_precision_score(labels, scores)
    
    return auroc, aupr

def get_edl_total_uncertainty(alpha: np.ndarray) -> np.ndarray:
    """Calculates the standard EDL uncertainty mass (K / S)."""
    S = np.sum(alpha, axis=1, keepdims=True)
    K = alpha.shape[1]
    uncertainty = K / S.squeeze()
    return uncertainty