"""
evaluate.py
-----------
Simple evaluation metrics for the early warning system.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

def evaluate_model(y_true: pd.Series, y_pred_proba: pd.Series, threshold: float = 0.5):
    """
    Evaluate model performance with key metrics.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred_proba : pd.Series
        Predicted probabilities
    threshold : float
        Classification threshold
    
    Returns:
    --------
    dict with evaluation metrics
    """
    # AUC-ROC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = np.trapz(precision, recall)
    
    # Binary predictions at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metrics
    results = {
        'auc_roc': auc,
        'auc_pr': auc_pr,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'detection_delay': None,  # Placeholder for future
    }
    
    return results

def print_evaluation(results: dict):
    """
    Print evaluation results in a readable format.
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print(f"AUC-PR:  {results['auc_pr']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    print("="*50)