"""
evaluate.py
-----------
Simple evaluation metrics for the early warning system.
Saves results automatically to outputs directory for tracking.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.metrics import average_precision_score


# Output directory
OUTPUT_DIR = Path("D:/quant-finance-ml/factor-exposure-sentinel/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    auc_pr = average_precision_score(y_true, y_pred_proba)

    
    # Binary predictions at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metrics
    results = {
        'auc_roc': float(auc),
        'auc_pr': float(auc_pr),
        'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
        'recall': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
        'f1': float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
        'false_positive_rate': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
        'true_positive_rate': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
        'threshold': float(threshold),
        'total_samples': int(len(y_true)),
        'positive_samples': int(y_true.sum()),
        'negative_samples': int(len(y_true) - y_true.sum()),
        'predictions_at_threshold': int(y_pred.sum()),
        'detection_delay': None,  # Placeholder for future
    }
    
    return results

def save_results(results: dict, feature_names: list = None, coefficients: list = None, 
                 model_name: str = "logistic_regression", run_id: str = None):
    """
    Save evaluation results to outputs directory with timestamp.
    
    Parameters:
    -----------
    results : dict
        Evaluation metrics from evaluate_model()
    feature_names : list, optional
        Feature names for importance tracking
    coefficients : list, optional
        Model coefficients for feature importance
    model_name : str
        Name of the model being evaluated
    run_id : str, optional
        Custom run identifier (auto-generated if None)
    """
    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run directory
    run_dir = OUTPUT_DIR / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)
    
    # Prepare results dictionary with metadata
    full_results = {
        'run_id': run_id,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': results,
    }
    
    # Add feature importance if provided
    if feature_names is not None and coefficients is not None:
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        full_results['feature_importance'] = importance.to_dict('records')
    
    # Save as JSON
    json_path = run_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save as CSV for easy tracking across runs
    csv_path = OUTPUT_DIR / "all_runs.csv"
    run_record = {
        'run_id': run_id,
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'auc_roc': results['auc_roc'],
        'auc_pr': results['auc_pr'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'false_positive_rate': results['false_positive_rate'],
        'true_positive_rate': results['true_positive_rate'],
        'threshold': results['threshold'],
        'total_samples': results['total_samples'],
        'positive_samples': results['positive_samples'],
        'negative_samples': results['negative_samples'],
        'predictions_at_threshold': results['predictions_at_threshold'],
    }
    
    # Append to CSV (create if doesn't exist)
    run_record_df = pd.DataFrame([run_record])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        updated = pd.concat([existing, run_record_df], ignore_index=True)
        updated.to_csv(csv_path, index=False)
    else:
        run_record_df.to_csv(csv_path, index=False)
    
    print(f"\n[Results saved to: {run_dir}]")
    
    return run_id

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
    print(f"Total Samples: {results['total_samples']}")
    print(f"Positive Samples: {results['positive_samples']} ({results['positive_samples']/results['total_samples']*100:.1f}%)")
    print("="*50)

def load_latest_results() -> pd.DataFrame:
    """
    Load all historical evaluation results for tracking.
    
    Returns:
    --------
    pd.DataFrame with all runs
    """
    csv_path = OUTPUT_DIR / "all_runs.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame()

def get_run_summary() -> str:
    """
    Get a summary of all runs for quick review.
    """
    df = load_latest_results()
    if df.empty:
        return "No results found."
    
    summary = f"""
    === RUN SUMMARY ===
    Total Runs: {len(df)}
    Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}
    
    Best AUC-ROC: {df['auc_roc'].max():.4f} (Run: {df.loc[df['auc_roc'].idxmax(), 'run_id']})
    Best F1: {df['f1'].max():.4f} (Run: {df.loc[df['f1'].idxmax(), 'run_id']})
    
    Recent Performance:
    {df[['run_id', 'timestamp', 'auc_roc', 'f1', 'precision', 'recall']].tail(5).to_string(index=False)}
    """
    return summary