"""
evaluate.py
-----------
Statistically rigorous evaluation metrics for the early warning system.
Always assumes the model is lying until proven otherwise.
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


def bootstrap_confidence_interval(y_true, y_pred_proba, n_iterations=1000, ci=0.95):
    """
    Calculate bootstrap confidence interval for AUC.
    If the CI includes 0.5, your model is not better than random.
    """
    aucs = []
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        auc = roc_auc_score(y_true_boot, y_pred_boot)
        aucs.append(auc)
    
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    
    return {
        'mean': np.mean(aucs),
        'std': np.std(aucs),
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_width': upper - lower,
        'includes_0.5': lower < 0.5 < upper,
        'is_significant': lower > 0.5
    }


def test_calibration(y_true, y_pred_proba, n_bins=10):
    """
    Test if predicted probabilities are well-calibrated.
    Poor calibration means you can't trust the probability values.
    Returns Expected Calibration Error (ECE).
    """
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    # Handle case where calibration_curve returns fewer bins
    if len(prob_true) < 2:
        return {
            'ece': 1.0,
            'is_calibrated': False,
            'warning': 'Insufficient probability range for calibration'
        }
    
    # Calculate ECE - only for bins that have data
    bin_counts = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_true)
    
    # Align lengths - take only bins that exist in calibration_curve
    # prob_true and prob_pred may have fewer bins than n_bins
    n_actual_bins = len(prob_true)
    bin_weights_aligned = bin_weights[:n_actual_bins]
    
    # Normalize to sum to 1 for actual bins
    if bin_weights_aligned.sum() > 0:
        bin_weights_aligned = bin_weights_aligned / bin_weights_aligned.sum()
    
    ece = np.sum(bin_weights_aligned * np.abs(prob_true - prob_pred))
    
    return {
        'ece': ece,
        'is_calibrated': ece < 0.10,
        'n_bins_used': n_actual_bins
    }

def evaluate_with_rigor(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate model with statistical rigor including confidence intervals.
    """
    # Standard metrics
    results = evaluate_model(y_true, y_pred_proba, threshold)
    
    # Bootstrap CI
    bootstrap = bootstrap_confidence_interval(y_true, y_pred_proba)
    results['bootstrap'] = bootstrap
    
    # Add CI to top-level for easy access
    results['ci_lower'] = bootstrap['ci_lower']
    results['ci_upper'] = bootstrap['ci_upper']
    results['is_significant'] = bootstrap['is_significant']
    
    # Calibration
    calibration = test_calibration(y_true, y_pred_proba)
    results['calibration'] = calibration
    
    # Add interpretation
    if bootstrap['is_significant'] and calibration['is_calibrated']:
        results['verdict'] = 'PASS'
    elif bootstrap['is_significant']:
        results['verdict'] = 'WARNING - Poor calibration'
    elif calibration['is_calibrated']:
        results['verdict'] = 'WARNING - Not significant (CI includes 0.5)'
    else:
        results['verdict'] = 'FAIL - Results are not reliable'
    
    return results


def print_evaluation(results: dict):
    """
    Print evaluation results with statistical rigor.
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
    
    # Statistical rigor
    if 'bootstrap' in results:
        b = results['bootstrap']
        print(f"\n📈 95% CI for AUC: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")
        print(f"   {'✅ Significant (CI > 0.5)' if b['is_significant'] else '❌ NOT significant (CI includes 0.5)'}")
    
    if 'calibration' in results:
        cal = results['calibration']
        print(f"\n🎯 Calibration (ECE): {cal['ece']:.4f}")
        print(f"   {'✅ Calibrated' if cal['is_calibrated'] else '❌ Poor calibration'}")
    
    if 'verdict' in results:
        print(f"\nVERDICT: {results['verdict']}")
    
    print("="*50)


def save_results(results: dict, feature_names: list = None, coefficients: list = None, 
                 model_name: str = "logistic_regression", run_id: str = None):
    """
    Save evaluation results to outputs directory with timestamp.
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
        'metrics': {k: v for k, v in results.items() if not isinstance(v, dict)},
    }
    
    # Add statistical rigor results separately
    if 'bootstrap' in results:
        full_results['statistical_rigor'] = {
            'bootstrap': results['bootstrap'],
            'calibration': results.get('calibration', {}),
            'verdict': results.get('verdict', 'UNKNOWN')
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
        'ci_lower': results.get('bootstrap', {}).get('ci_lower', np.nan),
        'ci_upper': results.get('bootstrap', {}).get('ci_upper', np.nan),
        'is_significant': results.get('bootstrap', {}).get('is_significant', False),
        'ece': results.get('calibration', {}).get('ece', np.nan),
        'verdict': results.get('verdict', 'UNKNOWN'),
        # Add these for better tracking
        'precision_at_threshold': results['precision'],
        'recall_at_threshold': results['recall'],
        'predictions_count': results['predictions_at_threshold'],
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


def load_latest_results() -> pd.DataFrame:
    """
    Load all historical evaluation results for tracking.
    """
    csv_path = OUTPUT_DIR / "all_runs.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame()


def get_run_summary() -> str:
    """
    Get a summary of all runs with statistical rigor.
    """
    df = load_latest_results()
    if df.empty:
        return "No results found."
    
    # Count passing vs failing
    pass_count = len(df[df['verdict'] == 'PASS']) if 'verdict' in df.columns else 0
    fail_count = len(df[df['verdict'] == 'FAIL']) if 'verdict' in df.columns else 0
    
    summary = f"""
    === RUN SUMMARY ===
    Total Runs: {len(df)}
    Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}
    
    Verdicts:
      PASS: {pass_count}
      FAIL: {fail_count}
      WARNING: {len(df) - pass_count - fail_count}
    
    Best AUC-ROC: {df['auc_roc'].max():.4f} (Run: {df.loc[df['auc_roc'].idxmax(), 'run_id']})
    Best F1: {df['f1'].max():.4f} (Run: {df.loc[df['f1'].idxmax(), 'run_id']})
    
    Recent Performance:
    {df[['run_id', 'timestamp', 'auc_roc', 'f1', 'precision', 'recall', 'verdict']].tail(5).to_string(index=False)}
    """
    return summary