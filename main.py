"""
main.py
-------
Simple pipeline orchestrator for Week 1 implementation.
Runs the entire workflow from data loading to evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import modules
from src.data_loader import fetch_all_data, fetch_etf_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features
from src.target_analysis import analyze_target, print_target_analysis, get_analysis_summary
from src.evaluate import evaluate_model, print_evaluation, save_results
from src.models import ModelFactory, get_feature_importance

from sklearn.metrics import confusion_matrix, roc_auc_score


def threshold_baseline(features, target, percentile=0.90):
    """Simple FCI threshold baseline."""
    threshold = features['fci'].quantile(percentile)
    y_pred = (features['fci'] > threshold).astype(int)
    return y_pred


def evaluate_threshold_baseline(features, target, train_idx, test_idx, percentile=0.90):
    """Evaluate threshold baseline on test data."""
    # Calculate threshold from training data only
    fci_train = features.loc[train_idx, 'fci'].dropna()
    threshold = fci_train.quantile(percentile)
    
    # Predict on test data
    fci_test = features.loc[test_idx, 'fci']
    y_pred = (fci_test > threshold).astype(int)
    y_true = target.loc[test_idx]
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    results = {
        'auc_roc': roc_auc_score(y_true, y_pred),
        'auc_pr': 0.0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'threshold': threshold,
        'total_samples': len(y_true),
        'positive_samples': y_true.sum(),
        'negative_samples': len(y_true) - y_true.sum(),
        'predictions_at_threshold': y_pred.sum(),
        'detection_delay': None,
    }
    
    return results


def main():
    print("="*60)
    print("FACTOR EXPOSURE SENTINEL - WEEK 1 IMPLEMENTATION")
    print("="*60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    # 2. Create portfolio
    print("\n[2] Creating portfolio...")
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    # 3. Create target
    print("\n[3] Creating target variable...")
    target = create_target(portfolio_returns, factors)
    print(f"   Event rate: {target.mean():.3f} ({target.sum()} events)")
    
    # 3b. Target analysis
    print("\n[3b] Target analysis...")
    target_results = analyze_target(target)
    print_target_analysis(target_results)
    print(f"   Summary: {get_analysis_summary(target_results)}")
    
    # 4. Create features
    print("\n[4] Creating features...")
    features = create_features(
        returns[ETF_TICKERS], 
        factors, 
        portfolio_weights,
        macro_data=vix
    )
    
    # Drop constant features
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # 4b. Data diagnostics
    print("\n[4b] Data diagnostics:")
    print(f"   Features shape: {features.shape}")
    print(f"   Target shape: {target.shape}")
    
    correlations = features.corrwith(target).sort_values(ascending=False)
    print("\n   Top correlations with target:")
    print(correlations.head(5))
    print("\n   Bottom correlations with target:")
    print(correlations.tail(5))
    
    # 5. Prepare train/test split
    print("\n[5] Preparing train/test split...")
    train_end = "2018-12-31"
    test_start = "2019-01-01"
    
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
    X_train = features.loc[train_idx]
    y_train = target.loc[train_idx]
    X_test = features.loc[test_idx]
    y_test = target.loc[test_idx]
    
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # 6. Threshold Baseline (Tier 1)
    print("\n[6] Evaluating Threshold Baseline (FCI > 90th percentile)...")
    threshold_results = evaluate_threshold_baseline(features, target, train_idx, test_idx, percentile=0.90)
    print_evaluation(threshold_results)
    
    # 7. Logistic Regression (Tier 2)
    print("\n[7] Training Logistic Regression...")
    lr_model = ModelFactory.get_model('logistic_regression')
    lr_results = ModelFactory.train_and_evaluate(
        lr_model, X_train, y_train, X_test, y_test, use_smote=True
    )
    
    print("\n[8] Evaluating Logistic Regression...")
    lr_eval = evaluate_model(y_test, lr_results['y_pred_proba'], threshold=lr_results['optimal_threshold'])
    print_evaluation(lr_eval)
    
    # Save LR results
    run_id = save_results(
        results=lr_eval,
        feature_names=X_train.columns.tolist(),
        coefficients=lr_results['model'].coef_[0].tolist(),
        model_name="logistic_regression_smote"
    )
    print(f"   Run ID: {run_id}")
    
    # 9. Random Forest (Tier 3)
    print("\n[9] Training Random Forest...")
    rf_model = ModelFactory.get_model('random_forest', n_estimators=100)
    rf_results = ModelFactory.train_and_evaluate(
        rf_model, X_train, y_train, X_test, y_test, use_smote=False
    )
    
    print("\n[10] Evaluating Random Forest...")
    rf_eval = evaluate_model(y_test, rf_results['y_pred_proba'], threshold=rf_results['optimal_threshold'])
    print_evaluation(rf_eval)
    
    # Save RF results
    run_id = save_results(
        results=rf_eval,
        feature_names=X_train.columns.tolist(),
        coefficients=rf_results['model'].feature_importances_.tolist(),
        model_name="random_forest"
    )
    print(f"   Run ID: {run_id}")
    
    # 11. Feature importance (Random Forest)
    print("\n[11] Feature importance (Random Forest):")
    importance = get_feature_importance(rf_results['model'], X_train.columns.tolist())
    print(importance.head(10).to_string(index=False))
    
    # 12. Model Comparison
    print("\n[12] Model Comparison:")
    print(f"   Threshold Baseline AUC-ROC: {threshold_results['auc_roc']:.4f}")
    print(f"   Logistic Regression AUC-ROC: {lr_eval['auc_roc']:.4f}")
    print(f"   Random Forest AUC-ROC: {rf_eval['auc_roc']:.4f}")
    
    best_model = max([(threshold_results['auc_roc'], 'Threshold'), 
                      (lr_eval['auc_roc'], 'Logistic Regression'),
                      (rf_eval['auc_roc'], 'Random Forest')], key=lambda x: x[0])
    print(f"\n   Best Model: {best_model[1]} (AUC-ROC: {best_model[0]:.4f})")
    
    print("\n" + "="*60)
    print("WEEK 1 IMPLEMENTATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()