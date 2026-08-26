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
from src.evaluate import evaluate_model, print_evaluation, save_results

# Simple Logistic Regression (no complex ML yet)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE


def threshold_baseline(features, target, percentile=0.90):
    """
    Simple FCI threshold baseline.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Features with 'fci' column
    target : pd.Series
        Target variable (for alignment only)
    percentile : float
        Percentile threshold for FCI (default 0.90 = top 10%)
    
    Returns:
    --------
    pd.Series with binary predictions
    """
    threshold = features['fci'].quantile(percentile)
    y_pred = (features['fci'] > threshold).astype(int)
    return y_pred


def evaluate_threshold_baseline(features, target, train_idx, test_idx, percentile=0.90):
    """
    Evaluate threshold baseline on test data.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Features with 'fci' column
    target : pd.Series
        Target variable
    train_idx : Index
        Training indices
    test_idx : Index
        Test indices
    percentile : float
        Percentile threshold for FCI
    
    Returns:
    --------
    dict with evaluation metrics
    """
    # Calculate threshold from training data only
    fci_train = features.loc[train_idx, 'fci'].dropna()
    threshold = fci_train.quantile(percentile)
    
    # Predict on test data
    fci_test = features.loc[test_idx, 'fci']
    y_pred = (fci_test > threshold).astype(int)
    y_true = target.loc[test_idx]
    
    # Calculate metrics manually
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    results = {
        'auc_roc': roc_auc_score(y_true, y_pred),
        'auc_pr': 0.0,  # Not applicable for threshold rule
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
    
    # 2. Create simple portfolio (equal weights for now)
    print("\n[2] Creating portfolio...")
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    
    # Calculate portfolio returns
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    # 3. Create target
    print("\n[3] Creating target variable...")
    target = create_target(portfolio_returns, factors)
    print(f"   Event rate: {target.mean():.3f} ({target.sum()} events)")
    
    # 4. Create features
    print("\n[4] Creating features...")
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')

    # Check data alignment
    print("\n[4b] Data diagnostics:")
    print(f"   Features shape: {features.shape}")
    print(f"   Target shape: {target.shape}")
    print(f"   Features index: {features.index.min()} to {features.index.max()}")
    print(f"   Target index: {target.index.min()} to {target.index.max()}")

    # Check correlation between features and target
    correlations = features.corrwith(target).sort_values(ascending=False)
    print("\n   Top correlations with target:")
    print(correlations.head(5))
    print("\n   Bottom correlations with target:")
    print(correlations.tail(5))
    
    # 5. Prepare data for modeling (chronological split)
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
    
    # 6. Evaluate Threshold Baseline (Tier 1)
    print("\n[6] Evaluating Threshold Baseline (FCI > 90th percentile)...")
    threshold_results = evaluate_threshold_baseline(features, target, train_idx, test_idx, percentile=0.90)
    print_evaluation(threshold_results)
    
    # 7. Train Logistic Regression with SMOTE (Tier 2)
    print("\n[7] Training logistic regression with SMOTE...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"   SMOTE applied: Original {len(y_train)} -> Resampled {len(y_train_resampled)} samples")
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 8. Predict and evaluate
    print("\n[8] Evaluating Logistic Regression...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold from training data
    precision, recall, thresholds = precision_recall_curve(y_train_resampled, 
                                                           model.predict_proba(X_train_resampled)[:, 1])
    # Use threshold that balances precision and recall
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0.5
    
    results = evaluate_model(y_test, y_pred_proba, threshold=optimal_threshold)
    print_evaluation(results)
    
    # Save results to outputs directory
    run_id = save_results(
        results=results,
        feature_names=X_train.columns.tolist(),
        coefficients=model.coef_[0].tolist(),
        model_name="logistic_regression_smote"
    )
    print(f"   Run ID: {run_id}")
    
    # 9. Feature importance
    print("\n[9] Feature importance:")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(importance.head(10).to_string(index=False))
    
    # 10. Comparison Summary
    print("\n[10] Model Comparison:")
    print(f"   Threshold Baseline AUC-ROC: {threshold_results['auc_roc']:.4f}")
    print(f"   Logistic Regression (SMOTE) AUC-ROC: {results['auc_roc']:.4f}")
    improvement = results['auc_roc'] - threshold_results['auc_roc']
    print(f"   Improvement: {improvement:+.4f}")
    
    print("\n" + "="*60)
    print("WEEK 1 IMPLEMENTATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()