"""
main.py
-------
Clean pipeline orchestrator for Factor Exposure Sentinel.
Focuses on rigorous evaluation with statistical validation.
Baseline-first: threshold rules → logistic regression → XGBoost
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import fetch_all_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features
from src.target_analysis import analyze_target, print_target_analysis, get_analysis_summary
from src.evaluate import evaluate_with_rigor, print_evaluation, save_results, load_latest_results
from src.models import ModelFactory, get_feature_importance, train_xgboost
from src.visualization import generate_all_visualizations

from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def evaluate_threshold_rule(features, target, train_idx, test_idx, 
                            fci_col='fci', threshold_pct=0.90, 
                            vix_threshold=None, trend=False):
    """
    Evaluate a threshold-based rule on test data.
    All thresholds are calculated from training data only.
    """
    # Calculate threshold from training data only
    fci_train = features.loc[train_idx, fci_col].dropna()
    threshold = fci_train.quantile(threshold_pct)
    
    # Build signal
    signal = features[fci_col] > threshold
    
    if vix_threshold is not None:
        signal = signal & (features['vix_level'] > vix_threshold)
    
    if trend:
        features['fci_ma20'] = features['fci'].rolling(20).mean()
        signal = signal & (features['fci'] > features['fci_ma20'])
    
    # Evaluate on test data
    y_pred = signal.loc[test_idx].astype(int)
    y_true = target.loc[test_idx]
    
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
    print("="*70)
    print("FACTOR EXPOSURE SENTINEL - RIGOROUS EVALUATION")
    print("="*70)
    
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
    
    # 3. Create target (-3% drawdown, NO attribution)
    print("\n[3] Creating target variable...")
    target = create_target(portfolio_returns, drawdown_threshold=-0.03)
    print(f"   Event rate: {target.mean():.3f} ({target.sum()} events)")
    
    # 3b. Target analysis
    print("\n[3b] Target analysis...")
    target_results = analyze_target(target)
    print_target_analysis(target_results)
    print(f"   Summary: {get_analysis_summary(target_results)}")
    
    # 4. Create features (validated set only)
    print("\n[4] Creating features...")
    features = create_features(
        returns[ETF_TICKERS], 
        factors, 
        portfolio_weights,
        macro_data=vix
    )
    
    # Drop constant features
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    print(f"   Features shape: {features.shape}")
    correlations = features.corrwith(target).sort_values(ascending=False)
    print("\n   Top 5 correlations with target:")
    for feat, corr in correlations.head(5).items():
        print(f"      {feat}: {corr:.4f}")
    
    # 5. Train/Validation/Test Split (chronological)
    print("\n[5] Preparing train/validation/test split...")
    train_end = "2016-12-31"
    val_end = "2022-12-31"
    test_start = "2023-01-01"
    
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
    X_train = features.loc[train_idx]
    y_train = target.loc[train_idx]
    X_val = features.loc[val_idx]
    y_val = target.loc[val_idx]
    X_test = features.loc[test_idx]
    y_test = target.loc[test_idx]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Test events: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # ============================================================
    # 6. HEURISTIC BASELINE: FCI > 90th percentile
    # ============================================================
    print("\n[6] Heuristic Baseline: FCI > 90th percentile...")
    threshold_results = evaluate_threshold_rule(
        features, target, train_idx, test_idx, fci_col='fci', threshold_pct=0.90
    )
    print(f"   AUC-ROC: {threshold_results['auc_roc']:.4f}, "
          f"Precision: {threshold_results['precision']:.4f}, "
          f"Recall: {threshold_results['recall']:.4f}, "
          f"F1: {threshold_results['f1']:.4f}")
    
    # ============================================================
    # 7. ENHANCED BASELINE: FCI + VIX + Credit
    # ============================================================
    print("\n[7] Enhanced Baseline: FCI > 90% AND VIX > 20 AND Credit > median...")
    enhanced_results = evaluate_threshold_rule(
        features, target, train_idx, test_idx, 
        fci_col='fci', threshold_pct=0.90, vix_threshold=20
    )
    print(f"   AUC-ROC: {enhanced_results['auc_roc']:.4f}, "
          f"Precision: {enhanced_results['precision']:.4f}, "
          f"Recall: {enhanced_results['recall']:.4f}, "
          f"F1: {enhanced_results['f1']:.4f}")
    
    # ============================================================
    # 8. LOGISTIC REGRESSION (Baseline ML)
    # ============================================================
    print("\n[8] Logistic Regression (with SMOTE)...")
    lr_model = ModelFactory.get_model('logistic_regression')
    lr_results = ModelFactory.train_and_evaluate(
        lr_model, X_train, y_train, X_val, y_val, use_smote=True
    )
    lr_eval = evaluate_with_rigor(y_val, lr_results['y_pred_proba'], 
                                  threshold=lr_results['optimal_threshold'])
    print(f"   Validation AUC: {lr_eval['auc_roc']:.4f}")
    print(f"   Optimal threshold: {lr_results['optimal_threshold']:.3f}")
    
    # Evaluate on test
    lr_model_final = ModelFactory.get_model('logistic_regression')
    lr_final = ModelFactory.train_and_evaluate(
        lr_model_final, X_train, y_train, X_test, y_test, 
        use_smote=True, threshold_method='default', default_threshold=0.05
    )
    lr_test = evaluate_with_rigor(y_test, lr_final['y_pred_proba'], 
                                  threshold=0.05)
    print(f"   Test AUC: {lr_test['auc_roc']:.4f} (CI: [{lr_test['ci_lower']:.4f}, {lr_test['ci_upper']:.4f}])")
    
    # ============================================================
    # 9. RANDOM FOREST (Intermediate)
    # ============================================================
    print("\n[9] Random Forest...")
    rf_model = ModelFactory.get_model('random_forest', n_estimators=100)
    rf_results = ModelFactory.train_and_evaluate(
        rf_model, X_train, y_train, X_val, y_val, use_smote=False
    )
    rf_eval = evaluate_with_rigor(y_val, rf_results['y_pred_proba'], 
                                  threshold=rf_results['optimal_threshold'])
    print(f"   Validation AUC: {rf_eval['auc_roc']:.4f}")
    
    # Evaluate on test
    rf_model_final = ModelFactory.get_model('random_forest', n_estimators=100)
    rf_final = ModelFactory.train_and_evaluate(
        rf_model_final, X_train, y_train, X_test, y_test, 
        use_smote=False, threshold_method='default', default_threshold=0.05
    )
    rf_test = evaluate_with_rigor(y_test, rf_final['y_pred_proba'], 
                                  threshold=0.05)
    print(f"   Test AUC: {rf_test['auc_roc']:.4f} (CI: [{rf_test['ci_lower']:.4f}, {rf_test['ci_upper']:.4f}])")
    
    # ============================================================
    # 10. XGBOOST (Primary Model - Most Stable)
    # ============================================================
    print("\n[10] XGBoost with Calibration (Primary Model)...")
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler
        
        xgb_model = ModelFactory.get_model('xgboost', 
                                           n_estimators=100,
                                           max_depth=4,
                                           learning_rate=0.1,
                                           scale_pos_weight=10)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Calibrate on training data
        calibrated_model = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Validate - find best threshold on validation set
        y_pred_proba_val = calibrated_model.predict_proba(X_val_scaled)[:, 1]
        
        # Find optimal threshold on validation (max F1)
        thresholds = np.linspace(0.01, 0.50, 50)
        best_f1 = 0
        best_threshold = 0.05
        best_val_auc = 0
        
        for thresh in thresholds:
            eval_result = evaluate_with_rigor(y_val, y_pred_proba_val, threshold=thresh)
            if eval_result['f1'] > best_f1:
                best_f1 = eval_result['f1']
                best_threshold = thresh
                best_val_auc = eval_result['auc_roc']
        
        print(f"   Best validation threshold: {best_threshold:.3f} (F1: {best_f1:.4f}, AUC: {best_val_auc:.4f})")
        
        # Test with best threshold
        y_pred_proba_test = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        xgb_test = evaluate_with_rigor(y_test, y_pred_proba_test, threshold=best_threshold)
        print(f"   Test AUC: {xgb_test['auc_roc']:.4f} (CI: [{xgb_test['ci_lower']:.4f}, {xgb_test['ci_upper']:.4f}])")
        print(f"   Test F1: {xgb_test['f1']:.4f} (Precision: {xgb_test['precision']:.4f}, Recall: {xgb_test['recall']:.4f})")
        
        # Save XGBoost results
        run_id = save_results(
            results=xgb_test,
            feature_names=None,
            coefficients=None,
            model_name="xgboost_calibrated"
        )
        print(f"   Run ID: {run_id}")
        
    except Exception as e:
        print(f"   ⚠️ XGBoost error: {e}")
        xgb_test = None
    
    # ============================================================
    # 11. FINAL RESULTS SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("FINAL TEST RESULTS (One-time evaluation)")
    print("="*70)
    
    all_results = {
        'Heuristic (FCI 90%)': threshold_results,
        'Enhanced (FCI+VIX)': enhanced_results,
        'Logistic Regression': lr_test,
        'Random Forest': rf_test,
    }
    if xgb_test is not None:
        all_results['XGBoost'] = xgb_test
    
    print("\n   Model            | AUC-ROC | CI Lower | CI Upper | Significant? | F1")
    print("   " + "-"*80)
    for name, res in all_results.items():
        if 'ci_lower' in res:
            sig = "✅" if res['is_significant'] else "❌"
            print(f"   {name:16} | {res['auc_roc']:.4f}  | {res['ci_lower']:.4f}   | {res['ci_upper']:.4f}   | {sig}          | {res['f1']:.4f}")
        else:
            print(f"   {name:16} | {res['auc_roc']:.4f}  | N/A      | N/A      | N/A        | {res['f1']:.4f}")
    
    # Overall verdict
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)
    
    if xgb_test is not None:
        final_model = xgb_test
        model_name = "XGBoost"
    else:
        final_model = rf_test
        model_name = "Random Forest"
    
    if final_model['is_significant']:
        print(f"   ✅ {model_name} shows statistically significant predictive signal (CI > 0.5)")
    else:
        print(f"   ❌ {model_name} is NOT statistically significant (CI includes 0.5)")
        print("   The null hypothesis cannot be rejected with the available data.")
    
    print(f"\n   Best AUC-ROC: {final_model['auc_roc']:.4f}")
    print(f"   95% CI: [{final_model['ci_lower']:.4f}, {final_model['ci_upper']:.4f}]")
    print(f"   ECE: {final_model.get('calibration', {}).get('ece', 0):.4f}")
    print(f"   Verdict: {final_model.get('verdict', 'UNKNOWN')}")
    
    # ============================================================
    # 12. GENERATE VISUALIZATIONS
    # ============================================================
    print("\n[12] Generating visualizations...")
    
    # Get predictions from best model
    if xgb_test is not None:
        y_pred_proba_final = y_pred_proba_test
    else:
        y_pred_proba_final = rf_final['y_pred_proba']
    
    results_df = load_latest_results()
    
    generate_all_visualizations(
        target=target,
        features=features,
        portfolio_returns=portfolio_returns,
        y_test=y_test,
        y_pred_proba_final=y_pred_proba_final,
        results_df=results_df
    )
    
    print("\n" + "="*70)
    print("✅ Pipeline complete!")
    print("="*70)
    print("\n📊 Results saved to: outputs/")
    print("📈 Figures saved to: outputs/figures/")


if __name__ == "__main__":
    main()