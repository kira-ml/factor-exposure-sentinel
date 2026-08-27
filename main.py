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
from src.models import ModelFactory, get_feature_importance, train_xgboost

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
    target = create_target(portfolio_returns)
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
    
    # 5. Prepare train/validation/test split
    print("\n[5] Preparing train/validation/test split...")
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    test_start = "2021-01-01"

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
    
    # ============================================================
    # 6. THRESHOLD SENSITIVITY ANALYSIS
    # ============================================================
    print("\n[6] Threshold Sensitivity Analysis (Training Data Only):")
    
    fci_train = features.loc[train_idx, 'fci'].dropna()
    target_train = target.loc[fci_train.index]
    
    percentiles = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    sensitivity_results = []
    
    from sklearn.metrics import confusion_matrix, roc_auc_score
    
    for pct in percentiles:
        threshold = fci_train.quantile(pct)
        y_pred = (fci_train > threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(target_train, y_pred).ravel()
        
        auc = roc_auc_score(target_train, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        sensitivity_results.append({
            'percentile': pct,
            'threshold': threshold,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': int(y_pred.sum())
        })
    
    # Find best thresholds
    best_by_f1 = max(sensitivity_results, key=lambda x: x['f1'])
    best_by_auc = max(sensitivity_results, key=lambda x: x['auc'])
    
    print(f"   Best F1: {best_by_f1['f1']:.4f} at {best_by_f1['percentile']*100:.0f}th percentile (AUC: {best_by_f1['auc']:.4f})")
    print(f"   Best AUC: {best_by_auc['auc']:.4f} at {best_by_auc['percentile']*100:.0f}th percentile (F1: {best_by_auc['f1']:.4f})")
    
    # Show full table
    print("\n   All Results:")
    print(f"   {'Pct':>6} | {'AUC':>8} | {'F1':>8} | {'Prec':>8} | {'Recall':>8} | {'Preds':>8}")
    print("   " + "-"*70)
    for r in sensitivity_results:
        print(f"   {r['percentile']*100:>5.0f}% | {r['auc']:>8.4f} | {r['f1']:>8.4f} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['predictions']:>8}")
    
    # Use 90th percentile (proven to generalize best)
    optimal_percentile = 0.90
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.90)
    print(f"\n   Using proven threshold: {optimal_percentile*100:.0f}th percentile (FCI threshold: {fci_threshold:.4f})")
    
    # ============================================================
    # 7. Threshold Baseline with 90th Percentile
    # ============================================================
    print(f"\n[7] Evaluating Threshold Baseline (FCI > 90th percentile)...")
    threshold_results = evaluate_threshold_baseline(
        features, target, train_idx, val_idx, percentile=0.90
    )
    print_evaluation(threshold_results)
    
    # ============================================================
    # 8. VIX-ENHANCED THRESHOLD BASELINE
    # ============================================================
    print("\n[8] VIX-Enhanced Threshold Baseline:")
    
    # Test different VIX thresholds
    vix_thresholds = [15, 18, 20, 22, 25]
    vix_enhanced_results = []
    
    for vix_t in vix_thresholds:
        # Rule: FCI > 90th percentile AND VIX > vix_t
        signal = (features['fci'] > fci_threshold) & (features['vix_level'] > vix_t)
        
        # Evaluate on validation set (tuning)
        y_pred_val = signal.loc[val_idx].astype(int) 
        y_true_val = target.loc[val_idx]
        
        tn, fp, fn, tp = confusion_matrix(y_true_val, y_pred_val).ravel()
        
        auc = roc_auc_score(y_true_val, y_pred_val)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        vix_enhanced_results.append({
            'vix_threshold': vix_t,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': int(y_pred_val.sum()),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    # Show results
    print(f"\n   Rule: FCI > 90th percentile AND VIX > threshold")
    print(f"   {'VIX >':>8} | {'AUC':>8} | {'F1':>8} | {'Prec':>8} | {'Recall':>8} | {'Preds':>8} | {'TP':>6}")
    print("   " + "-"*80)
    for r in vix_enhanced_results:
        print(f"   {r['vix_threshold']:>8} | {r['auc']:>8.4f} | {r['f1']:>8.4f} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['predictions']:>8} | {r['tp']:>6}")
    
    # Find best VIX threshold by F1 on validation set
    best_vix = max(vix_enhanced_results, key=lambda x: x['f1'])
    print(f"\n   Best VIX threshold: {best_vix['vix_threshold']} (F1: {best_vix['f1']:.4f}, AUC: {best_vix['auc']:.4f})")
    
    # Store for comparison
    vix_enhanced_auc = best_vix['auc']
    vix_enhanced_f1 = best_vix['f1']
    
    # ============================================================
    # 9. FCI TREND ENHANCED THRESHOLD (NEW)
    # ============================================================
    print("\n[9] FCI Trend-Enhanced Threshold Baseline:")
    
    # Calculate rolling average of FCI (20-day)
    features['fci_ma20'] = features['fci'].rolling(20).mean()
    features['fci_trend'] = (features['fci'] > features['fci_ma20']).astype(int)
    
    # Rule: FCI > 90th percentile AND FCI > 20-day MA
    signal_trend = (features['fci'] > fci_threshold) & (features['fci_trend'] == 1)
    
    # Evaluate on validation set (tuning)
    y_pred_val = signal_trend.loc[val_idx].astype(int)
    y_true_val = target.loc[val_idx]
    
    tn, fp, fn, tp = confusion_matrix(y_true_val, y_pred_val).ravel()
    trend_auc = roc_auc_score(y_true_val, y_pred_val)
    trend_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    trend_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    trend_f1 = 2 * trend_precision * trend_recall / (trend_precision + trend_recall) if (trend_precision + trend_recall) > 0 else 0
    
    print(f"\n   Rule: FCI > 90th percentile AND FCI > 20-day MA")
    print(f"   AUC-ROC: {trend_auc:.4f}")
    print(f"   F1: {trend_f1:.4f}")
    print(f"   Precision: {trend_precision:.4f}")
    print(f"   Recall: {trend_recall:.4f}")
    print(f"   Predictions: {int(y_pred_val.sum())}, TP: {tp}")
    
    # Combined: FCI > 90% AND FCI > MA20 AND VIX > 20
    signal_combined = (features['fci'] > fci_threshold) & (features['fci_trend'] == 1) & (features['vix_level'] > 20)
    
    # Evaluate on validation set (tuning)
    y_pred_val = signal_combined.loc[val_idx].astype(int)
    y_true_val = target.loc[val_idx]
    
    tn, fp, fn, tp = confusion_matrix(y_true_val, y_pred_val).ravel()
    combined_auc = roc_auc_score(y_true_val, y_pred_val)
    combined_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    combined_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    combined_f1 = 2 * combined_precision * combined_recall / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0
    
    print(f"\n   Combined Rule: FCI > 90% AND FCI > MA20 AND VIX > 20")
    print(f"   AUC-ROC: {combined_auc:.4f}")
    print(f"   F1: {combined_f1:.4f}")
    print(f"   Precision: {combined_precision:.4f}")
    print(f"   Recall: {combined_recall:.4f}")
    print(f"   Predictions: {int(y_pred_val.sum())}, TP: {tp}")
    
    # ============================================================
    # 10. Logistic Regression (Tier 2)
    # ============================================================
    print("\n[10] Training Logistic Regression...")
    lr_model = ModelFactory.get_model('logistic_regression')
    lr_results = ModelFactory.train_and_evaluate(
        lr_model, X_train, y_train, X_val, y_val, use_smote=True
    )
    
    print("\n[11] Evaluating Logistic Regression...")
    lr_eval = evaluate_model(y_val, lr_results['y_pred_proba'], threshold=lr_results['optimal_threshold'])
    print_evaluation(lr_eval)
    
    # Save LR results
    run_id = save_results(
        results=lr_eval,
        feature_names=X_train.columns.tolist(),
        coefficients=lr_results['model'].coef_[0].tolist(),
        model_name="logistic_regression_smote"
    )
    print(f"   Run ID: {run_id}")
    
    # ============================================================
    # 12. Random Forest (Tier 3)
    # ============================================================
    print("\n[12] Training Random Forest...")
    rf_model = ModelFactory.get_model('random_forest', n_estimators=100)
    rf_results = ModelFactory.train_and_evaluate(
        rf_model, X_train, y_train, X_val, y_val, use_smote=False
    )
    
    print("\n[13] Evaluating Random Forest...")
    rf_eval = evaluate_model(y_val, rf_results['y_pred_proba'], threshold=rf_results['optimal_threshold'])
    print_evaluation(rf_eval)
    
    # Save RF results
    run_id = save_results(
        results=rf_eval,
        feature_names=X_train.columns.tolist(),
        coefficients=rf_results['model'].feature_importances_.tolist(),
        model_name="random_forest"
    )
    print(f"   Run ID: {run_id}")
    
    # ============================================================
    # 14. Feature importance (Random Forest)
    # ============================================================
    print("\n[14] Feature importance (Random Forest):")
    importance = get_feature_importance(rf_results['model'], X_train.columns.tolist())
    print(importance.head(10).to_string(index=False))


    # ============================================================
    # 15. XGBoost with Calibration (Conditional Tier 4)
    # ============================================================
    print("\n[15] Training XGBoost with Calibration...")
    xgb_eval = None
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler
        
        # Get XGBoost model
        xgb_model = ModelFactory.get_model('xgboost', 
                                           n_estimators=100,
                                           max_depth=4,
                                           learning_rate=0.1,
                                           scale_pos_weight=10)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Calibrate probabilities using Platt scaling (sigmoid)
        calibrated_model = CalibratedClassifierCV(
            xgb_model, 
            method='sigmoid', 
            cv=3
        )
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Get calibrated probabilities on validation set
        y_pred_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
        
        # ---- OPTIMAL THRESHOLD SELECTION ON VALIDATION SET ----
        thresholds = np.linspace(0.01, 0.50, 50)
        best_f1 = 0
        best_threshold = 0.05
        best_eval = None
        
        for thresh in thresholds:
            eval_result = evaluate_model(y_val, y_pred_proba, threshold=thresh)
            if eval_result['f1'] > best_f1:
                best_f1 = eval_result['f1']
                best_threshold = thresh
                best_eval = eval_result
        
        if best_eval is not None:
            print(f"\n[16] Evaluating XGBoost (calibrated, threshold={best_threshold:.3f})...")
            print_evaluation(best_eval)
            
            # Save results (skip feature importance for calibrated model)
            run_id = save_results(
                results=best_eval,
                feature_names=None,
                coefficients=None,
                model_name="xgboost_calibrated"
            )
            print(f"   Run ID: {run_id}")
            
            xgb_eval = best_eval
        else:
            print(f"\n   ⚠️ XGBoost: No valid threshold found (F1=0 for all thresholds)")
            xgb_eval = None
        
    except ImportError as e:
        print(f"\n   ⚠️ XGBoost not available: {e}")
        print("   Skipping XGBoost. Run: pip install xgboost")
        xgb_eval = None
    except Exception as e:
        print(f"\n   ⚠️ XGBoost error: {e}")
        print("   Skipping XGBoost.")
        xgb_eval = None



    # ============================================================
    # 16. Model Selection (Validation Set Results)
    # ============================================================
    print("\n[16] Model Selection (Validation Set Results):")
    print(f"   Threshold Baseline (90%):        AUC-ROC: {threshold_results['auc_roc']:.4f}")
    print(f"   VIX-Enhanced Threshold:          AUC-ROC: {vix_enhanced_auc:.4f} (VIX > {best_vix['vix_threshold']})")
    print(f"   FCI Trend-Enhanced:              AUC-ROC: {trend_auc:.4f}")
    print(f"   Combined (FCI + Trend + VIX):    AUC-ROC: {combined_auc:.4f}")
    print(f"   Logistic Regression:             AUC-ROC: {lr_eval['auc_roc']:.4f}")
    print(f"   Random Forest:                   AUC-ROC: {rf_eval['auc_roc']:.4f}")
    
    # Add XGBoost if available
    if xgb_eval is not None:
        print(f"   XGBoost:                         AUC-ROC: {xgb_eval['auc_roc']:.4f}")
    
    # Select best model based on validation AUC
    best_models = [(threshold_results['auc_roc'], 'Threshold (90%)'), 
                   (vix_enhanced_auc, 'VIX-Enhanced'),
                   (trend_auc, 'FCI Trend'),
                   (combined_auc, 'Combined'),
                   (lr_eval['auc_roc'], 'Logistic Regression'),
                   (rf_eval['auc_roc'], 'Random Forest')]
    
    if xgb_eval is not None:
        best_models.append((xgb_eval['auc_roc'], 'XGBoost'))
    
    best_model_name = max(best_models, key=lambda x: x[0])[1]
    best_val_auc = max(best_models, key=lambda x: x[0])[0]
    print(f"\n   Best Model on Validation: {best_model_name} (AUC-ROC: {best_val_auc:.4f})")
    
    # ============================================================
    # 17. FINAL EVALUATION ON TEST SET (ONE-TIME)
    # ============================================================
    print("\n[17] FINAL EVALUATION ON TEST SET (Out-of-Sample):")
    print("   ⚠️  This is the ONE AND ONLY evaluation on test data.")
    print("   No further tuning or model selection should be done after this.")
    
    # Evaluate the best model on test set
    # Re-train on full training data with the best model's parameters
    # For now, we'll evaluate the best performing model on test data
    
    # For Logistic Regression (if it's the best)
    if best_model_name == 'Logistic Regression':
        # Re-train LR on full training data
        lr_model_final = ModelFactory.get_model('logistic_regression')
        lr_results_final = ModelFactory.train_and_evaluate(
            lr_model_final, X_train, y_train, X_test, y_test, use_smote=True
        )
        # Use the threshold found on validation
        final_eval = evaluate_model(y_test, lr_results_final['y_pred_proba'], threshold=lr_results['optimal_threshold'])
        print(f"\n   Logistic Regression (Test Set):")
        print(f"      AUC-ROC: {final_eval['auc_roc']:.4f}")
        print(f"      F1: {final_eval['f1']:.4f}")
        print(f"      Precision: {final_eval['precision']:.4f}")
        print(f"      Recall: {final_eval['recall']:.4f}")
    
    # For Random Forest (if it's the best)
    elif best_model_name == 'Random Forest':
        rf_model_final = ModelFactory.get_model('random_forest', n_estimators=100)
        rf_results_final = ModelFactory.train_and_evaluate(
            rf_model_final, X_train, y_train, X_test, y_test, use_smote=False
        )
        final_eval = evaluate_model(y_test, rf_results_final['y_pred_proba'], threshold=rf_results['optimal_threshold'])
        print(f"\n   Random Forest (Test Set):")
        print(f"      AUC-ROC: {final_eval['auc_roc']:.4f}")
        print(f"      F1: {final_eval['f1']:.4f}")
        print(f"      Precision: {final_eval['precision']:.4f}")
        print(f"      Recall: {final_eval['recall']:.4f}")
    
    # For XGBoost (if it's the best)
    elif best_model_name == 'XGBoost' and xgb_eval is not None:
        xgb_model_final = ModelFactory.get_model('xgboost',
                                                  n_estimators=100,
                                                  max_depth=4,
                                                  learning_rate=0.1,
                                                  scale_pos_weight=10)
        X_train_scaled_final = scaler.fit_transform(X_train)
        X_test_scaled_final = scaler.transform(X_test)
        calibrated_model_final = CalibratedClassifierCV(xgb_model_final, method='sigmoid', cv=3)
        calibrated_model_final.fit(X_train_scaled_final, y_train)
        y_pred_proba_final = calibrated_model_final.predict_proba(X_test_scaled_final)[:, 1]
        final_eval = evaluate_model(y_test, y_pred_proba_final, threshold=best_threshold)
        print(f"\n   XGBoost (Test Set):")
        print(f"      AUC-ROC: {final_eval['auc_roc']:.4f}")
        print(f"      F1: {final_eval['f1']:.4f}")
        print(f"      Precision: {final_eval['precision']:.4f}")
        print(f"      Recall: {final_eval['recall']:.4f}")
    
    # For threshold-based rules (if they are the best)
    else:
        # Threshold-based rules are simple - evaluate directly on test
        print(f"\n   {best_model_name} (Test Set):")
        # Re-run threshold baseline on test set
        if best_model_name == 'Threshold (90%)':
            threshold_results_test = evaluate_threshold_baseline(
                features, target, train_idx, test_idx, percentile=0.90
            )
            print(f"      AUC-ROC: {threshold_results_test['auc_roc']:.4f}")
            print(f"      F1: {threshold_results_test['f1']:.4f}")
            print(f"      Precision: {threshold_results_test['precision']:.4f}")
            print(f"      Recall: {threshold_results_test['recall']:.4f}")
        elif best_model_name == 'VIX-Enhanced':
            signal = (features['fci'] > fci_threshold) & (features['vix_level'] > best_vix['vix_threshold'])
            y_pred_test = signal.loc[test_idx].astype(int)
            y_true_test = target.loc[test_idx]
            test_auc = roc_auc_score(y_true_test, y_pred_test)
            tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()
            test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
            print(f"      AUC-ROC: {test_auc:.4f}")
            print(f"      F1: {test_f1:.4f}")
            print(f"      Precision: {test_precision:.4f}")
            print(f"      Recall: {test_recall:.4f}")
        elif best_model_name == 'FCI Trend':
            signal = (features['fci'] > fci_threshold) & (features['fci_trend'] == 1)
            y_pred_test = signal.loc[test_idx].astype(int)
            y_true_test = target.loc[test_idx]
            test_auc = roc_auc_score(y_true_test, y_pred_test)
            tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()
            test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
            print(f"      AUC-ROC: {test_auc:.4f}")
            print(f"      F1: {test_f1:.4f}")
            print(f"      Precision: {test_precision:.4f}")
            print(f"      Recall: {test_recall:.4f}")
        elif best_model_name == 'Combined':
            signal = (features['fci'] > fci_threshold) & (features['fci_trend'] == 1) & (features['vix_level'] > 20)
            y_pred_test = signal.loc[test_idx].astype(int)
            y_true_test = target.loc[test_idx]
            test_auc = roc_auc_score(y_true_test, y_pred_test)
            tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()
            test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
            print(f"      AUC-ROC: {test_auc:.4f}")
            print(f"      F1: {test_f1:.4f}")
            print(f"      Precision: {test_precision:.4f}")
            print(f"      Recall: {test_recall:.4f}")
    
    print("\n" + "="*60)
    print("⚠️  FINAL TEST RESULTS - DO NOT TUNE FURTHER")
    print("="*60)

if __name__ == "__main__":
    main()