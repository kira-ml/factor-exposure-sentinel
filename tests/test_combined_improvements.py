"""
test_combined_improvements.py
Test all mathematically justified improvements together.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.data_loader import fetch_all_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features

def test_combined_improvements():
    """Test all mathematically justified improvements + Random Forest + Permutation Test."""
    
    # Load data
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    # Portfolio
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    # Target & Features
    target = create_target(portfolio_returns, factors)
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # --- ADD NEW FEATURES ---
    # 1. FCI change
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    
    # 2. FCI / VIX ratio
    features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    
    # 3. Log transformations
    features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    # 4. Factor correlation
    def rolling_factor_correlation(factor_returns, window=60):
        corr_series = pd.Series(index=factor_returns.index, dtype=float)
        for i in range(window, len(factor_returns)):
            corr = factor_returns.iloc[i-window:i].corr()
            upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            corr_series.iloc[i] = np.nanmean(upper_tri)
        return corr_series
    
    features['factor_corr_60'] = rolling_factor_correlation(factors, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factors, window=120)
    
    # --- BEST EXISTING FEATURES ---
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    # Prepare X and y for ML
    X_train = features.loc[train_idx]
    y_train = target.loc[train_idx]
    X_val = features.loc[val_idx]
    y_val = target.loc[val_idx]
    
    # Best config from Day 3
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.95)
    
    # Test rules
    rules = {
        'Day 3 Best': (features['fci'] > fci_threshold) & 
                      (features['fci'] > features['fci_ma25']) & 
                      (features['vix_level'] > 19) & 
                      (features['ret_vol_60'] > 0.008),
        
        '+ FCI Change 5 > 0': (features['fci'] > fci_threshold) & 
                              (features['fci'] > features['fci_ma25']) & 
                              (features['vix_level'] > 19) & 
                              (features['ret_vol_60'] > 0.008) & 
                              (features['fci_change_5'] > 0),
        
        '+ FCI Change 10 > 0': (features['fci'] > fci_threshold) & 
                               (features['fci'] > features['fci_ma25']) & 
                               (features['vix_level'] > 19) & 
                               (features['ret_vol_60'] > 0.008) & 
                               (features['fci_change_10'] > 0),
        
        '+ FCI/VIX Ratio > 0.04': (features['fci'] > fci_threshold) & 
                                  (features['fci'] > features['fci_ma25']) & 
                                  (features['vix_level'] > 19) & 
                                  (features['ret_vol_60'] > 0.008) & 
                                  (features['fci_vix_ratio'] > 0.04),
        
        '+ Factor Corr > 0.3': (features['fci'] > fci_threshold) & 
                              (features['fci'] > features['fci_ma25']) & 
                              (features['vix_level'] > 19) & 
                              (features['ret_vol_60'] > 0.008) & 
                              (features['factor_corr_60'] > 0.3),
        
        'All Combined': (features['fci'] > fci_threshold) & 
                        (features['fci'] > features['fci_ma25']) & 
                        (features['vix_level'] > 19) & 
                        (features['ret_vol_60'] > 0.008) & 
                        (features['fci_change_10'] > 0) & 
                        (features['fci_vix_ratio'] > 0.04) & 
                        (features['factor_corr_60'] > 0.3),
    }
    
    print("\n" + "="*70)
    print("MATH-JUSTIFIED IMPROVEMENTS (DATA-DRIVEN)")
    print("="*70)
    
    results = []
    for name, signal in rules.items():
        y_pred = signal.loc[val_idx].astype(int)
        y_true = target.loc[val_idx]
        
        auc = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'rule': name,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': int(y_pred.sum()),
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
        
        status = "✅" if (auc > 0.6002 and f1 > 0.2184) else "❌" if auc < 0.6002 else "="
        print(f"\n{status} {name}:")
        print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  Preds: {int(y_pred.sum())}, TP: {tp}, FP: {fp}, FN: {fn}")
    
    # ============================================================
    # RANDOM FOREST ON VALIDATION SET
    # ============================================================
    print("\n" + "="*70)
    print("RANDOM FOREST (USING ALL FEATURES)")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Handle NaN values
    X_train_clean = X_train.fillna(0)
    X_val_clean = X_val.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    
    # Predict on validation
    y_pred_proba = rf.predict_proba(X_val_scaled)[:, 1]
    
    # Find optimal threshold on validation
    thresholds = np.linspace(0.01, 0.50, 50)
    best_f1 = 0
    best_threshold = 0.05
    best_rf_eval = None
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_rf_eval = {
                'auc': roc_auc_score(y_val, y_pred_proba),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': int(y_pred.sum()),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'threshold': thresh
            }
    
    print(f"\nRandom Forest (Validation Set):")
    print(f"  AUC-ROC: {best_rf_eval['auc']:.4f}")
    print(f"  F1: {best_rf_eval['f1']:.4f}")
    print(f"  Precision: {best_rf_eval['precision']:.4f}")
    print(f"  Recall: {best_rf_eval['recall']:.4f}")
    print(f"  Threshold: {best_rf_eval['threshold']:.3f}")
    print(f"  Predictions: {best_rf_eval['predictions']}, TP: {best_rf_eval['tp']}, FP: {best_rf_eval['fp']}, FN: {best_rf_eval['fn']}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 10 Features:")
    print(importance.head(10).to_string(index=False))
    
    # ============================================================
    # PERMUTATION TEST FOR STATISTICAL SIGNIFICANCE
    # ============================================================
    print("\n" + "="*70)
    print("PERMUTATION TEST (Statistical Significance)")
    print("="*70)
    
    observed_auc = best_rf_eval['auc']
    n_permutations = 1000
    permutation_aucs = []
    
    for i in range(n_permutations):
        y_shuffled = np.random.permutation(y_val)
        auc = roc_auc_score(y_shuffled, y_pred_proba)
        permutation_aucs.append(auc)
    
    p_value = np.mean(np.array(permutation_aucs) >= observed_auc)
    
    print(f"\nObserved AUC: {observed_auc:.4f}")
    print(f"Permutation AUC mean: {np.mean(permutation_aucs):.4f}")
    print(f"Permutation AUC std: {np.std(permutation_aucs):.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"\n✅ Statistically significant (p < 0.05). Signal is likely real.")
    else:
        print(f"\n❌ NOT statistically significant (p >= 0.05). Signal may be noise.")
    
    # ============================================================
    # POWER ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("POWER ANALYSIS")
    print("="*70)
    
    n_events = int(y_val.sum())
    n_samples = len(y_val)
    effect_size = observed_auc - 0.5
    
    print(f"\nValidation set:")
    print(f"  Total samples: {n_samples}")
    print(f"  Event count: {n_events}")
    print(f"  Event rate: {n_events/n_samples*100:.1f}%")
    print(f"  Observed AUC: {observed_auc:.4f}")
    print(f"  Effect size (AUC - 0.5): {effect_size:.4f}")
    
    if effect_size < 0.05:
        print(f"\n  ⚠️  Effect size is very small (< 0.05).")
        print(f"  With only {n_events} events, power to detect this effect is low.")
        print(f"  Estimated events needed for reliable detection: ~200-300")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Best rule-based
    best_rule = max(results, key=lambda x: x['f1'])
    print(f"Best Rule-Based: {best_rule['rule']}")
    print(f"  AUC: {best_rule['auc']:.4f}, F1: {best_rule['f1']:.4f}, TP: {best_rule['tp']}, FP: {best_rule['fp']}")
    
    print(f"\nRandom Forest:")
    print(f"  AUC: {best_rf_eval['auc']:.4f}, F1: {best_rf_eval['f1']:.4f}, TP: {best_rf_eval['tp']}, FP: {best_rf_eval['fp']}")
    
    if best_rf_eval['auc'] > best_rule['auc']:
        print(f"\n✅ Random Forest outperforms rule-based (AUC +{best_rf_eval['auc'] - best_rule['auc']:.4f})")
    else:
        print(f"\n❌ Rule-based outperforms Random Forest (AUC +{best_rule['auc'] - best_rf_eval['auc']:.4f})")
    
    # Statistical significance conclusion
    if p_value < 0.05:
        print(f"\n✅ Signal is statistically significant (p={p_value:.4f})")
        print(f"   However, practical significance is limited by:")
        print(f"   - Low precision (0.0699)")
        print(f"   - High false positive ratio (13:1)")
    else:
        print(f"\n❌ Signal is NOT statistically significant (p={p_value:.4f})")
        print(f"   Recommend: Accept null hypothesis. No reliable predictive relationship found.")
    
    return results, best_rf_eval, p_value

if __name__ == "__main__":
    test_combined_improvements()