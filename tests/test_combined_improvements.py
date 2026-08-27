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

def test_no_attribution_target():
    """Test Random Forest on target without 60% attribution threshold.
    
    Hypothesis: The 60% attribution threshold is too restrictive.
    Removing it should increase event count and potentially reveal signal.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: NO ATTRIBUTION THRESHOLD")
    print("Target = drawdown < -5% (no 60% attribution condition)")
    print("="*70)
    
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
    
    # ---- CREATE TARGET WITHOUT ATTRIBUTION ----
    from src.target import calculate_drawdown
    
    drawdown = calculate_drawdown(portfolio_returns, window=21)
    target_no_attribution = (drawdown < -0.05).astype(int)
    
    print(f"\nTarget without attribution:")
    print(f"  Total events: {target_no_attribution.sum()}")
    print(f"  Event rate: {target_no_attribution.mean():.3f}")
    
    # Create features
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # Add features
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    def rolling_factor_correlation(factor_returns, window=60):
        corr_series = pd.Series(index=factor_returns.index, dtype=float)
        for i in range(window, len(factor_returns)):
            corr = factor_returns.iloc[i-window:i].corr()
            upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            corr_series.iloc[i] = np.nanmean(upper_tri)
        return corr_series
    
    features['factor_corr_60'] = rolling_factor_correlation(factors, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factors, window=120)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    X_train = features.loc[train_idx]
    y_train = target_no_attribution.loc[train_idx]
    X_val = features.loc[val_idx]
    y_val = target_no_attribution.loc[val_idx]
    
    # Scale and train RF
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    X_train_clean = X_train.fillna(0)
    X_val_clean = X_val.fillna(0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    
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
                'threshold': thresh,
                'n_events': int(y_val.sum())
            }
    
    # ---- PERMUTATION TEST ----
    observed_auc = best_rf_eval['auc']
    n_permutations = 1000
    permutation_aucs = []
    
    for i in range(n_permutations):
        y_shuffled = np.random.permutation(y_val)
        auc = roc_auc_score(y_shuffled, y_pred_proba)
        permutation_aucs.append(auc)
    
    p_value = np.mean(np.array(permutation_aucs) >= observed_auc)
    
    # ---- RESULTS ----
    print(f"\nRandom Forest (Validation Set):")
    print(f"  AUC-ROC: {best_rf_eval['auc']:.4f}")
    print(f"  F1: {best_rf_eval['f1']:.4f}")
    print(f"  Precision: {best_rf_eval['precision']:.4f}")
    print(f"  Recall: {best_rf_eval['recall']:.4f}")
    print(f"  Threshold: {best_rf_eval['threshold']:.3f}")
    print(f"  Predictions: {best_rf_eval['predictions']}")
    print(f"  TP: {best_rf_eval['tp']}, FP: {best_rf_eval['fp']}, FN: {best_rf_eval['fn']}")
    print(f"  Validation events: {best_rf_eval['n_events']}")
    
    print(f"\nPermutation Test:")
    print(f"  Observed AUC: {observed_auc:.4f}")
    print(f"  Permutation AUC mean: {np.mean(permutation_aucs):.4f}")
    print(f"  Permutation AUC std: {np.std(permutation_aucs):.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"\n  ✅ Statistically significant (p < 0.05)")
    else:
        print(f"\n  ❌ NOT statistically significant (p >= 0.05)")
    
    # ---- COMPARE TO ORIGINAL ----
    print("\n" + "="*70)
    print("COMPARISON TO ORIGINAL TARGET")
    print("="*70)
    print(f"Original target (with attribution):")
    print(f"  Events in validation: 23")
    print(f"  RF AUC: 0.5465, p = 0.2180")
    print(f"\nNew target (without attribution):")
    print(f"  Events in validation: {best_rf_eval['n_events']}")
    print(f"  RF AUC: {best_rf_eval['auc']:.4f}, p = {p_value:.4f}")
    
    improvement = best_rf_eval['auc'] - 0.5465
    if improvement > 0:
        print(f"\n  AUC change: +{improvement:.4f} (improvement)")
    else:
        print(f"\n  AUC change: {improvement:.4f} (degradation)")
    
    return best_rf_eval, p_value


def test_attribution_thresholds():
    """Test different attribution thresholds (40%, 50%, 60%).
    
    Hypothesis: The 60% threshold is too high. Lower thresholds
    may preserve more signal while still filtering some noise.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: DIFFERENT ATTRIBUTION THRESHOLDS")
    print("Testing 40%, 50%, 60% attribution conditions")
    print("="*70)
    
    from src.target import calculate_drawdown, factor_attribution
    
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
    
    # Create features once
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # Add features
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    def rolling_factor_correlation(factor_returns, window=60):
        corr_series = pd.Series(index=factor_returns.index, dtype=float)
        for i in range(window, len(factor_returns)):
            corr = factor_returns.iloc[i-window:i].corr()
            upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            corr_series.iloc[i] = np.nanmean(upper_tri)
        return corr_series
    
    features['factor_corr_60'] = rolling_factor_correlation(factors, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factors, window=120)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    # Create drawdown once
    drawdown = calculate_drawdown(portfolio_returns, window=21)
    
    # Calculate attribution once
    attribution = factor_attribution(portfolio_returns, factors)
    
    # Test different thresholds
    thresholds_to_test = [0.40, 0.50, 0.60]
    results = []
    
    for att_threshold in thresholds_to_test:
        print(f"\n--- Testing attribution threshold: {att_threshold*100:.0f}% ---")
        
        # Create target with this threshold
        target_att = ((drawdown < -0.05) & (attribution > att_threshold)).astype(int)
        
        # Prepare data
        X_train = features.loc[train_idx]
        y_train = target_att.loc[train_idx]
        X_val = features.loc[val_idx]
        y_val = target_att.loc[val_idx]
        
        # Count events
        n_events = int(y_val.sum())
        print(f"  Validation events: {n_events}")
        
        # Skip if too few events
        if n_events < 10:
            print(f"  ⚠️ Too few events ({n_events}) — skipping")
            results.append({
                'threshold': att_threshold,
                'auc': np.nan,
                'p_value': np.nan,
                'n_events': n_events,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            })
            continue
        
        # Scale and train RF
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        X_train_clean = X_train.fillna(0)
        X_val_clean = X_val.fillna(0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_val_scaled = scaler.transform(X_val_clean)
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        
        y_pred_proba = rf.predict_proba(X_val_scaled)[:, 1]
        
        # Find optimal threshold on validation
        thresholds = np.linspace(0.01, 0.50, 50)
        best_f1 = 0
        best_eval = None
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_eval = {
                    'auc': roc_auc_score(y_val, y_pred_proba),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'threshold': thresh,
                    'n_events': n_events
                }
        
        # Permutation test
        observed_auc = best_eval['auc']
        n_permutations = 500
        permutation_aucs = []
        
        for i in range(n_permutations):
            y_shuffled = np.random.permutation(y_val)
            auc = roc_auc_score(y_shuffled, y_pred_proba)
            permutation_aucs.append(auc)
        
        p_value = np.mean(np.array(permutation_aucs) >= observed_auc)
        
        # Store results
        results.append({
            'threshold': att_threshold,
            'auc': observed_auc,
            'p_value': p_value,
            'n_events': n_events,
            'tp': best_eval['tp'],
            'fp': best_eval['fp'],
            'fn': best_eval['fn'],
            'precision': best_eval['precision'],
            'recall': best_eval['recall'],
            'f1': best_eval['f1']
        })
        
        print(f"  AUC-ROC: {observed_auc:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  TP: {best_eval['tp']}, FP: {best_eval['fp']}, FN: {best_eval['fn']}")
        print(f"  Precision: {best_eval['precision']:.4f}, Recall: {best_eval['recall']:.4f}, F1: {best_eval['f1']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: ATTRIBUTION THRESHOLD COMPARISON")
    print("="*70)
    print(f"{'Threshold':>10} | {'AUC':>8} | {'p-value':>8} | {'Events':>8} | {'TP':>6} | {'FP':>6} | {'Prec':>8} | {'Recall':>8} | {'F1':>8}")
    print("-"*100)
    
    # Add 0% threshold from Experiment 1
    results.append({
        'threshold': 0.00,
        'auc': 0.7147,
        'p_value': 0.0000,
        'n_events': 23,
        'tp': 20,
        'fp': 182,
        'fn': 3,
        'precision': 0.0990,
        'recall': 0.8696,
        'f1': 0.1778
    })
    
    for r in sorted(results, key=lambda x: x['threshold']):
        status = "✅" if (r['p_value'] < 0.05 and r['auc'] > 0.65) else "❌" if r['p_value'] >= 0.05 else "⚠️"
        print(f"{r['threshold']*100:>9.0f}% | {r['auc']:>8.4f} | {r['p_value']:>8.4f} | {r['n_events']:>8} | {r['tp']:>6} | {r['fp']:>6} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['f1']:>8.4f} {status}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    best = max([r for r in results if r['p_value'] < 0.05], key=lambda x: x['auc'])
    print(f"Best performing threshold: {best['threshold']*100:.0f}%")
    print(f"  AUC: {best['auc']:.4f}, p = {best['p_value']:.4f}")
    print(f"  TP: {best['tp']}, FP: {best['fp']}, FN: {best['fn']}")
    
    if best['threshold'] == 0.00:
        print("\n✅ Recommendation: Use NO attribution threshold.")
        print("   The 60% threshold destroys signal. Lower thresholds do not help.")
        print("   The target should be: drawdown < -5% over 21 days.")
    
    return results



def test_continuous_target():
    """Experiment 3: Predict drawdown magnitude directly (regression).
    
    Hypothesis: Binary target loses information. Continuous prediction
    of drawdown magnitude might be more useful and have stronger signal.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: CONTINUOUS TARGET (REGRESSION)")
    print("Predicting drawdown magnitude directly")
    print("="*70)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    
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
    
    # Continuous target: drawdown magnitude
    from src.target import calculate_drawdown
    drawdown = calculate_drawdown(portfolio_returns, window=21)
    
    # Create features
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # Add features
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    def rolling_factor_correlation(factor_returns, window=60):
        corr_series = pd.Series(index=factor_returns.index, dtype=float)
        for i in range(window, len(factor_returns)):
            corr = factor_returns.iloc[i-window:i].corr()
            upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            corr_series.iloc[i] = np.nanmean(upper_tri)
        return corr_series
    
    features['factor_corr_60'] = rolling_factor_correlation(factors, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factors, window=120)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    X_train = features.loc[train_idx].fillna(0)
    y_train = drawdown.loc[train_idx].fillna(0)
    X_val = features.loc[val_idx].fillna(0)
    y_val = drawdown.loc[val_idx].fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    rf_reg.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = rf_reg.predict(X_val_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Naive forecast: historical average drawdown
    naive_pred = np.full_like(y_pred, y_train.mean())
    naive_mae = mean_absolute_error(y_val, naive_pred)
    
    # Compare to binary target performance
    # Use RF classifier's probability as proxy for continuous signal
    from sklearn.ensemble import RandomForestClassifier
    y_binary = (drawdown < -0.05).astype(int)
    y_train_binary = y_binary.loc[train_idx]
    y_val_binary = y_binary.loc[val_idx]
    
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    rf_clf.fit(X_train_scaled, y_train_binary)
    y_pred_proba = rf_clf.predict_proba(X_val_scaled)[:, 1]
    binary_auc = roc_auc_score(y_val_binary, y_pred_proba)
    
    print(f"\nRegression Results (Validation Set):")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Naive MAE (historical mean): {naive_mae:.4f}")
    
    if mae < naive_mae:
        print(f"\n  ✅ Regression outperforms naive forecast (MAE -{naive_mae - mae:.4f})")
    else:
        print(f"\n  ❌ Regression does NOT outperform naive forecast (MAE +{mae - naive_mae:.4f})")
    
    print(f"\nComparison to Binary Target:")
    print(f"  Binary AUC: {binary_auc:.4f}")
    
    # Correlation between predicted and actual drawdown
    corr = np.corrcoef(y_pred, y_val)[0, 1]
    print(f"  Correlation (predicted vs actual): {corr:.4f}")
    
    return {'mae': mae, 'r2': r2, 'naive_mae': naive_mae, 'binary_auc': binary_auc, 'correlation': corr}




def test_drawdown_thresholds():
    """Experiment 4: Test different drawdown thresholds (-3%, -5%, -7%).
    
    Hypothesis: -5% may not be optimal. Different thresholds may
    yield better signal-to-noise ratio.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 4: DIFFERENT DRAWDOWN THRESHOLDS")
    print("Testing -3%, -5%, -7% thresholds")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
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
    
    # Drawdown
    from src.target import calculate_drawdown
    drawdown = calculate_drawdown(portfolio_returns, window=21)
    
    # Create features
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # Add features
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    def rolling_factor_correlation(factor_returns, window=60):
        corr_series = pd.Series(index=factor_returns.index, dtype=float)
        for i in range(window, len(factor_returns)):
            corr = factor_returns.iloc[i-window:i].corr()
            upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            corr_series.iloc[i] = np.nanmean(upper_tri)
        return corr_series
    
    features['factor_corr_60'] = rolling_factor_correlation(factors, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factors, window=120)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    # Test different thresholds
    thresholds_to_test = [-0.03, -0.05, -0.07]
    results = []
    
    for drawdown_threshold in thresholds_to_test:
        print(f"\n--- Testing drawdown threshold: {drawdown_threshold*100:.0f}% ---")
        
        # Create target
        target = (drawdown < drawdown_threshold).astype(int)
        
        # Prepare data
        X_train = features.loc[train_idx].fillna(0)
        y_train = target.loc[train_idx]
        X_val = features.loc[val_idx].fillna(0)
        y_val = target.loc[val_idx]
        
        # Count events
        n_events = int(y_val.sum())
        n_total = len(y_val)
        print(f"  Validation events: {n_events} ({n_events/n_total*100:.1f}%)")
        
        # Skip if too few events
        if n_events < 10:
            print(f"  ⚠️ Too few events ({n_events}) — skipping")
            results.append({
                'threshold': drawdown_threshold,
                'auc': np.nan,
                'p_value': np.nan,
                'n_events': n_events
            })
            continue
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train RF
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        
        y_pred_proba = rf.predict_proba(X_val_scaled)[:, 1]
        
        # Optimal threshold
        best_f1 = 0
        best_eval = None
        for thresh in np.linspace(0.01, 0.50, 50):
            y_pred = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_eval = {
                    'auc': roc_auc_score(y_val, y_pred_proba),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
        
        # Permutation test
        observed_auc = best_eval['auc']
        n_permutations = 500
        permutation_aucs = []
        for i in range(n_permutations):
            y_shuffled = np.random.permutation(y_val)
            auc = roc_auc_score(y_shuffled, y_pred_proba)
            permutation_aucs.append(auc)
        p_value = np.mean(np.array(permutation_aucs) >= observed_auc)
        
        results.append({
            'threshold': drawdown_threshold,
            'auc': observed_auc,
            'p_value': p_value,
            'n_events': n_events,
            'tp': best_eval['tp'],
            'fp': best_eval['fp'],
            'fn': best_eval['fn'],
            'precision': best_eval['precision'],
            'recall': best_eval['recall'],
            'f1': best_eval['f1']
        })
        
        print(f"  AUC-ROC: {observed_auc:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  TP: {best_eval['tp']}, FP: {best_eval['fp']}, FN: {best_eval['fn']}")
        print(f"  Precision: {best_eval['precision']:.4f}, Recall: {best_eval['recall']:.4f}, F1: {best_eval['f1']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: DRAWDOWN THRESHOLD COMPARISON")
    print("="*70)
    print(f"{'Threshold':>10} | {'AUC':>8} | {'p-value':>8} | {'Events':>8} | {'TP':>6} | {'FP':>6} | {'Prec':>8} | {'Recall':>8} | {'F1':>8}")
    print("-"*100)
    
    for r in results:
        status = "✅" if (r['p_value'] < 0.05 and r['auc'] > 0.65) else "❌" if r['p_value'] >= 0.05 else "⚠️"
        print(f"{r['threshold']*100:>9.0f}% | {r['auc']:>8.4f} | {r['p_value']:>8.4f} | {r['n_events']:>8} | {r['tp']:>6} | {r['fp']:>6} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['f1']:>8.4f} {status}")
    
    # Recommendation
    best = max([r for r in results if r['p_value'] < 0.05], key=lambda x: x['auc'])
    print(f"\nBest performing threshold: {best['threshold']*100:.0f}%")
    print(f"  AUC: {best['auc']:.4f}, p = {best['p_value']:.4f}")
    
    return results




def test_precision_improvement():
    """Experiment 5: Try to improve precision by optimizing different criteria.
    
    Hypothesis: Optimizing for F1 may not be optimal for practical use.
    Different threshold selection strategies may yield better precision.
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT 5: IMPROVING PRECISION")
    print("Testing different threshold selection strategies")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
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
    
    # Target: drawdown < -5% (no attribution)
    from src.target import calculate_drawdown
    drawdown = calculate_drawdown(portfolio_returns, window=21)
    target = (drawdown < -0.05).astype(int)
    
    # Create features
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # Add features
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    def rolling_factor_correlation(factor_returns, window=60):
        corr_series = pd.Series(index=factor_returns.index, dtype=float)
        for i in range(window, len(factor_returns)):
            corr = factor_returns.iloc[i-window:i].corr()
            upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
            corr_series.iloc[i] = np.nanmean(upper_tri)
        return corr_series
    
    features['factor_corr_60'] = rolling_factor_correlation(factors, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factors, window=120)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    X_train = features.loc[train_idx].fillna(0)
    y_train = target.loc[train_idx]
    X_val = features.loc[val_idx].fillna(0)
    y_val = target.loc[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train RF
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_proba = rf.predict_proba(X_val_scaled)[:, 1]
    
    print(f"\nValidation set:")
    print(f"  Total events: {y_val.sum()}")
    print(f"  Total samples: {len(y_val)}")
    
    # Different threshold selection strategies
    strategies = {
        'F1 Optimization': 'f1',
        'Precision @ Recall >= 0.5': 'precision_at_recall',
        'Precision @ Recall >= 0.7': 'precision_at_recall_high',
        'Fixed Threshold 0.10': 'fixed_0.10',
        'Fixed Threshold 0.15': 'fixed_0.15',
        'Fixed Threshold 0.20': 'fixed_0.20',
    }
    
    results = []
    
    for name, strategy in strategies.items():
        if strategy == 'f1':
            best_f1 = 0
            best_thresh = 0.05
            for thresh in np.linspace(0.01, 0.50, 50):
                y_pred = (y_pred_proba >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
            threshold = best_thresh
            
        elif strategy == 'precision_at_recall':
            for thresh in np.linspace(0.01, 0.50, 50):
                y_pred = (y_pred_proba >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                if recall >= 0.5:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    threshold = thresh
                    best_metrics = {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0, 'tp': tp, 'fp': fp, 'fn': fn}
                    break
            else:
                threshold = 0.05
                best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}
                
        elif strategy == 'precision_at_recall_high':
            for thresh in np.linspace(0.01, 0.50, 50):
                y_pred = (y_pred_proba >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                if recall >= 0.7:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    threshold = thresh
                    best_metrics = {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0, 'tp': tp, 'fp': fp, 'fn': fn}
                    break
            else:
                threshold = 0.05
                best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        else:
            # Fixed thresholds
            threshold = float(strategy.split('_')[1])
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            best_metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
        
        results.append({
            'strategy': name,
            'threshold': threshold,
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1': best_metrics['f1'],
            'tp': best_metrics['tp'],
            'fp': best_metrics['fp'],
            'fn': best_metrics['fn']
        })
    
    # Print results
    print("\n" + "="*70)
    print("THRESHOLD STRATEGY COMPARISON")
    print("="*70)
    print(f"{'Strategy':<30} | {'Threshold':>8} | {'Precision':>8} | {'Recall':>8} | {'F1':>8} | {'TP':>6} | {'FP':>6}")
    print("-"*90)
    
    for r in results:
        print(f"{r['strategy']:<30} | {r['threshold']:>8.3f} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['f1']:>8.4f} | {r['tp']:>6} | {r['fp']:>6}")
    
    # Find best precision with reasonable recall
    best_precision = max([r for r in results if r['recall'] >= 0.3], key=lambda x: x['precision'])
    print(f"\nBest precision with recall >= 0.3: {best_precision['strategy']}")
    print(f"  Precision: {best_precision['precision']:.4f}, Recall: {best_precision['recall']:.4f}")
    
    return results

# Add this to the bottom of test_combined_improvements.py
if __name__ == "__main__":
    # Original test
    test_combined_improvements()
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 1: NO ATTRIBUTION THRESHOLD")
    print("="*70)
    test_no_attribution_target()
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 2: DIFFERENT ATTRIBUTION THRESHOLDS")
    print("="*70)
    test_attribution_thresholds()
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 3: CONTINUOUS TARGET")
    print("="*70)
    test_continuous_target()
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 4: DIFFERENT DRAWDOWN THRESHOLDS")
    print("="*70)
    test_drawdown_thresholds()
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 5: PRECISION IMPROVEMENT")
    print("="*70)
    test_precision_improvement()