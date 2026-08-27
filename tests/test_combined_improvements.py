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
    """Test all mathematically justified improvements."""
    
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
    test_start = "2019-01-01"
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
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
        y_pred = signal.loc[test_idx].astype(int)
        y_true = target.loc[test_idx]
        
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
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Day 3 Best: AUC 0.6002, F1 0.2184, TP 19, FP 81")
    
    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest New: {best['rule']}")
    print(f"  AUC: {best['auc']:.4f}, F1: {best['f1']:.4f}, TP: {best['tp']}, FP: {best['fp']}")
    print(f"  Improvement: AUC +{best['auc']-0.6002:.4f}, F1 +{best['f1']-0.2184:.4f}")
    
    return results

if __name__ == "__main__":
    test_combined_improvements()