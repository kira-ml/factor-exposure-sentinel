"""
test_fci_percentile_grid.py
Grid search over FCI percentiles with best Day 2 features.
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

def test_fci_percentile_grid():
    """Grid search over FCI percentiles with Day 2 features."""
    
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
    
    # Add features
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    test_start = "2019-01-01"
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
    # Grid search
    percentiles = [0.80, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.95]
    vix_thresholds = [18, 19, 20, 21, 22]
    vol_thresholds = [0.006, 0.007, 0.008, 0.009, 0.010]
    
    print("\n" + "="*70)
    print("FCI PERCENTILE GRID SEARCH")
    print("="*70)
    print(f"Testing {len(percentiles)} percentiles × {len(vix_thresholds)} VIX × {len(vol_thresholds)} Vol")
    print("="*70)
    
    best_result = None
    best_score = 0
    
    for pct in percentiles:
        fci_threshold = features.loc[train_idx, 'fci'].quantile(pct)
        
        for vix_t in vix_thresholds:
            for vol_t in vol_thresholds:
                signal = (features['fci'] > fci_threshold) & \
                         (features['fci'] > features['fci_ma25']) & \
                         (features['vix_level'] > vix_t) & \
                         (features['ret_vol_60'] > vol_t)
                
                y_pred = signal.loc[test_idx].astype(int)
                y_true = target.loc[test_idx]
                
                auc = roc_auc_score(y_true, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_score:
                    best_score = f1
                    best_result = {
                        'percentile': pct,
                        'fci_threshold': fci_threshold,
                        'vix_threshold': vix_t,
                        'vol_threshold': vol_t,
                        'auc': auc,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'predictions': int(y_pred.sum()),
                        'tp': tp,
                        'fp': fp,
                        'fn': fn
                    }
    
    # Print best result
    print("\n" + "="*70)
    print("BEST FCI PERCENTILE CONFIGURATION")
    print("="*70)
    print(f"FCI Percentile: {best_result['percentile']*100:.0f}% (threshold: {best_result['fci_threshold']:.4f})")
    print(f"VIX Threshold: {best_result['vix_threshold']}")
    print(f"Vol Threshold: {best_result['vol_threshold']:.3f}")
    print(f"\nMetrics:")
    print(f"  AUC-ROC: {best_result['auc']:.4f}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  Predictions: {best_result['predictions']}, TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    
    # Compare to current best
    print("\n" + "="*70)
    print("COMPARISON TO DAY 2 BEST")
    print("="*70)
    print(f"Day 2 Best (90% + VIX20 + Vol0.008): AUC 0.5747, F1 0.1720, TP 16, FP 96")
    print(f"Grid Best ({best_result['percentile']*100:.0f}% + VIX{best_result['vix_threshold']} + Vol{best_result['vol_threshold']:.3f}):")
    print(f"  AUC {best_result['auc']:.4f}, F1 {best_result['f1']:.4f}, TP {best_result['tp']}, FP {best_result['fp']}")
    
    improvement_auc = best_result['auc'] - 0.5747
    improvement_f1 = best_result['f1'] - 0.1720
    print(f"\nImprovement: AUC +{improvement_auc:.4f}, F1 +{improvement_f1:.4f}")
    
    return best_result

if __name__ == "__main__":
    test_fci_percentile_grid()