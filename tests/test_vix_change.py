"""
test_vix_change.py
Test Combined Rule with VIX change filter.
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

def test_vix_change():
    """Test Combined Rule with VIX change filter."""
    
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
    
    # Split
    train_end = "2018-12-31"
    test_start = "2019-01-01"
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.90)
    features['fci_ma20'] = features['fci'].rolling(20).mean()
    features['vix_change_1d'] = features['vix_level'].pct_change()
    
    # Test VIX change thresholds
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
    
    print("\n" + "="*70)
    print("VIX CHANGE ENHANCEMENT")
    print("="*70)
    
    results = []
    for pct_change in thresholds:
        signal = (features['fci'] > fci_threshold) & \
                 (features['fci'] > features['fci_ma20']) & \
                 (features['vix_level'] > 20) & \
                 (features['vix_change_1d'] > pct_change)
        
        y_pred = signal.loc[test_idx].astype(int)
        y_true = target.loc[test_idx]
        
        auc = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'vix_change': pct_change,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': int(y_pred.sum()),
            'tp': tp,
            'fp': fp
        })
        
        print(f"VIX change > {pct_change*100:.0f}%: AUC={auc:.4f}, F1={f1:.4f}, Preds={int(y_pred.sum())}, TP={tp}, FP={fp}")
    
    return results

if __name__ == "__main__":
    test_vix_change()