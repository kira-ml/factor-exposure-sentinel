"""
test_momentum_features.py
Test momentum features as early warning signals.
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

def test_momentum_features():
    """Test momentum features (5-day, 10-day returns) with Combined Rule."""
    
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
    
    # Add momentum features
    features['ret_5d'] = portfolio_returns.rolling(5).sum()
    features['ret_10d'] = portfolio_returns.rolling(10).sum()
    features['ret_20d'] = portfolio_returns.rolling(20).sum()
    
    # Add return volatility
    features['ret_vol_5'] = portfolio_returns.rolling(5).std()
    features['ret_vol_10'] = portfolio_returns.rolling(10).std()
    
    # Split
    train_end = "2018-12-31"
    test_start = "2019-01-01"
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
    # Base rule components
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.90)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Test different momentum rules
    rules = {
        'Baseline (no momentum)': (features['fci'] > fci_threshold) & 
                                  (features['fci'] > features['fci_ma25']) & 
                                  (features['vix_level'] > 20) & 
                                  (features['ret_vol_60'] > 0.008),
        
        '+ 5d return < -0.01': (features['fci'] > fci_threshold) & 
                               (features['fci'] > features['fci_ma25']) & 
                               (features['vix_level'] > 20) & 
                               (features['ret_vol_60'] > 0.008) & 
                               (features['ret_5d'] < -0.01),
        
        '+ 10d return < -0.02': (features['fci'] > fci_threshold) & 
                                (features['fci'] > features['fci_ma25']) & 
                                (features['vix_level'] > 20) & 
                                (features['ret_vol_60'] > 0.008) & 
                                (features['ret_10d'] < -0.02),
        
        '+ 20d return < -0.03': (features['fci'] > fci_threshold) & 
                                (features['fci'] > features['fci_ma25']) & 
                                (features['vix_level'] > 20) & 
                                (features['ret_vol_60'] > 0.008) & 
                                (features['ret_20d'] < -0.03),
        
        '+ 5d vol > 0.01': (features['fci'] > fci_threshold) & 
                           (features['fci'] > features['fci_ma25']) & 
                           (features['vix_level'] > 20) & 
                           (features['ret_vol_60'] > 0.008) & 
                           (features['ret_vol_5'] > 0.01),
        
        '+ 10d vol > 0.015': (features['fci'] > fci_threshold) & 
                             (features['fci'] > features['fci_ma25']) & 
                             (features['vix_level'] > 20) & 
                             (features['ret_vol_60'] > 0.008) & 
                             (features['ret_vol_10'] > 0.015),
    }
    
    print("\n" + "="*70)
    print("MOMENTUM FEATURE ENHANCEMENT")
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
        
        status = "✅" if (auc > 0.5747 and f1 > 0.1720) else "❌" if auc < 0.5747 else "="
        print(f"\n{status} {name}:")
        print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  Preds: {int(y_pred.sum())}, TP: {tp}, FP: {fp}, FN: {fn}")
    
    # Find best
    best = max(results, key=lambda x: x['f1'])
    print(f"\n{'='*70}")
    print(f"Best momentum rule: {best['rule']}")
    print(f"  AUC: {best['auc']:.4f}, F1: {best['f1']:.4f}")
    
    return results

if __name__ == "__main__":
    test_momentum_features()