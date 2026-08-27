"""
test_volatility_features.py
Test adding rolling volatility features to Combined Rule.
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

def test_volatility_features():
    """Test Combined Rule with volatility filters."""
    
    # Load data
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    # Portfolio returns
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
    
    # Add volatility features
    features['ret_vol_20'] = portfolio_returns.rolling(20).std()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    features['vix_vol_20'] = features['vix_level'].rolling(20).std()
    
    # Test different rules
    rules = {
        'Baseline (no vol)': (features['fci'] > fci_threshold) & (features['fci'] > features['fci_ma20']) & (features['vix_level'] > 20),
        '+ Ret Vol 20': (features['fci'] > fci_threshold) & (features['fci'] > features['fci_ma20']) & (features['vix_level'] > 20) & (features['ret_vol_20'] > 0.005),
        '+ Ret Vol 60': (features['fci'] > fci_threshold) & (features['fci'] > features['fci_ma20']) & (features['vix_level'] > 20) & (features['ret_vol_60'] > 0.008),
        '+ VIX Vol 20': (features['fci'] > fci_threshold) & (features['fci'] > features['fci_ma20']) & (features['vix_level'] > 20) & (features['vix_vol_20'] > 2.0),
        '+ All Vol': (features['fci'] > fci_threshold) & (features['fci'] > features['fci_ma20']) & (features['vix_level'] > 20) & (features['ret_vol_20'] > 0.005) & (features['vix_vol_20'] > 2.0),
    }
    
    print("\n" + "="*70)
    print("VOLATILITY FEATURE ENHANCEMENT")
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
        
        print(f"\n{name}:")
        print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  Preds: {int(y_pred.sum())}, TP: {tp}, FP: {fp}, FN: {fn}")
    
    return results

if __name__ == "__main__":
    test_volatility_features()