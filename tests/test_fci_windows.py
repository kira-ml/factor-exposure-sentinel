"""
test_fci_windows.py
Test different FCI moving average windows for Combined Rule.
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

def test_fci_windows():
    """Test different MA windows for Combined Rule."""
    
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
    val_end = "2020-12-31"
    train_idx = features.loc[:train_end].dropna().index
    val_idx = features.loc[train_end:val_end].dropna().index
    
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.90)
    
    # Test windows
    windows = [5, 10, 15, 20, 25, 30, 45, 60]
    results = []
    
    for window in windows:
        features[f'fci_ma{window}'] = features['fci'].rolling(window).mean()
        
        # Combined Rule: FCI > threshold AND FCI > MA AND VIX > 20
        signal = (features['fci'] > fci_threshold) & \
                 (features['fci'] > features[f'fci_ma{window}']) & \
                 (features['vix_level'] > 20)
        
        y_pred = signal.loc[val_idx].astype(int)
        y_true = target.loc[val_idx]
        
        # Metrics
        auc = roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'window': window,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': int(y_pred.sum()),
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # Print results
    print("\n" + "="*70)
    print("FCI MA WINDOW OPTIMIZATION")
    print("="*70)
    print(f"{'Window':>8} | {'AUC':>8} | {'F1':>8} | {'Prec':>8} | {'Recall':>8} | {'Preds':>8} | {'TP':>6} | {'FP':>6}")
    print("-"*80)
    for r in results:
        print(f"{r['window']:>8} | {r['auc']:>8.4f} | {r['f1']:>8.4f} | {r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['predictions']:>8} | {r['tp']:>6} | {r['fp']:>6}")
    
    # Best by F1
    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest window: {best['window']} days (F1: {best['f1']:.4f}, AUC: {best['auc']:.4f})")
    
    return results

if __name__ == "__main__":
    test_fci_windows()