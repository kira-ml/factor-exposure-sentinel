"""
test_target_horizons.py
Test different prediction horizons for target definition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from src.data_loader import fetch_all_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features

def test_target_horizons():
    """Test different prediction horizons."""
    
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    horizons = [5, 10, 15, 21, 30, 45, 60]
    results = []
    
    for h in horizons:
        target = create_target(portfolio_returns, factors, horizon=h)
        features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
        features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
        features['fci_ma25'] = features['fci'].rolling(25).mean()
        features['ret_vol_60'] = portfolio_returns.rolling(60).std()
        
        train_end = "2018-12-31"
        train_idx = features.loc[:train_end].dropna().index
        test_idx = features.loc["2019-01-01":].dropna().index
        
        fci_threshold = features.loc[train_idx, 'fci'].quantile(0.95)
        signal = (features['fci'] > fci_threshold) & \
                 (features['fci'] > features['fci_ma25']) & \
                 (features['vix_level'] > 19) & \
                 (features['ret_vol_60'] > 0.008)
        
        y_pred = signal.loc[test_idx].astype(int)
        y_true = target.loc[test_idx]
        auc = roc_auc_score(y_true, y_pred)
        event_rate = y_true.mean()
        
        results.append({
            'horizon': h,
            'auc': auc,
            'event_rate': event_rate,
            'events': int(y_true.sum()),
            'predictions': int(y_pred.sum())
        })
        
        print(f"Horizon {h}d: AUC={auc:.4f}, Events={int(y_true.sum())}, Rate={event_rate:.3f}")
    
    best = max(results, key=lambda x: x['auc'])
    print(f"\nBest horizon: {best['horizon']} days (AUC: {best['auc']:.4f})")
    return results

if __name__ == "__main__":
    test_target_horizons()