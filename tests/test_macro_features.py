"""
test_macro_features.py
Test macro features (credit spreads) for regime filtering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.data_loader import fetch_all_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features

def fetch_credit_spread(start_date="2010-01-01", end_date="2024-12-31"):
    """Fetch BAA and 10-year Treasury yields from Yahoo Finance."""
    print("Fetching credit spread data...")
    
    # BAA corporate bond yield (ETF: LQD or use ^BAM)
    # Use HYG for high yield spread or AGG for broad
    baa = yf.download("LQD", start=start_date, end=end_date, progress=False)
    if 'Adj Close' in baa.columns:
        baa_series = baa['Adj Close']
    else:
        baa_series = baa['Close']
    
    # 10-year Treasury yield (^TNX)
    tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)
    if 'Adj Close' in tnx.columns:
        tnx_series = tnx['Adj Close']
    else:
        tnx_series = tnx['Close']
    
    # Align and compute spread
    credit_spread = (baa_series / tnx_series).dropna()
    return credit_spread

def test_macro_features():
    """Test credit spread as a macro regime filter."""
    
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
    
    # Add macro features
    try:
        credit_spread = fetch_credit_spread()
        features['credit_spread'] = credit_spread.reindex(features.index)
        features['credit_spread_change'] = features['credit_spread'].pct_change()
        features['credit_spread_vol'] = features['credit_spread'].rolling(20).std()
        has_credit = True
    except Exception as e:
        print(f"Warning: Could not fetch credit spread: {e}")
        has_credit = False
    
    # Add features
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_end = "2018-12-31"
    test_start = "2019-01-01"
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc[test_start:].dropna().index
    
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.90)
    
    # Base rule
    base_signal = (features['fci'] > fci_threshold) & \
                  (features['fci'] > features['fci_ma25']) & \
                  (features['vix_level'] > 20) & \
                  (features['ret_vol_60'] > 0.008)
    
    print("\n" + "="*70)
    print("MACRO FEATURE ENHANCEMENT")
    print("="*70)
    
    # Evaluate base
    y_pred = base_signal.loc[test_idx].astype(int)
    y_true = target.loc[test_idx]
    base_auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    base_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    base_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    base_f1 = 2 * base_precision * base_recall / (base_precision + base_recall) if (base_precision + base_recall) > 0 else 0
    
    print(f"\nBase Rule (No Macro):")
    print(f"  AUC: {base_auc:.4f}, F1: {base_f1:.4f}, TP: {tp}, FP: {fp}")
    
    if has_credit:
        # Test credit spread filters
        credit_rules = {
            'Credit Spread > median': (features['fci'] > fci_threshold) & 
                                      (features['fci'] > features['fci_ma25']) & 
                                      (features['vix_level'] > 20) & 
                                      (features['ret_vol_60'] > 0.008) & 
                                      (features['credit_spread'] > features['credit_spread'].median()),
            
            'Credit Spread > 75th percentile': (features['fci'] > fci_threshold) & 
                                               (features['fci'] > features['fci_ma25']) & 
                                               (features['vix_level'] > 20) & 
                                               (features['ret_vol_60'] > 0.008) & 
                                               (features['credit_spread'] > features['credit_spread'].quantile(0.75)),
            
            'Credit Spread Increasing': (features['fci'] > fci_threshold) & 
                                        (features['fci'] > features['fci_ma25']) & 
                                        (features['vix_level'] > 20) & 
                                        (features['ret_vol_60'] > 0.008) & 
                                        (features['credit_spread_change'] > 0.01),
        }
        
        for name, signal in credit_rules.items():
            y_pred = signal.loc[test_idx].astype(int)
            y_true = target.loc[test_idx]
            
            auc = roc_auc_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            status = "✅" if (auc > base_auc and f1 > base_f1) else "❌"
            print(f"\n{status} {name}:")
            print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, TP: {tp}, FP: {fp}")
    else:
        print("\n⚠️ Credit spread data unavailable. Skipping macro tests.")

if __name__ == "__main__":
    test_macro_features()