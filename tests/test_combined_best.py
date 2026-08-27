"""
test_combined_best.py
Validate the best combined model from Day 2 experiments.
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

def test_combined_best():
    """Validate best combined rule: FCI > 90% + MA25 + VIX > 20 + Ret Vol 60 > 0.008"""
    
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
    
    # Best features
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.90)
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Best rule
    signal = (features['fci'] > fci_threshold) & \
             (features['fci'] > features['fci_ma25']) & \
             (features['vix_level'] > 20) & \
             (features['ret_vol_60'] > 0.008)
    
    y_pred = signal.loc[test_idx].astype(int)
    y_true = target.loc[test_idx]
    
    # Metrics
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("="*70)
    print("BEST COMBINED MODEL (DAY 2)")
    print("="*70)
    print(f"Rule: FCI > 90% AND FCI > 25-day MA AND VIX > 20 AND Ret Vol 60 > 0.008")
    print(f"\nMetrics:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"  Predictions: {int(y_pred.sum())}")
    print(f"  Actual events: {y_true.sum()}")
    
    # Compare to baseline
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE")
    print("="*70)
    print(f"Baseline (20-day MA): AUC 0.5446, F1 0.1232, FP 124, TP 13")
    print(f"New Best (25-day MA + Vol60): AUC {auc:.4f}, F1 {f1:.4f}, FP {fp}, TP {tp}")
    print(f"\nImprovement: AUC +{auc-0.5446:.4f}, F1 +{f1-0.1232:.4f}, FP -{124-fp}")

if __name__ == "__main__":
    test_combined_best()