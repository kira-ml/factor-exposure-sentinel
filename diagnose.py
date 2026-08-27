"""
diagnose.py
-----------
Diagnostic script for Day 2 improvements.
Analyzes Combined Rule predictions vs actual events.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import fetch_all_data
from src.target import create_target
from src.features import create_features
from src.evaluate import evaluate_model

def diagnose_combined_rule():
    """Diagnose Combined Rule performance on test set."""
    
    # Load data
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    # Portfolio
    from src.data_loader import ETF_TICKERS
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    # Target
    target = create_target(portfolio_returns, factors)
    
    # Features
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    
    # FCI MA20
    features['fci_ma20'] = features['fci'].rolling(20).mean()
    
    # Split
    test_start = "2019-01-01"
    test_idx = features.loc[test_start:].dropna().index
    y_true = target.loc[test_idx]
    
    # Combined Rule
    fci_threshold = features.loc[:'2018-12-31', 'fci'].quantile(0.90)
    signal = (features['fci'] > fci_threshold) & (features['fci'] > features['fci_ma20']) & (features['vix_level'] > 20)
    y_pred = signal.loc[test_idx].astype(int)
    
    # Classify predictions
    results = pd.DataFrame({
        'date': test_idx,
        'y_true': y_true,
        'y_pred': y_pred,
        'fci': features.loc[test_idx, 'fci'],
        'fci_ma20': features.loc[test_idx, 'fci_ma20'],
        'vix_level': features.loc[test_idx, 'vix_level']
    })
    
    # TP, FP, FN, TN
    results['type'] = 'TN'
    results.loc[(results['y_true'] == 1) & (results['y_pred'] == 1), 'type'] = 'TP'
    results.loc[(results['y_true'] == 0) & (results['y_pred'] == 1), 'type'] = 'FP'
    results.loc[(results['y_true'] == 1) & (results['y_pred'] == 0), 'type'] = 'FN'
    
    tp_dates = results[results['type'] == 'TP']['date'].tolist()
    fp_dates = results[results['type'] == 'FP']['date'].tolist()
    fn_dates = results[results['type'] == 'FN']['date'].tolist()
    
    print("="*70)
    print("COMBINED RULE DIAGNOSTICS")
    print("="*70)
    print(f"Test period: {test_idx[0]} to {test_idx[-1]}")
    print(f"Total test days: {len(test_idx)}")
    print(f"Actual events: {y_true.sum()}")
    print(f"Predictions: {y_pred.sum()}")
    print(f"\nClassification breakdown:")
    print(f"  TP: {len(tp_dates)}")
    print(f"  FP: {len(fp_dates)}")
    print(f"  FN: {len(fn_dates)}")
    print(f"  TN: {len(results) - len(tp_dates) - len(fp_dates) - len(fn_dates)}")
    
    # Show examples
    print(f"\nTP Dates (correctly flagged):")
    for d in tp_dates[:5]:
        row = results.loc[d]
        print(f"  {d.date()}: FCI={row['fci']:.4f}, VIX={row['vix_level']:.1f}")
    
    print(f"\nFP Dates (false alarms):")
    for d in fp_dates[:5]:
        row = results.loc[d]
        print(f"  {d.date()}: FCI={row['fci']:.4f}, VIX={row['vix_level']:.1f}")
    
    print(f"\nFN Dates (missed events):")
    for d in fn_dates[:5]:
        row = results.loc[d]
        print(f"  {d.date()}: FCI={row['fci']:.4f}, VIX={row['vix_level']:.1f}")
    
    # Feature comparison
    print(f"\nFeature Comparison (mean values):")
    for label, group in results.groupby('type'):
        if len(group) > 0:
            print(f"  {label} (n={len(group)}): FCI={group['fci'].mean():.4f}, VIX={group['vix_level'].mean():.1f}")
    
    # Return for further analysis
    return results

if __name__ == "__main__":
    results = diagnose_combined_rule()