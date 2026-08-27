"""
test_credit_spread.py
Test credit spread as a macro filter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import roc_auc_score
from src.data_loader import fetch_all_data, ETF_TICKERS
from src.target import create_target
from src.features import create_features


def fetch_credit_spread(start_date="2010-01-01", end_date="2024-12-31"):
    """Fetch credit spread from FRED via yfinance.
    
    Uses ^HYG-LQD (High Yield vs IG spread) as primary proxy.
    Falls back to individual bond ETF spreads if unavailable.
    """
    # Primary: actual credit spread index from FRED
    spread_tickers = ["^HYG-LQD", "BAMLH0A0HYM2", "DCOILWTICO"]
    
    for ticker in spread_tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
            
            # Handle yfinance MultiIndex columns (>=0.2.31)
            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close'].squeeze() if 'Close' in df.columns.get_level_values(0) else df.iloc[:, 0]
            else:
                close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            
            close = close.dropna()
            if len(close) > 100:
                print(f"Using {ticker} as credit spread proxy ({len(close)} obs)")
                return close
                
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            continue
    
    # Fallback: construct synthetic spread from HYG/LQD ratio
    try:
        print("Falling back to synthetic HYG/LQD spread...")
        hyg = yf.download("HYG", start=start_date, end=end_date, progress=False)['Close'].squeeze()
        lqd = yf.download("LQD", start=start_date, end=end_date, progress=False)['Close'].squeeze()
        synthetic = (hyg / lqd).dropna()
        if len(synthetic) > 100:
            print(f"Synthetic spread ready ({len(synthetic)} obs)")
            return synthetic
    except Exception as e:
        print(f"Synthetic spread failed: {e}")
    
    print("WARNING: No credit spread data available")
    return None


def safe_roc_auc(y_true, y_pred):
    """Compute ROC AUC with edge-case handling."""
    try:
        if y_pred.nunique() < 2:
            return float('nan')
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return float('nan')


def test_credit_spread():
    """Test credit spread enhancement on best macro rule."""
    
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    portfolio_weights = pd.DataFrame(
        1 / len(ETF_TICKERS),
        index=etf_prices.index,
        columns=ETF_TICKERS
    )
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    target = create_target(portfolio_returns, factors)
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # --- Add credit spread features ---
    credit = fetch_credit_spread()
    has_credit = False
    if credit is not None:
        # Align to feature index, forward-fill gaps from differing calendars
        features['credit'] = credit.reindex(features.index).ffill()
        features['credit_change'] = features['credit'].pct_change()
        features['credit_vol'] = features['credit'].rolling(20).std()
        has_credit = True
    
    # --- Train / Test split ---
    train_end = "2018-12-31"
    train_idx = features.loc[:train_end].dropna().index
    test_idx = features.loc["2019-01-01":].dropna().index
    
    if len(test_idx) == 0:
        raise ValueError("No test observations after 2019-01-01. Check data range.")
    
    fci_threshold = features.loc[train_idx, 'fci'].quantile(0.95)
    
    # Base conditions (computed once, NaN-safe)
    base_cond = (
        (features['fci'] > fci_threshold) &
        (features['fci'] > features['fci_ma25']) &
        (features['vix_level'] > 19) &
        (features['ret_vol_60'] > 0.008)
    ).fillna(False)
    
    rules = {'Day 3 Best': base_cond}
    
    if has_credit:
        # Compute median ONLY on training data to avoid lookahead bias
        credit_train_median = features.loc[train_idx, 'credit'].median()
        
        rules['+ Credit > train median'] = base_cond & (
            features['credit'] > credit_train_median
        ).fillna(False)
        
        rules['+ Credit increasing'] = base_cond & (
            features['credit_change'] > 0
        ).fillna(False)
    
    # --- Evaluate ---
    print("\nCredit Spread Enhancement Results:")
    print("-" * 50)
    for name, signal in rules.items():
        y_pred = signal.loc[test_idx].astype(int)
        y_true = target.loc[test_idx]
        
        # Ensure alignment (both should share test_idx, but be safe)
        common_idx = y_pred.index.intersection(y_true.dropna().index)
        auc = safe_roc_auc(y_true.loc[common_idx], y_pred.loc[common_idx])
        n_signals = y_pred.sum()
        
        print(f"{name:<25s} | AUC={auc:.4f} | Signals={n_signals:>4d}/{len(common_idx)}")


if __name__ == "__main__":
    test_credit_spread()