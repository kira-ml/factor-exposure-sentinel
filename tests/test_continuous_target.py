"""
test_continuous_target.py
Test continuous target (drawdown magnitude) vs binary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import fetch_all_data, ETF_TICKERS
from src.target import calculate_drawdown
from src.features import create_features

def test_continuous_target():
    """Test continuous target (drawdown magnitude)."""
    
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    factors = data['factors']
    etf_prices = data['etf_prices']
    vix = data['vix']
    
    portfolio_weights = pd.DataFrame(1/len(ETF_TICKERS), 
                                     index=etf_prices.index, 
                                     columns=ETF_TICKERS)
    returns = etf_prices.pct_change()
    portfolio_returns = (returns[ETF_TICKERS] * portfolio_weights).sum(axis=1)
    
    # Continuous target: drawdown magnitude
    drawdown = calculate_drawdown(portfolio_returns, window=21)
    
    # Features
    features = create_features(returns[ETF_TICKERS], factors, portfolio_weights, macro_data=vix)
    features = features.drop(['hhi', 'turnover'], axis=1, errors='ignore')
    features['fci_ma25'] = features['fci'].rolling(25).mean()
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    # Split
    train_idx = features.loc[:'2018-12-31'].dropna().index
    test_idx = features.loc['2019-01-01':].dropna().index
    
    X_train = features.loc[train_idx].fillna(0)
    y_train = drawdown.loc[train_idx].fillna(0)
    X_test = features.loc[test_idx].fillna(0)
    y_test = drawdown.loc[test_idx].fillna(0)
    
    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Continuous Target (Drawdown Magnitude):")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  Feature importance:")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'coef': model.coef_
    }).sort_values('coef', key=abs, ascending=False)
    print(importance.head(5).to_string(index=False))

if __name__ == "__main__":
    test_continuous_target()