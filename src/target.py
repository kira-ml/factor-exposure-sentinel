"""
target.py
---------
Defines the target variable Y_t for factor concentration events.
Calculates drawdowns and factor attribution to label events.
"""

import pandas as pd
import numpy as np
from typing import Optional

def calculate_drawdown(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculate forward-looking drawdown over h days.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio daily returns
    window : int
        Forward-looking window (default 21 trading days = 1 month)
    
    Returns:
    --------
    pd.Series with drawdown values
    """
    # Forward rolling window drawdown
    drawdown = returns.rolling(window=window).apply(
        lambda x: (x + 1).prod() - 1
    )
    return drawdown.shift(-window)  # Shift forward to avoid look-ahead

def factor_attribution(portfolio_returns: pd.Series, 
                       factor_returns: pd.DataFrame,
                       window: int = 252) -> pd.Series:
    """
    Calculate rolling factor attribution using linear regression.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    factor_returns : pd.DataFrame
        Factor returns (Fama-French 5-factor)
    window : int
        Rolling window for regression
    
    Returns:
    --------
    pd.Series with R-squared of factor model (factor attribution)
    """
    # Use simple R-squared as factor attribution proxy
    from sklearn.linear_model import LinearRegression
    
    r2_series = pd.Series(index=portfolio_returns.index, dtype=float)
    
    for i in range(window, len(portfolio_returns)):
        X = factor_returns.iloc[i-window:i].values
        y = portfolio_returns.iloc[i-window:i].values
        
        if len(X) == window and not np.any(np.isnan(X)) and not np.any(np.isnan(y)):
            model = LinearRegression()
            model.fit(X, y)
            r2_series.iloc[i] = model.score(X, y)
    
    return r2_series

def create_target(portfolio_returns: pd.Series,
                  factor_returns: pd.DataFrame,
                  drawdown_threshold: float = -0.05,
                  factor_threshold: float = 0.60,
                  horizon: int = 21) -> pd.Series:
    """
    Create binary target variable Y_t.
    
    Y_t = 1 if:
        1. Drawdown over next h days < -5%
        2. Factor attribution > 60%
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio daily returns
    factor_returns : pd.DataFrame
        Factor returns for attribution
    drawdown_threshold : float
        Drawdown threshold (default -0.05 = -5%)
    factor_threshold : float
        Factor attribution threshold (default 0.60 = 60%)
    horizon : int
        Prediction horizon in days (default 21)
    
    Returns:
    --------
    pd.Series with binary target (1 = event, 0 = no event)
    """
    # Calculate drawdown
    drawdown = calculate_drawdown(portfolio_returns, window=horizon)
    
    # Calculate factor attribution (R-squared)
    attribution = factor_attribution(portfolio_returns, factor_returns)
    
    # Create target
    target = ((drawdown < drawdown_threshold) & (attribution > factor_threshold)).astype(int)
    
    return target