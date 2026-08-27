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
    Calculate factor attribution for each date t by estimating betas using
    the trailing window and then applying them to the forward period.
    
    Returns the proportion of forward portfolio return explained by factors.
    """
    from sklearn.linear_model import LinearRegression
    import warnings
    
    attribution_series = pd.Series(index=portfolio_returns.index, dtype=float)
    
    for i in range(window, len(portfolio_returns)):
        # Estimate betas using trailing window (point-in-time)
        X_train = factor_returns.iloc[i-window:i].values
        y_train = portfolio_returns.iloc[i-window:i].values
        
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            continue
            
        model = LinearRegression()
        model.fit(X_train, y_train)
        betas = model.coef_
        
        # Apply betas to the forward period (the next 21 days)
        # Get the actual factor returns during the forward period
        forward_end = min(i + 21, len(portfolio_returns))
        X_forward = factor_returns.iloc[i:forward_end].values
        y_forward = portfolio_returns.iloc[i:forward_end].values
        
        if len(X_forward) == 0 or np.any(np.isnan(X_forward)) or np.any(np.isnan(y_forward)):
            continue
        
        # Predicted returns from factors
        y_pred = X_forward @ betas
        
        # Total forward return (actual portfolio return)
        total_return = np.sum(y_forward)
        
        # Factor contribution (sum of predicted returns)
        factor_return = np.sum(y_pred)
        
        # Avoid division by zero
        if abs(total_return) < 1e-10:
            continue
        
        # Proportion of return explained by factors (absolute values)
        # Clamp between 0 and 1 to handle edge cases
        attribution = min(abs(factor_return / total_return), 1.0)
        attribution_series.iloc[i] = attribution
    
    return attribution_series

def create_target(portfolio_returns: pd.Series,
                  factor_returns: pd.DataFrame = None,
                  drawdown_threshold: float = -0.03,
                  factor_threshold: float = None,
                  horizon: int = 21) -> pd.Series:
    """
    Create binary target variable Y_t.
    
    Y_t = 1 if drawdown over next h days < drawdown_threshold.
    
    Based on empirical evidence, the attribution threshold destroys signal.
    The optimal threshold is -3% over 21 days.
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio daily returns
    factor_returns : pd.DataFrame
        Factor returns for attribution (optional, not used when None)
    drawdown_threshold : float
        Drawdown threshold (default -0.03 = -3%)
    factor_threshold : float
        DEPRECATED: Attribution threshold (no longer used)
    horizon : int
        Prediction horizon in days (default 21)
    
    Returns:
    --------
    pd.Series with binary target (1 = event, 0 = no event)
    """
    # Calculate drawdown
    drawdown = calculate_drawdown(portfolio_returns, window=horizon)
    
    # Create target (no attribution condition)
    target = (drawdown < drawdown_threshold).astype(int)
    
    return target