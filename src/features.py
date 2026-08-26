"""
features.py
-----------
Simple, point-in-time feature engineering for factor concentration.
No look-ahead bias allowed.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def rolling_betas(asset_returns: pd.DataFrame, 
                  factor_returns: pd.DataFrame,
                  window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling factor betas for each asset.
    
    Parameters:
    -----------
    asset_returns : pd.DataFrame
        Asset returns (each column is an asset)
    factor_returns : pd.DataFrame
        Factor returns
    window : int
        Rolling window for beta estimation
    
    Returns:
    --------
    pd.DataFrame with betas for each factor (averaged across assets)
    """
    betas = pd.DataFrame(index=asset_returns.index, 
                        columns=factor_returns.columns)
    
    for i in range(window, len(asset_returns)):
        X = factor_returns.iloc[i-window:i].values
        y = asset_returns.iloc[i-window:i].values
        
        if len(X) == window and not np.any(np.isnan(X)) and not np.any(np.isnan(y)):
            model = LinearRegression()
            model.fit(X, y)
            # Average betas across all assets
            betas.iloc[i] = model.coef_.mean(axis=0)
    
    return betas


def factor_concentration_index(betas: pd.DataFrame) -> pd.Series:
    """
    Calculate Factor Concentration Index (FCI).
    
    FCI = sum(beta_k^2) / (sum(|beta_k|))^2
    
    Parameters:
    -----------
    betas : pd.DataFrame
        Rolling betas for each factor
    
    Returns:
    --------
    pd.Series with FCI values (0-1, higher = more concentrated)
    """
    numerator = (betas ** 2).sum(axis=1)
    denominator = (betas.abs().sum(axis=1)) ** 2
    
    # Handle division by zero by setting FCI to 0 when denominator is 0
    fci = numerator / denominator.replace(0, np.nan)
    fci = fci.fillna(0)  # Fill NaN (from division by zero) with 0
    
    return fci


def portfolio_hhi(weights: pd.DataFrame) -> pd.Series:
    """
    Calculate Herfindahl-Hirschman Index for portfolio weights.
    
    HHI = sum(w_i^2)
    
    Parameters:
    -----------
    weights : pd.DataFrame
        Asset weights (each row is a date, each column is an asset)
    
    Returns:
    --------
    pd.Series with HHI values
    """
    return (weights ** 2).sum(axis=1)


def create_features(asset_returns: pd.DataFrame,
                    factor_returns: pd.DataFrame,
                    portfolio_weights: pd.DataFrame,
                    macro_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create all features for the model.
    
    Parameters:
    -----------
    asset_returns : pd.DataFrame
        Asset returns
    factor_returns : pd.DataFrame
        Factor returns
    portfolio_weights : pd.DataFrame
        Portfolio weights
    macro_data : pd.DataFrame, optional
        Macro data with VIX and other indicators
    
    Returns:
    --------
    pd.DataFrame with all features
    """
    # 1. Rolling betas
    betas = rolling_betas(asset_returns, factor_returns)
    
    # 2. Factor Concentration Index
    fci = factor_concentration_index(betas)
    
    # 3. Portfolio HHI
    hhi = portfolio_hhi(portfolio_weights)
    
    # 4. Portfolio turnover (simple proxy)
    turnover = portfolio_weights.diff().abs().sum(axis=1)
    
    # Combine features
    features = pd.DataFrame(index=asset_returns.index)
    features['fci'] = fci
    features['hhi'] = hhi
    features['turnover'] = turnover
    
    # Add individual factor exposures
    for factor in betas.columns:
        features[f'beta_{factor}'] = betas[factor]
    
    
    # Add macro features if available
    if macro_data is not None:
        features['vix_level'] = macro_data
        features['vix_change'] = macro_data.pct_change()
        features['vix_vol'] = macro_data.rolling(20).std()
    
    return features