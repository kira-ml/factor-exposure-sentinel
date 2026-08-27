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


def rolling_factor_correlation(factor_returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Calculate mean pairwise correlation of factors.
    
    Parameters:
    -----------
    factor_returns : pd.DataFrame
        Factor returns
    window : int
        Rolling window for correlation
    
    Returns:
    --------
    pd.Series with mean pairwise correlation
    """
    corr_series = pd.Series(index=factor_returns.index, dtype=float)
    for i in range(window, len(factor_returns)):
        corr = factor_returns.iloc[i-window:i].corr()
        # Mean of upper triangle (excluding diagonal)
        upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
        corr_series.iloc[i] = np.nanmean(upper_tri)
    return corr_series


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



def fetch_credit_spread(start_date="2010-01-01", end_date="2024-12-31"):
    """
    Fetch credit spread (HYG vs LQD ratio) as a proxy for credit risk.
    Returns a Series with datetime index.
    """
    import yfinance as yf
    try:
        # Try getting HYG and LQD
        hyg = yf.download("HYG", start=start_date, end=end_date, progress=False)
        lqd = yf.download("LQD", start=start_date, end=end_date, progress=False)
        
        if not hyg.empty and not lqd.empty:
            # Extract Adj Close or Close as Series
            if 'Adj Close' in hyg.columns:
                hyg_series = hyg['Adj Close']
            else:
                hyg_series = hyg['Close']
            
            if 'Adj Close' in lqd.columns:
                lqd_series = lqd['Adj Close']
            else:
                lqd_series = lqd['Close']
            
            # Ensure they are Series, not DataFrames
            if isinstance(hyg_series, pd.DataFrame):
                hyg_series = hyg_series.iloc[:, 0]
            if isinstance(lqd_series, pd.DataFrame):
                lqd_series = lqd_series.iloc[:, 0]
            
            # Calculate spread and return as Series
            spread = hyg_series / lqd_series
            spread = spread.rename('credit_spread')
            
            # Forward fill to handle any missing values
            spread = spread.ffill()
            
            return spread
    except Exception as e:
        print(f"Credit spread fetch failed: {e}")
    return None





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
    
    # ---- MATHEMATICALLY JUSTIFIED FEATURES ----
    # (Added AFTER macro_data is processed so vix_level exists)
    
    # 1. FCI change features (velocity of concentration)
    features['fci_change_5'] = features['fci'].diff(5)
    features['fci_change_10'] = features['fci'].diff(10)
    features['fci_change_20'] = features['fci'].diff(20)
    features['fci_change_30'] = features['fci'].diff(30)
    
    # 2. FCI / VIX ratio (interaction)
    if macro_data is not None:
        features['fci_vix_ratio'] = features['fci'] / (features['vix_level'] + 1)
    
    # 3. Log transformations for skewed features
    if macro_data is not None:
        features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    # 4. Factor correlation (crowding proxy)
    features['factor_corr_60'] = rolling_factor_correlation(factor_returns, window=60)
    features['factor_corr_120'] = rolling_factor_correlation(factor_returns, window=120)

    # Add credit spread if available
    credit_spread = fetch_credit_spread(asset_returns.index[0], asset_returns.index[-1])
    if credit_spread is not None:
        # Ensure it's a Series, not DataFrame
        if isinstance(credit_spread, pd.DataFrame):
            # Take first column if multiple
            if len(credit_spread.columns) > 1:
                credit_spread = credit_spread.iloc[:, 0]
            else:
                credit_spread = credit_spread.squeeze()
        
        features['credit_spread'] = credit_spread.reindex(features.index)
        features['credit_spread_change'] = features['credit_spread'].pct_change()

    
    return features




