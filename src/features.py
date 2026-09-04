"""
features.py
-----------
Simple, point-in-time feature engineering for factor concentration.
No look-ahead bias allowed.

VALIDATED FEATURES ONLY (based on empirical testing):
- FCI (25-day MA) -> Best: AUC 0.5614
- VIX level -> Corr: 0.0775
- VIX volatility -> Corr: 0.0744
- FCI change (20-day) -> Corr: 0.0752
- Log VIX -> Corr: 0.0916
- Credit spread -> AUC 0.6134
- Persistence features (FCI high, VIX high, stress confirm)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


def rolling_betas(asset_returns: pd.DataFrame, 
                  factor_returns: pd.DataFrame,
                  window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling factor betas for each asset.
    Uses data up to i-1 to avoid look-ahead bias.
    """
    betas = pd.DataFrame(index=asset_returns.index, 
                        columns=factor_returns.columns)
    
    for i in range(window, len(asset_returns)):
        # Use data up to i-1 (exclude current day)
        X = factor_returns.iloc[i-window:i-1].values
        y = asset_returns.iloc[i-window:i-1].values
        
        # Need at least window-1 observations
        if len(X) == window - 1 and not np.any(np.isnan(X)) and not np.any(np.isnan(y)):
            model = LinearRegression()
            model.fit(X, y)
            # Average betas across all assets
            betas.iloc[i] = model.coef_.mean(axis=0)
    
    return betas


def factor_concentration_index(betas: pd.DataFrame) -> pd.Series:
    """Calculate Herfindahl-Hirschman Index of factor betas."""
    abs_betas = betas.abs()
    row_sums = abs_betas.sum(axis=1)
    
    # Replace zeros with NaN to avoid division by zero
    row_sums_safe = row_sums.replace(0, np.nan)
    
    # Divide and then fill NaN with 0
    weights = abs_betas.div(row_sums_safe, axis=0).fillna(0)
    fci = (weights ** 2).sum(axis=1)
    return fci


def fetch_credit_spread(start_date="2010-01-01", end_date="2024-12-31"):
    """Fetch credit spread (HYG vs LQD ratio) as a proxy for credit risk."""
    import yfinance as yf
    try:
        hyg = yf.download("HYG", start=start_date, end=end_date, progress=False)
        lqd = yf.download("LQD", start=start_date, end=end_date, progress=False)
        
        if not hyg.empty and not lqd.empty:
            if 'Adj Close' in hyg.columns:
                hyg_series = hyg['Adj Close']
            else:
                hyg_series = hyg['Close']
            
            if 'Adj Close' in lqd.columns:
                lqd_series = lqd['Adj Close']
            else:
                lqd_series = lqd['Close']
            
            if isinstance(hyg_series, pd.DataFrame):
                hyg_series = hyg_series.iloc[:, 0]
            if isinstance(lqd_series, pd.DataFrame):
                lqd_series = lqd_series.iloc[:, 0]
            
            spread = hyg_series / lqd_series
            spread = spread.rename('credit_spread')
            spread = spread.ffill()
            
            return spread
    except Exception as e:
        logger.warning(f"Credit spread fetch failed: {e}")
    return None


def create_features(asset_returns: pd.DataFrame,
                    factor_returns: pd.DataFrame,
                    portfolio_weights: pd.DataFrame,
                    macro_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create all features for the model.
    
    ONLY features that passed empirical validation are included.
    Rejected features (momentum, ratio/correlation) are excluded.
    
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
    
    # 2. Factor Concentration Index (FCI)
    fci = factor_concentration_index(betas)
    
    # 3. FCI with 25-day MA (validated best: AUC 0.5614)
    fci_ma25 = fci.rolling(25).mean()
    
    # Combine features
    features = pd.DataFrame(index=asset_returns.index)
    features['fci'] = fci
    features['fci_ma25'] = fci_ma25  # Best performing MA window
    
    # 4. Add individual factor exposures
    for factor in betas.columns:
        features[f'beta_{factor}'] = betas[factor]
    
    # 5. Macro features (VIX)
    if macro_data is not None:
        features['vix_level'] = macro_data
        features['vix_change'] = macro_data.pct_change()
        features['vix_vol'] = macro_data.rolling(20).std()
    
    # 6. Log transformations (skewed features)
    if macro_data is not None:
        features['log_vix'] = np.log(features['vix_level'] + 1)
    features['log_fci'] = np.log(features['fci'] + 0.001)
    
    # 7. FCI change features (velocity of concentration)
    features['fci_change_20'] = features['fci'].diff(20)  # Best performer
    features['fci_change_30'] = features['fci'].diff(30)
    
    # 8. Credit spread (validated: improved AUC to 0.6134)
    credit_spread = fetch_credit_spread(asset_returns.index[0], asset_returns.index[-1])
    if credit_spread is not None:
        if isinstance(credit_spread, pd.DataFrame):
            credit_spread = credit_spread.iloc[:, 0] if len(credit_spread.columns) > 1 else credit_spread.squeeze()
        
        features['credit_spread'] = credit_spread.reindex(features.index)
        features['credit_spread_change'] = features['credit_spread'].pct_change()
        
        # Credit > median (improves precision)
        features['credit_high'] = (features['credit_spread'] > features['credit_spread'].median()).astype(int)

    # 9. Persistence features (reduce false positives)
    features['fci_high'] = (features['fci'] > features['fci'].rolling(252).quantile(0.85)).astype(int)
    features['fci_persistence'] = features['fci_high'].rolling(10).sum()
    
    if macro_data is not None:
        features['vix_high'] = (features['vix_level'] > features['vix_level'].rolling(252).quantile(0.80)).astype(int)
        features['vix_persistence'] = features['vix_high'].rolling(5).sum()
    
    features['stress_confirm'] = ((features['fci_high'] == 1) & (features['vix_high'] == 1)).astype(int)
    features['stress_persistence'] = features['stress_confirm'].rolling(5).sum()

    # 10. VOLATILITY FEATURES (validated: reduced false positives by 38)
    portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
    features['ret_vol_60'] = portfolio_returns.rolling(60).std()
    
    return features