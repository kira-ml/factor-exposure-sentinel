"""
data_loader.py
--------------
Fetches and caches all required data for the Factor Exposure Sentinel project:
- Fama-French 5 factors (daily) from local CSV
- ETF adjusted close prices from Yahoo Finance
- VIX (volatility index) from Yahoo Finance

All data is aligned to a daily frequency, with factors already converted to decimals.
Caching is used to avoid repeated downloads and heavy CSV parsing.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  CONFIGURATION
# -------------------------------------------------------------------
DATA_DIR = Path("D:/quant-finance-ml/factor-exposure-sentinel/data")
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ETF tickers for multi-asset portfolio construction
ETF_TICKERS = [
    "SPY",   # US equities
    "AGG",   # US bonds
    "GLD",   # Gold
    "IJS",   # Small cap value
    "EFA",   # Developed ex-US equities
]

FF_FILE = DATA_DIR / "F-F_Research_Data_5_Factors_2x3_daily.csv"
VIX_TICKER = "^VIX"


# -------------------------------------------------------------------
#  LOADING FUNCTIONS
# -------------------------------------------------------------------

def load_ff_factors(start_date="2010-01-01", end_date="2026-06-30", use_cache=True):
    """
    Load Fama-French 5-factor daily returns from local CSV.
    Handles parsing, footer removal, and date filtering.
    Returns DataFrame with datetime index and factor columns as decimals.
    """
    cache_file = CACHE_DIR / f"ff_factors_{start_date}_{end_date}.parquet"
    
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached FF factors from {cache_file}")
        return pd.read_parquet(cache_file)
    
    logger.info("Loading raw FF factors from CSV...")
    # Load raw data, skipping the first 4 rows (description header)
    df_raw = pd.read_csv(FF_FILE, skiprows=4)
    
    # Rename the first column to 'Date'
    df_raw.rename(columns={df_raw.columns[0]: 'Date'}, inplace=True)
    
    # Remove any non‑numeric rows (e.g., copyright footer)
    df_raw = df_raw[pd.to_numeric(df_raw['Date'], errors='coerce').notna()]
    df_raw = df_raw.dropna(subset=['Date'])
    
    # Convert Date to datetime
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%Y%m%d')
    
    # Convert factor columns to decimals (percentage points / 100)
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    for col in factor_cols:
        df_raw[col] = df_raw[col].astype(float) / 100.0
    
    # Set Date as index
    df_raw.set_index('Date', inplace=True)
    df_raw.index = pd.to_datetime(df_raw.index, format='%Y%m%d')
    
    # Filter to date range
    df = df_raw.loc[start_date:end_date].copy()
    
    # Cache the result
    if use_cache:
        df.to_parquet(cache_file)
        logger.info(f"Cached FF factors to {cache_file}")
    
    return df


def fetch_etf_data(tickers=ETF_TICKERS, start_date="2010-01-01", end_date="2026-06-30", use_cache=True):
    """
    Download daily adjusted close prices for a list of ETF tickers via yfinance.
    Returns DataFrame with columns = tickers, index = date.
    """
    cache_file = CACHE_DIR / f"etf_prices_{start_date}_{end_date}.parquet"
    
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached ETF prices from {cache_file}")
        return pd.read_parquet(cache_file)
    
    logger.info(f"Downloading ETF data for {tickers}...")
    prices = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Some tickers may not have 'Adj Close' (e.g., if delisted), fallback to 'Close'
        if not df.empty and 'Adj Close' in df.columns:
            prices[ticker] = df['Adj Close']
        elif not df.empty and 'Close' in df.columns:
            prices[ticker] = df['Close']
        else:
            # If no price data available, fill with NaN
            logger.warning(f"No price data for {ticker}, filling with NaN")
            prices[ticker] = pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='B'), dtype=float)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc[start_date:end_date]
    
    # Cache
    if use_cache:
        prices.to_parquet(cache_file)
        logger.info(f"Cached ETF prices to {cache_file}")
    
    return prices


def fetch_vix(start_date="2010-01-01", end_date="2026-06-30", use_cache=True):
    """
    Download VIX daily closing values via yfinance.
    Returns Series with datetime index.
    """
    cache_file = CACHE_DIR / f"vix_{start_date}_{end_date}.parquet"
    
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached VIX from {cache_file}")
        return pd.read_parquet(cache_file)['VIX']
    
    logger.info("Downloading VIX data...")
    vix = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False)
    
    # Extract Adj Close directly - handles both flat and MultiIndex
    if 'Adj Close' in vix.columns:
        vix_series = vix['Adj Close'].rename('VIX')
    elif isinstance(vix.columns, pd.MultiIndex):
        # Try to find Adj Close in MultiIndex
        if 'Adj Close' in vix.columns.get_level_values(0):
            vix_series = vix.xs('Adj Close', axis=1, level=0).rename('VIX')
        elif 'Adj Close' in vix.columns.get_level_values(1):
            vix_series = vix.xs('Adj Close', axis=1, level=1).rename('VIX')
        else:
            logger.warning("'Adj Close' not found in MultiIndex, using first column")
            vix_series = vix.iloc[:, 0].rename('VIX')
    else:
        # Fallback to first column if 'Adj Close' is not found
        logger.warning("'Adj Close' column not found, using first available column for VIX")
        vix_series = vix.iloc[:, 0].rename('VIX')
    
    # Cache
    if use_cache:
        vix_series.to_frame().to_parquet(cache_file)
        logger.info(f"Cached VIX to {cache_file}")
    
    return vix_series

def fetch_all_data(start_date="2010-01-01", end_date="2026-06-30", use_cache=True):
    """
    Orchestrate fetching of all data sources and return as a dictionary.
    Aligns all data to daily frequency (forward-fill factors? Actually factors are already daily).
    """
    logger.info("Fetching all data...")
    
    # 1. Fama-French factors
    factors = load_ff_factors(start_date, end_date, use_cache)
    
    # 2. ETF prices
    etf_prices = fetch_etf_data(ETF_TICKERS, start_date, end_date, use_cache)
    
    # 3. VIX
    vix = fetch_vix(start_date, end_date, use_cache)
    
    # Ensure all indices are DateTimeIndex and align to common date range
    common_idx = factors.index.intersection(etf_prices.index).intersection(vix.index)
    factors = factors.loc[common_idx]
    etf_prices = etf_prices.loc[common_idx]
    vix = vix.loc[common_idx]
    
    logger.info(f"All data aligned. Shape: {len(common_idx)} rows.")
    
    return {
        'factors': factors,
        'etf_prices': etf_prices,
        'vix': vix,
        'dates': common_idx
    }


# -------------------------------------------------------------------
#  TEST / STANDALONE EXECUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test
    data = fetch_all_data(start_date="2010-01-01", end_date="2024-12-31", use_cache=True)
    
    print("\n=== Factors ===")
    print(data['factors'].head())
    
    print("\n=== ETF Prices ===")
    print(data['etf_prices'].head())
    
    print("\n=== VIX ===")
    print(data['vix'].head())
    
    print(f"\nTotal rows: {len(data['dates'])}")