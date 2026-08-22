"""
Market data download and caching via yfinance.
"""

import streamlit as st
import yfinance as yf
import pandas as pd


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(tickers: list[str], start_date: str = "2010-01-01") -> pd.DataFrame:
    """Download and cache stock close prices.

    Returns a DataFrame with one column per ticker. Rows are forward-filled
    so tickers with different trading calendars / listing dates don't
    truncate the whole dataset; only leading rows that are still NaN (before
    ANY ticker has listed) are dropped.
    """
    if not tickers:
        return pd.DataFrame()

    unique_tickers = sorted(set(t.upper() for t in tickers))

    try:
        data = yf.download(
            unique_tickers,
            start=start_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            close = data['Close']
        else:
            close = data
    elif 'Close' in data.columns:
        # Single ticker — columns are just ['Open','High',...]
        close = pd.DataFrame({unique_tickers[0]: data['Close']})
    else:
        close = data

    # Forward-fill gaps (holidays per-ticker, later listings) with a cap so
    # delisted tickers don't freeze at their last price forever; then drop
    # only the leading rows where every ticker is still NaN.
    close = close.ffill(limit=10)
    first_valid = close.apply(lambda s: s.first_valid_index())
    if first_valid.isna().all():
        return pd.DataFrame()
    start_idx = first_valid.max()
    if start_idx is not None:
        close = close.loc[start_idx:]
    return close.dropna(how="all")
