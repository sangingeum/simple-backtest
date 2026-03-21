"""
Market data download and caching via yfinance.
"""

import streamlit as st
import yfinance as yf
import pandas as pd


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(tickers: list[str], start_date: str = "2010-01-01") -> pd.DataFrame:
    """Download and cache stock close prices.

    Returns a DataFrame with one column per ticker, NaN rows dropped.
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
            return data['Close'].dropna()
        return data.dropna()

    # Single ticker — data.columns are just ['Open','High',...]
    if 'Close' in data.columns:
        ticker_name = unique_tickers[0]
        return pd.DataFrame({ticker_name: data['Close']}).dropna()

    return data.dropna()
