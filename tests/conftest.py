"""Shared pytest fixtures — synthetic market data (no network)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=300)


@pytest.fixture
def price_data(dates):
    """Two-ticker synthetic price frame, deterministic."""
    rng = np.random.default_rng(42)
    ret_a = rng.normal(0.0004, 0.01, len(dates))
    ret_b = rng.normal(0.0002, 0.02, len(dates))
    a = 100 * np.cumprod(1 + ret_a)
    b = 50 * np.cumprod(1 + ret_b)
    return pd.DataFrame({"AAA": a, "BBB": b}, index=dates)


@pytest.fixture
def returns(price_data):
    return price_data.pct_change().dropna()
