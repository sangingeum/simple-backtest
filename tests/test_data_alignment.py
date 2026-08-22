"""Test that data-layer helpers handle mixed listing dates without truncation.

No network: we exercise the post-processing logic directly by building a
frame like yfinance would return and applying the same ffill/leading-drop
steps used in data.get_stock_data.
"""

import numpy as np
import pandas as pd


def _postprocess(close: pd.DataFrame) -> pd.DataFrame:
    """Mirror of data.get_stock_data's alignment logic."""
    close = close.ffill(limit=10)
    first_valid = close.apply(lambda s: s.first_valid_index())
    if first_valid.isna().all():
        return pd.DataFrame()
    start_idx = first_valid.max()
    if start_idx is not None:
        close = close.loc[start_idx:]
    return close.dropna(how="all")


def test_mixed_listing_dates_not_truncated():
    idx = pd.bdate_range("2010-01-04", periods=100)
    voo = pd.Series(np.linspace(100, 150, 100), index=idx)
    soxl = pd.concat([
        pd.Series([np.nan] * 50, index=idx[:50]),
        pd.Series(np.linspace(10, 30, 50), index=idx[50:]),
    ])
    close = pd.DataFrame({"SOXL": soxl, "VOO": voo})

    out = _postprocess(close)

    # Keeps the full overlapping history (rows from SOXL's first valid day),
    # NOT truncated per-gap or to the longest series' start.
    assert len(out) == 50
    assert out.index[0] == idx[50]
    assert out["VOO"].notna().all()
    assert out["SOXL"].notna().all()


def test_interior_holiday_gap_is_ffilled():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(range(10), index=idx, dtype=float)
    b = pd.Series(range(10), index=idx, dtype=float)
    b.iloc[4] = np.nan  # one ticker closed, the other traded
    out = _postprocess(pd.DataFrame({"A": a, "B": b}))
    assert len(out) == 10  # row kept via forward-fill
    assert out.iloc[4]["B"] == out.iloc[3]["B"]


def test_all_nan_frame_yields_empty():
    idx = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame({"A": [np.nan] * 5}, index=idx)
    assert _postprocess(df).empty


def test_ffill_is_capped_so_delisted_ticker_does_not_freeze():
    """A ticker missing >10 rows must NOT be frozen at its last price forever."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    a = pd.Series(range(30), index=idx, dtype=float)
    b = pd.Series([1.0] * 5 + [np.nan] * 25, index=idx)  # delisted after day 5
    out = _postprocess(pd.DataFrame({"A": a, "B": b}))
    # First 10 NaN rows after the last price are ffilled; after that B stays
    # NaN (rows kept via dropna(how='all') because A trades).
    assert out["B"].iloc[5:15].notna().all()
    assert out["B"].iloc[15:].isna().all()
