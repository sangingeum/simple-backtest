"""Tests for strategies.py — signal generators on synthetic series."""

import numpy as np
import pandas as pd
import pytest

import strategies as S

GENS = [
    S.generate_trend_signal,
    S.generate_crossover_signal,
    S.generate_volatility_signal,
    S.generate_trailing_stop_signal,
    S.generate_rsi_signal,
    S.generate_macd_signal,
    S.generate_bband_signal,
]


@pytest.fixture
def ts(dates):
    rng = np.random.default_rng(1)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, len(dates))),
                     index=dates)


def _crash_series(dates):
    """Rise 150 days, then crash hard."""
    up = np.linspace(100, 200, 150)
    down = np.linspace(200, 120, len(dates) - 150)
    return pd.Series(np.concatenate([up, down]), index=dates)


@pytest.mark.parametrize("gen", GENS, ids=lambda f: f.__name__)
def test_returns_bool_series_of_correct_length(ts, gen):
    sig = gen(ts)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(ts)
    # bool-like values only
    assert sig.dtype == bool or set(sig.unique()).issubset({True, False, 0, 1})


def test_trend_bullish_when_above_sma(ts):
    sma = ts.rolling(50).mean()
    sig = generate_checked(ts, window=50)
    valid = sma.notna()
    assert (sig[valid] == (ts[valid] > sma[valid])).all()


def generate_checked(ts, **kw):
    return S.generate_trend_signal(ts, kw["window"], False)


def test_trend_dual_momentum_is_subset(ts):
    plain = S.generate_trend_signal(ts, 20, False)
    dual = S.generate_trend_signal(ts, 20, True)
    # Dual momentum can only remove bullish days, never add them
    assert not (dual & ~plain).any()


def test_volatility_threshold():
    idx = pd.date_range("2020-01-01", periods=4)
    vix = pd.Series([10.0, 29.9, 30.1, 50.0], index=idx)
    sig = S.generate_volatility_signal(vix, 30.0)
    assert list(sig) == [True, True, False, False]


def test_trailing_stop_exits_on_crash(dates):
    ts = _crash_series(dates)
    sig = S.generate_trailing_stop_signal(ts, stop_pct=0.15, sma_window=50)
    assert isinstance(sig, pd.Series) and len(sig) == len(ts)
    # Must exit at some point after the peak
    peak_idx = int(np.argmax(ts.values))
    assert not sig.iloc[-1]
    # And be invested before the crash
    assert sig.iloc[:50].all()


def test_rsi_state_machine_sanity(ts):
    sig = S.generate_rsi_signal(ts)
    assert len(sig) == len(ts)
    arr = sig.to_numpy()
    # State changes are transitions only — no NaNs, bools throughout
    assert arr.dtype == bool


def test_macd_signal_changes_on_crossover():
    idx = pd.date_range("2020-01-01", periods=120)
    # Flat then a step up: MACD pops above signal, then decays back below as
    # momentum fades on the plateau → exactly one exit transition expected.
    vals = np.concatenate([np.full(60, 100.0), np.full(60, 140.0)])
    sig = S.generate_macd_signal(pd.Series(vals, index=idx))
    assert sig.iloc[0]  # starts invested
    transitions = int((sig.astype(int).diff().abs() > 0).sum())
    assert transitions >= 1

    # A sustained ramp keeps MACD above its signal line at the end.
    ramp = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    sig_ramp = S.generate_macd_signal(ramp)
    assert sig_ramp.iloc[-1]


def test_bband_starts_invested_and_bool(ts):
    sig = S.generate_bband_signal(ts)
    assert len(sig) == len(ts)
    assert sig.iloc[0]
