"""Tests for engine.calculate_metrics — metric math on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from engine import calculate_metrics


def _series(values, index):
    return pd.Series(values, index=index)


def test_basic_metrics(dates):
    hist = _series(np.linspace(100, 200, len(dates)), dates)
    rets = hist.pct_change().fillna(0)
    m = calculate_metrics(hist, rets, 100.0, dates[0], dates[-1])

    assert m["final_value"] == 200
    assert m["profit"] == pytest.approx(100.0)
    assert m["roi"] == pytest.approx(1.0)
    assert m["mdd"] == 0  # monotonic increase → no drawdown
    assert m["cagr"] > 0
    assert 0 <= m["win_rate"] <= 1
    # No losing days → sortino denominator empty → guarded to 0
    assert m["sortino"] == 0


def test_empty_history_returns_empty_dict():
    assert calculate_metrics(pd.Series(dtype=float), pd.Series(dtype=float),
                             0, None, None) == {}


def test_sharpe_zero_when_constant_returns(dates):
    hist = _series(np.full(len(dates), 100.0), dates)
    rets = pd.Series(0.0, index=dates)
    m = calculate_metrics(hist, rets, 100.0, dates[0], dates[-1])
    assert m["sharpe"] == 0
    assert m["volatility"] == 0


def test_sortino_nan_guard_when_no_negative_returns(dates):
    # Strictly increasing → zero negative days → neg std is NaN (empty series)
    vals = np.arange(100.0, 100.0 + len(dates))
    hist = _series(vals, dates)
    rets = hist.pct_change().fillna(0.001)
    m = calculate_metrics(hist, rets, 100.0, dates[0], dates[-1])
    assert not np.isnan(m["sortino"])
    assert not np.isnan(m["sharpe"])
    assert not np.isnan(m["volatility"])


def test_profit_factor(dates):
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.001, 0.01, len(dates)), index=dates)
    hist = 100 * (1 + rets).cumprod()
    m = calculate_metrics(hist, rets, 100.0, dates[0], dates[-1])

    gross_wins = rets[rets > 0].sum()
    gross_losses = -rets[rets < 0].sum()
    expected = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    assert m["profit_factor"] == pytest.approx(expected)


def test_profit_factor_infinite_with_no_losses(dates):
    rets = pd.Series(0.01, index=dates)
    hist = 100 * (1 + rets).cumprod()
    m = calculate_metrics(hist, rets, 100.0, dates[0], dates[-1])
    assert m["profit_factor"] == float("inf")


def test_drawdown_metrics_on_declining_series(dates):
    vals = np.linspace(200, 100, len(dates))
    hist = _series(vals, dates)
    rets = hist.pct_change().fillna(0)
    m = calculate_metrics(hist, rets, 200.0, dates[0], dates[-1])
    assert m["mdd"] < 0
    assert m["calmar"] < 0
