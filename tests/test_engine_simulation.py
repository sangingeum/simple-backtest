"""Tests for engine simulation loop — rebalance costs, tax, year boundary."""

import numpy as np
import pandas as pd
import pytest

from config import TAX_RATE
from engine import (
    BacktestState,
    _execute_rebalance,
    _handle_year_boundary,
    run_backtest,
)


# ── _execute_rebalance ────────────────────────────────────────────────────────


def test_rebalance_moves_to_targets():
    st = BacktestState(
        assets_value={"AAA": 100.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 100.0, "BBB": 0.0},
        cash_balance=0.0,
    )
    _execute_rebalance(st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5}, 0.0, 0.0, "Immediate")
    assert st.assets_value["AAA"] == pytest.approx(50.0)
    assert st.assets_value["BBB"] == pytest.approx(50.0)
    # Selling AAA at no gain (basis = value) → no realized gain
    assert st.annual_realized_gain == 0
    assert st.cash_balance >= 0


def test_rebalance_tax_on_gain():
    st = BacktestState(
        assets_value={"AAA": 200.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 100.0, "BBB": 0.0},
        cash_balance=0.0,
        last_year=2020,
    )
    # Sell half of AAA (basis 50 of 100) → gain 50
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5},
        slippage_rate=0.0, commission_fee=0.0,
        tax_settlement_mode="Immediate", tax_threshold=0.0,
    )
    expected = 50 * TAX_RATE
    assert st.annual_realized_gain == pytest.approx(50.0)
    # Portfolio is fully invested, so implied cash after targeting is 0;
    # immediate costs/tax are settled proportionally out of asset values.
    assert st.cash_balance == 0
    assert expected > 0
    total_after = sum(st.assets_value.values()) + st.cash_balance
    assert total_after == pytest.approx(200.0 - expected)


def test_immediate_tax_not_erased_when_fully_invested():
    """Regression: a fully-invested zero-cash rebalance must show real tax drag."""
    st = BacktestState(
        assets_value={"AAA": 200.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 100.0, "BBB": 0.0},
        cash_balance=0.0,
        last_year=2020,
    )
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5},
        slippage_rate=0.0, commission_fee=0.0,
        tax_settlement_mode="Immediate", tax_threshold=0.0,
    )
    expected_tax = 50 * TAX_RATE
    total_after = sum(st.assets_value.values()) + st.cash_balance
    # Before the fix the clamp wiped the entire liability (drag was 0).
    assert 200.0 - total_after == pytest.approx(expected_tax)
    assert st.cash_balance == 0.0


def test_clamp_never_increases_total_value():
    """Costs+tax can only reduce portfolio value; clamping never adds value."""
    st = BacktestState(
        assets_value={"AAA": 150.0, "BBB": 50.0},
        assets_cost_basis={"AAA": 150.0, "BBB": 50.0},  # no gains → no tax
        cash_balance=0.0,
        last_year=2020,
    )
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.25, "BBB": 0.75},
        slippage_rate=0.01, commission_fee=1.0,
        tax_settlement_mode="Immediate", tax_threshold=0.0,
    )
    total_after = sum(st.assets_value.values()) + st.cash_balance
    assert total_after <= 200.0
    assert st.cash_balance >= 0.0


def test_full_backtest_tax_drag_end_to_end(price_data):
    """End-to-end: Immediate-settlement run must underperform a no-tax run
    when a fully-invested book sells at a gain (signal flip forces rebalance)."""
    import engine as eng

    dates = price_data.index
    # Bull first half, bear second half → one sell-at-gain rebalance.
    signal = pd.Series([True] * (len(dates) // 2) + [False] * (len(dates) - len(dates) // 2), index=dates)

    def _run():
        return run_backtest(
            scenario_name="t",
            tickers=["AAA", "BBB"],
            weights=[0.5, 0.5],
            expenses=[0.0, 0.0],
            data=price_data,
            initial_capital=10_000.0,
            monthly_investment=0.0,
            inflation_rate=0.0,
            tax_threshold=0.0,
            strategy_mode="signal",
            slippage_rate=0.0,
            commission_fee=0.0,
            tax_settlement_mode="Immediate",
            signal_series=signal,
        )[1]

    mets_tax = _run()
    orig_rate = eng.TAX_RATE
    eng.TAX_RATE = 0.0
    try:
        mets_free = _run()
    finally:
        eng.TAX_RATE = orig_rate

    assert mets_tax["final_value"] < mets_free["final_value"]
    drag_pct = (
        (mets_free["final_value"] - mets_tax["final_value"])
        / mets_free["final_value"] * 100
    )
    # Before the bugfix this drag was ~0% for a fully-invested book.
    assert drag_pct > 0.1


def test_rebalance_tax_on_gain_with_cash_buffer():
    st = BacktestState(
        assets_value={"AAA": 200.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 100.0, "BBB": 0.0},
        cash_balance=200.0,
        last_year=2020,
    )
    # Sell half of AAA (basis 50 of 100) → gain 50, tax paid from cash
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.25, "BBB": 0.25},
        slippage_rate=0.0, commission_fee=0.0,
        tax_settlement_mode="Immediate", tax_threshold=0.0,
    )
    assert st.annual_realized_gain == pytest.approx(50.0)
    # equity 400 → targets 100 each; implied cash = 400-200 = 200; minus tax
    assert st.cash_balance == pytest.approx(200.0 - 50 * TAX_RATE)


def test_rebalance_annual_tax_defers_liability():
    st = BacktestState(
        assets_value={"AAA": 200.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 100.0, "BBB": 0.0},
        cash_balance=0.0,
    )
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5},
        slippage_rate=0.0, commission_fee=0.0,
        tax_settlement_mode="Annual", tax_threshold=0.0,
    )
    assert st.pending_tax_liability == pytest.approx(50 * TAX_RATE)
    # Fully invested → implied cash is 0; deferred tax stays as liability.
    assert st.cash_balance == 0


def test_rebalance_cash_never_negative():
    """Slippage + commission larger than residual cash must not go negative."""
    st = BacktestState(
        assets_value={"AAA": 10.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 10.0, "BBB": 0.0},
        cash_balance=0.0,
    )
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5},
        slippage_rate=0.05, commission_fee=5.0,
        tax_settlement_mode="Immediate",
    )
    assert st.cash_balance >= 0


def test_rebalance_buy_updates_basis():
    st = BacktestState(
        assets_value={"AAA": 50.0, "BBB": 50.0},
        assets_cost_basis={"AAA": 40.0, "BBB": 50.0},
        cash_balance=100.0,
    )
    before = st.assets_cost_basis.copy()
    _execute_rebalance(st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5}, 0.0, 0.0, "Immediate")
    assert st.assets_cost_basis["BBB"] > before["BBB"]
    assert st.cash_balance == pytest.approx(0.0, abs=1e-9)


def test_tax_threshold_respected():
    st = BacktestState(
        assets_value={"AAA": 300.0, "BBB": 0.0},
        assets_cost_basis={"AAA": 150.0, "BBB": 0.0},
        cash_balance=0.0,
    )
    # Sell half → gain 75, threshold 100 → nothing taxable yet
    _execute_rebalance(
        st, ["AAA", "BBB"], {"AAA": 0.5, "BBB": 0.5},
        0.0, 0.0, "Annual", tax_threshold=100.0,
    )
    assert st.pending_tax_liability == 0


# ── _handle_year_boundary ─────────────────────────────────────────────────────


def test_year_boundary_not_triggered_on_first_day():
    st = BacktestState(last_year=2020, pending_tax_liability=500.0,
                       current_monthly_inv=1000.0)
    _handle_year_boundary(st, pd.Timestamp("2020-01-02"), "Annual", 0.03)
    assert st.last_year == 2020
    assert st.pending_tax_liability == 500.0
    assert st.current_monthly_inv == 1000.0


def test_year_boundary_settles_and_inflates():
    st = BacktestState(last_year=2019, pending_tax_liability=500.0,
                       current_monthly_inv=1000.0, cash_balance=2000.0)
    _handle_year_boundary(st, pd.Timestamp("2020-01-02"), "Annual", 0.03)
    assert st.last_year == 2020
    assert st.pending_tax_liability == 0
    assert st.cash_balance == pytest.approx(1500.0)
    assert st.current_monthly_inv == pytest.approx(1030.0)
    assert st.annual_realized_gain == 0


# ── run_backtest ──────────────────────────────────────────────────────────────


def test_run_backtest_buy_and_hold_smoke(price_data):
    hist, mets = run_backtest(
        scenario_name="t",
        tickers=["AAA", "BBB"],
        weights=[0.6, 0.4],
        expenses=[0.0, 0.0],
        data=price_data,
        initial_capital=10_000,
        monthly_investment=0,
        inflation_rate=0.0,
        # Reference model below ignores taxes; raise the threshold so the
        # engine settles none either (immediate tax now reduces asset value).
        tax_threshold=1e12,
        strategy_mode="Monthly Rebalancing",
    )
    assert hist is not None and mets is not None
    assert len(hist) == len(price_data) - 1  # pct_change drops first row
    assert not np.isnan(mets["sharpe"])
    assert not np.isnan(mets["sortino"])
    assert "profit_factor" in mets
    assert mets["total_invested"] == 10_000
    # Replicate a monthly-rebalanced weighted portfolio independently.
    rets = price_data.pct_change()
    v = {"AAA": 6_000.0, "BBB": 4_000.0}
    prev_month = None
    for date in price_data.index[1:]:
        for t, w in (("AAA", 0.6), ("BBB", 0.4)):
            v[t] *= 1 + rets.at[date, t]
        if prev_month is not None and date.month != prev_month:
            total = sum(v.values())
            v = {"AAA": total * 0.6, "BBB": total * 0.4}
        prev_month = date.month
    assert hist.iloc[-1] == pytest.approx(sum(v.values()), rel=1e-9)


def test_run_backtest_with_signal_no_lookahead(price_data):
    """Signal shifted by 1 day: today's signal affects tomorrow's rebalance."""
    sig = pd.Series(True, index=price_data.index)
    sig.iloc[100:] = False
    hist, mets = run_backtest(
        scenario_name="t",
        tickers=["AAA"],
        weights=[1.0],
        expenses=[0.0],
        data=price_data[["AAA"]],
        initial_capital=10_000,
        monthly_investment=0,
        inflation_rate=0.0,
        tax_threshold=0,
        strategy_mode="Trend Following (SMA)",
        signal_series=sig,
        safe_assets=[],
        risk_off_invested_pct=0.0,
    )
    assert hist is not None
    # After the flip the portfolio should be fully in cash (no further market moves)
    tail = price_data.index[102:]
    base = hist.loc[tail[0]]
    for d in tail:
        assert hist.loc[d] == pytest.approx(base)


def test_run_backtest_missing_ticker_returns_none(price_data):
    hist, mets = run_backtest(
        scenario_name="t", tickers=["NOPE"], weights=[1.0], expenses=[0.0],
        data=price_data, initial_capital=100, monthly_investment=0,
        inflation_rate=0, tax_threshold=0, strategy_mode="Monthly Rebalancing",
    )
    assert hist is None and mets is None
