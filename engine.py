"""
Core backtest engine — simulation loop and metrics calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import STRAT_MONTHLY, TAX_RATE


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

@dataclass
class BacktestState:
    """Mutable state carried across every day of the simulation loop."""

    assets_value: dict[str, float] = field(default_factory=dict)
    assets_cost_basis: dict[str, float] = field(default_factory=dict)
    cash_balance: float = 0.0
    annual_realized_gain: float = 0.0
    pending_tax_liability: float = 0.0
    current_monthly_inv: float = 0.0
    total_invested: float = 0.0
    last_month: int | None = None
    last_year: int = 0
    prev_signal_bull: bool = True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    history_series: pd.Series,
    daily_returns: pd.Series,
    total_invested: float,
    start_date,
    end_date,
) -> dict:
    """Compute performance metrics from a portfolio value series."""
    if history_series.empty:
        return {}

    start_val = history_series.iloc[0]
    end_val = history_series.iloc[-1]

    std = daily_returns.std()
    sharpe = (
        (daily_returns.mean() / std) * np.sqrt(252)
        if pd.notna(std) and std != 0
        else 0
    )

    neg_returns = daily_returns[daily_returns < 0]
    neg_std = neg_returns.std()
    # Guard: an empty or constant negative-return series gives NaN, not 0.
    sortino = (
        (daily_returns.mean() / neg_std) * np.sqrt(252)
        if pd.notna(neg_std) and neg_std != 0
        else 0
    )

    volatility = (std * np.sqrt(252)) if pd.notna(std) else 0.0

    roll_max = history_series.cummax()
    drawdown = (history_series - roll_max) / roll_max
    mdd = drawdown.min()

    years = (end_date - start_date).days / 365.25
    if years <= 0:
        years = 1e-6

    total_return_factor = end_val / start_val
    cagr = (total_return_factor ** (1 / years)) - 1

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    pain_index = drawdown.abs().mean()
    pain_ratio = cagr / pain_index if pain_index != 0 else 0

    win_rate = (
        (daily_returns > 0).sum() / len(daily_returns)
        if len(daily_returns) > 0
        else 0
    )

    # Profit factor: gross wins / gross losses (inf when no losing days).
    gross_wins = daily_returns[daily_returns > 0].sum()
    gross_losses = -daily_returns[daily_returns < 0].sum()
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    profit = end_val - total_invested
    roi = profit / total_invested if total_invested != 0 else 0.0

    return {
        "final_value": end_val,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "mdd": mdd,
        "cagr": cagr,
        "calmar": calmar,
        "pain_index": pain_index,
        "pain_ratio": pain_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "profit": profit,
        "roi": roi,
        "total_invested": total_invested,
    }


# ---------------------------------------------------------------------------
# Helper: daily market returns & expense drag
# ---------------------------------------------------------------------------

def _apply_market_returns(
    state: BacktestState,
    daily_ret: pd.Series,
    tickers: list[str],
    expense_map: dict[str, float],
) -> None:
    """Apply one day of market returns and expense-ratio drag in-place."""
    for ticker in tickers:
        state.assets_value[ticker] *= 1 + daily_ret[ticker]
        state.assets_value[ticker] *= 1 - (expense_map.get(ticker, 0.0) / 252)


# ---------------------------------------------------------------------------
# Helper: rebalance execution
# ---------------------------------------------------------------------------

def _execute_rebalance(
    state: BacktestState,
    tickers: list[str],
    target_weights: dict[str, float],
    slippage_rate: float,
    commission_fee: float,
    tax_settlement_mode: str,
    tax_threshold: float = 0.0,
) -> None:
    """Sell / buy to match *target_weights* and settle transaction costs."""
    current_equity = sum(state.assets_value.values()) + state.cash_balance
    target_values = {
        t: current_equity * target_weights.get(t, 0.0) for t in tickers
    }

    total_trans_cost = 0.0
    total_tax_paid_now = 0.0

    for ticker in tickers:
        target_val = target_values[ticker]
        current_val = state.assets_value[ticker]
        trade_diff = target_val - current_val

        # Transaction costs
        if abs(trade_diff) > 0.01:
            total_trans_cost += abs(trade_diff) * slippage_rate + commission_fee

        # Tax on sells
        if trade_diff < 0:
            sell_amount = abs(trade_diff)
            avg_cost_basis = state.assets_cost_basis[ticker]
            pct_sold = sell_amount / current_val if current_val > 0 else 0
            basis_sold = avg_cost_basis * pct_sold

            realized_gain = sell_amount - basis_sold
            state.assets_cost_basis[ticker] -= basis_sold

            if realized_gain > 0:
                prev_taxable = max(0, state.annual_realized_gain - tax_threshold)
                state.annual_realized_gain += realized_gain
                new_taxable = max(0, state.annual_realized_gain - tax_threshold)
                taxable_now = new_taxable - prev_taxable

                if taxable_now > 0:
                    tax_on_trade = taxable_now * TAX_RATE
                    if tax_settlement_mode == "Immediate":
                        total_tax_paid_now += tax_on_trade
                    else:
                        state.pending_tax_liability += tax_on_trade
            else:
                state.annual_realized_gain += realized_gain

        elif trade_diff > 0:
            state.assets_cost_basis[ticker] += trade_diff

        state.assets_value[ticker] = target_val

    # Settle costs. Transaction costs and immediate taxes are deducted from
    # cash AFTER asset values are set to targets. When implied cash can't
    # cover them, settle the shortfall proportionally out of asset values
    # (scale every holding down) — otherwise a fully-invested portfolio
    # (implied_cash == 0) would silently erase the entire tax liability.
    allocated = sum(state.assets_value.values())
    implied_cash = current_equity - allocated
    shortfall = total_trans_cost + total_tax_paid_now - max(0.0, implied_cash)
    if shortfall > 0 and allocated > 0:
        scale = (allocated - shortfall) / allocated
        for ticker in tickers:
            state.assets_value[ticker] *= scale

    # Final safety net only: never let cash go negative (no invented leverage),
    # and never let scaling push total value up.
    state.cash_balance = max(
        0.0, implied_cash - total_trans_cost - total_tax_paid_now
    )
    new_total = sum(state.assets_value.values()) + state.cash_balance
    if new_total > current_equity:
        excess = new_total - current_equity
        state.cash_balance = max(0.0, state.cash_balance - excess)


# ---------------------------------------------------------------------------
# Helper: year-boundary handling
# ---------------------------------------------------------------------------

def _handle_year_boundary(
    state: BacktestState,
    date: pd.Timestamp,
    tax_settlement_mode: str,
    inflation_rate: float,
) -> None:
    """Settle annual taxes, apply inflation to monthly contribution.

    Triggered on the FIRST trading day of each new calendar year (not the
    last trading day of the old one). ``last_year`` is initialised to the
    first year of the backtest, so day one is never misread as a boundary —
    no spurious tax settlement or inflation compounding on day 1.
    """
    if date.year == state.last_year:
        return

    if tax_settlement_mode == "Annual" and state.pending_tax_liability > 0:
        state.cash_balance -= state.pending_tax_liability
        state.pending_tax_liability = 0.0

    state.current_monthly_inv *= 1 + inflation_rate
    state.annual_realized_gain = 0.0
    state.last_year = date.year


# ---------------------------------------------------------------------------
# Weight computation (unchanged logic, module-level function)
# ---------------------------------------------------------------------------

def _compute_current_weights(
    strategy_mode: str,
    effective_signal: pd.Series | None,
    current_signal_bull: bool,
    weight_map: dict[str, float],
    tickers: list[str],
    safe_assets: list[str] | None,
    risk_off_invested_pct: float,
) -> dict[str, float]:
    """Compute target weights for a rebalance event."""
    if strategy_mode == STRAT_MONTHLY or effective_signal is None:
        return weight_map.copy()

    if current_signal_bull:
        return weight_map.copy()

    # Risk-Off allocation
    risk_tickers = [t for t in tickers if safe_assets is None or t not in safe_assets]
    safe_tickers = [t for t in tickers if safe_assets is not None and t in safe_assets]

    total_risk_w = sum(weight_map.get(t, 0.0) for t in risk_tickers)
    total_safe_w = sum(weight_map.get(t, 0.0) for t in safe_tickers)

    vacated = total_risk_w * (1.0 - risk_off_invested_pct)

    current_weights: dict[str, float] = {}
    for t in tickers:
        if t in safe_tickers:
            share = weight_map[t] / total_safe_w if total_safe_w > 0 else 0.0
            current_weights[t] = weight_map[t] + vacated * share
        else:
            current_weights[t] = weight_map[t] * risk_off_invested_pct

    return current_weights


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    scenario_name: str,
    tickers: list[str],
    weights: list[float],
    expenses: list[float],
    data: pd.DataFrame,
    initial_capital: float,
    monthly_investment: float,
    inflation_rate: float,
    tax_threshold: float,
    strategy_mode: str,
    slippage_rate: float = 0.0,
    commission_fee: float = 0.0,
    tax_settlement_mode: str = "Immediate",
    signal_series: pd.Series | None = None,
    safe_assets: list[str] | None = None,
    risk_off_invested_pct: float = 0.0,
) -> tuple[pd.Series | None, dict | None]:
    """Run a full backtest simulation for one scenario.

    Returns ``(history_series, metrics_dict)`` or ``(None, None)`` on failure.
    """
    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------
    valid_tickers = [t for t in tickers if t in data.columns]
    if len(valid_tickers) != len(tickers):
        return None, None

    current_data = data[tickers].dropna()
    if current_data.empty:
        return None, None

    returns = current_data.pct_change().dropna()
    if returns.empty:
        return None, None

    weight_map = dict(zip(tickers, weights))
    expense_map = dict(zip(tickers, expenses))

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    state = BacktestState(
        assets_value={t: initial_capital * weight_map[t] for t in tickers},
        assets_cost_basis={t: initial_capital * weight_map[t] for t in tickers},
        cash_balance=0.0,
        annual_realized_gain=0.0,
        pending_tax_liability=0.0,
        current_monthly_inv=monthly_investment,
        total_invested=initial_capital,
        last_month=None,
        last_year=returns.index[0].year,
        prev_signal_bull=True,
    )

    values_history: list[float] = []
    daily_strategy_returns: list[float] = []

    # ------------------------------------------------------------------
    # Signal preparation — shift by 1 day, then align to returns index
    # ------------------------------------------------------------------
    effective_signal: pd.Series | None = (
        signal_series.shift(1, fill_value=True) if signal_series is not None else None
    )

    if effective_signal is not None:
        sig = effective_signal.reindex(returns.index, method="ffill").fillna(True)
        # Coerce to a strict bool so downstream comparisons are reliable even
        # when callers pass object/float series (0/1, np.bool_, etc.).
        # NOTE: fillna(True) MUST precede astype(bool) — astype maps NaN to
        # True, so filling first is the only way NaN becomes False-safe here.
        sig = sig.astype(bool)
        state.prev_signal_bull = bool(sig.iloc[0])
        effective_signal = sig

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    for date in returns.index:
        # --- Year boundary ---
        _handle_year_boundary(state, date, tax_settlement_mode, inflation_rate)

        prev_total = sum(state.assets_value.values()) + state.cash_balance

        # --- Market movement & expenses ---
        try:
            daily_ret = returns.loc[date]
        except KeyError:
            continue

        _apply_market_returns(state, daily_ret, tickers, expense_map)

        post_market_total = sum(state.assets_value.values()) + state.cash_balance
        strat_ret = (
            (post_market_total - prev_total) / prev_total if prev_total != 0 else 0
        )
        daily_strategy_returns.append(strat_ret)

        rebalance_needed = False

        # --- Signal check (indices already aligned — direct lookup) ---
        current_signal_bull = True
        if effective_signal is not None:
            current_signal_bull = effective_signal.loc[date]
            if current_signal_bull != state.prev_signal_bull:
                rebalance_needed = True

        # --- Monthly trigger ---
        is_monthly_trigger = (
            state.last_month is not None and date.month != state.last_month
        )

        if is_monthly_trigger:
            rebalance_needed = True
            state.total_invested += state.current_monthly_inv
            state.cash_balance += state.current_monthly_inv

        if rebalance_needed:
            current_weights = _compute_current_weights(
                strategy_mode,
                effective_signal,
                current_signal_bull,
                weight_map,
                tickers,
                safe_assets,
                risk_off_invested_pct,
            )
            _execute_rebalance(
                state, tickers, current_weights,
                slippage_rate, commission_fee, tax_settlement_mode,
                tax_threshold,
            )
            state.prev_signal_bull = current_signal_bull

        state.last_month = date.month
        values_history.append(
            sum(state.assets_value.values())
            + state.cash_balance
            - state.pending_tax_liability
        )

    # ------------------------------------------------------------------
    # Build result series & metrics
    # ------------------------------------------------------------------
    history_series = pd.Series(values_history, index=returns.index)
    strat_ret_series = pd.Series(daily_strategy_returns, index=returns.index)

    metrics = calculate_metrics(
        history_series,
        strat_ret_series,
        state.total_invested,
        returns.index[0],
        returns.index[-1],
    )
    return history_series, metrics
