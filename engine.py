"""
Core backtest engine — simulation loop and metrics calculation.
"""

import numpy as np
import pandas as pd

from config import TAX_RATE


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

    end_val = history_series.iloc[-1]

    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() != 0
        else 0
    )

    neg_returns = daily_returns[daily_returns < 0]
    sortino = (
        (daily_returns.mean() / neg_returns.std()) * np.sqrt(252)
        if neg_returns.std() != 0
        else 0
    )

    volatility = daily_returns.std() * np.sqrt(252)

    roll_max = history_series.cummax()
    drawdown = (history_series - roll_max) / roll_max
    mdd = drawdown.min()

    years = (end_date - start_date).days / 365.25
    if years <= 0:
        years = 1e-6

    total_return_factor = end_val / total_invested
    cagr = (total_return_factor ** (1 / years)) - 1

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    pain_index = drawdown.abs().mean()
    pain_ratio = cagr / pain_index if pain_index != 0 else 0

    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0

    return {
        'final_value': end_val,
        'sharpe': sharpe,
        'sortino': sortino,
        'volatility': volatility,
        'mdd': mdd,
        'cagr': cagr,
        'calmar': calmar,
        'pain_index': pain_index,
        'pain_ratio': pain_ratio,
        'win_rate': win_rate,
        'profit': end_val - total_invested,
        'total_invested': total_invested,
    }


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

    Returns (history_series, metrics_dict) or (None, None) on failure.
    """
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

    # Initialize assets and cash
    assets_value = {t: initial_capital * weight_map[t] for t in tickers}
    assets_cost_basis = {t: initial_capital * weight_map[t] for t in tickers}
    cash_balance = 0.0

    values_history: list[float] = []
    daily_strategy_returns: list[float] = []

    last_month = None
    last_year = returns.index[0].year
    annual_realized_gain = 0.0
    pending_tax_liability = 0.0

    current_monthly_inv = monthly_investment
    total_invested = initial_capital

    # Shift signal by 1 day — trade on PREVIOUS day's close (no lookahead)
    effective_signal = (
        signal_series.shift(1).fillna(True) if signal_series is not None else None
    )

    prev_signal_bull = True
    if effective_signal is not None:
        try:
            prev_signal_bull = effective_signal.iloc[0]
        except Exception:
            prev_signal_bull = True

    for date in returns.index:
        # --- Year boundary: inflation + annual tax settlement ---
        if date.year != last_year:
            if tax_settlement_mode == "Annual" and pending_tax_liability > 0:
                cash_balance -= pending_tax_liability
                pending_tax_liability = 0.0
            current_monthly_inv *= (1 + inflation_rate)
            annual_realized_gain = 0.0
            last_year = date.year

        prev_total = sum(assets_value.values()) + cash_balance

        # --- Market movement & expenses ---
        try:
            daily_ret = returns.loc[date]
        except KeyError:
            continue

        for ticker in tickers:
            assets_value[ticker] *= (1 + daily_ret[ticker])
            assets_value[ticker] *= (1 - (expense_map.get(ticker, 0.0) / 252))

        post_market_total = sum(assets_value.values()) + cash_balance
        strat_ret = (
            (post_market_total - prev_total) / prev_total if prev_total != 0 else 0
        )
        daily_strategy_returns.append(strat_ret)

        rebalance_needed = False

        # --- Signal check ---
        current_signal_bull = True
        if effective_signal is not None:
            try:
                current_signal_bull = effective_signal.loc[date]
                if current_signal_bull != prev_signal_bull:
                    rebalance_needed = True
            except KeyError:
                current_signal_bull = prev_signal_bull

        # --- Monthly trigger ---
        is_monthly_trigger = last_month is not None and date.month != last_month

        if is_monthly_trigger:
            rebalance_needed = True
            total_invested += current_monthly_inv
            cash_balance += current_monthly_inv
            post_market_total += current_monthly_inv

        if rebalance_needed:
            # Determine target allocations
            current_weights = _compute_current_weights(
                strategy_mode, effective_signal, current_signal_bull,
                weight_map, tickers, safe_assets, risk_off_invested_pct,
            )

            current_equity = sum(assets_value.values()) + cash_balance
            target_values = {
                t: current_equity * current_weights.get(t, 0.0) for t in tickers
            }

            # Execute trades
            total_trans_cost = 0.0
            total_tax_paid_now = 0.0

            for ticker in tickers:
                target_val = target_values[ticker]
                current_val = assets_value[ticker]
                trade_diff = target_val - current_val

                # Transaction costs
                if abs(trade_diff) > 0.01:
                    total_trans_cost += abs(trade_diff) * slippage_rate + commission_fee

                # Tax on sells
                if trade_diff < 0:
                    sell_amount = abs(trade_diff)
                    avg_cost_basis = assets_cost_basis[ticker]
                    pct_sold = sell_amount / current_val if current_val > 0 else 0
                    basis_sold = avg_cost_basis * pct_sold

                    realized_gain = sell_amount - basis_sold
                    assets_cost_basis[ticker] -= basis_sold

                    if realized_gain > 0:
                        prev_taxable = max(0, annual_realized_gain - tax_threshold)
                        annual_realized_gain += realized_gain
                        new_taxable = max(0, annual_realized_gain - tax_threshold)
                        taxable_now = new_taxable - prev_taxable

                        if taxable_now > 0:
                            tax_on_trade = taxable_now * TAX_RATE
                            if tax_settlement_mode == "Immediate":
                                total_tax_paid_now += tax_on_trade
                            else:
                                pending_tax_liability += tax_on_trade
                    else:
                        annual_realized_gain += realized_gain

                elif trade_diff > 0:
                    assets_cost_basis[ticker] += trade_diff

                assets_value[ticker] = target_val

            # Settle costs
            allocated = sum(assets_value.values())
            implied_cash = current_equity - allocated
            cash_balance = implied_cash - total_trans_cost - total_tax_paid_now

            prev_signal_bull = current_signal_bull

        last_month = date.month
        values_history.append(
            sum(assets_value.values()) + cash_balance - pending_tax_liability
        )

    history_series = pd.Series(values_history, index=returns.index)
    strat_ret_series = pd.Series(daily_strategy_returns, index=returns.index)

    metrics = calculate_metrics(
        history_series, strat_ret_series, total_invested,
        returns.index[0], returns.index[-1],
    )
    return history_series, metrics


def _compute_current_weights(
    strategy_mode, effective_signal, current_signal_bull,
    weight_map, tickers, safe_assets, risk_off_invested_pct,
) -> dict:
    """Compute target weights for a rebalance event."""
    from config import STRAT_MONTHLY

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

    current_weights = {}
    for t in tickers:
        if t in safe_tickers:
            share = weight_map[t] / total_safe_w if total_safe_w > 0 else 0.0
            current_weights[t] = weight_map[t] + vacated * share
        else:
            current_weights[t] = weight_map[t] * risk_off_invested_pct

    return current_weights
