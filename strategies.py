"""
Signal generation for each trading strategy.
"""

import numpy as np
import pandas as pd


def generate_trend_signal(
    ts: pd.Series, sma_window: int = 200, use_dual_momentum: bool = False
) -> pd.Series:
    """Trend Following (SMA): Risk-On when Price > SMA."""
    sma = ts.rolling(window=sma_window).mean()
    bullish = ts > sma
    if use_dual_momentum:
        mom = ts > ts.shift(21)
        return bullish & mom
    return bullish


def generate_crossover_signal(
    ts: pd.Series, sma_fast: int = 50, sma_slow: int = 200
) -> pd.Series:
    """SMA Crossover (Golden Cross): Risk-On when Fast SMA > Slow SMA."""
    fast = ts.rolling(window=sma_fast).mean()
    slow = ts.rolling(window=sma_slow).mean()
    return fast > slow


def generate_volatility_signal(ts: pd.Series, threshold: float = 30.0) -> pd.Series:
    """Volatility Targeting: Risk-Off when VIX > threshold."""
    return ts < threshold


def generate_trailing_stop_signal(
    ts: pd.Series, stop_pct: float = 0.15, sma_window: int = 200
) -> pd.Series:
    """Trailing Stop: Exit on X% drawdown from peak, re-enter when Price > SMA."""
    sma = ts.rolling(window=sma_window).mean()
    current_state = True
    current_hwm = ts.iloc[0]
    signal_list: list[bool] = []

    for date, price in ts.items():
        try:
            sma_val = sma.at[date]
        except Exception:
            sma_val = np.nan

        if current_state:  # Invested
            if price > current_hwm:
                current_hwm = price
            dd = (price - current_hwm) / current_hwm if current_hwm > 0 else 0
            if dd < -stop_pct:
                current_state = False  # Stop out
        else:  # In cash — check re-entry
            if not np.isnan(sma_val) and price > sma_val:
                current_state = True
                current_hwm = price  # Reset HWM on re-entry

        signal_list.append(current_state)

    return pd.Series(signal_list, index=ts.index)


def generate_rsi_signal(
    ts: pd.Series,
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> pd.Series:
    """RSI Mean Reversion: Risk-Off when RSI > overbought or < oversold.

    Uses a state machine:
    - When Risk-On:  exit if RSI > overbought
    - When Risk-Off: re-enter if RSI crosses back below overbought (from above)
                     OR crosses back above oversold (from below)
    """
    delta = ts.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # State machine for cleaner signals (avoid excessive whipsaw)
    is_invested = True
    signals: list[bool] = []

    for val in rsi:
        if np.isnan(val):
            signals.append(is_invested)
            continue

        if is_invested:
            # Exit when overbought
            if val > overbought:
                is_invested = False
        else:
            # Re-enter when RSI pulls back into normal range
            if val < overbought and val > oversold:
                is_invested = True

        signals.append(is_invested)

    return pd.Series(signals, index=ts.index)


def generate_macd_signal(
    ts: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """MACD Divergence: Risk-On when MACD line > Signal line.

    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal_period)
    Risk-On on bullish crossover, Risk-Off on bearish crossover.
    """
    ema_fast = ts.ewm(span=fast_period, adjust=False).mean()
    ema_slow = ts.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # State machine to avoid excessive switching
    is_invested = True
    signals: list[bool] = []

    for macd_val, sig_val in zip(macd_line, signal_line):
        if np.isnan(macd_val) or np.isnan(sig_val):
            signals.append(is_invested)
            continue

        if is_invested:
            if macd_val < sig_val:
                is_invested = False
        else:
            if macd_val > sig_val:
                is_invested = True

        signals.append(is_invested)

    return pd.Series(signals, index=ts.index)


def generate_bband_signal(
    ts: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
    squeeze_threshold: float = 0.04,
) -> pd.Series:
    """Bollinger Band Squeeze: detects low-volatility regimes for breakout entries.

    Bandwidth = (Upper - Lower) / Mid
    - When bandwidth < squeeze_threshold → a 'squeeze' is detected.
    - Risk-On when price breaks above the upper band after a squeeze.
    - Risk-Off when price drops below the lower band.
    """
    mid = ts.rolling(window=period).mean()
    std = ts.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid

    is_invested = True
    in_squeeze = False
    signals: list[bool] = []

    for i in range(len(ts)):
        price = ts.iloc[i]
        bw = bandwidth.iloc[i]
        ub = upper.iloc[i]
        lb = lower.iloc[i]

        if np.isnan(bw) or np.isnan(ub) or np.isnan(lb):
            signals.append(is_invested)
            continue

        # Track squeeze state
        if bw < squeeze_threshold:
            in_squeeze = True

        if is_invested:
            # Exit when price drops below lower band
            if price < lb:
                is_invested = False
                in_squeeze = False
        else:
            # Re-enter after squeeze when price breaks above upper band
            if in_squeeze and price > ub:
                is_invested = True
                in_squeeze = False
            # Also re-enter if price reclaims mid band (non-squeeze recovery)
            elif not in_squeeze and price > mid.iloc[i]:
                is_invested = True

        signals.append(is_invested)

    return pd.Series(signals, index=ts.index)
