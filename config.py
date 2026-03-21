"""
Constants, defaults, and configuration for the backtest simulator.
"""

# --- Expense Ratios ---
DEFAULT_EXPENSE_RATIOS = {
    'TQQQ': 0.0086, 'QLD': 0.0095, 'USD': 0.0095, 'GLD': 0.0040,
    'SOXL': 0.0076, 'SPXL': 0.0091, 'UPRO': 0.0091, 'VOO': 0.0003,
    'VTI': 0.0003,
}

# --- Tax ---
TAX_RATE = 0.22

# --- Strategy Constants ---
STRAT_MONTHLY = "Monthly Rebalancing"
STRAT_TREND = "Trend Following (SMA)"
STRAT_CROSS = "SMA Crossover (Golden Cross)"
STRAT_VOL = "Volatility Targeting (VIX)"
STRAT_TRAIL = "Trailing Stop (High Water Mark)"
STRAT_RSI = "RSI Mean Reversion"
STRAT_MACD = "MACD Divergence"
STRAT_BBAND = "Bollinger Band Squeeze"

ALL_STRATEGIES = [
    STRAT_MONTHLY, STRAT_TREND, STRAT_CROSS,
    STRAT_VOL, STRAT_TRAIL, STRAT_RSI,
    STRAT_MACD, STRAT_BBAND,
]

STRATEGY_DESCRIPTIONS = {
    STRAT_MONTHLY: (
        "The **Buy & Hold** benchmark. Rebalances to target weights on the "
        "1st of every month regardless of price."
    ),
    STRAT_TREND: (
        "Risk-On only when Price > SMA (e.g., 200-day). Acts as a 'Circuit "
        "Breaker' for secular bear markets. **Pros:** Avoids major crashes. "
        "**Cons:** Can 'whipsaw' during sideways markets."
    ),
    STRAT_CROSS: (
        "A 'Golden Cross' strategy. Risk-On when Fast SMA (e.g., 50) > Slow "
        "SMA (e.g., 200). **Pros:** Filters minor noise. **Cons:** More lag; "
        "may miss recovery gains."
    ),
    STRAT_VOL: (
        "Exits the market when VIX spikes above your threshold. **Pros:** "
        "Proactively exits before volatility decay. **Cons:** Panics are "
        "often short-lived."
    ),
    STRAT_TRAIL: (
        "Exits if signal ticker drops X% from its recent peak. Re-enters "
        "when Price > SMA. **Pros:** Hard limit on capital loss. "
        "**Cons:** Requires tuned re-entry. Assumes **Invested** at start."
    ),
    STRAT_RSI: (
        "Goes Risk-Off when RSI(14) crosses above the overbought level or "
        "drops below the oversold level. Re-enters on mean reversion. "
        "**Pros:** Captures regime extremes. **Cons:** RSI can stay "
        "overbought for extended trends."
    ),
    STRAT_MACD: (
        "Risk-On when the MACD line crosses above its Signal line (bullish "
        "divergence). Risk-Off on bearish crossover. **Pros:** Captures "
        "momentum shifts early. **Cons:** Prone to whipsaws in ranging markets."
    ),
    STRAT_BBAND: (
        "Uses Bollinger Band squeeze to detect low-volatility regimes. "
        "Risk-On when price breaks above the upper band after a squeeze. "
        "Risk-Off when price drops below the lower band. **Pros:** Captures "
        "explosive breakouts. **Cons:** May enter late in trending markets."
    ),
}

# --- Initial Seed Scenarios ---
INITIAL_SCENARIOS = {
    "USD 50/TQQQ 35/GLD 15": (['GLD', 'TQQQ', 'USD'], [0.15, 0.35, 0.50]),
    "SOXL 50/TQQQ 35/GLD 15": (['GLD', 'TQQQ', 'SOXL'], [0.15, 0.35, 0.50]),
    "USD 30/QLD 55/GLD 15": (['GLD', 'QLD', 'USD'], [0.15, 0.55, 0.30]),
    "USD 50/QLD 35/GLD 15": (['GLD', 'QLD', 'USD'], [0.15, 0.35, 0.50]),
    "USD 85% + GLD 15%": (['GLD', 'USD'], [0.15, 0.85]),
    "SOXL 85% + GLD 15%": (['GLD', 'SOXL'], [0.15, 0.85]),
    "TQQQ 85% + GLD 15%": (['GLD', 'TQQQ'], [0.15, 0.85]),
    "SPXL 85% + GLD 15%": (['GLD', 'SPXL'], [0.15, 0.85]),
    "QLD 85% + GLD 15%": (['GLD', 'QLD'], [0.15, 0.85]),
    "UPRO 85% + GLD 15%": (['GLD', 'UPRO'], [0.15, 0.85]),
    "VOO 85% + GLD 15%": (['GLD', 'VOO'], [0.15, 0.85]),
    "VOO 100%": (['VOO'], [1.00]),
    "VTI 100%": (['VTI'], [1.00]),
    "SOXL 100%": (['SOXL'], [1.00]),
}
