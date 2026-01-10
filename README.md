# ETF Portfolio Backtester

Interactive dashboard for backtesting ETF portfolios with advanced tax and cost simulations.

## Usage

```bash
uv run streamlit run main.py
```

## Features

- **Portfolio Management**: Compare predefined and custom ETF portfolios with drag-and-drop organization.
- **Simulation Engine**:
    - Monthly rebalancing.
    - Inflation adjustment.
    - Transaction costs (Slippage and Commissions).
    - Tax Logic: Marginal tax rates with options for Immediate or Annual settlement.
- **Analysis**:
    - Interactive performance charts.
    - Metrics: CAGR, Sharpe, Sortino, Volatility, Calmar, Max Drawdown.
    - Detailed breakdown of final values and costs.