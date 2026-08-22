# ETF Portfolio Backtester

Interactive Streamlit dashboard for backtesting ETF portfolios with realistic tax, cost, and strategy simulations.

## Usage

```bash
uv run streamlit run main.py
```

## Features

**Scenarios** -- Create, clone, import/export portfolio scenarios. Each scenario defines a set of tickers with target weights and expense ratios. Select scenarios via checkbox grid with bulk actions (All / None / Invert).

**Strategies** -- 8 built-in signal strategies:
- Monthly Rebalancing (buy-and-hold benchmark)
- Trend Following (SMA)
- SMA Crossover (Golden Cross)
- Volatility Targeting (VIX)
- Trailing Stop (High Water Mark)
- RSI Mean Reversion
- MACD Divergence
- Bollinger Band Squeeze

**Simulation Engine** -- Monthly rebalancing, inflation-adjusted contributions, transaction costs (slippage + commission), and tax logic with immediate or annual settlement modes.

**Analysis Views** -- Results are organized into three tabs:
- **Charts** -- Portfolio performance and drawdown over time, with log-scale and normalized toggles.
- **Metrics** -- Sortable table with CAGR, Sharpe, Sortino, Volatility, Max Drawdown, Calmar, Pain Ratio, Win Rate, Profit Factor. Default sorted by CAGR.
- **Analysis** -- Correlation heatmap, rolling returns (1yr / 3yr / 5yr), and annual returns bar chart.

## Data & Signal Notes

- Price data is forward-filled across per-ticker holidays, and only leading rows (before any ticker has listed) are dropped — a short-history ETF no longer truncates the whole dataset.
- Signals are generated on the **full** downloaded history and then sliced to the selected date range, so rolling windows (e.g. SMA 200) are properly warmed up at the start of the range instead of silently reading as "invested".
- Signals are shifted by one day before use: a flip rebalances on the next trading day (no look-ahead).

## Development

```bash
uv sync              # installs runtime + dev deps
uv run pytest -v     # offline test suite (no network required)
```

## Screenshots

![page_image_1](./docs/page_image_1.png)
![page_image_2](./docs/page_image_2.png)
![page_image_3](./docs/page_image_3.png)
![page_image_4](./docs/page_image_4.png)
![page_image_5](./docs/page_image_5.png)