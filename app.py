"""
Portfolio Backtester — main application orchestrator.

Ties together the sidebar, data layer, engine, and visualisation modules.
"""

import streamlit as st

from config import (
    STRAT_CROSS, STRAT_MONTHLY, STRAT_RSI, STRAT_TRAIL, STRAT_TREND, STRAT_VOL,
    STRAT_MACD, STRAT_BBAND,
)
from data import get_stock_data
from engine import run_backtest
from scenarios import load_scenarios
from strategies import (
    generate_crossover_signal,
    generate_rsi_signal,
    generate_trailing_stop_signal,
    generate_trend_signal,
    generate_volatility_signal,
    generate_macd_signal,
    generate_bband_signal,
)
from ui.charts import render_charts
from ui.metrics import render_detailed_breakdown, render_metrics_table
from ui.sidebar import render_scenario_manager, render_settings
from ui.views import render_analysis_views


def run() -> None:
    """Application entry point."""

    # --- Page config ---
    st.set_page_config(
        page_title="Portfolio Backtester",
        page_icon="📈",
        layout="wide",
    )

    # --- Custom CSS ---
    _inject_custom_css()

    # --- Session state init ---
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = load_scenarios()

    # --- Sidebar ---
    params = render_settings()
    selected_scenarios = render_scenario_manager()

    # --- Main Area ---
    st.title("ETF Portfolio Backtester")

    if not selected_scenarios:
        st.info("Select at least one scenario from the sidebar to begin.")
        return

    # Collect all tickers needed — sorted for deterministic cache keys
    all_tickers: set[str] = set()
    strategy_mode = params["strategy_mode"]

    if strategy_mode != STRAT_MONTHLY:
        all_tickers.add(params["signal_ticker"])

    for scen in selected_scenarios:
        all_tickers.update(scen["tickers"])

    sorted_tickers = sorted(all_tickers)

    # --- Download data ---
    with st.spinner("Downloading market data…"):
        full_data = get_stock_data(sorted_tickers)

    if full_data.empty:
        st.error("No data found for the selected tickers.")
        return

    # --- Date range filter ---
    min_date = full_data.index.min().date()
    max_date = full_data.index.max().date()

    st.subheader("Analysis Period")
    date_range = st.slider(
        "Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
    )
    start_date, end_date = date_range

    filtered_data = full_data[
        (full_data.index.date >= start_date) & (full_data.index.date <= end_date)
    ]

    if filtered_data.empty:
        st.warning("No data available for the selected date range.")
        return

    # --- Signal generation ---
    signal_series = _generate_signal(filtered_data, params)

    # --- Run backtests ---
    results: dict = {}
    progress = st.progress(0, text="Running backtests…")

    for i, scen in enumerate(selected_scenarios):
        hist, mets = run_backtest(
            scenario_name=scen["name"],
            tickers=scen["tickers"],
            weights=scen["weights"],
            expenses=scen["expenses"],
            data=filtered_data,
            initial_capital=params["initial_cash"],
            monthly_investment=params["monthly_cash"],
            inflation_rate=params["inflation_rate"],
            tax_threshold=params["tax_threshold"],
            strategy_mode=strategy_mode,
            slippage_rate=params["slippage_rate"],
            commission_fee=params["commission_fee"],
            tax_settlement_mode=params["tax_settlement_mode"],
            signal_series=signal_series,
            safe_assets=params["safe_assets"],
            risk_off_invested_pct=params["risk_off_invested_pct"],
        )
        if hist is not None:
            results[scen["name"]] = {"history": hist, "metrics": mets}

        progress.progress(
            (i + 1) / len(selected_scenarios),
            text=f"Completed {i + 1}/{len(selected_scenarios)}…",
        )

    progress.empty()

    if not results:
        st.warning("Not enough data to run backtest for the selected scenarios.")
        return

    # --- Summary stat cards ---
    _render_summary_cards(results)

    # --- Tabbed results ---
    tab_charts, tab_metrics, tab_analysis = st.tabs([
        "📈 Charts", "📊 Metrics", "🔬 Analysis",
    ])

    with tab_charts:
        render_charts(results)

    with tab_metrics:
        render_metrics_table(results)
        render_detailed_breakdown(start_date, end_date)

    with tab_analysis:
        render_analysis_views(results)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _generate_signal(data, params) -> "None | object":
    """Generate a signal series based on the selected strategy."""
    strategy = params["strategy_mode"]

    if strategy == STRAT_MONTHLY:
        return None

    signal_ticker = params["signal_ticker"]
    if signal_ticker not in data.columns:
        st.warning(f"Signal ticker **{signal_ticker}** not found in data.")
        return None

    ts = data[signal_ticker]

    if strategy == STRAT_TREND:
        return generate_trend_signal(
            ts, params["sma_window"], params["use_dual_momentum"],
        )
    elif strategy == STRAT_CROSS:
        return generate_crossover_signal(
            ts, params["sma_fast"], params["sma_slow"],
        )
    elif strategy == STRAT_VOL:
        return generate_volatility_signal(ts, params["vix_threshold"])
    elif strategy == STRAT_TRAIL:
        return generate_trailing_stop_signal(
            ts, params["trailing_stop_pct"], params["sma_window"],
        )
    elif strategy == STRAT_RSI:
        return generate_rsi_signal(
            ts, params["rsi_period"],
            params["rsi_overbought"], params["rsi_oversold"],
        )
    elif strategy == STRAT_MACD:
        return generate_macd_signal(
            ts, params["macd_fast"],
            params["macd_slow"], params["macd_signal"],
        )
    elif strategy == STRAT_BBAND:
        return generate_bband_signal(
            ts, params["bb_period"],
            params["bb_std"], params["bb_squeeze"],
        )

    return None


def _render_summary_cards(results: dict) -> None:
    """Top-level summary stat cards above the tabbed results."""
    if not results:
        return

    cagrs = {n: r["metrics"]["cagr"] for n, r in results.items()}
    sharpes = {n: r["metrics"]["sharpe"] for n, r in results.items()}
    finals = {n: r["metrics"]["final_value"] for n, r in results.items()}
    rois = {n: r["metrics"].get("roi", 0) for n, r in results.items()}

    best_cagr = max(cagrs, key=cagrs.get)
    worst_cagr = min(cagrs, key=cagrs.get)
    best_sharpe = max(sharpes, key=sharpes.get)
    avg_sharpe = sum(sharpes.values()) / len(sharpes) if sharpes else 0

    cols = st.columns(5)
    with cols[0]:
        st.metric(
            "Scenarios",
            len(results),
            help="Number of active scenarios in this run",
        )
    with cols[1]:
        st.metric(
            "🏆 Best CAGR",
            f"{cagrs[best_cagr] * 100:.1f}%",
            best_cagr,
        )
    with cols[2]:
        st.metric(
            "📉 Worst CAGR",
            f"{cagrs[worst_cagr] * 100:.1f}%",
            worst_cagr,
        )
    with cols[3]:
        st.metric(
            "⚡ Best Sharpe",
            f"{sharpes[best_sharpe]:.2f}",
            best_sharpe,
        )
    with cols[4]:
        st.metric(
            "Avg Sharpe",
            f"{avg_sharpe:.2f}",
            help="Average Sharpe ratio across all scenarios",
        )


def _inject_custom_css() -> None:
    """Inject CSS for visual polish — metric cards, tabs, dialogs."""
    st.markdown("""
    <style>
    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg,
            rgba(0, 173, 181, 0.08) 0%,
            rgba(57, 62, 70, 0.3) 100%);
        border: 1px solid rgba(0, 173, 181, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 173, 181, 0.15);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem;
        color: rgba(238, 238, 238, 0.7);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00E5FF;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.75rem;
        color: rgba(238, 238, 238, 0.5);
    }

    /* Tabs styling */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 600;
        padding: 8px 20px;
    }

    /* Sidebar scenario checkboxes */
    section[data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.85rem;
    }

    /* Data editor / dataframe improvements */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Dialog styling improvements */
    div[data-testid="stModal"] {
        backdrop-filter: blur(8px);
    }
    div[data-testid="stModal"] > div {
        border-radius: 16px;
        border: 1px solid rgba(0, 173, 181, 0.2);
    }

    /* Smooth transitions on interactive elements */
    button, .stButton > button {
        transition: all 0.15s ease;
    }
    </style>
    """, unsafe_allow_html=True)
