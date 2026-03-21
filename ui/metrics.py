"""
Metrics table — sortable numeric columns with formatted display,
plus highlight cards for top-ranked scenarios.
"""

import streamlit as st
import pandas as pd


def render_metrics_table(results: dict) -> None:
    """Render highlight stat cards and the performance metrics table.

    Args:
        results: {scenario_name: {"history": pd.Series, "metrics": dict}, …}
    """
    if not results:
        return

    st.subheader("Performance Metrics")

    # --- Highlight cards ---
    _render_highlight_cards(results)

    # --- Build DataFrame with raw numeric values ---
    rows: list[dict] = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Scenario": name,
            "Final Value": m["final_value"],
            "Profit": m["profit"],
            "CAGR": m["cagr"] * 100,       # Convert to pct for display
            "Sharpe": m["sharpe"],
            "Sortino": m["sortino"],
            "Volatility": m["volatility"] * 100,
            "Max DD": m["mdd"] * 100,
            "Calmar": m["calmar"],
            "Pain Idx": m["pain_index"] * 100,
            "Pain Ratio": m["pain_ratio"],
            "Win Rate": m["win_rate"] * 100,
            "Invested": m["total_invested"],
        })

    df = pd.DataFrame(rows)

    # Default sort by CAGR descending
    df = df.sort_values("CAGR", ascending=False).reset_index(drop=True)

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
            "Final Value": st.column_config.NumberColumn(
                "Final Value", format="$%,.0f",
            ),
            "Profit": st.column_config.NumberColumn(
                "Profit", format="$%,.0f",
            ),
            "CAGR": st.column_config.NumberColumn(
                "CAGR", format="%.2f%%",
                help="Compound Annual Growth Rate",
            ),
            "Sharpe": st.column_config.NumberColumn(
                "Sharpe", format="%.2f",
            ),
            "Sortino": st.column_config.NumberColumn(
                "Sortino", format="%.2f",
            ),
            "Volatility": st.column_config.NumberColumn(
                "Vol", format="%.2f%%",
            ),
            "Max DD": st.column_config.NumberColumn(
                "Max DD", format="%.2f%%",
                help="Maximum Drawdown",
            ),
            "Calmar": st.column_config.NumberColumn(
                "Calmar", format="%.2f",
            ),
            "Pain Idx": st.column_config.NumberColumn(
                "Pain Idx", format="%.2f%%",
            ),
            "Pain Ratio": st.column_config.NumberColumn(
                "Pain Ratio", format="%.2f",
            ),
            "Win Rate": st.column_config.NumberColumn(
                "Win Rate", format="%.1f%%",
                help="% of trading days with positive returns",
            ),
            "Invested": st.column_config.NumberColumn(
                "Invested", format="$%,.0f",
            ),
        },
    )


def _render_highlight_cards(results: dict) -> None:
    """Show top-ranked scenario highlight cards above the table."""
    if len(results) < 2:
        return

    best_cagr_name = max(results, key=lambda n: results[n]["metrics"]["cagr"])
    best_sharpe_name = max(results, key=lambda n: results[n]["metrics"]["sharpe"])
    lowest_mdd_name = max(results, key=lambda n: results[n]["metrics"]["mdd"])  # mdd is negative, max = least bad

    cols = st.columns(3)
    with cols[0]:
        cagr_val = results[best_cagr_name]["metrics"]["cagr"] * 100
        st.metric("🏆 Best CAGR", f"{cagr_val:.2f}%", best_cagr_name)
    with cols[1]:
        sharpe_val = results[best_sharpe_name]["metrics"]["sharpe"]
        st.metric("📊 Best Sharpe", f"{sharpe_val:.2f}", best_sharpe_name)
    with cols[2]:
        mdd_val = results[lowest_mdd_name]["metrics"]["mdd"] * 100
        st.metric("🛡️ Shallowest DD", f"{mdd_val:.2f}%", lowest_mdd_name)


def render_detailed_breakdown(start_date, end_date) -> None:
    """Render the detailed assumptions / methodology note."""
    with st.expander("ℹ️ Methodology & Assumptions"):
        st.markdown(
            f"""
            - **Rebalancing**: Monthly to target weights on the 1st trading day.
            - **Taxes**: Estimated at **22%** on realized gains exceeding the threshold.
            - **Expense ratios**: Applied daily (annual ratio / 252).
            - **Slippage & commission**: Deducted from cash on each rebalance.
            - **Data range**: `{start_date}` → `{end_date}`
            """
        )
