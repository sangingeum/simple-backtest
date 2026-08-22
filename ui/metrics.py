"""
Metrics table — sortable performance columns with ROI & Total Return,
plus premium highlight cards for top-ranked scenarios.

Theme: primary=#00ADB5, bg=#222831, secondary=#393E46, text=#EEEEEE
"""

import streamlit as st
import pandas as pd

# ── Theme tokens ──────────────────────────────────────────────────────────────
_PRIMARY = "#00ADB5"
_BG = "#222831"
_SECONDARY = "#393E46"
_TEXT = "#EEEEEE"


def render_metrics_table(results: dict) -> None:
    """Render highlight stat cards and the performance metrics table.

    Args:
        results: {scenario_name: {"history": pd.Series, "metrics": dict}, …}
    """
    if not results:
        return

    st.markdown(
        f'<h3 style="color:{_PRIMARY};">📊 Performance Metrics</h3>',
        unsafe_allow_html=True,
    )

    # --- Highlight cards ---
    _render_highlight_cards(results)

    # --- Build DataFrame with raw numeric values ---
    rows: list[dict] = []
    for name, res in results.items():
        m = res["metrics"]
        final_val = m["final_value"]
        total_inv = m["total_invested"]
        # ROI: (final - invested) / invested * 100
        roi = ((final_val - total_inv) / total_inv * 100) if total_inv > 0 else 0.0
        # Total Return: (final / start - 1) * 100  where start = first history value
        hist = res["history"]
        start_val = hist.iloc[0] if len(hist) > 0 else total_inv
        total_return = ((final_val / start_val - 1) * 100) if start_val > 0 else 0.0

        rows.append({
            "Scenario": name,
            "Final Value": final_val,
            "Invested": total_inv,
            "Profit": m["profit"],
            "ROI": roi,
            "Total Return": total_return,
            "CAGR": m["cagr"] * 100,
            "Sharpe": m["sharpe"],
            "Sortino": m["sortino"],
            "Volatility": m["volatility"] * 100,
            "Max DD": m["mdd"] * 100,
            "Calmar": m["calmar"],
            "Pain Idx": m["pain_index"] * 100,
            "Pain Ratio": m["pain_ratio"],
            "Win Rate": m["win_rate"] * 100,
            # Map inf → None so the column renders blank/— instead of "inf"
            # when there are no losing days (matches the tooltip).
            "Profit Factor": (
                None if m.get("profit_factor", 0.0) == float("inf")
                else m.get("profit_factor", 0.0)
            ),
        })

    df = pd.DataFrame(rows)

    # Default sort by CAGR descending
    df = df.sort_values("CAGR", ascending=False).reset_index(drop=True)

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Scenario": st.column_config.TextColumn(
                "Scenario", width="medium",
            ),
            "Final Value": st.column_config.NumberColumn(
                "Final Value", format="$%,.0f",
            ),
            "Invested": st.column_config.NumberColumn(
                "Invested", format="$%,.0f",
            ),
            "Profit": st.column_config.NumberColumn(
                "Profit", format="$%,.0f",
            ),
            "ROI": st.column_config.NumberColumn(
                "ROI",
                format="%.2f%%",
                help="Return on Investment: (Final − Invested) / Invested",
            ),
            "Total Return": st.column_config.NumberColumn(
                "Total Ret",
                format="%.2f%%",
                help="Total Return: (Final / Start − 1) × 100",
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
            "Profit Factor": st.column_config.NumberColumn(
                "Profit Factor", format="%.2f",
                help="Gross wins / gross losses (∞ shown as — when no losing days)",
            ),
        },
    )


def _render_highlight_cards(results: dict) -> None:
    """Show top-ranked scenario highlight cards above the table."""
    if len(results) < 2:
        return

    best_cagr_name = max(results, key=lambda n: results[n]["metrics"]["cagr"])
    best_sharpe_name = max(results, key=lambda n: results[n]["metrics"]["sharpe"])
    # mdd is negative — max gives least-bad drawdown
    lowest_mdd_name = max(results, key=lambda n: results[n]["metrics"]["mdd"])

    # Best ROI
    def _roi(n: str) -> float:
        m = results[n]["metrics"]
        inv = m["total_invested"]
        return ((m["final_value"] - inv) / inv * 100) if inv > 0 else 0
    best_roi_name = max(results, key=_roi)

    # Highest win rate
    best_wr_name = max(results, key=lambda n: results[n]["metrics"]["win_rate"])

    cols = st.columns(5)
    _card_style = (
        "border:1px solid {border};border-radius:10px;padding:12px 16px;"
        "background:{bg};text-align:center;"
    )

    with cols[0]:
        cagr_val = results[best_cagr_name]["metrics"]["cagr"] * 100
        st.markdown(
            f'<div style="{_card_style.format(border=_PRIMARY, bg=_SECONDARY)}">'
            f'<p style="color:{_TEXT};font-size:0.75rem;margin:0;">🏆 Best CAGR</p>'
            f'<p style="color:{_PRIMARY};font-size:1.5rem;font-weight:700;margin:4px 0;">'
            f'{cagr_val:.2f}%</p>'
            f'<p style="color:{_TEXT};font-size:0.7rem;margin:0;opacity:0.7;">'
            f'{best_cagr_name}</p></div>',
            unsafe_allow_html=True,
        )

    with cols[1]:
        sharpe_val = results[best_sharpe_name]["metrics"]["sharpe"]
        st.markdown(
            f'<div style="{_card_style.format(border="#7C4DFF", bg=_SECONDARY)}">'
            f'<p style="color:{_TEXT};font-size:0.75rem;margin:0;">📊 Best Sharpe</p>'
            f'<p style="color:#7C4DFF;font-size:1.5rem;font-weight:700;margin:4px 0;">'
            f'{sharpe_val:.2f}</p>'
            f'<p style="color:{_TEXT};font-size:0.7rem;margin:0;opacity:0.7;">'
            f'{best_sharpe_name}</p></div>',
            unsafe_allow_html=True,
        )

    with cols[2]:
        mdd_val = results[lowest_mdd_name]["metrics"]["mdd"] * 100
        st.markdown(
            f'<div style="{_card_style.format(border="#69F0AE", bg=_SECONDARY)}">'
            f'<p style="color:{_TEXT};font-size:0.75rem;margin:0;">🛡️ Shallowest DD</p>'
            f'<p style="color:#69F0AE;font-size:1.5rem;font-weight:700;margin:4px 0;">'
            f'{mdd_val:.2f}%</p>'
            f'<p style="color:{_TEXT};font-size:0.7rem;margin:0;opacity:0.7;">'
            f'{lowest_mdd_name}</p></div>',
            unsafe_allow_html=True,
        )

    with cols[3]:
        roi_val = _roi(best_roi_name)
        st.markdown(
            f'<div style="{_card_style.format(border="#FFD740", bg=_SECONDARY)}">'
            f'<p style="color:{_TEXT};font-size:0.75rem;margin:0;">💰 Best ROI</p>'
            f'<p style="color:#FFD740;font-size:1.5rem;font-weight:700;margin:4px 0;">'
            f'{roi_val:.1f}%</p>'
            f'<p style="color:{_TEXT};font-size:0.7rem;margin:0;opacity:0.7;">'
            f'{best_roi_name}</p></div>',
            unsafe_allow_html=True,
        )

    with cols[4]:
        wr_val = results[best_wr_name]["metrics"]["win_rate"] * 100
        st.markdown(
            f'<div style="{_card_style.format(border="#FF6F61", bg=_SECONDARY)}">'
            f'<p style="color:{_TEXT};font-size:0.75rem;margin:0;">🎯 Best Win Rate</p>'
            f'<p style="color:#FF6F61;font-size:1.5rem;font-weight:700;margin:4px 0;">'
            f'{wr_val:.1f}%</p>'
            f'<p style="color:{_TEXT};font-size:0.7rem;margin:0;opacity:0.7;">'
            f'{best_wr_name}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacing


def render_detailed_breakdown(start_date, end_date) -> None:
    """Render the detailed assumptions / methodology note."""
    with st.expander("ℹ️ Methodology & Assumptions"):
        st.markdown(
            f"""
            | Parameter | Detail |
            |---|---|
            | **Rebalancing** | Monthly to target weights on the 1st trading day |
            | **Taxes** | Estimated at **22%** on realized gains exceeding the threshold |
            | **Expense ratios** | Applied daily (annual ratio / 252) |
            | **Slippage & commission** | Deducted from cash on each rebalance |
            | **Data range** | `{start_date}` → `{end_date}` |
            """
        )
