"""
Plotly chart rendering — performance chart with reference lines, drawdown chart.

Theme: primary=#00ADB5, bg=#222831, secondary=#393E46, text=#EEEEEE
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

# ── Theme tokens ──────────────────────────────────────────────────────────────
_PRIMARY = "#00ADB5"
_BG = "#222831"
_SECONDARY = "#393E46"
_TEXT = "#EEEEEE"

# Rich colour palette — high-contrast on dark backgrounds
_COLORS = [
    "#00E5FF", "#FF6F61", "#7C4DFF", "#FFD740",
    "#69F0AE", "#FF4081", "#448AFF", "#FFAB40",
    "#B388FF", "#64FFDA", "#FF8A80", "#82B1FF",
    "#EA80FC", "#A7FFEB", "#FF80AB", "#8C9EFF",
]


def render_charts(results: dict) -> None:
    """Render performance and drawdown charts with toggles and reference lines.

    Args:
        results: {scenario_name: {"history": pd.Series, "metrics": dict}, …}
    """
    if not results:
        return

    # ── Toggles ──
    col1, col2 = st.columns([1, 1])
    with col1:
        use_log = st.toggle("📊 Log Scale", value=True, key="log_toggle")
    with col2:
        use_normalized = st.toggle(
            "📐 Normalized (start at 100)", value=False, key="norm_toggle",
        )

    # ── Subplots ──
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Performance", "Drawdown"),
    )

    # ── Compute reference lines ──
    # Initial capital: first value of the first scenario's history
    first_hist = next(iter(results.values()))["history"]
    initial_capital = first_hist.iloc[0] if len(first_hist) > 0 else 0

    # Max total invested across all scenarios
    max_invested = max(
        (res["metrics"].get("total_invested", 0) for res in results.values()),
        default=0,
    )

    for i, (name, res) in enumerate(results.items()):
        hist = res["history"]
        colour = _COLORS[i % len(_COLORS)]

        # Normalize if toggled
        display_hist = (hist / hist.iloc[0]) * 100 if use_normalized else hist

        # Performance line
        fig.add_trace(
            go.Scatter(
                x=display_hist.index,
                y=display_hist,
                mode="lines",
                name=name,
                line=dict(color=colour, width=2),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "📅 %{x|%b %d, %Y}<br>"
                    + (
                        "📈 Value: %{y:.1f}<br>"
                        if use_normalized
                        else "💰 Value: $%{y:,.0f}<br>"
                    )
                    + "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # Drawdown
        roll_max = hist.cummax()
        drawdown = (hist - roll_max) / roll_max * 100  # percent

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode="lines",
                name=f"{name} DD",
                line=dict(color=colour, width=1.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(colour, 0.12),
                showlegend=False,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "📅 %{x|%b %d, %Y}<br>"
                    "📉 Drawdown: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    # ── Reference lines on performance chart ──
    if not use_normalized:
        # Initial capital
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="rgba(238,238,238,0.35)",
            line_width=1,
            annotation_text=f"Initial ${initial_capital:,.0f}",
            annotation_position="top left",
            annotation_font=dict(color=_TEXT, size=10),
            row=1, col=1,
        )
        # Total invested (max across scenarios)
        if max_invested > initial_capital:
            fig.add_hline(
                y=max_invested,
                line_dash="dot",
                line_color=_hex_to_rgba(_PRIMARY, 0.5),
                line_width=1,
                annotation_text=f"Total Invested ${max_invested:,.0f}",
                annotation_position="bottom left",
                annotation_font=dict(color=_PRIMARY, size=10),
                row=1, col=1,
            )

    # Drawdown zero line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.2)",
        row=2, col=1,
    )

    # ── Layout ──
    y_title = "Value (Indexed to 100)" if use_normalized else "Portfolio Value ($)"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_SECONDARY,
        hovermode="x unified",
        height=780,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=80, b=40),
        font=dict(color=_TEXT),
    )

    fig.update_yaxes(
        title_text=y_title,
        type="log" if use_log else "linear",
        gridcolor="rgba(255,255,255,0.06)",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Drawdown (%)",
        gridcolor="rgba(255,255,255,0.06)",
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text="Date",
        gridcolor="rgba(255,255,255,0.06)",
        row=2, col=1,
    )

    # Style subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=13, color=_PRIMARY)

    st.plotly_chart(fig, use_container_width=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
