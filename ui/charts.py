"""
Plotly chart rendering — performance chart, drawdown chart, toggles.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd


def render_charts(results: dict) -> None:
    """Render performance and drawdown charts with toggles.

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
        use_normalized = st.toggle("📐 Normalized (start at 100)", value=False, key="norm_toggle")

    # ── Build performance chart ──
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Performance", "Drawdown"),
    )

    # Colour palette — rich, premium colours
    colors = [
        "#00E5FF", "#FF6F61", "#7C4DFF", "#FFD740",
        "#69F0AE", "#FF4081", "#448AFF", "#FFAB40",
        "#B388FF", "#64FFDA", "#FF8A80", "#82B1FF",
        "#EA80FC", "#A7FFEB", "#FF80AB", "#8C9EFF",
    ]

    for i, (name, res) in enumerate(results.items()):
        hist = res["history"]
        colour = colors[i % len(colors)]

        # Normalize if toggled
        display_hist = hist
        if use_normalized:
            display_hist = (hist / hist.iloc[0]) * 100

        # Performance line
        fig.add_trace(
            go.Scatter(
                x=display_hist.index,
                y=display_hist,
                mode="lines",
                name=name,
                line=dict(color=colour, width=2),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    + ("Value: %{y:.1f}<br>" if use_normalized else "Value: $%{y:,.0f}<br>")
                    + "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1, col=1,
        )

        # Drawdown
        roll_max = hist.cummax()
        drawdown = (hist - roll_max) / roll_max * 100  # Percent

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode="lines",
                name=f"{name} DD",
                line=dict(color=colour, width=1.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(colour, 0.15),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
            ),
            row=2, col=1,
        )

    y_title = "Value (Indexed to 100)" if use_normalized else "Portfolio Value ($)"

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=80, b=40),
    )

    fig.update_yaxes(
        title_text=y_title,
        type="log" if use_log else "linear",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Drawdown (%)",
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
