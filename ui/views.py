"""
Advanced analysis views — radar comparison, correlation heatmap,
rolling returns, annual returns.

Theme: primary=#00ADB5, bg=#222831, secondary=#393E46, text=#EEEEEE
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

# ── Theme tokens ──────────────────────────────────────────────────────────────
_PRIMARY = "#00ADB5"
_BG = "#222831"
_SECONDARY = "#393E46"
_TEXT = "#EEEEEE"

_COLORS = [
    "#00E5FF", "#FF6F61", "#7C4DFF", "#FFD740",
    "#69F0AE", "#FF4081", "#448AFF", "#FFAB40",
    "#B388FF", "#64FFDA", "#FF8A80", "#82B1FF",
    "#EA80FC", "#A7FFEB", "#FF80AB", "#8C9EFF",
]


def render_analysis_views(results: dict) -> None:
    """Render all analysis tabs: radar, correlation, rolling returns, annual returns.

    Args:
        results: {scenario_name: {"history": pd.Series, "metrics": dict}, …}
    """
    if not results or len(results) < 1:
        return

    tab_radar, tab_corr, tab_rolling, tab_annual = st.tabs([
        "🎯 Radar Comparison",
        "🔗 Correlation Matrix",
        "📈 Rolling Returns",
        "📅 Annual Returns",
    ])

    with tab_radar:
        _render_radar_chart(results)
    with tab_corr:
        _render_correlation_heatmap(results)
    with tab_rolling:
        _render_rolling_returns(results)
    with tab_annual:
        _render_annual_returns(results)


# ── Radar Chart ───────────────────────────────────────────────────────────────

# Metrics used for radar comparison with human-friendly labels
_RADAR_METRICS = [
    ("CAGR", "cagr", False),
    ("Sharpe", "sharpe", False),
    ("Sortino", "sortino", False),
    ("−Max DD", "mdd", True),       # inverted: less negative = better
    ("Win Rate", "win_rate", False),
    ("Calmar", "calmar", False),
]


def _render_radar_chart(results: dict) -> None:
    """Radar / spider chart comparing scenarios across key metrics.

    Each metric is normalized to 0-100 across the selected scenarios so
    differences are visually comparable regardless of absolute scale.
    """
    if len(results) < 2:
        st.info("Select at least **2 scenarios** to see the radar comparison.")
        return

    st.markdown(
        f'<p style="color:{_TEXT};font-size:0.85rem;">'
        "Each axis is normalized 0–100 across selected scenarios. "
        "Larger area = better overall risk-adjusted performance.</p>",
        unsafe_allow_html=True,
    )

    # Extract raw values
    names = list(results.keys())
    raw: dict[str, list[float]] = {name: [] for name in names}
    labels: list[str] = []

    for label, key, invert in _RADAR_METRICS:
        labels.append(label)
        vals = []
        for name in names:
            v = results[name]["metrics"].get(key, 0)
            if invert:
                # MDD is negative; negate so higher = better
                v = -v
            vals.append(float(v))

        # Normalize to 0-100
        vmin, vmax = min(vals), max(vals)
        span = vmax - vmin if vmax != vmin else 1.0
        for name, v in zip(names, vals):
            raw[name].append((v - vmin) / span * 100)

    # Close the polygon (repeat first point)
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    for i, name in enumerate(names):
        values_closed = raw[name] + [raw[name][0]]
        colour = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=_hex_to_rgba(colour, 0.10),
            name=name,
            line=dict(color=colour, width=2),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "%{theta}: %{r:.1f}/100"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=_SECONDARY,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(size=9, color=_TEXT),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(size=11, color=_TEXT),
            ),
        ),
        template="plotly_dark",
        paper_bgcolor=_BG,
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11, color=_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=60, t=40, b=80),
        font=dict(color=_TEXT),
    )

    st.plotly_chart(fig, use_container_width=True)


# ── Correlation Heatmap ───────────────────────────────────────────────────────


def _render_correlation_heatmap(results: dict) -> None:
    """Pairwise Pearson correlation of daily returns across all scenarios."""
    if len(results) < 2:
        st.info("Select at least **2 scenarios** to see correlations.")
        return

    # Build daily-returns DataFrame
    returns_df = pd.DataFrame({
        name: res["history"].pct_change().dropna()
        for name, res in results.items()
    }).dropna()

    if returns_df.empty:
        st.warning("Not enough overlapping data.")
        return

    corr = returns_df.corr()
    labels = list(corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=11, color=_TEXT),
        hovertemplate=(
            "<b>%{x}</b> vs <b>%{y}</b><br>"
            "Correlation: %{z:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_SECONDARY,
        height=max(400, 60 * len(labels)),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="Return Correlation Matrix", font=dict(color=_PRIMARY)),
        xaxis=dict(tickangle=-45),
        font=dict(color=_TEXT),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insights
    if len(labels) >= 2:
        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append((labels[i], labels[j], corr.iloc[i, j]))

        if pairs:
            pairs_sorted = sorted(pairs, key=lambda x: x[2])
            least = pairs_sorted[0]
            most = pairs_sorted[-1]

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "🔀 Least Correlated",
                    f"{least[2]:.3f}",
                    f"{least[0]} ↔ {least[1]}",
                )
            with col2:
                st.metric(
                    "🔗 Most Correlated",
                    f"{most[2]:.3f}",
                    f"{most[0]} ↔ {most[1]}",
                )


# ── Rolling Returns ───────────────────────────────────────────────────────────


def _render_rolling_returns(results: dict) -> None:
    """Rolling annualized returns for each scenario."""
    window_options = {"1 Year": 252, "3 Years": 252 * 3, "5 Years": 252 * 5}
    selected_window = st.radio(
        "Rolling Window",
        list(window_options.keys()),
        horizontal=True,
        key="rolling_window",
    )
    window = window_options[selected_window]

    fig = go.Figure()
    any_data = False

    for i, (name, res) in enumerate(results.items()):
        hist = res["history"]
        if len(hist) < window:
            continue

        rolling_ret = (hist / hist.shift(window)) ** (252 / window) - 1
        rolling_ret = rolling_ret.dropna() * 100

        if rolling_ret.empty:
            continue

        any_data = True
        colour = _COLORS[i % len(_COLORS)]

        fig.add_trace(go.Scatter(
            x=rolling_ret.index,
            y=rolling_ret,
            mode="lines",
            name=name,
            line=dict(color=colour, width=2),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "📅 %{x|%b %d, %Y}<br>"
                "📈 Rolling Return: %{y:.2f}%"
                "<extra></extra>"
            ),
        ))

    if not any_data:
        st.info(f"Not enough data for {selected_window} rolling window.")
        return

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_SECONDARY,
        height=500,
        title=dict(
            text=f"Rolling {selected_window} Annualized Return (%)",
            font=dict(color=_PRIMARY),
        ),
        yaxis_title="Return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11, color=_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(color=_TEXT),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    st.plotly_chart(fig, use_container_width=True)


# ── Annual Returns Bar Chart ─────────────────────────────────────────────────


def _render_annual_returns(results: dict) -> None:
    """Year-by-year return breakdown as a grouped bar chart."""
    annual_data: list[dict] = []

    for name, res in results.items():
        hist = res["history"]
        yearly_groups = hist.groupby(hist.index.year)

        for year, group in yearly_groups:
            if len(group) < 2:
                continue
            year_ret = (group.iloc[-1] / group.iloc[0] - 1) * 100
            annual_data.append({
                "Year": str(year),
                "Scenario": name,
                "Return (%)": year_ret,
            })

    if not annual_data:
        st.info("Not enough data for annual breakdown.")
        return

    df = pd.DataFrame(annual_data)

    fig = px.bar(
        df,
        x="Year",
        y="Return (%)",
        color="Scenario",
        barmode="group",
        template="plotly_dark",
        title="Annual Returns by Scenario",
        color_discrete_sequence=_COLORS,
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_SECONDARY,
        height=500,
        hovermode="x unified",
        title=dict(font=dict(color=_PRIMARY)),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11, color=_TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(color=_TEXT),
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)")

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    st.plotly_chart(fig, use_container_width=True)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
