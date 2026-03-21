"""
Advanced analysis views — correlation heatmap, rolling returns, annual returns.
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np


def render_analysis_views(results: dict) -> None:
    """Render all analysis tabs: correlation, rolling returns, annual returns.

    Args:
        results: {scenario_name: {"history": pd.Series, "metrics": dict}, …}
    """
    if not results or len(results) < 1:
        return

    tab_corr, tab_rolling, tab_annual = st.tabs([
        "Correlation Matrix",
        "Rolling Returns",
        "Annual Returns",
    ])

    with tab_corr:
        _render_correlation_heatmap(results)
    with tab_rolling:
        _render_rolling_returns(results)
    with tab_annual:
        _render_annual_returns(results)


# ── Correlation Heatmap ───────────────────────────────────────────────────────


def _render_correlation_heatmap(results: dict) -> None:
    """Pairwise Pearson correlation of daily returns across all scenarios."""
    if len(results) < 2:
        st.info("Select at least 2 scenarios to see correlations.")
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

    # Shorten names for readability
    labels = list(corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=max(400, 60 * len(labels)),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Return Correlation Matrix",
        xaxis=dict(tickangle=-45),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insights
    if len(labels) >= 2:
        # Find most and least correlated pairs
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
    """Rolling 1-year and 3-year annualized returns for each scenario."""
    window_options = {"1 Year": 252, "3 Years": 252 * 3, "5 Years": 252 * 5}
    selected_window = st.radio(
        "Rolling Window",
        list(window_options.keys()),
        horizontal=True,
        key="rolling_window",
    )
    window = window_options[selected_window]

    colors = [
        "#00E5FF", "#FF6F61", "#7C4DFF", "#FFD740",
        "#69F0AE", "#FF4081", "#448AFF", "#FFAB40",
        "#B388FF", "#64FFDA", "#FF8A80", "#82B1FF",
    ]

    fig = go.Figure()
    any_data = False

    for i, (name, res) in enumerate(results.items()):
        hist = res["history"]
        if len(hist) < window:
            continue

        # Rolling annualized return: (V_t / V_{t-window})^(252/window) - 1
        rolling_ret = (hist / hist.shift(window)) ** (252 / window) - 1
        rolling_ret = rolling_ret.dropna() * 100  # Convert to percent

        if rolling_ret.empty:
            continue

        any_data = True
        colour = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=rolling_ret.index,
            y=rolling_ret,
            mode="lines",
            name=name,
            line=dict(color=colour, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Rolling Return: %{y:.2f}%<extra>%{fullData.name}</extra>",
        ))

    if not any_data:
        st.info(f"Not enough data for {selected_window} rolling window.")
        return

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title=f"Rolling {selected_window} Annualized Return (%)",
        yaxis_title="Return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


# ── Annual Returns Bar Chart ─────────────────────────────────────────────────


def _render_annual_returns(results: dict) -> None:
    """Year-by-year return breakdown as a grouped bar chart."""
    annual_data: list[dict] = []

    for name, res in results.items():
        hist = res["history"]
        # Group by year
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
        color_discrete_sequence=[
            "#00E5FF", "#FF6F61", "#7C4DFF", "#FFD740",
            "#69F0AE", "#FF4081", "#448AFF", "#FFAB40",
            "#B388FF", "#64FFDA", "#FF8A80", "#82B1FF",
        ],
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    st.plotly_chart(fig, use_container_width=True)
