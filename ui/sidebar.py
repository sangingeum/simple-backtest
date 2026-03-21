"""
Sidebar UI — settings, strategy parameters, and scenario management.
"""

import streamlit as st
import pandas as pd

from config import (
    ALL_STRATEGIES,
    DEFAULT_EXPENSE_RATIOS,
    STRATEGY_DESCRIPTIONS,
    STRAT_CROSS,
    STRAT_MONTHLY,
    STRAT_RSI,
    STRAT_TRAIL,
    STRAT_TREND,
    STRAT_VOL,
    STRAT_MACD,
    STRAT_BBAND,
)
from scenarios import (
    create_scenario,
    delete_scenario,
    export_scenarios,
    import_scenarios,
    ensure_expenses,
)


def render_settings() -> dict:
    """Render sidebar settings and return a dict of all user-selected parameters."""
    st.sidebar.header("Settings")

    initial_cash = st.sidebar.number_input(
        "Initial Cash ($)", value=20000, step=1000
    )
    monthly_cash = st.sidebar.number_input(
        "Monthly Contribution ($)", value=1500, step=100
    )
    tax_threshold = st.sidebar.number_input(
        "Tax Free Threshold ($)", value=2000, step=500
    )
    inflation_rate = (
        st.sidebar.slider("Annual Inflation Rate (%)", 0.0, 10.0, 0.0, 0.1) / 100
    )
    slippage_rate = (
        st.sidebar.slider(
            "Slippage (%)", 0.0, 5.0, 0.1, 0.1,
            help="Estimated price impact per trade.",
        )
        / 100.0
    )
    commission_fee = st.sidebar.number_input(
        "Commission per Trade ($)", value=0.0, step=1.0,
        help="Flat fee per ticker traded per rebalance.",
    )
    tax_settlement_mode = st.sidebar.selectbox(
        "Tax Settlement", ["Immediate", "Annual"],
        help=(
            "'Immediate': Pay tax instantly on rebalance gains.\n"
            "'Annual': Defer payment until year-end."
        ),
    )

    st.sidebar.markdown("---")

    # --- Strategy ---
    st.sidebar.subheader("Strategy")

    strategy_mode = st.sidebar.selectbox(
        "Trading Strategy", ALL_STRATEGIES,
        help="Select the logic that determines Risk-On or Risk-Off.",
    )
    st.sidebar.info(STRATEGY_DESCRIPTIONS[strategy_mode])

    # Strategy-specific parameters
    signal_ticker = "QQQ"
    safe_assets = ["GLD"]
    sma_window = 200
    risk_off_invested_pct = 0.0
    sma_fast = 50
    sma_slow = 200
    vix_threshold = 30.0
    trailing_stop_pct = 0.15
    use_dual_momentum = False
    rsi_period = 14
    rsi_overbought = 70.0
    rsi_oversold = 30.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    bb_period = 20
    bb_std = 2.0
    bb_squeeze = 0.04

    if strategy_mode != STRAT_MONTHLY:
        st.sidebar.markdown("#### Signal & Safety")
        col_sig, col_safe = st.sidebar.columns(2)
        with col_sig:
            if strategy_mode == STRAT_VOL:
                signal_ticker = st.text_input(
                    "Vol Ticker", "^VIX",
                    help="Ticker used to measure volatility.",
                ).upper()
            else:
                signal_ticker = st.text_input(
                    "Signal Ticker", "QQQ",
                    help="Ticker used to generate Buy/Sell signals.",
                ).upper()
        with col_safe:
            safe_str = st.text_input(
                "Safe Assets", "GLD",
                help="Comma-separated tickers to HOLD during Risk-Off.",
            )
            safe_assets = [s.strip().upper() for s in safe_str.split(",") if s.strip()]

        risk_off_invested_pct = (
            st.sidebar.slider(
                "Risk-Off Invested %", 0, 100, 0, 10,
                help=(
                    "0% = Move all risk assets to Cash/Safe.\n"
                    "50% = Partial de-leveraging."
                ),
            )
            / 100.0
        )

    # Strategy-specific knobs
    if strategy_mode == STRAT_TREND:
        sma_window = st.sidebar.slider("SMA Window", 20, 300, 200, 10)
        use_dual_momentum = st.sidebar.checkbox(
            "Dual Momentum Filter", value=False,
            help="Stay Risk-On if 1-Month Return is positive even if Price < SMA.",
        )

    elif strategy_mode == STRAT_CROSS:
        c1, c2 = st.sidebar.columns(2)
        sma_fast = c1.number_input("Fast SMA", 10, 200, 50)
        sma_slow = c2.number_input("Slow SMA", 50, 400, 200)

    elif strategy_mode == STRAT_VOL:
        vix_threshold = st.sidebar.number_input(
            "VIX Threshold", 10.0, 100.0, 30.0, step=1.0,
        )
        st.sidebar.caption(f"Risk-Off when {signal_ticker} > {vix_threshold}")

    elif strategy_mode == STRAT_TRAIL:
        trailing_stop_pct = (
            st.sidebar.slider("Stop Loss %", 5, 50, 15, 1) / 100.0
        )
        sma_window = st.sidebar.slider("Re-Entry SMA", 20, 300, 200, 10)

    elif strategy_mode == STRAT_RSI:
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14, 1)
        c1, c2 = st.sidebar.columns(2)
        rsi_overbought = c1.number_input("Overbought", 60.0, 90.0, 70.0, step=5.0)
        rsi_oversold = c2.number_input("Oversold", 10.0, 40.0, 30.0, step=5.0)

    elif strategy_mode == STRAT_MACD:
        st.sidebar.markdown("**MACD Parameters**")
        c1, c2, c3 = st.sidebar.columns(3)
        macd_fast = c1.number_input("Fast EMA", 5, 50, 12)
        macd_slow = c2.number_input("Slow EMA", 10, 100, 26)
        macd_signal = c3.number_input("Signal", 3, 30, 9)

    elif strategy_mode == STRAT_BBAND:
        st.sidebar.markdown("**Bollinger Band Parameters**")
        c1, c2 = st.sidebar.columns(2)
        bb_period = c1.number_input("BB Period", 10, 50, 20)
        bb_std = c2.number_input("Std Devs", 1.0, 4.0, 2.0, step=0.5)
        bb_squeeze = st.sidebar.slider(
            "Squeeze Threshold", 0.01, 0.10, 0.04, 0.01,
            help="Bandwidth below this → squeeze detected.",
        )

    return {
        "initial_cash": initial_cash,
        "monthly_cash": monthly_cash,
        "tax_threshold": tax_threshold,
        "inflation_rate": inflation_rate,
        "slippage_rate": slippage_rate,
        "commission_fee": commission_fee,
        "tax_settlement_mode": tax_settlement_mode,
        "strategy_mode": strategy_mode,
        "signal_ticker": signal_ticker,
        "safe_assets": safe_assets,
        "sma_window": sma_window,
        "risk_off_invested_pct": risk_off_invested_pct,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "vix_threshold": vix_threshold,
        "trailing_stop_pct": trailing_stop_pct,
        "use_dual_momentum": use_dual_momentum,
        "rsi_period": rsi_period,
        "rsi_overbought": rsi_overbought,
        "rsi_oversold": rsi_oversold,
        "macd_fast": macd_fast,
        "macd_slow": macd_slow,
        "macd_signal": macd_signal,
        "bb_period": bb_period,
        "bb_std": bb_std,
        "bb_squeeze": bb_squeeze,
    }


# ── Scenario Manager ──────────────────────────────────────────────────────────

def render_scenario_manager() -> list[dict]:
    """Render scenario selection with checkbox grid, create/manage tabs.
    Returns list of active scenario dicts: [{name, tickers, weights, expenses}, …].
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenarios")

    all_names = list(st.session_state.scenarios.keys())

    # Initialize active list
    if "active_scenarios" not in st.session_state:
        st.session_state.active_scenarios = [
            n for n in all_names if any(
                tag in n for tag in ("VOO", "TQQQ", "QLD")
            )
        ][:5]

    # Sync: remove names that no longer exist
    st.session_state.active_scenarios = [
        n for n in st.session_state.active_scenarios if n in all_names
    ]

    # ── Quick actions ──
    btn_cols = st.sidebar.columns(3)
    with btn_cols[0]:
        if st.button("All", use_container_width=True, key="btn_select_all"):
            st.session_state.active_scenarios = list(all_names)
            for n in all_names:
                st.session_state[f"scen_cb_{n}"] = True
            st.rerun()
    with btn_cols[1]:
        if st.button("None", use_container_width=True, key="btn_clear_all"):
            st.session_state.active_scenarios = []
            for n in all_names:
                st.session_state[f"scen_cb_{n}"] = False
            st.rerun()
    with btn_cols[2]:
        if st.button("Invert", use_container_width=True, key="btn_invert"):
            new_active = [
                n for n in all_names
                if n not in st.session_state.active_scenarios
            ]
            st.session_state.active_scenarios = new_active
            for n in all_names:
                st.session_state[f"scen_cb_{n}"] = (n in new_active)
            st.rerun()

    # ── Checkbox grid for scenario selection ──
    _render_scenario_checkboxes(all_names)

    # ── Tabbed management interface ──
    tab_create, tab_manage, tab_io = st.sidebar.tabs([
        "New", "Manage", "Import/Export",
    ])
    with tab_create:
        _render_create_form()
    with tab_manage:
        _render_manage_ui()
    with tab_io:
        _render_import_export()

    # Build selected scenario list
    result: list[dict] = []
    for name in st.session_state.active_scenarios:
        if name not in st.session_state.scenarios:
            continue
        details = ensure_expenses(st.session_state.scenarios[name])
        result.append({
            "name": name,
            "tickers": details["tickers"],
            "weights": details["weights"],
            "expenses": details["expenses"],
        })
    return result


def _render_scenario_checkboxes(all_names: list[str]) -> None:
    """Render a scrollable checkbox list with inline delete buttons."""
    # Container with max height for scrollability (CSS injected in app.py)
    with st.sidebar.container(height=300):
        for name in all_names:
            # Build mini summary
            details = st.session_state.scenarios[name]
            tickers = details.get("tickers", [])
            weights = details.get("weights", [])
            summary = " · ".join(
                f"{t} {w:.0%}" for t, w in zip(tickers, weights)
            )

            col_cb, col_del = st.columns([5, 1])

            with col_cb:
                is_active = name in st.session_state.active_scenarios
                new_val = st.checkbox(
                    name,
                    value=is_active,
                    key=f"scen_cb_{name}",
                    help=summary,
                )

                # Update active list on toggle
                if new_val and name not in st.session_state.active_scenarios:
                    st.session_state.active_scenarios.append(name)
                elif not new_val and name in st.session_state.active_scenarios:
                    st.session_state.active_scenarios.remove(name)

            with col_del:
                if st.button("×", key=f"del_{name}", help=f"Delete {name}"):
                    delete_scenario(st.session_state.scenarios, name)
                    if name in st.session_state.active_scenarios:
                        st.session_state.active_scenarios.remove(name)
                    st.toast(f"Deleted: {name}")
                    st.rerun()


def _render_create_form() -> None:
    """Form to create a new custom scenario."""
    default_data = pd.DataFrame([
        {"Ticker": "AAPL", "Weight": 0.5, "Expense Ratio": 0.0},
        {"Ticker": "MSFT", "Weight": 0.5, "Expense Ratio": 0.0},
    ])
    edited_df = st.data_editor(
        default_data, num_rows="dynamic", use_container_width=True,
        key="new_scenario_editor",
    )

    # Live weight indicator
    total_w = edited_df["Weight"].astype(float).sum()
    colour = "green" if abs(total_w - 1.0) < 0.01 else "red"
    st.markdown(
        f"Total weight: :{colour}[**{total_w:.2f}**] "
        f"{'✅' if colour == 'green' else '⚠️ must equal 1.0'}"
    )

    # Auto-generate name from tickers
    tickers_preview = [
        t.strip().upper()
        for t in edited_df["Ticker"].astype(str).tolist()
        if t.strip()
    ]
    auto_name = " / ".join(
        f"{t} {w:.0%}" for t, w in zip(tickers_preview, edited_df["Weight"])
    ) if tickers_preview else "My Portfolio"

    new_name = st.text_input("Scenario Name", auto_name, key="new_scen_name")

    if st.button("Save Scenario", use_container_width=True, type="primary"):
        c_tickers = [
            t.strip().upper()
            for t in edited_df["Ticker"].astype(str).tolist()
            if t.strip()
        ]
        c_weights = edited_df["Weight"].astype(float).tolist()
        c_expenses = edited_df["Expense Ratio"].astype(float).tolist()

        err = create_scenario(
            st.session_state.scenarios, new_name, c_tickers, c_weights, c_expenses,
        )
        if err:
            st.error(err)
        else:
            st.session_state.active_scenarios.append(new_name)
            st.success(f"Created **{new_name}**!")
            st.rerun()


def _render_manage_ui() -> None:
    """Delete and clone scenarios."""
    all_names = list(st.session_state.scenarios.keys())
    if not all_names:
        st.caption("No scenarios to manage.")
        return

    # ── Clone ──
    st.markdown("**Clone a Scenario**")
    source = st.selectbox(
        "Source scenario", all_names, key="clone_source",
        label_visibility="collapsed",
    )
    if source:
        details = st.session_state.scenarios[source]
        clone_name = st.text_input(
            "New name", f"{source} (copy)", key="clone_name",
        )
        if st.button("Clone", use_container_width=True, key="btn_clone"):
            err = create_scenario(
                st.session_state.scenarios,
                clone_name,
                list(details["tickers"]),
                list(details["weights"]),
                list(details.get("expenses", [])),
            )
            if err:
                st.error(err)
            else:
                st.session_state.active_scenarios.append(clone_name)
                st.success(f"Cloned as **{clone_name}**!")
                st.rerun()

    st.markdown("---")

    # ── Delete ──
    st.markdown("**Delete Scenarios**")
    to_delete = st.multiselect(
        "Select scenarios to delete",
        options=all_names,
        key="delete_selector",
        label_visibility="collapsed",
    )
    if to_delete and st.button(
        f"Delete {len(to_delete)} scenario(s)",
        type="primary",
        use_container_width=True,
    ):
        for name in to_delete:
            delete_scenario(st.session_state.scenarios, name)
        st.toast(f"Deleted: {', '.join(to_delete)}")
        st.rerun()


def _render_import_export() -> None:
    """Import / export scenario sets as JSON."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Export**")
        json_str = export_scenarios(
            st.session_state.scenarios,
            list(st.session_state.scenarios.keys()),
        )
        st.download_button(
            "Download JSON",
            data=json_str,
            file_name="scenarios_export.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        st.markdown("**Import**")
        uploaded = st.file_uploader(
            "Upload JSON", type=["json"], key="import_upload", label_visibility="collapsed",
        )
        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            imported, skipped = import_scenarios(st.session_state.scenarios, content)
            if imported > 0:
                st.success(f"Imported {imported} scenario(s).")
                st.rerun()
            elif skipped > 0:
                st.warning("All scenarios already exist (skipped).")
            else:
                st.error("Invalid JSON file.")
