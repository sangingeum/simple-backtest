"""
Sidebar UI — settings, strategy parameters, and dialog-based scenario management.

Uses @st.dialog (streamlit >= 1.33) for all scenario CRUD operations so the
sidebar stays clean and focused on selection & strategy tuning.
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
    save_scenarios,
)

# ── Theme tokens ──────────────────────────────────────────────────────────────
_PRIMARY = "#00ADB5"
_BG = "#222831"
_SECONDARY = "#393E46"
_TEXT = "#EEEEEE"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy-parameter registry — drives the UI declaratively
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY_PARAMS: dict[str, list[dict]] = {
    STRAT_TREND: [
        {
            "key": "sma_window", "widget": "slider",
            "label": "SMA Window", "min": 20, "max": 300, "default": 200, "step": 10,
        },
        {
            "key": "use_dual_momentum", "widget": "checkbox",
            "label": "Dual Momentum Filter", "default": False,
            "help": "Stay Risk-On if 1-Month Return is positive even if Price < SMA.",
        },
    ],
    STRAT_CROSS: [
        {
            "key": "sma_fast", "widget": "number_input", "label": "Fast SMA",
            "min": 10, "max": 200, "default": 50, "column": 0,
        },
        {
            "key": "sma_slow", "widget": "number_input", "label": "Slow SMA",
            "min": 50, "max": 400, "default": 200, "column": 1,
        },
    ],
    STRAT_VOL: [
        {
            "key": "vix_threshold", "widget": "number_input",
            "label": "VIX Threshold", "min": 10.0, "max": 100.0,
            "default": 30.0, "step": 1.0,
        },
    ],
    STRAT_TRAIL: [
        {
            "key": "trailing_stop_pct", "widget": "slider",
            "label": "Stop Loss %", "min": 5, "max": 50, "default": 15, "step": 1,
            "divisor": 100.0,
        },
        {
            "key": "sma_window", "widget": "slider",
            "label": "Re-Entry SMA", "min": 20, "max": 300, "default": 200, "step": 10,
        },
    ],
    STRAT_RSI: [
        {
            "key": "rsi_period", "widget": "slider",
            "label": "RSI Period", "min": 5, "max": 30, "default": 14, "step": 1,
        },
        {
            "key": "rsi_overbought", "widget": "number_input",
            "label": "Overbought", "min": 60.0, "max": 90.0,
            "default": 70.0, "step": 5.0, "column": 0,
        },
        {
            "key": "rsi_oversold", "widget": "number_input",
            "label": "Oversold", "min": 10.0, "max": 40.0,
            "default": 30.0, "step": 5.0, "column": 1,
        },
    ],
    STRAT_MACD: [
        {
            "key": "macd_fast", "widget": "number_input", "label": "Fast EMA",
            "min": 5, "max": 50, "default": 12, "column": 0,
        },
        {
            "key": "macd_slow", "widget": "number_input", "label": "Slow EMA",
            "min": 10, "max": 100, "default": 26, "column": 1,
        },
        {
            "key": "macd_signal", "widget": "number_input", "label": "Signal",
            "min": 3, "max": 30, "default": 9, "column": 2,
        },
    ],
    STRAT_BBAND: [
        {
            "key": "bb_period", "widget": "number_input", "label": "BB Period",
            "min": 10, "max": 50, "default": 20, "column": 0,
        },
        {
            "key": "bb_std", "widget": "number_input", "label": "Std Devs",
            "min": 1.0, "max": 4.0, "default": 2.0, "step": 0.5, "column": 1,
        },
        {
            "key": "bb_squeeze", "widget": "slider",
            "label": "Squeeze Threshold", "min": 0.01, "max": 0.10,
            "default": 0.04, "step": 0.01,
            "help": "Bandwidth below this → squeeze detected.",
        },
    ],
}

# Default values for ALL strategy params (so the return dict is always complete)
_PARAM_DEFAULTS: dict[str, object] = {
    "sma_window": 200,
    "use_dual_momentum": False,
    "sma_fast": 50,
    "sma_slow": 200,
    "vix_threshold": 30.0,
    "trailing_stop_pct": 0.15,
    "rsi_period": 14,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_squeeze": 0.04,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  render_settings()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_settings() -> dict:
    """Render sidebar settings and return a dict of all user-selected parameters."""
    st.sidebar.markdown(
        f'<h3 style="color:{_PRIMARY};margin-bottom:4px;">⚙️ Settings</h3>',
        unsafe_allow_html=True,
    )

    # ── Core parameters ──
    initial_cash = st.sidebar.number_input(
        "Initial Cash ($)", value=20000, step=1000,
    )
    monthly_cash = st.sidebar.number_input(
        "Monthly Contribution ($)", value=1500, step=100,
    )
    tax_threshold = st.sidebar.number_input(
        "Tax Free Threshold ($)", value=2000, step=500,
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

    # ── Strategy selector ──
    st.sidebar.markdown(
        f'<h4 style="color:{_PRIMARY};">📈 Strategy</h4>',
        unsafe_allow_html=True,
    )

    strategy_mode = st.sidebar.selectbox(
        "Trading Strategy", ALL_STRATEGIES,
        help="Select the logic that determines Risk-On or Risk-Off.",
    )
    st.sidebar.info(STRATEGY_DESCRIPTIONS[strategy_mode])

    # ── Signal & Safety (shared across signal-based strategies) ──
    signal_ticker = "QQQ"
    safe_assets = ["GLD"]
    risk_off_invested_pct = 0.0

    if strategy_mode != STRAT_MONTHLY:
        st.sidebar.markdown("#### Signal & Safety")
        col_sig, col_safe = st.sidebar.columns(2)
        with col_sig:
            default_signal = "^VIX" if strategy_mode == STRAT_VOL else "QQQ"
            label = "Vol Ticker" if strategy_mode == STRAT_VOL else "Signal Ticker"
            signal_ticker = st.text_input(
                label, default_signal,
                help="Ticker used to generate signals.",
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

    # ── Strategy-specific knobs (data-driven) ──
    param_values = dict(_PARAM_DEFAULTS)  # start from defaults
    params_spec = STRATEGY_PARAMS.get(strategy_mode, [])
    if params_spec:
        _render_strategy_params(params_spec, param_values)

    # VIX caption
    if strategy_mode == STRAT_VOL:
        st.sidebar.caption(
            f"Risk-Off when {signal_ticker} > {param_values['vix_threshold']}"
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
        "risk_off_invested_pct": risk_off_invested_pct,
        # Strategy-specific (always present, defaults for irrelevant strategies)
        "sma_window": param_values["sma_window"],
        "use_dual_momentum": param_values["use_dual_momentum"],
        "sma_fast": param_values["sma_fast"],
        "sma_slow": param_values["sma_slow"],
        "vix_threshold": param_values["vix_threshold"],
        "trailing_stop_pct": param_values["trailing_stop_pct"],
        "rsi_period": param_values["rsi_period"],
        "rsi_overbought": param_values["rsi_overbought"],
        "rsi_oversold": param_values["rsi_oversold"],
        "macd_fast": param_values["macd_fast"],
        "macd_slow": param_values["macd_slow"],
        "macd_signal": param_values["macd_signal"],
        "bb_period": param_values["bb_period"],
        "bb_std": param_values["bb_std"],
        "bb_squeeze": param_values["bb_squeeze"],
    }


def _render_strategy_params(
    params_spec: list[dict],
    values: dict,
) -> None:
    """Render strategy-specific widgets from a declarative spec list.

    Params that define a ``"column"`` key are grouped into a column layout.
    """
    # Separate column-grouped params from standalone params
    col_params: dict[int, list[dict]] = {}
    standalone: list[dict] = []
    for p in params_spec:
        if "column" in p:
            col_params.setdefault(p["column"], []).append(p)
        else:
            standalone.append(p)

    # Render column-grouped params first
    if col_params:
        n_cols = max(col_params.keys()) + 1
        cols = st.sidebar.columns(n_cols)
        for col_idx, param_list in col_params.items():
            for p in param_list:
                with cols[col_idx]:
                    values[p["key"]] = _render_single_widget(p)

    # Render standalone params
    for p in standalone:
        values[p["key"]] = _render_single_widget(p, sidebar=True)


def _render_single_widget(spec: dict, *, sidebar: bool = False) -> object:
    """Create a single Streamlit widget from a spec dict."""
    container = st.sidebar if sidebar else st
    widget_type = spec["widget"]
    common = {"key": f"strat_{spec['key']}"}
    if "help" in spec:
        common["help"] = spec["help"]

    if widget_type == "slider":
        raw = container.slider(
            spec["label"],
            min_value=spec["min"],
            max_value=spec["max"],
            value=spec["default"],
            step=spec["step"],
            **common,
        )
        return raw / spec["divisor"] if "divisor" in spec else raw

    if widget_type == "number_input":
        return container.number_input(
            spec["label"],
            min_value=spec.get("min"),
            max_value=spec.get("max"),
            value=spec["default"],
            step=spec.get("step"),
            **common,
        )

    if widget_type == "checkbox":
        return container.checkbox(
            spec["label"],
            value=spec["default"],
            **common,
        )

    return spec["default"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scenario Manager — dialog-based
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_scenario_manager() -> list[dict]:
    """Render scenario selection with checkboxes and dialog-action buttons.

    Returns:
        list of active scenario dicts: [{name, tickers, weights, expenses}, …]
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<h3 style="color:{_PRIMARY};margin-bottom:4px;">📂 Scenarios</h3>',
        unsafe_allow_html=True,
    )

    all_names = list(st.session_state.scenarios.keys())

    # Initialize active list
    if "active_scenarios" not in st.session_state:
        st.session_state.active_scenarios = [
            n for n in all_names
            if any(tag in n for tag in ("VOO", "TQQQ", "QLD"))
        ][:5]

    # Sync: remove stale names
    st.session_state.active_scenarios = [
        n for n in st.session_state.active_scenarios if n in all_names
    ]

    # ── Quick selection actions ──
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

    # ── Scrollable checkbox list ──
    _render_scenario_checkboxes(all_names)

    # ── Action buttons row → dialogs ──
    st.sidebar.markdown(
        f'<p style="color:{_TEXT};font-size:0.8rem;margin:8px 0 4px 0;">'
        f'Manage scenarios</p>',
        unsafe_allow_html=True,
    )
    act1, act2 = st.sidebar.columns(2)
    act3, act4 = st.sidebar.columns(2)

    with act1:
        if st.button("➕ New", use_container_width=True, key="btn_new_scenario"):
            create_scenario_dialog()
    with act2:
        if st.button("📋 Clone", use_container_width=True, key="btn_clone_scenario"):
            clone_scenario_dialog()
    with act3:
        if st.button("📦 Import / Export", use_container_width=True, key="btn_io"):
            import_export_dialog()
    with act4:
        if st.button(
            "🗑️ Delete", use_container_width=True, key="btn_delete_scenario",
        ):
            delete_confirmation_dialog()

    # ── Build selected scenario list ──
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


# ── Checkbox list ─────────────────────────────────────────────────────────────


def _render_scenario_checkboxes(all_names: list[str]) -> None:
    """Render a scrollable checkbox list with inline edit buttons."""
    with st.sidebar.container(height=300):
        for name in all_names:
            details = st.session_state.scenarios[name]
            tickers = details.get("tickers", [])
            weights = details.get("weights", [])
            summary = " · ".join(
                f"{t} {w:.0%}" for t, w in zip(tickers, weights)
            )

            col_cb, col_edit = st.columns([5, 1])

            with col_cb:
                is_active = name in st.session_state.active_scenarios
                new_val = st.checkbox(
                    name, value=is_active,
                    key=f"scen_cb_{name}",
                    help=summary,
                )
                if new_val and name not in st.session_state.active_scenarios:
                    st.session_state.active_scenarios.append(name)
                elif not new_val and name in st.session_state.active_scenarios:
                    st.session_state.active_scenarios.remove(name)

            with col_edit:
                if st.button("✏️", key=f"edit_{name}", help=f"Edit {name}"):
                    edit_scenario_dialog(name)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  @st.dialog – modal dialogs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@st.dialog("Create New Scenario", width="large")
def create_scenario_dialog() -> None:
    """Modal dialog for creating a brand-new scenario."""
    st.markdown("Define the tickers, weights, and expense ratios for your portfolio.")

    default_data = pd.DataFrame([
        {"Ticker": "AAPL", "Weight": 0.5, "Expense Ratio": 0.0},
        {"Ticker": "MSFT", "Weight": 0.5, "Expense Ratio": 0.0},
    ])

    edited_df = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        key="dlg_new_scenario_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
            "Weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            ),
            "Expense Ratio": st.column_config.NumberColumn(
                "Expense Ratio", min_value=0.0, max_value=0.1, step=0.001, format="%.4f",
            ),
        },
    )

    # Live weight validation
    total_w = edited_df["Weight"].astype(float).sum()
    is_valid = abs(total_w - 1.0) < 0.01
    colour = "green" if is_valid else "red"
    st.markdown(
        f"**Total weight:** :{colour}[**{total_w:.2f}**] "
        f"{'✅' if is_valid else '⚠️ must equal 1.0'}"
    )

    # Auto-generated name from tickers
    tickers_preview = [
        t.strip().upper()
        for t in edited_df["Ticker"].astype(str).tolist()
        if t.strip()
    ]
    auto_name = (
        " / ".join(f"{t} {w:.0%}" for t, w in zip(tickers_preview, edited_df["Weight"]))
        if tickers_preview
        else "My Portfolio"
    )

    new_name = st.text_input("Scenario Name", auto_name, key="dlg_new_scen_name")

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("💾 Save Scenario", use_container_width=True, type="primary"):
            c_tickers = [
                t.strip().upper()
                for t in edited_df["Ticker"].astype(str).tolist()
                if t.strip()
            ]
            c_weights = edited_df["Weight"].astype(float).tolist()
            c_expenses = edited_df["Expense Ratio"].astype(float).tolist()

            err = create_scenario(
                st.session_state.scenarios, new_name,
                c_tickers, c_weights, c_expenses,
            )
            if err:
                st.error(err)
            else:
                st.session_state.active_scenarios.append(new_name)
                st.toast(f"✅ Created **{new_name}**")
                st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Edit Scenario", width="large")
def edit_scenario_dialog(name: str) -> None:
    """Modal dialog for editing an existing scenario's composition."""
    if name not in st.session_state.scenarios:
        st.error(f"Scenario '{name}' not found.")
        return

    details = ensure_expenses(st.session_state.scenarios[name])

    st.markdown(f"Editing: **{name}**")

    edit_df = pd.DataFrame({
        "Ticker": details["tickers"],
        "Weight": details["weights"],
        "Expense Ratio": details["expenses"],
    })

    edited_df = st.data_editor(
        edit_df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"dlg_edit_{name}",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
            "Weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            ),
            "Expense Ratio": st.column_config.NumberColumn(
                "Expense Ratio", min_value=0.0, max_value=0.1, step=0.001, format="%.4f",
            ),
        },
    )

    # Live weight validation
    total_w = edited_df["Weight"].astype(float).sum()
    is_valid = abs(total_w - 1.0) < 0.01
    colour = "green" if is_valid else "red"
    st.markdown(
        f"**Total weight:** :{colour}[**{total_w:.2f}**] "
        f"{'✅' if is_valid else '⚠️ must equal 1.0'}"
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("💾 Save Changes", use_container_width=True, type="primary"):
            new_tickers = [
                t.strip().upper()
                for t in edited_df["Ticker"].astype(str).tolist()
                if t.strip()
            ]
            new_weights = edited_df["Weight"].astype(float).tolist()
            new_expenses = edited_df["Expense Ratio"].astype(float).tolist()

            if not new_tickers:
                st.error("Please add at least one ticker.")
            elif not is_valid:
                st.error(f"Weights must sum to 1.0 (current: {total_w:.2f}).")
            else:
                st.session_state.scenarios[name] = {
                    "tickers": new_tickers,
                    "weights": new_weights,
                    "expenses": new_expenses,
                }
                save_scenarios(st.session_state.scenarios)
                st.toast(f"✅ Updated **{name}**")
                st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Delete Scenarios")
def delete_confirmation_dialog() -> None:
    """Confirmation modal before deleting one or more scenarios."""
    all_names = list(st.session_state.scenarios.keys())
    if not all_names:
        st.info("No scenarios to delete.")
        return

    st.markdown("Select scenarios to **permanently delete**:")
    to_delete = st.multiselect(
        "Scenarios",
        options=all_names,
        key="dlg_delete_selector",
        label_visibility="collapsed",
    )

    if to_delete:
        st.warning(
            f"⚠️ This will permanently delete **{len(to_delete)}** scenario(s). "
            "This cannot be undone."
        )

    col_del, col_cancel = st.columns(2)
    with col_del:
        if st.button(
            f"🗑️ Delete {len(to_delete)}" if to_delete else "Select scenarios above",
            use_container_width=True,
            type="primary",
            disabled=not to_delete,
        ):
            for n in to_delete:
                delete_scenario(st.session_state.scenarios, n)
                if n in st.session_state.active_scenarios:
                    st.session_state.active_scenarios.remove(n)
            st.toast(f"Deleted: {', '.join(to_delete)}")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Clone Scenario")
def clone_scenario_dialog() -> None:
    """Modal for cloning a scenario with a new name."""
    all_names = list(st.session_state.scenarios.keys())
    if not all_names:
        st.info("No scenarios to clone.")
        return

    source = st.selectbox("Source Scenario", all_names, key="dlg_clone_source")
    if not source:
        return

    details = ensure_expenses(st.session_state.scenarios[source])

    # Preview
    st.caption(
        "**Contents:** "
        + " · ".join(
            f"{t} {w:.0%}" for t, w in zip(details["tickers"], details["weights"])
        )
    )

    clone_name = st.text_input(
        "New Name", f"{source} (copy)", key="dlg_clone_name",
    )

    col_clone, col_cancel = st.columns(2)
    with col_clone:
        if st.button("📋 Clone", use_container_width=True, type="primary"):
            err = create_scenario(
                st.session_state.scenarios,
                clone_name,
                list(details["tickers"]),
                list(details["weights"]),
                list(details["expenses"]),
            )
            if err:
                st.error(err)
            else:
                st.session_state.active_scenarios.append(clone_name)
                st.toast(f"✅ Cloned as **{clone_name}**")
                st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("Import / Export Scenarios", width="large")
def import_export_dialog() -> None:
    """Modal for importing and exporting scenarios as JSON."""
    tab_export, tab_import = st.tabs(["📤 Export", "📥 Import"])

    with tab_export:
        st.markdown("Download all scenarios as a JSON file.")
        json_str = export_scenarios(
            st.session_state.scenarios,
            list(st.session_state.scenarios.keys()),
        )
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name="scenarios_export.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("Preview JSON"):
            st.code(json_str, language="json")

    with tab_import:
        st.markdown("Upload a previously exported JSON file.")
        uploaded = st.file_uploader(
            "Upload JSON",
            type=["json"],
            key="dlg_import_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            with st.expander("Preview uploaded JSON"):
                st.code(content, language="json")
            if st.button("📥 Import Scenarios", use_container_width=True, type="primary"):
                imported, skipped = import_scenarios(
                    st.session_state.scenarios, content,
                )
                if imported > 0:
                    st.success(f"Imported {imported} scenario(s).")
                    st.rerun()
                elif skipped > 0:
                    st.warning("All scenarios already exist (skipped).")
                else:
                    st.error("Invalid JSON file.")
