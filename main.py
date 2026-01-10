import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# --- Page Config ---
st.set_page_config(page_title="Portfolio Backtester", layout="wide")

# --- Constants ---
SCENARIOS_FILE = "scenarios.json"

DEFAULT_EXPENSE_RATIOS = {
    'TQQQ': 0.0086, 'QLD': 0.0095, 'USD': 0.0095, 'GLD': 0.0040,
    'SOXL': 0.0076, 'SPXL': 0.0091, 'UPRO': 0.0091, 'VOO': 0.0003, 'VTI': 0.0003
}

# Initial seed data (tuples of tickers, weights)
INITIAL_SCENARIOS = {
    "1. USD 50/TQQQ 35/GLD 15": (['GLD', 'TQQQ', 'USD'], [0.15, 0.35, 0.50]),
    "2. SOXL 50/TQQQ 35/GLD 15": (['GLD', 'TQQQ', 'SOXL'], [0.15, 0.35, 0.50]),
    "3. USD 30/QLD 55/GLD 15": (['GLD', 'QLD', 'USD'], [0.15, 0.55, 0.30]),
    "4. USD 50/QLD 35/GLD 15": (['GLD', 'QLD', 'USD'], [0.15, 0.35, 0.50]),
    "5. USD 85% + GLD 15%": (['GLD', 'USD'], [0.15, 0.85]),
    "6. SOXL 85% + GLD 15%": (['GLD', 'SOXL'], [0.15, 0.85]),
    "7. TQQQ 85% + GLD 15%": (['GLD', 'TQQQ'], [0.15, 0.85]),
    "8. SPXL 85% + GLD 15%": (['GLD', 'SPXL'], [0.15, 0.85]),
    "9. QLD 85% + GLD 15%": (['GLD', 'QLD'], [0.15, 0.85]),
    "10. UPRO 85% + GLD 15%": (['GLD', 'UPRO'], [0.15, 0.85]),
    "11. VOO 85% + GLD 15%": (['GLD', 'VOO'], [0.15, 0.85]),
    "12. VOO 100%": (['VOO'], [1.00]),
    "13. VTI 100%": (['VTI'], [1.00]),
    "14. SOXL 100%": (['SOXL'], [1.00]),
}

# --- Persistence Functions ---

def load_scenarios():
    """Load scenarios from JSON file or initialize with defaults."""
    if os.path.exists(SCENARIOS_FILE):
        try:
            with open(SCENARIOS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading scenarios: {e}")
            return {}
    
    # Initialize defaults
    scenarios = {}
    for name, (tickers, weights) in INITIAL_SCENARIOS.items():
        expenses = [DEFAULT_EXPENSE_RATIOS.get(t, 0.0) for t in tickers]
        scenarios[name] = {
            "tickers": tickers,
            "weights": weights,
            "expenses": expenses
        }
    return scenarios

def save_scenarios(scenarios):
    """Save scenarios to JSON file."""
    try:
        with open(SCENARIOS_FILE, 'w') as f:
            json.dump(scenarios, f, indent=4)
    except Exception as e:
        st.error(f"Error saving scenarios: {e}")

# Initialize Session State
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = load_scenarios()

# --- Functions ---

@st.cache_data
def get_stock_data(tickers, start_date="2010-01-01"):
    """Download and cache stock data."""
    if not tickers:
        return pd.DataFrame()
    unique_tickers = list(set([t.upper() for t in tickers]))
    try:
        data = yf.download(unique_tickers, start=start_date, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

    if 'Close' in data.columns:
        return data['Close'].dropna()
    elif len(unique_tickers) == 1:
        return pd.DataFrame({unique_tickers[0]: data['Close']}).dropna() if 'Close' in data else data.dropna()
    return data.dropna() 

def calculate_metrics(history_series, daily_returns, total_invested, start_date, end_date):
    if history_series.empty:
        return {}
        
    start_val = history_series.iloc[0]
    end_val = history_series.iloc[-1]
    
    rf = 0.0

    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    negative_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252) if negative_returns.std() != 0 else 0

    volatility = daily_returns.std() * np.sqrt(252)

    roll_max = history_series.cummax()
    drawdown = (history_series - roll_max) / roll_max
    mdd = drawdown.min()
    
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        years = 1e-6 
        
    total_return_factor = end_val / total_invested
    cagr = (total_return_factor ** (1/years)) - 1
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    return {
        'final_value': end_val,
        'sharpe': sharpe,
        'sortino': sortino,
        'volatility': volatility,
        'mdd': mdd,
        'cagr': cagr,
        'calmar': calmar,
        'profit': end_val - total_invested,
        'total_invested': total_invested
    }

def run_backtest(scenario_name, tickers, weights, expenses, data, initial_capital, monthly_investment, inflation_rate, tax_threshold, strategy_mode, signal_series=None, safe_assets=None, risk_off_invested_pct=0.0):
    valid_tickers = [t for t in tickers if t in data.columns]
    if len(valid_tickers) != len(tickers):
        return None, None
        
    current_data = data[tickers].dropna()
    if current_data.empty:
        return None, None

    returns = current_data.pct_change().dropna()
    if returns.empty:
        return None, None
    
    weight_map = dict(zip(tickers, weights))
    expense_map = dict(zip(tickers, expenses))
    
    # Initialize assets and cash
    assets_value = {ticker: initial_capital * weight_map[ticker] for ticker in tickers}
    assets_cost_basis = {ticker: initial_capital * weight_map[ticker] for ticker in tickers}
    cash_balance = 0.0
    
    values_history = []
    daily_strategy_returns = []
    
    last_month = None
    last_year = returns.index[0].year
    annual_realized_gain = 0 
    current_monthly_inv = monthly_investment
    total_invested = initial_capital
    
    # Strategy State
    prev_signal_bull = True # Default assumption or need to check start? 
    if strategy_mode == "Trend Following (QQQ SMA)" and signal_series is not None:
         # Initial state based on first day data
         try:
             prev_signal_bull = signal_series.loc[returns.index[0]]
         except:
             prev_signal_bull = True

    for date in returns.index:
        prev_total = sum(assets_value.values()) + cash_balance
        
        if date.year != last_year:
            current_monthly_inv *= (1 + inflation_rate)
            annual_realized_gain = 0 
            last_year = date.year

        try:
            daily_ret = returns.loc[date]
        except KeyError:
            continue
            
        for ticker in tickers:
            assets_value[ticker] *= (1 + daily_ret[ticker])
            expense_ratio = expense_map.get(ticker, 0.0) 
            daily_expense = expense_ratio / 252
            assets_value[ticker] *= (1 - daily_expense)
        
        post_market_total = sum(assets_value.values()) + cash_balance
        
        strat_ret = (post_market_total - prev_total) / prev_total if prev_total != 0 else 0
        daily_strategy_returns.append(strat_ret)



        rebalance_needed = False
        current_signal_bull = True

        # Check Trend Signal Logic
        if strategy_mode == "Trend Following (QQQ SMA)" and signal_series is not None:
            try:
                # Check signal for TODAY (closing prices determines state for tomorrow? 
                # Or we rebalance AT CLOSE today? 
                # Standard backtest: Signal calculated on Yesterday's Close implies trade at Open.
                # Here we have daily data. We can assume we trade at Close based on Close signal 
                # (Slight lookahead bias if not careful, but standard for 1D signals).
                # To be practically executable: Trade Next Open. 
                # But for this script's daily resolution: Trade continuously or Close-to-Close changes.
                # Let's use signal at date. If Signal changed from Yesterday, Rebalance NOW.
                current_signal_bull = signal_series.loc[date]
                
                if current_signal_bull != prev_signal_bull:
                    rebalance_needed = True
            except KeyError:
                current_signal_bull = prev_signal_bull # Keep status quo if data missing
        
        # Check Monthly Rebalance Logic
        if last_month is not None and date.month != last_month:
            rebalance_needed = True
            total_invested += current_monthly_inv
            # Add cash immediately
            cash_balance += current_monthly_inv

        if rebalance_needed:
            current_total_equity = sum(assets_value.values()) + cash_balance
            
            # Determine Target Weights
            current_weights = {}
            
            if strategy_mode == "Trend Following (QQQ SMA)":
                 if current_signal_bull:
                      # RISK ON: Full Allocation
                      current_weights = weight_map.copy()
                 else:
                      # RISK OFF: Safe Assets Only
                      # We Keep Safe Assets at their original prescribed weight?
                      # Or do we scale them up?
                      # User says: "Keep the 15% GLD". Implies do not scale up.
                      current_weights = {}
                      for t in tickers:
                           if safe_assets and t in safe_assets:
                                # Safe assets stay at target
                                current_weights[t] = weight_map[t]
                           else:
                                # Risk assets reduce to risk_off_invested_pct of their target
                                current_weights[t] = weight_map[t] * risk_off_invested_pct
            else:
                 # Standard Rebalancing
                 current_weights = weight_map.copy()
            
            # Calculate target value for each asset
            target_values = {}
            for ticker in tickers:
                 target_values[ticker] = current_total_equity * current_weights.get(ticker, 0.0)
            
            # The remainder goes to cash
            allocated_value = sum(target_values.values())
            target_cash = current_total_equity - allocated_value
            
            # Execute Trades
            for ticker in tickers:
                target_val = target_values[ticker]
                current_val = assets_value[ticker]
                
                if current_val > target_val:
                    sell_amount = current_val - target_val
                    profit_ratio = (current_val - assets_cost_basis[ticker]) / current_val if current_val != 0 else 0
                    realized_gain = max(0, sell_amount * profit_ratio)
                    annual_realized_gain += realized_gain
                    
                    if annual_realized_gain > tax_threshold:
                         tax = realized_gain * 0.22 
                         sell_amount -= tax 
                
                assets_value[ticker] = target_val
                
                if current_val > 0:
                    assets_cost_basis[ticker] = assets_cost_basis[ticker] * (target_val / current_val)
                    if target_val > current_val:
                         cost_of_new = target_val - current_val
                         assets_cost_basis[ticker] += cost_of_new
                else: 
                     assets_cost_basis[ticker] = target_val

            cash_balance = target_cash
            
            # Update State
            prev_signal_bull = current_signal_bull
        
        last_month = date.month
        values_history.append(sum(assets_value.values()) + cash_balance)
    
    history_series = pd.Series(values_history, index=returns.index)
    strat_ret_series = pd.Series(daily_strategy_returns, index=returns.index)
    
    metrics = calculate_metrics(history_series, strat_ret_series, total_invested, returns.index[0], returns.index[-1])
    return history_series, metrics

# --- Sidebar Controls ---
st.sidebar.header("Settings")

initial_cash = st.sidebar.number_input("Initial Cash ($)", value=20000, step=1000)
monthly_cash = st.sidebar.number_input("Monthly Contribution ($)", value=1500, step=100)
tax_threshold = st.sidebar.number_input("Tax Free Threshold ($)", value=2000, step=500)
inflation_rate = st.sidebar.slider("Annual Inflation Rate (%)", 0.0, 10.0, 0.0, 0.1) / 100

st.sidebar.markdown("### Strategy")
strategy_mode = st.sidebar.selectbox(
    "Rebalancing Strategy", 
    ["Monthly Rebalancing", "Trend Following (QQQ SMA)"],
    help="Monthly Rebalancing: Restore target weights every month.\nTrend Following: Risk-Off to Cash/Safe Assets when Signal Ticker < 200 SMA."
)

signal_ticker = "QQQ"
safe_assets = ["GLD"]
sma_window = 200
use_dual_momentum = False
risk_off_invested_pct = 0.0

if strategy_mode == "Trend Following (QQQ SMA)":
    st.sidebar.markdown("#### Trend Settings")
    col_sig, col_safe = st.sidebar.columns([1, 1])
    with col_sig:
        signal_ticker = st.text_input("Signal Ticker", "QQQ").upper()
    with col_safe:
        safe_assets_str = st.text_input("Safe Assets", "GLD")
        safe_assets = [s.strip().upper() for s in safe_assets_str.split(",") if s.strip()]
    
    sma_window = st.sidebar.slider("SMA Window", 20, 300, 200, 10)
    
    col_dual, col_dummy = st.sidebar.columns([1,0.1])
    with col_dual:
        use_dual_momentum = st.sidebar.checkbox(
            "Dual Momentum Filter", 
            help="If checked, stay invested if Price > SMA OR 1-Month Return > 0 (reduces fake-outs)."
        )
        
    risk_off_invested_pct = st.sidebar.slider(
        "Risk-Off Invested %", 
        0, 100, 0, 10, 
        help="Percentage of risk assets to KEEP during downturns (Partial De-leveraging)."
    ) / 100.0

st.sidebar.markdown("---")

# --- Scenario Management via Drag & Drop ---
from streamlit_sortables import sort_items

st.sidebar.subheader("Scenarios")
st.sidebar.caption("Drag items between lists to Select, Store, or Delete.")

# Prepare lists for sortables
all_scenario_names = list(st.session_state.scenarios.keys())

# Initialize Sortable State if not present or if format is wrong (list of lists)
if "sortable_state" not in st.session_state or (st.session_state.sortable_state and isinstance(st.session_state.sortable_state[0], list)):
    # Default: Predefined popular ones in Active, others in Library
    active = []
    library = []
    for name in all_scenario_names:
        if ("VOO" in name or "TQQQ" in name) and "Custom" not in name:
            active.append(name)
        else:
            library.append(name)
    
    st.session_state.sortable_state = [
        {"header": "✅ Active (Backtest)", "items": active},
        {"header": "📚 Library", "items": library},
        {"header": "🗑️ Trash (Drop to Delete)", "items": []}
    ]

# Sync Sortable State with Persistence
current_active = st.session_state.sortable_state[0]["items"]
current_library = st.session_state.sortable_state[1]["items"]
try:
    current_trash = st.session_state.sortable_state[2]["items"]
except IndexError:
    current_trash = []
    st.session_state.sortable_state.append({"header": "🗑️ Trash (Drop to Delete)", "items": []})

current_known = set(current_active + current_library + current_trash)
new_items = [n for n in all_scenario_names if n not in current_known]

if new_items:
     # Add new to Active
    st.session_state.sortable_state[0]["items"].extend(new_items)

# Cleanup deleted items from lists if they were deleted externally (or init issues)
valid_names = set(all_scenario_names)
st.session_state.sortable_state[0]["items"] = [n for n in st.session_state.sortable_state[0]["items"] if n in valid_names]
st.session_state.sortable_state[1]["items"] = [n for n in st.session_state.sortable_state[1]["items"] if n in valid_names]
# Trash items are processed then deleted

# Render Sortables
with st.sidebar:
    sorted_data = sort_items(
        st.session_state.sortable_state,
        multi_containers=True,
        direction="vertical"
    )

# Process Logic from Sortables
# sorted_data returns the same structure: list of dicts
active_names = sorted_data[0]["items"]
library_names = sorted_data[1]["items"]
trash_names = sorted_data[2]["items"]

# Update Session State for next rerun
st.session_state.sortable_state = sorted_data
# Clear trash items from state after processing?
# Actually if we modify sorted_data[2]["items"] to empty, it might reset UI immediately.
# But we need to delete the underlying scenarios.

# Handle Deletion
if trash_names:
    for name in trash_names:
        if name in st.session_state.scenarios:
            del st.session_state.scenarios[name]
    save_scenarios(st.session_state.scenarios)
    st.toast(f"Deleted: {', '.join(trash_names)}")
    
    # Empty the trash in the state for next render
    st.session_state.sortable_state[2]["items"] = []
    st.rerun()

# Build Selected Scenarios based on "Active" order
selected_scenarios = []
all_tickers = set()

# Always add signal ticker if used
if strategy_mode == "Trend Following (QQQ SMA)":
    all_tickers.add(signal_ticker)

for name in active_names:
    if name in st.session_state.scenarios:
        details = st.session_state.scenarios[name]
        # Compatibility check
        if "expenses" not in details:
             details["expenses"] = [DEFAULT_EXPENSE_RATIOS.get(t, 0.0) for t in details["tickers"]]
             
        selected_scenarios.append({
            "name": name,
            "tickers": details["tickers"],
            "weights": details["weights"],
            "expenses": details["expenses"]
        })
        for t in details["tickers"]:
            all_tickers.add(t)

st.sidebar.markdown("---")
st.sidebar.subheader("New Scenario")

# Add New Scenario
with st.sidebar.expander("Create Custom", expanded=False):
    st.write("Define tickers, weights, and expenses.")
    
    default_data = pd.DataFrame(
        [
            {"Ticker": "AAPL", "Weight": 0.5, "Expense Ratio": 0.0},
            {"Ticker": "MSFT", "Weight": 0.5, "Expense Ratio": 0.0}
        ]
    )
    
    edited_df = st.data_editor(default_data, num_rows="dynamic", width="stretch")
    
    new_name = st.text_input("Name", "My Custom Portfolio")
    
    if st.button("Save to Active"):
        c_tickers = [t.strip().upper() for t in edited_df["Ticker"].astype(str).tolist() if t.strip()]
        c_weights = edited_df["Weight"].astype(float).tolist()
        c_expenses = edited_df["Expense Ratio"].astype(float).tolist()
        
        total_weight = sum(c_weights)
        
        if not c_tickers:
            st.error("Please add at least one ticker.")
        elif abs(total_weight - 1.0) > 0.01:
            st.error(f"Weights must sum to 1.0. Current: {total_weight:.2f}")
        elif new_name in st.session_state.scenarios:
            st.error("Name already exists.")
        else:
            st.session_state.scenarios[new_name] = {
                "tickers": c_tickers,
                "weights": c_weights,
                "expenses": c_expenses
            }
            save_scenarios(st.session_state.scenarios)
            # Add to sortable state active list immediately
            st.session_state.sortable_state[0].append(new_name)
            st.success(f"Saved!")
            st.rerun()

# --- Main Logic ---

st.title("ETF Portfolio Backtester")

if not selected_scenarios:
    st.info("Please select at least one scenario from the sidebar.")
else:
    with st.spinner("Downloading market data..."):
        full_stock_data = get_stock_data(list(all_tickers))

    if full_stock_data.empty:
        st.error("No data found for the selected tickers.")
    else:
        min_date = full_stock_data.index.min().date()
        max_date = full_stock_data.index.max().date()
        
        st.subheader("Analysis Period")
        date_range = st.slider(
            "Select Date",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        start_date, end_date = date_range

        filtered_stock_data = full_stock_data[
            (full_stock_data.index.date >= start_date) & 
            (full_stock_data.index.date <= end_date)
        ]

        if filtered_stock_data.empty:
            st.warning("No data available for selected date range.")
        else:
            # Calculate SMA data if needed
            signal_series = None
            if strategy_mode == "Trend Following (QQQ SMA)":
                 if signal_ticker in filtered_stock_data.columns:
                     ts = filtered_stock_data[signal_ticker]
                     sma = ts.rolling(window=sma_window).mean()
                     
                     sma_bull = ts > sma
                     
                     if use_dual_momentum:
                         # 21 trading days approx 1 month
                         mom_bull = ts > ts.shift(21)
                         signal_series = sma_bull | mom_bull
                     else:
                         signal_series = sma_bull
                 else:
                     st.warning(f"Signal ticker {signal_ticker} not found in data.")

            results = {}
            
            for scen in selected_scenarios:
                hist, mets = run_backtest(
                    scen['name'], 
                    scen['tickers'], 
                    scen['weights'], 
                    scen['expenses'],
                    filtered_stock_data, 
                    initial_cash, 
                    monthly_cash, 
                    inflation_rate, 
                    tax_threshold,
                    strategy_mode,
                    signal_series,
                    safe_assets,
                    risk_off_invested_pct
                )

                if hist is not None:
                    results[scen['name']] = {'history': hist, 'metrics': mets}

            if not results:
                st.warning("Not enough data to run backtest.")
            else:
                # --- Visualization ---
                
                # 1. Performance Chart
                fig = go.Figure()
                
                for name, res in results.items():
                    fig.add_trace(go.Scatter(
                        x=res['history'].index, 
                        y=res['history'], 
                        mode='lines', 
                        name=f"{name}"
                    ))
                    
                fig.update_layout(
                    title="Portfolio Performance (Log Scale)",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    yaxis_type="log",
                    template="plotly_dark",
                    hovermode="x unified",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # 2. Metrics Table
                st.subheader("Performance Metrics")
                
                metrics_data = []
                for name, res in results.items():
                    m = res['metrics']
                    metrics_data.append({
                        "Scenario": name,
                        "Final Value": f"${m['final_value']:,.0f}",
                        "CAGR": f"{m['cagr']:.2%}",
                        "Sharpe": f"{m['sharpe']:.2f}",
                        "Sortino": f"{m['sortino']:.2f}",
                        "Volatility": f"{m['volatility']:.2%}",
                        "Calmar": f"{m['calmar']:.2f}",
                        "Max Drawdown": f"{m['mdd']:.2%}",
                        "Total Invested": f"${m['total_invested']:,.0f}"
                    })
                    
                metrics_df = pd.DataFrame(metrics_data)
                
                st.dataframe(
                    metrics_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
                    }
                )

                # 3. Detailed Summary
                with st.expander("See Detailed Breakdown"):
                    st.write("Calculations assume monthly rebalancing to target weights. Taxes are estimated at 22% on realized gains over the threshold.")
                    st.write(f"Data range: {start_date} to {end_date}")


