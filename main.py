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

def run_backtest(scenario_name, tickers, weights, expenses, data, initial_capital, monthly_investment, inflation_rate, tax_threshold, strategy_mode, slippage_rate=0.0, commission_fee=0.0, tax_settlement_mode="Immediate", signal_series=None, safe_assets=None, risk_off_invested_pct=0.0):
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
    annual_realized_gain = 0.0
    pending_tax_liability = 0.0 # For Annual settlement
    
    current_monthly_inv = monthly_investment
    total_invested = initial_capital
    
    # CRITICAL: Shift signal by 1 day to trade based on PREVIOUS day's close (removes lookahead bias)
    effective_signal = signal_series.shift(1).fillna(True) if signal_series is not None else None
    
    # Track previous signal to detect changes. 
    # Initial state: Assume signal at index 0 (which is shifted) is the start.
    prev_signal_bull = True
    if effective_signal is not None:
         # Handle potential NaN at start due to shift
         try:
             prev_signal_bull = effective_signal.iloc[0]
         except:
             prev_signal_bull = True

    for date in returns.index:
        # --- Start of Day Processing ---
        
        # 1. Inflation Adjustment & Annual Tax Settlement (Yearly)
        if date.year != last_year:
            # Settle taxes annually if selected
            if tax_settlement_mode == "Annual" and pending_tax_liability > 0:
                cash_balance -= pending_tax_liability
                pending_tax_liability = 0.0
            
            current_monthly_inv *= (1 + inflation_rate)
            annual_realized_gain = 0.0 # Reset Tax Bucket after payment year
            last_year = date.year

        prev_total = sum(assets_value.values()) + cash_balance

        # 2. Market Movement & Expenses
        try:
            daily_ret = returns.loc[date]
        except KeyError:
            continue
            
        for ticker in tickers:
            assets_value[ticker] *= (1 + daily_ret[ticker])
            expense_ratio = expense_map.get(ticker, 0.0) 
            # Apply daily expense ratio
            assets_value[ticker] *= (1 - (expense_ratio / 252))
        
        # Calculate Equity BEFORE trading
        post_market_total = sum(assets_value.values()) + cash_balance
        
        strat_ret = (post_market_total - prev_total) / prev_total if prev_total != 0 else 0
        daily_strategy_returns.append(strat_ret)

        rebalance_needed = False
        
        # Get signal for TODAY based on YESTERDAY'S data (effective_signal)
        current_signal_bull = True
        if effective_signal is not None:
            try:
                current_signal_bull = effective_signal.loc[date]
                if current_signal_bull != prev_signal_bull:
                    rebalance_needed = True
            except KeyError:
                 current_signal_bull = prev_signal_bull

        # Need to rebalance IF: 
        # 1. Signal Change
        # 2. Monthly Rebalance Trigger
        
        is_monthly_trigger = (last_month is not None and date.month != last_month)
        
        if is_monthly_trigger:
            rebalance_needed = True
            total_invested += current_monthly_inv
            # Add monthly contribution to Cash first
            cash_balance += current_monthly_inv
            # Update equity with new cash
            post_market_total += current_monthly_inv 
            
        if rebalance_needed:
            # We work with 'current_total_equity' as the pool to distribute.
            # However, taxes and slip/commissions will REDUCE this equity during the trade ops.
            # So we perform calculations in a way that respects the cash outflow.
            
            # --- 1. Determine Target Allocations --- 
            # (Same logic as before)
            current_weights = {}
            
            if strategy_mode == "Monthly Rebalancing" or effective_signal is None:
                 current_weights = weight_map.copy()
            else:
                 if current_signal_bull:
                      # RISK ON
                      current_weights = weight_map.copy()
                 else:
                      # RISK OFF
                      risk_tickers = [t for t in tickers if (safe_assets is None) or (t not in safe_assets)]
                      safe_tickers = [t for t in tickers if (safe_assets is not None) and (t in safe_assets)]
                      
                      total_risk_w = sum([weight_map.get(t, 0.0) for t in risk_tickers])
                      total_safe_w = sum([weight_map.get(t, 0.0) for t in safe_tickers])
                      
                      vacated_weight = total_risk_w * (1.0 - risk_off_invested_pct)
                      
                      for t in tickers:
                          if t in safe_tickers:
                              share_of_safe = weight_map[t] / total_safe_w if total_safe_w > 0 else 0.0
                              current_weights[t] = weight_map[t] + (vacated_weight * share_of_safe)
                          else:
                              current_weights[t] = weight_map[t] * risk_off_invested_pct

            # --- 2. Calculate Initial Targets ---
            target_values = {}
            current_equity_for_calc = sum(assets_value.values()) + cash_balance
            
            for ticker in tickers:
                 target_values[ticker] = current_equity_for_calc * current_weights.get(ticker, 0.0)
            
            # --- 3. Execute Trades ---
            total_trans_cost = 0.0
            total_tax_paid_now = 0.0
            
            for ticker in tickers:
                target_val = target_values[ticker]
                current_val = assets_value[ticker]
                
                trade_diff = target_val - current_val
                
                # Slippage & Commission
                if abs(trade_diff) > 0.01: # Ignore micro-trades
                    # Slippage: Cost based on value traded
                    slip_cost = abs(trade_diff) * slippage_rate
                    # Commission: Flat fee per trade
                    comm_cost = commission_fee
                    
                    total_trans_cost += (slip_cost + comm_cost)
                
                # Tax Logic (Only on Sells)
                if trade_diff < 0: # SELLING
                    sell_amount = abs(trade_diff)
                    
                    avg_cost_basis = assets_cost_basis[ticker]
                    # Logic update: We need to reduce the stored cost basis proportional to the sell
                    pct_sold = sell_amount / current_val if current_val > 0 else 0
                    basis_sold = avg_cost_basis * pct_sold
                    
                    realized_gain = sell_amount - basis_sold
                    
                    # Update Actua Cost Basis for remaining
                    assets_cost_basis[ticker] -= basis_sold
                    
                    if realized_gain > 0:
                        # Tax Calculation: Marginal
                        # We have 'annual_realized_gain' (already booked).
                        
                        # Amount ALREADY over threshold
                        prev_taxable_gain = max(0, annual_realized_gain - tax_threshold)
                        
                        # New Total Gain
                        annual_realized_gain += realized_gain
                        
                        # New Taxable Gain
                        new_taxable_gain = max(0, annual_realized_gain - tax_threshold)
                        
                        # Tax to pay NOW
                        taxable_now = new_taxable_gain - prev_taxable_gain
                        if taxable_now > 0:
                            tax_on_trade = taxable_now * 0.22 # 22% rate
                            
                            if tax_settlement_mode == "Immediate":
                                total_tax_paid_now += tax_on_trade
                            else:
                                pending_tax_liability += tax_on_trade
                                
                    else:
                        # Loss netting
                        annual_realized_gain += realized_gain 

                elif trade_diff > 0: # BUYING
                    # Increase Cost Basis
                    assets_cost_basis[ticker] += trade_diff
                
                # UPDATE POSITION
                assets_value[ticker] = target_val

            # --- 4. Settle Costs ---
            # Deduct Transaction Costs + IMMEDIATE Taxes
            
            allocated_value = sum(assets_value.values()) 
            implied_cash = current_equity_for_calc - allocated_value
            
            final_cash = implied_cash - total_trans_cost - total_tax_paid_now
            
            # If Annual, pending_tax_liability stays internal until Year End
            if tax_settlement_mode == "Annual":
                # Check if we need to force liquidate cash? 
                # No, we just deduct from cash at end of year. 
                # If cash goes negative at year end, we are effectively using margin.
                pass
            
            cash_balance = final_cash
            
            # Update State
            prev_signal_bull = current_signal_bull
        
        last_month = date.month
        values_history.append(sum(assets_value.values()) + cash_balance - pending_tax_liability) 
        # Note: We subtract pending liability from daily valuation to be honest about Net Worth?
        # Or should we show gross? Net Worth usually implies liability subtraction.
        # Decision: Show Net Equity (Assets + Cash - Liabilities)

    
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
slippage_rate = st.sidebar.slider("Slippage (%)", 0.0, 5.0, 0.1, 0.1, help="Estimated price impact per trade.") / 100.0
commission_fee = st.sidebar.number_input("Commission per Trade ($)", value=0.0, step=1.0, help="Flat fee per ticker traded per rebalance.")
tax_settlement_mode = st.sidebar.selectbox("Tax Settlement", ["Immediate", "Annual"], help="'Immediate': Pay tax instantly on rebalance gains.\n'Annual': Defer payment until Dec 31st each year (allows compounding).")


    # Strategy Options
# Strategy Options
STRAT_MONTHLY = "Monthly Rebalancing"
STRAT_TREND = "Trend Following (SMA)"
STRAT_CROSS = "SMA Crossover (Golden Cross)"
STRAT_VOL = "Volatility Targeting (VIX)"
STRAT_TRAIL = "Trailing Stop (High Water Mark)"

strategy_descriptions = {
    STRAT_MONTHLY: "The 'Buy & Hold' benchmark. Rebalances to target weights on the 1st of every month regardless of price.",
    STRAT_TREND: "Risk-On only when Price > SMA (e.g., 200-day). This acts as a 'Circuit Breaker' for secular bear markets. **Pros:** Avoids major crashes like 2008. **Cons:** Can 'whipsaw' (sell low, buy high) during sideways markets.",
    STRAT_CROSS: "A 'Golden Cross' strategy. Risk-On when a Fast SMA (e.g., 50) is above a Slow SMA (e.g., 200). **Pros:** Filters out minor price 'noise' better than a single SMA. **Cons:** Even more lagging; you will miss more of the initial recovery gains.",
    STRAT_VOL: "Exits the market when the VIX (Fear Index) spikes above your threshold. **Pros:** Proactively exits before 'volatility decay' eats your leveraged ETF gains. **Cons:** Market panics are often short-lived, leading to unnecessary exits.",
    STRAT_TRAIL: "Exits if the signal ticker drops X% from its recent peak. Only re-enters when Price > SMA. **Pros:** Hard limit on capital loss. **Cons:** Requires a perfect 're-entry' setting. **Note:** Assumes 'Invested' at start of simulation."
}

strategy_mode = st.sidebar.selectbox(
    "Trading Strategy", 
    [STRAT_MONTHLY, STRAT_TREND, STRAT_CROSS, STRAT_VOL, STRAT_TRAIL],
    help="Select the logic determines when to be 'Risk-On' (Invested) or 'Risk-Off' (Defensive)."
)

# Show Strategy Explanation
st.sidebar.info(strategy_descriptions[strategy_mode])

signal_ticker = "QQQ"
safe_assets = ["GLD"]
sma_window = 200
risk_off_invested_pct = 0.0

# Strategy Parameters
st.sidebar.markdown("#### Settings")

if strategy_mode != STRAT_MONTHLY:
    col_sig, col_safe = st.sidebar.columns([1, 1])
    with col_sig:
        # For Volatility strategy, Signal Ticker is VIX implicitly, but maybe user wants another?
        # User text implies VIX. Let's auto-set/lock for Vol, or allow custom.
        if strategy_mode == STRAT_VOL:
            signal_ticker = st.text_input("Vol Ticker", "^VIX", help="Ticker used to measure volatility (e.g. ^VIX).").upper()
        else:
            signal_ticker = st.text_input("Signal Ticker", "QQQ", help="Ticker used to generate Buy/Sell signals.").upper()
    with col_safe:
        safe_assets_str = st.text_input("Safe Assets", "GLD", help="Comma-separated tickers to HOLD during Risk-Off periods (e.g. GLD, BIL).")
        safe_assets = [s.strip().upper() for s in safe_assets_str.split(",") if s.strip()]
        
    risk_off_invested_pct = st.sidebar.slider(
        "Risk-Off Invested %", 
        0, 100, 0, 10, 
        help="0% = Move all risk assets to Cash/Safe Assets.\n50% = Keep half of risk assets invested (Partial De-leveraging)."
    ) / 100.0

# Specific Parameters
sma_fast = 50
sma_slow = 200
vix_threshold = 30.0
trailing_stop_pct = 0.15
use_dual_momentum = False

if strategy_mode == STRAT_TREND:
    sma_window = st.sidebar.slider(
        "SMA Window", 20, 300, 200, 10, 
        help="Number of days for the Simple Moving Average. Price < SMA triggers Risk-Off."
    )
    use_dual_momentum = st.sidebar.checkbox(
        "Dual Momentum Filter", 
        value=False,
        help="If checked, strategy stays Risk-On if 1-Month Return is Positive, even if Price < SMA. Prevents fake-outs."
    )
    
elif strategy_mode == STRAT_CROSS:
    c1, c2 = st.sidebar.columns(2)
    sma_fast = c1.number_input("Fast SMA", 10, 200, 50, help="Shorter moving average window.")
    sma_slow = c2.number_input("Slow SMA", 50, 400, 200, help="Longer moving average window.")
    
elif strategy_mode == STRAT_VOL:
    vix_threshold = st.sidebar.number_input(
        "VIX Threshold", 10.0, 100.0, 30.0, step=1.0,
        help="If VIX > this value, the strategy goes Risk-Off."
    )
    st.caption(f"Risk-Off when {signal_ticker} > {vix_threshold}")
    
elif strategy_mode == STRAT_TRAIL:
    trailing_stop_pct = st.sidebar.slider(
        "Stop Loss %", 5, 50, 15, 1,
        help="Exit trigger: % drop from the highest price seen during the trade."
    ) / 100.0
    sma_window = st.sidebar.slider(
        "Re-Entry SMA", 20, 300, 200, 10,
        help="Price must be ABOVE this SMA to re-enter the market after a Stop Loss exit."
    )


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
if strategy_mode != STRAT_MONTHLY:
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
            # Calculate Signal Series
            signal_series = None
            
            if strategy_mode != STRAT_MONTHLY:
                 if signal_ticker in filtered_stock_data.columns:
                     ts = filtered_stock_data[signal_ticker]
                     
                     if strategy_mode == STRAT_TREND:
                         sma = ts.rolling(window=sma_window).mean()
                         bullish = ts > sma
                         if use_dual_momentum:
                             mom = ts > ts.shift(21)
                             signal_series = bullish & mom
                         else:
                             signal_series = bullish
                             
                     elif strategy_mode == STRAT_CROSS:
                         fast = ts.rolling(window=sma_fast).mean()
                         slow = ts.rolling(window=sma_slow).mean()
                         signal_series = fast > slow
                         
                     elif strategy_mode == STRAT_VOL:
                         # Risk OFF if VIX > Threshold
                         # Signal Series is "Are we in Bull Mode?"
                         # Bull Mode = VIX < Threshold
                         signal_series = ts < vix_threshold
                         
                     elif strategy_mode == STRAT_TRAIL:
                         # State-based generation
                         # Needs loop
                         sma = ts.rolling(window=sma_window).mean()
                         signals = []
                         is_invested = True # Assume started invested? Or check SMA?
                         # Let's seed initial state with SMA check
                         if len(ts) > 0 and len(sma.dropna()) > 0:
                             # Find first valid index for SMA
                             first_valid = sma.first_valid_index()
                             if first_valid:
                                 # Before valid SMA, assume invested? Or not? 
                                 # Let's default True.
                                 pass
                                 
                         hwm = ts.iloc[0]
                         
                         signals = pd.Series(index=ts.index, dtype=bool)
                         signals.iloc[:] = True # Default True
                         
                         # Slow loop but necessary for state dependency
                         # Optimization: Vectorized HWM is easy, but the "Wait for SMA" reset is the tricky part.
                         # Logic:
                         # 1. Update HWM.
                         # 2. Check Drawdown. If DD < -StopPct -> Signal = False (Exit).
                         # 3. If Signal is False: Check Re-entry (Price > SMA). If True -> Signal = True (Enter). Reset HWM? 
                         #    Usually Trailing Stop resets HWM upon re-entry or just tracks Price.
                         
                         current_state = True # Bull
                         current_hwm = ts.iloc[0]
                         
                         signal_list = []
                         
                         for date, price in ts.items():
                             # Get SMA
                             try:
                                 sma_val = sma.at[date]
                             except:
                                 sma_val = np.nan
                             
                             if current_state: # Currently Invested
                                 if price > current_hwm:
                                     current_hwm = price
                                 
                                 dd = (price - current_hwm) / current_hwm if current_hwm > 0 else 0
                                 if dd < -trailing_stop_pct:
                                     current_state = False # Stop Out
                             else: # Currently in Cash
                                 # Check Re-entry
                                 if not np.isnan(sma_val) and price > sma_val:
                                     current_state = True
                                     current_hwm = price # Reset HWM to current price on re-entry
                             
                             signal_list.append(current_state)
                             
                         signal_series = pd.Series(signal_list, index=ts.index)

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
                    slippage_rate,
                    commission_fee,
                    tax_settlement_mode,
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


