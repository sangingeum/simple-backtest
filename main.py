import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Portfolio Backtester", layout="wide")

# --- Constants ---
# Default ETF Annual Expense Ratio (used if not specified)
DEFAULT_EXPENSE_RATIOS = {
    'TQQQ': 0.0086, 'QLD': 0.0095, 'USD': 0.0095, 'GLD': 0.0040,
    'SOXL': 0.0076, 'SPXL': 0.0091, 'UPRO': 0.0091, 'VOO': 0.0003, 'VTI': 0.0003
}

DEFAULT_SCENARIOS = {
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

# --- Functions ---

@st.cache_data
def get_stock_data(tickers, start_date="2010-01-01"):
    """Download and cache stock data."""
    if not tickers:
        return pd.DataFrame()
    # Normalize tickers to upper case and remove duplicates
    unique_tickers = list(set([t.upper() for t in tickers]))
    try:
        data = yf.download(unique_tickers, start=start_date, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

    if 'Close' in data.columns:
        return data['Close'].dropna()
    elif len(unique_tickers) == 1:
        # If single ticker, yfinance might not return MultiIndex
        return pd.DataFrame({unique_tickers[0]: data['Close']}).dropna() if 'Close' in data else data.dropna()
    return data.dropna() 

def calculate_metrics(history_series, daily_returns, total_invested, start_date, end_date):
    if history_series.empty:
        return {}
        
    start_val = history_series.iloc[0]
    end_val = history_series.iloc[-1]
    
    # Risk Free Rate (assumed 0 for simplicity in this context, or we could use a constant)
    rf = 0.0

    # Sharpe Ratio (Annualized)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    # Sortino Ratio
    negative_returns = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252) if negative_returns.std() != 0 else 0

    # Volatility (Annualized)
    volatility = daily_returns.std() * np.sqrt(252)

    # Max Drawdown
    roll_max = history_series.cummax()
    drawdown = (history_series - roll_max) / roll_max
    mdd = drawdown.min()
    
    # CAGR (Money Weighted Approximation)
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        years = 1e-6 
        
    total_return_factor = end_val / total_invested
    cagr = (total_return_factor ** (1/years)) - 1
    
    # Calmar Ratio
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

def run_backtest(scenario_name, tickers, weights, expenses, data, initial_capital, monthly_investment, inflation_rate, tax_threshold):
    # Filter data for specific tickers and time range
    # data passed here is already filtered by date range in the main app logic
    
    # Ensure all tickers exist in data
    valid_tickers = [t for t in tickers if t in data.columns]
    if len(valid_tickers) != len(tickers):
        # We might have missing data for some tickers in the selected range
        # Only proceed if we have all tickers, otherwise allocation is messed up
        return None, None
        
    current_data = data[tickers].dropna()
    if current_data.empty:
        return None, None

    returns = current_data.pct_change().dropna()
    if returns.empty:
        return None, None
    
    # Normalize weights just in case, though validation handles it
    weight_map = dict(zip(tickers, weights))
    expense_map = dict(zip(tickers, expenses))
    
    assets_value = {ticker: initial_capital * weight_map[ticker] for ticker in tickers}
    assets_cost_basis = {ticker: initial_capital * weight_map[ticker] for ticker in tickers}
    
    values_history = []
    daily_strategy_returns = []
    
    last_month = None
    last_year = returns.index[0].year
    annual_realized_gain = 0 
    current_monthly_inv = monthly_investment
    total_invested = initial_capital

    for date in returns.index:
        prev_total = sum(assets_value.values())
        
        if date.year != last_year:
            current_monthly_inv *= (1 + inflation_rate)
            annual_realized_gain = 0 
            last_year = date.year

        try:
            daily_ret = returns.loc[date]
        except KeyError:
            continue
            
        # 1. Apply Market Movement and Expenses
        for ticker in tickers:
            assets_value[ticker] *= (1 + daily_ret[ticker])
            expense_ratio = expense_map.get(ticker, 0.0) 
            daily_expense = expense_ratio / 252
            assets_value[ticker] *= (1 - daily_expense)
        
        post_market_total = sum(assets_value.values())
        
        # 2. Record daily return
        strat_ret = (post_market_total - prev_total) / prev_total if prev_total != 0 else 0
        daily_strategy_returns.append(strat_ret)

        # 3. Monthly Rebalancing and New Investment
        if last_month is not None and date.month != last_month:
            total_invested += current_monthly_inv
            current_total = sum(assets_value.values())
            target_total = current_total + current_monthly_inv
            
            for ticker in tickers:
                weight = weight_map[ticker]
                target_val = target_total * weight
                current_val = assets_value[ticker]
                
                if current_val > target_val:
                    sell_amount = current_val - target_val
                    profit_ratio = (current_val - assets_cost_basis[ticker]) / current_val if current_val != 0 else 0
                    realized_gain = max(0, sell_amount * profit_ratio)
                    annual_realized_gain += realized_gain
                    
                    if annual_realized_gain > tax_threshold:
                        if annual_realized_gain > tax_threshold:
                             tax = realized_gain * 0.22 
                             assets_value[ticker] -= tax
                
                assets_value[ticker] = target_val
                assets_cost_basis[ticker] = target_val 
        
        last_month = date.month
        values_history.append(sum(assets_value.values()))
    
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

st.sidebar.markdown("---")
st.sidebar.subheader("Select Scenarios")

selected_scenarios = []
all_tickers = set()

# Predefined Scenarios
for name, (tickers, weights) in DEFAULT_SCENARIOS.items():
    if st.sidebar.checkbox(name, value=("VOO" in name or "TQQQ" in name)):
        # Use default expenses
        expenses = [DEFAULT_EXPENSE_RATIOS.get(t, 0.0) for t in tickers]
        selected_scenarios.append({
            "name": name,
            "tickers": tickers,
            "weights": weights,
            "expenses": expenses
        })
        for t in tickers:
            all_tickers.add(t)

# Custom Scenario Builder
st.sidebar.markdown("---")
with st.sidebar.expander("Custom Scenario"):
    st.write("Add tickers, weights, and annual expense ratios.")
    
    # Initialize session state for data editor if needed, or just let it process
    # We use a default dataframe structure
    default_data = pd.DataFrame(
        [
            {"Ticker": "AAPL", "Weight": 0.5, "Expense Ratio": 0.0},
            {"Ticker": "MSFT", "Weight": 0.5, "Expense Ratio": 0.0}
        ]
    )
    
    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
    
    custom_name = st.text_input("Scenario Name", "My Custom Portfolio")
    add_custom = st.button("Add Custom Scenario")
    
    if add_custom:
        c_tickers = [t.strip().upper() for t in edited_df["Ticker"].astype(str).tolist() if t.strip()]
        c_weights = edited_df["Weight"].astype(float).tolist()
        c_expenses = edited_df["Expense Ratio"].astype(float).tolist()
        
        total_weight = sum(c_weights)
        
        if not c_tickers:
            st.error("Please add at least one ticker.")
        elif abs(total_weight - 1.0) > 0.01:
            st.error(f"Total weight must be 1.0. Current sum: {total_weight:.2f}")
        else:
            selected_scenarios.append({
                "name": f"Custom: {custom_name}",
                "tickers": c_tickers,
                "weights": c_weights,
                "expenses": c_expenses
            })
            for t in c_tickers:
                all_tickers.add(t)
            st.success("Custom scenario added!")

# --- Main Logic ---

st.title("ETF Portfolio Backtester")

if not selected_scenarios:
    st.info("Please select at least one scenario from the sidebar.")
else:
    # Time Range Selector
    # Get available data range first? efficient approach: get all data, then filter.
    with st.spinner("Downloading market data..."):
        full_stock_data = get_stock_data(list(all_tickers))

    if full_stock_data.empty:
        st.error("No data found for the selected tickers.")
    else:
        min_date = full_stock_data.index.min().date()
        max_date = full_stock_data.index.max().date()
        
        st.subheader("Analysis Period")
        date_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        start_date, end_date = date_range

        # Filter Stock Data based on selection
        # Need to handle Timestamp vs Date comparison
        filtered_stock_data = full_stock_data[
            (full_stock_data.index.date >= start_date) & 
            (full_stock_data.index.date <= end_date)
        ]

        if filtered_stock_data.empty:
            st.warning("No data available for selected date range.")
        else:
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
                    tax_threshold
                )
                if hist is not None:
                    results[scen['name']] = {'history': hist, 'metrics': mets}

            if not results:
                st.warning("Not enough data to run backtest for the selected range/tickers.")
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
                
                # Styling the dataframe for better readability
                st.dataframe(
                    metrics_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
                    }
                )

                # 3. Detailed Summary
                with st.expander("See Detailed Breakdown"):
                    st.write("Calculations assume monthly rebalancing to target weights. Taxes are estimated at 22% on realized gains over the threshold.")
                    st.write(f"Data range: {start_date} to {end_date}")

