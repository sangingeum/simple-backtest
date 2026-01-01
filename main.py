import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ETF Annual Expense Ratio
EXPENSE_RATIOS = {
    'TQQQ': 0.0086, 'QLD': 0.0095, 'USD': 0.0095, 'GLD': 0.0040,
    'SOXL': 0.0076, 'SPXL': 0.0091, 'UPRO': 0.0091, 'VOO': 0.0003, 'VTI': 0.0003
}

# --- Settings ---
initial_cash = 20000
monthly_cash = 1500
tax_level = 2000 
inflation = 0.0

def backtest_portfolio(tickers, weights, initial_capital=20000, monthly_investment=1500, inflation_rate=0.0):
    raw_data = yf.download(tickers, start="2010-01-01", auto_adjust=True)
    data = raw_data['Close'].dropna()
    returns = data.pct_change().dropna()
    
    assets_value = {ticker: initial_capital * weight for ticker, weight in zip(tickers, weights)}
    assets_cost_basis = {ticker: initial_capital * weight for ticker, weight in zip(tickers, weights)}
    
    values_history = []
    daily_strategy_returns = [] # To calculate Sharpe accurately
    
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

        daily_ret = returns.loc[date]
        
        # 1. Apply Market Movement and Expenses
        for ticker in tickers:
            assets_value[ticker] *= (1 + daily_ret[ticker])
            expense_ratio = EXPENSE_RATIOS.get(ticker, 0.0) 
            daily_expense = expense_ratio / 252
            assets_value[ticker] *= (1 - daily_expense)
        
        post_market_total = sum(assets_value.values())
        
        # 2. Record daily return (excluding the cash injection)
        strat_ret = (post_market_total - prev_total) / prev_total if prev_total != 0 else 0
        daily_strategy_returns.append(strat_ret)

        # 3. Monthly Rebalancing and New Investment
        if last_month is not None and date.month != last_month:
            total_invested += current_monthly_inv
            current_total = sum(assets_value.values())
            target_total = current_total + current_monthly_inv
            
            for ticker, weight in zip(tickers, weights):
                target_val = target_total * weight
                current_val = assets_value[ticker]
                
                if current_val > target_val:
                    sell_amount = current_val - target_val
                    profit_ratio = (current_val - assets_cost_basis[ticker]) / current_val
                    realized_gain = max(0, sell_amount * profit_ratio)
                    annual_realized_gain += realized_gain
                    
                    if annual_realized_gain > tax_level:
                        taxable_gain = max(0, realized_gain if (annual_realized_gain - realized_gain) > tax_level else annual_realized_gain - tax_level)
                        tax = taxable_gain * 0.22
                        assets_value[ticker] -= tax
                
                assets_value[ticker] = target_val
                assets_cost_basis[ticker] = target_val 
        
        last_month = date.month
        values_history.append(sum(assets_value.values()))
    
    # Calculate Metrics
    history_series = pd.Series(values_history, index=returns.index)
    strat_ret_series = pd.Series(daily_strategy_returns, index=returns.index)
    
    # Sharpe Ratio (Annualized)
    sharpe = (strat_ret_series.mean() / strat_ret_series.std()) * np.sqrt(252) if strat_ret_series.std() != 0 else 0
    
    # Max Drawdown
    roll_max = history_series.cummax()
    drawdown = (history_series - roll_max) / roll_max
    mdd = drawdown.min()
    
    # CAGR (Money Weighted Approximation)
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    total_return_factor = history_series.iloc[-1] / total_invested
    cagr = (total_return_factor ** (1/years)) - 1

    return {
        'history': history_series,
        'sharpe': sharpe,
        'mdd': mdd,
        'cagr': cagr,
        'final_value': history_series.iloc[-1],
        'total_invested': total_invested
    }

scenarios = {
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

results = {}
for name, (tickers, weights) in scenarios.items():
    results[name] = backtest_portfolio(tickers, weights, initial_cash, monthly_cash, inflation)

# --- Visualization ---
plt.figure(figsize=(14, 9))

# Create a color map to handle many scenarios
colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(results)))

for (name, res), color in zip(results.items(), colors):
    plt.plot(
        res['history'], 
        label=f"{name} (Sharpe: {res['sharpe']:.2f})", 
        linewidth=2, 
        color=color
    )

plt.yscale('log')
plt.title(f'Post-Tax Backtest (Log Scale)\nInitial: ${initial_cash}, Monthly: ${monthly_cash}', fontsize=15)
plt.ylabel('Portfolio Value (USD)')

# Place legend outside or use a tighter layout since there are many items
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout() # Prevents legend from being cut off
plt.show()

# Print Detailed Results
print(f"\n{'Scenario':<30} | {'Final Value':<12} | {'CAGR':<8} | {'Sharpe':<8} | {'MDD':<8}")
print("-" * 75)
sorted_res = sorted(results.items(), key=lambda x: x[1]['final_value'], reverse=True)
for name, res in sorted_res:
    print(f"{name:<30} | ${res['final_value']:>10,.0f} | {res['cagr']:>7.2%} | {res['sharpe']:>7.2f} | {res['mdd']:>7.2%}")