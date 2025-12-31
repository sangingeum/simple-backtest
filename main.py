import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ETF별 운용 보수 (Annual Expense Ratio)
EXPENSE_RATIOS = {
    'TQQQ': 0.0086, 'QLD': 0.0095, 'USD': 0.0095, 'GLD': 0.0040,
    'SOXL': 0.0076, 'SPXL': 0.0091, 'UPRO': 0.0091, 'VOO': 0.0003
}

# --- 설정값 ---
initial_cash = 20000
monthly_cash = 1500
tax_level = 2000  # 연간 비과세 한도
inflation = 0.0

def backtest_portfolio(tickers, weights, initial_capital=20000, monthly_investment=1500, inflation_rate=0.0):
    # 1. 데이터 다운로드
    raw_data = yf.download(tickers, start="2010-01-01", auto_adjust=True)
    data = raw_data['Close'].dropna()
    returns = data.pct_change().dropna()
    
    # 초기화
    assets_value = {ticker: initial_capital * weight for ticker, weight in zip(tickers, weights)}
    assets_cost_basis = {ticker: initial_capital * weight for ticker, weight in zip(tickers, weights)}
    
    values_history = []
    last_month = None
    last_year = returns.index[0].year
    
    annual_realized_gain = 0 
    current_monthly_inv = monthly_investment

    for date in returns.index:
        if date.year != last_year:
            current_monthly_inv *= (1 + inflation_rate)
            annual_realized_gain = 0 
            last_year = date.year

        daily_ret = returns.loc[date]
        
        for ticker in tickers:
            # 수익률 반영
            assets_value[ticker] *= (1 + daily_ret[ticker])
            
            # --- 수정된 부분: 딕셔너리에 없는 티커는 0으로 처리 ---
            expense_ratio = EXPENSE_RATIOS.get(ticker, 0.0) 
            daily_expense = expense_ratio / 252
            assets_value[ticker] *= (1 - daily_expense)
        
        if last_month is not None and date.month != last_month:
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
                    
                    # $2,000(약 250만원) 초과분에 대해 22% 세금 적용
                    if annual_realized_gain > tax_level:
                        taxable_gain = max(0, realized_gain if (annual_realized_gain - realized_gain) > tax_level else annual_realized_gain - tax_level)
                        tax = taxable_gain * 0.22
                        assets_value[ticker] -= tax
                
                assets_value[ticker] = target_val
                assets_cost_basis[ticker] = target_val 
        
        last_month = date.month
        values_history.append(sum(assets_value.values()))
    
    return pd.Series(values_history, index=returns.index)



scenarios = {
    "1. USD 50/TQQQ 35/GLD 15": (['GLD', 'TQQQ', 'USD'], [0.15, 0.35, 0.50]),
    "2. QLD 55/USD 30/GLD 15": (['GLD', 'QLD', 'USD'], [0.15, 0.55, 0.30]),
    "3. USD 85% + GLD 15%": (['GLD', 'USD'], [0.15, 0.85]),
    "4. SOXL 85% + GLD 15%": (['GLD', 'SOXL'], [0.15, 0.85]),
    "5. TQQQ 85% + GLD 15%": (['GLD', 'TQQQ'], [0.15, 0.85]),
    "6. SPXL 85% + GLD 15%": (['GLD', 'SPXL'], [0.15, 0.85]),
    "7. QLD 85% + GLD 15%": (['GLD', 'QLD'], [0.15, 0.85]),
    "8. UPRO 85% + GLD 15%": (['GLD', 'UPRO'], [0.15, 0.85]),
    "9. VOO 85% + GLD 15%": (['GLD', 'VOO'], [0.15, 0.85]),
    "10. VOO 100": (['VOO'], [1.0]),
    "11. SOXL 100": (['SOXL'], [1.0]),
}

results = {}
for name, (tickers, weights) in scenarios.items():
    results[name] = backtest_portfolio(tickers, weights, initial_cash, monthly_cash, inflation)

# 결과 시각화
plt.figure(figsize=(14, 8))
for name, res in results.items():
    plt.plot(res, label=name, linewidth=2)

plt.yscale('log')
plt.title(f'Post-Tax Backtest (Initial: ${initial_cash}, Monthly: ${monthly_cash})', fontsize=15)
plt.ylabel('Portfolio Value (USD, Log Scale)')
plt.legend(loc='upper left')

# --- 수정된 부분: 에러 발생하던 shadow=True 제거 ---
plt.grid(True, which="both", ls="-", alpha=0.3) 

plt.show()

print(f"\n--- 세금/보수 차감 후 최종 결과 ---")
sorted_res = sorted(results.items(), key=lambda x: x[1][-1], reverse=True)
for name, res in sorted_res:
    print(f"{name:40}: ${res[-1]:,.0f}")