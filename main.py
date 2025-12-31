import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def backtest_portfolio(tickers, weights, initial_capital=20000, monthly_investment=1500):
    # 1. 데이터 다운로드
    raw_data = yf.download(tickers, start="2014-01-01", auto_adjust=True)
    # MultiIndex 대응: 종가 데이터만 추출
    data = raw_data['Close'].dropna()
    returns = data.pct_change().dropna()
    
    # 자산별 보유 금액 초기화 (딕셔너리)
    assets_value = {ticker: initial_capital * weight for ticker, weight in zip(tickers, weights)}
    values_history = []
    last_month = None

    # 2. 일별 시뮬레이션
    for date in returns.index:
        daily_ret = returns.loc[date]
        
        # 가치 업데이트 (당일 수익률 반영)
        for ticker in tickers:
            assets_value[ticker] *= (1 + daily_ret[ticker])
        
        # 3. 리밸런싱 및 월급 투입 (매월 첫 영업일)
        if last_month is not None and date.month != last_month:
            # 현재 총 자산 + 이번 달 투자금
            current_total = sum(assets_value.values()) + monthly_investment
            
            # 목표 비중(weights)에 맞춰 전 자산 재분배 (정확한 리밸런싱)
            for ticker, weight in zip(tickers, weights):
                assets_value[ticker] = current_total * weight
        
        last_month = date.month
        values_history.append(sum(assets_value.values()))
    
    return pd.Series(values_history, index=returns.index)

# 설정값
initial_cash = 20000
monthly_cash = 1500

# 시나리오 정의 (GLD를 포함한 티커 리스트와 비중 리스트의 순서를 반드시 맞춰야 함)
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
    results[name] = backtest_portfolio(tickers, weights, initial_cash, monthly_cash)

# 4. 결과 시각화
plt.figure(figsize=(14, 8))
for name, res in results.items():
    plt.plot(res, label=name, linewidth=2)

plt.yscale('log')
plt.title(f'Retirement Strategy Backtest (Initial: ${initial_cash}, Monthly: ${monthly_cash})', fontsize=15)
plt.ylabel('Portfolio Value (USD, Log Scale)')
plt.legend(loc='upper left')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.show()

# 최종 결과 출력
print(f"\n--- 최종 자산 결과 (2014-현재) ---")
sorted_res = sorted(results.items(), key=lambda x: x[1][-1], reverse=True)
for name, res in sorted_res:
    print(f"{name:40}: ${res[-1]:,.0f}")