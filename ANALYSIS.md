# COMP6212 Coursework: Analytical Summary

## 1. Time Series Analysis with ARIMA

**Objective:** Implement ARIMA from first principles and apply to synthetic and real equity data for forecasting.

**Methodology:**
- **Part 1:** Built ARIMA(2,1,1) using only NumPy
  - First-order differencing (d=1)
  - AR(2) and MA(1) components with coefficients ϕ₁=7, ϕ₂=4, ψ₁=11
  - Forecasting via reversed integration
- **Part 2:** Applied to HSBC, AstraZeneca, and BP (Jan 2023-Jan 2024)
  - ADF test confirmed stationarity at d=1 for all series
  - ACF/PACF analysis identified optimal parameters:
    - HSBC: p=13, q=13
    - AstraZeneca: p=9, q=12 
    - BP: p=2, q=2
  - Hill-climbing optimization minimized rolling-window MSE
  - Evaluated on hold-out test set (33%) using RMSE/MAE

**Findings:**
- Synthetic forecast yielded extreme values (y₇ = -28, y₈ = -393), highlighting sensitivity to initial parameters
- BP's ARIMA(2,1,2) achieved RMSE of 7.19 GBX (~1.1% of asset value)
- Complex models (HSBC's ARIMA(13,1,13)) showed inferior out-of-sample performance

**Insight:** Model parsimony enhances robustness. Forecasting efficacy depends critically on parameter validation and error metric selection aligned with application context.

## 2. Algorithmic Trading: Pairs Trading

**Objective:** Compare mean-reversion (Z-score) against trend-following (TSMOM) strategies.

**Methodology:**
- Selected AZN.L/RKT.L pair based on pre-trade correlation (84%)
- Three Z-score variants:
  - Normalized price spread
  - Price ratio  
  - OLS hedge ratio spread
- TSMOM using dual moving averages on price ratio
- £1M capital, triggers at Z-score ±1 or MA crossovers

**Findings:**
- All Z-score strategies outperformed TSMOM
- OLS hedge ratio method achieved highest return (33%)
- TSMOM generated inferior signals for this pair/period

**Insight:** Spread definition quality drives strategy performance. Economically rigorous spread construction (OLS hedge ratio) outperforms simpler statistical alternatives.

## 3. Portfolio Optimization

**Objective:** Construct optimal portfolio via brute-force optimization versus 1/N benchmark.

**Methodology:**
- 10 FTSE stocks (2016-2017 training, 2018-2019 testing)
- 10M random portfolios with weights constrained 5-20%
- Maximized Sharpe ratio (Rf=4.25%) using sample moments
- Out-of-sample comparison against equal-weighted portfolio

**Findings:**
- Optimized portfolio superior in-sample (higher return, lower volatility)
- Underperformed 1/N out-of-sample (return: 0.000201 vs 0.000224)
- Maintained lower out-of-sample volatility (0.007609 vs 0.008190)

**Insight:** Historical moment estimation error dominates optimization benefits. Naive diversification provides robust out-of-sample performance despite theoretical suboptimality.

## 4. Stylised Facts Analysis

**Objective:** Validate heavy tails and volatility clustering in RR.L returns.

**Methodology:**
- 10 years daily data (2015-2025)
- Log returns analysis
- Excess kurtosis calculation
- ACF of returns and squared returns (50 lags)

**Findings:**
- Excess kurtosis = 16.9 (leptokurtic distribution)
- No significant ACF in raw returns
- Significant persistent ACF in squared returns

**Insight:** Empirical return properties invalidate normality and IID assumptions. Risk management requires models explicitly accounting for fat tails and volatility persistence (e.g., GARCH, non-Gaussian distributions).