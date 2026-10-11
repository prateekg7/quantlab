# Empirical Results & Research Report

All numerical values, tables, and charts in this report are fully reproducible and generated directly from the test suite and benchmark scripts in `benchmarks/run_experiments.py` and `benchmarks/plot_mc_convergence.py`.

---

## 1. Executive Summary & Honest Quantitative Verdict

A moving-average crossover strategy (e.g. 50/200 SMA) is often cited as a beginner's trend-following baseline. Across a comprehensive 15-year empirical evaluation (2010–2024) spanning US equities, mega-cap tech, commodities, and long-term Treasuries, we find:

1. **Volatility Reduction vs. Return Drag:**
   The crossover strategy successfully lowered annualized volatility across all evaluated assets (by 3% to 4% annualized). However, in equities (`SPY` and `AAPL`), it significantly lagged Buy & Hold on total return and Sharpe ratio.
2. **In-Sample Overfitting is Real:**
   When optimizing parameters on in-sample data (2010–2018), the winning parameters (Fast=50, Slow=150) achieved a higher Sharpe ratio than Buy & Hold (0.88 vs 0.80). However, out-of-sample (2019–2024), the strategy's Sharpe ratio degraded to 0.80, while Buy & Hold achieved 0.90.
3. **Regime Dependence:**
   The strategy provided dramatic downside protection in secular bear markets (e.g., cutting `TLT` max drawdown from -48.35% to -25.21% during the 2020–2023 bond collapse). Conversely, in fast, V-shaped equity crashes (such as COVID in March 2020), execution lag forced the strategy to absorb the initial crash and miss the sharp initial rebound.

---

## 2. In-Sample vs. Out-of-Sample Backtest (SPY)

### Protocol
- **Dataset:** SPY daily adjusted close prices (2010-01-04 to 2024-12-30).
- **Split Date:** `2018-12-31`.
  - **In-Sample (Train):** 2010–2018 (9 years, 2,263 trading days).
  - **Out-of-Sample (Test):** 2019–2024 (6 years, 1,510 trading days).
- **Parameter Grid:** Fast $\in \{5, 10, 20, 30, 50\}$, Slow $\in \{50, 100, 150, 200\}$ with $\text{fast} < \text{slow}$.
- **Optimization Criterion:** Maximum Sharpe ratio net of transaction costs on in-sample data only.
- **Out-of-Sample Evaluation:** Evaluated exactly once on the test period using the winning in-sample parameters.

### Winning Parameters
- **In-sample Winner:** Fast = **50**, Slow = **150**.

### Performance Table

| Period | Metric | Strategy (5 bps cost) | Strategy (0 bps cost) | Buy & Hold |
|---|---|---|---|---|
| **In-Sample** (2010–2018) | **CAGR** | +9.41% | +9.48% | **+11.43%** |
| | **Annualized Vol** | **10.91%** | **10.91%** | 14.91% |
| | **Sharpe Ratio** | **0.88** | **0.88** | 0.80 |
| | **Max Drawdown** | **-17.19%** | **-17.02%** | -19.35% |
| | **Trades / Total Cost**| 12 trades / 0.60% | 12 trades / 0.00% | 1 trade / 0.00% |
| **Out-of-Sample** (2019–2024) | **CAGR** | +12.20% | +12.27% | **+17.20%** |
| | **Annualized Vol** | **16.03%** | **16.03%** | 19.84% |
| | **Sharpe Ratio** | 0.80 | 0.80 | **0.90** |
| | **Max Drawdown** | -33.72% | -33.72% | -33.72% |
| | **Trades / Total Cost**| 7 trades / 0.35% | 7 trades / 0.00% | 1 trade / 0.00% |

### In-Sample vs. Out-of-Sample Observations
- **Sharpe Degradation:** In-sample, the strategy achieved an apparent edge (+0.08 higher Sharpe than Buy & Hold). Out-of-sample, this edge completely vanished (-0.10 lower Sharpe than Buy & Hold).
- **Cost Impact:** With a low-turnover strategy (12 trades over 9 years, 7 trades over 6 years), transaction costs of 5 bps reduced annualized CAGR by only ~0.07% per year. Transaction costs were not the primary driver of underperformance; strategy lag was.

---

## 3. Multi-Asset Comparison (2010–2024, 5 bps Cost)

We evaluated the SMA 50/200 strategy across 4 asset classes:
1. `SPY` (Broad US Equity Market)
2. `AAPL` (Mega-Cap Momentum & Growth Equity)
3. `GLD` (Commodity / Store of Value)
4. `TLT` (Long-Term US Treasury Bonds)

| Ticker | Asset Class | Strategy CAGR | B&H CAGR | Strategy Vol | B&H Vol | Strategy Sharpe | B&H Sharpe | Strategy Max DD | B&H Max DD | Trades |
|---|---|---|---|---|---|---|---|---|---|---|
| **SPY** | US Equities | +9.51% | **+13.70%** | **13.67%** | 17.05% | 0.73 | **0.84** | -33.72% | -33.72% | 13 |
| **AAPL** | Mega-Cap Tech | +19.41% | **+27.76%** | **23.46%** | 27.87% | 0.87 | **1.02** | -45.67% | **-43.80%** | 13 |
| **GLD** | Commodities | +2.38% | **+5.38%** | **11.58%** | 15.50% | 0.26 | **0.42** | **-35.46%** | -45.56% | 23 |
| **TLT** | Treasuries | +2.29% | **+2.66%** | **11.49%** | 15.29% | **0.25** | **0.25** | **-25.21%** | -48.35% | 26 |

### Cross-Asset Insights
- **Bonds (`TLT`):** SMA trend-following delivered substantial risk mitigation. During the historic 2020–2023 rate hiking cycle, `TLT` suffered a devastating -48.35% drawdown. The SMA strategy cut this peak-to-trough decline almost in half (-25.21%) by rotating into cash as yields rose.
- **Equities (`SPY`, `AAPL`):** Trend-following severely underperformed buy-and-hold during secular bull markets. Because equity returns exhibit positive drift over multi-year periods, the cash drag from staying out during sideways/bearish periods was higher than the avoided downside losses.

---

## 4. Monte Carlo Pricing & Variance Reduction

European call option benchmark ($S=100, K=100, r=5\%, \sigma=20\%, T=1\text{ yr}, q=0$).
Analytical Black-Scholes price: **10.4506**.

Simulated with $N = 100,000$ paths (seed = 42):

| Estimator Method | Price Estimate | Standard Error | 95% Confidence Interval | Variance Reduction |
|---|---|---|---|---|
| **Black-Scholes (Analytical)** | **10.4506** | — | — | — |
| **Plain Monte Carlo** | 10.4205 | 0.04677 | [10.3289, 10.5122] | Baseline |
| **Antithetic Variates (F-9)** | 10.4673 | 0.03308 | [10.4025, 10.5322] | **50.0%** |
| **Control Variates (F-10)** | 10.4668 | 0.01788 | [10.4318, 10.5019] | **85.4%** |
| **Combined (Antithetic + CV)** | **10.4409** | **0.00880** | [10.4236, 10.4581] | **96.5%** |

### Convergence Rate Verification (T-18)
- A log-log regression of standard error vs. path count $N \in [10^2, 10^6]$ measured an empirical slope of **$-0.4751$**, consistent with the theoretical $O(N^{-1/2})$ rate predicted by the Central Limit Theorem.
- Visual convergence plot saved at `docs/mc_convergence.png`.

---

## 5. Structural Limitations of Simple Trend-Following

1. **Whipsaw & Execution Lag:**
   Moving averages are low-pass filters: they introduce phase lag proportional to the window length. When markets turn violently (V-shaped bottoms), the strategy sells near the bottom and buys back after the rebound has already occurred.
2. **Cash Drag in Secular Bull Markets:**
   In long upward trends, an uninvested cash position yields 0% nominal return in this engine, creating an insurmountable performance deficit relative to 100% equity exposure.
3. **Overfitting Sensitivity:**
   Parameter optimization on in-sample data invariably overfits historical cycle lengths. The best training window rarely sustains its Sharpe advantage out-of-sample.
