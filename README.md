# QuantLab

[![CI](https://github.com/prateekg7/quantlab/actions/workflows/ci.yml/badge.svg)](https://github.com/prateekg7/quantlab/actions/workflows/ci.yml)

A quantitative finance library implemented from scratch in Python, emphasizing mathematical rigor, clean software engineering, and zero external pricing/backtesting library dependencies.

---

## Features

- **Options Pricing & Greeks (`quantlab.pricing`)**:
  - Analytical Black-Scholes-Merton pricer for European calls and puts with continuous dividend yield $q$.
  - Exact put-call parity compliance ($C - P = S e^{-qT} - K e^{-rT}$).
- **Backtesting Engine (`quantlab.backtest`)**:
  - Vectorized signal generation (e.g., dual Simple Moving Average crossover).
  - Explicit execution lag modeling ($p_t$ determined by information up to $t-1$) preventing look-ahead bias.
  - Linear turnover and transaction cost model in basis points ($c_{\text{bps}}$).
  - Frozen `BacktestResult` dataclass holding full equity curves, daily returns, trade count, and cumulative costs.
- **Risk & Performance Metrics (`quantlab.metrics`)**:
  - Simple & log returns, CAGR, sample annualized volatility ($\text{ddof}=1$), Sharpe ratio (annualized via $\sqrt{252}$), and maximum drawdown.
- **Data Loaders (`quantlab.data`)**:
  - CSV price series ingestion with validation (date parsing, duplicate checks, monotonicity).

---

## Repository Structure

```
quantlab/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow (lint, type check, pytest)
├── pyproject.toml              # Build config, dependencies, ruff, mypy, pytest settings
├── README.md                   # Documentation, install instructions, quickstart
├── PRD.md                      # Product requirements document
├── SPEC.md                     # Mathematical specifications & conventions
├── TASKS.md                    # Task tracker
├── LEARNING.md                 # Quant curriculum & study stages
├── src/quantlab/               # Package root
│   ├── data.py                 # Price data loader
│   ├── metrics.py              # Financial statistics & risk metrics
│   ├── pricing/                # Options pricing models
│   │   ├── black_scholes.py    # Analytical Black-Scholes pricer
│   │   ├── monte_carlo.py      # Monte Carlo pricer (with antithetic variates)
│   │   ├── binomial.py         # Binomial lattice pricer
│   │   └── greeks.py           # Analytical & finite-difference Greeks
│   ├── backtest/               # Backtesting engine
│   │   ├── signals.py          # Trading signal generators
│   │   └── engine.py           # Backtest execution & accounting
│   └── simulation/             # Discrete delta-hedging simulator
│       └── delta_hedge.py
├── tests/                      # Pytest suite with property and boundary tests
├── notebooks/                  # Visualization & research notebooks (imports only)
│   └── 01_first_backtest.ipynb
├── benchmarks/                 # Numerical timing & optimization benchmarks
└── docs/                       # Research logs, devlog, and plots
    ├── devlog.md
    ├── results.md
    ├── equity_curve.png
    └── drawdown.png
```

---

## Installation

### Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/prateekg7/quantlab.git
cd quantlab

# Install dependencies and project in editable mode
uv sync
```

### Using standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Analytical Options Pricing (Black-Scholes)

```python
from quantlab.pricing.black_scholes import black_scholes

# Price a European call option (S=100, K=100, r=5%, vol=20%, T=1 yr)
call_price = black_scholes(S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, is_call=True)
print(f"European Call Price: {call_price:.4f}")  # 10.4506

# Price the corresponding European put option
put_price = black_scholes(S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, is_call=False)
print(f"European Put Price: {put_price:.4f}")  # 5.5735
```

### 2. Backtesting an SMA Crossover Strategy

```python
import yfinance as yf
from quantlab.backtest.engine import run_backtest

# Download historical SPY prices
raw = yf.download("SPY", start="2010-01-01", end="2024-12-31", auto_adjust=True, progress=False)
prices = raw["Close"].squeeze().dropna()

# Run 50/200 SMA crossover backtest with 5 bps one-way cost
result = run_backtest(prices, fast=50, slow=200, cost_bps=5.0)

print(f"Strategy CAGR:        {result.metrics['cagr']:+.2%}")
print(f"Annualized Vol:       {result.metrics['vol']:.2%}")
print(f"Sharpe Ratio:         {result.metrics['sharpe']:.2f}")
print(f"Max Drawdown:         {result.metrics['max_drawdown']:.2%}")
print(f"Total Trades:         {result.n_trades}")
print(f"Cumulative Costs:     {result.metrics['total_cost']:.2%}")
```

#### Example Output

```text
Strategy CAGR:        +9.51%
Annualized Vol:       13.67%
Sharpe Ratio:         0.73
Max Drawdown:         -33.72%
Total Trades:         13
Cumulative Costs:     0.65%
```

---

## Benchmark Results: SPY (2010–2024)

Backtest comparison between the 50/200 SMA trend-following crossover strategy and Buy & Hold on SPY:

| Metric | SMA 50/200 (5 bps cost) | Buy & Hold | Difference |
|---|---|---|---|
| **CAGR** | +9.51% | +13.70% | -4.19% |
| **Annualized Volatility** | 13.67% | 17.05% | -3.38% (Lower risk) |
| **Sharpe Ratio** | 0.73 | 0.84 | -0.11 |
| **Max Drawdown** | -33.72% | -33.72% | Identical (COVID crash 2020) |
| **Total Trades** | 13 | 1 | 12 additional rebalances |
| **Cumulative Costs** | 0.65% | 0.00% | -0.65% drag |

### Strategy Takeaway & Findings
The 50/200 SMA strategy achieved its intended purpose of **reducing portfolio volatility** from 17.05% to 13.67% by rotating into cash during prolonged equity bear markets. However:
1. **Lag Cost:** Trend-following signals lag turning points. In sharp V-shaped recoveries (such as March–April 2020), the strategy incurred significant drawdown before exiting, and missed the initial sharp rebound before the fast SMA crossed back above the slow SMA.
2. **Risk-Adjusted Underperformance:** Over the secular 2010–2024 bull market, avoiding drawdowns did not compensate for the missed market returns, resulting in a lower Sharpe ratio (0.73 vs 0.84).

---

## Limitations

1. **Option Pricing Assumptions:**
   - The Black-Scholes model assumes constant volatility $\sigma$, constant risk-free rate $r$, and continuous log-normal price dynamics (GBM). It does not capture volatility smiles/skews, fat-tailed kurtosis, or jump risks.
   - Prices apply strictly to European options (no early exercise premium).
2. **Backtesting Scope:**
   - **Execution Model:** Assumes execution at daily closing prices with 1-day lag. Intraday price paths, order routing latency, and slippage beyond fixed bps are not modeled.
   - **Market Impact & Liquidity:** Assumes infinite liquidity with no market impact.
   - **Asset Universe:** Currently evaluates single-asset strategies without cross-asset allocation, cash yield, or short borrow fees.
3. **Educational Intent:**
   - Designed for algorithmic transparency and quantitative finance learning; not intended for live production execution.

---

## Development & Testing

All contributions must pass linting, static type checking, and unit tests:

```bash
# Run pytest with coverage
uv run pytest -q --cov=quantlab

# Lint with Ruff
uv run ruff check .

# Check formatting with Ruff
uv run ruff format --check .

# Type check with mypy
uv run mypy src tests
```

---

## License

MIT
