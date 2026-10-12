T-7

Risk-neutral probability of finishing in the money

$N(d_2)=\mathbb Q(S_T>K)$

This interpretation assumes the standard Black–Scholes model with continuous dividend yield. Consequently, $Ke^{-rT}N(d_2)$ is the discounted expected strike payment for a call.

## T-13: SMA Crossover vs Buy & Hold Reflection

The most surprising result is that the SMA 50/200 strategy reduced annualized volatility from 17.05% to 13.67%, but it also delivered lower returns, with a CAGR of 9.51% compared to 13.70% for buy-and-hold. Despite this reduction in volatility, its Sharpe ratio was also lower (0.73 vs. 0.84), suggesting that the strategy did not generate better returns per unit of total risk. Both strategies reported the same maximum drawdown of -33.72%, which is particularly surprising given the SMA strategy's lower volatility.

Overall, the SMA strategy did not beat buy-and-hold on a risk-adjusted basis. It traded lower volatility for lower returns without improving the Sharpe ratio or maximum drawdown. This suggests that, over the 2010–2024 period, the crossover strategy's downside protection was not sufficient to compensate for missed market gains and trend-following delays.

## T-14: Package Refactor (`src/quantlab/` layout)

Migrated to standard `src/` layout per PRD Section 7 and Python packaging best practices:
- Root build backend configured via `hatchling` in `pyproject.toml` pointing packages to `src/quantlab`.
- Modules modularized under `quantlab.pricing`, `quantlab.backtest`, `quantlab.simulation`, `quantlab.metrics`, and `quantlab.data`.
- Validated editable install (`pip install -e .`) and seamless package imports across the entire test suite.

## T-15: GitHub Actions Continuous Integration

Configured `.github/workflows/ci.yml` using `astral-sh/setup-uv` for caching and fast dependency resolution:
- Automatically runs `ruff check`, `ruff format --check`, `mypy src tests`, and `pytest --cov=quantlab` on every push and PR to `main`.
- Added the dynamic CI status badge to `README.md`.

## T-16: Plain Monte Carlo Pricing (`mc_price`, `MCResult`)

Implemented exact single-step terminal simulation under risk-neutral GBM:
$$S_T = S \exp\left( \left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z \right), \quad Z \sim \mathcal{N}(0, 1)$$
- Returns frozen `MCResult(price, std_error, ci_low, ci_high, n_paths)`.
- Standard error uses sample standard deviation with $\text{ddof}=1$: $\text{SE} = s / \sqrt{N}$.
- Verified 95% confidence interval coverage: Black-Scholes price falls within MC 95% CI in 95.5% of 200 seeded runs (matching theoretical 93–97% target).

## T-17: Variance Reduction via Antithetic Variates & Control Variate

- **Antithetic Variates (F-9):** Paired each $Z_i$ with $-Z_i$. Because option payoffs are monotonic in $Z$, $\text{Cov}(f(Z), f(-Z)) < 0$. Standard error is correctly computed over the $N/2$ independent pair averages $P_i = 0.5(f(Z_i) + f(-Z_i))$ (avoiding artificial underestimation from treating $2n$ correlated draws as i.i.d.). This reduces standard error by ~30% for ATM calls at zero extra RNG generation cost.
- **Control Variate (F-10):** Used discounted underlying price $Y = e^{-rT} S_T$ with known mean $\mathbb{E}[Y] = S e^{-qT}$. Optimal regression coefficient $\beta = \text{Cov}(X, Y) / \text{Var}(Y)$ reduces variance by >50%.

## T-18: Monte Carlo Convergence Analysis

Generated log-log convergence plot in `docs/mc_convergence.png`:
- Plotted standard error across $N \in [10^2, 10^6]$ for plain MC, antithetic variates, control variate, and combined.
- Empirical regression slope of $\log_{10}(\text{SE})$ against $\log_{10}(N)$ measured $-0.475 \approx -0.50$, confirming the $O(N^{-1/2})$ Central Limit Theorem convergence rate.

## T-19: Train/Test Optimization Protocol (`optimize_sma`)

Implemented `optimize_sma` with strict date separation at `train_end` (tested by `T-EN-3`):
- Grid search runs strictly on `prices <= train_end`. Asserted no look-ahead into test dates during search.
- Best in-sample parameters (Fast=50, Slow=150) evaluated once on out-of-sample data (`prices > train_end`).
- Observed realistic parameter decay: In-sample Sharpe of 0.88 degraded to 0.80 out-of-sample, trailing Buy & Hold (0.90).

## T-20: Multi-Asset Backtest Comparison

Implemented `multi_asset_comparison` evaluating 4 distinct asset classes (`SPY`, `AAPL`, `GLD`, `TLT`) against Buy & Hold:
- Highlighted that SMA crossover works best in secular bear markets (`TLT` drawdown reduced from -48.35% to -25.21%), while lagging in high-drift equities (`SPY`, `AAPL`) due to whipsaw and cash drag.

## T-21: Historical Volatility Metric (`historical_vol`)

Implemented `historical_vol` in `metrics.py` with type overloads:
- Uses log returns $r_t = \ln(p_t / p_{t-1})$ annualized via $\sqrt{252}$.
- Supports both full-sample (returning float) and rolling window (returning `pd.Series`).

## T-22: Empirical Results Documentation (`docs/results.md`)

Drafted comprehensive `docs/results.md`:
- Documents in-sample vs out-of-sample tables with and without costs.
- Includes multi-asset comparison table and Monte Carlo variance reduction table.
- Formulates honest, transparent conclusions regarding trend-following pitfalls and structural limitations.

## T-23: Analytical Black-Scholes Greeks (`bs_greeks`, `Greeks`)

Implemented analytical sensitivities for European calls and puts under Black-Scholes-Merton with continuous dividend yield $q$:
- $\Delta$: $\frac{\partial V}{\partial S} = e^{-qT} N(d_1)$ (call) and $-e^{-qT} N(-d_1)$ (put).
- $\Gamma$: $\frac{\partial^2 V}{\partial S^2} = \frac{e^{-qT} \phi(d_1)}{S \sigma \sqrt{T}}$ (identical for call and put).
- $\mathcal{V}$: $\frac{\partial V}{\partial \sigma} = S e^{-qT} \phi(d_1) \sqrt{T}$ (per 1.00 vol change, identical for call and put).
- $\Theta$: $\frac{\partial V}{\partial t} = -\frac{\partial V}{\partial T}$ (per calendar year).
- $\rho$: $\frac{\partial V}{\partial r} = K T e^{-rT} N(d_2)$ (call) and $-K T e^{-rT} N(-d_2)$ (put).
- Verified `T-GR-2` bounds ($\Delta \in [0, 1]$ for calls, $[-1, 0]$ for puts, $\Gamma \ge 0$, $\mathcal{V} \ge 0$).
- Verified exact consistency with Black-Scholes PDE: $\Theta + (r - q)S\Delta + \frac{1}{2}\sigma^2 S^2 \Gamma = rV$.
- Validated Put-Call Parity across all five Greeks.

## T-24: Finite-Difference Greeks & Common Random Numbers (`fd_greeks`)

Implemented central finite-difference sensitivities for arbitrary pricing callables:
- Central difference step $h_S = h_{\text{rel}} S$ yields $O(h^2)$ truncation error: $\Delta = \frac{V(S+h) - V(S-h)}{2h}$, $\Gamma = \frac{V(S+h) - 2V(S) + V(S-h)}{h^2}$.
- One-sided calendar decay for Theta: $h_T = \min(h_{\text{rel}}, T/2)$, $\Theta = \frac{V(T - h_T) - V(T)}{h_T}$.
- Matches analytical Greeks within strict `T-GR-1` tolerance thresholds ($\Delta < 10^{-5}, \Gamma < 10^{-4}, \mathcal{V} < 10^{-3}$).
- **Common Random Numbers (CRN):** Demonstrated that when pricing via Monte Carlo, resetting to identical RNG seeds across bump evaluations is mandatory; independent draws create independent sample noise $\sim \epsilon / h^2$, rendering finite-difference Gamma complete white noise.

## T-25: Cox-Ross-Rubinstein Binomial Tree (`binomial_price`)

Implemented recombining binomial lattice pricing for European and American options:
- Lattice parameters: $u = e^{\sigma \sqrt{\Delta t}}$, $d = 1/u$, $p = \frac{e^{(r-q)\Delta t} - d}{u - d}$. Raises `ValueError` if $p \notin [0, 1]$.
- Memory efficient $O(\text{steps})$ 1D array rolled backward in time.
- Validated `T-BT-1` convergence against Black-Scholes analytical values.
- Validated `T-BT-2` American early exercise properties: American call with $q=0$ strictly matches European call (early exercise never optimal); American put strictly $\ge$ European put, displaying significant early exercise premium for deep ITM puts.

## T-26: Binomial Convergence & Sawtooth Analysis (`docs/binomial_convergence.png`)

Generated convergence study comparing CRR tree prices against Black-Scholes for $n \in [10, 300]$:
- Observed classical sawtooth oscillations caused by node-strike misalignment (whether strike $K$ sits near a lattice node or midway between two nodes at maturity).
- Verified the asymptotic $O(1/n)$ decay envelope of pricing error.

## T-27: Implied Volatility via Newton-Raphson & Bisection Fallback (`implied_vol`)

Implemented root-finding algorithm to invert Black-Scholes option pricing:
- Enforces strict no-arbitrage bounds for calls ($\max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}$) and puts ($\max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}$), raising `ValueError` for invalid market prices (`T-IV-2`).
- Iterates via analytical Black-Scholes Vega ($\sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - C}{\mathcal{V}}$) starting at $\sigma_0 = 0.20$.
- Implements robust bisection fallback on $[10^{-6}, 5.0]$ whenever Vega $< 10^{-8}$ or iterates leave the valid domain, ensuring guaranteed convergence across extreme moneyness and long maturities (`T-IV-1`).