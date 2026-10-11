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