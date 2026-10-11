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