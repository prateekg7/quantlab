"""Run backtest experiments and multi-asset comparisons for docs/results.md.

Produces reproducible, verifiable numbers for:
1. Multi-asset comparison (SPY, AAPL, GLD, TLT) over 2010-2024.
2. In-sample (2010-2018) vs. Out-of-sample (2019-2024) with and without costs.
3. Monte Carlo pricing benchmark comparison.
"""

import yfinance as yf

from quantlab.backtest.engine import multi_asset_comparison, optimize_sma
from quantlab.metrics import annualized_vol, cagr, max_drawdown, sharpe, simple_returns
from quantlab.pricing.black_scholes import bs_price
from quantlab.pricing.monte_carlo import mc_price


def run_all() -> None:
    print("=" * 60)
    print("EXPERIMENT 1: Multi-Asset Backtest (2010-2024)")
    print("=" * 60)
    tickers = ["SPY", "AAPL", "GLD", "TLT"]
    prices_dict = {}

    for ticker in tickers:
        raw = yf.download(
            ticker, start="2010-01-01", end="2024-12-31", auto_adjust=True, progress=False
        )
        p = raw["Close"].squeeze().dropna()
        p.name = ticker
        prices_dict[ticker] = p
        print(f"Loaded {ticker}: {len(p)} bars ({p.index[0].date()} to {p.index[-1].date()})")

    df_multi = multi_asset_comparison(prices_dict, fast=50, slow=200, cost_bps=5.0)
    print("\nMulti-Asset Results Table (5 bps cost):")
    print(df_multi.to_string())

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: In-Sample vs Out-of-Sample on SPY (Split 2018-12-31)")
    print("=" * 60)
    spy_prices = prices_dict["SPY"]
    train_end = "2018-12-31"

    # With costs (5 bps)
    opt_costs = optimize_sma(spy_prices, train_end=train_end, cost_bps=5.0)
    print(f"Best parameters with 5 bps: Fast={opt_costs.best_fast}, Slow={opt_costs.best_slow}")
    print("In-sample (train) metrics (with costs):", opt_costs.train_metrics)
    print("Out-of-sample (test) metrics (with costs):", opt_costs.test_metrics)

    # Without costs (0 bps)
    opt_nocost = optimize_sma(
        spy_prices,
        train_end=train_end,
        fast_grid=[opt_costs.best_fast],
        slow_grid=[opt_costs.best_slow],
        cost_bps=0.0,
    )
    print("\nIn-sample metrics (0 bps):", opt_nocost.train_metrics)
    print("Out-of-sample metrics (0 bps):", opt_nocost.test_metrics)

    # Buy and hold for train and test periods
    spy_ret = simple_returns(spy_prices)
    train_ret = spy_ret.loc[spy_ret.index <= train_end]
    test_ret = spy_ret.loc[spy_ret.index > train_end]

    print("\nBuy & Hold In-Sample (Train 2010-2018):")
    print(f"  CAGR:   {cagr(train_ret):+.2%}")
    print(f"  Vol:    {annualized_vol(train_ret):.2%}")
    print(f"  Sharpe: {sharpe(train_ret):.2f}")
    print(f"  MaxDD:  {max_drawdown((1 + train_ret).cumprod()):.2%}")

    print("\nBuy & Hold Out-of-Sample (Test 2019-2024):")
    print(f"  CAGR:   {cagr(test_ret):+.2%}")
    print(f"  Vol:    {annualized_vol(test_ret):.2%}")
    print(f"  Sharpe: {sharpe(test_ret):.2f}")
    print(f"  MaxDD:  {max_drawdown((1 + test_ret).cumprod()):.2%}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Monte Carlo Pricing & Variance Reduction")
    print("=" * 60)
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs = bs_price(S, K, r, sigma, T, "call")
    print(f"Analytical BS Call Price: {bs:.4f}")

    seed = 42
    n_paths = 100_000
    res_plain = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=seed)
    res_anti = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=seed, antithetic=True)
    res_cv = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=seed, control_variate=True)
    res_both = mc_price(
        S, K, r, sigma, T, "call", n_paths=n_paths, seed=seed, antithetic=True, control_variate=True
    )

    vr_anti = 1.0 - (res_anti.std_error / res_plain.std_error) ** 2
    vr_cv = 1.0 - (res_cv.std_error / res_plain.std_error) ** 2
    vr_both = 1.0 - (res_both.std_error / res_plain.std_error) ** 2

    print(
        f"Plain MC:     Price={res_plain.price:.4f}, SE={res_plain.std_error:.5f}, "
        f"95% CI=[{res_plain.ci_low:.4f}, {res_plain.ci_high:.4f}]"
    )
    print(
        f"Antithetic:   Price={res_anti.price:.4f}, SE={res_anti.std_error:.5f}, "
        f"Var Red={vr_anti:.1%}"
    )
    print(f"Control Var:  Price={res_cv.price:.4f}, SE={res_cv.std_error:.5f}, Var Red={vr_cv:.1%}")
    print(
        f"Combined:     Price={res_both.price:.4f}, SE={res_both.std_error:.5f}, "
        f"Var Red={vr_both:.1%}"
    )


if __name__ == "__main__":
    run_all()
