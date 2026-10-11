import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import (
    BacktestResult,
    OptimizationResult,
    compute_costs,
    compute_turnover,
    multi_asset_comparison,
    optimize_sma,
    run_backtest,
)
from quantlab.backtest.signals import lag_signal, sma_crossover_signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prices(n: int, start: str = "2024-01-02") -> pd.Series:
    return pd.Series(
        np.linspace(100.0, 200.0, n),
        index=pd.date_range(start, periods=n, freq="B"),
    )


def make_positions(prices: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    raw = sma_crossover_signal(prices, fast=fast, slow=slow)
    return lag_signal(raw)


# ---------------------------------------------------------------------------
# compute_turnover tests
# ---------------------------------------------------------------------------


def test_turnover_zero_when_position_unchanged() -> None:
    """Turnover is 0 on days where the position does not change."""
    pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=float)
    tv = compute_turnover(pos)

    assert tv.iloc[0] == pytest.approx(0.0)  # entered from 0 -> 0, no change
    assert tv.iloc[1] == pytest.approx(0.0)  # 0 -> 0
    assert tv.iloc[2] == pytest.approx(1.0)  # 0 -> 1, entry
    assert tv.iloc[3] == pytest.approx(0.0)  # 1 -> 1, no change
    assert tv.iloc[4] == pytest.approx(0.0)  # 1 -> 1, no change
    assert tv.iloc[5] == pytest.approx(1.0)  # 1 -> 0, exit


def test_turnover_first_bar_is_entry_cost() -> None:
    """If we hold a position from bar 0, the first bar incurs entry turnover."""
    pos = pd.Series([1.0, 1.0, 1.0])
    tv = compute_turnover(pos)
    assert tv.iloc[0] == pytest.approx(1.0)  # entered from 0 -> 1


def test_turnover_long_short_reversal_is_two() -> None:
    """Reversing from long (+1) to short (-1) should produce turnover of 2."""
    pos = pd.Series([1.0, -1.0])
    tv = compute_turnover(pos)
    assert tv.iloc[1] == pytest.approx(2.0)


def test_turnover_preserves_index() -> None:
    prices = make_prices(30)
    pos = make_positions(prices)
    tv = compute_turnover(pos)
    assert tv.index.equals(pos.index)
    assert len(tv) == len(pos)


def test_turnover_all_nonnegative() -> None:
    prices = make_prices(50)
    pos = make_positions(prices)
    tv = compute_turnover(pos)
    assert (tv >= 0).all()


def test_turnover_zero_cost_when_no_trades() -> None:
    """T-EN-2 partial: zero trades -> zero cost."""
    # A position that never changes -> zero turnover after bar 0 entry
    pos = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    tv = compute_turnover(pos)
    # Only bar 0 has turnover (the entry); all others are 0
    assert tv.iloc[1:].sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T-EN-2: Cost model tests
# ---------------------------------------------------------------------------


def test_zero_cost_bps_produces_zero_costs() -> None:
    """T-EN-2: cost_bps=0 means no friction regardless of turnover."""
    prices = make_prices(50)
    pos = make_positions(prices)
    tv = compute_turnover(pos)
    costs = compute_costs(tv, cost_bps=0.0)
    assert (costs == 0.0).all()


def test_positive_cost_bps_gives_positive_costs_when_trading() -> None:
    """T-EN-2: positive cost_bps and nonzero turnover -> positive costs on trade days."""
    pos = pd.Series([0.0, 1.0, 1.0, 0.0], dtype=float)
    tv = compute_turnover(pos)
    costs = compute_costs(tv, cost_bps=10.0)

    # Only days with actual turnover have positive costs
    assert costs.iloc[0] == pytest.approx(0.0)  # entry from 0 -> 0, no trade
    assert costs.iloc[1] == pytest.approx(10.0 / 10_000)  # 0 -> 1 entry
    assert costs.iloc[2] == pytest.approx(0.0)  # no change
    assert costs.iloc[3] == pytest.approx(10.0 / 10_000)  # 1 -> 0 exit


def test_cost_scales_linearly_with_bps() -> None:
    """Cost is proportional to cost_bps."""
    tv = pd.Series([0.0, 1.0, 2.0, 0.5])
    c1 = compute_costs(tv, cost_bps=5.0)
    c2 = compute_costs(tv, cost_bps=10.0)
    pd.testing.assert_series_equal(c2, c1 * 2.0)


def test_cost_scales_linearly_with_turnover() -> None:
    """Cost is proportional to turnover."""
    tv1 = pd.Series([0.0, 1.0, 0.0, 1.0])
    tv2 = pd.Series([0.0, 2.0, 0.0, 2.0])  # double turnover (e.g. L/S reversal)
    c1 = compute_costs(tv1, cost_bps=5.0)
    c2 = compute_costs(tv2, cost_bps=5.0)
    pd.testing.assert_series_equal(c2, c1 * 2.0)


def test_long_short_reversal_costs_twice_as_much() -> None:
    """Reversing L/S (turnover=2) costs exactly twice a single entry (turnover=1)."""
    tv_entry = compute_turnover(pd.Series([0.0, 1.0]))  # 0 -> 1, turnover=1
    tv_reversal = compute_turnover(pd.Series([1.0, -1.0]))  # 1 -> -1, turnover=2

    c_entry = compute_costs(tv_entry, cost_bps=10.0).iloc[1]
    c_reversal = compute_costs(tv_reversal, cost_bps=10.0).iloc[1]
    assert c_reversal == pytest.approx(c_entry * 2.0)


def test_negative_cost_bps_raises() -> None:
    """Negative cost_bps is nonsensical and should raise ValueError."""
    tv = pd.Series([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="cost_bps must be non-negative"):
        compute_costs(tv, cost_bps=-1.0)


def test_cost_preserves_index() -> None:
    prices = make_prices(30)
    pos = make_positions(prices)
    tv = compute_turnover(pos)
    costs = compute_costs(tv, cost_bps=5.0)
    assert costs.index.equals(tv.index)
    assert len(costs) == len(tv)


# ---------------------------------------------------------------------------
# T-EN-1: always-long with zero cost reproduces buy-and-hold
# ---------------------------------------------------------------------------


def test_always_long_zero_cost_equals_buy_and_hold() -> None:
    """T-EN-1: always-long (zero cost) strategy return equals simple_returns of prices.

    On a strictly increasing price series fast SMA is always above slow SMA
    once the warm-up ends.  With cost_bps=0 the strategy return on each live
    day is exactly 1 * asset_ret.  The equity curve must equal the buy-and-hold
    equity curve computed only over those same days.
    """
    n = 60
    prices = pd.Series(
        np.linspace(100.0, 200.0, n),
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    fast, slow = 5, 20

    result = run_backtest(prices, fast=fast, slow=slow, cost_bps=0.0)

    # Find the first day where the strategy is actually long (past warm-up + lag)
    first_long = result.positions[result.positions == 1].index[0]

    # Slice the strategy equity from that day onward and normalise to 1.0
    strat_equity = result.equity.loc[first_long:]
    strat_normalised = strat_equity / strat_equity.iloc[0]

    # Build buy-and-hold equity over the same date range, also normalised to 1.0
    prices_live = prices.loc[first_long:]
    bah_normalised = prices_live / prices_live.iloc[0]

    # They must be numerically identical bar-by-bar
    import math

    for t, (s, b) in enumerate(zip(strat_normalised, bah_normalised, strict=True)):
        assert math.isclose(s, b, rel_tol=1e-9), (
            f"Bar {t}: strategy equity={s:.8f}, buy-and-hold equity={b:.8f}"
        )


def test_backtest_result_structure() -> None:
    """BacktestResult fields have correct types and consistent lengths."""
    prices = make_prices(60)
    result = run_backtest(prices, fast=5, slow=20, cost_bps=5.0)

    assert isinstance(result, BacktestResult)
    assert isinstance(result.returns, pd.Series)
    assert isinstance(result.positions, pd.Series)
    assert isinstance(result.equity, pd.Series)
    assert isinstance(result.n_trades, int)
    assert isinstance(result.metrics, dict)

    # All series must share the same index
    assert result.returns.index.equals(result.positions.index)
    assert result.returns.index.equals(result.equity.index)

    # Required metric keys present
    for key in ("cagr", "vol", "sharpe", "max_drawdown", "total_cost"):
        assert key in result.metrics

    # Equity starts at the compounded return of the first bar (not necessarily 1.0,
    # but must be positive)
    assert (result.equity > 0).all()


def test_equity_starts_from_one_when_always_invested() -> None:
    """Equity curve's first value equals 1 + strategy_return on the first bar."""
    prices = make_prices(60)
    result = run_backtest(prices, fast=5, slow=20, cost_bps=0.0)
    # equity = (1 + returns).cumprod() -- first value equals 1 + returns[0]
    assert result.equity.iloc[0] == pytest.approx(1.0 + result.returns.iloc[0])


# ---------------------------------------------------------------------------
# T-EN-2: costs strictly lower equity; zero trades mean zero cost
# ---------------------------------------------------------------------------


def test_positive_cost_strictly_lowers_equity() -> None:
    """T-EN-2: with trades occurring, positive cost_bps gives lower final equity."""
    prices = make_prices(60)
    fast, slow = 5, 20

    result_free = run_backtest(prices, fast=fast, slow=slow, cost_bps=0.0)
    result_costly = run_backtest(prices, fast=fast, slow=slow, cost_bps=50.0)

    # Must have at least one trade for the test to be meaningful
    assert result_costly.n_trades > 0, "Expected at least one trade on this price series"
    assert result_costly.equity.iloc[-1] < result_free.equity.iloc[-1]
    assert result_costly.metrics["total_cost"] > 0.0


def test_zero_trades_means_zero_total_cost() -> None:
    """T-EN-2: on a monotone-increasing series only one trade (entry) occurs."""
    prices = make_prices(60)
    result = run_backtest(prices, fast=5, slow=20, cost_bps=100.0)

    from quantlab.backtest.engine import compute_turnover

    tv = compute_turnover(result.positions)
    # Find index of the single entry (first nonzero turnover)
    first_trade_idx = tv[tv > 0].index[0]
    # All turnover after the entry must be zero
    post_entry = tv.loc[tv.index > first_trade_idx]
    assert post_entry.sum() == pytest.approx(0.0)


def test_n_trades_counts_position_change_days() -> None:
    """n_trades equals the number of days with nonzero turnover."""
    prices = make_prices(60)
    result = run_backtest(prices, fast=5, slow=20, cost_bps=0.0)

    from quantlab.backtest.engine import compute_turnover

    tv = compute_turnover(result.positions)
    expected_trades = int((tv > 0).sum())
    assert result.n_trades == expected_trades


def test_t_en_3_optimize_sma_no_data_leakage() -> None:
    """T-EN-3: optimize_sma never touches data after train_end during search."""
    # Generate 120 days of data
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    prices = pd.Series(np.linspace(100.0, 150.0, 120), index=dates)

    train_end = str(dates[79].date())  # 80 days train, 40 days test
    train_cutoff = pd.Timestamp(train_end)

    opt = optimize_sma(
        prices,
        train_end=train_end,
        fast_grid=[5, 10],
        slow_grid=[20, 30],
        cost_bps=5.0,
    )

    assert isinstance(opt, OptimizationResult)
    # Train returns and positions must never exceed train_cutoff
    assert opt.train_result.returns.index.max() <= train_cutoff
    assert opt.train_result.positions.index.max() <= train_cutoff
    assert opt.train_result.equity.index.max() <= train_cutoff

    # Grid search results must all have been run on train data
    assert not opt.grid_results.empty
    assert (opt.grid_results["fast"] < opt.grid_results["slow"]).all()

    # Out-of-sample test results start strictly after train_cutoff
    assert opt.test_result.returns.index.min() > train_cutoff
    assert opt.test_result.positions.index.min() > train_cutoff


def test_optimize_sma_best_selection_and_test_metrics() -> None:
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    # Up-trend then down-trend
    p = np.concatenate([np.linspace(100, 150, 75), np.linspace(150, 110, 75)])
    prices = pd.Series(p, index=dates)

    train_end = str(dates[99].date())
    opt = optimize_sma(
        prices,
        train_end=train_end,
        fast_grid=[5, 10, 15],
        slow_grid=[30, 50],
        cost_bps=5.0,
    )

    assert opt.best_fast < opt.best_slow
    for k in ("cagr", "vol", "sharpe", "max_drawdown", "total_cost"):
        assert k in opt.train_metrics
        assert k in opt.test_metrics

    # Test evaluated ONCE on the winning parameters
    assert isinstance(opt.test_result, BacktestResult)


def test_optimize_sma_invalid_inputs() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = pd.Series(np.linspace(100, 150, 50), index=dates)

    # Empty test set
    with pytest.raises(ValueError, match="Both train and test periods must contain price data"):
        optimize_sma(prices, train_end=str(dates[-1].date()))

    # Empty train set
    with pytest.raises(ValueError, match="Both train and test periods must contain price data"):
        optimize_sma(prices, train_end="2019-01-01")

    # No valid pairs (fast >= slow)
    with pytest.raises(ValueError, match="No valid"):
        optimize_sma(prices, train_end=str(dates[25].date()), fast_grid=[50], slow_grid=[20])


def test_multi_asset_comparison_4_assets() -> None:
    """T-20: Multi-asset comparison table across 4 assets."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")

    prices_dict = {
        "ASSET_A": pd.Series(np.linspace(100, 150, 100), index=dates),
        "ASSET_B": pd.Series(np.linspace(100, 80, 100), index=dates),
        "ASSET_C": pd.Series(100 + 10 * np.sin(np.linspace(0, 3.14, 100)), index=dates),
        "ASSET_D": pd.Series(np.linspace(50, 120, 100), index=dates),
    }

    df = multi_asset_comparison(prices_dict, fast=10, slow=30, cost_bps=5.0)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df.index) == {"ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"}

    expected_cols = {
        "strat_cagr",
        "bah_cagr",
        "strat_vol",
        "bah_vol",
        "strat_sharpe",
        "bah_sharpe",
        "strat_max_dd",
        "bah_max_dd",
        "n_trades",
        "total_cost",
    }
    assert expected_cols.issubset(set(df.columns))
