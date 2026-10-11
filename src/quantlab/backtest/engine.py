"""Backtesting engine: cost model and full run_backtest loop."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.backtest.signals import lag_signal, sma_crossover_signal
from quantlab.metrics import annualized_vol, cagr, max_drawdown, sharpe, simple_returns


def compute_turnover(positions: pd.Series) -> pd.Series:
    """Compute the per-day turnover as the absolute change in position.

    Turnover measures how much the position changes on each day.  In long-only
    mode positions move between {0, 1}, so the maximum daily turnover is 1.0.
    In long/short mode positions move between {-1, 1}, so a full reversal
    produces a turnover of 2.0 (correctly capturing the cost of closing the
    short and opening the long, or vice-versa).

    Formula:
        turnover(t) = | position(t) - position(t-1) |

    The very first bar has no prior position, so turnover defaults to
    |position(0) - 0| = |position(0)|.  This ensures an entry cost is charged
    on the first day a position is taken.

    Parameters:
        positions: Series of positions, typically produced by
            ``lag_signal(sma_crossover_signal(...))``.

    Returns:
        pd.Series of non-negative floats with the same index as ``positions``.
    """
    # diff() gives position(t) - position(t-1); the first element is NaN which
    # we fill with the position itself (as if we entered from 0).
    return positions.diff().fillna(positions).abs()


def compute_costs(turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Convert per-day turnover into a monetary cost in decimal-return units.

    ``cost_bps`` is the one-way cost per unit of notional turnover expressed in
    basis points (1 bps = 0.01%).  Dividing by 10 000 converts to a decimal
    fraction that can be subtracted directly from the gross return.

    Formula:
        cost(t) = turnover(t) * cost_bps / 10_000

    Parameters:
        turnover: Per-day turnover series (output of ``compute_turnover``).
        cost_bps: Transaction cost in basis points per unit of turnover (>= 0).

    Returns:
        pd.Series of non-negative cost values with the same index as ``turnover``.

    Raises:
        ValueError: If ``cost_bps`` is negative.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be non-negative, got {cost_bps}")
    return turnover * cost_bps / 10_000


@dataclass(frozen=True)
class BacktestResult:
    """Results from a single backtest run.

    Attributes:
        returns:   Net strategy return per day (after costs), aligned to the
                   asset return index (starts on the second price bar).
        positions: Position held on each day (0, 1, or -1).
        equity:    Cumulative equity curve: (1 + returns).cumprod(), starts at 1.0.
        n_trades:  Number of days on which a position change (trade) occurred.
        metrics:   Summary statistics: cagr, vol, sharpe, max_drawdown, total_cost.
    """

    returns: pd.Series
    positions: pd.Series
    equity: pd.Series
    n_trades: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class OptimizationResult:
    """Results from SMA parameter optimization with train/test separation.

    Attributes:
        best_fast: Winning fast SMA window chosen on training data.
        best_slow: Winning slow SMA window chosen on training data.
        train_metrics: Performance metrics on the in-sample training set.
        test_metrics: Performance metrics on the out-of-sample test set.
        train_result: BacktestResult for winning parameters on the training set.
        test_result: BacktestResult for winning parameters on the test set.
        grid_results: DataFrame of all parameter combinations evaluated during search.
    """

    best_fast: int
    best_slow: int
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_result: BacktestResult
    test_result: BacktestResult
    grid_results: pd.DataFrame


def run_backtest(
    prices: pd.Series,
    fast: int,
    slow: int,
    cost_bps: float = 0.0,
    long_short: bool = False,
) -> BacktestResult:
    """Run a full SMA-crossover backtest on a price series.

    Pipeline (no look-ahead at any step):
        1. asset_ret(t)  = simple return of prices on day t
        2. signal(t)     = sma_crossover_signal(prices, fast, slow, long_short)
        3. position(t)   = signal(t-1)   [lag_signal — look-ahead protection]
        4. turnover(t)   = |position(t) - position(t-1)|
        5. cost(t)       = turnover(t) * cost_bps / 10_000
        6. strategy_ret(t) = position(t) * asset_ret(t) - cost(t)
        7. equity(t)     = prod_{s<=t}(1 + strategy_ret(s))

    All series are aligned on the asset return index (day 2 onward), so
    ``positions`` and ``returns`` share the same DatetimeIndex throughout.

    Parameters:
        prices:     Daily price series, sorted ascending with a DatetimeIndex.
        fast:       Fast SMA window (number of observations, must be < slow).
        slow:       Slow SMA window (number of observations).
        cost_bps:   One-way transaction cost per unit of turnover (>= 0).
        long_short: If True, the strategy goes short when the fast SMA is below
                    the slow SMA.  If False (default), it goes flat instead.

    Returns:
        BacktestResult dataclass with returns, positions, equity, n_trades, metrics.

    Raises:
        ValueError: Propagated from ``sma_crossover_signal`` (fast >= slow, etc.)
                    or ``compute_costs`` (negative cost_bps).
    """
    # --- Step 1: asset returns (n-1 bars, starting on day 2) ---
    asset_ret = simple_returns(prices)

    # --- Step 2+3: signal on full price history, then lag to get positions ---
    raw_signal = sma_crossover_signal(prices, fast=fast, slow=slow, long_short=long_short)
    # Align positions to the asset_ret index: we need position(t) for each
    # day t that has an asset return.  raw_signal is defined on prices.index;
    # lag_signal shifts by 1, so position(t) = raw_signal(t-1).
    positions_full = lag_signal(raw_signal)
    # Reindex to asset_ret's index (drops the very first price bar, which has
    # no return and no lagged position to use).
    positions = positions_full.reindex(asset_ret.index)

    # --- Step 4+5: turnover and cost ---
    turnover = compute_turnover(positions)
    costs = compute_costs(turnover, cost_bps)

    # --- Step 6: net strategy return ---
    strategy_ret = positions * asset_ret - costs

    # --- Step 7: equity curve ---
    equity = (1.0 + strategy_ret).cumprod()

    # --- Step 8: n_trades (days where a position change occurred) ---
    n_trades = int((turnover > 0).sum())

    # --- Step 9: summary metrics (suppress zero-variance warning for flat periods) ---
    total_cost = float(costs.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        metrics: dict[str, float] = {
            "cagr": cagr(strategy_ret),
            "vol": annualized_vol(strategy_ret),
            "sharpe": sharpe(strategy_ret),
            "max_drawdown": max_drawdown(equity),
            "total_cost": total_cost,
        }

    return BacktestResult(
        returns=strategy_ret,
        positions=positions,
        equity=equity,
        n_trades=n_trades,
        metrics=metrics,
    )


def optimize_sma(
    prices: pd.Series,
    train_end: str | pd.Timestamp,
    fast_grid: Sequence[int] | None = None,
    slow_grid: Sequence[int] | None = None,
    cost_bps: float = 0.0,
    long_short: bool = False,
) -> OptimizationResult:
    r"""Optimize SMA crossover parameters on in-sample data and evaluate out-of-sample.

    Protocol (F-19, T-EN-3):
        1. Split prices by date at ``train_end``. The grid search evaluates parameter
           combinations strictly on data up to ``train_end`` (asserting max date <= train_end).
        2. Objective: maximize Sharpe ratio net of costs on the training set.
        3. The winning parameter pair (best_fast, best_slow) is evaluated ONCE on the
           test set (prices after train_end).
        4. To avoid losing warm-up days at the start of the test period, indicators
           are computed on the full price history, and returns are evaluated for t > train_end.

    Parameters:
        prices: Daily price series with DatetimeIndex.
        train_end: End date of training period (inclusive, e.g. "2018-12-31").
        fast_grid: Sequence of fast SMA windows (default: (5, 10, 20, 30, 50)).
        slow_grid: Sequence of slow SMA windows (default: (50, 100, 150, 200)).
        cost_bps: One-way transaction costs in basis points.
        long_short: If True, strategy shorts when fast < slow; else flat.

    Returns:
        OptimizationResult dataclass.
    """
    train_cutoff = pd.Timestamp(train_end)
    train_prices = prices.loc[prices.index <= train_cutoff]
    test_prices = prices.loc[prices.index > train_cutoff]

    if train_prices.empty or test_prices.empty:
        raise ValueError("Both train and test periods must contain price data.")

    fast_candidates = list(fast_grid) if fast_grid is not None else [5, 10, 20, 30, 50]
    slow_candidates = list(slow_grid) if slow_grid is not None else [50, 100, 150, 200]

    grid = [(f, s) for f in fast_candidates for s in slow_candidates if f < s]
    if not grid:
        raise ValueError("No valid (fast, slow) pairs with fast < slow.")

    # T-EN-3: Assert that training prices never touch data after train_end
    assert train_prices.index.max() <= train_cutoff, (
        "Data leakage: post-cutoff prices in train set."
    )

    best_sharpe = -float("inf")
    best_pair: tuple[int, int] | None = None
    grid_rows: list[dict[str, float | int]] = []

    for f, s in grid:
        res_train = run_backtest(
            train_prices, fast=f, slow=s, cost_bps=cost_bps, long_short=long_short
        )
        assert res_train.returns.index.max() <= train_cutoff

        sh = res_train.metrics["sharpe"]
        score = -float("inf") if (math.isnan(sh) or np.isnan(sh)) else float(sh)

        grid_rows.append(
            {
                "fast": f,
                "slow": s,
                "sharpe": sh,
                "cagr": res_train.metrics["cagr"],
                "vol": res_train.metrics["vol"],
                "max_drawdown": res_train.metrics["max_drawdown"],
                "total_cost": res_train.metrics["total_cost"],
                "n_trades": res_train.n_trades,
            }
        )

        if score > best_sharpe:
            best_sharpe = score
            best_pair = (f, s)

    if best_pair is None:
        best_pair = grid[0]

    best_fast, best_slow = best_pair
    best_train_res = run_backtest(
        train_prices, fast=best_fast, slow=best_slow, cost_bps=cost_bps, long_short=long_short
    )

    # Evaluate on test period ONCE: full history used to warm up SMAs
    full_res = run_backtest(
        prices, fast=best_fast, slow=best_slow, cost_bps=cost_bps, long_short=long_short
    )

    test_returns = full_res.returns.loc[full_res.returns.index > train_cutoff]
    test_positions = full_res.positions.loc[full_res.positions.index > train_cutoff]
    test_turnover = compute_turnover(test_positions)
    test_costs = compute_costs(test_turnover, cost_bps)
    test_equity = (1.0 + test_returns).cumprod()
    test_trades = int((test_turnover > 0).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        test_metrics: dict[str, float] = {
            "cagr": cagr(test_returns),
            "vol": annualized_vol(test_returns),
            "sharpe": sharpe(test_returns),
            "max_drawdown": max_drawdown(test_equity),
            "total_cost": float(test_costs.sum()),
        }

    test_res = BacktestResult(
        returns=test_returns,
        positions=test_positions,
        equity=test_equity,
        n_trades=test_trades,
        metrics=test_metrics,
    )

    return OptimizationResult(
        best_fast=best_fast,
        best_slow=best_slow,
        train_metrics=best_train_res.metrics,
        test_metrics=test_metrics,
        train_result=best_train_res,
        test_result=test_res,
        grid_results=pd.DataFrame(grid_rows),
    )


def multi_asset_comparison(
    prices_dict: dict[str, pd.Series],
    fast: int = 50,
    slow: int = 200,
    cost_bps: float = 5.0,
    long_short: bool = False,
) -> pd.DataFrame:
    r"""Run SMA crossover strategy across multiple assets and compare to buy-and-hold.

    Parameters:
        prices_dict: Dictionary mapping asset ticker symbol to its price series.
        fast: Fast SMA window (default 50).
        slow: Slow SMA window (default 200).
        cost_bps: One-way cost in basis points (default 5.0).
        long_short: If True, long/short strategy; else long-only.

    Returns:
        DataFrame comparing Strategy vs Buy & Hold across CAGR, Volatility,
        Sharpe ratio, Max Drawdown, and trade count for all assets.
    """
    rows = []
    for ticker, prices in prices_dict.items():
        res = run_backtest(prices, fast=fast, slow=slow, cost_bps=cost_bps, long_short=long_short)
        asset_ret = simple_returns(prices)
        bah_equity = (1.0 + asset_ret).cumprod()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bah_cagr = cagr(asset_ret)
            bah_vol = annualized_vol(asset_ret)
            bah_sh = sharpe(asset_ret)
            bah_mdd = max_drawdown(bah_equity)

        rows.append(
            {
                "ticker": ticker,
                "strat_cagr": res.metrics["cagr"],
                "bah_cagr": bah_cagr,
                "strat_vol": res.metrics["vol"],
                "bah_vol": bah_vol,
                "strat_sharpe": res.metrics["sharpe"],
                "bah_sharpe": bah_sh,
                "strat_max_dd": res.metrics["max_drawdown"],
                "bah_max_dd": bah_mdd,
                "n_trades": res.n_trades,
                "total_cost": res.metrics["total_cost"],
            }
        )

    return pd.DataFrame(rows).set_index("ticker")
