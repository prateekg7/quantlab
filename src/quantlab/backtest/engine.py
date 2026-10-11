"""Backtesting engine: cost model and full run_backtest loop."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

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


@dataclass
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
