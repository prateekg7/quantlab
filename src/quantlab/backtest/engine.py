"""Backtesting engine: cost model and full run_backtest loop."""

from __future__ import annotations

import pandas as pd


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
