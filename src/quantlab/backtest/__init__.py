"""Backtesting engine, signal generators, and parameter optimization."""

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

__all__ = [
    "BacktestResult",
    "OptimizationResult",
    "compute_costs",
    "compute_turnover",
    "lag_signal",
    "multi_asset_comparison",
    "optimize_sma",
    "run_backtest",
    "sma_crossover_signal",
]
