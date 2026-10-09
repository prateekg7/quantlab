import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Simple returns p_t / p_{t-1} - 1."""
    if prices.min() <= 0:
        raise ValueError("Prices must be positive.")
    if len(prices) < 2:
        raise ValueError("Prices must contain at least two values.")
    return (prices / prices.shift(1) - 1).dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns ln(p_t / p_{t-1})."""
    if prices.min() <= 0:
        raise ValueError("Prices must be positive.")
    if len(prices) < 2:
        raise ValueError("Prices must contain at least two values.")
    return (prices / prices.shift(1)).dropna().apply(np.log)
