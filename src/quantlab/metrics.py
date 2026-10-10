import math
import warnings

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


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR: (prod(1 + r))^(periods_per_year / n) - 1."""
    if returns.empty:
        raise ValueError("Returns series cannot be empty.")
    n = len(returns)
    compounded = np.prod(1 + returns)
    return float(math.pow(compounded, periods_per_year / n) - 1)


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised volatility: std(r, ddof=1) * sqrt(periods_per_year)."""
    if returns.empty:
        raise ValueError("Returns series cannot be empty.")
    return float(returns.std() * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Sharpe ratio: mean(r - rf/periods_per_year) / std(r, ddof=1) * sqrt(periods_per_year)."""
    if returns.empty:
        raise ValueError("Returns series cannot be empty.")
    vol = returns.std()
    if vol == 0.0:
        warnings.warn(
            "Returns have zero variance; Sharpe ratio is undefined.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    mean_excess_return = returns.mean() - (rf / periods_per_year)
    return float((mean_excess_return / vol) * math.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown: equity / equity.cummax() - 1."""
    if equity.empty:
        raise ValueError("Equity series cannot be empty.")
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """Max drawdown: min of drawdown(); a value <= 0."""
    if equity.empty:
        raise ValueError("Equity series cannot be empty.")
    return float(drawdown(equity).min())
