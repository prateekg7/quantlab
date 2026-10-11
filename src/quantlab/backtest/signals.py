"""Signal generators for systematic strategies.

Relevant LEARNING.md stages:
- Stage 2: Moving averages, trend-following signals
- Stage 7: Look-ahead bias and the importance of lagging signals
"""

import pandas as pd


def sma_crossover_signal(
    prices: pd.Series,
    fast: int,
    slow: int,
    long_short: bool = False,
) -> pd.Series:
    """Generate a raw SMA crossover signal on each price bar (not yet position-ready).

    A signal of 1 means "go long" (fast SMA > slow SMA).
    A signal of 0 means "stay flat" (long-only mode, or before the warm-up period ends).
    A signal of -1 means "go short" (long/short mode when fast SMA <= slow SMA).

    IMPORTANT: this is the *raw* signal at the close of day t, using prices up to and
    including day t.  It must be lagged by one day with `lag_signal` before use as a
    position to avoid look-ahead bias.

    Formula:
        SMA_fast(t) = mean(prices[t - fast + 1 : t + 1])
        SMA_slow(t) = mean(prices[t - slow + 1 : t + 1])
        signal(t)   = 1 if SMA_fast(t) > SMA_slow(t) else (0 or -1)

    Parameters:
        prices: Daily price series with a DatetimeIndex, sorted ascending.
        fast: Lookback window for the fast moving average (number of observations).
        slow: Lookback window for the slow moving average (number of observations).
        long_short: If True, the "out" signal is -1 (short). If False (default), it is 0 (flat).

    Returns:
        pd.Series of signals with the same index as `prices`.
        Values are in {0, 1} (long-only) or {-1, 1} (long/short).
        Signal is 0 before `slow` observations are available (warm-up period).

    Raises:
        ValueError: If fast >= slow, or either window is < 1.
    """
    if fast < 1 or slow < 1:
        raise ValueError(f"fast and slow must be >= 1, got fast={fast}, slow={slow}")
    if fast >= slow:
        raise ValueError(f"fast must be strictly less than slow, got fast={fast}, slow={slow}")

    sma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    sma_slow = prices.rolling(window=slow, min_periods=slow).mean()

    # Default signal is 0 (flat). We only assign a directional signal where the slow
    # SMA has enough data (i.e. where sma_slow is not NaN — this enforces the warm-up).
    out_val = -1 if long_short else 0
    signal = pd.Series(0, index=prices.index, dtype=float)

    has_slow = sma_slow.notna()
    signal[has_slow & (sma_fast > sma_slow)] = 1.0
    signal[has_slow & (sma_fast <= sma_slow)] = float(out_val)

    return signal
