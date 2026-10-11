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


def lag_signal(signal: pd.Series, lag: int = 1) -> pd.Series:
    """Shift a signal forward in time to convert it into a position series.

    In any real trading system you can only act on a signal *after* you observe it.
    A signal computed at the close of day t can only be executed at the open of day t+1
    — or equivalently, it becomes the position *held during* day t+1.

    This one-line shift is the sole mechanism that protects a backtest from look-ahead
    bias: without it, the strategy would "know" the current day's closing price before
    deciding whether to buy or sell, which is impossible in practice.

    Formula:
        position(t) = signal(t - lag)

    The first `lag` entries become NaN after shifting; they are filled with 0 (flat)
    because there is no prior signal to act on.

    Parameters:
        signal: Raw signal series produced by `sma_crossover_signal` (or similar).
        lag: Number of periods to shift forward (default 1 = one trading day).

    Returns:
        pd.Series with the same index as `signal`, values shifted by `lag`.
        Leading NaNs (from the shift) are replaced with 0.

    Raises:
        ValueError: If lag < 1.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    # shift(1) moves every value one row forward:
    # the position on day t becomes the signal from day t-1.
    # fill_value=0 replaces the leading NaN that shift introduces (no position before history).
    return signal.shift(lag, fill_value=0.0)
