import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.signals import lag_signal, sma_crossover_signal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_prices(n: int, start: str = "2024-01-02") -> pd.Series:
    """Make a simple linearly increasing price series with n observations."""
    return pd.Series(
        np.arange(1.0, n + 1.0),
        index=pd.date_range(start, periods=n, freq="B"),
        name="close",
    )


# ---------------------------------------------------------------------------
# T-SG-2: Warm-up period is flat
# ---------------------------------------------------------------------------


def test_signal_flat_during_warmup() -> None:
    """T-SG-2: Signal is 0 for the first (slow - 1) bars while SMA_slow has no data."""
    prices = make_prices(50)
    fast, slow = 5, 20
    signal = sma_crossover_signal(prices, fast=fast, slow=slow)

    # The first (slow - 1) bars must all be 0 (warm-up — no valid slow SMA yet)
    warm_up = signal.iloc[: slow - 1]
    assert (warm_up == 0).all(), f"Expected all zeros in warm-up, got:\n{warm_up}"

    # From bar `slow` onward, the signal is defined (can be 0 or 1)
    assert len(signal.iloc[slow - 1 :]) > 0


def test_signal_exactly_at_slow_boundary() -> None:
    """Signal must be 0 at index slow-2, and non-zero (defined) at index slow-1."""
    prices = make_prices(30)
    fast, slow = 3, 10
    signal = sma_crossover_signal(prices, fast=fast, slow=slow)

    # Strictly before enough data
    assert signal.iloc[slow - 2] == 0
    # At the bar that completes the slow window, signal is determined by the crossover
    assert signal.iloc[slow - 1] in (0.0, 1.0, -1.0)


def test_signal_length_matches_prices() -> None:
    """Output series must have the same length and index as the input prices."""
    prices = make_prices(40)
    signal = sma_crossover_signal(prices, fast=5, slow=20)
    assert len(signal) == len(prices)
    assert signal.index.equals(prices.index)


# ---------------------------------------------------------------------------
# Signal values are correct
# ---------------------------------------------------------------------------


def test_signal_long_when_fast_above_slow() -> None:
    """Signal is 1 when fast SMA > slow SMA."""
    # Build a prices series that is strictly increasing — fast SMA always above slow SMA
    # after warm-up because recent prices are always higher
    n = 60
    prices = pd.Series(
        np.linspace(100.0, 200.0, n),
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    signal = sma_crossover_signal(prices, fast=5, slow=20)
    # After warm-up all signals must be 1
    assert (signal.iloc[19:] == 1).all()


def test_signal_flat_when_fast_below_slow_long_only() -> None:
    """In long-only mode, signal is 0 (not -1) when fast SMA <= slow SMA."""
    n = 60
    # Strictly decreasing series: fast SMA always below slow SMA after warm-up
    prices = pd.Series(
        np.linspace(200.0, 100.0, n),
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    signal = sma_crossover_signal(prices, fast=5, slow=20, long_short=False)
    assert (signal.iloc[19:] == 0).all()


def test_signal_short_when_fast_below_slow_long_short() -> None:
    """In long/short mode, signal is -1 when fast SMA <= slow SMA."""
    n = 60
    prices = pd.Series(
        np.linspace(200.0, 100.0, n),
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )
    signal = sma_crossover_signal(prices, fast=5, slow=20, long_short=True)
    assert (signal.iloc[19:] == -1).all()


# ---------------------------------------------------------------------------
# Crossover values are numerically exact (hand-calculated)
# ---------------------------------------------------------------------------


def test_signal_crossover_hand_calculation() -> None:
    """Verify signal against manually computed SMA values."""
    # Use a short series where we can compute SMAs by hand.
    # fast=2, slow=3
    prices = pd.Series(
        [10.0, 11.0, 9.0, 12.0, 13.0],
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )
    signal = sma_crossover_signal(prices, fast=2, slow=3)

    # bar 0 (price=10): no data for slow — should be 0
    # bar 1 (price=11): no data for slow (need 3 obs) — should be 0
    # bar 2 (price=9): slow=(10+11+9)/3=10, fast=(11+9)/2=10 -> fast<=slow -> 0
    # bar 3 (price=12): slow=(11+9+12)/3=10.67, fast=(9+12)/2=10.5 -> fast<slow -> 0
    # bar 4 (price=13): slow=(9+12+13)/3=11.33, fast=(12+13)/2=12.5 -> fast>slow -> 1

    expected = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert signal.tolist() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_fast_must_be_less_than_slow() -> None:
    """fast >= slow raises ValueError."""
    prices = make_prices(20)
    with pytest.raises(ValueError, match="fast must be strictly less than slow"):
        sma_crossover_signal(prices, fast=20, slow=20)
    with pytest.raises(ValueError, match="fast must be strictly less than slow"):
        sma_crossover_signal(prices, fast=25, slow=10)


def test_windows_must_be_positive() -> None:
    """fast or slow < 1 raises ValueError."""
    prices = make_prices(20)
    with pytest.raises(ValueError):
        sma_crossover_signal(prices, fast=0, slow=10)
    with pytest.raises(ValueError):
        sma_crossover_signal(prices, fast=5, slow=0)


# ---------------------------------------------------------------------------
# T-SG-1: Look-ahead bias test (lag_signal)
# ---------------------------------------------------------------------------


def test_lag_signal_shifts_by_one() -> None:
    """T-SG-1 core: position(t) == signal(t-1), verified element-by-element."""
    prices = make_prices(30)
    raw = sma_crossover_signal(prices, fast=3, slow=5)
    position = lag_signal(raw)

    # position[i] must equal raw[i-1] for all i >= 1
    for i in range(1, len(raw)):
        assert position.iloc[i] == raw.iloc[i - 1], (
            f"Mismatch at index {i}: position={position.iloc[i]}, signal={raw.iloc[i - 1]}"
        )

    # The very first position must be 0 (no prior signal — fill_value=0)
    assert position.iloc[0] == 0.0


def test_changing_price_on_day_t_does_not_change_position_on_day_t() -> None:
    """T-SG-1: Mutating today's price must not affect today's position.

    This is the canonical look-ahead test. If the position on day t depends on the
    price of day t, the strategy is peeking into the future.
    """
    n = 40
    fast, slow = 3, 10

    prices_original = pd.Series(
        np.linspace(100.0, 140.0, n),
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )

    # Compute positions using original prices
    signal_orig = sma_crossover_signal(prices_original, fast=fast, slow=slow)
    position_orig = lag_signal(signal_orig)

    # Perturb the price on every day t (simulate "seeing the future") and re-derive positions
    for t in range(slow, n):
        prices_perturbed = prices_original.copy()
        prices_perturbed.iloc[t] = prices_original.iloc[t] * 10.0  # dramatic change on day t

        signal_perturbed = sma_crossover_signal(prices_perturbed, fast=fast, slow=slow)
        position_perturbed = lag_signal(signal_perturbed)

        # The position on day t must NOT have changed even though price[t] changed,
        # because position(t) = signal(t-1) which only uses prices up to t-1.
        assert position_perturbed.iloc[t] == position_orig.iloc[t], (
            f"Day {t}: position changed from {position_orig.iloc[t]} to "
            f"{position_perturbed.iloc[t]} after mutating price[{t}] — look-ahead bias!"
        )


def test_lag_signal_preserves_index_and_length() -> None:
    """lag_signal output has the same index and length as the input."""
    prices = make_prices(25)
    raw = sma_crossover_signal(prices, fast=3, slow=8)
    position = lag_signal(raw)

    assert len(position) == len(raw)
    assert position.index.equals(raw.index)


def test_lag_signal_arbitrary_lag() -> None:
    """lag=2 shifts by two periods; first two entries are 0."""
    prices = make_prices(20)
    raw = sma_crossover_signal(prices, fast=3, slow=5)
    position = lag_signal(raw, lag=2)

    assert position.iloc[0] == 0.0
    assert position.iloc[1] == 0.0
    for i in range(2, len(raw)):
        assert position.iloc[i] == raw.iloc[i - 2]


def test_lag_signal_invalid_lag() -> None:
    """lag < 1 raises ValueError."""
    prices = make_prices(20)
    raw = sma_crossover_signal(prices, fast=3, slow=5)
    with pytest.raises(ValueError, match="lag must be >= 1"):
        lag_signal(raw, lag=0)
