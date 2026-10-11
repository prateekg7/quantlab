import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import compute_costs, compute_turnover
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
