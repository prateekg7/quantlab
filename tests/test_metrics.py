import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import annualized_vol, cagr, log_returns, sharpe, simple_returns


@pytest.fixture
def prices() -> pd.Series:
    return pd.Series(
        [100.0, 102.0, 101.0, 103.02, 103.02],
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )


def test_simple_returns_values(prices: pd.Series) -> None:
    r = simple_returns(prices)
    expected = [0.02, 101 / 102 - 1, 0.02, 0.0]
    assert r.to_numpy() == pytest.approx(expected)


def test_simple_returns_index_and_length(prices: pd.Series) -> None:
    r = simple_returns(prices)
    assert len(r) == len(prices) - 1
    assert r.index.equals(prices.index[1:])
    assert not r.isna().any()


def test_log_returns_values(prices: pd.Series) -> None:
    r = log_returns(prices)
    expected = [math.log(102 / 100), math.log(101 / 102), math.log(103.02 / 101), 0.0]
    assert r.to_numpy() == pytest.approx(expected)


def test_log_return_is_log1p_of_simple_return(prices: pd.Series) -> None:
    expected = np.log1p(simple_returns(prices))
    pd.testing.assert_series_equal(log_returns(prices), expected)


def test_inputs_are_not_mutated(prices: pd.Series) -> None:
    before = prices.copy()
    simple_returns(prices)
    log_returns(prices)
    pd.testing.assert_series_equal(prices, before)


@pytest.mark.parametrize("fn", [simple_returns, log_returns])
def test_needs_at_least_two_prices(fn: Callable[[pd.Series], pd.Series]) -> None:
    one = pd.Series([100.0], index=pd.date_range("2024-01-02", periods=1))
    with pytest.raises(ValueError, match="at least two"):
        fn(one)


@pytest.mark.parametrize("fn", [simple_returns, log_returns])
def test_non_positive_prices_raise(fn: Callable[[pd.Series], pd.Series]) -> None:
    bad = pd.Series(
        [100.0, 0.0, 101.0],
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
    with pytest.raises(ValueError, match="positive"):
        fn(bad)


def test_cagr(prices: pd.Series) -> None:
    r = simple_returns(prices)
    expected = math.pow(1.0302, 252 / 4) - 1
    assert cagr(r) == pytest.approx(expected)


def test_annualized_vol(prices: pd.Series) -> None:
    r = simple_returns(prices)
    expected_r = [0.02, 101 / 102 - 1, 0.02, 0.0]
    expected_vol = np.std(expected_r, ddof=1) * math.sqrt(252)
    assert annualized_vol(r) == pytest.approx(expected_vol)


def test_sharpe_zero_variance() -> None:
    r = pd.Series([0.01, 0.01, 0.01, 0.01])
    with pytest.warns(UserWarning, match="zero variance"):
        val = sharpe(r)
    assert math.isnan(val)


def test_sharpe(prices: pd.Series) -> None:
    r = simple_returns(prices)
    expected_r = np.array([0.02, 101 / 102 - 1, 0.02, 0.0])
    expected_vol = np.std(expected_r, ddof=1)
    expected_sharpe = np.mean(expected_r) / expected_vol * math.sqrt(252)
    assert sharpe(r) == pytest.approx(expected_sharpe)


@pytest.mark.parametrize("fn", [cagr, annualized_vol, sharpe])
def test_empty_input_raises(fn: Callable[..., float]) -> None:
    r = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        fn(r)
