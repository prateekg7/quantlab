from typing import Literal, cast

import numpy as np
import pytest

from quantlab.pricing.black_scholes import bs_price


def test_bs_price_known_values() -> None:
    """T-BS-1: spot=100, strike=100, r=0.05, sigma=0.2, T=1.0."""
    call_price = bs_price(100.0, 100.0, 0.05, 0.2, 1.0, "call")
    put_price = bs_price(100.0, 100.0, 0.05, 0.2, 1.0, "put")

    assert call_price == pytest.approx(10.4506, abs=1e-3)
    assert put_price == pytest.approx(5.5735, abs=1e-3)


def test_bs_price_t_zero_intrinsic() -> None:
    """T=0 returns intrinsic value max(S - K, 0) or max(K - S, 0)."""
    # In the money
    assert bs_price(110.0, 100.0, 0.05, 0.2, 0.0, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.05, 0.2, 0.0, "put") == pytest.approx(10.0)

    # Out of the money
    assert bs_price(90.0, 100.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)
    assert bs_price(110.0, 100.0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)

    # At the money
    assert bs_price(100.0, 100.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)
    assert bs_price(100.0, 100.0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)


def test_bs_price_with_dividends() -> None:
    """Dividend yield q reduces call price and increases put price."""
    c_no_div = bs_price(100.0, 100.0, 0.05, 0.2, 1.0, "call", q=0.0)
    c_with_div = bs_price(100.0, 100.0, 0.05, 0.2, 1.0, "call", q=0.03)
    assert c_with_div < c_no_div

    p_no_div = bs_price(100.0, 100.0, 0.05, 0.2, 1.0, "put", q=0.0)
    p_with_div = bs_price(100.0, 100.0, 0.05, 0.2, 1.0, "put", q=0.03)
    assert p_with_div > p_no_div


def test_bs_price_vectorized() -> None:
    """Vectorized inputs for S and sigma."""
    spots = np.array([90.0, 100.0, 110.0])
    call_prices = bs_price(spots, 100.0, 0.05, 0.2, 1.0, "call")
    assert isinstance(call_prices, np.ndarray)
    assert len(call_prices) == 3

    # Element-wise agreement
    for s, p in zip(spots, call_prices, strict=True):
        assert p == pytest.approx(bs_price(float(s), 100.0, 0.05, 0.2, 1.0, "call"))


@pytest.mark.parametrize(
    ("S", "K", "r", "sigma", "T", "option"),
    [
        (100.0, 100.0, 0.05, 0.2, -0.1, "call"),  # T < 0
        (100.0, 100.0, 0.05, 0.0, 1.0, "call"),  # sigma == 0
        (100.0, 100.0, 0.05, -0.2, 1.0, "call"),  # sigma < 0
        (0.0, 100.0, 0.05, 0.2, 1.0, "call"),  # S == 0
        (-10.0, 100.0, 0.05, 0.2, 1.0, "call"),  # S < 0
        (100.0, 0.0, 0.05, 0.2, 1.0, "call"),  # K == 0
        (100.0, -10.0, 0.05, 0.2, 1.0, "call"),  # K < 0
        (100.0, 100.0, 0.05, 0.2, 1.0, "other"),  # invalid option type
    ],
)
def test_bs_price_invalid_inputs(
    S: float, K: float, r: float, sigma: float, T: float, option: str
) -> None:
    with pytest.raises(ValueError):
        bs_price(S, K, r, sigma, T, cast(Literal["call", "put"], option))
