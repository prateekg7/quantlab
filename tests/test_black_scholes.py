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


@pytest.mark.parametrize("q", [0.0, 0.03, 0.07])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
@pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
def test_put_call_parity(
    S: float, K: float, T: float, sigma: float, q: float, r: float = 0.05
) -> None:
    """T-BS-2: Put-call parity C - P = S e^{-qT} - K e^{-rT} holds to 1e-8."""
    call = bs_price(S, K, r, sigma, T, "call", q=q)
    put = bs_price(S, K, r, sigma, T, "put", q=q)

    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_put_call_parity_vectorized() -> None:
    """T-BS-2: Put-call parity with array inputs."""
    spots = np.array([80.0, 100.0, 120.0])
    calls = bs_price(spots, 100.0, 0.05, 0.2, 1.0, "call", q=0.02)
    puts = bs_price(spots, 100.0, 0.05, 0.2, 1.0, "put", q=0.02)

    lhs = calls - puts
    rhs = spots * np.exp(-0.02 * 1.0) - 100.0 * np.exp(-0.05 * 1.0)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8)


@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_price_bounds(S: float, K: float, q: float, r: float = 0.05, T: float = 1.0) -> None:
    """T-BS-3: No-arbitrage theoretical price bounds for calls and puts."""
    call = bs_price(S, K, r, 0.2, T, "call", q=q)
    put = bs_price(S, K, r, 0.2, T, "put", q=q)

    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    # Call bounds: max(S e^{-qT} - K e^{-rT}, 0) <= Call <= S e^{-qT}
    call_lower_bound = max(discounted_spot - discounted_strike, 0.0)
    assert call >= call_lower_bound - 1e-12
    assert call <= discounted_spot + 1e-12

    # Put bounds: max(K e^{-rT} - S e^{-qT}, 0) <= Put <= K e^{-rT}
    put_lower_bound = max(discounted_strike - discounted_spot, 0.0)
    assert put >= put_lower_bound - 1e-12
    assert put <= discounted_strike + 1e-12


def test_price_monotonicity_spot() -> None:
    """T-BS-3: Call price is strictly increasing in spot S; put price is strictly decreasing."""
    spots = [70.0, 90.0, 100.0, 110.0, 130.0]
    calls = [bs_price(s, 100.0, 0.05, 0.2, 1.0, "call") for s in spots]
    puts = [bs_price(s, 100.0, 0.05, 0.2, 1.0, "put") for s in spots]

    # Verify strictly increasing for calls
    assert all(c1 < c2 for c1, c2 in zip(calls[:-1], calls[1:], strict=True))
    # Verify strictly decreasing for puts
    assert all(p1 > p2 for p1, p2 in zip(puts[:-1], puts[1:], strict=True))


def test_price_monotonicity_volatility() -> None:
    """T-BS-3: Both call and put prices strictly increase in volatility sigma."""
    vols = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    calls = [bs_price(100.0, 100.0, 0.05, v, 1.0, "call") for v in vols]
    puts = [bs_price(100.0, 100.0, 0.05, v, 1.0, "put") for v in vols]

    assert all(c1 < c2 for c1, c2 in zip(calls[:-1], calls[1:], strict=True))
    assert all(p1 < p2 for p1, p2 in zip(puts[:-1], puts[1:], strict=True))


def test_bs_price_t_zero_intrinsic() -> None:
    """T-BS-4: T=0 returns exact intrinsic value."""
    # In the money
    assert bs_price(110.0, 100.0, 0.05, 0.2, 0.0, "call") == pytest.approx(10.0)
    assert bs_price(90.0, 100.0, 0.05, 0.2, 0.0, "put") == pytest.approx(10.0)

    # Out of the money
    assert bs_price(90.0, 100.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)
    assert bs_price(110.0, 100.0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)

    # At the money
    assert bs_price(100.0, 100.0, 0.05, 0.2, 0.0, "call") == pytest.approx(0.0)
    assert bs_price(100.0, 100.0, 0.05, 0.2, 0.0, "put") == pytest.approx(0.0)


def test_bs_price_t_approaching_zero_limit() -> None:
    """T-BS-4: As T -> 0, price smoothly converges to intrinsic value."""
    # ITM call (S=110, K=100 -> intrinsic 10)
    call_near_expiry = bs_price(110.0, 100.0, 0.05, 0.2, 1e-7, "call")
    assert call_near_expiry == pytest.approx(10.0, abs=1e-4)

    # ITM put (S=90, K=100 -> intrinsic 10)
    put_near_expiry = bs_price(90.0, 100.0, 0.05, 0.2, 1e-7, "put")
    assert put_near_expiry == pytest.approx(10.0, abs=1e-4)

    # OTM call (S=90, K=100 -> intrinsic 0)
    call_otm = bs_price(90.0, 100.0, 0.05, 0.2, 1e-7, "call")
    assert call_otm == pytest.approx(0.0, abs=1e-4)


def test_bs_price_deep_itm_otm_asymptotics() -> None:
    """T-BS-4: Deep ITM / OTM boundary behavior."""
    # Deep OTM call (S=1, K=100) -> 0
    assert bs_price(1.0, 100.0, 0.05, 0.2, 1.0, "call") == pytest.approx(0.0, abs=1e-6)
    # Deep ITM put (S=1, K=100) -> K * e^{-rT} - S
    assert bs_price(1.0, 100.0, 0.05, 0.2, 1.0, "put") == pytest.approx(
        100.0 * np.exp(-0.05 * 1.0) - 1.0, abs=1e-3
    )
    # As S -> 0+, put converges to discounted strike K * e^{-rT}
    assert bs_price(1e-6, 100.0, 0.05, 0.2, 1.0, "put") == pytest.approx(
        100.0 * np.exp(-0.05 * 1.0), abs=1e-3
    )

    # Deep ITM call (S=1000, K=100) -> S - K * e^{-rT}
    assert bs_price(1000.0, 100.0, 0.05, 0.2, 1.0, "call") == pytest.approx(
        1000.0 - 100.0 * np.exp(-0.05 * 1.0), abs=1e-3
    )
    # Deep OTM put (S=1000, K=100) -> 0
    assert bs_price(1000.0, 100.0, 0.05, 0.2, 1.0, "put") == pytest.approx(0.0, abs=1e-6)


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
def test_bs_price_invalid_inputs_scalar(
    S: float, K: float, r: float, sigma: float, T: float, option: str
) -> None:
    """T-BS-4: Invalid scalar inputs raise ValueError."""
    with pytest.raises(ValueError):
        bs_price(S, K, r, sigma, T, cast(Literal["call", "put"], option))


def test_bs_price_invalid_inputs_vectorized() -> None:
    """T-BS-4: Invalid array inputs raise ValueError."""
    # S has a non-positive element
    with pytest.raises(ValueError, match="S must be strictly positive"):
        bs_price(np.array([100.0, 0.0, 110.0]), 100.0, 0.05, 0.2, 1.0, "call")

    # K has a non-positive element
    with pytest.raises(ValueError, match="K must be strictly positive"):
        bs_price(100.0, np.array([100.0, -50.0]), 0.05, 0.2, 1.0, "call")

    # sigma has a non-positive element
    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        bs_price(100.0, 100.0, 0.05, np.array([0.2, -0.1]), 1.0, "call")
