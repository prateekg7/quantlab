"""Tests for Cox-Ross-Rubinstein binomial tree option pricing (T-25, F-12, F-13, T-BT-1, T-BT-2)."""

import math

import pytest

from quantlab.pricing.binomial import binomial_price
from quantlab.pricing.black_scholes import bs_price


def test_t_bt_1_convergence_error_shrinks() -> None:
    """T-BT-1: Binomial price error vs Black-Scholes shrinks as steps grow (200 -> 1000)."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    bs_call = float(bs_price(S, K, r, sigma, T, "call"))
    bs_put = float(bs_price(S, K, r, sigma, T, "put"))

    # Coarse grid vs fine grid
    bt_call_coarse = binomial_price(S, K, r, sigma, T, "call", steps=50)
    bt_call_200 = binomial_price(S, K, r, sigma, T, "call", steps=200)
    bt_call_1000 = binomial_price(S, K, r, sigma, T, "call", steps=1000)

    bt_put_coarse = binomial_price(S, K, r, sigma, T, "put", steps=50)
    bt_put_200 = binomial_price(S, K, r, sigma, T, "put", steps=200)
    bt_put_1000 = binomial_price(S, K, r, sigma, T, "put", steps=1000)

    err_call_coarse = abs(bt_call_coarse - bs_call)
    err_call_200 = abs(bt_call_200 - bs_call)
    err_call_1000 = abs(bt_call_1000 - bs_call)

    err_put_coarse = abs(bt_put_coarse - bs_put)
    err_put_200 = abs(bt_put_200 - bs_put)
    err_put_1000 = abs(bt_put_1000 - bs_put)

    # 1000 steps should be strictly more accurate than 50 steps and within 0.005 of BS
    assert err_call_1000 < err_call_coarse
    assert err_put_1000 < err_put_coarse
    assert err_call_1000 < 0.005
    assert err_put_1000 < 0.005
    assert err_call_200 < 0.015
    assert err_put_200 < 0.015


def test_t_bt_2_american_properties() -> None:
    """T-BT-2: American put >= European put; American call (q=0) equals European call."""
    configs = [
        (100.0, 100.0, 0.05, 0.20, 1.0),  # ATM
        (80.0, 100.0, 0.05, 0.20, 1.0),  # ITM put / OTM call
        (120.0, 100.0, 0.05, 0.20, 1.0),  # OTM put / ITM call
        (60.0, 100.0, 0.08, 0.30, 2.0),  # Deep ITM put
    ]

    for S, K, r, sigma, T in configs:
        # 1. American call with q=0 equals European call (early exercise is never optimal)
        eur_call = binomial_price(S, K, r, sigma, T, "call", q=0.0, steps=300, american=False)
        ame_call = binomial_price(S, K, r, sigma, T, "call", q=0.0, steps=300, american=True)
        assert ame_call == pytest.approx(eur_call, abs=1e-6)

        # 2. American put >= European put (early exercise has non-negative option value)
        eur_put = binomial_price(S, K, r, sigma, T, "put", q=0.0, steps=300, american=False)
        ame_put = binomial_price(S, K, r, sigma, T, "put", q=0.0, steps=300, american=True)
        assert ame_put >= eur_put - 1e-9


def test_american_put_early_exercise_premium() -> None:
    """Deep in-the-money American put has strictly positive early exercise premium."""
    # S=70, K=100, r=0.08, sigma=0.15, T=1.0: intrinsic is 30.0
    S, K, r, sigma, T = 70.0, 100.0, 0.08, 0.15, 1.0
    eur_put = binomial_price(S, K, r, sigma, T, "put", steps=400, american=False)
    ame_put = binomial_price(S, K, r, sigma, T, "put", steps=400, american=True)

    premium = ame_put - eur_put
    assert premium > 0.50  # Substantial early exercise premium
    assert ame_put >= K - S  # Must be at least intrinsic value


def test_american_call_with_dividends() -> None:
    """With high dividend yield q > r, American call can have an early exercise premium."""
    S, K, r, sigma, T, q = 120.0, 100.0, 0.02, 0.20, 1.0, 0.10
    eur_call = binomial_price(S, K, r, sigma, T, "call", q=q, steps=400, american=False)
    ame_call = binomial_price(S, K, r, sigma, T, "call", q=q, steps=400, american=True)

    # When dividend yield is high, early exercise before dividend capture is optimal
    assert ame_call > eur_call


def test_binomial_put_call_parity() -> None:
    """European binomial tree satisfies discrete put-call parity."""
    S, K, r, sigma, T, q = 100.0, 95.0, 0.04, 0.25, 0.5, 0.01
    call_px = binomial_price(S, K, r, sigma, T, "call", q=q, steps=500, american=False)
    put_px = binomial_price(S, K, r, sigma, T, "put", q=q, steps=500, american=False)

    expected_diff = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert call_px - put_px == pytest.approx(expected_diff, abs=1e-3)


def test_binomial_boundary_at_expiry() -> None:
    """At T=0 expiry, binomial_price returns exact intrinsic value."""
    assert binomial_price(110.0, 100.0, 0.05, 0.20, 0.0, "call") == 10.0
    assert binomial_price(90.0, 100.0, 0.05, 0.20, 0.0, "call") == 0.0
    assert binomial_price(90.0, 100.0, 0.05, 0.20, 0.0, "put") == 10.0
    assert binomial_price(110.0, 100.0, 0.05, 0.20, 0.0, "put") == 0.0


def test_binomial_p_outside_bounds_raises() -> None:
    """Coarse tree with extreme parameters leads to p outside [0, 1] and raises ValueError."""
    # With T=1, steps=1: dt=1. r=1.5, q=0.0, sigma=0.01:
    # u = exp(0.01) ~ 1.01005, d = 0.99005. drift = exp(1.5) ~ 4.48.
    # p = (4.48 - 0.99) / (1.01 - 0.99) ~ 174.5 >> 1.0
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        binomial_price(100.0, 100.0, 1.5, 0.01, 1.0, "call", steps=1)


def test_binomial_input_validation() -> None:
    """Verify input validation for invalid parameters."""
    with pytest.raises(ValueError, match="Spot price S must be positive"):
        binomial_price(0.0, 100.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Strike price K must be positive"):
        binomial_price(100.0, -10.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Volatility sigma must be positive"):
        binomial_price(100.0, 100.0, 0.05, 0.0, 1.0, "call")

    with pytest.raises(ValueError, match="Time to maturity T must be non-negative"):
        binomial_price(100.0, 100.0, 0.05, 0.20, -0.5, "call")

    with pytest.raises(ValueError, match="Number of steps must be at least 1"):
        binomial_price(100.0, 100.0, 0.05, 0.20, 1.0, "call", steps=0)

    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        binomial_price(100.0, 100.0, 0.05, 0.20, 1.0, "binary")  # type: ignore[arg-type]
