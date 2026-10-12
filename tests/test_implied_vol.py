"""Tests for implied volatility calculation (T-27, F-14, T-IV-1, T-IV-2)."""

from typing import Literal

import pytest

from quantlab.pricing.black_scholes import bs_price
from quantlab.pricing.implied_vol import implied_vol


def test_t_iv_1_inversion_grid() -> None:
    """T-IV-1: implied_vol(bs_price(sigma=x)) ≈ x across a grid of strikes and maturities."""
    strikes = [80.0, 100.0, 120.0]
    maturities = [0.25, 1.0, 2.0]
    volatilities = [0.10, 0.20, 0.40, 0.80]
    options: tuple[Literal["call", "put"], ...] = ("call", "put")
    dividends = [0.0, 0.02]

    S = 100.0
    r = 0.05

    for K in strikes:
        for T in maturities:
            for sigma_true in volatilities:
                for q in dividends:
                    for opt in options:
                        px = float(bs_price(S, K, r, sigma_true, T, opt, q=q))
                        iv = implied_vol(px, S, K, r, T, opt, q=q, tol=1e-8)
                        assert iv == pytest.approx(sigma_true, abs=1e-6)


def test_t_iv_2_out_of_bounds_raises() -> None:
    """T-IV-2: Out-of-bounds market price raises ValueError."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0

    # For Call:
    # Lower bound: max(100 - 100*exp(-0.05), 0) ~ 4.877
    # Upper bound: 100.0
    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol(1.0, S, K, r, T, "call")  # Below intrinsic lower bound

    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol(105.0, S, K, r, T, "call")  # Above spot upper bound

    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol(0.0, S, K, r, T, "call")  # Zero price

    # For Put:
    # Lower bound: max(100*exp(-0.05) - 100, 0) = 0.0
    # Upper bound: 100*exp(-0.05) ~ 95.123
    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol(0.0, S, K, r, T, "put")  # Non-positive price

    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        implied_vol(100.0, S, K, r, T, "put")  # Above discounted strike upper bound


def test_bisection_fallback_extreme_vol() -> None:
    """Verify bisection fallback converges cleanly for high volatility or low vega."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0

    # High volatility (where Newton might step outside normal bounds)
    sigma_high = 2.50
    px_high = float(bs_price(S, K, r, sigma_high, T, "call"))
    iv_high = implied_vol(px_high, S, K, r, T, "call", tol=1e-7)
    assert iv_high == pytest.approx(sigma_high, abs=1e-4)

    # Low volatility
    sigma_low = 0.02
    px_low = float(bs_price(S, K, r, sigma_low, T, "call"))
    iv_low = implied_vol(px_low, S, K, r, T, "call", tol=1e-7)
    assert iv_low == pytest.approx(sigma_low, abs=1e-4)


def test_deep_out_of_the_money_vega_near_zero() -> None:
    """OTM option with very small vega falls back to bisection and recovers true vol."""
    # S=50, K=120, T=0.25 (deep OTM call)
    S, K, r, T = 50.0, 120.0, 0.03, 0.25
    sigma_true = 0.35
    px = float(bs_price(S, K, r, sigma_true, T, "call"))
    iv = implied_vol(px, S, K, r, T, "call", tol=1e-7)
    assert iv == pytest.approx(sigma_true, abs=1e-4)


def test_implied_vol_with_dividends() -> None:
    """Verify implied volatility correctly respects continuous dividend yield q."""
    S, K, r, sigma_true, T, q = 100.0, 100.0, 0.05, 0.30, 0.75, 0.04
    call_px = float(bs_price(S, K, r, sigma_true, T, "call", q=q))
    put_px = float(bs_price(S, K, r, sigma_true, T, "put", q=q))

    assert implied_vol(call_px, S, K, r, T, "call", q=q) == pytest.approx(sigma_true, abs=1e-6)
    assert implied_vol(put_px, S, K, r, T, "put", q=q) == pytest.approx(sigma_true, abs=1e-6)


def test_implied_vol_input_validation() -> None:
    """Verify input validation for invalid parameters."""
    with pytest.raises(ValueError, match="Spot price S must be positive"):
        implied_vol(10.0, 0.0, 100.0, 0.05, 1.0, "call")

    with pytest.raises(ValueError, match="Strike price K must be positive"):
        implied_vol(10.0, 100.0, 0.0, 0.05, 1.0, "call")

    with pytest.raises(ValueError, match="Time to maturity T must be positive"):
        implied_vol(10.0, 100.0, 100.0, 0.05, 0.0, "call")

    with pytest.raises(ValueError, match="Tolerance tol must be positive"):
        implied_vol(10.0, 100.0, 100.0, 0.05, 1.0, "call", tol=0.0)

    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        implied_vol(10.0, 100.0, 100.0, 0.05, 1.0, "call", max_iter=0)

    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        implied_vol(10.0, 100.0, 100.0, 0.05, 1.0, "strangle")  # type: ignore[arg-type]
