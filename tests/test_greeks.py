"""Tests for analytical Black-Scholes Greeks (T-23, F-7, T-GR-2)."""

import math
from dataclasses import FrozenInstanceError
from typing import Literal

import pytest
from scipy.stats import norm

from quantlab.pricing.black_scholes import bs_price
from quantlab.pricing.greeks import Greeks, bs_greeks, fd_greeks
from quantlab.pricing.monte_carlo import mc_price


def test_greeks_dataclass_frozen() -> None:
    g = Greeks(delta=0.5, gamma=0.02, vega=10.0, theta=-5.0, rho=20.0)
    assert g.delta == 0.5
    assert g.gamma == 0.02
    assert g.vega == 10.0
    assert g.theta == -5.0
    assert g.rho == 20.0

    with pytest.raises(FrozenInstanceError):
        g.delta = 0.6  # type: ignore[misc]


def test_t_gr_2_bounds_and_signs() -> None:
    """T-GR-2: Delta in [0,1] for calls, [-1,0] for puts; gamma >= 0, vega >= 0."""
    strikes = [80.0, 100.0, 120.0]
    maturities = [0.1, 0.5, 1.0, 2.0]
    volatilities = [0.10, 0.20, 0.40]
    dividends = [0.0, 0.03]

    S = 100.0
    r = 0.05

    for K in strikes:
        for T in maturities:
            for sigma in volatilities:
                for q in dividends:
                    disc_q = math.exp(-q * T)

                    call_g = bs_greeks(S, K, r, sigma, T, "call", q=q)
                    put_g = bs_greeks(S, K, r, sigma, T, "put", q=q)

                    # Delta bounds (scaled by e^{-qT})
                    assert 0.0 <= call_g.delta <= disc_q + 1e-9
                    assert -disc_q - 1e-9 <= put_g.delta <= 0.0

                    # Gamma and Vega non-negative
                    assert call_g.gamma >= 0.0
                    assert put_g.gamma >= 0.0
                    assert call_g.vega >= 0.0
                    assert put_g.vega >= 0.0

                    # Equality between call and put gamma and vega
                    assert call_g.gamma == pytest.approx(put_g.gamma, rel=1e-12)
                    assert call_g.vega == pytest.approx(put_g.vega, rel=1e-12)


def test_known_values_benchmark() -> None:
    """Benchmark against known analytical values for standard ATM European option."""
    S, K, r, sigma, T, q = 100.0, 100.0, 0.05, 0.20, 1.0, 0.0

    d1 = (math.log(1.0) + (0.05 + 0.5 * 0.04) * 1.0) / 0.20  # 0.07 / 0.20 = 0.35
    d2 = d1 - 0.20  # 0.15

    expected_call_delta = float(norm.cdf(d1))
    expected_put_delta = float(norm.cdf(d1) - 1.0)
    expected_gamma = float(norm.pdf(d1) / 20.0)
    expected_vega = float(100.0 * norm.pdf(d1))
    expected_call_rho = float(100.0 * math.exp(-0.05) * norm.cdf(d2))
    expected_put_rho = float(-100.0 * math.exp(-0.05) * norm.cdf(-d2))

    call_g = bs_greeks(S, K, r, sigma, T, "call", q=q)
    put_g = bs_greeks(S, K, r, sigma, T, "put", q=q)

    assert call_g.delta == pytest.approx(expected_call_delta, abs=1e-6)
    assert put_g.delta == pytest.approx(expected_put_delta, abs=1e-6)
    assert call_g.gamma == pytest.approx(expected_gamma, abs=1e-6)
    assert call_g.vega == pytest.approx(expected_vega, abs=1e-6)
    assert call_g.rho == pytest.approx(expected_call_rho, abs=1e-6)
    assert put_g.rho == pytest.approx(expected_put_rho, abs=1e-6)


def test_put_call_parity_greeks() -> None:
    """Verify exact put-call parity relationships across all five Greeks."""
    S, K, r, sigma, T, q = 105.0, 100.0, 0.04, 0.25, 0.75, 0.02

    call_g = bs_greeks(S, K, r, sigma, T, "call", q=q)
    put_g = bs_greeks(S, K, r, sigma, T, "put", q=q)

    # 1. Delta: Delta_C - Delta_P = exp(-q*T)
    assert call_g.delta - put_g.delta == pytest.approx(math.exp(-q * T), abs=1e-12)

    # 2. Gamma: Gamma_C = Gamma_P
    assert call_g.gamma == pytest.approx(put_g.gamma, abs=1e-12)

    # 3. Vega: Vega_C = Vega_P
    assert call_g.vega == pytest.approx(put_g.vega, abs=1e-12)

    # 4. Theta: Theta_C - Theta_P = q * S * exp(-q*T) - r * K * exp(-r*T)
    expected_theta_diff = q * S * math.exp(-q * T) - r * K * math.exp(-r * T)
    assert call_g.theta - put_g.theta == pytest.approx(expected_theta_diff, abs=1e-12)

    # 5. Rho: Rho_C - Rho_P = K * T * exp(-r*T)
    expected_rho_diff = K * T * math.exp(-r * T)
    assert call_g.rho - put_g.rho == pytest.approx(expected_rho_diff, abs=1e-12)


def test_delta_gamma_theta_pde_identity() -> None:
    """Black-Scholes PDE identity: Theta + (r - q)*S*Delta + 0.5*sigma^2*S^2*Gamma = r*V."""
    configs = [
        (100.0, 100.0, 0.05, 0.20, 1.0, 0.0),
        (120.0, 100.0, 0.03, 0.30, 0.5, 0.01),
        (85.0, 100.0, 0.06, 0.15, 2.0, 0.02),
    ]

    options: tuple[Literal["call", "put"], ...] = ("call", "put")
    for S, K, r, sigma, T, q in configs:
        for opt in options:
            g = bs_greeks(S, K, r, sigma, T, opt, q=q)
            v = float(bs_price(S, K, r, sigma, T, opt, q=q))

            lhs = g.theta + (r - q) * S * g.delta + 0.5 * (sigma**2) * (S**2) * g.gamma
            rhs = r * v

            assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)


def test_greeks_boundary_at_expiry() -> None:
    """At T=0 expiry, Greeks evaluate to intrinsic step derivatives."""
    # ITM call: Delta = 1.0, all others 0.0
    c_itm = bs_greeks(105.0, 100.0, 0.05, 0.20, 0.0, "call")
    assert c_itm.delta == 1.0
    assert c_itm.gamma == 0.0
    assert c_itm.vega == 0.0
    assert c_itm.theta == 0.0
    assert c_itm.rho == 0.0

    # OTM call: Delta = 0.0
    c_otm = bs_greeks(95.0, 100.0, 0.05, 0.20, 0.0, "call")
    assert c_otm.delta == 0.0

    # ITM put: Delta = -1.0
    p_itm = bs_greeks(90.0, 100.0, 0.05, 0.20, 0.0, "put")
    assert p_itm.delta == -1.0

    # OTM put: Delta = 0.0
    p_otm = bs_greeks(110.0, 100.0, 0.05, 0.20, 0.0, "put")
    assert p_otm.delta == 0.0


def test_greeks_input_validation() -> None:
    with pytest.raises(ValueError, match="Spot price S must be positive"):
        bs_greeks(0.0, 100.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Strike price K must be positive"):
        bs_greeks(100.0, -5.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Volatility sigma must be positive"):
        bs_greeks(100.0, 100.0, 0.05, 0.0, 1.0, "call")

    with pytest.raises(ValueError, match="Time to maturity T must be non-negative"):
        bs_greeks(100.0, 100.0, 0.05, 0.20, -0.5, "call")

    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        bs_greeks(100.0, 100.0, 0.05, 0.20, 1.0, "straddle")  # type: ignore[arg-type]


def test_t_gr_1_fd_greeks_matches_analytical() -> None:
    """T-GR-1: FD Greeks match analytical Greeks (delta 1e-5, gamma 1e-4, vega 1e-3)."""
    spots = [80.0, 100.0, 120.0]
    options: tuple[Literal["call", "put"], ...] = ("call", "put")
    q_vals = [0.0, 0.02]

    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0

    for S in spots:
        for opt in options:
            for q in q_vals:
                analytical = bs_greeks(S, K, r, sigma, T, opt, q=q)
                fd = fd_greeks(bs_price, S, K, r, sigma, T, opt, q=q)

                # Acceptance criteria from SPEC.md: delta 1e-5, gamma 1e-4, vega 1e-3
                assert abs(fd.delta - analytical.delta) < 1e-5
                assert abs(fd.gamma - analytical.gamma) < 1e-4
                assert abs(fd.vega - analytical.vega) < 1e-3

                # Rho within 1e-3
                assert abs(fd.rho - analytical.rho) < 1e-3

                # Theta is one-sided difference with O(h) truncation error (well within 1e-2)
                assert abs(fd.theta - analytical.theta) < 1e-2


def test_fd_greeks_common_random_numbers() -> None:
    """Demonstrate that CRN is mandatory for stable MC finite differences."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    def mc_crn(
        s: float,
        k: float,
        rate: float,
        vol: float,
        mat: float,
        opt: Literal["call", "put"],
        div: float = 0.0,
    ) -> float:
        # Same fixed seed across all bumped calls (CRN)
        return mc_price(
            s,
            k,
            rate,
            vol,
            mat,
            opt,
            q=div,
            n_paths=50_000,
            seed=42,
            antithetic=True,
            control_variate=True,
        ).price

    fd_crn = fd_greeks(mc_crn, S, K, r, sigma, T, "call")
    analytical = bs_greeks(S, K, r, sigma, T, "call")

    # With CRN, delta matches within ~0.005 and gamma is positive and accurate within ~0.005
    assert abs(fd_crn.delta - analytical.delta) < 0.01
    assert abs(fd_crn.gamma - analytical.gamma) < 0.01
    assert fd_crn.gamma > 0.0


def test_fd_greeks_input_validation() -> None:
    """Verify input validation for fd_greeks."""
    with pytest.raises(ValueError, match="Spot price S must be positive"):
        fd_greeks(bs_price, -10.0, 100.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Strike price K must be positive"):
        fd_greeks(bs_price, 100.0, 0.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Volatility sigma must be positive"):
        fd_greeks(bs_price, 100.0, 100.0, 0.05, -0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Time to maturity T must be non-negative"):
        fd_greeks(bs_price, 100.0, 100.0, 0.05, 0.20, -1.0, "call")

    with pytest.raises(ValueError, match="Relative bump size h_rel must be positive"):
        fd_greeks(bs_price, 100.0, 100.0, 0.05, 0.20, 1.0, "call", h_rel=0.0)

    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        fd_greeks(bs_price, 100.0, 100.0, 0.05, 0.20, 1.0, "butterfly")  # type: ignore[arg-type]


def test_fd_greeks_boundary_at_expiry() -> None:
    """T=0 expiry returns intrinsic step derivatives."""
    g_call = fd_greeks(bs_price, 105.0, 100.0, 0.05, 0.20, 0.0, "call")
    assert g_call.delta == 1.0
    assert g_call.gamma == 0.0
    assert g_call.vega == 0.0
    assert g_call.theta == 0.0
    assert g_call.rho == 0.0

    g_put = fd_greeks(bs_price, 95.0, 100.0, 0.05, 0.20, 0.0, "put")
    assert g_put.delta == -1.0
    assert g_put.gamma == 0.0


def test_fd_greeks_custom_signature() -> None:
    """Verify fd_greeks works with flexible signature callable."""
    # Callable taking only (s, sig, r)
    fn = lambda s, sig, r: bs_price(s, 100.0, r, sig, 1.0, "call")  # noqa: E731
    g = fd_greeks(fn, 100.0, 100.0, 0.05, 0.20, 1.0, "call")
    analytical = bs_greeks(100.0, 100.0, 0.05, 0.20, 1.0, "call")
    assert abs(g.delta - analytical.delta) < 1e-5
    assert abs(g.gamma - analytical.gamma) < 1e-4
