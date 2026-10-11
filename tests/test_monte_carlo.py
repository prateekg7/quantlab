"""Tests for Monte Carlo option pricer and variance reduction methods."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quantlab.pricing.black_scholes import bs_price
from quantlab.pricing.monte_carlo import MCResult, mc_price


def test_mc_result_dataclass_frozen() -> None:
    res = MCResult(price=10.0, std_error=0.1, ci_low=9.8, ci_high=10.2, n_paths=1000)
    assert res.price == 10.0
    assert res.std_error == 0.1
    assert res.ci_low == 9.8
    assert res.ci_high == 10.2
    assert res.n_paths == 1000

    with pytest.raises(FrozenInstanceError):
        res.price = 11.0  # type: ignore[misc]


def test_t_mc_1_seeded_rerun_is_bit_identical() -> None:
    """T-MC-1: Seeded rerun must produce bit-identical results (NF-4)."""
    res1 = mc_price(100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=50_000, seed=42)
    res2 = mc_price(100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=50_000, seed=42)

    assert res1.price == res2.price
    assert res1.std_error == res2.std_error
    assert res1.ci_low == res2.ci_low
    assert res1.ci_high == res2.ci_high
    assert res1.n_paths == res2.n_paths

    # Also test with antithetic and control variate
    res_av1 = mc_price(
        100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=50_000, seed=123, antithetic=True
    )
    res_av2 = mc_price(
        100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=50_000, seed=123, antithetic=True
    )
    assert res_av1.price == res_av2.price
    assert res_av1.std_error == res_av2.std_error


def test_t_mc_2_ci_coverage_200_runs() -> None:
    """T-MC-2: BS price lies inside MC 95% CI in roughly 93-97% of 200 seeded runs (F-8)."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs = bs_price(S, K, r, sigma, T, "call")

    n_runs = 200
    n_paths = 10_000
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 10_000_000, size=n_runs)
    inside_count = 0

    for seed in seeds:
        res = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=int(seed))
        if res.ci_low <= bs <= res.ci_high:
            inside_count += 1

    coverage = inside_count / n_runs
    # SPEC: BS price lies inside MC 95% CI in roughly 93-97% of 200 seeded runs
    assert 0.93 <= coverage <= 0.97, f"Coverage {coverage:.1%} outside expected 93-97% range"


def test_t_mc_3_error_shrinks_10x_for_100x_paths() -> None:
    """T-MC-3: 100x more paths cuts std error by roughly 10x (accept 7x to 14x) (F-8)."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    res_1k = mc_price(S, K, r, sigma, T, "call", n_paths=1_000, seed=42)
    res_100k = mc_price(S, K, r, sigma, T, "call", n_paths=100_000, seed=42)

    ratio = res_1k.std_error / res_100k.std_error
    assert 7.0 <= ratio <= 14.0, f"Error reduction ratio {ratio:.2f} not in [7, 14]"


def test_t_mc_4_antithetic_variance_reduction() -> None:
    """T-MC-4: Antithetic variates give lower std error than plain at equal n_paths (F-9)."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    n_paths = 50_000

    res_plain = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=99, antithetic=False)
    res_anti = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=99, antithetic=True)

    assert res_anti.std_error < res_plain.std_error
    # Typical standard error reduction for ATM option is around 20-40%
    assert res_anti.std_error / res_plain.std_error < 0.85

    # Check put as well
    res_put_plain = mc_price(S, K, r, sigma, T, "put", n_paths=n_paths, seed=99, antithetic=False)
    res_put_anti = mc_price(S, K, r, sigma, T, "put", n_paths=n_paths, seed=99, antithetic=True)
    assert res_put_anti.std_error < res_put_plain.std_error


def test_t_mc_4_control_variate_variance_reduction() -> None:
    """T-MC-4: Control variate gives lower std error than plain at equal n_paths (F-10)."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    n_paths = 50_000

    res_plain = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=99, control_variate=False)
    res_cv = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=99, control_variate=True)

    assert res_cv.std_error < res_plain.std_error
    # Asset price control variate is highly correlated with ATM call (correlation ~0.7-0.9)
    assert res_cv.std_error / res_plain.std_error < 0.70


def test_both_variance_reductions_combined() -> None:
    """Combining antithetic and control variate yields even lower standard error."""
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    n_paths = 50_000

    res_plain = mc_price(S, K, r, sigma, T, "call", n_paths=n_paths, seed=10)
    res_both = mc_price(
        S, K, r, sigma, T, "call", n_paths=n_paths, seed=10, antithetic=True, control_variate=True
    )

    assert res_both.std_error < res_plain.std_error
    bs = bs_price(S, K, r, sigma, T, "call")
    assert res_both.ci_low <= bs <= res_both.ci_high


def test_mc_put_call_parity() -> None:
    """C - P approx S*exp(-qT) - K*exp(-rT) within MC uncertainty."""
    S, K, r, sigma, T, q = 100.0, 95.0, 0.04, 0.25, 0.75, 0.02
    n_paths = 100_000

    call_res = mc_price(S, K, r, sigma, T, "call", q=q, n_paths=n_paths, seed=42)
    put_res = mc_price(S, K, r, sigma, T, "put", q=q, n_paths=n_paths, seed=42)

    lhs = call_res.price - put_res.price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    combined_se = np.sqrt(call_res.std_error**2 + put_res.std_error**2)

    assert abs(lhs - rhs) < 3.0 * combined_se


def test_mc_boundary_conditions() -> None:
    # At T=0, returns intrinsic with 0 std error
    call_t0 = mc_price(105.0, 100.0, 0.05, 0.20, 0.0, "call")
    assert call_t0.price == 5.0
    assert call_t0.std_error == 0.0
    assert call_t0.ci_low == 5.0
    assert call_t0.ci_high == 5.0

    put_t0 = mc_price(90.0, 100.0, 0.05, 0.20, 0.0, "put")
    assert put_t0.price == 10.0
    assert put_t0.std_error == 0.0

    # Deterministic limit sigma = 0
    call_sig0 = mc_price(100.0, 100.0, 0.05, 0.0, 1.0, "call")
    det_st = 100.0 * np.exp(0.05)
    expected_call = np.exp(-0.05) * (det_st - 100.0)
    assert np.isclose(call_sig0.price, expected_call)
    assert call_sig0.std_error == 0.0


def test_mc_input_validation() -> None:
    with pytest.raises(ValueError, match="Spot price S must be positive"):
        mc_price(0.0, 100.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Strike price K must be positive"):
        mc_price(100.0, -10.0, 0.05, 0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Volatility sigma must be non-negative"):
        mc_price(100.0, 100.0, 0.05, -0.20, 1.0, "call")

    with pytest.raises(ValueError, match="Time to maturity T must be non-negative"):
        mc_price(100.0, 100.0, 0.05, 0.20, -1.0, "call")

    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        mc_price(100.0, 100.0, 0.05, 0.20, 1.0, "forward")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n_paths must be at least 2"):
        mc_price(100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=1)
