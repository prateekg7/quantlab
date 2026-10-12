"""Analytical and finite-difference option Greeks."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from scipy.stats import norm


@dataclass(frozen=True)
class Greeks:
    """Analytical or numerical sensitivity Greeks for an option.

    Attributes:
        delta: First derivative of option price with respect to spot price S (dV/dS).
        gamma: Second derivative of option price with respect to spot price S (d^2V/dS^2).
        vega: First derivative with respect to volatility sigma (dV/dsigma), per 1.00 vol.
        theta: Time decay derivative (dV/dt = -dV/dT), per calendar year.
        rho: First derivative with respect to interest rate r (dV/dr), per 1.00 rate.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bs_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: Literal["call", "put"],
    q: float = 0.0,
) -> Greeks:
    r"""Compute analytical Black-Scholes Greeks for European call or put options.

    Formulas (continuous dividend yield q):
        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        Delta:
            Call:  exp(-q * T) * N(d1)
            Put:   exp(-q * T) * (N(d1) - 1) = -exp(-q * T) * N(-d1)

        Gamma:
            Both:  exp(-q * T) * phi(d1) / (S * sigma * sqrt(T))

        Vega:
            Both:  S * exp(-q * T) * phi(d1) * sqrt(T)
            (Convention: per 1.00 change in sigma)

        Theta (dV/dt = -dV/dT, per calendar year):
            Call: -[S * exp(-q * T) * phi(d1) * sigma] / (2 * sqrt(T))
                  - r * K * exp(-r * T) * N(d2) + q * S * exp(-q * T) * N(d1)
            Put:  -[S * exp(-q * T) * phi(d1) * sigma] / (2 * sqrt(T))
                  + r * K * exp(-r * T) * N(-d2) - q * S * exp(-q * T) * N(-d1)

        Rho:
            Call:  K * T * exp(-r * T) * N(d2)
            Put:  -K * T * exp(-r * T) * N(-d2)

    Relationship:
        Theta + (r - q) * S * Delta + 0.5 * sigma^2 * S^2 * Gamma = r * V
        (Black-Scholes PDE / Delta-Gamma-Theta relationship)

    Parameters:
        S: Spot price of the underlying asset (S > 0).
        K: Strike price of the option (K > 0).
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility (sigma > 0).
        T: Time to expiry in years (T >= 0).
        option: 'call' or 'put'.
        q: Continuous dividend yield (default 0.0).

    Returns:
        Greeks dataclass containing delta, gamma, vega, theta, rho.

    Raises:
        ValueError: If S <= 0, K <= 0, sigma <= 0, T < 0, or option not in ('call', 'put').
    """
    if S <= 0.0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0.0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if sigma <= 0.0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")
    if T < 0.0:
        raise ValueError(f"Time to maturity T must be non-negative, got {T}")
    if option not in ("call", "put"):
        raise ValueError(f"Option type must be 'call' or 'put', got {option!r}")

    # Boundary at expiry T = 0
    if T == 0.0:
        if option == "call":
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = math.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    phi_d1 = float(norm.pdf(d1))
    n_d1 = float(norm.cdf(d1))
    n_d2 = float(norm.cdf(d2))

    gamma = float(disc_q * phi_d1 / (S * vol_sqrt_t))
    vega = float(S * disc_q * phi_d1 * sqrt_t)
    theta_decay = -(S * disc_q * phi_d1 * sigma) / (2.0 * sqrt_t)

    if option == "call":
        delta = float(disc_q * n_d1)
        theta = float(theta_decay - r * K * disc_r * n_d2 + q * S * disc_q * n_d1)
        rho = float(K * T * disc_r * n_d2)
    else:
        n_minus_d1 = float(norm.cdf(-d1))
        n_minus_d2 = float(norm.cdf(-d2))
        delta = float(-disc_q * n_minus_d1)
        theta = float(theta_decay + r * K * disc_r * n_minus_d2 - q * S * disc_q * n_minus_d1)
        rho = float(-K * T * disc_r * n_minus_d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def fd_greeks(
    price_fn: Callable[..., Any],
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: Literal["call", "put"] = "call",
    q: float = 0.0,
    h_rel: float = 1e-4,
) -> Greeks:
    r"""Compute option Greeks via central finite differences (F-11).

    Formulas:
        Delta = (V(S + h_S) - V(S - h_S)) / (2 * h_S),  where h_S = h_rel * S
        Gamma = (V(S + h_S) - 2 * V(S) + V(S - h_S)) / (h_S^2)
        Vega  = (V(sigma + h_sigma) - V(sigma - h_sigma)) / (2 * h_sigma)
        Rho   = (V(r + h_r) - V(r - h_r)) / (2 * h_r),  h_r = max(h_rel * |r|, 1e-5)
        Theta = (V(T - h_T) - V(T)) / h_T,  h_T = min(h_rel, T / 2) (one-sided time decay)

    Assumptions & Notes:
        - Central difference yields truncation error O(h^2) for Delta, Gamma, Vega, Rho.
        - Theta uses a backward step in time to maturity T (advancing calendar time t),
          yielding truncation error O(h_T).
        - If price_fn is a Monte Carlo simulator, Common Random Numbers (CRN) must be used
          (same seed for all evaluations), otherwise finite-difference Gamma is dominated by noise.

    Parameters:
        price_fn: Option pricing callable returning float or an object with a .price attribute.
            Can accept arguments (S, K, r, sigma, T, option, q) or subsets thereof.
        S: Spot price of underlying asset (S > 0).
        K: Strike price (K > 0).
        r: Continuously compounded risk-free rate.
        sigma: Volatility (sigma > 0).
        T: Time to maturity in years (T >= 0).
        option: Option type, either "call" or "put".
        q: Continuous dividend yield.
        h_rel: Relative step size for finite differences (default 1e-4).

    Returns:
        Greeks: Frozen dataclass with delta, gamma, vega, theta, rho.
    """
    if S <= 0.0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0.0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if sigma <= 0.0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")
    if T < 0.0:
        raise ValueError(f"Time to maturity T must be non-negative, got {T}")
    if h_rel <= 0.0:
        raise ValueError(f"Relative bump size h_rel must be positive, got {h_rel}")
    if option not in ("call", "put"):
        raise ValueError(f"Option type must be 'call' or 'put', got {option!r}")

    def _eval(
        s: float,
        k: float,
        rate: float,
        vol: float,
        mat: float,
        opt: Literal["call", "put"],
        div: float,
    ) -> float:
        try:
            val = price_fn(s, k, rate, vol, mat, opt, div)
        except TypeError:
            try:
                val = price_fn(s, k, rate, vol, mat, opt)
            except TypeError:
                try:
                    val = price_fn(s, k, rate, vol, mat)
                except TypeError:
                    val = price_fn(s, vol, rate)
        if hasattr(val, "price"):
            return float(val.price)
        return float(val)

    if T == 0.0:
        if option == "call":
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    # Base price
    v0 = _eval(S, K, r, sigma, T, option, q)

    # Spot bump (Delta, Gamma)
    h_s = h_rel * S
    v_s_up = _eval(S + h_s, K, r, sigma, T, option, q)
    v_s_dn = _eval(S - h_s, K, r, sigma, T, option, q)
    delta = (v_s_up - v_s_dn) / (2.0 * h_s)
    gamma = (v_s_up - 2.0 * v0 + v_s_dn) / (h_s**2)

    # Volatility bump (Vega)
    h_vol = max(h_rel * sigma, 1e-5)
    vol_up = sigma + h_vol
    vol_dn = max(sigma - h_vol, 1e-7)
    v_vol_up = _eval(S, K, r, vol_up, T, option, q)
    v_vol_dn = _eval(S, K, r, vol_dn, T, option, q)
    vega = (v_vol_up - v_vol_dn) / (vol_up - vol_dn)

    # Rate bump (Rho)
    h_r = max(h_rel * abs(r), 1e-5)
    v_r_up = _eval(S, K, r + h_r, sigma, T, option, q)
    v_r_dn = _eval(S, K, r - h_r, sigma, T, option, q)
    rho = (v_r_up - v_r_dn) / (2.0 * h_r)

    # Time bump (Theta: calendar time advance = maturity reduction)
    h_t = min(h_rel, T / 2.0)
    v_t_dn = _eval(S, K, r, sigma, T - h_t, option, q)
    theta = (v_t_dn - v0) / h_t

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
