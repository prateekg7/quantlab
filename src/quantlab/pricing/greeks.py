"""Analytical and finite-difference option Greeks."""

import math
from dataclasses import dataclass
from typing import Literal

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
