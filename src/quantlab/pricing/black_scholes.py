"""Black-Scholes analytical pricing for European options."""

from typing import Literal, overload

import numpy as np
from scipy.stats import norm


@overload
def bs_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: Literal["call", "put"],
    q: float = ...,
) -> float: ...


@overload
def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    T: float,
    option: Literal["call", "put"],
    q: float = ...,
) -> float | np.ndarray: ...


def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    T: float,
    option: Literal["call", "put"],
    q: float = 0.0,
) -> float | np.ndarray:
    r"""Compute European call or put price using the Black-Scholes-Merton model.

    Formula:
        d1 = [ln(S / K) + (r - q + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        Call = S * e^(-q * T) * N(d1) - K * e^(-r * T) * N(d2)
        Put  = K * e^(-r * T) * N(-d2) - S * e^(-q * T) * N(-d1)

    Assumptions:
        - Underlying asset follows Geometric Brownian Motion with constant volatility sigma.
        - Risk-free interest rate r and continuous dividend yield q are constant.
        - Frictionless market with no transaction costs, continuous trading, and no arbitrage.
        - European exercise style (exercise only at expiry T).

    Parameters:
        S: Spot price of the underlying asset (S > 0).
        K: Strike price of the option (K > 0).
        r: Continuously compounded risk-free rate.
        sigma: Annualized volatility (sigma > 0).
        T: Time to expiration in years (T >= 0).
        option: Option type, either "call" or "put".
        q: Continuously compounded dividend yield (defaults to 0.0).

    Returns:
        Option price as a float (if inputs are scalar) or np.ndarray.

    Raises:
        ValueError: If T < 0, S <= 0, K <= 0, sigma <= 0, or option is not "call"/"put".
    """
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")

    s_arr = np.asarray(S)
    k_arr = np.asarray(K)
    sigma_arr = np.asarray(sigma)

    if np.any(s_arr <= 0):
        raise ValueError("S must be strictly positive")
    if np.any(k_arr <= 0):
        raise ValueError("K must be strictly positive")
    if np.any(sigma_arr <= 0):
        raise ValueError("sigma must be strictly positive")

    # At expiry (T = 0), the option value equals its intrinsic payoff
    if T == 0.0:
        if option == "call":
            payoff = np.maximum(s_arr - k_arr, 0.0)
        else:
            payoff = np.maximum(k_arr - s_arr, 0.0)
        return float(payoff) if np.ndim(payoff) == 0 else payoff

    sqrt_t = np.sqrt(T)
    d1 = (np.log(s_arr / k_arr) + (r - q + 0.5 * sigma_arr**2) * T) / (sigma_arr * sqrt_t)
    d2 = d1 - sigma_arr * sqrt_t

    if option == "call":
        price = s_arr * np.exp(-q * T) * norm.cdf(d1) - k_arr * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = k_arr * np.exp(-r * T) * norm.cdf(-d2) - s_arr * np.exp(-q * T) * norm.cdf(-d1)

    return float(price) if np.ndim(price) == 0 else price
