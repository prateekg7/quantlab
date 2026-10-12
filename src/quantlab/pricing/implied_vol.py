"""Implied volatility calculation using Newton-Raphson with bisection fallback."""

import math
from typing import Literal

from scipy.stats import norm

from quantlab.pricing.black_scholes import bs_price


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option: Literal["call", "put"],
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    r"""Invert the Black-Scholes formula to find implied volatility (F-14).

    Formulas:
        Solve f(sigma) = BS(S, K, r, sigma, T, option, q) - market_price = 0.
        Newton step:
            sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)
            where vega = S * exp(-q * T) * phi(d1) * sqrt(T)

    No-Arbitrage Bounds:
        Call:
            Lower bound: max(S * exp(-q * T) - K * exp(-r * T), 0.0)
            Upper bound: S * exp(-q * T)
        Put:
            Lower bound: max(K * exp(-r * T) - S * exp(-q * T), 0.0)
            Upper bound: K * exp(-r * T)

    Algorithm:
        - Initial guess: sigma_0 = 0.20 (20% annualized volatility).
        - Newton iteration with analytical Black-Scholes vega.
        - Robustness fallback: if vega < 1e-8 or iterate leaves (1e-6, 5.0),
          smoothly switch to bisection search on [1e-6, 5.0].

    Parameters:
        market_price: Observed market price of the option (market_price > 0).
        S: Spot price of underlying asset (S > 0).
        K: Strike price (K > 0).
        r: Continuously compounded risk-free rate.
        T: Time to maturity in years (T > 0).
        option: Option type, either "call" or "put".
        q: Continuous dividend yield (default 0.0).
        tol: Convergence tolerance on price residual (default 1e-8).
        max_iter: Maximum number of Newton iterations (default 100).

    Returns:
        float: Implied annualized volatility sigma.

    Raises:
        ValueError: If parameters violate domain constraints or
            market_price violates no-arbitrage bounds.
    """
    if S <= 0.0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0.0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if T <= 0.0:
        raise ValueError(f"Time to maturity T must be positive for implied volatility, got {T}")
    if tol <= 0.0:
        raise ValueError(f"Tolerance tol must be positive, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if option not in ("call", "put"):
        raise ValueError(f"Option type must be 'call' or 'put', got {option!r}")

    # No-arbitrage theoretical price bounds
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    sqrt_t = math.sqrt(T)

    if option == "call":
        lower_bound = max(S * disc_q - K * disc_r, 0.0)
        upper_bound = S * disc_q
    else:
        lower_bound = max(K * disc_r - S * disc_q, 0.0)
        upper_bound = K * disc_r

    if market_price <= lower_bound or market_price >= upper_bound:
        raise ValueError(
            f"Market price {market_price:.6f} violates no-arbitrage bounds "
            f"[{lower_bound:.6f}, {upper_bound:.6f}] for {option}."
        )

    def _bs_and_vega(vol: float) -> tuple[float, float]:
        price = float(bs_price(S, K, r, vol, T, option, q=q))
        vol_sqrt_t = vol * sqrt_t
        d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / vol_sqrt_t
        vega = float(S * disc_q * norm.pdf(d1) * sqrt_t)
        return price, vega

    def _bisection(low: float = 1e-6, high: float = 5.0) -> float:
        # Expand upper bracket if needed
        p_high = float(bs_price(S, K, r, high, T, option, q=q))
        while p_high < market_price and high < 20.0:
            high *= 2.0
            p_high = float(bs_price(S, K, r, high, T, option, q=q))

        p_tol = min(tol, tol * max(market_price, 1e-8))

        for _ in range(max_iter * 2):
            mid = 0.5 * (low + high)
            p_mid = float(bs_price(S, K, r, mid, T, option, q=q))
            diff = p_mid - market_price
            if abs(diff) < p_tol or (high - low) < tol:
                return mid
            if diff > 0.0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    # Initial guess
    sigma = 0.20
    use_bisection = False

    for _ in range(max_iter):
        price, vega = _bs_and_vega(sigma)
        diff = price - market_price

        if vega < 1e-8:
            use_bisection = True
            break

        step = diff / vega

        if abs(diff) < tol and abs(step) < tol:
            return sigma

        sigma_next = sigma - step

        if abs(step) < tol and abs(diff) < 1e-6:
            return sigma_next

        if sigma_next <= 1e-6 or sigma_next >= 5.0:
            use_bisection = True
            break

        sigma = sigma_next
    else:
        use_bisection = True

    if use_bisection:
        return _bisection(low=1e-6, high=5.0)

    return sigma
