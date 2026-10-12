"""Cox-Ross-Rubinstein (CRR) binomial tree option pricing for European and American options."""

import math
from typing import Literal

import numpy as np


def binomial_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: Literal["call", "put"],
    q: float = 0.0,
    steps: int = 200,
    american: bool = False,
) -> float:
    r"""Compute European or American option price using the Cox-Ross-Rubinstein binomial tree.

    Formulas:
        dt = T / steps
        u  = exp(sigma * sqrt(dt))
        d  = 1 / u = exp(-sigma * sqrt(dt))
        p  = (exp((r - q) * dt) - d) / (u - d)

        Backward induction:
            V_node = exp(-r * dt) * [p * V_up + (1 - p) * V_down]
            If american:
                V_node = max(V_node, intrinsic(S_node))

    Assumptions:
        - Underlying follows a recombining multiplicative binomial lattice approximating GBM.
        - Risk-neutral up probability p must lie in [0, 1]; raises ValueError otherwise.
        - European style checks payoff at maturity T only; American style allows early exercise
          at any lattice node before or at maturity.
        - Memory complexity is O(steps) via a 1-D NumPy array backward induction.

    Parameters:
        S: Spot price of the underlying asset (S > 0).
        K: Strike price of the option (K > 0).
        r: Continuously compounded risk-free interest rate.
        sigma: Annualized volatility (sigma > 0).
        T: Time to maturity in years (T >= 0).
        option: Option type, either "call" or "put".
        q: Continuous dividend yield (default 0.0).
        steps: Number of discrete time steps in the tree (steps >= 1, default 200).
        american: Whether to allow early exercise (American style, default False).

    Returns:
        float: Estimated option price at the root node (t = 0).

    Raises:
        ValueError: If parameters violate domain constraints or
            risk-neutral probability p is not in [0, 1].
    """
    if S <= 0.0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0.0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if sigma <= 0.0:
        raise ValueError(f"Volatility sigma must be positive, got {sigma}")
    if T < 0.0:
        raise ValueError(f"Time to maturity T must be non-negative, got {T}")
    if steps < 1:
        raise ValueError(f"Number of steps must be at least 1, got {steps}")
    if option not in ("call", "put"):
        raise ValueError(f"Option type must be 'call' or 'put', got {option!r}")

    # Immediate payoff at expiry
    if T == 0.0:
        if option == "call":
            return float(max(S - K, 0.0))
        return float(max(K - S, 0.0))

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u

    # Risk-neutral probability of an up move
    drift = math.exp((r - q) * dt)
    denom = u - d
    p = (drift - d) / denom

    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Risk-neutral probability p = {p:.6f} is outside [0, 1]. "
            f"Lattice condition violated (dt = {dt:.6f}). Increase steps to refine time grid."
        )

    disc = math.exp(-r * dt)

    # Terminal asset prices at step N: S * u^(2*j - N) for j in [0, ..., N]
    j = np.arange(steps + 1, dtype=np.float64)
    s_nodes = S * (u ** (2.0 * j - steps))

    # Terminal option payoffs
    if option == "call":
        values = np.maximum(s_nodes - K, 0.0)
    else:
        values = np.maximum(K - s_nodes, 0.0)

    # Backward induction using 1D array rolled in-place
    p_comp = 1.0 - p
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + p_comp * values[:-1])
        if american:
            j_i = np.arange(i + 1, dtype=np.float64)
            s_nodes_i = S * (u ** (2.0 * j_i - i))
            if option == "call":
                intrinsic = np.maximum(s_nodes_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - s_nodes_i, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])
