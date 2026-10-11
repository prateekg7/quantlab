"""Monte Carlo option pricing with variance reduction."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MCResult:
    """Result of a Monte Carlo option pricing simulation.

    Attributes:
        price: Estimated expected discounted option payoff.
        std_error: Standard error of the sample mean estimator.
        ci_low: Lower bound of the 95% confidence interval (price - 1.96 * std_error).
        ci_high: Upper bound of the 95% confidence interval (price + 1.96 * std_error).
        n_paths: Total number of simulated price paths used in the estimate.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int


def mc_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: Literal["call", "put"],
    q: float = 0.0,
    n_paths: int = 100_000,
    seed: int | np.random.Generator | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MCResult:
    r"""Price a European option via Monte Carlo simulation under risk-neutral GBM.

    Formulas:
        Under the risk-neutral measure Q, the asset price follows Geometric Brownian Motion:
            S_T = S * exp((r - q - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z),  Z ~ N(0, 1)

        Discounted payoff:
            X_i = exp(-r * T) * max(S_T^(i) - K, 0)   (call)
            X_i = exp(-r * T) * max(K - S_T^(i), 0)   (put)

        Estimator:
            price = mean(X)
            std_error = std(X, ddof=1) / sqrt(n)
            95% CI = [price - 1.96 * std_error, price + 1.96 * std_error]

    Variance Reduction:
        - Antithetic Variates (F-9):
          Simulates pairs (Z_i, -Z_i). The pair average P_i = 0.5 * (f(Z_i) + f(-Z_i)) has
          strictly lower variance due to negative covariance Cov(f(Z), f(-Z)) < 0.
          Standard error is computed over the n_paths / 2 pair averages.
        - Control Variate (F-10):
          Uses discounted asset price Y = exp(-r*T) * S_T with known expectation
          E[Y] = S * exp(-q*T). Optimal coefficient beta = Cov(X, Y) / Var(Y).
          Estimator: X_cv = X - beta * (Y - E[Y]).

    Parameters:
        S: Spot price of the underlying asset (S > 0).
        K: Strike price of the option (K > 0).
        r: Continuously compounded risk-free interest rate.
        sigma: Annualized volatility of the underlying asset (sigma >= 0).
        T: Time to expiry in years (T >= 0).
        option: 'call' or 'put'.
        q: Continuous dividend yield (default 0.0).
        n_paths: Number of simulated price paths (must be >= 2).
        seed: Random seed or NumPy Generator for reproducible sampling.
        antithetic: Whether to use antithetic variates (F-9).
        control_variate: Whether to use the underlying asset as a control variate (F-10).

    Returns:
        MCResult containing price estimate, standard error, 95% CI, and path count.
    """
    if S <= 0.0:
        raise ValueError(f"Spot price S must be positive, got {S}")
    if K <= 0.0:
        raise ValueError(f"Strike price K must be positive, got {K}")
    if sigma < 0.0:
        raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")
    if T < 0.0:
        raise ValueError(f"Time to maturity T must be non-negative, got {T}")
    if option not in ("call", "put"):
        raise ValueError(f"Option type must be 'call' or 'put', got {option!r}")
    if n_paths <= 1:
        raise ValueError(f"n_paths must be at least 2, got {n_paths}")

    # Boundary: at expiry T = 0
    if T == 0.0:
        intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return MCResult(
            price=float(intrinsic),
            std_error=0.0,
            ci_low=float(intrinsic),
            ci_high=float(intrinsic),
            n_paths=n_paths,
        )

    # Deterministic limit: sigma = 0
    if sigma == 0.0:
        s_t = S * np.exp((r - q) * T)
        discount = np.exp(-r * T)
        payoff = max(s_t - K, 0.0) if option == "call" else max(K - s_t, 0.0)
        det_price = discount * payoff
        return MCResult(
            price=float(det_price),
            std_error=0.0,
            ci_low=float(det_price),
            ci_high=float(det_price),
            n_paths=n_paths,
        )

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    drift = (r - q - 0.5 * sigma**2) * T
    vol_sqrt_t = sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    expected_y = S * np.exp(-q * T)

    if antithetic:
        # Round up to ensure an integer number of pairs
        n_pairs = (n_paths + 1) // 2
        actual_n_paths = 2 * n_pairs

        z = rng.standard_normal(n_pairs)
        s_t_pos = S * np.exp(drift + vol_sqrt_t * z)
        s_t_neg = S * np.exp(drift - vol_sqrt_t * z)

        if option == "call":
            payoff_pos = np.maximum(s_t_pos - K, 0.0)
            payoff_neg = np.maximum(s_t_neg - K, 0.0)
        else:
            payoff_pos = np.maximum(K - s_t_pos, 0.0)
            payoff_neg = np.maximum(K - s_t_neg, 0.0)

        x_pos = discount * payoff_pos
        x_neg = discount * payoff_neg
        p_x = 0.5 * (x_pos + x_neg)

        if control_variate:
            y_pos = discount * s_t_pos
            y_neg = discount * s_t_neg
            p_y = 0.5 * (y_pos + y_neg)

            cov_matrix = np.cov(p_x, p_y, ddof=1)
            var_y = float(cov_matrix[1, 1])
            beta = float(cov_matrix[0, 1] / var_y) if var_y > 0 else 0.0
            estimator = p_x - beta * (p_y - expected_y)
        else:
            estimator = p_x

        price = float(np.mean(estimator))
        std_error = float(np.std(estimator, ddof=1) / np.sqrt(n_pairs))
        total_paths = actual_n_paths

    else:
        z = rng.standard_normal(n_paths)
        s_t = S * np.exp(drift + vol_sqrt_t * z)

        if option == "call":
            payoff = np.maximum(s_t - K, 0.0)
        else:
            payoff = np.maximum(K - s_t, 0.0)

        x = discount * payoff

        if control_variate:
            y = discount * s_t
            cov_matrix = np.cov(x, y, ddof=1)
            var_y = float(cov_matrix[1, 1])
            beta = float(cov_matrix[0, 1] / var_y) if var_y > 0 else 0.0
            estimator = x - beta * (y - expected_y)
        else:
            estimator = x

        price = float(np.mean(estimator))
        std_error = float(np.std(estimator, ddof=1) / np.sqrt(n_paths))
        total_paths = n_paths

    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error

    return MCResult(
        price=price,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paths=total_paths,
    )
