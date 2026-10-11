"""Generate Monte Carlo convergence log-log plot (T-18).

Plots standard error vs. path count N on a log-log scale for:
- Plain Monte Carlo
- Antithetic Variates (F-9)
- Control Variate (F-10)
- Antithetic + Control Variate Combined

Verifies the Central Limit Theorem theoretical convergence rate of O(N^{-1/2})
(slope = -0.50 on log-log scale).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab.pricing.monte_carlo import mc_price

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_PATH = DOCS_DIR / "mc_convergence.png"


def generate_convergence_plot() -> None:
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    seed = 42

    path_counts = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]

    se_plain = []
    se_anti = []
    se_cv = []
    se_both = []

    for n in path_counts:
        res_p = mc_price(S, K, r, sigma, T, "call", n_paths=n, seed=seed)
        res_a = mc_price(S, K, r, sigma, T, "call", n_paths=n, seed=seed, antithetic=True)
        res_c = mc_price(S, K, r, sigma, T, "call", n_paths=n, seed=seed, control_variate=True)
        res_b = mc_price(
            S,
            K,
            r,
            sigma,
            T,
            "call",
            n_paths=n,
            seed=seed,
            antithetic=True,
            control_variate=True,
        )

        se_plain.append(res_p.std_error)
        se_anti.append(res_a.std_error)
        se_cv.append(res_c.std_error)
        se_both.append(res_b.std_error)

    # Compute empirical slope via linear regression on log-log data
    log_n = np.log10(path_counts)
    log_se_plain = np.log10(se_plain)
    slope_plain, intercept_plain = np.polyfit(log_n, log_se_plain, 1)

    # Plot styling
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.loglog(
        path_counts,
        se_plain,
        "o-",
        label=f"Plain MC (slope = {slope_plain:.3f})",
        color="#2b5c8f",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        path_counts,
        se_anti,
        "s--",
        label="Antithetic Variates (F-9)",
        color="#d95f02",
        linewidth=1.8,
        markersize=5,
    )
    ax.loglog(
        path_counts,
        se_cv,
        "^--",
        label="Control Variate (F-10)",
        color="#7570b3",
        linewidth=1.8,
        markersize=5,
    )
    ax.loglog(
        path_counts,
        se_both,
        "d-.",
        label="Antithetic + Control Variate",
        color="#1b9e77",
        linewidth=1.8,
        markersize=5,
    )

    # Reference theoretical slope -0.5 line
    ref_line = 10 ** (intercept_plain - 0.5 * (log_n - log_n[0]))
    ax.loglog(
        path_counts,
        ref_line,
        ":",
        color="gray",
        label=r"Theoretical $O(N^{-1/2})$ ($\mathrm{slope} = -0.500$)",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set_title(
        r"Monte Carlo Convergence: Standard Error vs. Path Count $N$"
        "\n"
        r"European Call ($S=100, K=100, r=5\%, \sigma=20\%, T=1$)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel(r"Number of Simulated Paths $N$", fontsize=11)
    ax.set_ylabel(r"Standard Error ($\mathrm{SE} = s / \sqrt{N}$)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right", frameon=True)

    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Convergence plot saved to: {OUTPUT_PATH}")
    print(f"Empirical slope for Plain MC: {slope_plain:.4f} (theoretical: -0.5000)")


if __name__ == "__main__":
    generate_convergence_plot()
