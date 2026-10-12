"""Generate Binomial Convergence plot against Black-Scholes (T-26).

Plots European option binomial tree prices and pricing errors vs number of steps,
illustrating the characteristic sawtooth oscillations and O(1/n) convergence.
Saves figure to docs/binomial_convergence.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab.pricing.binomial import binomial_price
from quantlab.pricing.black_scholes import bs_price

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_PATH = DOCS_DIR / "binomial_convergence.png"


def generate_binomial_convergence_plot() -> None:
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    bs_exact = float(bs_price(S, K, r, sigma, T, "call"))

    step_range = np.arange(10, 301, dtype=int)
    prices = np.zeros(len(step_range))
    errors = np.zeros(len(step_range))

    for idx, n in enumerate(step_range):
        p = binomial_price(S, K, r, sigma, T, "call", steps=int(n))
        prices[idx] = p
        errors[idx] = abs(p - bs_exact)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)

    # Subplot 1: Price Convergence
    ax1.plot(
        step_range,
        prices,
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.85,
        label=r"Binomial Tree Price $V_{\mathrm{tree}}(n)$",
    )
    ax1.axhline(
        bs_exact,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Black-Scholes Analytical ({bs_exact:.4f})",
    )
    ax1.set_ylabel("Option Price ($)", fontsize=11)
    ax1.set_title(
        r"CRR Binomial Tree Convergence vs. Black-Scholes Analytical Price"
        "\n"
        r"European Call ($S=100, K=100, r=5\%, \sigma=20\%, T=1$)",
        fontsize=12,
        pad=10,
    )
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Absolute Error & Sawtooth Oscillation (Log-scale)
    ax2.semilogy(
        step_range,
        errors,
        color="#2ca02c",
        linewidth=1.0,
        alpha=0.75,
        label=r"Absolute Error $|V_{\mathrm{tree}}(n) - V_{\mathrm{BS}}|$",
    )

    # Theoretical O(1/n) reference envelope
    c_envelope = np.mean(errors[:20] * step_range[:20])
    ax2.semilogy(
        step_range,
        c_envelope / step_range,
        ":",
        color="gray",
        linewidth=1.5,
        alpha=0.9,
        label=r"Asymptotic Envelope $O(1/n)$",
    )

    ax2.set_xlabel(r"Number of Tree Steps ($n$)", fontsize=11)
    ax2.set_ylabel("Absolute Error (log scale)", fontsize=11)
    ax2.set_title("Pricing Error: Sawtooth Oscillation and Decay", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved binomial convergence plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_binomial_convergence_plot()
