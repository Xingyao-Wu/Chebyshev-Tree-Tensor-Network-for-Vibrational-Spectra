from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("/Users/link/Desktop/A_Python_code/Chebyshev_upload/Chebyshev/master_thesis")
OUTPUT_PDF = OUTPUT_DIR / "mu0_real_contraction_time_panels_ab_effective_hamiltonian_large.pdf"

# Runtime values are from CH3CN_T3NS_12dim_125N_mu0_real.ipynb saved output.
METHODS = [
    {
        "method": "cheb_ttns_CBC",
        "label": "CBC",
        "color": "tab:blue",
        "cheb_vector_s": 50.5934,
        "heff_s": 234.396,
    },
    {
        "method": "cheb_ttns_Density",
        "label": "DM",
        "color": "tab:orange",
        "cheb_vector_s": 728.203,
        "heff_s": 240.13,
    },
    {
        "method": "cheb_ttns_Direct_Truncate",
        "label": "Direct Truncate",
        "color": "tab:green",
        "cheb_vector_s": 417.181,
        "heff_s": 264.521,
    },
    {
        "method": "cheb_ttns_variational_paper",
        "label": "Variational",
        "color": "tab:red",
        "cheb_vector_s": 2770.75,
        "heff_s": 266.922,
    },
]

# First run checkpoint at 51 states plus the resumed 99-state run.
LOBPCG_FIRST_SEGMENT_S = 3115.624032974243
LOBPCG_RESUME_SEGMENT_S = 3641.0814139842987
LOBPCG_TOTAL_S = LOBPCG_FIRST_SEGMENT_S + LOBPCG_RESUME_SEGMENT_S


def main() -> None:
    fig, (ax_runtime, ax_total) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.8),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )

    labels = [row["label"] for row in METHODS]
    colors = [row["color"] for row in METHODS]
    cheb_values = np.array([row["cheb_vector_s"] for row in METHODS], dtype=np.float64)
    heff_values = np.array([row["heff_s"] for row in METHODS], dtype=np.float64)
    total_values = cheb_values + heff_values

    x_runtime = np.arange(len(METHODS), dtype=np.float64)
    bars_runtime = ax_runtime.bar(x_runtime, cheb_values, width=0.6, color=colors)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=row["color"], label=row["label"])
        for row in METHODS
    ]
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, color="0.75", label="Effective Hamiltonian")
    )

    ax_runtime.set_title("Contraction time comparison")
    ax_runtime.set_ylabel("125 Chebyshev vectors time [s]")
    ax_runtime.set_xticks(x_runtime)
    ax_runtime.set_xticklabels(labels, rotation=20, ha="right")
    ax_runtime.grid(True, axis="y", alpha=0.25)
    ax_runtime.legend(handles=legend_handles, fontsize=8, ncol=3, loc="upper left")

    runtime_pad = 0.02 * float(np.nanmax(cheb_values))
    ax_runtime.set_ylim(0.0, float(np.nanmax(cheb_values)) * 1.16)
    for bar, value in zip(bars_runtime, cheb_values):
        ax_runtime.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + runtime_pad,
            f"{value:.1f} s",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )

    x_total = np.arange(len(METHODS) + 1, dtype=np.float64)
    ax_total.bar(
        x_total[:-1],
        cheb_values,
        width=0.6,
        color=colors,
        alpha=0.85,
        label="Chebyshev vectors",
    )
    ax_total.bar(
        x_total[:-1],
        heff_values,
        width=0.6,
        bottom=cheb_values,
        color="0.75",
        edgecolor="0.75",
        label="Effective Hamiltonian",
    )
    ax_total.bar(
        x_total[-1],
        LOBPCG_TOTAL_S,
        width=0.6,
        color="black",
        label="LOBPCG",
    )

    total_labels = labels + ["LOBPCG"]
    total_values_with_lobpcg = np.append(total_values, LOBPCG_TOTAL_S)
    ax_total.set_title("Total contraction time comparison (Chebyshev vectors + Effective Hamiltonian)")
    ax_total.set_ylabel("Total time [s]")
    ax_total.set_xticks(x_total)
    ax_total.set_xticklabels(total_labels, rotation=20, ha="right")
    ax_total.grid(True, axis="y", alpha=0.25)
    ax_total.legend(fontsize=8, ncol=3, loc="upper left")

    total_pad = 0.02 * float(np.nanmax(total_values_with_lobpcg))
    ax_total.set_ylim(0.0, float(np.nanmax(total_values_with_lobpcg)) * 1.16)
    for x, total, heff in zip(x_total[:-1], total_values, heff_values):
        heff_pct = 100.0 * heff / total if total > 0.0 else np.nan
        label = f"total {total:.1f} s"
        if np.isfinite(heff_pct):
            label += f"\nEffective Hamiltonian {heff_pct:.1f}%"
        ax_total.text(
            x,
            total + total_pad,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
    ax_total.text(
        x_total[-1],
        LOBPCG_TOTAL_S + total_pad,
        f"LOBPCG {LOBPCG_TOTAL_S:.1f} s",
        ha="center",
        va="bottom",
        fontsize=8,
        clip_on=False,
    )

    for panel_label, ax in zip(("(a)", "(b)"), (ax_runtime, ax_total)):
        ax.text(
            -0.075,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PDF}")
    print(f"LOBPCG total time [s]: {LOBPCG_TOTAL_S:.6f}")


if __name__ == "__main__":
    main()
