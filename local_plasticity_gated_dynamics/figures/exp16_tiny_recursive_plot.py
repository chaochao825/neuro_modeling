"""Generate the scoped Exp16 seed-level comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_exp16(
    results_root: Path,
    *,
    prefix: str = "exp16_tiny_recursive_smoke",
) -> dict[str, Path]:
    conditions = pd.read_csv(results_root / f"{prefix}_conditions.csv")
    comparison = pd.read_csv(results_root / f"{prefix}_comparison.csv")
    raw = pd.read_csv(results_root / f"{prefix}_raw.csv.gz")
    aggregates = raw.loc[raw["stage"].eq("aggregate")].copy()
    labels = {
        "micro_trm_bptt": "micro-TRM-like",
        "flat_shared_compute_matched": "single-state matched",
    }
    ordered = [name for name in labels if name in set(conditions["condition"])]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    x = np.arange(len(ordered))
    means = [
        float(
            conditions.loc[conditions["condition"].eq(name), "mean_exact_accuracy"].iloc[0]
        )
        for name in ordered
    ]
    axes[0].bar(x, means, color=["#3569b7", "#9a9a9a"], width=0.62)
    for index, name in enumerate(ordered):
        values = aggregates.loc[
            aggregates["condition"].eq(name), "exact_accuracy"
        ].to_numpy(float)
        axes[0].scatter(
            np.full(len(values), index), values, color="black", s=18, zorder=3
        )
    axes[0].set_xticks(x, [labels[name] for name in ordered], rotation=15, ha="right")
    axes[0].set_ylabel("Held-out exact accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Seed-level performance")

    if comparison.empty:
        axes[1].text(0.5, 0.5, "No paired comparison", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        row = comparison.iloc[0]
        estimate = float(row["estimate"])
        low = float(row["seed_bootstrap_ci_low"])
        high = float(row["seed_bootstrap_ci_high"])
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].errorbar(
            [0],
            [estimate],
            yerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color="#3569b7",
            capsize=4,
        )
        axes[1].set_xlim(-0.7, 0.7)
        axes[1].set_xticks([0], ["micro-TRM − matched"])
        axes[1].set_ylabel("Paired exact-accuracy difference")
        axes[1].set_title(f"{row['conclusion']} ({int(row['n_complete_seeds'])} seeds)")
    figure.suptitle("Exp16 baseline-only recursive reasoning audit")
    figure.tight_layout()
    png = results_root / f"{prefix}.png"
    pdf = results_root / f"{prefix}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {"figure_png": png, "figure_pdf": pdf}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="exp16_tiny_recursive_smoke")
    args = parser.parse_args()
    plot_exp16(Path(args.results_root), prefix=args.prefix)


if __name__ == "__main__":
    main()
