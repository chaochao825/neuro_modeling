"""Plot data-bound shared-dynamics comparisons from aggregated CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "common": "#4C78A8",
    "shared": "#E45756",
    "separate": "#54A24B",
    "random": "#B279A2",
    "orthogonal": "#F58518",
    "shuffled": "#72B7B2",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(results_root: Path, output_base: Path) -> None:
    summary = pd.read_csv(results_root / "model_summary.csv")
    comparisons = pd.read_csv(results_root / "comparisons.csv")
    if summary.empty or comparisons.empty:
        raise ValueError("summary/comparison tables must be non-empty")
    dims = sorted(summary["latent_dim"].unique())
    target_dim = 4 if 4 in dims else dims[0]
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    for model in ("common", "shared", "separate"):
        values = summary.loc[summary["model"] == model].sort_values("latent_dim")
        x = values["latent_dim"].to_numpy()
        y = values["nll_per_scalar_mean"].to_numpy()
        sd = values["nll_per_scalar_std"].fillna(0.0).to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.8, color=COLORS[model], label=model)
        ax.fill_between(x, y - sd, y + sd, color=COLORS[model], alpha=0.14, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, labels=[str(int(value)) for value in dims])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Held-out NLL / scalar (lower is better)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_title("A  Held-out predictive likelihood", loc="left", fontweight="bold")

    ax = axes[0, 1]
    target = summary.loc[summary["latent_dim"] == target_dim].copy()
    for row in target.itertuples(index=False):
        ax.errorbar(
            row.parameter_count_median,
            row.nll_per_scalar_mean,
            yerr=0.0 if np.isnan(row.nll_per_scalar_std) else row.nll_per_scalar_std,
            fmt="o",
            markersize=6,
            capsize=2,
            color=COLORS.get(row.model, "#333333"),
            label=row.model,
        )
    ax.set_xscale("log")
    ax.set_xlabel(f"Fitted parameter count (d={int(target_dim)})")
    ax.set_ylabel("Held-out NLL / scalar")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.set_title("B  Performance–parameter comparison", loc="left", fontweight="bold")

    ax = axes[1, 0]
    shared = summary.loc[summary["model"] == "shared"].sort_values("latent_dim")
    x = shared["latent_dim"].to_numpy()
    rank = shared["effective_rank_mean"].to_numpy()
    rank_sd = shared["effective_rank_std"].fillna(0.0).to_numpy()
    line_rank = ax.plot(
        x,
        rank,
        marker="o",
        color=COLORS["shared"],
        linewidth=1.8,
        label="effective rank",
    )[0]
    ax.fill_between(x, rank - rank_sd, rank + rank_sd, color=COLORS["shared"], alpha=0.14)
    line_ceiling = ax.plot(
        x,
        x,
        linestyle="--",
        color="#777777",
        linewidth=1,
        label="rank ceiling",
    )[0]
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, labels=[str(int(value)) for value in dims])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Transition effective rank")
    ax2 = ax.twinx()
    line_energy = ax2.plot(
        x,
        shared["top_k_singular_energy_mean"].to_numpy(),
        marker="s",
        color="#4C78A8",
        linewidth=1.5,
        label="top-k energy",
    )[0]
    ax2.set_ylabel("Top-k singular energy")
    ax2.set_ylim(0.0, 1.05)
    ax.legend(
        [line_rank, line_energy, line_ceiling],
        ["effective rank", "top-k energy", "rank ceiling"],
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
        ncol=3,
        fontsize=7,
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_title("C  Learned transition dimensionality", loc="left", fontweight="bold")

    ax = axes[1, 1]
    control_columns = {
        "random": "random_minus_shared_nll",
        "orthogonal": "orthogonal_minus_shared_nll",
        "shuffled": "shuffled_minus_shared_nll",
    }
    for model, column in control_columns.items():
        grouped = comparisons.groupby("latent_dim")[column]
        mean = grouped.mean().reindex(dims)
        sd = grouped.std().fillna(0.0).reindex(dims)
        x = np.asarray(dims, dtype=float)
        y = mean.to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.6, color=COLORS[model], label=model)
        ax.fill_between(x, y - sd.to_numpy(), y + sd.to_numpy(), color=COLORS[model], alpha=0.12)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, labels=[str(int(value)) for value in dims])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Control NLL − aligned shared NLL")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_title("D  Basis-alignment controls", loc="left", fontweight="bold")

    fig.text(
        0.5,
        -0.01,
        "Bands/error bars: SD across computational neuron-subset seeds; not biological uncertainty.",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(__file__).resolve().parent / "visual_context_shared_dynamics",
    )
    args = parser.parse_args()
    plot(args.results_root, args.output_base)


if __name__ == "__main__":
    main()
