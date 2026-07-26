#!/usr/bin/env python3
"""Render the Exp39 result directly from seed- and block-level CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "fixed": "#7A7A7A",
    "imm": "#E69F00",
    "factorized": "#0072B2",
    "oracle": "#009E73",
    "h": "#D55E00",
    "q": "#CC79A7",
    "r": "#56B4E9",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, half


def render(result_dir: Path, output_stem: Path) -> tuple[Path, Path, Path]:
    """Create PDF, SVG, and 300-dpi PNG with no placeholder values."""

    seed_metrics = pd.read_csv(result_dir / "seed_metrics.csv")
    block_metrics = pd.read_csv(
        result_dir / "block_metrics.csv", dtype={"cell": str}
    )
    clamp = pd.read_csv(result_dir / "comparisons_and_clamps.csv")
    required_methods = {
        "selected_fixed",
        "seen_mode_imm",
        "factorized",
        "oracle_factorial_imm",
        "oracle_dynamic",
    }
    if not required_methods <= set(seed_metrics["method"]):
        raise RuntimeError("Exp39 seed metrics are incomplete")
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))

    pivot = seed_metrics.pivot(
        index="seed", columns="method", values="heldout_nll"
    )
    gains = {
        "Fixed": pivot["selected_fixed"] - pivot["factorized"],
        "Seen IMM": pivot["seen_mode_imm"] - pivot["factorized"],
    }
    ax = axes[0, 0]
    for index, (label, values) in enumerate(gains.items()):
        x = np.full(len(values), index, dtype=float)
        offsets = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(
            x + offsets,
            values,
            s=13,
            color=COLORS["factorized"],
            alpha=0.58,
            edgecolors="none",
        )
        mean, half = _mean_ci(values.to_numpy(float))
        ax.errorbar(
            index,
            mean,
            yerr=half,
            fmt="o",
            color="black",
            capsize=3,
            markersize=4,
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(gains)), gains.keys())
    ax.set_ylabel("Paired predictive-NLL gain (nats)")
    ax.text(-0.16, 1.04, "a", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    methods = [
        ("selected_fixed", "Fixed", COLORS["fixed"]),
        ("seen_mode_imm", "Seen IMM", COLORS["imm"]),
        ("factorized", "Factorized", COLORS["factorized"]),
        ("oracle_factorial_imm", "8-mode oracle", COLORS["oracle"]),
        ("oracle_dynamic", "Dynamic oracle", "#004D40"),
    ]
    for index, (method, _, color) in enumerate(methods):
        values = pivot[method].to_numpy(float)
        offsets = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=12,
            color=color,
            alpha=0.50,
            edgecolors="none",
        )
        mean, half = _mean_ci(values)
        ax.errorbar(
            index,
            mean,
            yerr=half,
            fmt="o",
            color="black",
            capsize=3,
            markersize=3.8,
            linewidth=1.1,
        )
    ax.set_xticks(
        range(len(methods)),
        [item[1] for item in methods],
        rotation=28,
        ha="right",
    )
    ax.set_ylabel("Held-out predictive NLL (nats)")
    ax.text(-0.16, 1.04, "b", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    clamp = clamp.loc[clamp["factor"].isin(("h", "q", "r"))].copy()
    for index, factor in enumerate(("h", "q", "r")):
        values = clamp.loc[clamp["factor"] == factor, "selectivity"].to_numpy(float)
        offsets = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=13,
            color=COLORS[factor],
            alpha=0.58,
            edgecolors="none",
        )
        mean, half = _mean_ci(values)
        ax.errorbar(
            index,
            mean,
            yerr=half,
            fmt="o",
            color="black",
            capsize=3,
            markersize=4,
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks((0, 1, 2), ("Hazard", "Process", "Observation"))
    ax.set_ylabel("Selective clamp penalty (nats)")
    ax.text(-0.16, 1.04, "c", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    factorized = block_metrics.loc[block_metrics["method"] == "factorized"]
    separations: dict[str, list[float]] = {"h": [], "q": [], "r": []}
    for seed, frame in factorized.groupby("seed"):
        del seed
        for factor in separations:
            high = np.log(
                frame.loc[
                    frame[f"{factor}_high"] == 1,
                    f"mean_{factor}_estimate",
                ]
            ).mean()
            low = np.log(
                frame.loc[
                    frame[f"{factor}_high"] == 0,
                    f"mean_{factor}_estimate",
                ]
            ).mean()
            true_high = np.log(
                frame.loc[frame[f"{factor}_high"] == 1, f"true_{factor}"]
            ).mean()
            true_low = np.log(
                frame.loc[frame[f"{factor}_high"] == 0, f"true_{factor}"]
            ).mean()
            separations[factor].append(float((high - low) / (true_high - true_low)))
    for index, factor in enumerate(("h", "q", "r")):
        values = np.asarray(separations[factor])
        offsets = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=13,
            color=COLORS[factor],
            alpha=0.58,
            edgecolors="none",
        )
        mean, half = _mean_ci(values)
        ax.errorbar(
            index,
            mean,
            yerr=half,
            fmt="o",
            color="black",
            capsize=3,
            markersize=4,
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color=COLORS["oracle"], linewidth=0.8, linestyle=":")
    ax.set_xticks((0, 1, 2), ("Hazard", "Process", "Observation"))
    ax.set_ylabel("Recovered log-range / true log-range")
    ax.text(-0.16, 1.04, "d", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout(w_pad=1.7, h_pad=1.8)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png"))
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return paths  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem is not None
        else result_dir / "figure_exp39"
    )
    for path in render(result_dir, output_stem):
        print(path)


if __name__ == "__main__":
    main()
