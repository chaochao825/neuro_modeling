#!/usr/bin/env python3
"""Publication-layout amendment for the verified Exp39 result."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#777777"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _summary_dot(
    ax: plt.Axes,
    *,
    index: int,
    values: np.ndarray,
    color: str,
) -> None:
    offsets = np.linspace(-0.085, 0.085, len(values))
    ax.scatter(
        index + offsets,
        values,
        s=12,
        color=color,
        alpha=0.52,
        edgecolors="none",
        zorder=2,
    )
    mean = float(np.mean(values))
    half = (
        0.0
        if len(values) < 2
        else 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    )
    ax.errorbar(
        index,
        mean,
        yerr=half,
        fmt="o",
        color="black",
        capsize=2.5,
        markersize=3.5,
        linewidth=1.0,
        zorder=3,
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=10,
    )


def render(result_dir: Path, output_stem: Path) -> tuple[Path, Path, Path]:
    seed = pd.read_csv(result_dir / "seed_metrics.csv")
    block = pd.read_csv(result_dir / "block_metrics.csv", dtype={"cell": str})
    clamp = pd.read_csv(result_dir / "comparisons_and_clamps.csv")
    pivot = seed.pivot(index="seed", columns="method", values="heldout_nll")
    required = {
        "selected_fixed",
        "seen_mode_imm",
        "factorized",
        "oracle_factorial_imm",
        "oracle_dynamic",
    }
    if not required <= set(pivot):
        raise RuntimeError("formal Exp39 methods are incomplete")
    _style()
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.35), constrained_layout=True
    )

    ax = axes[0, 0]
    fixed_gain = (
        pivot["selected_fixed"] - pivot["factorized"]
    ).to_numpy(float)
    imm_gain = (
        pivot["seen_mode_imm"] - pivot["factorized"]
    ).to_numpy(float)
    _summary_dot(ax, index=0, values=fixed_gain, color=BLUE)
    _summary_dot(ax, index=1, values=imm_gain, color=BLUE)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xticks((0, 1), ("Fixed", "Seen IMM"))
    ax.set_ylabel("Paired NLL gain (nats)")
    _panel_label(ax, "a")

    ax = axes[0, 1]
    methods = (
        ("selected_fixed", "Fixed", GREY),
        ("seen_mode_imm", "Seen IMM", ORANGE),
        ("factorized", "Factorized", BLUE),
        ("oracle_factorial_imm", "8-mode", GREEN),
        ("oracle_dynamic", "Dynamic", "#00695C"),
    )
    for index, (method, _, color) in enumerate(methods):
        _summary_dot(
            ax,
            index=index,
            values=pivot[method].to_numpy(float),
            color=color,
        )
    ax.set_xticks(
        range(len(methods)),
        [value[1] for value in methods],
        rotation=24,
        ha="right",
    )
    ax.set_ylabel("Held-out NLL (nats)")
    _panel_label(ax, "b")

    ax = axes[1, 0]
    clamp = clamp.loc[clamp["factor"].isin(("h", "q", "r"))]
    factor_style = (
        ("h", "Hazard", VERMILLION),
        ("q", "Process", PURPLE),
        ("r", "Observation", SKY),
    )
    for index, (factor, _, color) in enumerate(factor_style):
        values = clamp.loc[
            clamp["factor"] == factor, "selectivity"
        ].to_numpy(float)
        _summary_dot(ax, index=index, values=values, color=color)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(3), [value[1] for value in factor_style])
    ax.set_ylabel("Clamp selectivity (nats)")
    _panel_label(ax, "c")

    ax = axes[1, 1]
    factorized = block.loc[block["method"] == "factorized"]
    for index, (factor, _, color) in enumerate(factor_style):
        ratios: list[float] = []
        for _, frame in factorized.groupby("seed"):
            estimate_high = np.log(
                frame.loc[
                    frame[f"{factor}_high"] == 1,
                    f"mean_{factor}_estimate",
                ]
            ).mean()
            estimate_low = np.log(
                frame.loc[
                    frame[f"{factor}_high"] == 0,
                    f"mean_{factor}_estimate",
                ]
            ).mean()
            true_high = np.log(
                frame.loc[
                    frame[f"{factor}_high"] == 1, f"true_{factor}"
                ]
            ).mean()
            true_low = np.log(
                frame.loc[
                    frame[f"{factor}_high"] == 0, f"true_{factor}"
                ]
            ).mean()
            ratios.append(
                float(
                    (estimate_high - estimate_low)
                    / (true_high - true_low)
                )
            )
        _summary_dot(
            ax, index=index, values=np.asarray(ratios), color=color
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.axhline(1.0, color=GREEN, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(3), [value[1] for value in factor_style])
    ax.set_ylabel("Recovered / true log-range")
    _panel_label(ax, "d")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = tuple(
        output_stem.with_suffix(f".{suffix}")
        for suffix in ("pdf", "svg", "png")
    )
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return outputs  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem is not None
        else result_dir / "figure_exp39_publication_v2"
    )
    for path in render(result_dir, output_stem):
        print(path)


if __name__ == "__main__":
    main()
