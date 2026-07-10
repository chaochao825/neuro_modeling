"""Create a cross-dataset summary figure for the Minimal_computation reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_BASE = ROOT / "figures" / "paper_comparison_summary"

RUNS = [
    ("C. elegans", RESULTS / "c_elegans_neuron13_max32.json", "#4c78a8", "o"),
    ("Hippocampus", RESULTS / "hippocampus_neuron13_max30.json", "#f58518", "s"),
    ("Visual spontaneous", RESULTS / "visual_spontaneous_neuron13_max30.json", "#54a24b", "^"),
    ("Visual responding", RESULTS / "visual_responding_neuron13_max30.json", "#b279a2", "D"),
]


def load_runs() -> list[dict]:
    loaded = []
    for label, path, color, marker in RUNS:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["label"] = label
        data["color"] = color
        data["marker"] = marker
        loaded.append(data)
    return loaded


def main() -> None:
    runs = load_runs()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax_entropy, ax_residual, ax_reduction, ax_ratio = axes.ravel()

    labels = []
    reductions = []
    residual_ratios = []
    bar_colors = []
    completion_labels = []

    for data in runs:
        nums = np.asarray(data["nums_in"], dtype=float)
        entropy = np.asarray(data["entropies"], dtype=float)
        residual = np.asarray(data["residual_errors"], dtype=float)
        entropy0 = float(data["independent_entropy"])
        label = data["label"]
        color = data["color"]

        ax_entropy.plot(
            nums,
            entropy / entropy0,
            marker=data["marker"],
            linewidth=2,
            color=color,
            label=label,
        )
        ax_residual.plot(
            nums,
            residual,
            marker=data["marker"],
            linewidth=2,
            color=color,
            label=label,
        )

        labels.append(label)
        reductions.append(100.0 * (entropy0 - float(entropy[-1])) / entropy0)
        residual_ratios.append(float(residual[-1]) / 2.0)
        bar_colors.append(color)
        if data["complete_num_inputs"] is None:
            completion_labels.append("not complete")
        else:
            completion_labels.append(f"{data['complete_num_inputs']} inputs")

    ax_entropy.set_xlabel("Selected inputs")
    ax_entropy.set_ylabel("Entropy / independent entropy")
    ax_entropy.set_ylim(0.0, 1.05)
    ax_entropy.grid(color="#dddddd", linewidth=0.6)
    ax_entropy.legend(frameon=False, fontsize=8)
    ax_entropy.text(0.01, 0.98, "A", transform=ax_entropy.transAxes, fontweight="bold", va="top")

    ax_residual.axhline(2.0, color="#333333", linestyle="--", linewidth=1)
    ax_residual.set_xlabel("Selected inputs")
    ax_residual.set_ylabel("Max normalized corr. error")
    ax_residual.set_yscale("log")
    ax_residual.grid(color="#dddddd", linewidth=0.6)
    ax_residual.text(0.01, 0.98, "B", transform=ax_residual.transAxes, fontweight="bold", va="top")

    x = np.arange(len(labels))
    ax_reduction.bar(x, reductions, color=bar_colors, width=0.68)
    ax_reduction.set_xticks(x)
    ax_reduction.set_xticklabels(labels, rotation=25, ha="right")
    ax_reduction.set_ylabel("Entropy reduction at max sweep (%)")
    ax_reduction.set_ylim(0, max(reductions) * 1.18)
    ax_reduction.grid(axis="y", color="#dddddd", linewidth=0.6)
    for i, value in enumerate(reductions):
        ax_reduction.text(i, value + 1.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_reduction.text(0.01, 0.98, "C", transform=ax_reduction.transAxes, fontweight="bold", va="top")

    ax_ratio.bar(x, residual_ratios, color=bar_colors, width=0.68)
    ax_ratio.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(labels, rotation=25, ha="right")
    ax_ratio.set_ylabel("Final residual / complete threshold")
    ax_ratio.set_yscale("log")
    ax_ratio.grid(axis="y", color="#dddddd", linewidth=0.6)
    for i, (value, text) in enumerate(zip(residual_ratios, completion_labels)):
        ax_ratio.text(i, value * 1.12, text, ha="center", va="bottom", fontsize=8)
    ax_ratio.text(0.01, 0.98, "D", transform=ax_ratio.transAxes, fontweight="bold", va="top")

    fig.text(
        0.5,
        -0.01,
        "Historical residual-approximation baselines; not block-Schur/MATLAB-parity results.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(OUT_BASE.with_suffix(".png"))
    print(OUT_BASE.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
