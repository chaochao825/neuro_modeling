"""Plot historical synthetic calibration beside mixed-selector dependency fits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MINIMAL_ROOT = ROOT.parent / "minimal_computation_python"
OUT_BASE = Path(__file__).with_name("integrated_goal_status")

MINIMAL_RUNS = [
    (
        "C. elegans",
        MINIMAL_ROOT / "results" / "c_elegans_matlab_schur_neuron13_max32.json",
        "#4c78a8",
    ),
    ("Hippocampus", MINIMAL_ROOT / "results" / "hippocampus_neuron13_max30.json", "#f58518"),
    ("Visual spontaneous", MINIMAL_ROOT / "results" / "visual_spontaneous_neuron13_max30.json", "#54a24b"),
    ("Visual responding", MINIMAL_ROOT / "results" / "visual_responding_neuron13_max30.json", "#b279a2"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def level_color(level: str) -> str:
    return {
        "strong": "#1b9e77",
        "moderate": "#7570b3",
        "weak": "#d95f02",
        "refuted": "#d73027",
    }.get(level, "#666666")


def panel_framework_scores(ax, matrix: dict) -> None:
    keys = list(matrix)
    labels = [k.split("_")[0] for k in keys]
    scores = [matrix[k]["score"] for k in keys]
    colors = [level_color(matrix[k]["level"]) for k in keys]
    ax.bar(labels, scores, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_ylabel("Evidence score")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.55, 0.8, 1.0])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    for i, key in enumerate(keys):
        ax.text(i, scores[i] + 0.035, matrix[key]["level"], ha="center", va="bottom", fontsize=8)
    ax.text(0.01, 0.98, "A", transform=ax.transAxes, fontweight="bold", va="top")
    ax.set_title("H1-H5 synthetic calibration (seed 7)", loc="left", fontweight="bold")


def panel_minimal_entropy(ax, minimal: list[dict]) -> None:
    labels = [row["label"] for row in minimal]
    reductions = [
        100.0 * (row["independent_entropy"] - row["entropies"][-1]) / row["independent_entropy"]
        for row in minimal
    ]
    colors = [row["color"] for row in minimal]
    x = np.arange(len(labels))
    ax.bar(x, reductions, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Entropy reduction at max sweep (%)")
    ax.set_ylim(0, max(reductions) * 1.18)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    for i, val in enumerate(reductions):
        ax.text(i, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.text(0.01, 0.98, "B", transform=ax.transAxes, fontweight="bold", va="top")
    ax.set_title("Equal-time dependency fits (mixed selectors)", loc="left", fontweight="bold")


def panel_minimal_complete(ax, minimal: list[dict]) -> None:
    labels = [row["label"] for row in minimal]
    ratios = [
        row["residual_errors"][-1] / row.get("run_config", {}).get("corr_error_threshold", 2.0)
        for row in minimal
    ]
    colors = [row["color"] for row in minimal]
    x = np.arange(len(labels))
    ax.bar(x, ratios, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Final residual / complete threshold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    for i, row in enumerate(minimal):
        label = "not complete" if row["complete_num_inputs"] is None else f"{row['complete_num_inputs']} inputs"
        ax.text(i, ratios[i] * 1.12, label, ha="center", va="bottom", fontsize=8)
    ax.text(0.01, 0.98, "C", transform=ax.transAxes, fontweight="bold", va="top")
    ax.set_title("Dependency-fit criterion (mixed selectors)", loc="left", fontweight="bold")


def panel_goal_coverage(ax) -> None:
    rows = [
        "Synthetic\nsimulations",
        "Public-data\ninterface",
        "Equal-time\ndependency fit",
        "Causal / real\nexperiments",
    ]
    cols = ["H1", "H2", "H3", "H4", "H5"]
    # Coverage is a status audit, not an evidence score:
    # 1.0 implemented and run; 0.5 partial interface or submodule; 0.2 design-only; 0.0 absent.
    data = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    colors = {
        1.0: "#2c7fb8",
        0.5: "#c7e9b4",
        0.2: "#edf8b1",
        0.0: "#f7f7f7",
    }
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = float(data[i, j])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=colors[val], edgecolor="#333333", linewidth=0.6))
            txt = {1.0: "run", 0.5: "sub/api", 0.2: "design", 0.0: "-"}[float(data[i, j])]
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#222222")
    legend_x = len(cols) - 1.45
    legend_y = len(rows) - 0.18
    for offset, (label, color) in enumerate([("run", colors[1.0]), ("sub/api", colors[0.5]), ("design", colors[0.2])]):
        ax.add_patch(Rectangle((legend_x + offset * 0.72, legend_y), 0.16, 0.16, facecolor=color, edgecolor="#333333", linewidth=0.4, clip_on=False))
        ax.text(legend_x + offset * 0.72 + 0.2, legend_y + 0.08, label, va="center", fontsize=7, clip_on=False)
    ax.text(0.01, 0.98, "D", transform=ax.transAxes, fontweight="bold", va="top")
    ax.set_title("Original-goal coverage audit", loc="left", fontweight="bold")


def main() -> None:
    matrix = load_json(REPORTS / "decision_matrix.json")
    minimal = []
    for label, path, color in MINIMAL_RUNS:
        row = load_json(path)
        row["label"] = label
        row["color"] = color
        minimal.append(row)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    panel_framework_scores(axes[0, 0], matrix)
    panel_minimal_entropy(axes[0, 1], minimal)
    panel_minimal_complete(axes[1, 0], minimal)
    panel_goal_coverage(axes[1, 1])
    fig.text(
        0.5,
        -0.01,
        "C. elegans: block-Schur; mouse panels: historical residual approximation. "
        "Equal-time fits do not test H1-H5.",
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
