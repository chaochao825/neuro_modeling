"""Data-bound diagnostic figure for the immutable Exp23 failure probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.plot_style import COLORS, save_figure, setup_style


def plot_exp23_probe(output_dir: Path) -> tuple[Path, Path]:
    seed_path = output_dir / "seed_contrasts.csv"
    condition_path = output_dir / "condition_summary.csv"
    if not seed_path.exists() or not condition_path.exists():
        raise FileNotFoundError("Exp23 probe seed and condition summaries are required")
    seeds = pd.read_csv(seed_path)
    conditions = pd.read_csv(condition_path)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25))

    metrics = (
        ("local_matched_gain", "Local, matched"),
        ("local_natural_gain", "Local, natural"),
        ("bptt_matched_gain", "BPTT, matched"),
        ("bptt_natural_gain", "BPTT, natural"),
    )
    for task_index, task in enumerate(("current", "delayed")):
        frame = seeds[seeds["task_variant"] == task]
        x = np.arange(len(metrics), dtype=np.float64) + (task_index - 0.5) * 0.22
        means = [frame[column].mean() for column, _ in metrics]
        sem = [
            frame[column].std(ddof=1) / np.sqrt(frame[column].count())
            for column, _ in metrics
        ]
        axes[0].errorbar(
            x,
            means,
            yerr=sem,
            marker="o",
            linestyle="none",
            capsize=2,
            color=COLORS[task_index],
            label=task,
        )
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].axhline(
        0.03,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label="local endpoint MCID",
    )
    axes[0].set_xticks(
        range(len(metrics)), [label for _, label in metrics], rotation=25, ha="right"
    )
    axes[0].set_ylabel("Balanced-accuracy gain over frozen")
    axes[0].set_title("Mean $\\pm$ SEM (post-hoc)")
    axes[0].legend(loc="best")

    chosen = conditions[
        conditions["condition"].isin(
            ["local_eprop", "exact_forward_sensitivity", "bptt_axis_only"]
        )
    ]
    labels = []
    values = []
    colors = []
    for task_index, task in enumerate(("current", "delayed")):
        frame = chosen[chosen["task_variant"] == task].set_index("condition")
        for condition, short in (
            ("local_eprop", "local"),
            ("exact_forward_sensitivity", "exact"),
            ("bptt_axis_only", "BPTT"),
        ):
            labels.append(f"{task}\n{short}")
            values.append(frame.loc[condition, "median_axis_rescale_ratio"])
            colors.append(COLORS[task_index])
    axes[1].bar(range(len(values)), values, color=colors)
    axes[1].axhline(1.0, color="0.45", linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[1].set_ylabel("Median matched / natural axis L2")

    current = seeds[seeds["task_variant"] == "current"]
    delayed = seeds[seeds["task_variant"] == "delayed"]
    axes[2].scatter(
        current["local_test_loss_change"],
        current["local_matched_gain"],
        color=COLORS[0],
        s=20,
        alpha=0.8,
        label="current",
    )
    axes[2].scatter(
        delayed["local_test_loss_change"],
        delayed["local_matched_gain"],
        color=COLORS[1],
        s=20,
        alpha=0.8,
        label="delayed",
    )
    axes[2].axhline(0.0, color="0.65", linewidth=0.8)
    axes[2].axvline(0.0, color="0.65", linewidth=0.8)
    axes[2].set_xlabel("Local test task-loss change")
    axes[2].set_ylabel("Local balanced-accuracy gain")
    axes[2].legend(loc="best")

    for label, ax in zip("abc", axes, strict=True):
        ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8)
    save_figure(fig, "exp23_failure_probe", output_dir)
    plt.close(fig)
    return (
        output_dir / "exp23_failure_probe.png",
        output_dir / "exp23_failure_probe.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/exp23_failure_probe")
    args = parser.parse_args()
    for path in plot_exp23_probe(Path(args.output_dir)):
        print(path.resolve())


if __name__ == "__main__":
    main()
