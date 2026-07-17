"""Data-bound multi-panel figure for the Exp30 exploratory trend."""

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


DISPLAY = {
    "routing": "Input routing",
    "low_rank": "1D internal state",
    "associative": "Associative memory",
    "associative_shuffled": "Shuffled memory",
}


def plot_exp30(output_dir: Path) -> tuple[Path, Path]:
    conditions_path = output_dir / "conditions.csv"
    seeds_path = output_dir / "seed_summary.csv"
    if not conditions_path.exists() or not seeds_path.exists():
        raise FileNotFoundError("Exp30 conditions.csv and seed_summary.csv are required")
    conditions = pd.read_csv(conditions_path)
    seeds = pd.read_csv(seeds_path)
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35))

    modes = ("routing", "low_rank", "associative", "associative_shuffled")
    for index, mode in enumerate(modes):
        frame = conditions[conditions["actuator_mode"] == mode].sort_values(
            "memory_demand"
        )
        if frame.empty:
            raise ValueError(f"missing Exp30 condition rows for {mode}")
        sem = frame["sd_score"].fillna(0.0) / np.sqrt(frame["n_seeds"])
        axes[0].errorbar(
            frame["memory_demand"],
            frame["mean_score"],
            yerr=sem,
            marker="o",
            linewidth=1.7,
            markersize=4.0,
            capsize=2.0,
            color=COLORS[index],
            label=DISPLAY[mode],
        )
    axes[0].axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    axes[0].set_xlabel("Associative-memory demand, $\\mu$")
    axes[0].set_ylabel("Train-normalized held-out score")
    axes[0].set_xticks(sorted(conditions["memory_demand"].unique()))
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.2,
    )
    axes[0].text(-0.18, 1.10, "a", transform=axes[0].transAxes, fontweight="bold")

    columns = (
        "fixed_best_macro_score",
        "matched_macro_score",
        "combined_macro_score",
    )
    labels = ("Fixed single", "Demand matched", "Combined")
    positions = np.arange(len(columns), dtype=np.float64)
    for _, row in seeds.iterrows():
        axes[1].plot(
            positions,
            [row[column] for column in columns],
            color="0.75",
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )
    for index, column in enumerate(columns):
        values = seeds[column].to_numpy(dtype=np.float64)
        jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else 0.0
        axes[1].scatter(
            np.full(len(values), positions[index]) + jitter,
            values,
            s=18,
            color=COLORS[index],
            alpha=0.8,
            zorder=2,
        )
        axes[1].plot(
            [positions[index] - 0.16, positions[index] + 0.16],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=2.0,
            zorder=3,
        )
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Seed-level macro score")
    axes[1].text(-0.18, 1.10, "b", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "exp30_associative_actuator_trend", output_dir)
    plt.close(fig)
    return (
        output_dir / "exp30_associative_actuator_trend.png",
        output_dir / "exp30_associative_actuator_trend.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="results/exp30_associative_actuator_trend_smoke"
    )
    args = parser.parse_args()
    paths = plot_exp30(Path(args.output_dir))
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
