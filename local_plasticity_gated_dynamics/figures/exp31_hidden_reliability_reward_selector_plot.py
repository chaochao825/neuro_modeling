"""Data-bound publication figure for the Exp31 hidden-demand panel."""

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
    "train_fixed_best_macro_accuracy": "Train-fixed",
    "matched_probe_random_macro_accuracy": "Matched random",
    "reward_only_local_macro_accuracy": "Reward-only local",
    "oracle_hidden_train_map_macro_accuracy": "Hidden-state oracle",
}


def plot_exp31(output_dir: Path) -> tuple[Path, Path]:
    conditions_path = output_dir / "conditions.csv"
    seeds_path = output_dir / "seed_summary.csv"
    if not conditions_path.exists() or not seeds_path.exists():
        raise FileNotFoundError("Exp31 conditions.csv and seed_summary.csv are required")
    conditions = pd.read_csv(conditions_path)
    seeds = pd.read_csv(seeds_path)
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.3))

    fixed = conditions[
        conditions["actuator_mode"].isin(["fixed_routing", "fixed_associative"])
    ]
    pivot = fixed.pivot_table(
        index=["association_load", "distractor_writes", "interference_pressure"],
        columns="actuator_mode",
        values="mean_full_accuracy",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.sort_values("interference_pressure")
    advantage = pivot["fixed_associative"] - pivot["fixed_routing"]
    axes[0, 0].plot(
        pivot["interference_pressure"],
        advantage,
        marker="o",
        color=COLORS[0],
        linewidth=1.7,
    )
    axes[0, 0].axhline(0.0, color="0.65", linewidth=0.8)
    axes[0, 0].set_xlabel("Interference pressure, $(L-1+D)/d$")
    axes[0, 0].set_ylabel("Memory minus routing accuracy")

    memory = conditions[conditions["actuator_mode"] == "fixed_associative"]
    for index, reliability in enumerate(sorted(memory["direct_reliability"].unique())):
        frame = (
            memory[memory["direct_reliability"] == reliability]
            .groupby("interference_pressure", as_index=False)
            .agg(
                mean=("mean_full_accuracy", "mean"),
                sd=("mean_full_accuracy", "std"),
                n=("n_seeds", "max"),
            )
            .sort_values("interference_pressure")
        )
        axes[0, 1].errorbar(
            frame["interference_pressure"],
            frame["mean"],
            yerr=frame["sd"].fillna(0.0) / np.sqrt(frame["n"]),
            marker="o",
            linewidth=1.5,
            capsize=2,
            color=COLORS[index],
            label=f"Hidden cue reliability {reliability:.2f}",
        )
    shuffled = conditions[
        conditions["actuator_mode"] == "associative_query_shuffled"
    ]["mean_full_accuracy"].mean()
    axes[0, 1].axhline(shuffled, color="0.55", linestyle="--", linewidth=1.0, label="Query-shuffled")
    axes[0, 1].set_xlabel("Interference pressure, $(L-1+D)/d$")
    axes[0, 1].set_ylabel("Associative accuracy")
    axes[0, 1].legend(loc="best")

    columns = tuple(DISPLAY)
    x = np.arange(len(columns), dtype=np.float64)
    for _, row in seeds.iterrows():
        axes[1, 0].plot(
            x,
            [row[column] for column in columns],
            color="0.78",
            alpha=0.65,
            linewidth=0.7,
            zorder=1,
        )
    for index, column in enumerate(columns):
        values = seeds[column].to_numpy(float)
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else 0.0
        axes[1, 0].scatter(
            np.full(len(values), x[index]) + jitter,
            values,
            color=COLORS[index],
            s=17,
            zorder=2,
        )
        axes[1, 0].plot(
            [x[index] - 0.16, x[index] + 0.16],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=1.8,
            zorder=3,
        )
    axes[1, 0].set_xticks(x, [DISPLAY[column] for column in columns], rotation=18, ha="right")
    axes[1, 0].set_ylabel("Full-block accuracy (probe included)")

    contrasts = (
        ("reward_only_minus_fixed", "Local − fixed"),
        ("reward_only_minus_random", "Local − random"),
        ("associative_query_specificity", "Memory − shuffled"),
    )
    for index, (column, label) in enumerate(contrasts):
        values = seeds[column].to_numpy(float)
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else 0.0
        axes[1, 1].scatter(
            np.full(len(values), index) + jitter,
            values,
            color=COLORS[index],
            s=20,
        )
        axes[1, 1].plot(
            [index - 0.17, index + 0.17],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=1.8,
        )
    axes[1, 1].axhline(0.0, color="0.65", linewidth=0.8)
    axes[1, 1].set_xticks(range(len(contrasts)), [item[1] for item in contrasts], rotation=18, ha="right")
    axes[1, 1].set_ylabel("Seed-level paired contrast")

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8, h_pad=2.0)
    save_figure(fig, "exp31_hidden_reliability_reward_selector", output_dir)
    plt.close(fig)
    return (
        output_dir / "exp31_hidden_reliability_reward_selector.png",
        output_dir / "exp31_hidden_reliability_reward_selector.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="results/exp31_hidden_reliability_reward_selector"
    )
    args = parser.parse_args()
    for path in plot_exp31(Path(args.output_dir)):
        print(path.resolve())


if __name__ == "__main__":
    main()
