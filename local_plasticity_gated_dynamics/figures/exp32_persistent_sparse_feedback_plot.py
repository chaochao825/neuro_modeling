"""Data-bound publication figure for Exp32 sparse persistent reward belief."""

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
    "train_fixed_best": "Train-fixed",
    "cumulative_sample_average": "No forgetting",
    "persistent_rpe_local": "Persistent local",
    "credit_shuffled_local": "Opposite eligibility",
    "bayes_reward_filter": "Bayes reward filter",
    "oracle_hidden_state": "Hidden-state oracle",
}


def _paired_effect(raw: pd.DataFrame) -> pd.DataFrame:
    pivot = raw.pivot_table(
        index=["seed", "hazard", "feedback_fraction", "feedback_delay"],
        columns="selector_mode",
        values="full_stream_accuracy",
        aggfunc="first",
    ).reset_index()
    pivot["local_minus_fixed"] = (
        pivot["persistent_rpe_local"] - pivot["train_fixed_best"]
    )
    return pivot


def plot_exp32(output_dir: Path) -> tuple[Path, Path]:
    raw_path = output_dir / "raw_metrics.csv.gz"
    seeds_path = output_dir / "seed_summary.csv"
    summary_path = output_dir / "summary.json"
    if not raw_path.exists() or not seeds_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Exp32 raw_metrics.csv.gz, seed_summary.csv and summary.json are required"
        )
    raw = pd.read_csv(raw_path)
    seeds = pd.read_csv(seeds_path)
    summary = pd.read_json(summary_path, typ="series")
    effect = _paired_effect(raw)
    primary_hazard = float(summary["primary_hazard"])
    primary_delay = int(summary["primary_feedback_delay"])
    primary_fraction = float(summary["primary_feedback_fraction"])

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4))

    selected = effect[effect["feedback_delay"] == primary_delay]
    for index, hazard in enumerate(sorted(selected["hazard"].unique())):
        frame = (
            selected[np.isclose(selected["hazard"], hazard)]
            .groupby("feedback_fraction", as_index=False)
            .agg(
                mean=("local_minus_fixed", "mean"),
                sd=("local_minus_fixed", "std"),
                n=("seed", "nunique"),
            )
            .sort_values("feedback_fraction")
        )
        axes[0, 0].errorbar(
            frame["feedback_fraction"],
            frame["mean"],
            yerr=frame["sd"].fillna(0.0) / np.sqrt(frame["n"]),
            marker="o",
            linewidth=1.4,
            capsize=2,
            color=COLORS[index % len(COLORS)],
            label=f"hazard {hazard:g}",
        )
    axes[0, 0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0, 0].set_xscale("log", base=2)
    fractions = sorted(selected["feedback_fraction"].unique())
    fraction_labels = {
        0.03125: "1/32",
        0.0625: "1/16",
        0.125: "1/8",
        0.25: "1/4",
        0.5: "1/2",
    }
    axes[0, 0].set_xticks(
        fractions,
        [fraction_labels.get(float(value), f"{value:g}") for value in fractions],
    )
    axes[0, 0].set_xlabel("Observable reward fraction")
    axes[0, 0].set_ylabel("Persistent local minus train-fixed")
    axes[0, 0].legend(ncol=2, loc="best")

    delay_frame = effect[
        np.isclose(effect["hazard"], primary_hazard)
        & np.isclose(effect["feedback_fraction"], primary_fraction)
    ]
    delay_summary = (
        delay_frame.groupby("feedback_delay", as_index=False)
        .agg(
            mean=("local_minus_fixed", "mean"),
            sd=("local_minus_fixed", "std"),
            n=("seed", "nunique"),
        )
        .sort_values("feedback_delay")
    )
    axes[0, 1].errorbar(
        delay_summary["feedback_delay"],
        delay_summary["mean"],
        yerr=delay_summary["sd"].fillna(0.0) / np.sqrt(delay_summary["n"]),
        marker="o",
        color=COLORS[0],
        linewidth=1.6,
        capsize=2,
    )
    axes[0, 1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0, 1].set_xlabel("Reward delay (trials)")
    axes[0, 1].set_ylabel("Persistent local minus train-fixed")

    method_order = tuple(DISPLAY)
    x = np.arange(len(method_order), dtype=np.float64)
    for _, row in seeds.iterrows():
        axes[1, 0].plot(
            x,
            [row[f"{mode}_accuracy"] for mode in method_order],
            color="0.80",
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    for index, mode in enumerate(method_order):
        values = seeds[f"{mode}_accuracy"].to_numpy(float)
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else 0.0
        axes[1, 0].scatter(
            np.full(len(values), index) + jitter,
            values,
            color=COLORS[index % len(COLORS)],
            s=15,
            zorder=2,
        )
        axes[1, 0].plot(
            [index - 0.15, index + 0.15],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=1.7,
            zorder=3,
        )
    axes[1, 0].set_xticks(
        x, [DISPLAY[mode] for mode in method_order], rotation=22, ha="right"
    )
    axes[1, 0].set_ylabel("Primary full-stream accuracy")

    surface = (
        effect[effect["feedback_delay"] == primary_delay]
        .groupby(["hazard", "feedback_fraction"])["local_minus_fixed"]
        .mean()
        .unstack("feedback_fraction")
        .sort_index()
        .sort_index(axis=1)
    )
    maximum = max(0.02, float(np.nanmax(np.abs(surface.to_numpy(float)))))
    image = axes[1, 1].imshow(
        surface.to_numpy(float),
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
        aspect="auto",
        origin="lower",
    )
    axes[1, 1].set_xticks(
        range(surface.shape[1]),
        [f"{value:g}" for value in surface.columns],
    )
    axes[1, 1].set_yticks(
        range(surface.shape[0]),
        [f"{value:g}" for value in surface.index],
    )
    axes[1, 1].set_xlabel("Observable reward fraction")
    axes[1, 1].set_ylabel("Hidden-state hazard")
    colorbar = fig.colorbar(image, ax=axes[1, 1], shrink=0.84)
    colorbar.set_label("Local minus fixed accuracy")
    for row in range(surface.shape[0]):
        for column in range(surface.shape[1]):
            value = float(surface.iloc[row, column])
            axes[1, 1].text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > maximum * 0.55 else "black",
            )

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.09, 1.06, label, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8, h_pad=2.0)
    save_figure(fig, "exp32_persistent_sparse_feedback", output_dir)
    plt.close(fig)
    return (
        output_dir / "exp32_persistent_sparse_feedback.png",
        output_dir / "exp32_persistent_sparse_feedback.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="results/exp32_persistent_sparse_feedback"
    )
    args = parser.parse_args()
    for path in plot_exp32(Path(args.output_dir)):
        print(path.resolve())


if __name__ == "__main__":
    main()
