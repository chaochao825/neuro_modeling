#!/usr/bin/env python3
"""Render the preregistered Exp36 collector-level result panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOTTED_CONDITIONS = (
    "current_frame",
    "cumulative",
    "fixed_forgetting",
    "jsd_change_reset",
    "oracle_change_reset",
)
LABELS = {
    "current_frame": "Current",
    "cumulative": "Cumulative",
    "fixed_forgetting": "Forgetting",
    "jsd_change_reset": "Change reset",
    "oracle_change_reset": "Oracle",
    "change_reset_over_cumulative_switch": "Hidden: change − cumulative",
    "change_reset_over_fixed_forgetting_switch": "Hidden: change − forgetting",
    "change_reset_over_fixed_forgetting_post_switch": "Post-switch: change − forgetting",
    "change_reset_over_cumulative_natural": "Natural: change − cumulative",
}
COLORS = {
    "natural": "#0072B2",
    "hidden_switch": "#D55E00",
    "change": "#009E73",
    "control": "#CC79A7",
}


def _validate(collectors: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    collector_required = {
        "collector_id",
        "panel",
        "condition",
        "accuracy",
        "false_alarms_per_1000",
        "median_detection_delay",
    }
    comparison_required = {
        "comparison",
        "mean_difference",
        "ci_low",
        "ci_high",
    }
    if not collector_required <= set(collectors.columns):
        raise ValueError("collector panel misses Exp36 plotting columns")
    if not comparison_required <= set(comparisons.columns):
        raise ValueError("comparison panel misses Exp36 plotting columns")
    if collectors.duplicated(["collector_id", "panel", "condition"]).any():
        raise ValueError("collector panel contains duplicate cells")
    expected_comparisons = set(LABELS) - set(PLOTTED_CONDITIONS)
    if set(comparisons["comparison"]) != expected_comparisons:
        raise ValueError("comparison panel does not match the registered family")


def _wide(
    collectors: pd.DataFrame, *, panel: str, metric: str = "accuracy"
) -> pd.DataFrame:
    return collectors.loc[collectors["panel"] == panel].pivot(
        index="collector_id", columns="condition", values=metric
    )


def make_figure(
    collectors: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    max_false_alarms: float = 5.0,
    max_delay: float = 8.0,
) -> plt.Figure:
    _validate(collectors, comparisons)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    ax = axes[0, 0]
    x = np.arange(len(PLOTTED_CONDITIONS), dtype=float)
    for offset, panel in ((-0.12, "natural"), (0.12, "hidden_switch")):
        wide = _wide(collectors, panel=panel)
        values = wide.loc[:, list(PLOTTED_CONDITIONS)].to_numpy(float)
        for row in values:
            ax.plot(
                x + offset,
                row,
                color=COLORS[panel],
                alpha=0.16,
                linewidth=0.7,
                zorder=1,
            )
        ax.scatter(
            x + offset,
            np.mean(values, axis=0),
            s=42,
            color=COLORS[panel],
            edgecolor="white",
            linewidth=0.6,
            label="Natural" if panel == "natural" else "Hidden switch",
            zorder=3,
        )
    ax.set_xticks(x, [LABELS[name] for name in PLOTTED_CONDITIONS], rotation=20)
    ax.set_ylabel("Collector-level accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[0, 1]
    ordered = comparisons.set_index("comparison").loc[
        [item for item in LABELS if item not in PLOTTED_CONDITIONS]
    ]
    y = np.arange(len(ordered), dtype=float)[::-1]
    means = ordered["mean_difference"].to_numpy(float)
    lows = ordered["ci_low"].to_numpy(float)
    highs = ordered["ci_high"].to_numpy(float)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="o",
        color="#222222",
        ecolor="#666666",
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.0, color="#777777", linewidth=1, linestyle="--")
    ax.axvline(0.03, color="#E69F00", linewidth=1, linestyle=":")
    ax.set_yticks(y, [LABELS[name] for name in ordered.index])
    ax.set_xlabel("Paired accuracy difference")
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1, 0]
    natural = _wide(
        collectors.loc[collectors["condition"] == "jsd_change_reset"],
        panel="natural",
        metric="false_alarms_per_1000",
    )["jsd_change_reset"]
    hidden = _wide(
        collectors.loc[collectors["condition"] == "jsd_change_reset"],
        panel="hidden_switch",
        metric="median_detection_delay",
    )["jsd_change_reset"]
    common = natural.index.intersection(hidden.index)
    ax.scatter(
        natural.loc[common],
        hidden.loc[common],
        color=COLORS["change"],
        edgecolor="white",
        linewidth=0.5,
        s=38,
    )
    for collector in common:
        ax.annotate(
            str(collector),
            (float(natural.loc[collector]), float(hidden.loc[collector])),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6,
            alpha=0.75,
        )
    ax.axvline(max_false_alarms, color="#D55E00", linestyle="--", linewidth=1)
    ax.axhline(max_delay, color="#D55E00", linestyle="--", linewidth=1)
    ax.set_xlabel("Natural false alarms / 1,000 frames")
    ax.set_ylabel("Median detection delay (frames)")
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1, 1]
    hidden_wide = _wide(collectors, panel="hidden_switch")
    controls = (
        ("jsd_score_no_reset", "Score only"),
        ("matched_shifted_reset", "Shifted reset"),
    )
    rng = np.random.default_rng(0)
    for index, (condition, label) in enumerate(controls):
        difference = (
            hidden_wide["jsd_change_reset"] - hidden_wide[condition]
        ).to_numpy(float)
        jitter = rng.uniform(-0.07, 0.07, size=difference.size)
        ax.scatter(
            np.full(difference.size, index) + jitter,
            difference,
            s=24,
            alpha=0.7,
            color=COLORS["control"],
            edgecolor="none",
        )
        ax.scatter(
            [index],
            [np.mean(difference)],
            marker="D",
            s=48,
            color="#222222",
            zorder=3,
        )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(controls)), [label for _, label in controls])
    ax.set_ylabel("Change-reset accuracy advantage")
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)

    figure.tight_layout(w_pad=2.0, h_pad=2.0)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    root = Path(args.summary_dir).expanduser().resolve()
    collectors = pd.read_csv(root / "collector_panel.csv")
    comparisons = pd.read_csv(root / "primary_collector_comparisons.csv")
    figure = make_figure(collectors, comparisons)
    prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else root / "exp36_change_aware_prefix"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)
    print(prefix)


if __name__ == "__main__":
    main()
