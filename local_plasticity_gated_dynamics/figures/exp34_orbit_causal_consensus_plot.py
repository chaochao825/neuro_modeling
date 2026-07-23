"""Publication figure bound to summarized Exp34 ORBIT artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = (
    "prototype",
    "gain",
    "delta",
    "temporal",
    "selection_fixed_best",
    "memoryless_consensus",
    "instantaneous_majority",
    "delayed_consensus",
    "causal_consensus",
    "oracle_per_frame",
)
METHOD_LABELS = (
    "Proto",
    "Gain",
    "Delta",
    "Temporal",
    "Val-fixed",
    "Memoryless",
    "Majority",
    "Delayed",
    "Causal",
    "Oracle",
)
COLORS = {
    "prototype": "#4C78A8",
    "gain": "#72B7B2",
    "delta": "#54A24B",
    "temporal": "#F58518",
    "selection_fixed_best": "#9D755D",
    "memoryless_consensus": "#B279A2",
    "instantaneous_majority": "#76B7B2",
    "delayed_consensus": "#FF9DA6",
    "causal_consensus": "#E45756",
    "oracle_per_frame": "#BAB0AC",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(
    user_panel: pd.DataFrame,
    headroom: pd.DataFrame,
    raw: pd.DataFrame,
) -> plt.Figure:
    required_user = {"user_id", "condition", "user_video_mean_accuracy"}
    if not required_user <= set(user_panel.columns):
        raise ValueError("user panel lacks required columns")
    if not {"user_id", "oracle_gain", "action_disagreement"} <= set(headroom.columns):
        raise ValueError("headroom panel lacks required columns")
    if not {"condition", "frame_accuracy", "mean_event_l1"} <= set(raw.columns):
        raise ValueError("raw panel lacks required columns")
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    wide = user_panel.pivot(
        index="user_id", columns="condition", values="user_video_mean_accuracy"
    )

    ax = axes[0, 0]
    positions = np.arange(len(METHOD_ORDER))
    means = np.asarray([wide[name].mean() for name in METHOD_ORDER])
    ax.bar(
        positions,
        100.0 * means,
        color=[COLORS[name] for name in METHOD_ORDER],
        width=0.72,
        edgecolor="black",
        linewidth=0.35,
    )
    for user_number, (_, row) in enumerate(wide.iterrows()):
        jitter = (user_number - (len(wide) - 1) / 2.0) * 0.035
        ax.scatter(
            positions + jitter,
            100.0 * np.asarray([row[name] for name in METHOD_ORDER]),
            s=13,
            facecolor="white",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_xticks(positions, METHOD_LABELS, rotation=38, ha="right")
    ax.set_ylabel("Task-video accuracy (%)")
    ax.text(-0.15, 1.04, "a", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    comparisons = (
        ("selection_fixed_best", "Val-fixed"),
        ("memoryless_consensus", "Memoryless"),
        ("instantaneous_majority", "Majority"),
        ("delayed_consensus", "Delayed"),
    )
    causal = wide["causal_consensus"]
    for x, (condition, label) in enumerate(comparisons):
        difference = 100.0 * (causal - wide[condition])
        ax.scatter(
            np.full(len(difference), x),
            difference,
            color=COLORS[condition],
            edgecolor="black",
            linewidth=0.4,
            s=26,
        )
        ax.hlines(float(difference.mean()), x - 0.22, x + 0.22, color="black", lw=2)
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(comparisons)), [item[1] for item in comparisons])
    ax.set_ylabel("Causal-consensus gain (pp)")
    ax.text(-0.15, 1.04, "b", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    fixed = wide["selection_fixed_best"]
    causal = wide["causal_consensus"]
    for index, user_id in enumerate(wide.index):
        ax.plot(
            [0, 1],
            100.0 * np.asarray([fixed.loc[user_id], causal.loc[user_id]]),
            color="#777777",
            linewidth=0.8,
            alpha=0.75,
        )
        ax.scatter(
            [0, 1],
            100.0 * np.asarray([fixed.loc[user_id], causal.loc[user_id]]),
            color=[COLORS["selection_fixed_best"], COLORS["causal_consensus"]],
            edgecolor="black",
            linewidth=0.35,
            s=22,
            label=str(user_id) if index == 0 else None,
        )
    ax.set_xticks([0, 1], ["Val-fixed", "Causal"])
    ax.set_ylabel("Paired user accuracy (%)")
    ax.text(-0.15, 1.04, "c", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    cost_frame = (
        raw.groupby("condition", as_index=False)[["frame_accuracy", "mean_event_l1"]]
        .mean()
        .set_index("condition")
    )
    label_offsets = {
        "prototype": (4, 2),
        "gain": (4, 2),
        "delta": (4, 2),
        "temporal": (4, -14),
        "selection_fixed_best": (4, 7),
        "memoryless_consensus": (8, 7),
        "instantaneous_majority": (8, -14),
        "delayed_consensus": (4, 2),
        "causal_consensus": (4, 2),
    }
    for condition in METHOD_ORDER[:-1]:
        row = cost_frame.loc[condition]
        ax.scatter(
            float(row["mean_event_l1"]),
            100.0 * float(row["frame_accuracy"]),
            color=COLORS[condition],
            edgecolor="black",
            linewidth=0.35,
            s=28,
        )
        ax.annotate(
            METHOD_LABELS[METHOD_ORDER.index(condition)],
            (float(row["mean_event_l1"]), 100.0 * float(row["frame_accuracy"])),
            xytext=label_offsets[condition],
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean event L1 (full bank charged)")
    ax.set_ylabel("Task-video accuracy (%)")
    ax.text(-0.15, 1.04, "d", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "results/exp34_orbit_causal_consensus"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "figures/exp34_orbit_causal_consensus.pdf"),
    )
    args = parser.parse_args()
    root = Path(args.summary_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    raw_path = root / "raw_video_panel.csv"
    if not raw_path.is_file():
        raw_path = root / "raw_video_panel.csv.gz"
    if not raw_path.is_file():
        raise FileNotFoundError("Exp34 summary lacks raw video panel")
    figure = make_figure(
        pd.read_csv(root / "user_panel.csv"),
        pd.read_csv(root / "headroom_panel.csv"),
        pd.read_csv(raw_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
