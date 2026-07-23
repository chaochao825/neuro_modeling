"""Publication plot bound to the summarized Exp33 ORBIT artifacts."""

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
    "train_fixed_best",
    "reward_only_local",
    "credit_shuffled_local",
    "oracle_per_frame",
)
METHOD_LABELS = (
    "Proto",
    "Gain",
    "Delta",
    "Temporal",
    "Fit-fixed",
    "Local",
    "Credit-shuf.",
    "Oracle",
)
COLORS = {
    "prototype": "#4C78A8",
    "gain": "#72B7B2",
    "delta": "#54A24B",
    "temporal": "#F58518",
    "train_fixed_best": "#9D755D",
    "reward_only_local": "#E45756",
    "credit_shuffled_local": "#B279A2",
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
    if not {"condition", "n_frames"} <= set(raw.columns):
        raise ValueError("raw panel lacks required columns")
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7))

    ax = axes[0, 0]
    wide = user_panel.pivot(
        index="user_id", columns="condition", values="user_video_mean_accuracy"
    )
    means = np.asarray([wide[name].mean() for name in METHOD_ORDER])
    positions = np.arange(len(METHOD_ORDER))
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
            s=15,
            facecolor="white",
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    ax.set_xticks(positions, METHOD_LABELS, rotation=35, ha="right")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.text(-0.15, 1.04, "a", transform=ax.transAxes, fontweight="bold")

    ax = axes[0, 1]
    comparisons = (
        ("train_fixed_best", "Fit-fixed", "#9D755D"),
        ("credit_shuffled_local", "Credit-shuffled", "#B279A2"),
    )
    local = wide["reward_only_local"]
    for x, (condition, label, color) in enumerate(comparisons):
        difference = 100.0 * (local - wide[condition])
        ax.scatter(
            np.full(len(difference), x),
            difference,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=28,
            alpha=0.85,
        )
        ax.hlines(
            float(difference.mean()), x - 0.22, x + 0.22, color="black", linewidth=2
        )
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1], [item[1] for item in comparisons])
    ax.set_ylabel("Local accuracy difference (pp)")
    ax.text(-0.15, 1.04, "b", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 0]
    for user_id, frame in headroom.groupby("user_id", sort=True):
        ax.scatter(
            100.0 * frame["action_disagreement"],
            100.0 * frame["oracle_gain"],
            s=22,
            alpha=0.55,
            label=str(user_id),
        )
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.axvline(5.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Frames with actuator disagreement (%)")
    ax.set_ylabel("Per-frame oracle headroom (pp)")
    if headroom["user_id"].nunique() <= 8:
        ax.legend(loc="best", ncol=2)
    ax.text(-0.15, 1.04, "c", transform=ax.transAxes, fontweight="bold")

    ax = axes[1, 1]
    local_rows = raw.loc[raw["condition"] == "reward_only_local"]
    fractions = []
    for action in range(4):
        column = f"action_{action}_fraction"
        if column in local_rows:
            fractions.append(float(local_rows[column].fillna(0.0).mean()))
        else:
            fractions.append(0.0)
    ax.bar(
        np.arange(4),
        100.0 * np.asarray(fractions),
        color=[COLORS[name] for name in METHOD_ORDER[:4]],
        width=0.7,
        edgecolor="black",
        linewidth=0.35,
    )
    ax.set_xticks(np.arange(4), METHOD_LABELS[:4], rotation=20, ha="right")
    ax.set_ylabel("Selected frames (%)")
    ax.text(-0.15, 1.04, "d", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "results/exp33_orbit_streaming_fewshot"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "figures/exp33_orbit_streaming_fewshot.pdf"),
    )
    args = parser.parse_args()
    root = Path(args.summary_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    user_panel = pd.read_csv(root / "user_panel.csv")
    headroom = pd.read_csv(root / "headroom_panel.csv")
    raw = pd.read_csv(root / "raw_video_panel.csv")
    figure = make_figure(user_panel, headroom, raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
