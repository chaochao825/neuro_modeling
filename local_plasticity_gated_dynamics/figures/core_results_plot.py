"""Multi-panel Phase-1 figure bound to results/raw_metrics.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.claims import select_latest_attempts  # noqa: E402

try:
    from .plot_style import COLORS, LINE_STYLES, save_figure, setup_style
except ImportError:  # direct script execution
    from plot_style import COLORS, LINE_STYLES, save_figure, setup_style


METRICS = (
    ("effective_rank", "Effective rank (singular-value entropy)"),
    ("latent_r2", "Held-out latent $R^2$"),
    ("rollout_normalized_rmse", "100-step rollout NRMSE"),
    ("plasticity_cost", "Plasticity path cost ($L_1$)"),
)


def _profile(frame: pd.DataFrame) -> pd.DataFrame:
    if "profile" not in frame:
        return frame
    formal = frame.loc[frame["profile"] == "formal"]
    return formal if not formal.empty else frame.loc[frame["profile"] == "smoke"]


def _latest_attempt(frame: pd.DataFrame) -> pd.DataFrame:
    return select_latest_attempts(frame)


def plot_core_results(raw: pd.DataFrame):
    setup_style()
    if "experiment" in raw:
        selected = raw.loc[
            raw["experiment"] == "exp01_feedback_dimension_sweep"
        ].copy()
    else:
        selected = raw.iloc[:0].copy()
    selected = _latest_attempt(_profile(selected))
    if "grid" in selected:
        selected = selected.loc[selected["grid"] == "core"]
    if "status" in selected:
        selected = selected.loc[selected["status"] == "complete"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    axes = axes.ravel()
    if selected.empty:
        for index, ax in enumerate(axes):
            ax.text(0.5, 0.5, "No complete feedback-sweep metrics", ha="center", va="center")
            ax.text(0.02, 0.96, f"({chr(97 + index)})", transform=ax.transAxes, va="top")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig
    selected["feedback_dim"] = pd.to_numeric(selected["feedback_dim"])
    # A seed is the independent replicate.  Collapse any accidental repeated
    # rows within a seed/cell before computing between-seed error bars.
    value_columns = [metric for metric, _ in METRICS if metric in selected]
    selected = (
        selected.groupby(
            ["seed", "feedback_mode", "feedback_dim"],
            as_index=False,
            dropna=False,
        )[value_columns]
        .mean(numeric_only=True)
    )
    for axis_index, (metric, ylabel) in enumerate(METRICS):
        ax = axes[axis_index]
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        for mode_index, (mode, group) in enumerate(selected.groupby("feedback_mode", sort=True)):
            summary = group.groupby("feedback_dim")[metric].agg(["mean", "std", "count"]).reset_index()
            error = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
            ax.errorbar(
                summary["feedback_dim"],
                summary["mean"],
                yerr=error,
                color=COLORS[mode_index % len(COLORS)],
                linestyle=LINE_STYLES[mode_index % len(LINE_STYLES)],
                marker=("o", "s", "^", "D")[mode_index % 4],
                linewidth=1.7,
                markersize=4,
                capsize=2,
                label=mode,
            )
        ax.set_xscale("log", base=2)
        ticks = sorted(selected["feedback_dim"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(value)) for value in ticks])
        ax.set_xlabel("Feedback dimension (channels)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.text(0.02, 0.96, f"({chr(97 + axis_index)})", transform=ax.transAxes, va="top")
    axes[0].legend(loc="best", ncol=2)
    axes[0].set_title("Core grid only; uncertainty across seeds", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()
    root = Path(args.results_root)
    raw_path = root / "raw_metrics.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    fig = plot_core_results(raw)
    save_figure(fig, "core_results", root)
    plt.close(fig)


if __name__ == "__main__":
    main()
