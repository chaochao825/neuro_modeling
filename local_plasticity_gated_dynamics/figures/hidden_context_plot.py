"""Seed-level P2 hidden-context heatmaps and intervention contrasts."""

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
    from .plot_style import COLORS, save_figure, setup_style
except ImportError:  # direct script execution
    from plot_style import COLORS, save_figure, setup_style


def _complete_p2(raw: pd.DataFrame) -> pd.DataFrame:
    if "experiment" not in raw:
        return raw.iloc[:0].copy()
    selected = raw.loc[raw["experiment"].eq("exp09_hidden_context_gate")].copy()
    if "profile" in selected:
        formal = selected.loc[selected["profile"].eq("formal")]
        selected = (
            formal
            if not formal.empty
            else selected.loc[selected["profile"].eq("smoke")]
        )
    selected = select_latest_attempts(selected)
    if "status" not in selected:
        return selected.iloc[:0].copy()
    return selected.loc[selected["status"].eq("complete")].copy()


def _empty_axes(axes: np.ndarray) -> None:
    for index, ax in enumerate(axes.ravel()):
        ax.text(
            0.5, 0.5, "No complete hidden-context metrics", ha="center", va="center"
        )
        ax.text(0.02, 0.96, f"({chr(97 + index)})", transform=ax.transAxes, va="top")
        ax.set_xticks([])
        ax.set_yticks([])


def _paired_effect(
    frame: pd.DataFrame,
    *,
    metric: str,
    first_gate: str,
    second_gate: str,
) -> pd.DataFrame:
    base = frame.loc[frame["intervention"].eq("none")].copy()
    base[metric] = pd.to_numeric(base[metric], errors="coerce")
    pivot = base.pivot_table(
        index=["seed", "cue_reliability", "context_hazard"],
        columns="gate_model",
        values=metric,
        aggfunc="mean",
    )
    if first_gate not in pivot or second_gate not in pivot:
        return pd.DataFrame()
    effect = (pivot[first_gate] - pivot[second_gate]).rename("effect").dropna()
    return effect.reset_index()


def _draw_heatmap(
    ax: plt.Axes,
    effects: pd.DataFrame,
    *,
    label: str,
    panel: str,
) -> None:
    if effects.empty:
        ax.text(0.5, 0.5, "No paired cells", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    summary = effects.groupby(["context_hazard", "cue_reliability"], as_index=False)[
        "effect"
    ].mean()
    matrix = (
        summary.pivot(
            index="context_hazard", columns="cue_reliability", values="effect"
        )
        .sort_index()
        .sort_index(axis=1)
    )
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    limit = max(limit, 1e-6)
    image = ax.imshow(
        matrix.to_numpy(),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        origin="lower",
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([f"{value:g}" for value in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"{value:g}" for value in matrix.index])
    ax.set_xlabel("Cue reliability $q$")
    ax.set_ylabel("Context hazard $h$")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 * limit else "black",
                fontsize=8,
            )
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label(label)
    ax.text(0.02, 0.96, panel, transform=ax.transAxes, va="top")


def plot_hidden_context(raw: pd.DataFrame):
    setup_style()
    selected = _complete_p2(raw)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    if selected.empty or not {
        "seed",
        "cue_reliability",
        "context_hazard",
        "gate_model",
        "intervention",
    } <= set(selected):
        _empty_axes(axes)
        fig.tight_layout()
        return fig

    for column in ("cue_reliability", "context_hazard"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    accuracy_effect = _paired_effect(
        selected,
        metric="behavior_balanced_accuracy",
        first_gate="md_recurrent_belief",
        second_gate="no_gate",
    )
    _draw_heatmap(
        axes[0, 0],
        accuracy_effect,
        label="MD minus no-gate accuracy",
        panel="(a)",
    )
    nll_effect = _paired_effect(
        selected,
        metric="context_nll",
        first_gate="no_gate",
        second_gate="md_recurrent_belief",
    )
    _draw_heatmap(
        axes[0, 1],
        nll_effect,
        label="No-gate minus MD NLL (nats/trial)",
        panel="(b)",
    )

    intact = selected.loc[selected["intervention"].eq("none")].copy()
    if "behavior_balanced_accuracy" in intact:
        intact["behavior_balanced_accuracy"] = pd.to_numeric(
            intact["behavior_balanced_accuracy"], errors="coerce"
        )
        seed_q = (
            intact.groupby(["seed", "cue_reliability", "gate_model"], as_index=False)[
                "behavior_balanced_accuracy"
            ]
            .mean()
            .dropna()
        )
        for index, (gate, group) in enumerate(seed_q.groupby("gate_model", sort=True)):
            summary = group.groupby("cue_reliability")[
                "behavior_balanced_accuracy"
            ].agg(["mean", "std", "count"])
            error = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
            axes[1, 0].errorbar(
                summary.index,
                summary["mean"],
                yerr=error,
                color=COLORS[index % len(COLORS)],
                marker=("o", "s", "^", "D", "v")[index % 5],
                linewidth=1.5,
                capsize=2,
                label=gate,
            )
        axes[1, 0].legend(loc="best", fontsize=7)
    axes[1, 0].set_xlabel("Cue reliability $q$")
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[1, 0].grid(True, linestyle=":", alpha=0.25)
    axes[1, 0].text(0.02, 0.96, "(c)", transform=axes[1, 0].transAxes, va="top")

    md = selected.loc[selected["gate_model"].eq("md_recurrent_belief")].copy()
    if "behavior_balanced_accuracy" in md:
        md["behavior_balanced_accuracy"] = pd.to_numeric(
            md["behavior_balanced_accuracy"], errors="coerce"
        )
        pivot = md.pivot_table(
            index=["seed", "cue_reliability", "context_hazard"],
            columns="intervention",
            values="behavior_balanced_accuracy",
            aggfunc="mean",
        )
        labels = (
            [name for name in ("clamp", "delay", "shuffle") if name in pivot]
            if "none" in pivot
            else []
        )
        if labels:
            seed_drops = {}
            for name in labels:
                effect = pivot["none"] - pivot[name]
                if name == "delay":
                    # P2k is preregistered only for the high-hazard panel.
                    hazards = effect.index.get_level_values("context_hazard")
                    effect = effect[hazards.isin((0.10, 0.20))]
                seed_drops[name] = effect.groupby(level="seed").mean()
            drops = pd.DataFrame(seed_drops)
            means = drops.mean(axis=0)
            errors = drops.sem(axis=0).fillna(0.0)
            axes[1, 1].bar(
                np.arange(len(labels)),
                means.to_numpy(),
                yerr=errors.to_numpy(),
                color=[COLORS[index + 1] for index in range(len(labels))],
                capsize=3,
            )
            axes[1, 1].set_xticks(np.arange(len(labels)))
            axes[1, 1].set_xticklabels(
                ["delay\n(h=.10/.20)" if name == "delay" else name for name in labels]
            )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("Intact minus intervention accuracy")
    axes[1, 1].text(0.02, 0.96, "(d)", transform=axes[1, 1].transAxes, va="top")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parents[1] / "results"),
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    raw_path = root / "raw_metrics.csv"
    raw = (
        pd.read_csv(raw_path, low_memory=False) if raw_path.exists() else pd.DataFrame()
    )
    figure = plot_hidden_context(raw)
    save_figure(figure, "hidden_context", root)
    plt.close(figure)


if __name__ == "__main__":
    main()
