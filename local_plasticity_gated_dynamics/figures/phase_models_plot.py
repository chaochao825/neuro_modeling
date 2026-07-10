"""Phase, context-task, and real-data summaries bound to raw_metrics.csv."""

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


def _profile(frame: pd.DataFrame) -> pd.DataFrame:
    if "profile" not in frame:
        return frame
    formal = frame.loc[frame["profile"] == "formal"]
    return formal if not formal.empty else frame.loc[frame["profile"] == "smoke"]


def _complete_profile(frame: pd.DataFrame) -> pd.DataFrame:
    selected = select_latest_attempts(_profile(frame))
    if "status" in selected:
        selected = selected.loc[selected["status"] == "complete"]
    return selected


def _bar_metric(ax, frame, category, metric, ylabel, *, unit):
    if frame.empty or metric not in frame:
        ax.text(0.5, 0.5, "Complete metrics unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        return
    data = frame.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    if unit not in data:
        ax.text(0.5, 0.5, f"Missing {unit} identifiers", ha="center", va="center")
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        return
    # Folds, views, and repeated cells are nested observations.  Reduce them
    # to one value per declared independent unit/category before uncertainty.
    independent = (
        data.dropna(subset=[category, unit, metric])
        .groupby([unit, category], as_index=False, dropna=False)[metric]
        .mean()
    )
    summary = (
        independent.groupby(category)[metric]
        .agg(["mean", "std", "count"])
        .dropna(subset=["mean"])
        .reset_index()
    )
    error = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    positions = np.arange(len(summary))
    ax.bar(positions, summary["mean"], yerr=error, capsize=2, color=COLORS[: len(summary)])
    ax.set_xticks(positions)
    ax.set_xticklabels(summary[category], rotation=25, ha="right")
    ax.set_ylabel(ylabel)


def plot_phase_models(raw: pd.DataFrame):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    phase_all = (
        raw.loc[raw["experiment"] == "exp04_phase_gating"]
        if "experiment" in raw
        else raw.iloc[:0]
    )
    phase = _complete_profile(phase_all)
    _bar_metric(
        axes[0, 0], phase, "phase_condition", "decoding_accuracy", "Decoding accuracy", unit="seed"
    )
    axes[0, 0].set_title("Rate-matched phase gate")
    _bar_metric(
        axes[0, 1],
        phase,
        "phase_condition",
        "cross_validated_transfer_r2_gain",
        r"Cross-validated transfer $\Delta R^2$",
        unit="seed",
    )
    axes[0, 1].set_title("Rate-matched phase gate")
    context_all = raw.loc[
        raw["experiment"].isin(
            ["exp02_context_ei_oracle_gate", "exp03_context_ei_learned_gate"]
        )
    ] if "experiment" in raw else raw.iloc[:0]
    context = _complete_profile(context_all)
    if not context.empty and {"experiment", "condition"} <= set(context):
        if "architecture" in context:
            preferred = "ei_n512_fi20_gain1"
            available = context["architecture"].dropna().astype(str)
            if preferred not in set(available):
                ei_names = sorted(name for name in set(available) if name.startswith("ei_"))
                preferred = ei_names[0] if ei_names else sorted(set(available))[0]
            context = context.loc[context["architecture"].astype(str) == preferred]
        context = context.loc[context["condition"].isin(["local", "bptt", "no-gate"])]
        context["task_condition"] = (
            context["experiment"].map(
                {
                    "exp02_context_ei_oracle_gate": "oracle",
                    "exp03_context_ei_learned_gate": "learned",
                }
            ).fillna("context")
            + ":"
            + context["condition"].astype(str)
        )
    category = "task_condition" if "task_condition" in context else "condition"
    _bar_metric(
        axes[1, 0], context, category, "accuracy", "Context-task accuracy", unit="seed"
    )
    axes[1, 0].set_title("Independent seed summaries")

    sequence_all = (
        raw.loc[raw["experiment"] == "exp05_sequence_real_data"]
        if "experiment" in raw
        else raw.iloc[:0]
    )
    ibl_all = (
        raw.loc[raw["experiment"] == "exp06_ibl_context_switch"]
        if "experiment" in raw
        else raw.iloc[:0]
    )
    # Never pool distinct datasets. Prefer complete sequence-memory evidence;
    # failed-only sequence artifacts must not hide available complete IBL fits.
    sequence = _complete_profile(sequence_all)
    ibl = _complete_profile(ibl_all)
    real = sequence if not sequence.empty else ibl
    real_label = "Sequence-memory LDS" if not sequence.empty else "IBL LDS"
    if "fold" in real:
        real = real.loc[real["fold"].astype(str) != "unseen_combination"]
    _bar_metric(
        axes[1, 1],
        real.loc[real["model_family"].isin(["common", "shared", "full"])]
        if "model_family" in real
        else real.iloc[:0],
        "model_family",
        "heldout_nll_per_scalar",
        "Held-out NLL / scalar",
        unit="session_id",
    )
    axes[1, 1].set_title(f"{real_label}; folds nested in session")
    for index, ax in enumerate(axes.ravel()):
        ax.text(0.02, 0.96, f"({chr(97 + index)})", transform=ax.transAxes, va="top")
        ax.grid(axis="y", linestyle=":", alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()
    root = Path(args.results_root)
    raw_path = root / "raw_metrics.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    fig = plot_phase_models(raw)
    save_figure(fig, "phase_models", root)
    plt.close(fig)


if __name__ == "__main__":
    main()
