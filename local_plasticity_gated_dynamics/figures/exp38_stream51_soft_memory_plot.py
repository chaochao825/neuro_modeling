#!/usr/bin/env python3
"""Render result-bound qualification or external panels for Exp38."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#666666"

QUALIFICATION_EFFECTS = (
    ("stable_accumulation_gain", "Stable accumulation", "stable_gain_mcid"),
    ("oracle_adaptation_headroom", "Oracle headroom", "oracle_headroom_mcid"),
    ("cumulative_post_switch_harm", "Cumulative harm", "cumulative_harm_mcid"),
)
QUALIFICATION_GATES = (
    ("stable_accumulation_gate", "Accumulation"),
    ("oracle_headroom_gate", "Oracle"),
    ("cumulative_harm_gate", "Cumulative harm"),
    ("reachability_gate", "Reachability"),
)
EXTERNAL_CONDITIONS = (
    "current_frame",
    "cumulative",
    "fixed_forgetting",
    "sliding_window",
    "soft_memory",
    "hard_memory",
    "matched_shifted_memory",
    "oracle_memory",
)
CONDITION_LABELS = {
    "current_frame": "Current",
    "cumulative": "Cumulative",
    "fixed_forgetting": "Fixed forgetting",
    "sliding_window": "Window",
    "soft_memory": "Soft memory",
    "hard_memory": "Hard memory",
    "matched_shifted_memory": "Shifted timing",
    "oracle_memory": "Oracle",
}
COMPARISON_LABELS = {
    "soft_over_fixed_forgetting_hidden": "Soft − fixed",
    "soft_over_sliding_window_hidden": "Soft − window",
    "soft_over_hard_hidden": "Soft − hard",
    "soft_over_shifted_timing_hidden": "Soft − shifted",
    "soft_noninferior_current_natural": "Soft − current",
}


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
    )


def _qualification_rows(receipt: Mapping[str, Any]) -> pd.DataFrame:
    if receipt.get("receipt_type") != "exp38_qualification_gate":
        raise ValueError("not an Exp38 qualification receipt")
    rows: list[dict[str, Any]] = []
    seed_results = receipt.get("seed_results")
    if not isinstance(seed_results, list) or not seed_results:
        raise ValueError("qualification receipt has no seed results")
    for item in seed_results:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("qualification"), Mapping
        ):
            raise ValueError("qualification seed result is incomplete")
        rows.append(
            {
                "seed": int(item["seed"]),
                "passed": bool(item["passed"]),
                **dict(item["qualification"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    required = {
        "seed",
        "passed",
        "stable_accumulation_gain",
        "oracle_adaptation_headroom",
        "cumulative_post_switch_harm",
        "reachability_auc",
        "reachability_recall",
        "reachability_false_alarms_per_1000",
        "reachability_median_delay",
        *(name for name, _ in QUALIFICATION_GATES),
    }
    if not required <= set(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("qualification receipt misses registered plotting fields")
    numeric = frame[
        [
            "stable_accumulation_gain",
            "oracle_adaptation_headroom",
            "cumulative_post_switch_harm",
            "reachability_auc",
            "reachability_recall",
            "reachability_false_alarms_per_1000",
            "reachability_median_delay",
        ]
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("qualification plotting values must be finite")
    return frame


def make_qualification_figure(
    receipt: Mapping[str, Any], config: Mapping[str, Any]
) -> plt.Figure:
    """Plot every registered qualification gate without averaging it away."""

    frame = _qualification_rows(receipt)
    thresholds = config.get("qualification")
    if not isinstance(thresholds, Mapping):
        raise ValueError("configuration has no qualification thresholds")
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    seeds = frame["seed"].astype(str).tolist()
    x = np.arange(len(frame), dtype=float)

    ax = axes[0, 0]
    offsets = np.linspace(-0.22, 0.22, len(QUALIFICATION_EFFECTS))
    colors = (BLUE, ORANGE, GREEN)
    for offset, color, (field, label, threshold_field) in zip(
        offsets, colors, QUALIFICATION_EFFECTS, strict=True
    ):
        threshold = float(thresholds[threshold_field])
        margin = frame[field].to_numpy(float) - threshold
        ax.scatter(
            x + offset,
            margin,
            label=label,
            color=color,
            s=42,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("Assembly seed")
    ax.set_ylabel("Effect minus registered MCID")
    ax.legend(frameon=False, ncol=1, loc="best")
    _panel_label(ax, "a")

    ax = axes[0, 1]
    auc_margin = frame["reachability_auc"].to_numpy(float) - float(
        thresholds["min_reachability_auc"]
    )
    recall_margin = frame["reachability_recall"].to_numpy(float) - float(
        thresholds["min_reachability_recall"]
    )
    ax.scatter(
        auc_margin,
        recall_margin,
        c=np.where(frame["reachability_gate"].to_numpy(bool), GREEN, RED),
        s=55,
        edgecolor="white",
        linewidth=0.6,
    )
    for index, seed in enumerate(seeds):
        ax.annotate(seed, (auc_margin[index], recall_margin[index]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axvline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("Reachability AUC minus threshold")
    ax.set_ylabel("Switch recall minus threshold")
    _panel_label(ax, "b")

    ax = axes[1, 0]
    false_alarm_margin = float(thresholds["max_false_alarms_per_1000"]) - frame[
        "reachability_false_alarms_per_1000"
    ].to_numpy(float)
    delay_margin = float(thresholds["max_median_delay"]) - frame[
        "reachability_median_delay"
    ].to_numpy(float)
    ax.scatter(
        false_alarm_margin,
        delay_margin,
        c=np.where(frame["reachability_gate"].to_numpy(bool), GREEN, RED),
        s=55,
        edgecolor="white",
        linewidth=0.6,
    )
    for index, seed in enumerate(seeds):
        ax.annotate(seed, (false_alarm_margin[index], delay_margin[index]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axvline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("False-alarm allowance (per 1,000 frames)")
    ax.set_ylabel("Delay allowance (frames)")
    _panel_label(ax, "c")

    ax = axes[1, 1]
    gate_matrix = np.column_stack(
        [frame[field].astype(bool).to_numpy() for field, _ in QUALIFICATION_GATES]
    ).astype(int)
    ax.imshow(
        gate_matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=matplotlib.colors.ListedColormap([RED, GREEN]),
        interpolation="nearest",
    )
    ax.set_xticks(
        np.arange(len(QUALIFICATION_GATES)),
        [label for _, label in QUALIFICATION_GATES],
        rotation=22,
        ha="right",
    )
    ax.set_yticks(np.arange(len(frame)), seeds)
    ax.set_xlabel("Joint gate (red = fail, green = pass)")
    ax.set_ylabel("Assembly seed")
    external = bool(receipt.get("external_stage_authorized"))
    status = "EXTERNAL UNLOCKED" if external else "EXTERNAL LOCKED"
    ax.text(
        0.98,
        0.03,
        status,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GREEN if external else RED,
        fontweight="bold",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    _panel_label(ax, "d")

    figure.tight_layout(w_pad=2.0, h_pad=2.0)
    return figure


def _validate_external(
    condition_summary: pd.DataFrame, comparisons: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_required = {"panel", "condition", "video_equal_accuracy"}
    comparison_required = {
        "comparison",
        "mean_difference",
        "ci_low",
        "ci_high",
        "holm_adjusted_pvalue",
    }
    if not condition_required <= set(condition_summary.columns):
        raise ValueError("external condition summary is incomplete")
    if not comparison_required <= set(comparisons.columns):
        raise ValueError("external comparison summary is incomplete")
    expected_cells = {
        (panel, condition)
        for panel in ("natural", "hidden_switch")
        for condition in EXTERNAL_CONDITIONS
    }
    observed_cells = set(
        zip(condition_summary["panel"], condition_summary["condition"], strict=True)
    )
    if observed_cells != expected_cells or condition_summary.duplicated(
        ["panel", "condition"]
    ).any():
        raise ValueError("external condition panel does not match registered cells")
    if set(comparisons["comparison"]) != set(COMPARISON_LABELS):
        raise ValueError("external comparisons do not match registered family")
    for columns, label in (
        (["video_equal_accuracy"], "condition"),
        (["mean_difference", "ci_low", "ci_high", "holm_adjusted_pvalue"], "comparison"),
    ):
        table = condition_summary if label == "condition" else comparisons
        numeric = table[columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"external {label} plotting values must be finite")
    return condition_summary.copy(), comparisons.copy()


def make_external_figure(
    condition_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: Mapping[str, Any],
) -> plt.Figure:
    """Plot independent external outcomes after qualification unlock only."""

    means, effects = _validate_external(condition_summary, comparisons)
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.5))
    colors = [GRAY, PURPLE, ORANGE, "#56B4E9", GREEN, RED, "#999999", BLUE]
    x = np.arange(len(EXTERNAL_CONDITIONS), dtype=float)

    for ax, panel, label in (
        (axes[0, 0], "natural", "a"),
        (axes[0, 1], "hidden_switch", "b"),
    ):
        values = (
            means.loc[means["panel"] == panel]
            .set_index("condition")
            .loc[list(EXTERNAL_CONDITIONS), "video_equal_accuracy"]
            .to_numpy(float)
        )
        ax.bar(x, values, color=colors, width=0.74)
        ax.set_xticks(
            x,
            [CONDITION_LABELS[name] for name in EXTERNAL_CONDITIONS],
            rotation=28,
            ha="right",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Video-equal accuracy")
        ax.set_title("Natural videos" if panel == "natural" else "Hidden switches")
        _panel_label(ax, label)

    order = list(COMPARISON_LABELS)
    ordered = effects.set_index("comparison").loc[order]
    y = np.arange(len(order), dtype=float)[::-1]
    values = ordered["mean_difference"].to_numpy(float)
    lows = ordered["ci_low"].to_numpy(float)
    highs = ordered["ci_high"].to_numpy(float)
    ax = axes[1, 0]
    ax.errorbar(
        values,
        y,
        xerr=np.vstack((values - lows, highs - values)),
        fmt="o",
        color="#222222",
        ecolor=GRAY,
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.0, color=GRAY, linestyle="--", linewidth=1)
    ax.axvline(
        float(config["analysis"]["hidden_switch_mcid"]),
        color=ORANGE,
        linestyle=":",
        linewidth=1,
    )
    ax.set_yticks(y, [COMPARISON_LABELS[name] for name in order])
    ax.set_xlabel("Paired source-video accuracy difference")
    _panel_label(ax, "c")

    ax = axes[1, 1]
    pvalues = ordered["holm_adjusted_pvalue"].to_numpy(float)
    ax.barh(y, -np.log10(np.maximum(pvalues, np.finfo(float).tiny)), color=BLUE)
    ax.axvline(-np.log10(float(config["analysis"]["alpha"])), color=RED, linestyle="--", linewidth=1)
    ax.set_yticks(y, [COMPARISON_LABELS[name] for name in order])
    ax.set_xlabel(r"$-\log_{10}$(Holm-adjusted p)")
    _panel_label(ax, "d")

    figure.tight_layout(w_pad=2.2, h_pad=2.0)
    return figure


def save_all(figure: plt.Figure, prefix: Path) -> tuple[Path, ...]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        path = prefix.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qualification", "external"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage == "qualification":
        receipt = json.loads(args.input.read_text(encoding="utf-8"))
        figure = make_qualification_figure(receipt, config)
    else:
        figure = make_external_figure(
            pd.read_csv(args.input / "condition_summary.csv"),
            pd.read_csv(args.input / "comparisons.csv"),
            config,
        )
    paths = save_all(figure, args.output_prefix)
    plt.close(figure)
    print(json.dumps({"outputs": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
