#!/usr/bin/env python3
"""Render the outcome-bound Exp43 development figure.

The oracle panels are mechanism-localization diagnostics, not deployable
comparisons.  Cell-wise and parameter panels are explicitly descriptive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#777777"

LEARNED = "learned_event_learned_qr"
ORACLE_EVENT = "oracle_event_learned_qr"
ORACLE_QR = "learned_event_oracle_qr"
ORACLE_BOTH = "oracle_event_oracle_qr"
TOTAL_VARIANCE = "h_plus_total_variance"
SEEN_IMM = "generator_supported_seen_mode_imm"

ARM_ORDER = (LEARNED, ORACLE_EVENT, ORACLE_QR, ORACLE_BOTH)
ARM_LABELS = (
    "Learned event\nlearned Q/R",
    "Oracle event\nlearned Q/R",
    "Learned event\noracle Q/R",
    "Oracle event\noracle Q/R",
)
TEST_CELLS = ("011", "101", "110", "111")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=10,
    )


def _load(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_files = ("seed_metrics.csv", "block_metrics.csv", "summary.json")
    missing = [name for name in required_files if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Exp43 artifacts: {missing}")
    seed = pd.read_csv(result_dir / "seed_metrics.csv")
    block = pd.read_csv(result_dir / "block_metrics.csv", dtype={"cell": str})
    with (result_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    methods = set(seed["method"].astype(str))
    required_methods = set(ARM_ORDER) | {TOTAL_VARIANCE, SEEN_IMM}
    if not required_methods <= methods:
        raise RuntimeError(
            "Exp43 method panel is incomplete: "
            f"{sorted(required_methods - methods)}"
        )
    if set(block["cell"].astype(str)) != set(TEST_CELLS):
        raise RuntimeError("Exp43 test-cell panel is incomplete")
    if summary.get("development_only") is not True:
        raise RuntimeError("refusing to render an unlabelled Exp43 result")
    if summary.get("claim_upgrade_allowed") is not False:
        raise RuntimeError("Exp43 claim boundary is not fail-closed")
    return seed, block, summary


def _weighted_cell_metric(block: pd.DataFrame, metric: str) -> pd.DataFrame:
    needed = {"seed", "method", "cell", "block_length", metric}
    if not needed <= set(block.columns):
        raise RuntimeError(f"block_metrics.csv lacks {sorted(needed - set(block.columns))}")
    frame = block.loc[:, list(needed)].copy()
    frame["weighted"] = frame[metric] * frame["block_length"]
    grouped = frame.groupby(["seed", "method", "cell"], as_index=False).agg(
        weighted=("weighted", "sum"),
        samples=("block_length", "sum"),
    )
    grouped[metric] = grouped["weighted"] / grouped["samples"]
    return grouped.loc[:, ["seed", "method", "cell", metric]]


def cell_gains(block: pd.DataFrame, metric: str = "nll") -> pd.DataFrame:
    """Return paired per-seed, per-cell comparator-minus-learned gains."""

    aggregated = _weighted_cell_metric(block, metric)
    pivot = aggregated.pivot(
        index=["seed", "cell"], columns="method", values=metric
    )
    required = {LEARNED, TOTAL_VARIANCE, SEEN_IMM}
    if not required <= set(pivot.columns):
        raise RuntimeError("Exp43 cell-wise deployable methods are incomplete")
    output = pd.DataFrame(index=pivot.index)
    output["total_variance"] = pivot[TOTAL_VARIANCE] - pivot[LEARNED]
    output["seen_imm"] = pivot[SEEN_IMM] - pivot[LEARNED]
    return output.reset_index()


def learned_qr_by_cell(block: pd.DataFrame) -> pd.DataFrame:
    """Return length-weighted learned Q/R estimates by seed and test cell."""

    learned = block.loc[block["method"] == LEARNED].copy()
    q = _weighted_cell_metric(learned, "mean_q").rename(
        columns={"mean_q": "learned_q"}
    )
    r = _weighted_cell_metric(learned, "mean_r").rename(
        columns={"mean_r": "learned_r"}
    )
    merged = q.merge(r, on=["seed", "method", "cell"], validate="one_to_one")
    merged["true_q"] = merged["cell"].map(
        lambda cell: 0.04 if str(cell)[1] == "1" else 0.0025
    )
    merged["true_r"] = merged["cell"].map(
        lambda cell: 0.16 if str(cell)[2] == "1" else 0.01
    )
    return merged


def _headroom_values(
    summary: dict[str, Any], key: str
) -> tuple[np.ndarray, float]:
    headroom = summary.get("oracle_headroom", {}).get(key, {})
    values = headroom.get("values_by_seed")
    if not isinstance(values, dict) or len(values) != 8:
        raise RuntimeError(f"Exp43 oracle headroom is incomplete for {key}")
    ordered = np.asarray([float(values[seed]) for seed in sorted(values)], dtype=float)
    if not np.isclose(ordered.mean(), float(headroom["mean"]), atol=1e-12):
        raise RuntimeError(f"Exp43 oracle-headroom replay failed for {key}")
    return ordered, float(headroom["mean"])


def render(result_dir: Path, output_stem: Path) -> tuple[Path, Path, Path]:
    """Render PDF, SVG, and PNG from the immutable Exp43 result package."""

    seed, block, summary = _load(result_dir)
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.55), constrained_layout=True)

    # a: registered 2x2 causal exchange, paired by seed.
    ax = axes[0, 0]
    pivot = seed.pivot(index="seed", columns="method", values="overall_nll")
    x = np.arange(len(ARM_ORDER), dtype=float)
    for _, row in pivot.loc[:, list(ARM_ORDER)].iterrows():
        ax.plot(x, row.to_numpy(float), color=GREY, alpha=0.35, linewidth=0.65)
    means = pivot.loc[:, list(ARM_ORDER)].mean(axis=0).to_numpy(float)
    ax.plot(x, means, color="black", marker="o", markersize=3.5, linewidth=1.3)
    ax.set_xticks(x, ARM_LABELS)
    ax.set_ylabel("Held-out NLL (nats; lower is better)")
    _panel_label(ax, "a")

    # b: registered oracle-path headroom, with path-specific MCIDs.
    ax = axes[0, 1]
    headroom_specs = (
        ("event_8step_nll", "Event\n(next 8)", 0.02, ORANGE),
        ("qr_overall_nll", "Q/R\n(overall)", 0.02, BLUE),
        ("both_nll", "Both\n(overall)", 0.03, GREEN),
    )
    for index, (key, _, threshold, color) in enumerate(headroom_specs):
        values, mean = _headroom_values(summary, key)
        offsets = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=13,
            color=color,
            alpha=0.58,
            edgecolors="none",
            zorder=2,
        )
        ax.plot(index, mean, "o", color="black", markersize=3.8, zorder=3)
        ax.plot(
            [index - 0.18, index + 0.18],
            [threshold, threshold],
            color=VERMILLION,
            linewidth=1.2,
            zorder=3,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xticks(
        range(3), [spec[1] for spec in headroom_specs]
    )
    ax.set_ylabel("Learned minus oracle NLL (nats)")
    ax.plot(
        [],
        [],
        color=VERMILLION,
        linewidth=1.2,
        label="Registered MCID",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.11, 1.0), handlelength=1.8)
    _panel_label(ax, "b")

    # c: descriptive cell-wise deployable comparison.
    ax = axes[1, 0]
    gains = cell_gains(block)
    comparator_specs = (
        ("total_variance", "Total variance", BLUE, -0.11),
        ("seen_imm", "Seen IMM", ORANGE, 0.11),
    )
    for column, label, color, shift in comparator_specs:
        for index, cell in enumerate(TEST_CELLS):
            values = gains.loc[gains["cell"] == cell, column].to_numpy(float)
            offsets = np.linspace(-0.045, 0.045, len(values))
            ax.scatter(
                index + shift + offsets,
                values,
                s=10,
                color=color,
                alpha=0.48,
                edgecolors="none",
            )
            ax.plot(index + shift, values.mean(), "o", color=color, markersize=4)
        ax.plot([], [], "o", color=color, label=label, markersize=4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(4), TEST_CELLS)
    ax.set_xlabel("Unseen uncertainty cell (h, Q, R)")
    ax.set_ylabel("Comparator minus learned NLL (nats)")
    ax.legend(loc="upper left", ncols=2, handletextpad=0.3, columnspacing=0.7)
    _panel_label(ax, "c")

    # d: descriptive calibration audit; open symbols are generator truth.
    ax = axes[1, 1]
    qr = learned_qr_by_cell(block)
    positions = np.arange(len(TEST_CELLS), dtype=float)
    for column, label, color, marker, shift in (
        ("learned_q", "Learned Q", BLUE, "o", -0.06),
        ("learned_r", "Learned R", ORANGE, "s", 0.06),
    ):
        means = np.asarray(
            [qr.loc[qr["cell"] == cell, column].mean() for cell in TEST_CELLS]
        )
        ax.plot(
            positions + shift,
            means,
            marker=marker,
            color=color,
            linewidth=1.0,
            markersize=4,
            label=label,
        )
    for column, label, color, marker, shift in (
        ("true_q", "True Q", BLUE, "o", -0.06),
        ("true_r", "True R", ORANGE, "s", 0.06),
    ):
        values = np.asarray(
            [qr.loc[qr["cell"] == cell, column].iloc[0] for cell in TEST_CELLS]
        )
        ax.scatter(
            positions + shift,
            values,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            s=27,
            label=label,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xticks(positions, TEST_CELLS)
    ax.set_xlabel("Unseen uncertainty cell (h, Q, R)")
    ax.set_ylabel("Variance estimate (log scale)")
    ax.legend(loc="lower right", ncols=2, handletextpad=0.3, columnspacing=0.7)
    _panel_label(ax, "d")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = tuple(
        output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")
    )
    for path in outputs:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return outputs  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem is not None
        else result_dir.parent / "exp43_fast_slow_causal_decomposition_development"
    )
    for path in render(result_dir, output_stem):
        print(path)


if __name__ == "__main__":
    main()
