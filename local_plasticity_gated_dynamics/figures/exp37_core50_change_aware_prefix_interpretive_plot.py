#!/usr/bin/env python3
"""Post-outcome interpretive Exp37 figure with a detector-scale diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.exp37_core50_change_aware_prefix_plot import make_figure


def make_interpretive_figure(
    sessions: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostic: Mapping[str, Any],
) -> Figure:
    """Replace the undefined-delay panel with a labeled post-hoc scale audit."""

    if diagnostic.get("analysis_status") != "post_hoc_failure_diagnostic_only":
        raise ValueError("Exp37 interpretive figure requires the post-hoc diagnostic")
    if diagnostic.get("claim_upgrade_allowed") is not False:
        raise ValueError("post-hoc diagnostic cannot upgrade the claim")
    maximum = float(diagnostic["true_switch_frames"]["maximum"])
    threshold = float(diagnostic["frozen_minimum_alarm_threshold"])
    if not 0.0 <= maximum < threshold <= 1.0:
        raise ValueError("diagnostic does not show a frozen-threshold scale gap")

    figure = make_figure(sessions, comparisons)
    axis = figure.axes[2]
    axis.clear()
    axis.barh(
        [1, 0],
        [threshold, maximum],
        color=["#D55E00", "#0072B2"],
        height=0.5,
    )
    axis.set_yticks([1, 0], ["Minimum frozen threshold", "Max score at true switch"])
    axis.set_xlim(0.0, max(0.22, 1.1 * threshold))
    axis.set_xlabel("Change posterior probability")
    axis.set_title("Post-hoc s2 scale diagnostic", fontsize=9, loc="left")
    for y_value, value in ((1, threshold), (0, maximum)):
        axis.text(
            value + 0.004,
            y_value,
            f"{value:.4f}",
            va="center",
            fontsize=8,
        )
    axis.text(-0.12, 1.04, "c", transform=axis.transAxes, fontweight="bold", fontsize=11)
    figure.tight_layout(w_pad=2.0, h_pad=2.0)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    root = Path(args.summary_dir).expanduser().resolve()
    sessions = pd.read_csv(root / "session_panel.csv")
    comparisons = pd.read_csv(root / "primary_session_comparisons.csv")
    diagnostic = json.loads(
        (root / "post_hoc_detector_scale_diagnostic.json").read_text(encoding="utf-8")
    )
    figure = make_interpretive_figure(sessions, comparisons, diagnostic)
    prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else root / "exp37_core50_change_aware_prefix_interpretive"
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
