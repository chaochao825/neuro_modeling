"""Plot the hash-validated formal exp14 multi-session neural snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.plot_style import COLORS, save_figure, setup_style  # noqa: E402
from scripts.summarize_exp14 import (  # noqa: E402
    DEFAULT_PREFIX,
    load_validated_exp14_snapshot,
)

FAMILY_ORDER = ("common", "shared", "full")
PANEL_ORDER = (
    ("stimulus_pre", "primary_past_safe"),
    ("stimulus_pre", "full_trial_sensitivity"),
    ("movement_pre", "primary_past_safe"),
    ("movement_pre", "full_trial_sensitivity"),
)
PANEL_LABELS = (
    "Stim / primary\n(registered core)",
    "Stim / full\n(sensitivity)",
    "Move / primary\n(sensitivity)",
    "Move / full\n(sensitivity)",
)


def plot_exp14(results_root: Path, prefix: str = DEFAULT_PREFIX) -> plt.Figure:
    conditions, comparisons, _raw, _ = load_validated_exp14_snapshot(
        results_root, prefix=prefix
    )
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    ax_nll, ax_gain, ax_retention, ax_parameters = axes.ravel()
    family_colors = {"common": "#888888", "shared": COLORS[0], "full": COLORS[2]}
    x = np.arange(len(PANEL_ORDER))
    width = 0.23
    indexed = conditions.set_index(["view", "panel", "model_family"])
    for offset, family in zip((-width, 0.0, width), FAMILY_ORDER, strict=True):
        values = [
            float(indexed.loc[(*pair, family), "animal_mean_nll_per_count"])
            for pair in PANEL_ORDER
        ]
        ax_nll.bar(x + offset, values, width, label=family, color=family_colors[family])
    ax_nll.set_xticks(x, PANEL_LABELS)
    ax_nll.set_ylabel("Held-out NLL per count")
    ax_nll.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax_nll.text(-0.1, 1.03, "a", transform=ax_nll.transAxes, fontweight="bold")

    comparison = comparisons.set_index(["view", "panel"])
    gains = np.asarray(
        [comparison.loc[pair, "shared_vs_common_estimate"] for pair in PANEL_ORDER],
        dtype=float,
    )
    low = np.asarray(
        [comparison.loc[pair, "shared_vs_common_ci_low"] for pair in PANEL_ORDER],
        dtype=float,
    )
    high = np.asarray(
        [comparison.loc[pair, "shared_vs_common_ci_high"] for pair in PANEL_ORDER],
        dtype=float,
    )
    primary_conclusion = comparison.loc[PANEL_ORDER[0], "core_conclusion"]
    primary_color = (
        COLORS[2]
        if primary_conclusion == "support"
        else COLORS[1]
        if primary_conclusion == "oppose"
        else "#777777"
    )
    colors = [primary_color, "#aaaaaa", "#aaaaaa", "#aaaaaa"]
    ax_gain.axhline(0, color="black", ls="--", lw=0.8)
    for position, gain, ci_low, ci_high, color in zip(
        x, gains, low, high, colors, strict=True
    ):
        ax_gain.errorbar(
            position,
            gain,
            yerr=np.asarray([[gain - ci_low], [ci_high - gain]]),
            fmt="none",
            ecolor=color,
            capsize=3,
        )
    ax_gain.scatter(x, gains, c=colors, zorder=3)
    ax_gain.set_xticks(x, PANEL_LABELS)
    ax_gain.set_ylabel("Common - shared NLL/count (positive favors shared)")
    ax_gain.text(-0.1, 1.03, "b", transform=ax_gain.transAxes, fontweight="bold")

    ratios = np.asarray(
        [comparison.loc[pair, "retained_full_gain_ratio"] for pair in PANEL_ORDER],
        dtype=float,
    )
    finite = np.isfinite(ratios)
    ax_retention.axhline(
        0.9, color="black", ls="--", lw=0.9, label="Registered 90% margin"
    )
    ax_retention.bar(
        x[finite], ratios[finite], color=np.asarray(colors, dtype=object)[finite]
    )
    if not finite.any():
        ax_retention.text(
            0.28,
            0.5,
            "Retention undefined\n(full did not improve common)",
            transform=ax_retention.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
    ax_retention.set_xticks(x, PANEL_LABELS)
    ax_retention.set_ylabel("Shared/full gain ratio")
    ax_retention.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax_retention.text(
        -0.1, 1.03, "c", transform=ax_retention.transAxes, fontweight="bold"
    )

    shared = np.asarray(
        [comparison.loc[pair, "shared_parameter_count"] for pair in PANEL_ORDER],
        dtype=float,
    )
    full = np.asarray(
        [comparison.loc[pair, "full_parameter_count"] for pair in PANEL_ORDER],
        dtype=float,
    )
    ax_parameters.bar(x - width / 2, shared, width, color=COLORS[0], label="shared")
    ax_parameters.bar(x + width / 2, full, width, color=COLORS[2], label="full")
    ax_parameters.set_yscale("log")
    ax_parameters.set_xticks(x, PANEL_LABELS)
    ax_parameters.set_ylabel("Parameters (log scale)")
    ax_parameters.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax_parameters.text(
        -0.1, 1.03, "d", transform=ax_parameters.transAxes, fontweight="bold"
    )
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=12)
    fig.suptitle(
        "IBL shared-basis audit (only Stim / primary is registered core)", y=1.01
    )
    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    figure = plot_exp14(Path(args.results_root), args.prefix)
    save_figure(figure, args.prefix, Path(args.results_root))
    plt.close(figure)


if __name__ == "__main__":
    main()
