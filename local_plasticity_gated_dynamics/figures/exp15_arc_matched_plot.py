"""Plot the verified-source Exp15 ARC matched-compute snapshot."""

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
from scripts.summarize_exp15_arc_matched import (  # noqa: E402
    ARC_CONDITIONS,
    PREFIX,
    load_published_snapshot,
)


LABELS = ("Slow/fast belief", "Flat matched")


def plot_exp15_arc_matched(results_root: Path) -> plt.Figure:
    conditions, comparison, _manifest = load_published_snapshot(results_root)
    indexed = conditions.set_index("condition").loc[list(ARC_CONDITIONS)]
    comp = comparison.iloc[0]
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    ax_accuracy, ax_coverage, ax_difference, ax_compute = axes.ravel()

    x = np.arange(len(ARC_CONDITIONS))
    accuracy = 100.0 * indexed["exact_accuracy"].to_numpy(dtype=float)
    low = 100.0 * indexed["exact_accuracy_ci_low"].to_numpy(dtype=float)
    high = 100.0 * indexed["exact_accuracy_ci_high"].to_numpy(dtype=float)
    ax_accuracy.bar(x, accuracy, color=(COLORS[0], "#777777"), width=0.62)
    ax_accuracy.errorbar(
        x,
        accuracy,
        yerr=np.vstack([accuracy - low, high - accuracy]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    ax_accuracy.set_xticks(x, LABELS)
    ax_accuracy.set_ylabel("Exact tasks (%)")
    ax_accuracy.set_title("Held-out ARC-AGI-1")
    ax_accuracy.text(
        -0.12, 1.04, "a", transform=ax_accuracy.transAxes, fontweight="bold"
    )

    coverage = 100.0 * float(comp["candidate_coverage"])
    threshold = 100.0 * float(comp["minimum_candidate_coverage"])
    ax_coverage.bar([0], [coverage], color=COLORS[4], width=0.6)
    ax_coverage.axhline(
        threshold, color=COLORS[1], linestyle="--", label="Registered gate"
    )
    ax_coverage.set_xticks([0], ["Finite proposal\nlibrary"])
    ax_coverage.set_ylabel("Candidate coverage (%)")
    ax_coverage.set_ylim(0.0, 105.0)
    ax_coverage.legend(loc="upper right")
    ax_coverage.text(
        -0.12, 1.04, "b", transform=ax_coverage.transAxes, fontweight="bold"
    )

    estimate = 100.0 * float(comp["estimate"])
    ci_low = 100.0 * float(comp["ci_low"])
    ci_high = 100.0 * float(comp["ci_high"])
    color = COLORS[2] if comp["claim_conclusion"] == "support" else "#777777"
    ax_difference.errorbar(
        [estimate],
        [0],
        xerr=np.asarray([[estimate - ci_low], [ci_high - estimate]]),
        fmt="o",
        color=color,
        ecolor=color,
        capsize=4,
    )
    ax_difference.axvline(0.0, color="black", linewidth=0.8)
    ax_difference.set_yticks([0], ["Slow/fast − flat"])
    ax_difference.set_xlabel("Exact-accuracy difference (percentage points)")
    ax_difference.set_title(f"Conclusion: {comp['claim_conclusion']}")
    ax_difference.text(
        -0.12, 1.04, "c", transform=ax_difference.transAxes, fontweight="bold"
    )

    measured = indexed["mean_measured_compute_units"].to_numpy(dtype=float)
    charged = indexed["mean_charged_compute_units"].to_numpy(dtype=float)
    width = 0.34
    ax_compute.bar(x - width / 2, measured, width, color=COLORS[5], label="Measured")
    ax_compute.bar(x + width / 2, charged, width, color=COLORS[3], label="Charged")
    ax_compute.set_xticks(x, LABELS)
    ax_compute.set_ylabel("Abstract compute proxy (a.u.)")
    ax_compute.set_yscale("log")
    ax_compute.legend(ncol=2, loc="upper center")
    ax_compute.text(-0.12, 1.04, "d", transform=ax_compute.transAxes, fontweight="bold")

    fig.suptitle("Task specialization does not overcome proposal coverage", y=1.01)
    fig.tight_layout()
    save_figure(fig, PREFIX, results_root)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    plot_exp15_arc_matched(Path(args.results_root))


if __name__ == "__main__":
    main()
