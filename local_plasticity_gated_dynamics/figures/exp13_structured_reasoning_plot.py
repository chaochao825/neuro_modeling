"""Plot a data-bound exp13 structured-task formal result panel."""

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
from src.analysis.structured_benchmark import STRUCTURED_CONDITIONS  # noqa: E402
from src.analysis.structured_formal import (  # noqa: E402
    load_validated_structured_snapshot,
)


DISPLAY = {
    "support_heuristic": "Support\nheuristic",
    "flat_local": "Flat\nlocal",
    "hierarchical_local": "Hier.\nlocal",
    "trace_local": "Trace\nlocal",
    "gru_bptt": "GRU\nBPTT",
    "candidate_oracle": "Candidate\noracle",
}


def _load(results_root: Path, prefix: str):
    conditions, comparisons, raw, _ = load_validated_structured_snapshot(
        results_root,
        prefix=prefix,
        require_published_root=prefix == "exp13_arc_formal",
    )
    return conditions.set_index("condition"), comparisons, raw


def plot_exp13(results_root: Path, prefix: str = "exp13_arc_formal") -> plt.Figure:
    conditions, comparisons, raw = _load(results_root, prefix)
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    ax_accuracy, ax_difference, ax_parameters, ax_seeds = axes.ravel()
    colors = [COLORS[4], "#999999", COLORS[0], COLORS[2], COLORS[1], "#666666"]
    positions = np.arange(len(conditions))

    values = conditions["exact_accuracy"].to_numpy(dtype=float) * 100.0
    lower = values - conditions["exact_accuracy_ci_low"].to_numpy(dtype=float) * 100.0
    upper = conditions["exact_accuracy_ci_high"].to_numpy(dtype=float) * 100.0 - values
    bars = ax_accuracy.bar(
        positions,
        values,
        color=colors,
        width=0.72,
        yerr=np.vstack([lower, upper]),
        capsize=2.5,
        linewidth=0.5,
        edgecolor="black",
    )
    bars[-1].set_hatch("//")
    ax_accuracy.set_xticks(positions, [DISPLAY[item] for item in conditions.index])
    ax_accuracy.set_ylabel("Exact task accuracy (%)")
    ax_accuracy.set_ylim(
        0.0,
        min(
            100.0,
            max(1.0, conditions["exact_accuracy_ci_high"].max() * 100.0 + 0.25),
        ),
    )
    ax_accuracy.text(
        -0.10, 1.03, "a", transform=ax_accuracy.transAxes, fontweight="bold"
    )

    comparison_labels = [
        "Hier. - flat",
        "Trace - flat",
        "Hier. - heuristic",
        "Hier. - GRU",
        "Hier. - 0.9 GRU",
        "Trace - hier.",
    ]
    y = np.arange(len(comparisons))[::-1]
    estimates = comparisons["estimate"].to_numpy(dtype=float) * 100.0
    low = comparisons["ci_low"].to_numpy(dtype=float) * 100.0
    high = comparisons["ci_high"].to_numpy(dtype=float) * 100.0
    conclusion_colors = {
        "support": COLORS[2],
        "oppose": COLORS[1],
        "inconclusive": "#777777",
    }
    for row_index, row in comparisons.reset_index(drop=True).iterrows():
        location = y[row_index]
        color = conclusion_colors[str(row["conclusion"])]
        ax_difference.plot(
            [low[row_index], high[row_index]], [location, location], color=color, lw=1.8
        )
        ax_difference.scatter(
            estimates[row_index], location, color=color, s=26, zorder=3
        )
    ax_difference.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_difference.set_yticks(y, comparison_labels)
    ax_difference.set_xlabel("Paired exact-accuracy contrast (percentage points)")
    ax_difference.text(
        -0.10, 1.03, "b", transform=ax_difference.transAxes, fontweight="bold"
    )

    total = conditions["parameter_count"].to_numpy(dtype=float)
    trainable = conditions["trainable_parameter_count"].to_numpy(dtype=float)
    width = 0.34
    ax_parameters.bar(
        positions - width / 2,
        np.maximum(total, 1.0),
        width=width,
        color="#777777",
        label="Total state/readout",
    )
    ax_parameters.bar(
        positions + width / 2,
        np.maximum(trainable, 1.0),
        width=width,
        color=COLORS[0],
        label="Trainable",
    )
    ax_parameters.set_yscale("log")
    ax_parameters.set_xticks(positions, [DISPLAY[item] for item in conditions.index])
    ax_parameters.set_ylabel("Parameters (log scale; zero shown at 1)")
    ax_parameters.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0
    )
    ax_parameters.text(
        -0.10, 1.03, "c", transform=ax_parameters.transAxes, fontweight="bold"
    )

    seed_condition = raw.groupby(["seed", "condition"], as_index=False)["exact"].mean()
    distributions = [
        seed_condition.loc[seed_condition["condition"] == condition, "exact"].to_numpy()
        * 100.0
        for condition in STRUCTURED_CONDITIONS
    ]
    box = ax_seeds.boxplot(
        distributions,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_seeds.set_xticks(positions, [DISPLAY[item] for item in conditions.index])
    ax_seeds.set_ylabel("Per-seed exact task accuracy (%)")
    maximum_seed_accuracy = max(
        max(values_in_seed, default=0.0) for values_in_seed in distributions
    )
    ax_seeds.set_ylim(0.0, max(1.0, maximum_seed_accuracy + 0.15))
    ax_seeds.text(-0.10, 1.03, "d", transform=ax_seeds.transAxes, fontweight="bold")

    for axis in axes.ravel():
        axis.tick_params(axis="x", pad=3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.subplots_adjust(left=0.11)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--prefix", default="exp13_arc_formal")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    figure = plot_exp13(results_root, args.prefix)
    save_figure(figure, args.prefix, results_root)
    plt.close(figure)


if __name__ == "__main__":
    main()
