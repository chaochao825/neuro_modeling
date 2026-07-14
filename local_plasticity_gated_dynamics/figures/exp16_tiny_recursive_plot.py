"""Generate the scoped Exp16 seed-level comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_exp16(
    results_root: Path,
    *,
    prefix: str = "exp16_tiny_recursive_smoke",
) -> dict[str, Path]:
    from scripts.summarize_exp16_tiny_recursive import (
        latest_attempt_metrics,
        load_published_snapshot,
    )

    raw, conditions, comparison, manifest = load_published_snapshot(
        results_root, prefix=prefix
    )
    selected_raw = latest_attempt_metrics(raw, manifest)
    aggregates = selected_raw.loc[selected_raw["stage"].eq("aggregate")].copy()
    labels = {
        "micro_trm_bptt": "micro-TRM-like",
        "single_state_core_call_matched": "single-state matched",
    }
    ordered = [name for name in labels if name in set(conditions["condition"])]
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    x = np.arange(len(ordered))
    blank_available = (
        bool(ordered)
        and "mean_blank_cell_accuracy" in conditions
        and "blank_cell_accuracy" in aggregates
    )
    blank_means = (
        [
            float(
                conditions.loc[
                    conditions["condition"].eq(name), "mean_blank_cell_accuracy"
                ].iloc[0]
            )
            for name in ordered
        ]
        if blank_available
        else []
    )
    blank_observed = (
        aggregates["blank_cell_accuracy"].dropna().to_numpy(float)
        if blank_available
        else np.asarray([], dtype=float)
    )
    blank_available = bool(
        blank_available
        and len(blank_observed)
        and np.isfinite(blank_means).all()
        and np.isfinite(blank_observed).all()
    )
    if blank_available:
        axes[0].bar(x, blank_means, color=["#3569b7", "#9a9a9a"], width=0.62)
        for index, name in enumerate(ordered):
            values = aggregates.loc[
                aggregates["condition"].eq(name), "blank_cell_accuracy"
            ].to_numpy(float)
            axes[0].scatter(
                np.full(len(values), index), values, color="black", s=18, zorder=3
            )
        axes[0].set_xticks(
            x, [labels[name] for name in ordered], rotation=15, ha="right"
        )
        axes[0].axhline(
            1.0 / 9.0, color="#b24a3a", linestyle="--", linewidth=1.0
        )
        axes[0].set_ylabel("Held-out blank-cell accuracy")
        upper_values = [*blank_means, *blank_observed.tolist()]
        if "blank_seed_bootstrap_ci_high" in conditions:
            upper_values.extend(
                conditions["blank_seed_bootstrap_ci_high"].dropna().tolist()
            )
        upper = min(1.0, max(0.25, 1.15 * max(upper_values)))
        axes[0].set_ylim(0.0, upper)
        axes[0].set_title("Continuous endpoint")
    else:
        axes[0].text(
            0.5,
            0.5,
            "Blank-cell endpoint unavailable\nfor this snapshot",
            ha="center",
            va="center",
        )
        axes[0].set_axis_off()

    exact_available = (
        bool(ordered)
        and "mean_exact_accuracy" in conditions
        and "exact_accuracy" in aggregates
    )
    exact_means = (
        [
            float(
                conditions.loc[
                    conditions["condition"].eq(name), "mean_exact_accuracy"
                ].iloc[0]
            )
            for name in ordered
        ]
        if exact_available
        else []
    )
    exact_available = bool(
        exact_available
        and len(aggregates)
        and np.isfinite(exact_means).all()
        and np.isfinite(aggregates["exact_accuracy"].to_numpy(float)).all()
    )
    if exact_available:
        axes[1].bar(x, exact_means, color=["#3569b7", "#9a9a9a"], width=0.62)
        for index, name in enumerate(ordered):
            values = aggregates.loc[
                aggregates["condition"].eq(name), "exact_accuracy"
            ].to_numpy(float)
            axes[1].scatter(
                np.full(len(values), index), values, color="black", s=18, zorder=3
            )
        axes[1].set_xticks(
            x, [labels[name] for name in ordered], rotation=15, ha="right"
        )
        axes[1].set_ylabel("Held-out exact accuracy")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Board-level endpoint")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Exact endpoint unavailable\nfor this snapshot",
            ha="center",
            va="center",
        )
        axes[1].set_axis_off()

    if comparison.empty:
        axes[2].text(0.5, 0.5, "No paired comparison", ha="center", va="center")
        axes[2].set_axis_off()
    elif {
        "blank_accuracy_estimate",
        "blank_seed_bootstrap_ci_low",
        "blank_seed_bootstrap_ci_high",
    }.issubset(comparison) and np.isfinite(
        comparison.loc[
            comparison.index[0],
            [
                "blank_accuracy_estimate",
                "blank_seed_bootstrap_ci_low",
                "blank_seed_bootstrap_ci_high",
            ],
        ].to_numpy(float)
    ).all():
        row = comparison.iloc[0]
        estimate = float(row["blank_accuracy_estimate"])
        low = float(row["blank_seed_bootstrap_ci_low"])
        high = float(row["blank_seed_bootstrap_ci_high"])
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].errorbar(
            [0],
            [estimate],
            yerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color="#3569b7",
            capsize=4,
        )
        axes[2].set_xlim(-0.7, 0.7)
        axes[2].set_xticks([0], ["micro two-state - single-state"])
        axes[2].set_ylabel("Paired blank-accuracy difference")
        axes[2].set_title(f"{row['conclusion']} ({int(row['n_complete_seeds'])} seeds)")
    else:
        axes[2].text(
            0.5,
            0.5,
            "Paired blank endpoint unavailable\nfor this snapshot",
            ha="center",
            va="center",
        )
        axes[2].set_axis_off()
    figure.suptitle("Exp16 baseline-only recursive reasoning audit")
    figure.tight_layout()
    png = results_root / f"{prefix}.png"
    pdf = results_root / f"{prefix}.pdf"
    existing = [path for path in (png, pdf) if path.exists()]
    if existing:
        raise FileExistsError(
            "Exp16 figures are immutable; choose a new prefix: "
            + ", ".join(str(path) for path in existing)
        )
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {"figure_png": png, "figure_pdf": pdf}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="exp16_tiny_recursive_smoke")
    args = parser.parse_args()
    plot_exp16(Path(args.results_root), prefix=args.prefix)


if __name__ == "__main__":
    main()
