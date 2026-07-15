"""Plot pass@1/pass@2 and shape diagnostics for an Exp18 snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_exp18(
    results_root: str | Path, *, prefix: str = "exp18_arc_recursive_canary"
) -> dict[str, Path]:
    root = Path(results_root)
    conditions = pd.read_csv(root / f"{prefix}_conditions.csv")
    if conditions.empty:
        raise ValueError("Exp18 condition summary is empty")
    required = {
        "condition",
        "pass_at_1_mean",
        "pass_at_2_mean",
        "shape_exact_fraction_mean",
    }
    if not required.issubset(conditions):
        raise ValueError("Exp18 condition summary lacks required columns")
    aggregate = conditions.groupby("condition", sort=True)[
        ["pass_at_1_mean", "pass_at_2_mean", "shape_exact_fraction_mean"]
    ].mean()
    labels = [value.replace("_", "\n") for value in aggregate.index]
    x = np.arange(len(aggregate))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10.4, 4.6))
    axis.bar(x - width, aggregate["pass_at_1_mean"], width, label="pass@1")
    axis.bar(x, aggregate["pass_at_2_mean"], width, label="pass@2")
    axis.bar(
        x + width,
        aggregate["shape_exact_fraction_mean"],
        width,
        label="shape exact",
    )
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Task-macro score")
    axis.set_title("Exp18 direct ARC generation (demo-only TTA)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    outputs = {
        "png": root / f"{prefix}.png",
        "pdf": root / f"{prefix}.pdf",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Exp18 figures are immutable; choose another prefix: "
            + ", ".join(str(path) for path in existing)
        )
    figure.savefig(outputs["png"], dpi=220, bbox_inches="tight")
    figure.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="exp18_arc_recursive_canary")
    args = parser.parse_args()
    outputs = plot_exp18(args.results_root, prefix=args.prefix)
    print("\n".join(f"{name}: {path}" for name, path in outputs.items()))


if __name__ == "__main__":
    main()

