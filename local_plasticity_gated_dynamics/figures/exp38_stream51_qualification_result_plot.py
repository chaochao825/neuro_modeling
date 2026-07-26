#!/usr/bin/env python3
"""Render the locked Exp38 qualification result without overlapping seed labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.exp38_stream51_soft_memory_plot import (
    make_qualification_figure,
    save_all,
)


def make_result_figure(receipt: dict[str, object], config: dict[str, object]) -> plt.Figure:
    """Reuse the frozen panels and apply display-only, result-agnostic cleanup."""

    figure = make_qualification_figure(receipt, config)
    for ax in (figure.axes[1], figure.axes[2]):
        for label in tuple(ax.texts):
            if label.get_text().isdigit():
                label.remove()
        ax.text(
            0.02,
            0.98,
            "Each point is one assembly seed",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )
    figure.axes[3].set_xlabel("Joint gate (orange = fail, green = pass)")
    figure.tight_layout(w_pad=2.0, h_pad=2.0)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    figure = make_result_figure(receipt, config)
    paths = save_all(figure, args.output_prefix)
    plt.close(figure)
    print(json.dumps({"outputs": [str(path.resolve()) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
