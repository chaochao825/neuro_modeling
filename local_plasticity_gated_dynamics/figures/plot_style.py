"""Publication-style defaults shared by data-bound figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


DPI = 300
FONT_SIZE = 10
FIG_DIR = Path(__file__).resolve().parents[1] / "results"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LINE_STYLES = ("-", "--", "-.", ":")


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig: plt.Figure, name: str, results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        results_root / f"{name}.pdf",
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        results_root / f"{name}.png", format="png", dpi=DPI, bbox_inches="tight"
    )
