"""Generate the seed-level Exp27 actuator-selector evidence figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_COLUMNS = {
    "seed",
    "routing_utility",
    "gain_utility",
    "low_rank_utility",
    "fixed_best_utility",
    "oracle_utility",
    "gru_bptt_utility",
    "local_three_factor_utility",
    "local_minus_fixed_best",
    "oracle_minus_fixed_best",
    "local_selection_accuracy",
    "gru_selection_accuracy",
}

COLORS = {
    "fixed": "#6B7280",
    "local": "#0072B2",
    "gru": "#D55E00",
    "oracle": "#009E73",
    "routing": "#56B4E9",
    "gain": "#E69F00",
    "low_rank": "#CC79A7",
}


def _validated_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"seed endpoint table lacks columns: {sorted(missing)}")
    if frame.shape[0] < 2:
        raise ValueError("at least two independent seeds are required")
    values = frame.copy().sort_values("seed", kind="mergesort").reset_index(drop=True)
    if values["seed"].duplicated().any():
        raise ValueError("seed endpoint table must contain one row per seed")
    numeric = values[list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("seed endpoint table contains non-finite values")
    return values


def _seed_strip(
    axis: plt.Axes,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
) -> None:
    for position, (column, color) in enumerate(zip(columns, colors, strict=True)):
        values = frame[column].to_numpy(dtype=np.float64)
        offsets = np.linspace(-0.10, 0.10, values.size)
        axis.scatter(
            np.full(values.size, position) + offsets,
            values,
            color=color,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.35,
            s=22,
            zorder=2,
        )
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        axis.errorbar(
            position,
            mean,
            yerr=1.96 * standard_error,
            color="black",
            marker="D",
            markersize=5,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )
    axis.set_xticks(range(len(labels)), labels)


def plot_selector_evidence(
    seed_endpoints: pd.DataFrame,
    output_prefix: str | Path,
    *,
    dpi: int = 220,
    title: str | None = None,
    contrast_title: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write PNG/PDF/SVG evidence views from one row per outer seed."""

    frame = _validated_endpoints(seed_endpoints)
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValueError("title must be a non-empty string when provided")
    if contrast_title is not None and (
        not isinstance(contrast_title, str) or not contrast_title.strip()
    ):
        raise ValueError(
            "contrast_title must be a non-empty string when provided"
        )
    output = Path(output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)

    _seed_strip(
        axes[0, 0],
        frame,
        (
            "fixed_best_utility",
            "local_three_factor_utility",
            "gru_bptt_utility",
            "oracle_utility",
        ),
        ("Fixed best", "Local 3-factor", "GRU-BPTT", "Oracle"),
        (
            COLORS["fixed"],
            COLORS["local"],
            COLORS["gru"],
            COLORS["oracle"],
        ),
    )
    axes[0, 0].set_ylabel("Held-out balanced accuracy")
    axes[0, 0].set_title("A  Strict-unseen utility by outer seed", loc="left")

    oracle_gain = frame["oracle_minus_fixed_best"].to_numpy(dtype=np.float64)
    local_gain = frame["local_minus_fixed_best"].to_numpy(dtype=np.float64)
    maximum = max(float(np.max(oracle_gain)), float(np.max(local_gain)), 0.01)
    minimum = min(float(np.min(oracle_gain)), float(np.min(local_gain)), 0.0)
    span = maximum - minimum
    grid = np.linspace(minimum - 0.05 * span, maximum + 0.05 * span, 100)
    axes[0, 1].scatter(
        oracle_gain,
        local_gain,
        color=COLORS["local"],
        edgecolor="white",
        linewidth=0.5,
        s=32,
        alpha=0.8,
    )
    axes[0, 1].plot(grid, 0.8 * grid, "--", color="#333333", label="80% oracle gain")
    axes[0, 1].plot(grid, grid, ":", color=COLORS["oracle"], label="Oracle ceiling")
    axes[0, 1].axhline(0.0, color="#AAAAAA", linewidth=0.8)
    axes[0, 1].set_xlabel("Oracle − fixed-best gain")
    axes[0, 1].set_ylabel("Local − fixed-best gain")
    axes[0, 1].set_title(
        contrast_title or "B  Registered non-inferiority contrast", loc="left"
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    local_accuracy = frame["local_selection_accuracy"].to_numpy(dtype=np.float64)
    gru_accuracy = frame["gru_selection_accuracy"].to_numpy(dtype=np.float64)
    for local_value, gru_value in zip(local_accuracy, gru_accuracy, strict=True):
        axes[1, 0].plot(
            (0, 1),
            (local_value, gru_value),
            color="#BBBBBB",
            linewidth=0.7,
            alpha=0.65,
            zorder=1,
        )
    axes[1, 0].scatter(
        np.zeros(frame.shape[0]), local_accuracy, color=COLORS["local"], s=24, zorder=2
    )
    axes[1, 0].scatter(
        np.ones(frame.shape[0]), gru_accuracy, color=COLORS["gru"], s=24, zorder=2
    )
    axes[1, 0].set_xticks((0, 1), ("Local 3-factor", "GRU-BPTT"))
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].set_ylabel("Oracle-family selection accuracy")
    axes[1, 0].set_title("C  Family-selection accuracy", loc="left")

    family_differences = pd.DataFrame(
        {
            "Routing": frame["local_three_factor_utility"] - frame["routing_utility"],
            "Gain": frame["local_three_factor_utility"] - frame["gain_utility"],
            "Low-rank": frame["local_three_factor_utility"] - frame["low_rank_utility"],
        }
    )
    _seed_strip(
        axes[1, 1],
        family_differences,
        ("Routing", "Gain", "Low-rank"),
        ("Routing", "Gain", "Low-rank"),
        (COLORS["routing"], COLORS["gain"], COLORS["low_rank"]),
    )
    axes[1, 1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    axes[1, 1].set_ylabel("Local selector − fixed-family utility")
    axes[1, 1].set_title("D  Fixed-family comparisons", loc="left")

    profile = "formal" if frame.shape[0] == 30 else "development"
    default_title = (
        f"Exp27 frozen-family actuator selector "
        f"({profile}; n={frame.shape[0]} seeds)"
    )
    figure.suptitle(
        title or default_title,
        fontsize=12,
        fontweight="bold",
    )
    outputs = tuple(output.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))
    for path in outputs:
        save_options = {"dpi": dpi} if path.suffix == ".png" else {}
        figure.savefig(path, bbox_inches="tight", **save_options)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_endpoints", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    frame = pd.read_csv(args.seed_endpoints)
    for path in plot_selector_evidence(frame, args.output_prefix, dpi=args.dpi):
        print(path)


if __name__ == "__main__":
    main()
