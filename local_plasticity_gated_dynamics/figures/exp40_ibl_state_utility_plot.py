"""Publication figure bound to the animal-level Exp40 summary artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.plot_style import COLORS, save_figure, setup_style  # noqa: E402


FIGURE_NAME = "exp40_ibl_state_utility"


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{column} needs at least two finite animal values")
    return values


def _mean_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(20_000, values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _strip_with_interval(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    x: float,
    color: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.10, 0.10, size=values.size)
    ax.scatter(
        np.full(values.size, x) + jitter,
        values,
        s=17,
        color=color,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    mean, low, high = _mean_interval(values, seed + 100)
    ax.errorbar(
        [x],
        [mean],
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        color="black",
        markerfacecolor="white",
        markersize=5,
        linewidth=1.4,
        capsize=3,
        zorder=5,
    )


def make_figure(effects: pd.DataFrame, output_root: Path) -> None:
    setup_style()
    complete = effects.loc[effects["endpoint_status"].astype(str).eq("complete")]
    context = _finite(complete, "context_nll_gain_hmm_minus_semimarkov")
    registered = _finite(complete, "primary_gain_selected_baseline_minus_factorized")
    probe = _finite(complete, "probe_gain_selected_baseline_minus_factorized")
    release = _finite(complete, "release_clamp_nll_harm")
    precision = _finite(complete, "concentration_clamp_nll_harm")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    ax = axes[0]
    _strip_with_interval(ax, context, x=0.0, color=COLORS[0], seed=40)
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.axhline(0.02, color="0.65", linewidth=0.8, linestyle=":")
    ax.set_xticks([0.0], ["Semi-Markov\nvs learned HMM"])
    ax.set_ylabel("Context NLL gain (nats/trial)")
    ax.text(-0.20, 1.02, "a", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    for first, second in zip(registered, probe, strict=True):
        ax.plot([0, 1], [first, second], color="0.78", linewidth=0.6, zorder=0)
    _strip_with_interval(ax, registered, x=0.0, color=COLORS[1], seed=41)
    _strip_with_interval(ax, probe, x=1.0, color=COLORS[2], seed=42)
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.axhline(0.005, color="0.65", linewidth=0.8, linestyle=":")
    ax.set_xticks([0.0, 1.0], ["Registered", "Assay probe"])
    ax.set_ylabel("Baseline − factorized NLL")
    ax.text(-0.20, 1.02, "b", transform=ax.transAxes, fontweight="bold")

    ax = axes[2]
    for first, second in zip(release, precision, strict=True):
        ax.plot([0, 1], [first, second], color="0.78", linewidth=0.6, zorder=0)
    _strip_with_interval(ax, release, x=0.0, color=COLORS[3], seed=43)
    _strip_with_interval(ax, precision, x=1.0, color=COLORS[4], seed=44)
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.axhline(0.002, color="0.65", linewidth=0.8, linestyle=":")
    ax.set_xticks([0.0, 1.0], ["Release\nclamp", "Precision\nclamp"])
    ax.set_ylabel("Clamp NLL harm")
    ax.text(-0.20, 1.02, "c", transform=ax.transAxes, fontweight="bold")

    for axis in axes:
        axis.tick_params(axis="x", length=0)
        axis.margins(x=0.25)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, FIGURE_NAME, output_root)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effects",
        type=Path,
        default=PROJECT_ROOT / "results" / "exp40_ibl_state_utility_animal_effects.csv",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    effects = pd.read_csv(args.effects)
    make_figure(effects, args.output_root)
    for suffix in ("pdf", "png"):
        path = args.output_root / f"{FIGURE_NAME}.{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"figure output is missing or empty: {path}")
        print(path)


if __name__ == "__main__":
    main()
