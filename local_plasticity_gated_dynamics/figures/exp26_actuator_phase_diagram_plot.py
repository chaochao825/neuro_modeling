"""Publication plot for the Exp26 held-out actuator phase diagram.

The discovery split is represented only by thresholds already present in the
seed-endpoint table.  In particular, this module never estimates a trend from
held-out generator outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FONT_SIZE = 10
DPI = 300
FIGURE_STEM = "exp26_actuator_phase_diagram"
MODE_ORDER = ("frozen", "routing", "gain", "low_rank", "rgl")
MODE_LABELS = {
    "frozen": "Frozen",
    "routing": "Routing",
    "gain": "Gain",
    "low_rank": "Low-rank",
    "rgl": "RGL\n(ceiling)",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#8A8A8A",
    "light_gray": "#D0D0D0",
}

METRIC_COLUMNS = {
    "seed",
    "generator_id",
    "generator_split",
    "actuator_mode",
    "chi",
    "alpha",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "test_balanced_accuracy",
    "functional_budget_valid",
    "status",
}
ENDPOINT_COLUMNS = {
    "seed",
    "spearman_rho",
    "classifier_balanced_accuracy",
    "classifier_auroc",
    "chi_minus_alpha_auroc",
    "discovery_threshold",
}
NUMERIC_METRIC_COLUMNS = (
    "chi",
    "alpha",
    "transition_rank",
    "input_rank",
    "delay",
    "noise_std",
    "test_balanced_accuracy",
)
NUMERIC_ENDPOINT_COLUMNS = (
    "spearman_rho",
    "classifier_balanced_accuracy",
    "classifier_auroc",
    "chi_minus_alpha_auroc",
    "discovery_threshold",
)


def _setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _require_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: set[str],
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} is empty; refusing to draw an unbound figure")
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _boolean_values(series: pd.Series) -> pd.Series:
    """Resolve audit flags without treating the string ``'False'`` as true."""

    resolved: list[bool] = []
    for value in series.tolist():
        if pd.isna(value):
            resolved.append(False)
        elif isinstance(value, (bool, np.bool_)):
            resolved.append(bool(value))
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            resolved.append(bool(value))
        elif isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
            resolved.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {
            "true",
            "false",
            "1",
            "0",
        }:
            resolved.append(value.strip().lower() in {"true", "1"})
        else:
            raise ValueError(f"invalid functional_budget_valid value: {value!r}")
    return pd.Series(resolved, index=series.index, dtype=bool)


def _validated_inputs(
    metrics: pd.DataFrame,
    seed_endpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_frame(metrics, name="metrics", columns=METRIC_COLUMNS)
    _require_frame(
        seed_endpoints,
        name="seed_endpoints",
        columns=ENDPOINT_COLUMNS,
    )
    clean_metrics = metrics.copy()
    clean_endpoints = seed_endpoints.copy()
    for column in NUMERIC_METRIC_COLUMNS:
        clean_metrics[column] = pd.to_numeric(clean_metrics[column], errors="raise")
    for column in NUMERIC_ENDPOINT_COLUMNS:
        clean_endpoints[column] = pd.to_numeric(
            clean_endpoints[column], errors="raise"
        )
    clean_metrics["seed"] = pd.to_numeric(
        clean_metrics["seed"], errors="raise", downcast="integer"
    )
    clean_endpoints["seed"] = pd.to_numeric(
        clean_endpoints["seed"], errors="raise", downcast="integer"
    )
    for name, frame in (
        ("metrics", clean_metrics),
        ("seed_endpoints", clean_endpoints),
    ):
        seed_values = frame["seed"].to_numpy(dtype=float)
        if not np.all(np.isfinite(seed_values)) or not np.allclose(
            seed_values, np.round(seed_values)
        ):
            raise ValueError(f"{name} seeds must be finite integers")
        frame["seed"] = seed_values.astype(np.int64)
    if clean_endpoints["seed"].duplicated().any():
        raise ValueError("seed_endpoints must contain exactly one row per seed")
    metric_seeds = set(clean_metrics["seed"].astype(int))
    endpoint_seeds = set(clean_endpoints["seed"].astype(int))
    if not metric_seeds or metric_seeds != endpoint_seeds:
        raise ValueError("metrics and seed_endpoints must contain the same seeds")
    if not np.all(np.isfinite(clean_endpoints["discovery_threshold"])):
        raise ValueError("discovery_threshold must be finite for every seed")
    clean_metrics["functional_budget_valid"] = _boolean_values(
        clean_metrics["functional_budget_valid"]
    )
    clean_metrics["generator_split"] = (
        clean_metrics["generator_split"].astype(str).str.strip().str.lower()
    )
    clean_metrics["actuator_mode"] = (
        clean_metrics["actuator_mode"].astype(str).str.strip().str.lower()
    )
    clean_metrics["status"] = clean_metrics["status"].astype(str).str.strip().str.lower()
    if not clean_metrics["generator_split"].eq("heldout").any():
        raise ValueError("metrics contain no heldout generator rows")
    return clean_metrics, clean_endpoints.sort_values("seed").reset_index(drop=True)


def heldout_advantage(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return paired held-out low-rank advantage; RGL is deliberately excluded."""

    eligible = metrics.loc[
        metrics["generator_split"].eq("heldout")
        & metrics["status"].eq("complete")
        & metrics["functional_budget_valid"]
        & metrics["actuator_mode"].isin(("routing", "gain", "low_rank"))
    ].copy()
    finite_columns = [
        "chi",
        "alpha",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
        "test_balanced_accuracy",
    ]
    if not eligible.empty and not np.all(
        np.isfinite(eligible[finite_columns].to_numpy(dtype=float))
    ):
        raise ValueError("complete, budget-valid heldout rows must be finite")
    pair_index = ["seed", "generator_id", "generator_split"]
    if eligible.duplicated(pair_index + ["actuator_mode"]).any():
        raise ValueError("duplicate heldout actuator rows prevent strict pairing")
    if eligible.empty:
        return pd.DataFrame(
            columns=pair_index
            + [
                "routing",
                "gain",
                "low_rank",
                "chi",
                "alpha",
                "transition_rank",
                "input_rank",
                "delay",
                "noise_std",
                "advantage",
            ]
        )
    metadata = (
        eligible.groupby(pair_index, sort=False)[
            [
                "chi",
                "alpha",
                "transition_rank",
                "input_rank",
                "delay",
                "noise_std",
            ]
        ]
        .agg(["first", "nunique"])
    )
    if bool((metadata.xs("nunique", axis=1, level=1) != 1).any(axis=None)):
        raise ValueError("task metadata differ across paired actuator modes")
    pivot = eligible.pivot(
        index=pair_index,
        columns="actuator_mode",
        values="test_balanced_accuracy",
    )
    required_modes = ["routing", "gain", "low_rank"]
    for mode in required_modes:
        if mode not in pivot:
            pivot[mode] = np.nan
    pivot = pivot.dropna(subset=required_modes)
    result = pivot.reset_index()
    first = metadata.xs("first", axis=1, level=1).reindex(pivot.index)
    for column in first.columns:
        result[column] = first[column].to_numpy()
    result["advantage"] = result["low_rank"] - result[["routing", "gain"]].max(
        axis=1
    )
    return result


def _mean_interval(
    values: pd.Series | np.ndarray,
    *,
    random_seed: int,
    samples: int = 5_000,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return mean, float(low), float(high)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZE + 1,
        fontweight="bold",
        va="bottom",
    )


def _draw_advantage(
    axis: plt.Axes,
    advantage: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> None:
    thresholds = endpoints["discovery_threshold"].to_numpy(dtype=float)
    threshold_low = float(np.min(thresholds))
    threshold_high = float(np.max(thresholds))
    axis.axvspan(
        threshold_low,
        threshold_high,
        color=COLORS["orange"],
        alpha=0.20,
        linewidth=0,
        label="Discovery threshold range",
        zorder=0,
    )
    axis.axvline(
        float(np.median(thresholds)),
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )
    axis.axhline(0.0, color="black", linestyle=":", linewidth=0.9, zorder=1)
    if advantage.empty:
        axis.text(
            0.5,
            0.5,
            "No complete budget-matched triplets",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    else:
        ranks = sorted(advantage["transition_rank"].unique())
        rank_colors = (
            COLORS["blue"],
            COLORS["vermillion"],
            COLORS["green"],
            COLORS["purple"],
            COLORS["gray"],
        )
        markers = ("o", "s", "^", "D", "P")
        delays = sorted(advantage["delay"].unique())
        marker_for_delay = {
            value: markers[index % len(markers)] for index, value in enumerate(delays)
        }
        for index, rank in enumerate(ranks):
            subset = advantage.loc[advantage["transition_rank"].eq(rank)]
            for delay, delay_subset in subset.groupby("delay", sort=True):
                axis.scatter(
                    delay_subset["chi"],
                    delay_subset["advantage"],
                    s=26,
                    marker=marker_for_delay[delay],
                    facecolor=rank_colors[index % len(rank_colors)],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.72,
                    label=(
                        rf"rank $\Delta A$={rank:g}"
                        if delay == delays[0]
                        else None
                    ),
                    zorder=2,
                )
    axis.set_xlabel(r"Gramian demand index $\chi$ (unitless)")
    axis.set_ylabel(
        "$\\Delta$ balanced accuracy\n(low-rank - best routing/gain)"
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="best", fontsize=FONT_SIZE)


def _strip_offsets(count: int) -> np.ndarray:
    if count < 2:
        return np.zeros(count, dtype=float)
    return np.linspace(-0.13, 0.13, count, dtype=float)


def _draw_seed_endpoints(axis: plt.Axes, endpoints: pd.DataFrame) -> None:
    specifications = (
        ("spearman_rho", r"Spearman $\rho$", 0.0, COLORS["blue"]),
        (
            "classifier_balanced_accuracy",
            "Classifier BA",
            0.5,
            COLORS["vermillion"],
        ),
        ("classifier_auroc", "Classifier AUROC", 0.5, COLORS["green"]),
    )
    for position, (column, _label, null, color) in enumerate(specifications):
        values = endpoints[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        shown = values[finite]
        axis.scatter(
            position + _strip_offsets(shown.size),
            shown,
            s=24,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.65,
            zorder=2,
        )
        mean, low, high = _mean_interval(
            shown, random_seed=2600 + position
        )
        if np.isfinite(mean):
            axis.errorbar(
                position,
                mean,
                yerr=np.asarray([[mean - low], [high - mean]]),
                fmt="D",
                color="black",
                markerfacecolor="white",
                markersize=5,
                capsize=3,
                linewidth=1.2,
                zorder=3,
            )
        axis.hlines(
            null,
            position - 0.27,
            position + 0.27,
            color=COLORS["gray"],
            linestyle=":",
            linewidth=0.9,
            zorder=0,
        )
    axis.set_xticks(
        np.arange(len(specifications)), [item[1] for item in specifications]
    )
    axis.tick_params(axis="x", rotation=15)
    axis.set_ylabel("Seed-level endpoint (unitless)")


def _draw_incremental_auc(axis: plt.Axes, endpoints: pd.DataFrame) -> None:
    values = endpoints["chi_minus_alpha_auroc"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    axis.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    axis.scatter(
        _strip_offsets(values.size),
        values,
        s=28,
        facecolor=COLORS["purple"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.72,
        zorder=2,
    )
    mean, low, high = _mean_interval(values, random_seed=2620)
    if np.isfinite(mean):
        axis.errorbar(
            0.0,
            mean,
            yerr=np.asarray([[mean - low], [high - mean]]),
            fmt="D",
            color="black",
            markerfacecolor="white",
            markersize=6,
            capsize=4,
            linewidth=1.3,
            zorder=3,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No finite seed endpoint",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.set_xlim(-0.42, 0.42)
    axis.set_xticks([0.0], [r"Gramian $\chi$ vs. raw $\alpha$"])
    axis.set_ylabel(r"AUROC($\chi$) - AUROC($\alpha$)")


def _draw_coverage(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    observed = set(metrics["actuator_mode"].unique())
    modes = [mode for mode in MODE_ORDER if mode in observed]
    modes.extend(sorted(observed - set(modes)))
    if not modes:
        raise ValueError("metrics contain no actuator modes")
    totals = metrics.groupby("actuator_mode", sort=False).size().reindex(modes)
    complete = metrics["status"].eq("complete")
    valid = complete & metrics["functional_budget_valid"]
    budget_invalid = complete & ~metrics["functional_budget_valid"]
    incomplete = ~complete
    categories = (
        (valid, "Complete + budget valid", COLORS["green"]),
        (budget_invalid, "Complete, budget invalid", COLORS["orange"]),
        (incomplete, "Failed/incomplete", COLORS["gray"]),
    )
    positions = np.arange(len(modes))
    bottom = np.zeros(len(modes), dtype=float)
    for mask, label, color in categories:
        counts = (
            metrics.loc[mask].groupby("actuator_mode", sort=False).size().reindex(
                modes, fill_value=0
            )
        )
        percentages = 100.0 * counts.to_numpy(dtype=float) / totals.to_numpy(
            dtype=float
        )
        axis.bar(
            positions,
            percentages,
            bottom=bottom,
            width=0.72,
            color=color,
            label=label,
            linewidth=0,
        )
        bottom += percentages
    for position, total in zip(positions, totals, strict=True):
        axis.text(
            position,
            102.0,
            f"n={int(total)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )
    axis.set_xticks(positions, [MODE_LABELS.get(mode, mode) for mode in modes])
    axis.set_ylim(0.0, 112.0)
    axis.set_ylabel("Planned cells (%)")
    axis.set_xlabel("Actuator mode")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)


def make_figure(
    metrics: pd.DataFrame,
    seed_endpoints: pd.DataFrame,
) -> plt.Figure:
    """Build the four-panel bound figure after fail-closed schema validation."""

    clean_metrics, clean_endpoints = _validated_inputs(metrics, seed_endpoints)
    advantage = heldout_advantage(clean_metrics)
    _setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.7))
    _draw_advantage(axes[0, 0], advantage, clean_endpoints)
    _draw_seed_endpoints(axes[0, 1], clean_endpoints)
    _draw_incremental_auc(axes[1, 0], clean_endpoints)
    _draw_coverage(axes[1, 1], clean_metrics)
    for label, axis in zip(("A", "B", "C", "D"), axes.ravel(), strict=True):
        _panel_label(axis, label)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(w_pad=2.2, h_pad=2.6)
    return figure


def plot_exp26(
    metrics: pd.DataFrame,
    seed_endpoints: pd.DataFrame,
    output_dir: Path | str,
    *,
    stem: str = FIGURE_STEM,
) -> tuple[Path, Path]:
    """Render and save deterministic PDF and SVG vector outputs."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure = make_figure(metrics, seed_endpoints)
    pdf_path = destination / f"{stem}.pdf"
    svg_path = destination / f"{stem}.svg"
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(figure)
    for path in (pdf_path, svg_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"figure output is missing or empty: {path}")
    return pdf_path, svg_path


def _read_csv(path: Path, *, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{name} CSV is empty: {path}") from error
    if frame.empty:
        raise ValueError(f"{name} CSV contains no rows: {path}")
    return frame


def main(argv: Sequence[str] | None = None) -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed-endpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = _read_csv(args.metrics, name="metrics")
    endpoints = _read_csv(args.seed_endpoints, name="seed_endpoints")
    paths = plot_exp26(metrics, endpoints, args.output_dir)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
