"""Bind the 30-seed N=256 exp10 formal grid to scoped statistics and a figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.plot_style import COLORS, save_figure, setup_style  # noqa: E402
from src.analysis.run_provenance import (  # noqa: E402
    build_exp10_run_manifest,
    latest_exp10_formal_attempts,
    validate_exp10_checkpoint_contract,
    validate_exp10_run_manifest,
)


EXPERIMENT = "exp10_hidden_context_ei_bridge"
FIGURE_NAME = "exp10_bridge_formal"
BOOTSTRAP_SEED = 20260714
N_BOOTSTRAP = 100_000
Q_VALUES = (0.70, 0.85)
H_VALUES = (0.05, 0.20)
BASE_CONDITIONS = (
    ("no_gate", "none", "No gate"),
    ("learned_hmm", "none", "Learned HMM"),
    ("md_recurrent_belief", "none", "MD-like"),
    ("oracle_bayes", "none", "Oracle Bayes"),
)
COMPARISONS = (
    ("hmm_context_vs_no_gate", "simulated_hidden_context_inference"),
    ("md_context_vs_no_gate", "simulated_hidden_context_inference"),
    ("hmm_behavior_vs_no_gate", "separately_refit_functional_pipeline"),
    ("md_behavior_vs_no_gate", "separately_refit_functional_pipeline"),
    ("oracle_behavior_vs_no_gate", "descriptive_oracle_ceiling"),
    ("md_retains_90pct_oracle_gain", "separately_refit_noninferiority_margin"),
    ("md_vs_clamp", "fixed_checkpoint_within_model_counterfactual"),
    ("md_vs_delay", "fixed_checkpoint_within_model_counterfactual"),
    ("md_vs_shuffle", "fixed_checkpoint_within_model_counterfactual"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _latest_formal_rows(results_root: Path) -> pd.DataFrame:
    """Load latest formal attempt per seed, or the bound committed snapshot."""

    latest_attempts = latest_exp10_formal_attempts(results_root)
    rows: list[dict[str, object]] = []
    sources: list[str] = []
    if latest_attempts:
        for seed in range(30):
            latest = latest_attempts[seed]
            status_path = latest / "status.json"
            if not status_path.is_file():
                raise RuntimeError(
                    f"latest exp10 formal attempt for seed {seed} lacks status.json"
                )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if str(status.get("status")) != "complete":
                raise RuntimeError(
                    "latest exp10 formal attempt is not complete; refusing to "
                    f"fall back to an older success (seed {seed}, {latest.name})"
                )
            metrics_path = latest / "metrics.jsonl"
            if (
                not metrics_path.is_file()
                or not metrics_path.read_text(encoding="utf-8").strip()
            ):
                raise RuntimeError(
                    f"latest exp10 formal attempt for seed {seed} has no metrics"
                )
            current = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows.extend(current)
            sources.extend([_source_label(metrics_path)] * len(current))
        frame = pd.DataFrame(rows)
        frame["source_metrics_path"] = sources
    else:
        snapshot = results_root / "exp10_bridge_formal_raw.csv.gz"
        if not snapshot.is_file():
            raise FileNotFoundError(
                "no complete 30-seed N=256 exp10 formal grid or scoped snapshot"
            )
        frame = pd.read_csv(snapshot, low_memory=False)
    required = {
        "seed",
        "status",
        "profile",
        "cue_reliability",
        "context_hazard",
        "network_n_units",
        "gate_model",
        "intervention",
        "behavior_balanced_accuracy",
        "context_nll",
        "bridge_protocol_id",
        "recurrent_learning",
        "base_conditions_share_readout",
        "base_comparison_scope",
        "efficiency_claim_eligible",
        "three_factor_plasticity_claim_eligible",
        "intervention_postfit",
        "intervention_reuses_intact_gate_checkpoint",
        "intervention_reuses_intact_readout",
        "intervention_reuses_intact_receiver",
        "readout_checkpoint_id",
        "gate_checkpoint_id",
        "network_initialization_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"exp10 formal rows lack columns: {missing}")
    condition_pairs = set(
        map(tuple, frame[["gate_model", "intervention"]].drop_duplicates().to_numpy())
    )
    expected_pairs = {
        ("no_gate", "none"),
        ("learned_hmm", "none"),
        ("md_recurrent_belief", "none"),
        ("oracle_bayes", "none"),
        ("md_recurrent_belief", "clamp"),
        ("md_recurrent_belief", "delay"),
        ("md_recurrent_belief", "shuffle"),
    }
    cell_sizes = frame.groupby(["seed", "cue_reliability", "context_hazard"]).size()
    if (
        len(frame) != 840
        or frame["seed"].nunique() != 30
        or set(frame["status"].astype(str)) != {"complete"}
        or set(frame["profile"].astype(str)) != {"formal"}
        or set(frame["network_n_units"].astype(int)) != {256}
        or set(np.round(frame["cue_reliability"].astype(float), 8)) != set(Q_VALUES)
        or set(np.round(frame["context_hazard"].astype(float), 8)) != set(H_VALUES)
        or condition_pairs != expected_pairs
        or not cell_sizes.eq(7).all()
        or frame["bridge_protocol_id"].astype(str).nunique() != 1
        or frame["recurrent_learning"].astype(bool).any()
        or frame["base_conditions_share_readout"].astype(bool).any()
        or set(frame["base_comparison_scope"].astype(str))
        != {"separately_train_optimized_pipeline_comparison"}
        or frame["efficiency_claim_eligible"].astype(bool).any()
        or frame["three_factor_plasticity_claim_eligible"].astype(bool).any()
    ):
        raise ValueError("exp10 formal rows violate the registered N=256 contract")
    validate_exp10_checkpoint_contract(frame)
    return frame.sort_values(
        ["seed", "cue_reliability", "context_hazard", "gate_model", "intervention"],
        kind="mergesort",
    ).reset_index(drop=True)


def _cell_pivot(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.pivot(
        index=["seed", "cue_reliability", "context_hazard"],
        columns=["gate_model", "intervention"],
        values=metric,
    )


def _comparison_cell_values(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in {"hmm_context_vs_no_gate", "md_context_vs_no_gate"}:
        pivot = _cell_pivot(frame, "context_nll")
        gate = "learned_hmm" if name.startswith("hmm") else "md_recurrent_belief"
        return pivot[("no_gate", "none")] - pivot[(gate, "none")]
    pivot = _cell_pivot(frame, "behavior_balanced_accuracy")
    no_gate = pivot[("no_gate", "none")]
    if name == "hmm_behavior_vs_no_gate":
        return pivot[("learned_hmm", "none")] - no_gate
    if name == "md_behavior_vs_no_gate":
        return pivot[("md_recurrent_belief", "none")] - no_gate
    if name == "oracle_behavior_vs_no_gate":
        return pivot[("oracle_bayes", "none")] - no_gate
    if name == "md_retains_90pct_oracle_gain":
        return (pivot[("md_recurrent_belief", "none")] - no_gate) - 0.9 * (
            pivot[("oracle_bayes", "none")] - no_gate
        )
    intervention = name.removeprefix("md_vs_")
    if intervention not in {"clamp", "delay", "shuffle"}:
        raise ValueError(f"unknown exp10 formal comparison {name!r}")
    return (
        pivot[("md_recurrent_belief", "none")]
        - pivot[("md_recurrent_belief", intervention)]
    )


def _seed_macro_values(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = _comparison_cell_values(frame, name)
    return values.groupby("seed").mean().sort_index().to_numpy(float)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(N_BOOTSTRAP, values.size))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def formal_comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired seed-primary inference after equal q/h-cell macro averaging."""

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for name, scope in COMPARISONS:
        values = _seed_macro_values(frame, name)
        low, high = _bootstrap_ci(values, rng)
        p_value = (
            1.0
            if np.allclose(values, 0.0)
            else float(wilcoxon(values, alternative="two-sided").pvalue)
        )
        cell_values = _comparison_cell_values(frame, name)
        cell_means = cell_values.groupby(["cue_reliability", "context_hazard"]).mean()
        rows.append(
            {
                "comparison": name,
                "comparison_scope": scope,
                "metric": (
                    "context_nll"
                    if "context_vs_no_gate" in name
                    else "behavior_balanced_accuracy"
                ),
                "n_seeds": int(values.size),
                "n_q_h_cells": int(cell_means.size),
                "mean_difference": float(values.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wilcoxon_two_sided_p": p_value,
                "positive_seed_count": int(np.count_nonzero(values > 0.0)),
                "zero_seed_count": int(np.count_nonzero(values == 0.0)),
                "minimum_q_h_cell_mean": float(cell_means.min()),
                "maximum_q_h_cell_mean": float(cell_means.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_p"] = multipletests(
        summary["wilcoxon_two_sided_p"].to_numpy(), method="holm"
    )[1]
    summary["classification"] = "inconclusive"
    support = (summary["holm_p"] < 0.05) & (summary["bootstrap_ci_low"] > 0.0)
    oppose = (summary["holm_p"] < 0.05) & (summary["bootstrap_ci_high"] < 0.0)
    summary.loc[support, "classification"] = "support"
    summary.loc[oppose, "classification"] = "oppose"
    summary["conclusion"] = summary["classification"]
    summary.loc[
        summary["comparison_scope"].eq("simulated_hidden_context_inference")
        & summary["classification"].eq("support"),
        "conclusion",
    ] = "support_simulated_hidden_context_inference"
    summary.loc[
        summary["comparison_scope"].eq("separately_refit_functional_pipeline")
        & summary["classification"].eq("support"),
        "conclusion",
    ] = "support_functional_pipeline_formal"
    summary.loc[
        summary["comparison_scope"].eq("separately_refit_noninferiority_margin")
        & summary["classification"].eq("support"),
        "conclusion",
    ] = "support_macro_average_90pct_oracle_gain_margin"
    summary.loc[
        summary["comparison_scope"].eq("fixed_checkpoint_within_model_counterfactual")
        & summary["classification"].eq("support"),
        "conclusion",
    ] = "support_within_model_counterfactual"
    summary.loc[
        summary["comparison"].eq("oracle_behavior_vs_no_gate")
        & summary["classification"].eq("support"),
        "conclusion",
    ] = "descriptive_oracle_ceiling_support"
    summary["profile"] = "formal"
    summary["network_n_units"] = 256
    summary["statistics_unit"] = "seed"
    summary["within_seed_aggregation"] = "equal_macro_average_across_4_q_h_cells"
    summary["multiple_comparison_correction"] = "Holm_across_exp10_formal_family"
    summary["base_conditions_share_readout"] = False
    summary["recurrent_learning"] = False
    summary["biological_mechanism_claim_eligible"] = False
    summary["three_factor_plasticity_claim_eligible"] = False
    summary["efficiency_claim_eligible"] = False
    summary["bridge_protocol_id"] = str(frame["bridge_protocol_id"].iloc[0])
    summary["all_q_h_cell_means_positive"] = summary["minimum_q_h_cell_mean"] > 0.0
    return summary


def _plot_seed_points(
    axis: plt.Axes,
    values_by_group: list[np.ndarray],
    labels: list[str],
    rng: np.random.Generator,
) -> None:
    for index, values in enumerate(values_by_group):
        axis.scatter(
            rng.normal(index + 1, 0.045, size=values.size),
            values,
            s=12,
            alpha=0.62,
            color=COLORS[index + 1],
        )
        low, high = _bootstrap_ci(values, rng)
        mean = float(values.mean())
        axis.errorbar(
            index + 1,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color="black",
            markersize=4,
            linewidth=1.1,
            capsize=3,
        )
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=27)


def make_figure(frame: pd.DataFrame, results_root: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.35))
    rng = np.random.default_rng(1978)

    behavior = _cell_pivot(frame, "behavior_balanced_accuracy")
    groups = [
        behavior[(gate, intervention)]
        .groupby("seed")
        .mean()
        .sort_index()
        .to_numpy(float)
        * 100.0
        for gate, intervention, _ in BASE_CONDITIONS
    ]
    boxes = axes[0].boxplot(
        groups,
        tick_labels=[label for _, _, label in BASE_CONDITIONS],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for index, (box, values) in enumerate(zip(boxes["boxes"], groups, strict=True)):
        box.set_facecolor(COLORS[index])
        box.set_alpha(0.25)
        axes[0].scatter(
            rng.normal(index + 1, 0.045, size=values.size),
            values,
            s=11,
            color=COLORS[index],
            alpha=0.58,
        )
    axes[0].set_ylabel("Held-out balanced accuracy (%)")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].text(
        0.98,
        0.03,
        "N=256; n=30 seeds\n4 q/h cells; frozen recurrent",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
    )

    pipeline_names = (
        "hmm_behavior_vs_no_gate",
        "md_behavior_vs_no_gate",
        "oracle_behavior_vs_no_gate",
    )
    _plot_seed_points(
        axes[1],
        [_seed_macro_values(frame, name) * 100.0 for name in pipeline_names],
        ["Learned HMM", "MD-like", "Oracle Bayes"],
        rng,
    )
    axes[1].set_ylabel("Pipeline Δ vs separately refit no-gate (pp)")

    intervention_names = ("md_vs_clamp", "md_vs_delay", "md_vs_shuffle")
    _plot_seed_points(
        axes[2],
        [_seed_macro_values(frame, name) * 100.0 for name in intervention_names],
        ["Clamp", "Delay", "Shuffle"],
        rng,
    )
    axes[2].set_ylabel("Intact − intervention Δ (pp)")

    context_names = ("hmm_context_vs_no_gate", "md_context_vs_no_gate")
    _plot_seed_points(
        axes[3],
        [_seed_macro_values(frame, name) for name in context_names],
        ["Learned HMM", "MD-like"],
        rng,
    )
    axes[3].set_ylabel("Context NLL improvement vs uniform")

    for index, axis in enumerate(axes):
        axis.text(
            -0.17,
            1.03,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.45)
    save_figure(fig, FIGURE_NAME, results_root)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()
    results_root = Path(args.results_root).resolve()
    frame = _latest_formal_rows(results_root)
    raw_path = results_root / "exp10_bridge_formal_raw.csv.gz"
    frame.to_csv(
        raw_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        lineterminator="\n",
    )
    published_raw = pd.read_csv(raw_path, low_memory=False)
    manifest_path = results_root / "exp10_bridge_formal_run_manifest.csv"
    if latest_exp10_formal_attempts(results_root):
        run_manifest = build_exp10_run_manifest(results_root, published_raw)
        run_manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "exp10 formal support requires a published clean-run manifest"
            )
        run_manifest = pd.read_csv(manifest_path, low_memory=False)
        validate_exp10_run_manifest(run_manifest, published_raw)
    summary = formal_comparison_summary(frame)
    summary["scoped_raw_sha256"] = _file_sha256(raw_path)
    summary["run_manifest_sha256"] = _file_sha256(manifest_path)
    summary["run_git_commit"] = str(run_manifest["git_commit"].iloc[0])
    summary["run_git_dirty"] = False
    summary.to_csv(
        results_root / "exp10_bridge_formal_summary.csv",
        index=False,
        lineterminator="\n",
    )
    make_figure(frame, results_root)


if __name__ == "__main__":
    main()
