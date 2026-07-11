"""Bind the 30-seed exp10 pilot metrics to a publication-style figure."""

from __future__ import annotations

import argparse
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


EXPERIMENT = "exp10_hidden_context_ei_bridge"
FIGURE_NAME = "exp10_bridge_pilot"
BOOTSTRAP_SEED = 20260711
N_BOOTSTRAP = 100_000

CONDITION_ORDER = (
    ("no_gate", "none", "No gate"),
    ("learned_hmm", "none", "Learned HMM"),
    ("md_recurrent_belief", "none", "MD-like"),
    ("oracle_bayes", "none", "Oracle Bayes"),
)


def _is_registered_pilot_frame(frame: pd.DataFrame) -> bool:
    required = {
        "seed",
        "status",
        "profile",
        "cue_reliability",
        "context_hazard",
        "network_n_units",
        "recurrent_learning",
        "gate_model",
        "intervention",
    }
    if len(frame) != 7 or not required <= set(frame.columns):
        return False
    return bool(
        frame["seed"].nunique() == 1
        and set(frame["status"].astype(str)) == {"complete"}
        and set(frame["profile"].astype(str)) == {"smoke"}
        and set(frame["network_n_units"].astype(int)) == {32}
        and np.allclose(frame["cue_reliability"].astype(float), 0.70)
        and np.allclose(frame["context_hazard"].astype(float), 0.10)
        and not frame["recurrent_learning"].astype(bool).any()
        and frame[["gate_model", "intervention"]].drop_duplicates().shape[0] == 7
    )


def _latest_complete_rows(results_root: Path) -> pd.DataFrame:
    experiment_root = results_root / "runs" / EXPERIMENT
    rows: list[dict[str, object]] = []
    source_paths: list[str] = []
    for seed_dir in sorted(experiment_root.glob("seed_*")):
        complete: list[Path] = []
        for run_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            status_path = run_dir / "status.json"
            if not status_path.exists():
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if str(status.get("status", "")).startswith("complete"):
                complete.append(run_dir)
        for run_dir in reversed(complete):
            metrics_path = run_dir / "metrics.jsonl"
            candidate_rows = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            candidate = pd.DataFrame(candidate_rows)
            if not _is_registered_pilot_frame(candidate):
                continue
            rows.extend(candidate_rows)
            source_paths.extend(
                [str(metrics_path.relative_to(PROJECT_ROOT))] * len(candidate_rows)
            )
            break
    frame = pd.DataFrame(rows)
    # A newer formal N=256 attempt must not replace the immutable N=32 pilot.
    # If the complete 30-seed pilot is not available in run directories, bind
    # the figure to the committed protocol snapshot rather than mixing attempts.
    if frame.empty or frame["seed"].nunique() != 30 or len(frame) != 210:
        snapshot = results_root / "exp10_bridge_pilot_raw.csv.gz"
        if not snapshot.is_file():
            raise FileNotFoundError(
                f"no complete 30-seed {EXPERIMENT} pilot or committed snapshot"
            )
        frame = pd.read_csv(snapshot)
    else:
        frame["source_metrics_path"] = source_paths
    if frame["seed"].nunique() != 30 or len(frame) != 210:
        raise ValueError(
            "pilot figure requires exactly 30 seeds x 7 pre-registered conditions"
        )
    if set(frame["status"]) != {"complete"}:
        raise ValueError("pilot snapshot contains non-complete conditions")
    if frame.groupby("seed").size().nunique() != 1:
        raise ValueError("pilot condition grid is not balanced across seeds")
    required_provenance = {
        "profile",
        "cue_reliability",
        "context_hazard",
        "network_n_units",
        "bridge_protocol_id",
        "recurrent_learning",
        "base_conditions_share_readout",
        "base_comparison_scope",
        "efficiency_claim_eligible",
        "three_factor_plasticity_claim_eligible",
    }
    missing = sorted(required_provenance - set(frame.columns))
    if missing:
        raise ValueError(f"pilot snapshot lacks provenance columns: {missing}")
    if (
        set(frame["profile"].astype(str)) != {"smoke"}
        or set(frame["network_n_units"].astype(int)) != {32}
        or not np.allclose(frame["cue_reliability"].astype(float), 0.70)
        or not np.allclose(frame["context_hazard"].astype(float), 0.10)
        or frame["bridge_protocol_id"].astype(str).nunique() != 1
        or frame["recurrent_learning"].astype(bool).any()
        or frame["base_conditions_share_readout"].astype(bool).any()
        or set(frame["base_comparison_scope"].astype(str))
        != {"separately_train_optimized_pipeline_comparison"}
        or frame["efficiency_claim_eligible"].astype(bool).any()
        or frame["three_factor_plasticity_claim_eligible"].astype(bool).any()
    ):
        raise ValueError("pilot snapshot disagrees with the registered N=32 scope")
    return frame.sort_values(
        ["seed", "gate_model", "intervention"], kind="mergesort"
    ).reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(N_BOOTSTRAP, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(
        index="seed",
        columns=["gate_model", "intervention"],
        values="behavior_balanced_accuracy",
    )
    comparisons = (
        ("oracle_vs_no_gate", ("oracle_bayes", "none"), ("no_gate", "none")),
        ("hmm_vs_no_gate", ("learned_hmm", "none"), ("no_gate", "none")),
        (
            "md_vs_no_gate",
            ("md_recurrent_belief", "none"),
            ("no_gate", "none"),
        ),
        (
            "md_vs_clamp",
            ("md_recurrent_belief", "none"),
            ("md_recurrent_belief", "clamp"),
        ),
        (
            "md_vs_delay",
            ("md_recurrent_belief", "none"),
            ("md_recurrent_belief", "delay"),
        ),
        (
            "md_vs_shuffle",
            ("md_recurrent_belief", "none"),
            ("md_recurrent_belief", "shuffle"),
        ),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for name, candidate, reference in comparisons:
        differences = (pivot[candidate] - pivot[reference]).to_numpy(float)
        low, high = _bootstrap_ci(differences, rng)
        try:
            p_value = float(
                wilcoxon(differences, alternative="greater", zero_method="pratt").pvalue
            )
        except ValueError:
            p_value = 1.0
        rows.append(
            {
                "comparison": name,
                "candidate_gate": candidate[0],
                "candidate_intervention": candidate[1],
                "reference_gate": reference[0],
                "reference_intervention": reference[1],
                "n_seeds": differences.size,
                "mean_balanced_accuracy_difference": float(differences.mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wilcoxon_greater_p": p_value,
                "positive_seed_count": int(np.count_nonzero(differences > 0.0)),
                "zero_seed_count": int(np.count_nonzero(differences == 0.0)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_p"] = multipletests(
        summary["wilcoxon_greater_p"].to_numpy(), method="holm"
    )[1]
    summary["conclusion"] = np.where(
        (summary["holm_p"] < 0.05) & (summary["bootstrap_ci_low"] > 0.0),
        "functional_pipeline_support_pilot",
        "inconclusive_functional_pipeline_pilot",
    )
    intervention_mask = summary["comparison"].isin(
        {"md_vs_clamp", "md_vs_delay", "md_vs_shuffle"}
    )
    intervention_support = (
        intervention_mask
        & (summary["holm_p"] < 0.05)
        & (summary["bootstrap_ci_low"] > 0.0)
    )
    summary.loc[intervention_mask, "conclusion"] = (
        "inconclusive_within_model_counterfactual"
    )
    summary.loc[intervention_support, "conclusion"] = (
        "within_model_counterfactual_support_pilot"
    )
    summary.loc[summary["comparison"] == "oracle_vs_no_gate", "conclusion"] = (
        "descriptive_ceiling_support"
    )
    summary["comparison_scope"] = np.where(
        intervention_mask,
        "fixed_receiver_readout_within_model_counterfactual",
        "separately_refit_readout_functional_pipeline",
    )
    summary["base_conditions_share_readout"] = False
    summary["recurrent_learning"] = False
    summary["biological_mechanism_claim_eligible"] = False
    summary["three_factor_plasticity_claim_eligible"] = False
    summary["efficiency_claim_eligible"] = False
    summary["statistics_unit"] = "seed"
    summary["multiple_comparison_correction"] = "Holm_within_exp10_pilot_family"
    return summary


def _paired_differences(
    frame: pd.DataFrame,
    comparisons: list[tuple[tuple[str, str], tuple[str, str]]],
) -> list[np.ndarray]:
    pivot = frame.pivot(
        index="seed",
        columns=["gate_model", "intervention"],
        values="behavior_balanced_accuracy",
    )
    return [
        (pivot[candidate] - pivot[reference]).to_numpy(float) * 100.0
        for candidate, reference in comparisons
    ]


def make_figure(frame: pd.DataFrame, results_root: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    rng = np.random.default_rng(1977)

    groups = [
        frame[(frame["gate_model"] == gate) & (frame["intervention"] == intervention)]
        .sort_values("seed")["behavior_balanced_accuracy"]
        .to_numpy(float)
        * 100.0
        for gate, intervention, _ in CONDITION_ORDER
    ]
    labels = [label for _, _, label in CONDITION_ORDER]
    boxes = axes[0].boxplot(
        groups,
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for index, (box, values) in enumerate(zip(boxes["boxes"], groups, strict=True)):
        box.set_facecolor(COLORS[index])
        box.set_alpha(0.28)
        jitter = rng.normal(index + 1, 0.045, size=values.size)
        axes[0].scatter(
            jitter,
            values,
            s=12,
            alpha=0.62,
            color=COLORS[index],
            edgecolors="none",
        )
    axes[0].set_ylabel("Held-out balanced accuracy (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[0].text(
        0.98,
        0.03,
        "N=32; q=.70; h=.10\nfrozen recurrent; n=30 seeds",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
    )

    vs_no = _paired_differences(
        frame,
        [
            (("learned_hmm", "none"), ("no_gate", "none")),
            (("md_recurrent_belief", "none"), ("no_gate", "none")),
            (("oracle_bayes", "none"), ("no_gate", "none")),
        ],
    )
    delta_labels = ["Learned HMM", "MD-like", "Oracle Bayes"]
    for index, values in enumerate(vs_no):
        jitter = rng.normal(index + 1, 0.045, size=values.size)
        axes[1].scatter(jitter, values, s=13, alpha=0.65, color=COLORS[index + 1])
        low, high = _bootstrap_ci(values, rng)
        axes[1].errorbar(
            index + 1,
            values.mean(),
            yerr=[[values.mean() - low], [high - values.mean()]],
            fmt="D",
            markersize=4,
            color="black",
            capsize=3,
            linewidth=1.2,
        )
    axes[1].axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(range(1, 4), delta_labels, rotation=30)
    axes[1].set_ylabel("Pipeline Δ vs separately refit no-gate (pp)")
    axes[1].text(-0.16, 1.03, "b", transform=axes[1].transAxes, fontweight="bold")

    interventions = _paired_differences(
        frame,
        [
            (
                ("md_recurrent_belief", "none"),
                ("md_recurrent_belief", "clamp"),
            ),
            (
                ("md_recurrent_belief", "none"),
                ("md_recurrent_belief", "delay"),
            ),
            (
                ("md_recurrent_belief", "none"),
                ("md_recurrent_belief", "shuffle"),
            ),
        ],
    )
    for index, values in enumerate(interventions):
        jitter = rng.normal(index + 1, 0.045, size=values.size)
        axes[2].scatter(jitter, values, s=13, alpha=0.65, color=COLORS[index + 2])
        low, high = _bootstrap_ci(values, rng)
        axes[2].errorbar(
            index + 1,
            values.mean(),
            yerr=[[values.mean() - low], [high - values.mean()]],
            fmt="D",
            markersize=4,
            color="black",
            capsize=3,
            linewidth=1.2,
        )
    axes[2].axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(range(1, 4), ["Clamp", "Delay", "Shuffle"], rotation=25)
    axes[2].set_ylabel("Intact − intervention Δ (pp)")
    axes[2].text(-0.16, 1.03, "c", transform=axes[2].transAxes, fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.8)
    save_figure(fig, FIGURE_NAME, results_root)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()
    results_root = Path(args.results_root).resolve()
    frame = _latest_complete_rows(results_root)
    frame.to_csv(
        results_root / "exp10_bridge_pilot_raw.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        lineterminator="\n",
    )
    _comparison_summary(frame).to_csv(
        results_root / "exp10_bridge_pilot_summary.csv",
        index=False,
        lineterminator="\n",
    )
    make_figure(frame, results_root)


if __name__ == "__main__":
    main()
