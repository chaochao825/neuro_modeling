"""Animal-primary summary and figure for the formal IBL behavior cohort."""

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


EXPERIMENT = "exp11_ibl_behavior_belief"
FIGURE_NAME = "exp11_ibl_behavior_real"
BOOTSTRAP_SEED = 20260712


def _source_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_real_rows(results_root: Path) -> pd.DataFrame:
    """Use the latest formal attempt, never a hand-picked latest success."""

    experiment_root = results_root / "runs" / EXPERIMENT
    run_dirs: list[Path] = []
    if experiment_root.is_dir():
        for seed_dir in experiment_root.glob("seed_*"):
            for run_dir in seed_dir.iterdir():
                config_path = run_dir / "config.json"
                if not config_path.is_file():
                    continue
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if str(config.get("profile", "")) == "formal":
                    run_dirs.append(run_dir)
    if run_dirs:
        # Timestamp-prefixed immutable directory names make this the latest
        # attempt, including failed or interrupted retries.
        latest = max(run_dirs, key=lambda path: path.name)
        metrics_path = latest / "metrics.jsonl"
        if (
            not metrics_path.is_file()
            or not metrics_path.read_text(encoding="utf-8").strip()
        ):
            raise RuntimeError(
                "latest formal exp11 attempt has no metric rows; refusing to "
                "fall back to an older successful attempt"
            )
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        frame = pd.DataFrame(rows)
        frame["source_metrics_path"] = _source_label(metrics_path)
        status_path = latest / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {"status": "missing_status"}
        )
        frame["source_run_status"] = str(status.get("status", "unknown"))
        frame["source_run_attempt"] = latest.name
        planned_path = latest / "planned_conditions.json"
        if planned_path.is_file():
            planned_rows = json.loads(planned_path.read_text(encoding="utf-8"))
            if not isinstance(planned_rows, list):
                raise ValueError("exp11 planned_conditions.json must be a list")
            observed_keys = {
                (str(row.get("eid", "")), str(row.get("condition", ""))) for row in rows
            }
            missing_rows = []
            for planned_row in planned_rows:
                key = (
                    str(planned_row.get("eid", "")),
                    str(planned_row.get("condition", "")),
                )
                if key in observed_keys:
                    continue
                missing_rows.append(
                    {
                        **planned_row,
                        "status": "missing_from_metrics",
                        "error_type": "InterruptedRunMissingMetric",
                        "error": (
                            "planned condition has no metric row in the latest "
                            "formal attempt"
                        ),
                        "source_metrics_path": _source_label(metrics_path),
                        "source_run_status": str(status.get("status", "unknown")),
                        "source_run_attempt": latest.name,
                    }
                )
            if missing_rows:
                frame = pd.concat(
                    [frame, pd.DataFrame(missing_rows)],
                    ignore_index=True,
                    sort=False,
                )
    else:
        snapshot = results_root / "exp11_ibl_behavior_real_raw.csv.gz"
        if not snapshot.is_file():
            raise FileNotFoundError("no formal exp11 run or committed raw snapshot")
        frame = pd.read_csv(snapshot)
        if "source_run_status" not in frame.columns:
            frame["source_run_status"] = "snapshot_status_unavailable"
    if not {"condition", "status"} <= set(frame.columns):
        raise ValueError("exp11 rows require condition and status")
    optional_columns = {
        "eid",
        "animal_id",
        "profile",
        "context_nll",
        "behavior_log_loss",
        "official_bwm_mask_present",
        "cohort_manifest_sha256",
        "algorithmic_seed_is_statistical_unit",
        "behavior_only_benchmark",
        "neural_activity_analyzed",
        "eligible_for_context_inference_support",
        "eligible_for_behavior_pipeline_evaluation",
    }
    for name in optional_columns - set(frame.columns):
        frame[name] = np.nan
    complete = frame.loc[frame["status"].astype(str).eq("complete")]
    if not complete.empty:
        required_complete = {
            "profile",
            "context_nll",
            "behavior_log_loss",
            "official_bwm_mask_present",
            "cohort_manifest_sha256",
            "algorithmic_seed_is_statistical_unit",
            "behavior_only_benchmark",
            "neural_activity_analyzed",
        }
        if complete[list(required_complete)].isna().any().any():
            raise ValueError("complete exp11 rows lack required behavior provenance")
        if set(complete["profile"].astype(str)) != {"formal"}:
            raise ValueError("exp11 real summary accepts formal rows only")
        if set(complete["algorithmic_seed_is_statistical_unit"].astype(str)) != {
            "False"
        }:
            raise ValueError("algorithmic seed must not be a real-data replicate")
        if set(complete["behavior_only_benchmark"].astype(str)) != {"True"}:
            raise ValueError("exp11 real rows must be explicitly behavior-only")
        if set(complete["neural_activity_analyzed"].astype(str)) != {"False"}:
            raise ValueError("exp11 behavior benchmark must not claim neural activity")
    cohort_hashes = frame["cohort_manifest_sha256"].dropna().astype(str)
    if cohort_hashes.nunique() > 1:
        raise ValueError("exp11 rows do not share one frozen cohort manifest")
    return frame


def _hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Resample animals, then sessions nested within sampled animals."""

    animals = sorted(paired["animal_id"].astype(str).unique())
    if len(animals) < 2:
        return float("nan"), float("nan")
    grouped = {
        animal: paired.loc[
            paired["animal_id"].astype(str).eq(animal), "difference"
        ].to_numpy(float)
        for animal in animals
    }
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for bootstrap_index in range(int(n_bootstrap)):
        sampled_animals = rng.choice(animals, size=len(animals), replace=True)
        animal_means = []
        for animal in sampled_animals:
            values = grouped[str(animal)]
            animal_means.append(float(np.mean(rng.choice(values, size=values.size))))
        means[bootstrap_index] = float(np.mean(animal_means))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def real_data_comparison_summary(
    frame: pd.DataFrame, *, n_bootstrap: int = 100_000
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute intention-to-analyze, animal-primary behavior conclusions."""

    required = {"eid", "animal_id", "condition", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"exp11 comparison frame lacks columns: {missing}")
    expected_conditions = {
        "no_memory",
        "exponential_history",
        "learned_categorical_hmm",
        "oracle_ceiling",
    }
    scoped = frame.loc[
        frame["condition"].astype(str).isin(expected_conditions)
        & frame["eid"].notna()
        & frame["animal_id"].notna()
    ].copy()
    planned = scoped[["eid", "animal_id"]].drop_duplicates()
    if planned["eid"].duplicated().any():
        raise ValueError("one exp11 eid maps to multiple animals")
    n_planned_sessions = int(len(planned))
    n_planned_animals = int(planned["animal_id"].astype(str).nunique())
    cohort_hashes = (
        frame.get("cohort_manifest_sha256", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .unique()
    )
    cohort_hash = str(cohort_hashes[0]) if len(cohort_hashes) == 1 else "unavailable"
    if "source_run_status" in frame.columns:
        run_statuses = frame["source_run_status"].dropna().astype(str).unique()
        source_run_status = (
            str(run_statuses[0]) if len(run_statuses) == 1 else "mixed_or_missing"
        )
        run_attempt_finalized = source_run_status in {
            "complete",
            "complete_with_failures",
        }
    else:
        # Direct unit-level calls do not represent an artifact-selection path.
        source_run_status = "direct_frame_not_applicable"
        run_attempt_finalized = True
    comparisons = (
        (
            "hmm_context_nll_gain",
            "learned_categorical_hmm",
            "context_nll",
            True,
        ),
        (
            "history_context_nll_gain",
            "exponential_history",
            "context_nll",
            False,
        ),
        (
            "hmm_behavior_log_loss_gain",
            "learned_categorical_hmm",
            "behavior_log_loss",
            False,
        ),
        (
            "history_behavior_log_loss_gain",
            "exponential_history",
            "behavior_log_loss",
            False,
        ),
    )
    paired_by_claim: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for claim_index, (
        claim,
        candidate_name,
        metric,
        requires_valid_hmm_fit,
    ) in enumerate(comparisons):
        if metric not in scoped.columns:
            scoped[metric] = np.nan
        candidate_all = scoped.loc[
            scoped["condition"].astype(str).eq(candidate_name)
        ].copy()
        reference_all = scoped.loc[
            scoped["condition"].astype(str).eq("no_memory")
        ].copy()
        candidate_all[metric] = pd.to_numeric(candidate_all[metric], errors="coerce")
        reference_all[metric] = pd.to_numeric(reference_all[metric], errors="coerce")
        candidate = candidate_all.loc[
            candidate_all["status"].astype(str).eq("complete")
            & candidate_all[metric].notna()
            & candidate_all["official_bwm_mask_present"].astype(str).eq("True"),
            ["eid", "animal_id", metric],
        ].rename(columns={metric: "candidate"})
        reference = reference_all.loc[
            reference_all["status"].astype(str).eq("complete")
            & reference_all[metric].notna()
            & reference_all["official_bwm_mask_present"].astype(str).eq("True"),
            ["eid", "animal_id", metric],
        ].rename(columns={metric: "reference"})
        paired = candidate.merge(
            reference, on=["eid", "animal_id"], how="inner", validate="one_to_one"
        )
        paired["difference"] = paired["reference"] - paired["candidate"]
        paired_by_claim[claim] = paired
        animal_means = paired.groupby("animal_id", sort=True)["difference"].mean()
        low, high = _hierarchical_bootstrap(
            paired,
            n_bootstrap=n_bootstrap,
            seed=BOOTSTRAP_SEED + claim_index,
        )
        if animal_means.size and not np.allclose(animal_means.to_numpy(), 0.0):
            p_value = float(
                wilcoxon(animal_means.to_numpy(), alternative="two-sided").pvalue
            )
        else:
            p_value = 1.0
        n_paired_sessions = int(len(paired))
        n_failed_or_missing = max(0, n_planned_sessions - n_paired_sessions)
        if candidate_name == "learned_categorical_hmm":
            valid_column = "eligible_for_context_inference_support"
            if valid_column not in candidate_all.columns:
                valid_eids: set[str] = set()
            else:
                valid_eids = set(
                    candidate_all.loc[
                        candidate_all["status"].astype(str).eq("complete")
                        & candidate_all[valid_column].astype(str).eq("True"),
                        "eid",
                    ].astype(str)
                )
            n_valid_gate_sessions = len(valid_eids & set(planned["eid"].astype(str)))
            n_invalid_gate_sessions = max(0, n_planned_sessions - n_valid_gate_sessions)
            gate_fit_valid_rate = (
                n_valid_gate_sessions / n_planned_sessions
                if n_planned_sessions
                else float("nan")
            )
        else:
            n_valid_gate_sessions = n_planned_sessions
            n_invalid_gate_sessions = 0
            gate_fit_valid_rate = float("nan")
        rows.append(
            {
                "claim": claim,
                "candidate": candidate_name,
                "reference": "no_memory",
                "metric": metric,
                "difference_direction": "reference_minus_candidate_positive_is_better",
                "n_sessions": int(len(paired)),
                "n_animals": int(animal_means.size),
                "n_planned_sessions": n_planned_sessions,
                "n_planned_animals": n_planned_animals,
                "n_paired_complete_sessions": n_paired_sessions,
                "n_failed_or_missing_sessions": n_failed_or_missing,
                "n_valid_gate_sessions": n_valid_gate_sessions,
                "n_invalid_gate_sessions": n_invalid_gate_sessions,
                "gate_fit_valid_rate": gate_fit_valid_rate,
                "requires_valid_hmm_fit": requires_valid_hmm_fit,
                "source_run_status": source_run_status,
                "run_attempt_finalized": run_attempt_finalized,
                "animal_mean_difference": float(animal_means.mean())
                if animal_means.size
                else float("nan"),
                "hierarchical_bootstrap_ci_low": low,
                "hierarchical_bootstrap_ci_high": high,
                "animal_wilcoxon_two_sided_p": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_p"] = multipletests(
        summary["animal_wilcoxon_two_sided_p"].to_numpy(), method="holm"
    )[1]
    sufficient = (summary["n_sessions"] >= 20) & (summary["n_animals"] >= 5)
    complete_cohort = (
        (summary["n_planned_sessions"] > 0)
        & (summary["n_failed_or_missing_sessions"] == 0)
        & (summary["n_paired_complete_sessions"] == summary["n_planned_sessions"])
    )
    valid_gate = (~summary["requires_valid_hmm_fit"]) | (
        summary["n_invalid_gate_sessions"] == 0
    )
    inference_eligible = (
        sufficient & complete_cohort & valid_gate & run_attempt_finalized
    )
    summary["conclusion"] = "inconclusive"
    support = (
        inference_eligible
        & (summary["holm_p"] < 0.05)
        & (summary["hierarchical_bootstrap_ci_low"] > 0.0)
    )
    oppose = (
        inference_eligible
        & (summary["holm_p"] < 0.05)
        & (summary["hierarchical_bootstrap_ci_high"] < 0.0)
    )
    summary.loc[support, "conclusion"] = "support"
    summary.loc[oppose, "conclusion"] = "oppose"
    summary.loc[~sufficient, "conclusion"] = "inconclusive_insufficient_cohort"
    summary.loc[sufficient & ~complete_cohort, "conclusion"] = (
        "inconclusive_incomplete_cohort"
    )
    summary.loc[sufficient & complete_cohort & ~valid_gate, "conclusion"] = (
        "inconclusive_invalid_gate_fit"
    )
    summary.loc[~summary["run_attempt_finalized"], "conclusion"] = (
        "inconclusive_failed_or_unfinalized_attempt"
    )
    summary["cohort_complete_for_inference"] = complete_cohort
    summary["all_hmm_predictions_included_before_validity_gate"] = True
    summary["statistics_unit"] = "animal_primary_session_nested"
    summary["multiple_comparison_correction"] = "Holm_across_exp11_claim_family"
    summary["cohort_manifest_sha256"] = cohort_hash
    summary["evidence_scope"] = "IBL_trials_only_behavior_hidden_block_inference"
    summary["behavior_only_benchmark"] = True
    summary["neural_activity_analyzed"] = False
    summary["biological_mechanism_claim_eligible"] = False
    summary["shared_neural_dynamics_claim_eligible"] = False
    return summary, paired_by_claim


def _animal_values(paired: pd.DataFrame) -> np.ndarray:
    return paired.groupby("animal_id", sort=True)["difference"].mean().to_numpy(float)


def make_figure(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired: dict[str, pd.DataFrame],
    results_root: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    complete = frame.loc[frame["status"].astype(str).eq("complete")]
    condition_order = (
        ("no_memory", "Uniform belief"),
        ("exponential_history", "Exp. history"),
        ("learned_categorical_hmm", "Learned HMM"),
        ("oracle_ceiling", "Truth-access\noracle ceiling"),
    )
    values = [
        complete.loc[
            complete["condition"].astype(str).eq(condition), "context_nll"
        ].to_numpy(float)
        for condition, _ in condition_order
    ]
    boxes = axes[0].boxplot(
        values,
        tick_labels=[label for _, label in condition_order],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    rng = np.random.default_rng(41)
    for index, (box, observations) in enumerate(
        zip(boxes["boxes"], values, strict=True)
    ):
        box.set_facecolor(COLORS[index])
        box.set_alpha(0.25)
        axes[0].scatter(
            rng.normal(index + 1, 0.045, observations.size),
            observations,
            s=11,
            color=COLORS[index],
            alpha=0.55,
        )
    axes[0].axhline(np.log(2.0), color="0.35", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Held-out binary context NLL")
    axes[0].tick_params(axis="x", rotation=27)
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[0].text(
        0.98,
        0.98,
        "descriptive session distributions",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )

    panels = (
        (
            axes[1],
            ["history_context_nll_gain", "hmm_context_nll_gain"],
            "Context NLL Δ vs uniform belief",
        ),
        (
            axes[2],
            ["history_behavior_log_loss_gain", "hmm_behavior_log_loss_gain"],
            "Choice log-loss Δ vs no-belief history baseline",
        ),
    )
    for panel_index, (axis, claims, ylabel) in enumerate(panels, start=1):
        for index, claim in enumerate(claims):
            animal_values = _animal_values(paired[claim])
            axis.scatter(
                rng.normal(index + 1, 0.045, animal_values.size),
                animal_values,
                s=14,
                alpha=0.65,
                color=COLORS[index + 1],
            )
            if animal_values.size:
                row = summary.loc[summary["claim"].astype(str).eq(claim)].iloc[0]
                mean = float(row["animal_mean_difference"])
                low = float(row["hierarchical_bootstrap_ci_low"])
                high = float(row["hierarchical_bootstrap_ci_high"])
                axis.plot(
                    index + 1,
                    mean,
                    marker="D",
                    markersize=5,
                    color="black",
                )
                if np.isfinite([mean, low, high]).all():
                    axis.errorbar(
                        index + 1,
                        mean,
                        yerr=[[mean - low], [high - mean]],
                        color="black",
                        capsize=3,
                        linewidth=1.1,
                    )
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        axis.set_xticks([1, 2], ["Exp. history", "Learned HMM"], rotation=25)
        axis.set_ylabel(ylabel)
        axis.text(
            -0.16,
            1.03,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.text(
            0.98,
            0.98,
            "points: animal means\nsessions nested within animal",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
        )
    fig.tight_layout(w_pad=1.8)
    save_figure(fig, FIGURE_NAME, results_root)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--n-bootstrap", type=int, default=100_000)
    args = parser.parse_args()
    results_root = Path(args.results_root).resolve()
    frame = load_real_rows(results_root)
    frame.to_csv(
        results_root / "exp11_ibl_behavior_real_raw.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        lineterminator="\n",
    )
    summary, paired = real_data_comparison_summary(
        frame, n_bootstrap=int(args.n_bootstrap)
    )
    summary.to_csv(
        results_root / "exp11_ibl_behavior_real_summary.csv",
        index=False,
        lineterminator="\n",
    )
    make_figure(frame, summary, paired, results_root)


if __name__ == "__main__":
    main()
