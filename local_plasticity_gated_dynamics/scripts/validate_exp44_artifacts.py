#!/usr/bin/env python3
"""Independently replay Exp44 artifact integrity and its registered decision.

This validator is intentionally downstream of the frozen experiment runner. It
does not import Exp44 decision code and cannot change a scientific artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACTORIZED = "factorized_local_em"
FIXED = "fixed_gain"
TOTAL = "total_uncertainty"
AUTOCOV = "autocovariance_qr"
PARTICLE = "hierarchical_particle"
ORACLE = "oracle_qr"
METHODS = (FIXED, TOTAL, FACTORIZED, AUTOCOV, PARTICLE, ORACLE)
BASELINES = (FIXED, TOTAL, AUTOCOV, PARTICLE, ORACLE)
METRICS = ("conditional_update_nll", "conditional_update_mse")
ATOL = 1e-11


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=ATOL):
        raise ValueError(f"{name} mismatch: {actual!r} != {expected!r}")


def _bootstrap_interval(
    values: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap input must be one finite participant vector")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        indices = rng.integers(0, len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _verify_hashes(result: Path, manifest: Mapping[str, Any]) -> None:
    required = {
        "candidate_scores.csv",
        "cell_metrics.csv",
        "comparisons.csv",
        "config.json",
        "environment.json",
        "participant_folds.csv",
        "participant_metrics.csv",
        "report.md",
        "run.log",
        "selected_candidates.json",
        "summary.json",
        "trace_diagnostics.csv",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("manifest does not cover every required raw artifact")
    for name, expected in artifacts.items():
        path = result / str(name)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")

    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("manifest source hashes are missing")
    for relative, expected in sources.items():
        path = PROJECT_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")

    protocol = PROJECT_ROOT / "docs" / "exp44_piray_daw_qr_behavior_protocol_20260730.md"
    if _sha256(protocol) != manifest.get("protocol_sha256"):
        raise ValueError("protocol hash mismatch")


def _verify_structure(
    participant: pd.DataFrame,
    cells: pd.DataFrame,
    folds: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping[str, Any],
) -> int:
    n_participants = int(config["data"]["expected_participants"])
    expected_ids = set(range(n_participants))
    if set(participant["participant_id"]) != expected_ids:
        raise ValueError("participant_metrics does not contain the expected people")
    if len(participant) != n_participants * len(METHODS):
        raise ValueError("participant_metrics row count mismatch")
    if participant.duplicated(["participant_id", "method"]).any():
        raise ValueError("duplicate participant-method rows")
    if set(participant["method"]) != set(METHODS):
        raise ValueError("participant_metrics method set mismatch")

    expected_blocks = int(config["data"]["expected_blocks"])
    if len(cells) != n_participants * len(METHODS) * expected_blocks:
        raise ValueError("cell_metrics row count mismatch")
    if cells.duplicated(["participant_id", "method", "block_id"]).any():
        raise ValueError("duplicate participant-method-block rows")

    if (
        len(folds) != n_participants
        or folds["participant_id"].nunique() != n_participants
    ):
        raise ValueError("participant fold coverage mismatch")
    expected_folds = set(range(int(config["cross_validation"]["participant_folds"])))
    if set(folds["fold"]) != expected_folds:
        raise ValueError("outer fold labels mismatch")

    if len(diagnostics) != len(METHODS) * expected_blocks:
        raise ValueError("trace diagnostic row count mismatch")
    if diagnostics.duplicated(["method", "block_id"]).any():
        raise ValueError("duplicate method-block diagnostics")
    return n_participants


def _replay_comparisons(
    participant: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    if len(comparisons) != len(METRICS) * len(BASELINES):
        raise ValueError("comparison table row count mismatch")
    if comparisons.duplicated(["metric", "baseline"]).any():
        raise ValueError("duplicate registered comparison")
    resamples = int(config["analysis"]["bootstrap_resamples"])
    bootstrap_seed = int(config["analysis"]["bootstrap_seed"])
    replayed_p: list[float] = []
    rows: list[pd.Series] = []
    for metric_index, metric in enumerate(METRICS):
        pivot = participant.pivot(
            index="participant_id", columns="method", values=metric
        )
        if set(pivot.columns) != set(METHODS) or pivot.isna().any().any():
            raise ValueError(f"incomplete participant matrix for {metric}")
        for baseline_index, baseline in enumerate(BASELINES):
            difference = (pivot[baseline] - pivot[FACTORIZED]).to_numpy()
            selected = comparisons.loc[
                (comparisons["metric"] == metric)
                & (comparisons["baseline"] == baseline)
            ]
            if len(selected) != 1:
                raise ValueError(f"missing comparison: {metric}/{baseline}")
            row = selected.iloc[0]
            _require_close(difference.mean(), row["mean_gain"], "mean gain")
            _require_close(np.median(difference), row["median_gain"], "median gain")
            if int(np.sum(difference > 0.0)) != int(row["positive_participants"]):
                raise ValueError("positive-participant count mismatch")
            low, high = _bootstrap_interval(
                difference,
                resamples=resamples,
                seed=bootstrap_seed
                + metric_index * len(BASELINES)
                + baseline_index,
            )
            _require_close(low, row["ci_low"], "bootstrap lower bound")
            _require_close(high, row["ci_high"], "bootstrap upper bound")
            if np.ptp(difference) <= np.finfo(np.float64).eps:
                p_value = 1.0 if difference[0] == 0.0 else 0.0
            else:
                p_value = float(ttest_1samp(difference, 0.0).pvalue)
            _require_close(p_value, row["p_raw"], "raw p-value")
            replayed_p.append(p_value)
            rows.append(row)
    adjusted = multipletests(replayed_p, alpha=0.05, method="holm")[1]
    for row, value in zip(rows, adjusted, strict=True):
        _require_close(value, row["p_holm"], "Holm-adjusted p-value")


def _comparison_row(
    comparisons: pd.DataFrame, metric: str, baseline: str
) -> pd.Series:
    selected = comparisons.loc[
        (comparisons["metric"] == metric) & (comparisons["baseline"] == baseline)
    ]
    if len(selected) != 1:
        raise ValueError(f"missing comparison: {metric}/{baseline}")
    return selected.iloc[0]


def _replay_decision(
    comparisons: pd.DataFrame,
    cells: pd.DataFrame,
    diagnostics: pd.DataFrame,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, bool]:
    analysis = config["analysis"]
    primary: dict[str, dict[str, Any]] = {}
    clauses: dict[str, bool] = {}
    for baseline in (FIXED, TOTAL):
        nll = _comparison_row(comparisons, METRICS[0], baseline)
        mse = _comparison_row(comparisons, METRICS[1], baseline)
        nll_pass = bool(
            nll["mean_gain"] >= float(analysis["minimum_nll_gain"])
            and nll["ci_low"] > 0.0
        )
        mse_pass = bool(
            mse["mean_gain"] > float(analysis["minimum_mse_gain"])
            and mse["ci_low"] > 0.0
        )
        clause_suffix = "fixed" if baseline == FIXED else baseline
        clauses[f"nll_gain_vs_{clause_suffix}"] = nll_pass
        clauses[f"mse_gain_vs_{clause_suffix}"] = mse_pass
        primary[baseline] = {
            "nll_mean_gain": float(nll["mean_gain"]),
            "nll_ci_low": float(nll["ci_low"]),
            "nll_passes": nll_pass,
            "mse_mean_gain": float(mse["mean_gain"]),
            "mse_ci_low": float(mse["ci_low"]),
            "mse_passes": mse_pass,
        }

    factor = diagnostics.loc[diagnostics["method"] == FACTORIZED]
    q_high = factor["true_process_variance"].max()
    q_low = factor["true_process_variance"].min()
    r_high = factor["true_observation_variance"].max()
    r_low = factor["true_observation_variance"].min()
    q_effect = float(
        factor.loc[factor["true_process_variance"] == q_high, "mean_gain"].mean()
        - factor.loc[factor["true_process_variance"] == q_low, "mean_gain"].mean()
    )
    r_effect = float(
        factor.loc[factor["true_observation_variance"] == r_low, "mean_gain"].mean()
        - factor.loc[factor["true_observation_variance"] == r_high, "mean_gain"].mean()
    )
    clauses["directional_qr_effects"] = bool(q_effect > 0.0 and r_effect > 0.0)

    pivot = cells.pivot_table(
        index=["participant_id", "block_id"],
        columns="method",
        values="conditional_update_nll",
    )
    cell_gain = (pivot[TOTAL] - pivot[FACTORIZED]).groupby("block_id").mean()
    clauses["cellwise_noninferiority_vs_total"] = bool(
        np.all(cell_gain.to_numpy() >= float(analysis["cell_noninferiority_margin"]))
    )

    factor_over_fixed = float(
        _comparison_row(comparisons, METRICS[0], FIXED)["mean_gain"]
    )
    particle_vs_factor = float(
        _comparison_row(comparisons, METRICS[0], PARTICLE)["mean_gain"]
    )
    particle_over_fixed = factor_over_fixed - particle_vs_factor
    if particle_over_fixed > 0.0:
        retention: float | None = factor_over_fixed / particle_over_fixed
        retention_pass = bool(
            retention >= float(analysis["particle_gain_retention"])
        )
        retention_applicable = True
    else:
        retention = None
        retention_pass = True
        retention_applicable = False
    clauses["particle_gain_retention"] = retention_pass

    passed = bool(all(clauses.values()))
    if dict(summary["clauses"]) != clauses:
        raise ValueError("registered gate clauses do not replay")
    if bool(summary["development_gate_passed"]) != passed:
        raise ValueError("development gate does not replay")
    if summary["conclusion"] != ("support" if passed else "oppose"):
        raise ValueError("scientific conclusion does not replay")
    if bool(summary["confirmation_unlocked"]) != passed:
        raise ValueError("Experiment 2 lock does not replay")
    if bool(summary["popgym_unlocked"]):
        raise ValueError("POPGym cannot be unlocked by Experiment 1")
    if bool(summary["claim_upgrade_allowed"]):
        raise ValueError("development evidence cannot upgrade a claim")
    if summary["independent_statistical_unit"] != "participant":
        raise ValueError("inferential unit is not participant")

    _require_close(q_effect, summary["q_gain_effect"], "Q gain effect")
    _require_close(r_effect, summary["r_gain_effect"], "R gain effect")
    if bool(summary["particle_gain_retention_applicable"]) != retention_applicable:
        raise ValueError("particle-retention applicability mismatch")
    if retention is None:
        if summary["particle_gain_retention"] is not None:
            raise ValueError("particle-retention value should be null")
    else:
        _require_close(retention, summary["particle_gain_retention"], "particle retention")
    for baseline, expected in primary.items():
        actual = summary["primary_comparisons"][baseline]
        if actual.keys() != expected.keys():
            raise ValueError(f"primary comparison fields mismatch: {baseline}")
        for key, value in expected.items():
            if isinstance(value, bool):
                if bool(actual[key]) != value:
                    raise ValueError(f"primary decision mismatch: {baseline}/{key}")
            else:
                _require_close(value, actual[key], f"primary {baseline}/{key}")
    summary_cells = summary["cell_gain_total_minus_factorized"]
    if set(summary_cells) != {str(int(index)) for index in cell_gain.index}:
        raise ValueError("cell-gain key set mismatch")
    for block, value in cell_gain.items():
        _require_close(value, summary_cells[str(int(block))], "cell NLL gain")
    return clauses


def validate_exp44_artifacts(result: Path) -> dict[str, Any]:
    result = result.resolve()
    if not result.is_dir():
        raise FileNotFoundError(result)
    manifest = _read_json(result / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Exp44 manifest is not complete")
    _verify_hashes(result, manifest)

    config = _read_json(result / "config.json")
    summary = _read_json(result / "summary.json")
    environment = _read_json(result / "environment.json")
    if config.get("stage") != "development" or config["data"].get("experiment") != 1:
        raise ValueError(
            "artifact is not the registered Experiment 1 development run"
        )
    if environment.get("git", {}).get("dirty") is not False:
        raise ValueError("run did not start from a clean worktree")
    commit = str(environment.get("git", {}).get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("environment lacks an immutable git commit")

    participant = pd.read_csv(result / "participant_metrics.csv")
    cells = pd.read_csv(result / "cell_metrics.csv")
    folds = pd.read_csv(result / "participant_folds.csv")
    comparisons = pd.read_csv(result / "comparisons.csv")
    diagnostics = pd.read_csv(result / "trace_diagnostics.csv")
    n_participants = _verify_structure(
        participant, cells, folds, diagnostics, config
    )
    _replay_comparisons(participant, comparisons, config)
    clauses = _replay_decision(comparisons, cells, diagnostics, summary, config)
    return {
        "status": "pass",
        "validation_scope": "postoutcome_independent_artifact_and_decision_replay",
        "manifest_sha256": _sha256(result / "manifest.json"),
        "frozen_commit": commit,
        "n_participants": n_participants,
        "participant_metric_rows": int(len(participant)),
        "cell_metric_rows": int(len(cells)),
        "artifact_hashes_replayed": int(len(manifest["artifacts"])),
        "comparison_rows_replayed": int(len(comparisons)),
        "clauses": clauses,
        "development_gate_passed": bool(summary["development_gate_passed"]),
        "conclusion": str(summary["conclusion"]),
        "experiment2_locked": not bool(summary["confirmation_unlocked"]),
        "popgym_locked": not bool(summary["popgym_unlocked"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = validate_exp44_artifacts(args.result)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_name(f".{args.receipt.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.receipt)
    print(payload, end="")


if __name__ == "__main__":
    main()
