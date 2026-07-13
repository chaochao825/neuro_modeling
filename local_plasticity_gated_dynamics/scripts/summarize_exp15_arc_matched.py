"""Validate and publish the additive Exp15 ARC matched-compute audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp15_task_specialized_reasoning import (  # noqa: E402
    _bootstrap_accuracy,
    _paired_group_comparison,
)
from src.utils.reproducibility import derive_seed  # noqa: E402


PREFIX = "exp15_arc_matched_formal"
ARC_CONDITIONS = ("arc_slow_fast_program", "arc_flat_program_matched")
ARC_COMPARISON = "slow_fast_vs_flat_matched"
REGISTERED_FORMAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "formal" / "exp15_task_specialized_arc.json"
)
REGISTERED_FORMAL_CONFIG_SHA256 = (
    "cbf7bf6ac6bf7fd77e522bd70aac93b535370d80740b97c539e799885fbbbed2"
)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Exp15 metrics must be a non-empty JSONL object stream")
    return rows


def _current_git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    return commit, bool(status.strip())


def _registered_run_config(seed: int) -> dict[str, object]:
    if _sha256(REGISTERED_FORMAL_CONFIG_PATH) != REGISTERED_FORMAL_CONFIG_SHA256:
        raise ValueError("registered Exp15 ARC formal config hash differs from code")
    registered = _json(REGISTERED_FORMAL_CONFIG_PATH)
    if not isinstance(registered, dict) or seed not in registered.get("seeds", []):
        raise ValueError("registered Exp15 ARC formal config does not contain run seed")
    return {
        "experiment": "exp15_task_specialized_reasoning",
        "seed": seed,
        **registered,
        "config_path": str(REGISTERED_FORMAL_CONFIG_PATH.resolve()),
        "training_algorithm": "target_free_task_specialized_dynamics",
        "used_autograd": False,
        "used_bptt": False,
        "spiking_required": False,
        "reference_scope": "task_design_only_not_bdh_or_hrm_reimplementation",
    }


def _assert_close(observed: object, expected: object, *, label: str) -> None:
    if not np.isclose(
        float(observed), float(expected), rtol=1e-12, atol=1e-12, equal_nan=True
    ):
        raise ValueError(f"{label} differs from raw recomputation")


def _normalized_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _assert_published_rows_match_raw(
    published: pd.DataFrame,
    raw_rows: list[dict[str, Any]],
    *,
    key: str,
) -> None:
    if len(published) != len(raw_rows) or published[key].astype(str).duplicated().any():
        raise ValueError(f"Exp15 published {key} row family is malformed")
    raw_by_key = {str(row[key]): row for row in raw_rows}
    if set(published[key].astype(str)) != set(raw_by_key):
        raise ValueError(f"Exp15 published {key} row family differs from raw")
    for observed in published.to_dict("records"):
        expected = raw_by_key[str(observed[key])]
        for column, expected_value in expected.items():
            if column not in observed:
                raise ValueError(f"Exp15 published table lacks raw column {column}")
            observed_value = _normalized_scalar(observed[column])
            expected_value = _normalized_scalar(expected_value)
            if (
                isinstance(expected_value, (int, float))
                and not isinstance(expected_value, bool)
                and isinstance(observed_value, (int, float))
                and not isinstance(observed_value, bool)
            ):
                if not np.isclose(
                    float(observed_value),
                    float(expected_value),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Exp15 published {key} column {column} differs from raw"
                    )
            elif observed_value != expected_value:
                raise ValueError(
                    f"Exp15 published {key} column {column} differs from raw"
                )


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    names = (
        "config.json",
        "environment.json",
        "manifest.json",
        "metrics.jsonl",
        "planned_conditions.json",
        "run.log",
        "source_provenance.json",
        "status.json",
    )
    missing = [name for name in names if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Exp15 formal run lacks artifacts: {missing}")
    return {
        name.replace(".", "_") + "_sha256": _sha256(run_dir / name) for name in names
    }


def validate_formal_run(
    run_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Recompute all published Exp15 ARC rows from one immutable run."""

    path = Path(run_dir)
    hashes = _artifact_hashes(path)
    config = _json(path / "config.json")
    environment = _json(path / "environment.json")
    manifest = _json(path / "manifest.json")
    planned = _json(path / "planned_conditions.json")
    status = _json(path / "status.json")
    provenance = _json(path / "source_provenance.json")
    rows = _records(path / "metrics.jsonl")

    seed = int(config.get("seed", -1))
    if config != _registered_run_config(seed):
        raise ValueError(
            "run config differs from the registered Exp15 ARC formal config"
        )
    if (
        config.get("experiment") != "exp15_task_specialized_reasoning"
        or config.get("profile") != "formal"
        or config.get("family") != "arc"
        or tuple(config.get("conditions", ())) != ARC_CONDITIONS
    ):
        raise ValueError("run config is not the registered Exp15 ARC formal panel")
    run_id = str(manifest.get("run_id", ""))
    if (
        status.get("status") != "complete"
        or int(status.get("condition_failures", -1)) != 0
        or int(status.get("condition_invalid", -1)) != 0
        or manifest.get("status") != "complete"
        or int(manifest.get("seed", -1)) != seed
        or manifest.get("profile") != "formal"
    ):
        raise ValueError("Exp15 ARC formal run is incomplete")
    git_receipt = environment.get("git")
    if not isinstance(git_receipt, Mapping):
        raise ValueError("Exp15 ARC environment lacks nested git provenance")
    git_commit = str(git_receipt.get("commit", ""))
    current_commit, current_dirty = _current_git_state()
    if (
        _GIT_COMMIT.fullmatch(git_commit) is None
        or bool(git_receipt.get("dirty", True))
        or current_dirty
        or git_commit != current_commit
    ):
        raise ValueError("Exp15 ARC formal publication requires a clean commit")
    planned_conditions = tuple(row.get("condition") for row in planned)
    if planned_conditions != ARC_CONDITIONS:
        raise ValueError("Exp15 planned condition panel differs from config")
    if any(
        row.get("run_id") != run_id
        or int(row.get("seed", -1)) != seed
        or row.get("experiment") != "exp15_task_specialized_reasoning"
        for row in rows
    ):
        raise ValueError("Exp15 metric identity differs from the run manifest")
    failed = [row for row in rows if row.get("status") in {"failed", "invalid"}]
    if failed:
        raise ValueError("Exp15 formal metrics retain failed/invalid records")

    task_rows = [row for row in rows if row.get("stage") == "task_test"]
    aggregates = [row for row in rows if row.get("stage") == "aggregate"]
    comparisons = [row for row in rows if row.get("stage") == "comparison"]
    if (
        {row.get("condition") for row in aggregates} != set(ARC_CONDITIONS)
        or len(aggregates) != len(ARC_CONDITIONS)
        or len(comparisons) != 1
        or comparisons[0].get("comparison") != ARC_COMPARISON
    ):
        raise ValueError("Exp15 formal derived row family is incomplete")
    task_keys = [(row.get("condition"), row.get("task_id")) for row in task_rows]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Exp15 formal task panel contains duplicates")
    by_condition = {
        condition: [row for row in task_rows if row.get("condition") == condition]
        for condition in ARC_CONDITIONS
    }
    task_id_sets = [
        {str(row["task_id"]) for row in values} for values in by_condition.values()
    ]
    if not task_id_sets[0] or task_id_sets[0] != task_id_sets[1]:
        raise ValueError("Exp15 formal conditions do not share an identical task panel")
    provenance_fingerprints = provenance.get("test_task_fingerprints")
    if (
        not isinstance(provenance_fingerprints, Mapping)
        or int(provenance.get("n_test_tasks", -1)) != len(task_id_sets[0])
        or len(task_id_sets[0]) < int(config["minimum_test_tasks"])
        or set(str(key) for key in provenance_fingerprints) != task_id_sets[0]
    ):
        raise ValueError("Exp15 formal task panel differs from source provenance")
    for condition, condition_task_rows in by_condition.items():
        for row in condition_task_rows:
            if str(row["public_fingerprint"]) != str(
                provenance_fingerprints[str(row["task_id"])]
            ):
                raise ValueError(
                    f"{condition} public task fingerprint differs from provenance"
                )

    source_receipt = provenance.get("source_manifest_receipt")
    acquisition_receipt = provenance.get("source_acquisition_receipt")
    data_config = config.get("data")
    if (
        not isinstance(source_receipt, Mapping)
        or not isinstance(acquisition_receipt, Mapping)
        or not isinstance(data_config, Mapping)
        or provenance.get("source_manifest_verified") is not True
        or source_receipt.get("source_tree_verified") is not True
        or source_receipt.get("manifest_sha256") != data_config.get("manifest_sha256")
        or source_receipt.get("license_sha256") != data_config.get("license_sha256")
        or source_receipt.get("split_counts")
        != data_config.get("expected_split_counts")
        or acquisition_receipt.get("source_acquisition_verified") is not True
        or acquisition_receipt.get("acquisition_manifest_sha256")
        != data_config.get("acquisition_manifest_sha256")
        or acquisition_receipt.get("validation_sha256")
        != data_config.get("validation_receipt_sha256")
        or acquisition_receipt.get("source_manifest_sha256")
        != data_config.get("manifest_sha256")
        or provenance.get("license_status") != "verified"
        or provenance.get("test_split_role") != "ood"
    ):
        raise ValueError("Exp15 ARC source provenance is not fully verified")

    aggregate_by_condition = {str(row["condition"]): row for row in aggregates}
    condition_rows: list[dict[str, object]] = []
    for condition in ARC_CONDITIONS:
        raw_condition = by_condition[condition]
        aggregate = aggregate_by_condition[condition]
        expected_accuracy_seed = derive_seed(
            seed, "exp15", "arc", condition, "bootstrap"
        )
        expected_functional_seed = derive_seed(
            seed, "exp15", "arc", condition, "functional_bootstrap"
        )
        if (
            int(aggregate["n_bootstrap"]) != int(config["n_bootstrap"])
            or int(aggregate["accuracy_bootstrap_seed"]) != expected_accuracy_seed
            or int(aggregate["functional_bootstrap_seed"]) != expected_functional_seed
            or int(aggregate["n_tasks"]) != len(raw_condition)
            or int(aggregate["n_independent_source_groups"])
            != len({str(row["source_group"]) for row in raw_condition})
        ):
            raise ValueError(f"{condition} aggregate panel/bootstrap contract differs")
        exact = np.asarray([bool(row["exact"]) for row in raw_condition], dtype=float)
        groups = tuple(str(row["source_group"]) for row in raw_condition)
        estimate, low, high = _bootstrap_accuracy(
            exact,
            groups,
            n_bootstrap=int(aggregate["n_bootstrap"]),
            seed=int(aggregate["accuracy_bootstrap_seed"]),
        )
        functional_estimate, functional_low, functional_high = _bootstrap_accuracy(
            exact,
            groups,
            n_bootstrap=int(aggregate["n_bootstrap"]),
            seed=int(aggregate["functional_bootstrap_seed"]),
        )
        for label, observed, expected in (
            ("exact accuracy", aggregate["exact_accuracy"], estimate),
            ("exact CI low", aggregate["exact_accuracy_ci_low"], low),
            ("exact CI high", aggregate["exact_accuracy_ci_high"], high),
            (
                "functional success",
                aggregate["functional_success_rate"],
                functional_estimate,
            ),
            (
                "functional CI low",
                aggregate["functional_success_ci_low"],
                functional_low,
            ),
            (
                "functional CI high",
                aggregate["functional_success_ci_high"],
                functional_high,
            ),
            (
                "candidate coverage",
                aggregate["candidate_coverage"],
                np.mean([bool(row["candidate_covered"]) for row in raw_condition]),
            ),
            (
                "mean measured compute",
                aggregate["mean_measured_compute_units"],
                np.mean(
                    [float(row["measured_compute_units"]) for row in raw_condition]
                ),
            ),
            (
                "mean charged compute",
                aggregate["mean_charged_compute_units"],
                np.mean([float(row["charged_compute_units"]) for row in raw_condition]),
            ),
        ):
            _assert_close(observed, expected, label=f"{condition} {label}")
        if aggregate.get("formal_data_eligible") is not True:
            raise ValueError("Exp15 ARC aggregate unexpectedly fails data eligibility")
        condition_rows.append(dict(aggregate))

    observed_comparison = comparisons[0]
    expected_comparison_seed = derive_seed(
        seed, "exp15", "arc", ARC_COMPARISON, "paired_bootstrap"
    )
    if (
        int(observed_comparison["n_bootstrap"]) != int(config["n_bootstrap"])
        or int(observed_comparison["bootstrap_seed"]) != expected_comparison_seed
    ):
        raise ValueError("Exp15 paired bootstrap contract differs from registration")
    recomputed = _paired_group_comparison(
        by_condition[ARC_CONDITIONS[0]],
        by_condition[ARC_CONDITIONS[1]],
        n_bootstrap=int(observed_comparison["n_bootstrap"]),
        seed=int(observed_comparison["bootstrap_seed"]),
    )
    for key in (
        "estimate",
        "ci_low",
        "ci_high",
        "wilcoxon_p",
        "wilcoxon_p_holm",
        "candidate_coverage",
    ):
        _assert_close(
            observed_comparison[key], recomputed[key], label=f"comparison {key}"
        )
    for key in (
        "n_independent_source_groups",
        "n_nonzero_source_groups",
        "candidate_fingerprints_matched",
        "candidate_coverage_matched",
        "charged_compute_matched",
    ):
        if observed_comparison[key] != recomputed[key]:
            raise ValueError(f"comparison {key} differs from raw recomputation")
    comparison_config = config.get("registered_comparison")
    if not isinstance(comparison_config, Mapping):
        raise ValueError("Exp15 formal comparison registration is missing")
    minimum_coverage = float(comparison_config["minimum_candidate_coverage"])
    alpha = float(comparison_config["alpha"])
    expected_eligibility = bool(
        float(recomputed["candidate_coverage"]) >= minimum_coverage
        and recomputed["candidate_fingerprints_matched"]
        and recomputed["candidate_coverage_matched"]
        and recomputed["charged_compute_matched"]
    )
    if bool(observed_comparison["core_claim_eligible"]) != expected_eligibility:
        raise ValueError("Exp15 comparison claim gate differs from recomputation")
    if bool(observed_comparison["coverage_gate_passed"]) != bool(
        float(recomputed["candidate_coverage"]) >= minimum_coverage
    ) or not np.isclose(
        float(observed_comparison["minimum_candidate_coverage"]),
        minimum_coverage,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Exp15 comparison coverage gate differs from registration")
    if (
        observed_comparison.get("formal_data_eligible") is not True
        or observed_comparison.get("registered_ood_split") is not True
        or observed_comparison.get("matched_advantage_comparator_registered")
        is not True
    ):
        raise ValueError("Exp15 comparison provenance gates are not satisfied")
    if (
        expected_eligibility
        and float(recomputed["wilcoxon_p_holm"]) < alpha
        and float(recomputed["ci_low"]) > 0.0
    ):
        expected_conclusion = "support"
    elif (
        expected_eligibility
        and float(recomputed["wilcoxon_p_holm"]) < alpha
        and float(recomputed["ci_high"]) < 0.0
    ):
        expected_conclusion = "oppose"
    else:
        expected_conclusion = "inconclusive"
    if observed_comparison.get("claim_conclusion") != expected_conclusion:
        raise ValueError("Exp15 comparison conclusion differs from recomputation")

    raw = pd.DataFrame(rows)
    conditions = pd.DataFrame(condition_rows)
    comparison = pd.DataFrame([dict(observed_comparison)])
    run_manifest = {
        "family": "arc",
        "run_id": run_id,
        "attempt_name": path.name,
        "status": "complete",
        "seed": seed,
        "git_commit": git_commit,
        "git_dirty": False,
        "registered_formal_config_sha256": REGISTERED_FORMAL_CONFIG_SHA256,
        "source_manifest_sha256": str(data_config["manifest_sha256"]),
        "source_license_sha256": str(data_config["license_sha256"]),
        "source_tree_verified": True,
        "metrics_row_count": len(rows),
        **hashes,
    }
    return conditions, comparison, raw, run_manifest


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(
            f"refusing to overwrite a different published artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def publish_snapshot(
    run_dir: str | Path, results_root: str | Path
) -> Mapping[str, Path]:
    conditions, comparison, _raw, manifest_row = validate_formal_run(run_dir)
    root = Path(results_root)
    raw_path = root / f"{PREFIX}_raw.jsonl"
    raw_payload = (Path(run_dir) / "metrics.jsonl").read_bytes()
    _write_new(raw_path, raw_payload)
    manifest_row["published_raw_sha256"] = hashlib.sha256(raw_payload).hexdigest()
    run_manifest_path = root / f"{PREFIX}_run_manifest.csv"
    run_manifest_payload = _csv_bytes(pd.DataFrame([manifest_row]))
    _write_new(run_manifest_path, run_manifest_payload)
    run_manifest_sha = hashlib.sha256(run_manifest_payload).hexdigest()
    raw_sha = str(manifest_row["published_raw_sha256"])
    for frame in (conditions, comparison):
        frame["published_raw_sha256"] = raw_sha
        frame["published_run_manifest_sha256"] = run_manifest_sha
    conditions_path = root / f"{PREFIX}_conditions.csv"
    comparison_path = root / f"{PREFIX}_comparison.csv"
    _write_new(conditions_path, _csv_bytes(conditions))
    _write_new(comparison_path, _csv_bytes(comparison))

    indexed = conditions.set_index("condition")
    comp = comparison.iloc[0]
    report = f"""# Exp15 ARC verified-source matched-compute audit

This additive run used clean commit `{manifest_row["git_commit"]}` and verified all 800 ARC-AGI-1 JSON files plus the Apache-2.0 license against the reviewed per-file source manifest `{manifest_row["source_manifest_sha256"]}`. The separate acquisition and validation receipts were also verified. Query targets were used only by the held-out scorer and candidate-coverage diagnostic.

## Absolute task performance

- Slow/fast selector: {100 * float(indexed.loc["arc_slow_fast_program", "exact_accuracy"]):.4f}% exact (95% source-group CI {100 * float(indexed.loc["arc_slow_fast_program", "exact_accuracy_ci_low"]):.4f}%–{100 * float(indexed.loc["arc_slow_fast_program", "exact_accuracy_ci_high"]):.4f}%).
- Flat matched selector: {100 * float(indexed.loc["arc_flat_program_matched", "exact_accuracy"]):.4f}% exact (95% source-group CI {100 * float(indexed.loc["arc_flat_program_matched", "exact_accuracy_ci_low"]):.4f}%–{100 * float(indexed.loc["arc_flat_program_matched", "exact_accuracy_ci_high"]):.4f}%).

## Registered paired comparison

The slow/fast-minus-flat exact-accuracy difference is {100 * float(comp["estimate"]):.4f} percentage points (95% paired source-group bootstrap CI {100 * float(comp["ci_low"]):.4f} to {100 * float(comp["ci_high"]):.4f}; Holm p={float(comp["wilcoxon_p_holm"]):.4g}). Candidate fingerprints and charged compute are matched. Compute is an audited abstract operation proxy, not FLOPs, wall-clock time, or energy. Candidate coverage is only {100 * float(comp["candidate_coverage"]):.4f}% versus the preregistered {100 * float(comp["minimum_candidate_coverage"]):.1f}% gate, so `core_claim_eligible={str(bool(comp["core_claim_eligible"])).lower()}` and the conclusion is **{comp["claim_conclusion"]}**.

## Interpretation

The source-provenance defect is repaired, but source eligibility alone does not create evidence for hierarchical advantage. The finite proposal library remains the limiting factor; low matrix/state dimension and task-specific architecture are not counted as support without held-out behavioral gain.

Published raw SHA-256: `{raw_sha}`. Run-manifest SHA-256: `{run_manifest_sha}`.
"""
    report_path = root / f"{PREFIX}_report.md"
    _write_new(report_path, report.encode("utf-8"))
    return {
        "raw": raw_path,
        "run_manifest": run_manifest_path,
        "conditions": conditions_path,
        "comparison": comparison_path,
        "report": report_path,
    }


def load_published_snapshot(
    results_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(results_root)
    raw_path = root / f"{PREFIX}_raw.jsonl"
    manifest_path = root / f"{PREFIX}_run_manifest.csv"
    conditions = pd.read_csv(root / f"{PREFIX}_conditions.csv")
    comparison = pd.read_csv(root / f"{PREFIX}_comparison.csv")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 1:
        raise ValueError("Exp15 published run manifest must contain one run")
    raw_sha = _sha256(raw_path)
    manifest_sha = _sha256(manifest_path)
    if str(manifest.iloc[0]["published_raw_sha256"]) != raw_sha:
        raise ValueError("Exp15 published raw differs from its run manifest")
    for frame in (conditions, comparison):
        if not frame["published_raw_sha256"].astype(str).eq(raw_sha).all():
            raise ValueError("Exp15 derived table differs from published raw binding")
        if (
            not frame["published_run_manifest_sha256"]
            .astype(str)
            .eq(manifest_sha)
            .all()
        ):
            raise ValueError("Exp15 derived table differs from run-manifest binding")
    raw_records = _records(raw_path)
    _assert_published_rows_match_raw(
        conditions,
        [row for row in raw_records if row.get("stage") == "aggregate"],
        key="condition",
    )
    _assert_published_rows_match_raw(
        comparison,
        [row for row in raw_records if row.get("stage") == "comparison"],
        key="comparison",
    )
    return conditions, comparison, manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    outputs = publish_snapshot(args.run_dir, args.results_root)
    from figures.exp15_arc_matched_plot import plot_exp15_arc_matched

    plot_exp15_arc_matched(Path(args.results_root))
    print(
        json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
