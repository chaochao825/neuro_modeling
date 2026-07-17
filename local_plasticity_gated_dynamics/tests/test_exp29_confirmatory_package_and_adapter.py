"""Replay, fallback, and leakage tests for the Exp29 package/adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from experiments import exp29_confirmatory_source_panel as exp29
from experiments.common import load_json_config
from scripts import package_exp29_confirmatory_source_panel as packager
from src.analysis.actuator_manifest import manifest_hash
from src.data.actuator_selector_dataset import (
    build_frozen_selector_meta_training,
    load_exp26_selector_source,
)
from src.data.exp29_feasibility_selector_dataset import (
    build_confirmatory_selector_folds,
    confirmatory_source_from_package,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "formal" / "exp29_confirmatory_source_panel.json"
)
FORMAL_ROOT = (
    PROJECT_ROOT / "results" / "exp26_actuator_matching_formal_v2_e08beaf" / "formal"
)
FAKE_GIT = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}


def _runtime() -> dict[str, str | None]:
    return exp26.scientific_runtime_versions()


def _synthetic_collection() -> packager.PanelCollection:
    config = load_json_config(CONFIG_PATH)
    contract = exp29.validate_source_contract(
        config, current_git=FAKE_GIT, runtime_versions=_runtime()
    )
    provenance = exp29.build_evidence_provenance(
        contract, run_label=exp29.REQUIRED_RUN_LABEL
    )
    evidence = exp29.evidence_row_fields(provenance)
    cells = exp26._manifest(config)
    manifest_receipt = manifest_hash(cells)
    first_heldout = next(cell for cell in cells if cell.generator_split == "heldout")
    rows: list[dict[str, object]] = []
    attempts: list[packager.AttemptReceipt] = []
    for seed in exp29.EXPECTED_SEEDS:
        counts = {status: 0 for status in packager.ROW_STATUSES}
        for cell in cells:
            frozen_validation = 0.60 + 0.0001 * (seed - 60)
            frozen_test = 0.61 + 0.0001 * (seed - 60)
            frozen_fingerprint = f"frozen-{seed}-{cell.generator_id}"
            for mode_index, mode in enumerate(exp29.EXPECTED_MODES):
                infeasible = bool(
                    seed == 60
                    and cell.generator_id == first_heldout.generator_id
                    and mode == "routing"
                )
                status = "infeasible" if infeasible else "complete"
                validation = (
                    frozen_validation
                    if mode == "frozen" or infeasible
                    else 0.70 + 0.01 * mode_index
                )
                test = (
                    frozen_test
                    if mode == "frozen" or infeasible
                    else 0.71 + 0.01 * mode_index
                )
                active = mode != "frozen"
                feasible = not infeasible
                row: dict[str, object] = {
                    "run_id": f"fixture-{seed}",
                    "experiment": exp29.EXPERIMENT,
                    "seed": seed,
                    "profile": exp29.PROFILE,
                    "status": status,
                    "statistics_unit": "seed",
                    "recorded_at": "2026-07-17T00:00:00+00:00",
                    **exp29._dimensions(
                        cell,
                        mode=mode,
                        manifest_receipt=manifest_receipt,
                        evidence=evidence,
                    ),
                    "chi": float(cell.alpha),
                    "state_demand": 0.1 + 0.01 * cell.transition_rank,
                    "input_demand": 0.1 + 0.01 * cell.input_rank,
                    "dataset_fingerprint": f"dataset-{seed}-{cell.generator_id}",
                    "train_split_fingerprint": f"train-{seed}-{cell.generator_id}",
                    "validation_split_fingerprint": (
                        f"validation-{seed}-{cell.generator_id}"
                    ),
                    "test_split_fingerprint": f"test-{seed}-{cell.generator_id}",
                    "validation_balanced_accuracy": validation,
                    "test_balanced_accuracy": test,
                    "deployment_validation_balanced_accuracy": validation,
                    "deployment_test_balanced_accuracy": test,
                    "actuator_feasible": feasible,
                    "deployment_available": True,
                    "deployment_mode": "frozen" if infeasible else mode,
                    "deployment_fallback_mode": "frozen",
                    "deployment_fallback_applied": infeasible,
                    "matched_budget_support_eligible": active and feasible,
                    "functional_budget_valid": (not active) or feasible,
                    "effective_dynamics_strictly_stable": True,
                    "unconditional_cell_retained": True,
                    "correction_fingerprint": (
                        frozen_fingerprint
                        if mode == "frozen"
                        else f"{mode}-{seed}-{cell.generator_id}"
                    ),
                    "_attempt_path": f"/fixture/seed_{seed}",
                    "_run_status": "complete",
                    "_effective_status": status,
                }
                if infeasible:
                    row.update(
                        {
                            "infeasible_reason": "budget_scale_above_cap",
                            "fallback_frozen_validation_balanced_accuracy": frozen_validation,
                            "fallback_frozen_test_balanced_accuracy": frozen_test,
                            "fallback_frozen_correction_fingerprint": frozen_fingerprint,
                        }
                    )
                rows.append(row)
                counts[status] += 1
        attempts.append(
            packager.AttemptReceipt(
                seed=seed,
                path=f"/fixture/seed_{seed}",
                run_status="complete",
                run_id=f"fixture-{seed}",
                planned_coverage_valid=True,
                observed_row_count=exp29.EXPECTED_ROWS_PER_SEED,
                row_status_counts=counts,
                file_sha256={name: "c" * 64 for name in packager.ATTEMPT_FILES},
            )
        )
    for index, row in enumerate(rows):
        row["source_panel_row_index"] = index
    return packager.PanelCollection(
        rows=tuple(rows),
        attempts=tuple(attempts),
        config=config,
        config_sha256=contract.config_sha256,
        config_file_sha256=contract.config_file_sha256,
        source_contract=dict(config["source_binding"]),
        provenance_identity=provenance,
    )


@pytest.fixture(scope="module")
def packaged_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, packager.SourcePanelPackage]:
    directory = tmp_path_factory.mktemp("exp29-package") / "package"
    packager.write_source_panel_package(_synthetic_collection(), directory)
    package = packager.load_source_panel_package(directory)
    return directory, package


def _meta_training():
    source = load_exp26_selector_source(
        FORMAL_ROOT / "raw_metrics.csv.gz",
        FORMAL_ROOT / "conclusion.json",
        expected_profile="formal",
        expected_raw_sha256=(
            "b3ef5e22c241f832b1fd50254f87e3890ec45057bfeda3a784cbd218623a1193"
        ),
    )
    return build_frozen_selector_meta_training(source)


def _rewrite_hashes(directory: Path) -> None:
    raw_path = directory / "raw_metrics.jsonl"
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    receipt_path = directory / "source_panel_receipt.json"
    conclusion_path = directory / "conclusion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    receipt["raw_metrics_sha256"] = raw_sha
    conclusion["raw_metrics_sha256"] = raw_sha
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    payload_sha = exp29._canonical_sha256(payload)
    receipt["receipt_payload_sha256"] = payload_sha
    conclusion["source_panel_receipt_payload_sha256"] = payload_sha
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    conclusion_path.write_text(
        json.dumps(conclusion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_package_replays_complete_cartesian_feasibility_policy(
    packaged_source: tuple[Path, packager.SourcePanelPackage],
) -> None:
    _, package = packaged_source
    coverage = package.receipt["coverage"]
    assert coverage["source_panel_valid"] is True
    assert package.receipt["statistics_unit"] == "seed"
    assert coverage["observed_row_count"] == 30 * 88 * 5
    assert coverage["row_status_counts"] == {
        "complete": 13199,
        "failed": 0,
        "infeasible": 1,
        "invalid": 0,
    }
    assert coverage["matched_budget_support_row_count"] == 30 * 88 * 4 - 1
    assert coverage["infeasible_rate_by_seed_family"]["60"]["routing"] == pytest.approx(
        1.0 / 88.0
    )


def test_package_loader_rejects_semantic_tamper_even_after_rehash(
    packaged_source: tuple[Path, packager.SourcePanelPackage], tmp_path: Path
) -> None:
    source_dir, _ = packaged_source
    target = tmp_path / "tampered"
    shutil.copytree(source_dir, target)
    raw_path = target / "raw_metrics.jsonl"
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(rows):
        row = json.loads(line)
        if row["status"] == "infeasible":
            row["test_balanced_accuracy"] += 0.05
            rows[index] = json.dumps(row, sort_keys=True, separators=(",", ":"))
            break
    raw_path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    _rewrite_hashes(target)
    with pytest.raises(ValueError, match="coverage is not reproducible"):
        packager.load_source_panel_package(target)


def test_package_loader_rejects_statistics_unit_tamper_after_rehash(
    packaged_source: tuple[Path, packager.SourcePanelPackage], tmp_path: Path
) -> None:
    source_dir, _ = packaged_source
    target = tmp_path / "statistics-unit-tampered"
    shutil.copytree(source_dir, target)
    raw_path = target / "raw_metrics.jsonl"
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["statistics_unit"] = "neuron"
    rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    raw_path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    _rewrite_hashes(target)

    with pytest.raises(ValueError, match="statistics unit must be seed"):
        packager.load_source_panel_package(target)


def test_duplicate_seed_attempts_fail_before_favourable_selection(
    tmp_path: Path,
) -> None:
    config = load_json_config(CONFIG_PATH)
    base = tmp_path / "runs" / exp29.EXPERIMENT / "seed_0060"
    for name in ("attempt-a", "attempt-b"):
        attempt = base / name
        attempt.mkdir(parents=True)
        (attempt / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        attempt_config = {
            **config,
            "experiment": exp29.EXPERIMENT,
            "seed": 60,
            "run_label": exp29.REQUIRED_RUN_LABEL,
        }
        (attempt / "config.json").write_text(
            json.dumps(attempt_config), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="selective rerun is forbidden"):
        packager.collect_source_panel(
            tmp_path,
            config_path=CONFIG_PATH,
            current_git=FAKE_GIT,
            runtime_versions=_runtime(),
        )


def test_adapter_keeps_all_rows_and_scores_infeasible_choice_as_frozen(
    packaged_source: tuple[Path, packager.SourcePanelPackage],
) -> None:
    _, package = packaged_source
    source = confirmatory_source_from_package(package)
    assert source.statistics_unit == "seed"
    assert source.seeds.shape == (30 * 88,)
    assert int(np.sum(~source.candidate_feasible[:, 0])) == 1
    assert source.infeasible_rate_by_seed_family()[60]["routing"] == pytest.approx(
        1.0 / 88.0
    )
    meta = _meta_training()
    folds = build_confirmatory_selector_folds(meta, source)
    assert len(folds) == 30
    assert len({id(fold.meta_training) for fold in folds}) == 1
    fold = folds[0]
    assert fold.test_seed == 60
    assert fold.test_raw_features.shape[0] == 44
    infeasible_rows = np.flatnonzero(~fold.candidate_feasible[:, 0])
    assert infeasible_rows.size == 1
    choices = np.zeros(fold.test_raw_features.shape[0], dtype=np.int64)
    deployed = fold.deployment_utility(choices)
    row = int(infeasible_rows[0])
    assert deployed[row] == fold.frozen_test_utilities[row]
    assert fold.matched_budget_support_mask[row, 0] is np.False_
    oracle_indices, oracle_utilities = fold.oracle()
    assert oracle_indices[row] != 1
    assert oracle_utilities[row] >= fold.frozen_test_utilities[row]


def test_adapter_rejects_forged_package_object(
    packaged_source: tuple[Path, packager.SourcePanelPackage],
) -> None:
    _, package = packaged_source
    forged = packager.SourcePanelPackage(
        receipt=package.receipt,
        rows=package.rows,
        receipt_payload_sha256="0" * 64,
        receipt_file_sha256=package.receipt_file_sha256,
        conclusion_file_sha256=package.conclusion_file_sha256,
        raw_metrics_sha256=package.raw_metrics_sha256,
    )
    with pytest.raises(ValueError, match="integrity"):
        confirmatory_source_from_package(forged)


def test_adapter_independently_rejects_nonboolean_feasibility(
    packaged_source: tuple[Path, packager.SourcePanelPackage],
) -> None:
    _, package = packaged_source
    rows = [dict(row) for row in package.rows]
    target = next(
        row
        for row in rows
        if row["actuator_mode"] == "routing" and row["status"] == "complete"
    )
    target["actuator_feasible"] = "true"
    raw_sha = hashlib.sha256(packager._canonical_jsonl(rows)).hexdigest()
    receipt = dict(package.receipt)
    receipt["raw_metrics_sha256"] = raw_sha
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    receipt_sha = exp29._canonical_sha256(payload)
    receipt["receipt_payload_sha256"] = receipt_sha
    forged = packager.SourcePanelPackage(
        receipt=receipt,
        rows=tuple(rows),
        receipt_payload_sha256=receipt_sha,
        receipt_file_sha256=package.receipt_file_sha256,
        conclusion_file_sha256=package.conclusion_file_sha256,
        raw_metrics_sha256=raw_sha,
    )
    with pytest.raises(ValueError, match="status/feasibility"):
        confirmatory_source_from_package(forged)
