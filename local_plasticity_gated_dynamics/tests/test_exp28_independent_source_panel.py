"""Focused contracts for the hash-locked independent Exp26 source panel."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from experiments import exp28_exp26_independent_source_panel as independent
from experiments.common import load_json_config
from scripts import package_exp28_independent_source_panel as packager
from src.analysis.actuator_manifest import GeneratorCell
from src.utils import artifacts
from src.utils.artifacts import ExperimentRun


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "formal"
    / "exp28_exp26_independent_source_panel.json"
)
FAKE_GIT = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}


def _config() -> dict[str, object]:
    return load_json_config(CONFIG_PATH)


def _runtime() -> dict[str, str | None]:
    return exp26.scientific_runtime_versions()


def _contract() -> independent.SourceContract:
    return independent.validate_source_contract(
        _config(), current_git=FAKE_GIT, runtime_versions=_runtime()
    )


def _cell() -> GeneratorCell:
    return GeneratorCell(
        generator_id="fixture-cell",
        generator_split="heldout",
        alpha=0.5,
        transition_rank=2,
        input_rank=2,
        delay=4,
        noise_std=0.3,
        rotation_seed=281,
    )


def _rewrite_package_metadata(
    package: Path,
    receipt: dict[str, object],
    conclusion: dict[str, object],
) -> None:
    payload = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    receipt_sha = independent._canonical_sha256(payload)
    receipt["receipt_payload_sha256"] = receipt_sha
    conclusion["coverage"] = receipt["coverage"]
    conclusion["source_panel_valid"] = receipt["coverage"]["source_panel_valid"]
    conclusion["raw_metrics_sha256"] = receipt["raw_metrics_sha256"]
    conclusion["source_panel_receipt_payload_sha256"] = receipt_sha
    (package / "source_panel_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (package / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_formal_independent_config_copies_and_binds_exp26() -> None:
    config = _config()
    contract = independent.validate_source_contract(
        config, current_git=FAKE_GIT, runtime_versions=_runtime()
    )
    assert config["profile"] == "independent_test"
    assert tuple(config["seeds"]) == tuple(range(30, 60))
    assert len(independent.planned_conditions(config)) == 88 * 5
    assert config["actuator"]["max_scale"] == 128.0
    assert contract.source_config_sha256 == (
        "07ad3f16d9de6b5906155d95f215e9434e478ca992fd023adfabcd21a0005ecf"
    )
    assert contract.source_manifest_sha256 == (
        "a1c17a1e88c731f6678760865cf51d7236ae771bf839645c401e5cff8798ebfa"
    )
    assert contract.source_preflight_receipt_sha256 == (
        "bad665691233c9611fcdcce897c642d517a938b78adbabadee783c5e8cb1a671"
    )
    assert contract.source_critical_code_sha256 == (
        "7c593319af7e456ba6a8754965a90b6390da57994f28ccc23dcf93713a10157d"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seeds", list(range(31, 61)), "seeds 30--59"),
        ("actuator.max_scale", 127.0, "actuator differs"),
        (
            "source_binding.source_manifest_sha256",
            "0" * 64,
            "manifest hash",
        ),
    ],
)
def test_source_contract_rejects_panel_or_hash_tampering(
    field: str,
    value: object,
    message: str,
) -> None:
    config = json.loads(json.dumps(_config()))
    target: dict[str, object] = config
    parts = field.split(".")
    for part in parts[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[parts[-1]] = value
    with pytest.raises(ValueError, match=message):
        independent.validate_source_contract(
            config, current_git=FAKE_GIT, runtime_versions=_runtime()
        )


def test_source_contract_rejects_dirty_or_non_python311_runtime() -> None:
    with pytest.raises(ValueError, match="clean identifiable"):
        independent.validate_source_contract(
            _config(),
            current_git={**FAKE_GIT, "dirty": True},
            runtime_versions=_runtime(),
        )
    runtime = dict(_runtime())
    runtime["python"] = "3.12.0"
    with pytest.raises(ValueError, match="Python 3.11"):
        independent.validate_source_contract(
            _config(), current_git=FAKE_GIT, runtime_versions=runtime
        )


def test_runner_retains_every_mode_when_carrier_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = _cell()
    contract = _contract()
    monkeypatch.setattr(
        independent, "validate_source_contract", lambda config: contract
    )
    monkeypatch.setattr(exp26, "_manifest", lambda config: (cell,))
    monkeypatch.setattr(
        exp26,
        "_planned_conditions",
        lambda config: [
            {
                "generator_id": cell.generator_id,
                "generator_split": cell.generator_split,
                "alpha": cell.alpha,
                "transition_rank": cell.transition_rank,
                "input_rank": cell.input_rank,
                "delay": cell.delay,
                "noise_std": cell.noise_std,
                "rotation_seed": cell.rotation_seed,
                "actuator_mode": mode,
                "condition": mode,
                "manifest_hash": "fixture",
            }
            for mode in independent.EXPECTED_MODES
        ],
    )

    def fail_carrier(*args: object, **kwargs: object) -> object:
        raise RuntimeError("carrier fixture failure")

    monkeypatch.setattr(exp26, "make_carrier", fail_carrier)
    monkeypatch.setattr(exp26, "git_identity", lambda: dict(FAKE_GIT))
    monkeypatch.setattr(
        exp26,
        "scientific_runtime_versions",
        lambda: dict(contract.runtime_versions),
    )
    path = independent.run_seed(
        {"profile": "independent_test"},
        30,
        tmp_path,
        run_label=independent.REQUIRED_RUN_LABEL,
    )
    rows = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert len(rows) == len(independent.EXPECTED_MODES)
    assert {row["actuator_mode"] for row in rows} == set(
        independent.EXPECTED_MODES
    )
    assert all(row["status"] == "failed" for row in rows)
    assert status["status"] == "complete_with_failures"
    assert status["condition_failures"] == len(independent.EXPECTED_MODES)


def _complete_collection() -> packager.PanelCollection:
    config = _config()
    rows: list[dict[str, object]] = []
    cells = exp26._manifest(config)
    for seed in independent.EXPECTED_SEEDS:
        for cell in cells:
            for mode in independent.EXPECTED_MODES:
                rows.append(
                    {
                        "seed": seed,
                        "generator_id": cell.generator_id,
                        "actuator_mode": mode,
                        "status": "complete",
                        "functional_budget_valid": True,
                        "effective_dynamics_strictly_stable": True,
                        "profile": independent.PROFILE,
                    }
                )
    attempts = tuple(
        packager.AttemptReceipt(
            seed=seed,
            path=f"seed-{seed}",
            run_status="complete",
            run_id=f"run-{seed}",
            planned_coverage_valid=True,
            observed_row_count=independent.EXPECTED_ROWS_PER_SEED,
            observed_complete_rows=independent.EXPECTED_ROWS_PER_SEED,
            observed_failed_rows=0,
            observed_invalid_rows=0,
            file_sha256={},
        )
        for seed in independent.EXPECTED_SEEDS
    )
    contract = _contract()
    return packager.PanelCollection(
        rows=tuple(rows),
        attempts=attempts,
        config=config,
        config_sha256=contract.independent_config_sha256,
        config_file_sha256=contract.independent_config_file_sha256,
        source_contract=config["source_binding"],
        provenance_identity=None,
    )


def test_coverage_requires_exact_30_by_88_by_5_panel() -> None:
    collection = _complete_collection()
    coverage = packager.panel_coverage(collection)
    assert coverage["expected_row_count"] == 30 * 88 * 5 == 13200
    assert coverage["observed_row_count"] == 13200
    assert coverage["cartesian_complete"] is True
    assert coverage["all_functional_budgets_valid"] is True
    assert coverage["all_effective_dynamics_stable"] is True
    assert coverage["source_panel_valid"] is True


def test_scientific_failure_is_retained_and_invalidates_source_panel() -> None:
    collection = _complete_collection()
    failed_rows = list(collection.rows)
    failed_rows[17] = {
        **failed_rows[17],
        "status": "failed",
        "functional_budget_valid": False,
        "failure_reason": "functional_budget",
    }
    attempts = list(collection.attempts)
    attempts[0] = replace(
        attempts[0],
        run_status="complete_with_failures",
        observed_complete_rows=independent.EXPECTED_ROWS_PER_SEED - 1,
        observed_failed_rows=1,
    )
    failed = replace(
        collection,
        rows=tuple(failed_rows),
        attempts=tuple(attempts),
    )
    coverage = packager.panel_coverage(failed)
    assert coverage["observed_row_count"] == 13200
    assert coverage["cartesian_complete"] is True
    assert coverage["row_status_counts"]["failed"] == 1
    assert coverage["all_failures_retained"] is True
    assert coverage["source_panel_valid"] is False


def test_source_package_is_hash_bound_and_always_inconclusive(tmp_path: Path) -> None:
    config = _config()
    contract = _contract()
    row = {
        "seed": 30,
        "generator_id": exp26._manifest(config)[0].generator_id,
        "actuator_mode": independent.EXPECTED_MODES[0],
        "status": "failed",
        "profile": independent.PROFILE,
        "source_panel_row_index": 0,
    }
    collection = packager.PanelCollection(
        rows=(row,),
        attempts=(),
        config=config,
        config_sha256=contract.independent_config_sha256,
        config_file_sha256=contract.independent_config_file_sha256,
        source_contract=config["source_binding"],
        provenance_identity=None,
    )
    output = packager.write_source_panel_package(collection, tmp_path / "package")
    receipt = json.loads(
        (output / "source_panel_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["profile"] == "independent_test"
    assert receipt["conclusion"] == "inconclusive"
    assert receipt["standalone_inference_performed"] is False
    with pytest.raises(ValueError, match="lacks run provenance"):
        packager.load_source_panel_package(output, require_complete=True)

    raw_path = output / "raw_metrics.jsonl"
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw metrics hash"):
        packager.load_source_panel_package(output, require_complete=False)


def test_collector_refuses_duplicate_attempt_selection(tmp_path: Path) -> None:
    config = _config()
    for suffix in ("first", "second"):
        attempt = (
            tmp_path
            / "runs"
            / independent.EXPERIMENT
            / "seed_0030"
            / f"20260717T000000.000000Z_{suffix}"
        )
        attempt.mkdir(parents=True)
        (attempt / "metrics.jsonl").touch()
        (attempt / "config.json").write_text(
            json.dumps(
                {
                    **config,
                    "experiment": independent.EXPERIMENT,
                    "seed": 30,
                    "run_label": independent.REQUIRED_RUN_LABEL,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="refusing favourable selection"):
        packager.collect_source_panel(
            tmp_path,
            config_path=CONFIG_PATH,
            current_git=FAKE_GIT,
            runtime_versions=_runtime(),
        )


def test_runner_artifact_schema_collects_without_formal_support_spoofing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    contract = _contract()
    provenance = independent.build_evidence_provenance(
        contract, run_label=independent.REQUIRED_RUN_LABEL
    )
    evidence = independent.evidence_row_fields(provenance)
    packages = {
        "numpy": contract.runtime_versions["numpy"],
        "scipy": contract.runtime_versions["scipy"],
        "pandas": contract.runtime_versions["pandas"],
        "scikit-learn": contract.runtime_versions["scikit_learn"],
        "statsmodels": contract.runtime_versions["statsmodels"],
    }
    monkeypatch.setattr(
        artifacts,
        "_software_provenance",
        lambda: {"packages": packages, "git": dict(FAKE_GIT)},
    )
    run_config = {**config, "evidence_provenance": provenance}
    with ExperimentRun(
        independent.EXPERIMENT,
        30,
        run_config,
        results_root=tmp_path,
        run_label=independent.REQUIRED_RUN_LABEL,
    ) as run:
        run.register_conditions(independent.planned_conditions(run_config))
        receipt = contract.source_manifest_sha256
        for cell in exp26._manifest(config):
            state_demand = 1.0 + float(cell.alpha)
            input_demand = 2.0 - float(cell.alpha)
            chi = state_demand / (state_demand + input_demand)
            for mode in independent.EXPECTED_MODES:
                run.record(
                    {
                        "status": "complete",
                        "profile": independent.PROFILE,
                        "chi": chi,
                        "state_demand": state_demand,
                        "input_demand": input_demand,
                        "validation_balanced_accuracy": 0.6,
                        "test_balanced_accuracy": 0.6,
                        "functional_budget_valid": True,
                        "effective_dynamics_strictly_stable": True,
                    },
                    **independent._dimensions(
                        cell,
                        mode=mode,
                        manifest_receipt=receipt,
                        evidence=evidence,
                    ),
                )

    collection = packager.collect_source_panel(
        tmp_path,
        config_path=CONFIG_PATH,
        current_git=FAKE_GIT,
        runtime_versions=_runtime(),
    )
    assert len(collection.attempts) == 1
    assert len(collection.rows) == 88 * 5
    assert all(row["profile"] == "independent_test" for row in collection.rows)
    coverage = packager.panel_coverage(collection)
    assert coverage["missing_seeds"] == list(range(31, 60))
    assert coverage["source_panel_valid"] is False
    output = packager.write_source_panel_package(collection, tmp_path / "package")
    loaded = packager.load_source_panel_package(output, require_complete=False)
    assert loaded.receipt["conclusion"] == "inconclusive"
    assert len(loaded.rows) == 440
    assert loaded.raw_metrics_sha256 == loaded.receipt["raw_metrics_sha256"]
    assert loaded.receipt_payload_sha256 == loaded.receipt[
        "receipt_payload_sha256"
    ]
    assert loaded.receipt_file_sha256 == hashlib.sha256(
        (output / "source_panel_receipt.json").read_bytes()
    ).hexdigest()

    forged = tmp_path / "forged-package"
    shutil.copytree(output, forged)
    forged_receipt = json.loads(
        (forged / "source_panel_receipt.json").read_text(encoding="utf-8")
    )
    forged_conclusion = json.loads(
        (forged / "conclusion.json").read_text(encoding="utf-8")
    )
    forged_receipt["coverage"]["source_panel_valid"] = True
    _rewrite_package_metadata(forged, forged_receipt, forged_conclusion)
    with pytest.raises(ValueError, match="coverage is not reproducible"):
        packager.load_source_panel_package(forged, require_complete=False)

    forged_contract = tmp_path / "forged-contract-package"
    shutil.copytree(output, forged_contract)
    contract_receipt = json.loads(
        (forged_contract / "source_panel_receipt.json").read_text(encoding="utf-8")
    )
    contract_conclusion = json.loads(
        (forged_contract / "conclusion.json").read_text(encoding="utf-8")
    )
    contract_receipt["source_contract"]["source_manifest_sha256"] = "f" * 64
    contract_receipt["source_contract_sha256"] = independent._canonical_sha256(
        contract_receipt["source_contract"]
    )
    _rewrite_package_metadata(
        forged_contract, contract_receipt, contract_conclusion
    )
    with pytest.raises(ValueError, match="known config/contract binding"):
        packager.load_source_panel_package(forged_contract, require_complete=False)

    forged_raw = tmp_path / "forged-raw-package"
    shutil.copytree(output, forged_raw)
    raw_rows = [
        json.loads(line)
        for line in (forged_raw / "raw_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    raw_rows[0]["generator_id"] = raw_rows[5]["generator_id"]
    forged_raw_payload = packager._canonical_jsonl(raw_rows)
    (forged_raw / "raw_metrics.jsonl").write_bytes(forged_raw_payload)
    raw_receipt = json.loads(
        (forged_raw / "source_panel_receipt.json").read_text(encoding="utf-8")
    )
    raw_conclusion = json.loads(
        (forged_raw / "conclusion.json").read_text(encoding="utf-8")
    )
    raw_receipt["raw_metrics_sha256"] = hashlib.sha256(
        forged_raw_payload
    ).hexdigest()
    _rewrite_package_metadata(forged_raw, raw_receipt, raw_conclusion)
    with pytest.raises(ValueError, match="Cartesian schema"):
        packager.load_source_panel_package(forged_raw, require_complete=False)

    truncated = tmp_path / "truncated-package"
    shutil.copytree(output, truncated)
    raw_path = truncated / "raw_metrics.jsonl"
    raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
    raw_path.write_bytes(("\n".join(raw_lines[:-1]) + "\n").encode("utf-8"))
    truncated_receipt = json.loads(
        (truncated / "source_panel_receipt.json").read_text(encoding="utf-8")
    )
    truncated_conclusion = json.loads(
        (truncated / "conclusion.json").read_text(encoding="utf-8")
    )
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    truncated_receipt["raw_metrics_sha256"] = raw_sha
    truncated_receipt["raw_metrics_row_count"] = 439
    truncated_receipt["attempts"][0]["observed_row_count"] = 439
    truncated_receipt["attempts"][0]["observed_complete_rows"] = 439
    truncated_receipt["coverage"]["observed_row_count"] = 439
    truncated_receipt["coverage"]["row_status_counts"]["complete"] = 439
    _rewrite_package_metadata(truncated, truncated_receipt, truncated_conclusion)
    honest_partial = packager.load_source_panel_package(
        truncated, require_complete=False
    )
    assert len(honest_partial.rows) == 439
    with pytest.raises(ValueError, match="incomplete or scientifically invalid"):
        packager.load_source_panel_package(truncated, require_complete=True)
