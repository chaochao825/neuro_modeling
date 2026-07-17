"""Contracts for the transparent Exp28 reachability-ceiling amendment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from experiments import exp28_exp26_independent_source_panel as independent
from experiments.common import load_json_config
from scripts import package_exp28_independent_source_panel as packager
from src.utils import artifacts
from src.utils.artifacts import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = (
    PROJECT_ROOT / "configs/formal/exp28_exp26_independent_source_panel.json"
)
AMENDED_CONFIG = (
    PROJECT_ROOT
    / "configs/formal/exp28_exp26_independent_source_panel_ceiling_amendment_1.json"
)
FAKE_GIT = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}


def _runtime() -> dict[str, str | None]:
    return exp26.scientific_runtime_versions()


def _amended_config() -> dict[str, object]:
    return load_json_config(AMENDED_CONFIG)


def _amended_contract() -> independent.SourceContract:
    return independent.validate_source_contract(
        _amended_config(), current_git=FAKE_GIT, runtime_versions=_runtime()
    )


def test_amendment_is_hash_locked_reachability_only_and_nonconfirmatory() -> None:
    config = _amended_config()
    contract = _amended_contract()
    amendment = config["protocol_amendment"]
    assert isinstance(amendment, dict)
    assert independent.canonical_config_sha256(config) == (
        independent.AMENDED_CONFIG_CANONICAL_SHA256
    )
    assert config["actuator"]["max_scale"] == 256.0
    assert amendment["previous_max_scale"] == 128.0
    assert amendment["amended_max_scale"] == 256.0
    assert amendment["decision_rule"] == (
        "next_power_of_two_strictly_above_previous_ceiling"
    )
    assert amendment["trigger_class"] == "reachability_only"
    assert amendment["performance_metrics_inspected"] is True
    assert amendment["performance_metrics_inspection_timing"] == (
        "after_deterministic_amendment_decision_draft"
    )
    assert amendment["performance_metrics_used_for_amendment"] is False
    assert amendment["confirmatory_independence_restored"] is False
    assert amendment["selective_rerun_permitted"] is False
    assert amendment["further_ceiling_amendments_permitted"] is False
    assert amendment["rerun_scope"] == "all_30_seeds_x_88_generators_x_5_modes"
    assert contract.protocol_amendment_sha256 == independent._canonical_sha256(
        amendment
    )
    assert contract.functional_budget_max_scale == 256.0


def test_frozen_v1_contract_still_replays_unchanged() -> None:
    config = load_json_config(FROZEN_CONFIG)
    contract = independent.validate_source_contract(
        config, current_git=FAKE_GIT, runtime_versions=_runtime()
    )
    assert contract.protocol_version == independent.PROTOCOL_VERSION
    assert contract.required_run_label == independent.REQUIRED_RUN_LABEL
    assert contract.functional_budget_max_scale == 128.0
    assert contract.protocol_amendment is None
    assert contract.protocol_amendment_sha256 is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("actuator", "max_scale"), 512.0, "config hash is not registered"),
        (
            ("protocol_amendment", "performance_metrics_inspected"),
            False,
            "amendment is not registered",
        ),
        (
            ("protocol_amendment", "selective_rerun_permitted"),
            True,
            "amendment is not registered",
        ),
        (
            (
                "protocol_amendment",
                "trigger_evidence",
                "source_panel_receipt_file_sha256",
            ),
            "0" * 64,
            "amendment is not registered",
        ),
    ],
)
def test_amended_contract_rejects_any_ceiling_or_audit_tampering(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = json.loads(json.dumps(_amended_config()))
    target = config
    for part in path[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        independent.validate_source_contract(
            config, current_git=FAKE_GIT, runtime_versions=_runtime()
        )


def test_amended_contract_actually_verifies_preserved_trigger_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = independent._file_sha256

    def corrupt_receipt(path: Path) -> str:
        if path.name == "source_panel_receipt.json" and "ceiling128" in str(path):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(independent, "_file_sha256", corrupt_receipt)
    with pytest.raises(ValueError, match="trigger artifact hash"):
        independent.validate_source_contract(
            _amended_config(), current_git=FAKE_GIT, runtime_versions=_runtime()
        )


def test_amended_artifact_and_package_bind_amendment_without_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _amended_config()
    contract = _amended_contract()
    provenance = independent.build_evidence_provenance(
        contract, run_label=contract.required_run_label
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
        run_label=contract.required_run_label,
    ) as run:
        run.register_conditions(independent.planned_conditions(run_config))
        for cell in exp26._manifest(config):
            for mode in independent.EXPECTED_MODES:
                run.record(
                    {
                        "status": "complete",
                        "profile": independent.PROFILE,
                        "chi": 0.5,
                        "state_demand": 1.0,
                        "input_demand": 1.0,
                        "validation_balanced_accuracy": 0.5,
                        "test_balanced_accuracy": 0.5,
                        "functional_budget_valid": True,
                        "effective_dynamics_strictly_stable": True,
                    },
                    **independent._dimensions(
                        cell,
                        mode=mode,
                        manifest_receipt=contract.source_manifest_sha256,
                        evidence=evidence,
                    ),
                )

    collection = packager.collect_source_panel(
        tmp_path,
        config_path=AMENDED_CONFIG,
        run_label=contract.required_run_label,
        current_git=FAKE_GIT,
        runtime_versions=_runtime(),
    )
    output = packager.write_source_panel_package(collection, tmp_path / "package")
    loaded = packager.load_source_panel_package(output, require_complete=False)
    assert len(loaded.rows) == 440
    assert loaded.receipt["schema_version"] == (
        packager.AMENDED_SOURCE_PACKAGE_SCHEMA_VERSION
    )
    assert loaded.receipt["protocol_amendment_sha256"] == (
        contract.protocol_amendment_sha256
    )
    assert loaded.receipt["inference_status"] == (
        "post_hoc_amended_sensitivity_non_confirmatory"
    )
    assert loaded.receipt["confirmatory_independence_restored"] is False
    assert all(
        row["protocol_amendment_performance_metrics_inspected"] is True
        for row in loaded.rows
    )
