"""Contracts for the untouched feasibility-aware Exp29 source panel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from experiments import exp29_confirmatory_source_panel as exp29
from experiments.common import load_json_config
from src.analysis.actuator_manifest import GeneratorCell
from src.models.task_matched_actuators import ActuatorFitError


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "formal"
    / "exp29_confirmatory_source_panel.json"
)
FAKE_GIT = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}


def _config() -> dict[str, object]:
    return load_json_config(CONFIG_PATH)


def _runtime() -> dict[str, str | None]:
    return exp26.scientific_runtime_versions()


def _contract() -> exp29.SourceContract:
    return exp29.validate_source_contract(
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
        rotation_seed=291,
    )


def _metrics(mode: str) -> dict[str, object]:
    return {
        "status": "complete",
        "experiment_protocol_version": "fixture",
        "statistics_unit": "seed",
        "split_unit": "block",
        "time_points_randomly_split": False,
        "profile": "confirmatory_test",
        "dev_only": False,
        "training_algorithm": "train_only_closed_form_task_matched_actuators",
        "used_autograd": False,
        "used_bptt": False,
        "chi": 0.5,
        "state_demand": 0.2,
        "input_demand": 0.3,
        "target_train_balanced_accuracy": 0.8,
        "target_validation_balanced_accuracy": 0.8,
        "target_test_balanced_accuracy": 0.8,
        "train_balanced_accuracy": 0.7,
        "validation_balanced_accuracy": 0.6 if mode == "frozen" else 0.75,
        "test_balanced_accuracy": 0.61 if mode == "frozen" else 0.76,
        "dataset_fingerprint": "dataset",
        "train_split_fingerprint": "train",
        "validation_split_fingerprint": "validation",
        "test_split_fingerprint": "test",
        "correction_fingerprint": f"correction-{mode}",
        "functional_budget_valid": True,
        "effective_dynamics_strictly_stable": True,
    }


def _patch_one_cell(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_error: BaseException | None = None,
) -> exp29.SourceContract:
    cell = _cell()
    contract = _contract()
    monkeypatch.setattr(exp29, "validate_source_contract", lambda config: contract)
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
            for mode in exp29.EXPECTED_MODES
        ],
    )
    monkeypatch.setattr(exp26, "_carrier_config", lambda config: object())
    monkeypatch.setattr(exp26, "make_carrier", lambda config, seed: object())
    monkeypatch.setattr(
        exp26,
        "_setup_generator",
        lambda config, carrier, generator, **kwargs: object(),
    )

    def condition(
        config: object, setup: object, *, mode: str
    ) -> tuple[dict[str, object], bool]:
        if mode == "routing" and active_error is not None:
            raise active_error
        return _metrics(mode), True

    monkeypatch.setattr(exp26, "_condition_metrics", condition)
    monkeypatch.setattr(exp26, "git_identity", lambda: dict(FAKE_GIT))
    monkeypatch.setattr(
        exp26,
        "scientific_runtime_versions",
        lambda: dict(contract.runtime_versions),
    )
    return contract


def test_registered_config_freezes_unseen_seeds_cap_and_policy() -> None:
    config = _config()
    contract = exp29.validate_source_contract(
        config, current_git=FAKE_GIT, runtime_versions=_runtime()
    )
    assert tuple(config["seeds"]) == tuple(range(60, 90))
    assert tuple(config["preregistration"]["meta_training_seeds"]) == tuple(range(30))
    assert config["actuator"]["max_scale"] == 256.0
    assert config["preregistration"]["selective_rerun_permitted"] is False
    assert config["feasibility_policy"]["deployment_fallback_mode"] == "frozen"
    assert config["feasibility_policy"]["selector_candidate_modes"] == [
        "routing",
        "gain",
        "low_rank",
    ]
    assert config["feasibility_policy"]["nonselector_control_mode"] == "rgl"
    assert contract.config_sha256 == exp29.REGISTERED_CONFIG_CANONICAL_SHA256
    assert contract.implementation_file_sha256 == (exp29.exp29_implementation_sha256())
    assert len(exp29.planned_conditions(config)) == 88 * 5


def test_runner_source_hash_normalizes_only_registered_config_literal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runner.py"
    source.write_text(
        'REGISTERED_CONFIG_CANONICAL_SHA256 = ("' + "a" * 64 + '")\nVALUE = 1\n',
        encoding="utf-8",
    )
    first = exp29._normalized_runner_source_sha256(source)
    source.write_text(
        'REGISTERED_CONFIG_CANONICAL_SHA256 = ("' + "b" * 64 + '")\nVALUE = 1\n',
        encoding="utf-8",
    )
    assert exp29._normalized_runner_source_sha256(source) == first
    source.write_text(
        'REGISTERED_CONFIG_CANONICAL_SHA256 = ("' + "b" * 64 + '")\nVALUE = 2\n',
        encoding="utf-8",
    )
    assert exp29._normalized_runner_source_sha256(source) != first


def test_contract_rejects_exp29_implementation_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = exp29.exp29_implementation_sha256()
    monkeypatch.setattr(
        exp29,
        "exp29_implementation_sha256",
        lambda: {**observed, exp29.PACKAGER_RELATIVE_PATH: "0" * 64},
    )
    with pytest.raises(ValueError, match="implementation binding"):
        exp29.validate_source_contract(
            _config(), current_git=FAKE_GIT, runtime_versions=_runtime()
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("seeds", list(range(61, 91))),
        ("actuator.max_scale", 512.0),
        ("preregistration.selective_rerun_permitted", True),
        ("feasibility_policy.deployment_fallback_mode", "routing"),
    ],
)
def test_contract_rejects_seed_ceiling_or_policy_tampering(
    path: str, value: object
) -> None:
    config = json.loads(json.dumps(_config()))
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError, match="config hash"):
        exp29.validate_source_contract(
            config, current_git=FAKE_GIT, runtime_versions=_runtime()
        )


def test_contract_rejects_dirty_tree_and_wrong_runtime() -> None:
    with pytest.raises(ValueError, match="clean identifiable"):
        exp29.validate_source_contract(
            _config(),
            current_git={**FAKE_GIT, "dirty": True},
            runtime_versions=_runtime(),
        )
    runtime = dict(_runtime())
    runtime["python"] = "3.12.0"
    with pytest.raises(ValueError, match="Python 3.11"):
        exp29.validate_source_contract(
            _config(), current_git=FAKE_GIT, runtime_versions=runtime
        )


def test_runner_records_feasible_modes_and_exact_frozen_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_one_cell(monkeypatch)
    path = exp29.run_seed(
        {"profile": "confirmatory_test"},
        60,
        tmp_path,
        run_label=exp29.REQUIRED_RUN_LABEL,
    )
    rows = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"complete"}
    frozen = next(row for row in rows if row["actuator_mode"] == "frozen")
    assert frozen["matched_budget_support_eligible"] is False
    assert frozen["deployment_fallback_applied"] is False
    for row in rows:
        assert row["statistics_unit"] == "seed"
        if row["actuator_mode"] != "frozen":
            assert row["matched_budget_support_eligible"] is True


def test_runner_retains_cap_exceedance_as_terminal_frozen_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = ActuatorFitError(
        "functional-budget scale is non-finite or exceeds max_scale"
    )
    _patch_one_cell(monkeypatch, active_error=error)
    path = exp29.run_seed(
        {"profile": "confirmatory_test"},
        60,
        tmp_path,
        run_label=exp29.REQUIRED_RUN_LABEL,
    )
    rows = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    frozen = next(row for row in rows if row["actuator_mode"] == "frozen")
    routing = next(row for row in rows if row["actuator_mode"] == "routing")
    assert routing["status"] == "infeasible"
    assert routing["infeasible_reason"] == "budget_scale_above_cap"
    assert routing["matched_budget_support_eligible"] is False
    assert routing["deployment_fallback_applied"] is True
    assert routing["deployment_mode"] == "frozen"
    assert routing["test_balanced_accuracy"] == frozen["test_balanced_accuracy"]
    assert (
        routing["fallback_frozen_correction_fingerprint"]
        == frozen["correction_fingerprint"]
    )
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["condition_failures"] == 0


def test_runner_rejects_any_nonregistered_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_one_cell(monkeypatch)
    with pytest.raises(ValueError, match="60--89"):
        exp29.run_seed(
            {"profile": "confirmatory_test"},
            59,
            tmp_path,
            run_label=exp29.REQUIRED_RUN_LABEL,
        )
