"""Contracts for the strict train-only Exp26 budget preflight."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from experiments import exp26_formal_budget_preflight as preflight
from src.analysis.actuator_manifest import GeneratorCell, manifest_hash
from src.tasks import actuator_matching as actuator_task


def _config(*, max_scale: float = 1e6) -> dict[str, object]:
    return {
        "profile": "formal",
        "dev_only": False,
        "seeds": list(range(30)),
        "training_algorithm": "train_only_closed_form_task_matched_actuators",
        "used_autograd": False,
        "used_bptt": False,
        "carrier": {
            "n_neurons": 12,
            "n_inputs": 4,
            "n_outputs": 1,
            "inhibitory_fraction": 0.25,
            "spectral_radius": 0.6,
            "input_scale": 0.3,
        },
        "task": {
            "n_train_blocks": 2,
            "n_validation_blocks": 1,
            "n_test_blocks": 1,
            "trials_per_block": 4,
            "input_steps": 2,
            "input_std": 1.0,
            "delta_a_log10_range": [-1.0, -0.5],
            "delta_b_log10_range": [-1.0, -0.5],
            "stability_limit": 0.9,
        },
        "actuator": {
            "rank_a_capacity": 2,
            "rank_b_capacity": 2,
            "ridge": 1e-6,
            "max_scale": max_scale,
            "degeneracy_tolerance": 1e-12,
            "budget_relative_tolerance": 1e-8,
            "context_center_tolerance": 1e-12,
        },
    }


def _cell(identifier: str = "cell-a") -> GeneratorCell:
    return GeneratorCell(
        generator_id=identifier,
        generator_split="discovery",
        alpha=0.5,
        transition_rank=2,
        input_rank=2,
        delay=0,
        noise_std=0.1,
        rotation_seed=11,
    )


def _clean_provenance() -> dict[str, object]:
    return {
        "stable_during_run": True,
        "git_dirty": False,
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "end_snapshot": {
            "git_dirty": False,
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
        },
    }


def _bind_fixture_policy(
    config: dict[str, object],
    records: list[preflight.PreflightRecord],
) -> tuple[dict[str, object], list[preflight.PreflightRecord], float]:
    observed = max(float(record.required_budget_scale) for record in records)
    headroom = 1.25
    ceiling = preflight._next_power_of_two(headroom * observed)
    resolved = json.loads(json.dumps(config))
    resolved["seeds"] = sorted({record.seed for record in records})
    resolved["actuator"]["max_scale"] = ceiling
    resolved["budget_preflight"] = {
        "revision": "fixture-observed-bound",
        "receipt_schema": "exp26_budget_preflight_v2_observed_bound",
        "fit_scope": "training_blocks_only",
        "n_registered_active_fits": len(records),
        "required_scale_max": observed,
        "headroom_multiplier": headroom,
        "rounding_rule": "next_power_of_two",
        "validation_test_behavior_used": False,
        "validation_test_rollout_used": False,
    }
    rebound = [replace(record, registered_max_scale=ceiling) for record in records]
    return resolved, rebound, observed


def _summary(
    config: dict[str, object],
    records: list[preflight.PreflightRecord],
) -> dict[str, object]:
    return preflight.summarize_records(
        records,
        config_sha256=preflight.registered_config_sha256(config),
        receipt_manifest_hash=manifest_hash((_cell(),)),
        provenance=_clean_provenance(),
        config=config,
        expected_panel_size=len(records),
    )


def test_registered_hash_reuses_exp26_canonical_runtime_exclusions() -> None:
    config = _config()
    runtime_augmented = {
        **config,
        "config_path": "C:\\checkout\\formal.json",
        "run_label": "formal-panel",
        "seed": 99,
        "experiment": "runtime-only",
        "evidence_provenance": {"runtime": True},
    }
    assert preflight.registered_config_sha256(runtime_augmented) == (
        exp26.canonical_config_sha256(config)
    )
    changed = json.loads(json.dumps(config))
    changed["actuator"]["ridge"] = 1e-5
    assert preflight.registered_config_sha256(changed) != (
        preflight.registered_config_sha256(config)
    )


def test_formal_contract_requires_30_unique_non_bptt_seeds() -> None:
    assert preflight.validate_formal_contract(_config()) == tuple(range(30))
    too_short = _config()
    too_short["seeds"] = list(range(29))
    with pytest.raises(ValueError, match="exactly 30"):
        preflight.validate_formal_contract(too_short)
    duplicate = _config()
    duplicate["seeds"] = [0] * 30
    with pytest.raises(ValueError, match="exactly 30"):
        preflight.validate_formal_contract(duplicate)
    reordered = _config()
    reordered["seeds"] = list(reversed(range(30)))
    with pytest.raises(ValueError, match="registered seeds"):
        preflight.validate_formal_contract(reordered)
    bptt = _config()
    bptt["used_bptt"] = True
    with pytest.raises(ValueError, match="non-BPTT"):
        preflight.validate_formal_contract(bptt)


def test_frozen_formal_budget_policy_recomputes_ceiling_and_panel() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "formal"
        / "exp26_actuator_phase_diagram.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = preflight.validate_formal_contract(config)
    cells = exp26._manifest(config)
    assert preflight.validate_registered_budget_policy(config, seeds, cells) == 128.0

    tampered = json.loads(json.dumps(config))
    tampered["actuator"]["max_scale"] = 127.0
    with pytest.raises(ValueError, match="does not match"):
        preflight.validate_registered_budget_policy(tampered, seeds, cells)

    schema_tampered = json.loads(json.dumps(config))
    schema_tampered["budget_preflight"]["receipt_schema"] = "wrong-schema"
    with pytest.raises(ValueError, match="schema or scope"):
        preflight.validate_registered_budget_policy(schema_tampered, seeds, cells)


def test_observed_max_ceiling_panel_and_clean_provenance_jointly_pass() -> None:
    records = preflight.audit_cells(_config(), (6,), (_cell(),))
    config, rebound, observed = _bind_fixture_policy(_config(), records)
    summary = _summary(config, rebound)
    assert summary["schema_version"] == (
        "exp26_budget_preflight_v2_observed_bound"
    )
    assert summary["receipt_schema"] == summary["schema_version"]
    assert summary["observed_required_scale_max"] == pytest.approx(observed)
    assert summary["registered_required_scale_max"] == pytest.approx(observed)
    assert summary["observed_max_matches"] is True
    assert summary["ceiling_binding_valid"] is True
    assert summary["panel_binding_valid"] is True
    assert summary["policy_valid"] is True
    assert summary["provenance_clean"] is True
    assert summary["provenance_stable_during_run"] is True
    assert summary["n_unreachable"] == 0
    assert summary["preflight_passed"] is True


def test_tampered_required_max_fails_even_when_it_rounds_to_same_ceiling() -> None:
    records = preflight.audit_cells(_config(), (6,), (_cell(),))
    config, rebound, observed = _bind_fixture_policy(_config(), records)
    original_ceiling = config["actuator"]["max_scale"]
    delta = max(1e-4, observed * 1e-4)
    candidate = observed + delta
    if preflight._next_power_of_two(1.25 * candidate) != original_ceiling:
        candidate = observed - delta
    assert candidate > 0.0
    assert preflight._next_power_of_two(1.25 * candidate) == original_ceiling
    config["budget_preflight"]["required_scale_max"] = candidate
    summary = _summary(config, rebound)
    assert summary["rounded_policy_ceiling"] == original_ceiling
    assert summary["observed_rounded_policy_ceiling"] == original_ceiling
    assert summary["ceiling_binding_valid"] is True
    assert summary["observed_max_matches"] is False
    assert summary["required_scale_match_absolute_error"] == pytest.approx(
        abs(candidate - observed)
    )
    assert summary["policy_valid"] is False
    assert summary["preflight_passed"] is False


def test_formal_required_max_tamper_inside_128_bin_is_detected() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "formal"
        / "exp26_actuator_phase_diagram.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observed = float(config["budget_preflight"]["required_scale_max"])
    records = preflight.audit_cells(_config(max_scale=128.0), (6,), (_cell(),))
    rebound = [replace(record, registered_max_scale=128.0) for record in records]
    tampered = json.loads(json.dumps(config))
    tampered["budget_preflight"]["required_scale_max"] = observed - 0.01
    assert preflight._next_power_of_two(1.25 * observed) == 128.0
    assert preflight._next_power_of_two(1.25 * (observed - 0.01)) == 128.0
    bindings = preflight._budget_policy_bindings(
        tampered,
        rebound,
        observed_required_scale_max=observed,
        expected_panel_size=len(rebound),
    )
    assert bindings["rounded_policy_ceiling"] == 128.0
    assert bindings["observed_rounded_policy_ceiling"] == 128.0
    assert bindings["observed_max_matches"] is False
    assert bindings["policy_valid"] is False


def test_tampered_registered_receipt_schema_fails_closed() -> None:
    records = preflight.audit_cells(_config(), (6,), (_cell(),))
    config, rebound, _ = _bind_fixture_policy(_config(), records)
    config["budget_preflight"]["receipt_schema"] = "older-schema"
    summary = _summary(config, rebound)
    assert summary["schema_version"] == (
        "exp26_budget_preflight_v2_observed_bound"
    )
    assert summary["registered_receipt_schema"] == "older-schema"
    assert summary["receipt_schema_matches"] is False
    assert summary["policy_contract_valid"] is False
    assert summary["policy_valid"] is False
    assert summary["preflight_passed"] is False


def test_duplicate_condition_cannot_satisfy_registered_panel_by_count_only() -> None:
    records = preflight.audit_cells(_config(), (6,), (_cell(),))
    config, rebound, _ = _bind_fixture_policy(_config(), records)
    duplicated = [*rebound[:-1], rebound[0]]
    summary = _summary(config, duplicated)
    assert summary["observed_panel_size"] == summary["registered_panel_size"]
    assert summary["observed_unique_condition_count"] == len(duplicated) - 1
    assert summary["panel_cartesian_complete"] is False
    assert summary["panel_binding_valid"] is False
    assert summary["policy_valid"] is False
    assert summary["preflight_passed"] is False


def test_policy_mismatch_receipt_is_written_before_failure(tmp_path: Path) -> None:
    records = preflight.audit_cells(_config(), (7,), (_cell(),))
    config, rebound, observed = _bind_fixture_policy(_config(), records)
    config["budget_preflight"]["required_scale_max"] = observed + 0.01
    path, summary = preflight.write_receipt(
        tmp_path / "mismatch",
        config,
        (_cell(),),
        rebound,
        provenance=_clean_provenance(),
    )
    assert path.is_dir()
    assert (path / "preflight_cells.jsonl").is_file()
    assert (path / "preflight_summary.json").is_file()
    assert summary["observed_max_matches"] is False
    assert summary["policy_valid"] is False
    assert summary["preflight_passed"] is False


def test_preflight_calls_formal_fit_and_never_touches_heldout_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    original_make_split = actuator_task._make_split
    original_fit = exp26.fit_task_matched_actuator
    calls: list[str] = []
    split_calls: list[str] = []

    def split_spy(*args: object, **kwargs: object) -> object:
        split_name = str(kwargs["split_name"])
        split_calls.append(split_name)
        if split_name != "train":
            raise AssertionError("a validation/test split factory was called")
        return original_make_split(*args, **kwargs)

    def fit_spy(*args: object, **kwargs: object) -> object:
        calls.append(str(kwargs["mode"]))
        return original_fit(*args, **kwargs)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("held-out rollout/readout/behavior was accessed")

    monkeypatch.setattr(actuator_task, "_make_split", split_spy)
    monkeypatch.setattr(exp26, "fit_task_matched_actuator", fit_spy)
    monkeypatch.setattr(exp26, "make_dataset", forbidden)
    for name in (
        "_rollout",
        "_fit_shared_readout",
        "_split_metrics",
        "_balanced_accuracy",
    ):
        monkeypatch.setattr(exp26, name, forbidden)

    records = preflight.audit_cells(config, (3,), (_cell(),))
    assert split_calls == ["train"]
    assert calls == list(preflight.ACTIVE_MODES)
    assert len(records) == 4
    assert all(record.fit_status == "complete" for record in records)
    assert all(record.reachable_under_registered_max_scale for record in records)
    assert all(not record.validation_rollout_accessed for record in records)
    assert all(not record.test_rollout_accessed for record in records)
    assert all(not record.validation_behavior_accessed for record in records)
    assert all(not record.test_behavior_accessed for record in records)
    assert len({record.train_split_fingerprint for record in records}) == 1
    assert len({record.training_fingerprint for record in records}) == 1


def test_preflight_recovers_and_flags_required_scale_above_registered_bound() -> None:
    config = _config(max_scale=1e-12)
    records = preflight.audit_cells(config, (4,), (_cell(),))
    assert len(records) == len(preflight.ACTIVE_MODES)
    assert all(record.fit_status == "blocked_max_scale" for record in records)
    assert all(record.diagnostic_refit_used for record in records)
    assert all(record.max_scale_exceeded is True for record in records)
    assert all(not record.reachable_under_registered_max_scale for record in records)
    assert all(
        record.required_budget_scale is not None
        and record.required_budget_scale > 1e-12
        for record in records
    )
    summary = preflight.summarize_records(
        records,
        config_sha256=preflight.registered_config_sha256(config),
        receipt_manifest_hash=manifest_hash((_cell(),)),
        provenance={"fixture": True},
    )
    assert summary["all_reachable_under_registered_max_scale"] is False
    assert summary["n_unreachable"] == 4
    assert summary["n_max_scale_blockers"] == 4
    assert summary["n_other_failures"] == 0
    assert summary["required_budget_scale_quantiles"]["max"] > 1e-12
    assert summary["worst_condition"]["max_scale_exceeded"] is True


def test_setup_failures_retain_every_active_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dataset(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(
        actuator_task, "make_actuator_matching_train_split", fail_dataset
    )
    records = preflight.audit_cells(_config(), (1,), (_cell(),))
    assert len(records) == len(preflight.ACTIVE_MODES)
    assert {record.actuator_mode for record in records} == set(
        preflight.ACTIVE_MODES
    )
    assert all(record.fit_status == "setup_error" for record in records)
    assert all(record.error_type == "RuntimeError" for record in records)
    assert all(not record.reachable_under_registered_max_scale for record in records)


def test_receipt_binds_config_manifest_raw_cells_and_code_tree(tmp_path: Path) -> None:
    config = _config()
    cells = (_cell(),)
    records = preflight.audit_cells(config, (7,), cells)
    path, summary = preflight.write_receipt(
        tmp_path / "receipt", config, cells, records
    )
    assert path.is_dir()
    expected_config_hash = preflight.registered_config_sha256(config)
    assert (path / "config.sha256").read_text(encoding="ascii").strip() == (
        expected_config_hash
    )
    assert (path / "manifest.sha256").read_text(encoding="ascii").strip() == (
        manifest_hash(cells)
    )
    raw = [
        json.loads(line)
        for line in (path / "preflight_cells.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(raw) == len(preflight.ACTIVE_MODES)
    assert raw == [asdict(record) for record in records]
    persisted = json.loads(
        (path / "preflight_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["config_sha256"] == expected_config_hash
    assert persisted["manifest_hash"] == manifest_hash(cells)
    assert persisted["fit_scope"] == "training_blocks_only"
    assert persisted["validation_behavior_accessed"] is False
    assert persisted["test_behavior_accessed"] is False
    provenance = persisted["code_tree_provenance"]
    assert len(provenance["critical_code_sha256"]) == 64
    assert provenance["stable_during_run"] is True
    assert persisted["provenance_stable_during_run"] is True
    assert set(provenance["critical_file_sha256"]) == set(
        preflight._CRITICAL_CODE_FILES
    )
    assert summary["n_records"] == 4


def test_code_change_during_scan_fails_preflight_even_when_budget_is_reachable() -> None:
    config = _config()
    records = preflight.audit_cells(config, (8,), (_cell(),))
    summary = preflight.summarize_records(
        records,
        config_sha256=preflight.registered_config_sha256(config),
        receipt_manifest_hash=manifest_hash((_cell(),)),
        provenance={"stable_during_run": False},
    )
    assert summary["all_reachable_under_registered_max_scale"] is True
    assert summary["provenance_stable_during_run"] is False
    assert summary["preflight_passed"] is False


def test_dirty_tree_fails_preflight_even_when_stable_and_reachable() -> None:
    config = _config()
    records = preflight.audit_cells(config, (9,), (_cell(),))
    summary = preflight.summarize_records(
        records,
        config_sha256=preflight.registered_config_sha256(config),
        receipt_manifest_hash=manifest_hash((_cell(),)),
        provenance={
            "stable_during_run": True,
            "git_dirty": True,
            "end_snapshot": {"git_dirty": True},
        },
    )
    assert summary["all_reachable_under_registered_max_scale"] is True
    assert summary["provenance_stable_during_run"] is True
    assert summary["provenance_clean"] is False
    assert summary["preflight_passed"] is False


def test_cli_returns_nonzero_after_writing_an_unreachable_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked"
    observed: dict[str, object] = {}

    def fake_run(
        config: object, output_dir: object, *, workers: int
    ) -> tuple[Path, dict[str, object]]:
        observed["output_dir"] = Path(output_dir)
        observed["workers"] = workers
        Path(output_dir).mkdir()
        return Path(output_dir), {
            "all_reachable_under_registered_max_scale": False,
            "n_unreachable": 2,
        }

    monkeypatch.setattr(preflight, "load_json_config", lambda path: _config())
    monkeypatch.setattr(preflight, "run_formal_preflight", fake_run)
    exit_code = preflight.main(
        ["--config", "fixture.json", "--output-dir", str(output), "--workers", "3"]
    )
    assert exit_code == 2
    assert observed == {"output_dir": output, "workers": 3}
    assert output.is_dir()


def test_default_receipt_path_uses_ignored_run_staging_tree() -> None:
    path = preflight._default_output_dir()
    relative = path.relative_to(preflight.PROJECT_ROOT)
    assert relative.parts[:3] == (
        "results",
        "runs",
        preflight.EXPERIMENT,
    )
