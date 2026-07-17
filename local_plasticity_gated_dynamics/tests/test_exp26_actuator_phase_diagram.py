"""Contracts for the preregistered Exp26 actuator phase diagram."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments import exp26_actuator_phase_diagram as exp26
from experiments import exp26_formal_budget_preflight as preflight
from experiments.common import load_json_config


ROOT = Path(__file__).resolve().parents[1]


def _config(profile: str = "smoke") -> dict[str, object]:
    return load_json_config(
        ROOT / "configs" / profile / "exp26_actuator_phase_diagram.json"
    )


def _records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )


@pytest.fixture(scope="module")
def clean_formal_preflight(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    config = _config("formal")
    cells = exp26._manifest(config)
    required_max = float(config["budget_preflight"]["required_scale_max"])
    max_scale = float(config["actuator"]["max_scale"])
    records: list[preflight.PreflightRecord] = []
    final_key = (29, cells[-1].generator_id, preflight.ACTIVE_MODES[-1])
    for seed in config["seeds"]:
        for cell in cells:
            for mode in preflight.ACTIVE_MODES:
                required = (
                    required_max
                    if (seed, cell.generator_id, mode) == final_key
                    else 1.0
                )
                records.append(
                    preflight.PreflightRecord(
                        seed=int(seed),
                        generator_id=cell.generator_id,
                        generator_split=cell.generator_split,
                        alpha=cell.alpha,
                        transition_rank=cell.transition_rank,
                        input_rank=cell.input_rank,
                        delay=cell.delay,
                        noise_std=cell.noise_std,
                        rotation_seed=cell.rotation_seed,
                        actuator_mode=mode,
                        registered_max_scale=max_scale,
                        required_budget_scale=required,
                        max_scale_exceeded=False,
                        reachable_under_registered_max_scale=True,
                        fit_status="complete",
                        diagnostic_refit_used=False,
                        target_l2_rms=1.0,
                        raw_current_l2_rms=1.0 / required,
                        budget_relative_error=0.0,
                        train_split_fingerprint=f"train-{seed}-{cell.generator_id}",
                        training_fingerprint=f"fit-{seed}-{cell.generator_id}",
                        process_noise_fingerprint=f"noise-{seed}-{cell.generator_id}",
                    )
                )
    snapshot = preflight.code_tree_provenance()
    snapshot["git_dirty"] = False
    snapshot["worktree_content_sha256"] = "0" * 64
    end_snapshot = dict(snapshot)
    provenance = {
        **snapshot,
        "stable_during_run": True,
        "end_snapshot": end_snapshot,
    }
    receipt_path, summary = preflight.write_receipt(
        tmp_path_factory.mktemp("exp26-preflight") / "receipt",
        config,
        cells,
        records,
        provenance=provenance,
    )
    assert summary["preflight_passed"] is True
    run_git = {
        "commit": snapshot["git_commit"],
        "tree": snapshot["git_tree"],
        "dirty": False,
    }
    return config, receipt_path, run_git


def test_exp26_manifest_and_seed_contract_are_preregistered() -> None:
    smoke = _config()
    formal = _config("formal")
    assert smoke["seeds"] == [9000, 9001]
    assert formal["seeds"] == list(range(30))
    assert set(smoke["seeds"]).isdisjoint(formal["seeds"])
    assert smoke["dev_only"] is True
    assert formal["dev_only"] is False
    assert len(exp26._manifest(smoke)) == 24
    assert len(exp26._manifest(formal)) == 88
    assert len(exp26._planned_conditions(smoke)) == 24 * 5
    assert len(exp26._planned_conditions(formal)) == 88 * 5
    assert formal["used_autograd"] is False
    assert formal["used_bptt"] is False
    assert smoke["protocol_version"] == exp26.PROTOCOL_VERSION
    assert formal["protocol_version"] == exp26.PROTOCOL_VERSION


def test_canonical_config_hash_excludes_only_run_provenance() -> None:
    config = _config()
    altered = {
        **config,
        "config_path": "some/other/location.json",
        "experiment": exp26.EXPERIMENT,
        "seed": 9000,
        "run_label": "panel-a",
        "evidence_provenance": {"receipt": "ignored"},
    }
    assert exp26.canonical_config_sha256(altered) == (
        exp26.canonical_config_sha256(config)
    )
    changed = json.loads(json.dumps(config))
    changed["analysis"]["tie_margin"] = 0.02
    assert exp26.canonical_config_sha256(changed) != (
        exp26.canonical_config_sha256(config)
    )


def test_registered_moments_encode_input_epoch_delay_and_noise() -> None:
    config = _config()
    carrier = exp26.make_carrier(exp26._carrier_config(config), 9000)
    dataset_config = exp26._dataset_config(config)
    state, inputs = exp26._registered_second_moments(
        carrier,
        dataset_config,
        delay=3,
        noise_std=0.3,
    )
    assert state.shape[0] == dataset_config.input_steps + 3
    assert inputs.shape[0] == state.shape[0]
    np.testing.assert_allclose(
        inputs[: dataset_config.input_steps],
        np.broadcast_to(np.eye(4), (dataset_config.input_steps, 4, 4)),
    )
    np.testing.assert_array_equal(inputs[dataset_config.input_steps :], 0.0)
    np.testing.assert_array_equal(state[0], 0.0)
    assert np.trace(state[-1]) > 0.0


def test_one_smoke_seed_is_paired_budget_matched_and_failure_retaining(
    tmp_path: Path,
) -> None:
    config = _config()
    path = exp26.run_seed(
        config,
        9000,
        tmp_path,
        run_label="smoke-panel",
    )
    planned = json.loads(
        (path / "planned_conditions.json").read_text(encoding="utf-8")
    )
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    run_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    records = _records(path)
    assert len(planned) == 120
    assert len(records) == len(planned)
    assert status["status"] == "complete"
    assert status["run_label"] == "smoke-panel"
    assert run_config["run_label"] == "smoke-panel"
    assert run_manifest["run_label"] == "smoke-panel"
    evidence = run_config["evidence_provenance"]
    assert run_manifest["evidence_provenance"] == evidence
    assert evidence["schema_version"] == exp26.EVIDENCE_SCHEMA_VERSION
    assert evidence["canonical_config_sha256"] == exp26.canonical_config_sha256(
        config
    )
    assert evidence["manifest_sha256"] == config["manifest"]["expected_hash"]
    assert evidence["source_config_file_sha256"] == hashlib.sha256(
        Path(config["config_path"]).read_bytes()
    ).hexdigest()
    assert evidence["analysis"] == {
        "tie_margin": config["analysis"]["tie_margin"],
        "bootstrap_samples": config["analysis"]["bootstrap_samples"],
        "permutation_samples": config["analysis"]["permutation_samples"],
        "statistics_seed": config["analysis"]["statistics_seed"],
    }
    assert evidence["budget_preflight"] is None
    assert set(evidence["runtime_versions"]) == {
        "python",
        "numpy",
        "scipy",
        "scikit_learn",
        "pandas",
        "statsmodels",
    }
    assert set(evidence["git"]) == {"commit", "tree", "dirty"}
    assert set(records["status"]) == {"complete"}
    assert records["run_label"].eq("smoke-panel").all()
    assert records["formal_config_sha256"].eq(
        evidence["canonical_config_sha256"]
    ).all()
    assert records["registered_manifest_sha256"].eq(
        evidence["manifest_sha256"]
    ).all()
    assert records["run_python_version"].eq(
        evidence["runtime_versions"]["python"]
    ).all()
    assert records["run_git_tree"].eq(evidence["git"]["tree"]).all()
    assert (~records["preflight_required"]).all()
    assert records["preflight_passed"].isna().all()
    assert records["preflight_receipt_sha256"].isna().all()
    assert records["preflight_git_commit"].isna().all()
    assert records["preflight_git_tree"].isna().all()
    assert set(records["actuator_mode"]) == set(exp26.MODES)
    assert records["statistics_unit"].eq("seed").all()
    assert records["split_unit"].eq("block").all()
    assert (~records["time_points_randomly_split"]).all()
    assert records["readout_fit_train_only"].all()
    assert records["readout_shared_across_modes"].all()
    assert records["paired_noise_across_modes"].all()
    assert records["demand_marginal_decomposition_valid"].all()
    assert records["generator_state_input_cross_moment_zero_by_construction"].all()
    assert (~records["amplitudes_equalized_by_demand"]).all()
    assert (~records["effective_corrections_dale_constrained"]).all()
    assert records["functional_budget_valid"].all()
    active = records[records["actuator_mode"] != "frozen"]
    assert active["functional_budget_l2_relative_error"].max() <= 1e-8
    assert records.groupby("generator_id")["training_fingerprint"].nunique().max() == 1
    assert records.groupby("generator_id")["test_tape_fingerprint"].nunique().max() == 1
    assert records.groupby("generator_id")["base_recurrent_fingerprint"].nunique().max() == 1
    generators = records.drop_duplicates("generator_id")
    middle = generators[generators["alpha"] == 0.5]
    assert np.max(np.abs(middle["chi"] - middle["alpha"])) > 0.2


def test_smoke_endpoint_roles_follow_task_demand_without_manual_alignment(
    tmp_path: Path,
) -> None:
    records = _records(exp26.run_seed(_config(), 9000, tmp_path))
    pivot = records.pivot(
        index=["generator_id", "alpha", "chi"],
        columns="actuator_mode",
        values="test_balanced_accuracy",
    ).reset_index()
    input_cells = pivot[pivot["alpha"] == 0.0]
    state_cells = pivot[pivot["alpha"] == 1.0]
    np.testing.assert_allclose(input_cells["routing"], 1.0)
    np.testing.assert_allclose(state_cells["low_rank"], 1.0)
    assert float(np.mean(input_cells["routing"] - input_cells["low_rank"])) >= 0.4
    assert float(
        np.mean(state_cells["low_rank"] - state_cells[["routing", "gain"]].max(axis=1))
    ) >= 0.3
    advantage = pivot["low_rank"] - pivot[["routing", "gain"]].max(axis=1)
    assert float(pd.Series(pivot["chi"]).corr(pd.Series(advantage), method="spearman")) > 0.7


def test_clean_v2_preflight_is_fully_bound_into_formal_evidence(
    clean_formal_preflight: tuple[dict[str, object], Path, dict[str, object]],
) -> None:
    config, receipt_path, run_git = clean_formal_preflight
    cells = exp26._manifest(config)
    binding = exp26.validate_budget_preflight_receipt(
        config,
        cells,
        receipt_path,
        current_git=run_git,
    )
    assert set(binding) == {
        "required",
        "receipt_schema",
        "receipt_sha256",
        "preflight_passed",
        "registered_config_sha256",
        "manifest_sha256",
        "observed_required_scale_max",
        "policy_required_scale_max",
        "derived_ceiling",
        "provenance_clean",
        "provenance_stable_during_run",
        "git_commit",
        "git_tree",
    }
    assert binding["receipt_schema"] == exp26.BUDGET_PREFLIGHT_SCHEMA_VERSION
    assert binding["preflight_passed"] is True
    assert binding["provenance_clean"] is True
    assert binding["provenance_stable_during_run"] is True
    assert binding["registered_config_sha256"] == exp26.canonical_config_sha256(
        config
    )
    assert binding["manifest_sha256"] == config["manifest"]["expected_hash"]
    assert binding["observed_required_scale_max"] == pytest.approx(
        config["budget_preflight"]["required_scale_max"]
    )
    assert binding["derived_ceiling"] == config["actuator"]["max_scale"]
    assert len(str(binding["receipt_sha256"])) == 64

    provenance = exp26.build_evidence_provenance(
        config,
        manifest_sha256=str(binding["manifest_sha256"]),
        budget_preflight=binding,
        run_git=run_git,
    )
    row = exp26.evidence_row_fields(provenance, run_label="formal-panel")
    assert provenance["budget_preflight"] == binding
    assert row["preflight_required"] is True
    assert row["preflight_passed"] is True
    assert row["preflight_receipt_sha256"] == binding["receipt_sha256"]
    assert row["preflight_git_commit"] == binding["git_commit"]
    assert row["preflight_git_tree"] == binding["git_tree"]


def test_formal_missing_receipt_label_or_registered_seed_fails_before_artifacts(
    tmp_path: Path,
) -> None:
    config = _config("formal")
    cases = (
        {
            "seed": 0,
            "run_label": "formal-panel",
            "preflight_receipt": None,
            "match": "preflight-receipt",
        },
        {
            "seed": 0,
            "run_label": None,
            "preflight_receipt": tmp_path / "missing-receipt",
            "match": "run_label",
        },
        {
            "seed": 31,
            "run_label": "formal-panel",
            "preflight_receipt": tmp_path / "missing-receipt",
            "match": "not registered",
        },
    )
    for index, case in enumerate(cases):
        results_root = tmp_path / f"runs-{index}"
        with pytest.raises(ValueError, match=str(case["match"])):
            exp26.run_seed(
                config,
                int(case["seed"]),
                results_root,
                run_label=case["run_label"],
                preflight_receipt=case["preflight_receipt"],
            )
        assert not results_root.exists()

    smoke_root = tmp_path / "smoke-unregistered"
    with pytest.raises(ValueError, match="not registered"):
        exp26.run_seed(_config(), 31, smoke_root)
    assert not smoke_root.exists()


def test_dirty_or_tampered_preflight_fails_before_formal_artifacts(
    clean_formal_preflight: tuple[dict[str, object], Path, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, receipt_path, run_git = clean_formal_preflight
    dirty_path = tmp_path / "dirty-receipt"
    shutil.copytree(receipt_path, dirty_path)
    code_path = dirty_path / "code_tree_provenance.json"
    summary_path = dirty_path / "preflight_summary.json"
    code = json.loads(code_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    code["git_dirty"] = True
    code["end_snapshot"]["git_dirty"] = True
    summary["code_tree_provenance"] = code
    summary["provenance_clean"] = False
    summary["preflight_passed"] = False
    code_path.write_text(json.dumps(code, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp26, "git_identity", lambda: run_git)
    results_root = tmp_path / "formal-runs"
    with pytest.raises(ValueError, match="not clean"):
        exp26.run_seed(
            config,
            0,
            results_root,
            run_label="formal-panel",
            preflight_receipt=dirty_path,
        )
    assert not results_root.exists()

    tampered_path = tmp_path / "tampered-receipt"
    shutil.copytree(receipt_path, tampered_path)
    (tampered_path / "manifest.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="manifest binding"):
        exp26.run_seed(
            config,
            0,
            results_root,
            run_label="formal-panel",
            preflight_receipt=tampered_path,
        )
    assert not results_root.exists()

    leaky_path = tmp_path / "held-out-access-receipt"
    shutil.copytree(receipt_path, leaky_path)
    leaky_summary_path = leaky_path / "preflight_summary.json"
    leaky_summary = json.loads(leaky_summary_path.read_text(encoding="utf-8"))
    leaky_summary["validation_behavior_accessed"] = True
    leaky_summary_path.write_text(
        json.dumps(leaky_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train-only capability"):
        exp26.run_seed(
            config,
            0,
            results_root,
            run_label="formal-panel",
            preflight_receipt=leaky_path,
        )
    assert not results_root.exists()


def test_formal_git_change_during_seed_marks_attempt_failed(
    clean_formal_preflight: tuple[dict[str, object], Path, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, receipt_path, run_git = clean_formal_preflight
    changed_git = {**run_git, "dirty": True}
    identities = iter((run_git, changed_git))
    monkeypatch.setattr(exp26, "git_identity", lambda: next(identities))

    def fail_setup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("skip expensive setup in git-stability test")

    monkeypatch.setattr(exp26, "_setup_generator", fail_setup)
    results_root = tmp_path / "formal-git-change"
    with pytest.raises(RuntimeError, match="git identity changed"):
        exp26.run_seed(
            config,
            0,
            results_root,
            run_label="formal-panel",
            preflight_receipt=receipt_path,
        )
    statuses = list(results_root.rglob("status.json"))
    assert len(statuses) == 1
    status = json.loads(statuses[0].read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert "git identity changed" in status["error"]
