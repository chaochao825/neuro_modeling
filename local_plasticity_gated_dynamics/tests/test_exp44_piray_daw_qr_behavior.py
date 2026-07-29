from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.exp44_piray_daw_qr_behavior as exp44
from experiments.exp44_piray_daw_qr_behavior import (
    AUTOCOV,
    FACTORIZED,
    FIXED,
    METHODS,
    ORACLE,
    PARTICLE,
    TOTAL,
    Candidate,
    DEPLOYABLE_METHODS,
    _build_candidates,
    _candidate_id,
    compare_methods,
    cross_validated_behavior,
    development_decision,
    run,
    validate_config,
)
from src.data.piray_daw import PirayDawDataset
from src.models.piray_daw_qr_controller import run_fixed_gain


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "development" / "exp44_piray_daw_qr_behavior_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _dataset() -> PirayDawDataset:
    n, trials, blocks = 223, 50, 4
    bag = np.tile(np.linspace(55.0, 65.0, trials)[:, None], (1, blocks))
    bucket = np.empty((n, trials, blocks))
    bucket[:, 0, :] = 60.0
    for trial in range(trials - 1):
        bucket[:, trial + 1, :] = bucket[:, trial, :] + 0.5 * (
            bag[None, trial, :] - bucket[:, trial, :]
        )
    return PirayDawDataset(
        experiment=1,
        bucket=bucket,
        response_time=np.full((n, trials, blocks), 500.0),
        randomization_order=np.tile(np.arange(4), (n, 1)),
        age=np.full(n, 30.0),
        gender=tuple("x" for _ in range(n)),
        bag=bag,
        bird=bag,
        true_process_variance=np.array([4.0, 49.0, 4.0, 49.0]),
        true_observation_variance=np.array([16.0, 16.0, 64.0, 64.0]),
        source_path=Path("fixture.mat"),
        source_sha256="fixture",
    )


def test_frozen_development_config_is_valid_and_confirmation_locked() -> None:
    config = _config()
    validate_config(config)
    config["confirmation_lock"]["allow_confirmation"] = True
    with pytest.raises(ValueError, match="remain locked"):
        validate_config(config)


def test_candidate_ids_are_parameter_order_invariant() -> None:
    assert _candidate_id("x", {"a": 1, "b": 2}) == _candidate_id(
        "x", {"b": 2, "a": 1}
    )


def test_cross_validation_holds_out_participants_and_retains_all_methods() -> None:
    dataset = _dataset()
    candidates: dict[str, list[Candidate]] = {}
    for method in METHODS:
        gain = 0.5 if method == FACTORIZED else 0.4
        trace = run_fixed_gain(dataset.bag, gain=gain)
        parameters = {"gain": gain}
        candidates[method] = [
            Candidate(method, _candidate_id(method, parameters), parameters, trace)
        ]
    outputs, folds = cross_validated_behavior(dataset, candidates, _config())
    participant, cell, scores, selected = outputs
    assert len(participant) == dataset.n_participants * len(METHODS)
    assert len(cell) == dataset.n_participants * len(METHODS) * 4
    assert set(participant["method"]) == set(METHODS)
    assert folds["participant_id"].nunique() == dataset.n_participants
    assert scores.groupby(["selection_scope", "method"])["selected"].sum().eq(1).all()
    assert set(selected) == set(METHODS)


def test_deployable_candidates_ignore_bird_true_labels_and_bucket_outcomes() -> None:
    dataset = _dataset()
    config = _config()
    for key in (
        "fixed_gain_grid",
        "initial_q_grid",
        "initial_r_grid",
        "factorized_q_rate_grid",
        "factorized_r_rate_grid",
        "initial_total_variance_grid",
        "total_rate_grid",
        "total_q_fraction_grid",
        "autocovariance_decay_grid",
        "autocovariance_prior_mass_grid",
        "particle_mu_q_grid",
        "particle_mu_r_grid",
        "particle_log_step_grid",
    ):
        config["selection"][key] = config["selection"][key][:1]
    config["selection"]["particle_count"] = 32
    config["selection"]["particle_seeds"] = [44001]
    permutation = np.array([3, 2, 1, 0])
    altered = PirayDawDataset(
        experiment=1,
        bucket=dataset.bucket + 7.0,
        response_time=dataset.response_time,
        randomization_order=dataset.randomization_order,
        age=dataset.age,
        gender=dataset.gender,
        bag=dataset.bag,
        bird=dataset.bird + 1000.0,
        true_process_variance=dataset.true_process_variance[permutation],
        true_observation_variance=dataset.true_observation_variance[permutation],
        source_path=dataset.source_path,
        source_sha256=dataset.source_sha256,
    )
    original_candidates = _build_candidates(dataset, config)
    altered_candidates = _build_candidates(altered, config)
    for method in DEPLOYABLE_METHODS:
        np.testing.assert_array_equal(
            original_candidates[method][0].trace.gain,
            altered_candidates[method][0].trace.gain,
        )


def _passing_decision_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    for metric in ("conditional_update_nll", "conditional_update_mse"):
        comparison_rows.extend(
            [
                {"baseline": FIXED, "metric": metric, "mean_gain": 0.010, "ci_low": 0.006},
                {"baseline": TOTAL, "metric": metric, "mean_gain": 0.008, "ci_low": 0.003},
                {"baseline": AUTOCOV, "metric": metric, "mean_gain": 0.002, "ci_low": -0.001},
                {"baseline": PARTICLE, "metric": metric, "mean_gain": 0.001, "ci_low": -0.001},
                {"baseline": ORACLE, "metric": metric, "mean_gain": 0.001, "ci_low": -0.001},
            ]
        )
    comparisons = pd.DataFrame(comparison_rows)
    rows = []
    for participant in range(5):
        for block in range(4):
            rows.extend(
                [
                    {
                        "participant_id": participant,
                        "block_id": block,
                        "method": TOTAL,
                        "conditional_update_nll": 1.1,
                    },
                    {
                        "participant_id": participant,
                        "block_id": block,
                        "method": FACTORIZED,
                        "conditional_update_nll": 1.0,
                    },
                ]
            )
    diagnostics = pd.DataFrame(
        {
            "method": [FACTORIZED] * 4,
            "true_process_variance": [4.0, 49.0, 4.0, 49.0],
            "true_observation_variance": [16.0, 16.0, 64.0, 64.0],
            "mean_gain": [0.3, 0.6, 0.2, 0.5],
        }
    )
    return comparisons, pd.DataFrame(rows), diagnostics


def test_development_gate_is_a_strict_conjunction() -> None:
    comparisons, cells, diagnostics = _passing_decision_inputs()
    decision = development_decision(comparisons, cells, diagnostics, _config())
    assert decision["development_gate_passed"]
    assert decision["confirmation_unlocked"]
    assert not decision["popgym_unlocked"]
    comparisons.loc[
        (comparisons["baseline"] == TOTAL)
        & (comparisons["metric"] == "conditional_update_nll"),
        "mean_gain",
    ] = 0.0
    decision = development_decision(comparisons, cells, diagnostics, _config())
    assert not decision["development_gate_passed"]
    assert not decision["confirmation_unlocked"]


def test_method_comparison_uses_one_row_per_participant() -> None:
    rows = []
    for participant in range(20):
        for method_index, method in enumerate(METHODS):
            rows.append(
                {
                    "participant_id": participant,
                    "method": method,
                    "conditional_update_nll": 1.0 + 0.01 * method_index,
                    "conditional_update_mse": 2.0 + 0.02 * method_index,
                }
            )
    config = _config()
    config["analysis"]["bootstrap_resamples"] = 100
    comparisons = compare_methods(pd.DataFrame(rows), config)
    assert set(comparisons["baseline"]) == set(METHODS) - {FACTORIZED}
    assert comparisons["n_participants"].eq(20).all()


def test_execute_writes_complete_failure_preserving_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset()
    candidates: dict[str, list[Candidate]] = {}
    for method in METHODS:
        gain = 0.5 if method == FACTORIZED else 0.4
        parameters = {"gain": gain}
        candidates[method] = [
            Candidate(
                method,
                _candidate_id(method, parameters),
                parameters,
                run_fixed_gain(dataset.bag, gain=gain),
            )
        ]
    monkeypatch.setattr(exp44, "load_piray_daw", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(exp44, "_build_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(exp44, "_environment", lambda: {"git": {"dirty": False}})
    config_path = tmp_path / "config.json"
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "result"
    summary = run(config_path, output)
    assert summary["stage"] == "development"
    assert not summary["claim_upgrade_allowed"]
    required = {
        "config.json",
        "environment.json",
        "participant_folds.csv",
        "participant_metrics.csv",
        "cell_metrics.csv",
        "candidate_scores.csv",
        "comparisons.csv",
        "trace_diagnostics.csv",
        "selected_candidates.json",
        "summary.json",
        "report.md",
        "run.log",
        "manifest.json",
        "exp44_piray_daw_qr_behavior.png",
        "exp44_piray_daw_qr_behavior.pdf",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    with pytest.raises(FileExistsError):
        run(config_path, output)


def test_execute_rejects_dirty_start_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp44, "_environment", lambda: {"git": {"dirty": True}})
    config_path = tmp_path / "config.json"
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "dirty_result"
    with pytest.raises(RuntimeError, match="clean git worktree"):
        run(config_path, output)
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert failure["type"] == "RuntimeError"
    assert manifest["status"] == "failed"
