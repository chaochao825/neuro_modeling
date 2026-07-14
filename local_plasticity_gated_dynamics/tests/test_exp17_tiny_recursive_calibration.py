from __future__ import annotations

import json

import pytest

from experiments.common import load_json_config
from experiments.exp16_tiny_recursive_sudoku import calibration_candidate_sha256
from experiments.exp17_tiny_recursive_calibration import run_seed
from scripts.summarize_exp17_tiny_recursive_calibration import (
    publish_summary,
    summarize_runs,
)
from src.data.structured_protocol import TargetStore


def _rows(path):
    return [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_exp17_calibrates_without_invoking_any_hidden_target_scorer(
    tmp_path, monkeypatch
) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")

    def forbidden_score(self, task, prediction):  # pragma: no cover - must not run
        raise AssertionError("Exp17 must not invoke TargetStore.score")

    monkeypatch.setattr(TargetStore, "score", forbidden_score)
    run_path = run_seed(config, 0, tmp_path)
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"

    rows = _rows(run_path)
    candidates = [row for row in rows if row["stage"] == "calibration_candidate"]
    selections = [row for row in rows if row["stage"] == "calibration_selection"]
    assert len(candidates) == 2
    assert len(selections) == 1
    assert not any(row["stage"] == "task_test" for row in rows)
    assert all(row["test_data_used_for_fit_or_selection"] is False for row in rows)
    assert {
        (row["training_data_sha256"], row["epoch_permutation_sha256"])
        for row in candidates
    } == {
        (
            candidates[0]["training_data_sha256"],
            candidates[0]["epoch_permutation_sha256"],
        )
    }
    assert len({row["initialization_sha256"] for row in candidates}) == 1
    blank_candidate = next(
        row for row in candidates if row["candidate"] == "blank_only_reference"
    )
    expected_hash_config = {
        **config,
        "model": {**config["model"]},
        "training": {
            **config["training"],
            **config["candidates"]["blank_only_reference"]["training"],
        },
        "augmentations_per_task": 0,
    }
    assert blank_candidate["candidate_config_sha256"] == (
        calibration_candidate_sha256(expected_hash_config)
    )
    provenance = json.loads(
        (run_path / "source_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["dataset_adapter_loaded_test_records"] is True
    assert provenance["test_prediction_array_requested"] is False
    assert provenance["public_test_prediction_adapter_called"] is False
    assert provenance["hidden_target_scorer_called"] is False
    assert provenance["test_targets_remained_opaque_in_target_store"] is True


def test_exp17_rejects_a_non_test_free_selection_contract(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")
    config["selection_contract"]["test_access_forbidden"] = False
    with pytest.raises(ValueError, match="test-free"):
        run_seed(config, 0, tmp_path)


def test_exp17_rejects_path_unsafe_candidate_names(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")
    config["candidates"]["../escape"] = config["candidates"].pop(
        "all_tokens_candidate"
    )
    with pytest.raises(ValueError, match="path-safe"):
        run_seed(config, 0, tmp_path)


@pytest.mark.parametrize("reserved", ["CON", "nul", "COM1"])
def test_exp17_rejects_windows_reserved_candidate_names(tmp_path, reserved) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")
    config["candidates"][reserved] = config["candidates"].pop(
        "all_tokens_candidate"
    )
    with pytest.raises(ValueError, match="path-safe"):
        run_seed(config, 0, tmp_path)


def test_exp17_rejects_casefold_colliding_candidate_names(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")
    config["candidates"]["BLANK_ONLY_REFERENCE"] = config["candidates"].pop(
        "all_tokens_candidate"
    )
    with pytest.raises(ValueError, match="path-safe"):
        run_seed(config, 0, tmp_path)


def test_exp17_cross_seed_summary_is_validation_only_and_immutable(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")
    first = run_seed(config, 0, tmp_path / "runs")
    second = run_seed(config, 1, tmp_path / "runs")
    outputs = publish_summary(
        [first, second], tmp_path / "publication", prefix="exp17_test"
    )
    decision = json.loads(outputs["decision"].read_text(encoding="utf-8"))
    assert decision["status"] == "frozen_validation_only"
    assert decision["test_prediction_array_requested"] is False
    assert decision["hidden_target_scorer_called"] is False
    assert decision["confirmation_test_still_required"] is True
    assert decision["calibration_code_sha256"]
    assert "hidden-target scorer was not called" in outputs["report"].read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="immutable"):
        publish_summary([first, second], tmp_path / "publication", prefix="exp17_test")
    second_config = json.loads((second / "config.json").read_text(encoding="utf-8"))
    second_config["semantic_config_sha256"] = "0" * 64
    (second / "config.json").write_text(
        json.dumps(second_config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="semantic Exp17 configs differ"):
        summarize_runs([first, second])


def test_exp17_all_candidate_failures_are_publishable_but_not_frozen(
    tmp_path, monkeypatch
) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate calibration failure")

    monkeypatch.setattr(
        "experiments.exp17_tiny_recursive_calibration.fit_tiny_recursive", fail_fit
    )
    run_path = run_seed(config, 0, tmp_path / "runs")
    assert (run_path / "source_provenance.json").is_file()
    outputs = publish_summary(
        [run_path], tmp_path / "publication", prefix="exp17_failed"
    )
    decision = json.loads(outputs["decision"].read_text(encoding="utf-8"))
    assert decision["status"] == "insufficient_validation_evidence"
    assert decision["selected_candidate"] is None
    assert decision["all_runs_clean"] is False
    assert decision["all_candidates_complete"] is False


def test_exp17_dataset_failure_still_writes_a_publishable_provenance(tmp_path) -> None:
    config = load_json_config("configs/smoke/exp17_tiny_recursive_calibration.json")
    config["synthetic_fixture"]["n_train_tasks"] = 1
    run_path = run_seed(config, 0, tmp_path / "runs")
    provenance = json.loads(
        (run_path / "source_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "failed"
    assert provenance["test_data_used_for_fit_or_selection"] is False
    outputs = publish_summary(
        [run_path], tmp_path / "publication", prefix="exp17_dataset_failed"
    )
    decision = json.loads(outputs["decision"].read_text(encoding="utf-8"))
    assert decision["status"] == "insufficient_validation_evidence"
