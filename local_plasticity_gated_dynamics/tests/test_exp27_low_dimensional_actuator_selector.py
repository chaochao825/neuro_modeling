"""Contracts for the frozen-dictionary Exp27 selector experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments import exp27_low_dimensional_actuator_selector as exp27
from experiments.common import load_json_config
from src.data.actuator_selector_dataset import build_outer_seed_loso


ROOT = Path(__file__).resolve().parents[1]


def _config(profile: str = "smoke") -> dict[str, object]:
    return load_json_config(
        ROOT / "configs" / profile / "exp27_low_dimensional_actuator_selector.json"
    )


def _records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    )


def _canonical_receipt_hash(receipt: dict[str, object]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_exp27_profiles_freeze_seed_hash_and_hyperparameter_contracts() -> None:
    smoke = _config()
    formal = _config("formal")
    assert exp27._validate_config(smoke) == (9000, 9001)
    assert exp27._validate_config(formal) == tuple(range(30))
    assert smoke["protocol_version"] == exp27.PROTOCOL_VERSION
    assert formal["protocol_version"] == exp27.PROTOCOL_VERSION
    assert smoke["dictionary"] == list(exp27.CANDIDATE_MODES)
    assert smoke["selectors"] == list(exp27.SELECTORS)
    assert "alpha" not in smoke["feature_columns"]
    assert "demand_cross_relative_magnitude" not in smoke["feature_columns"]
    assert formal["local_selector"] == {
        "epochs": 200,
        "learning_rate": 0.05,
        "temperature": 1.0,
        "teacher_temperature": 0.05,
        "l2": 1e-4,
        "eligibility_decay": 0.8,
        "belief_retention": 0.8,
    }
    assert formal["gru_selector"] == {
        "hidden_dim": 8,
        "epochs": 200,
        "learning_rate": 0.02,
        "weight_decay": 1e-4,
        "teacher_temperature": 0.05,
        "device": "cpu",
        "deterministic": True,
    }
    assert formal["analysis"]["bootstrap_samples"] == 20_000
    assert formal["analysis"]["permutation_samples"] == 100_000
    assert formal["required_run_label"] == "exp27-formal-v1"


def test_exp27_source_and_loso_fold_are_seed_and_composition_safe() -> None:
    config = _config()
    source, candidates, receipt = exp27._validate_frozen_source(config)
    fold = build_outer_seed_loso(source, outer_seed=9000)
    assert source.unique_seeds == (9000, 9001)
    assert set(fold.train_seeds) == {9001}
    assert set(fold.test_seeds) == {9000}
    assert fold.train_raw_features.shape == (12, 7)
    assert fold.test_raw_features.shape == (12, 7)
    assert int(np.sum(fold.test_unseen_composition)) == 4
    assert int(np.sum(fold.test_composition_overlap)) == 8
    assert np.array_equal(
        fold.test_unseen_composition,
        ~fold.test_composition_overlap,
    )
    assert candidates.shape[0] == 2 * 24 * 3
    assert (
        receipt["raw_metrics_sha256"]
        == (config["source_exp26"]["expected_raw_metrics_sha256"])
    )
    assert receipt["receipt_sha256"] == _canonical_receipt_hash(receipt)


def test_one_smoke_outer_seed_retains_full_schema_receipts_and_audits(
    tmp_path: Path,
) -> None:
    config = _config()
    raw_path = Path(config["source_exp26"]["raw_metrics_path"])
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    path = exp27.run_seed(
        config,
        9000,
        tmp_path,
        run_label="exp27-smoke-test",
    )
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before_hash
    records = _records(path)
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert len(planned) == len(records) == 12 * 4
    assert set(records["status"]) == {"complete"}
    assert set(records["selector"]) == set(exp27.SELECTORS)
    assert records.groupby("generator_id")["selector"].nunique().eq(4).all()
    assert records["generator_split"].eq("heldout").all()
    assert records["source_seed"].eq(records["outer_seed"]).all()
    assert records["outer_seed"].eq(9000).all()
    assert records["strict_unseen_composition"].sum() == 4 * 4
    assert records["primary_endpoint_eligible"].equals(
        records["strict_unseen_composition"]
    )
    assert records["outer_seed_excluded_from_training"].all()
    assert records["training_source_seed_count"].eq(1).all()
    assert (
        records["training_source_seeds"]
        .map(lambda value: tuple(value) == (9001,))
        .all()
    )
    assert records["train_split"].eq("discovery").all()
    assert records["train_endpoint"].eq("validation_balanced_accuracy").all()
    assert records["test_endpoint"].eq("test_balanced_accuracy").all()
    assert records["carrier_frozen"].all()
    assert records["actuator_dictionary_frozen"].all()
    assert (~records["actuator_basis_trained"]).all()
    assert (~records["readout_trained"]).all()
    assert (~records["hidden_belief_inference_enabled"]).all()
    assert records["task_descriptor_cues_privileged"].all()
    assert (~records["update_budget_matched_across_selectors"]).all()

    local = records[records["selector"] == "local_three_factor"]
    gru = records[records["selector"] == "gru_bptt"]
    assert (~local["used_autograd"]).all() and (~local["used_bptt"]).all()
    assert gru["used_autograd"].all() and gru["used_bptt"].all()
    assert local["local_learning_main_model"].all()
    assert gru["bptt_baseline_isolated"].all()
    assert (local["plasticity_l1"] > 0.0).all()
    assert (gru["plasticity_l2"] > 0.0).all()
    assert local["selector_teacher_counterfactual"].all()
    assert (~local["selector_teacher_is_scalar_reward"]).all()

    for _, group in records.groupby("generator_id", sort=False):
        utilities = group.iloc[0][
            [
                "candidate_routing_utility",
                "candidate_gain_utility",
                "candidate_low_rank_utility",
            ]
        ].to_numpy(dtype=np.float64)
        expected_oracle = exp27.CANDIDATE_MODES[int(np.argmax(utilities))]
        assert group["oracle_mode"].eq(expected_oracle).all()
        oracle = group[group["selector"] == "oracle"].iloc[0]
        assert oracle["mode_selected"] == expected_oracle
        assert oracle["selection_correct"]
    means = records.iloc[0][
        [
            "train_mean_candidate_routing_utility",
            "train_mean_candidate_gain_utility",
            "train_mean_candidate_low_rank_utility",
        ]
    ].to_numpy(dtype=np.float64)
    expected_fixed = exp27.CANDIDATE_MODES[int(np.argmax(means))]
    assert records["fixed_best_mode"].eq(expected_fixed).all()
    assert (
        records.loc[records["selector"] == "fixed_best", "mode_selected"]
        .eq(expected_fixed)
        .all()
    )

    normalizer = json.loads(
        (path / "normalizer_receipt.json").read_text(encoding="utf-8")
    )
    selector_receipts = json.loads(
        (path / "selector_training_receipts.json").read_text(encoding="utf-8")
    )
    assert normalizer["schema_version"] == "exp27_selector_normalizer_v1"
    assert normalizer["fit_scope"] == exp27.TRAINING_SCOPE
    assert normalizer["train_n"] == 12
    assert len(normalizer["center"]) == len(normalizer["scale"]) == 7
    assert normalizer["receipt_sha256"] == _canonical_receipt_hash(normalizer)
    assert selector_receipts["schema_version"] == "exp27_selector_training_receipts_v1"
    assert selector_receipts["receipt_sha256"] == _canonical_receipt_hash(
        selector_receipts
    )
    assert selector_receipts["local_three_factor"]["used_bptt"] is False
    assert selector_receipts["gru_bptt"]["used_bptt"] is True
    assert selector_receipts["local_three_factor"]["fit_receipt"]["epochs"] == 20
    assert selector_receipts["gru_bptt"]["fit_receipt"]["epochs"] == 20
    for selector in ("local_three_factor", "gru_bptt"):
        model_receipt = selector_receipts[selector]
        assert len(model_receipt["test_generator_ids"]) == 12
        assert np.asarray(model_receipt["test_probabilities"]).shape == (12, 3)
        assert len(model_receipt["test_decision_fingerprint"]) == 64
    assert records["normalizer_receipt_sha256"].nunique() == 1
    assert records["selector_training_receipts_sha256"].nunique() == 1
    assert (
        records["exp27_config_sha256"].eq(exp27.canonical_config_sha256(config)).all()
    )


def test_learned_selectors_are_deterministic_and_training_fold_normalized() -> None:
    config = _config()
    source, _, _ = exp27._validate_frozen_source(config)
    fold = build_outer_seed_loso(source, 9000)
    first = exp27._fit_learned_selectors(fold, config, seed=9000)
    second = exp27._fit_learned_selectors(fold, config, seed=9000)
    for selector in ("local_three_factor", "gru_bptt"):
        np.testing.assert_array_equal(first[0][selector], second[0][selector])
        assert first[1][selector] == second[1][selector]
        assert (
            first[2][selector]["receipt_sha256"]
            == (second[2][selector]["receipt_sha256"])
        )
    assert first[3]["fingerprint"] == second[3]["fingerprint"]
    assert first[4] == second[4] == {}


def test_tampered_source_conclusion_fails_closed_and_retains_all_cells(
    tmp_path: Path,
) -> None:
    config = _config()
    original = Path(config["source_exp26"]["conclusion_path"])
    tampered = tmp_path / "tampered_conclusion.json"
    shutil.copyfile(original, tampered)
    with tampered.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    config["source_exp26"]["conclusion_path"] = str(tampered)
    with pytest.raises(ValueError, match="conclusion SHA-256 mismatch"):
        exp27.run_seed(config, 9000, tmp_path / "results")
    attempts = list(
        (tmp_path / "results" / "runs" / exp27.EXPERIMENT / "seed_9000").iterdir()
    )
    assert len(attempts) == 1
    path = attempts[0]
    records = _records(path)
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert len(records) == 12 * 4
    assert set(records["status"]) == {"failed"}
    assert status["status"] == "failed"


def test_formal_requires_registered_label_and_clean_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("formal")
    with pytest.raises(ValueError, match="registered shared run_label"):
        exp27.run_seed(config, 0, tmp_path)
    monkeypatch.setattr(
        exp27,
        "git_identity",
        lambda: {"commit": "a" * 40, "tree": "b" * 40, "dirty": True},
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        exp27.run_seed(
            config,
            0,
            tmp_path,
            run_label="exp27-formal-v1",
        )
    attempts = list((tmp_path / "runs" / exp27.EXPERIMENT / "seed_0000").iterdir())
    assert len(attempts) == 1
    records = _records(attempts[0])
    assert len(records) == 44 * 4
    assert set(records["status"]) == {"failed"}


def test_registered_contract_rejects_feature_or_source_hash_tuning() -> None:
    config = _config()
    config["feature_columns"] = [*config["feature_columns"], "alpha"]
    with pytest.raises(ValueError, match="feature contract"):
        exp27._validate_config(config)
    config = _config()
    config["source_exp26"]["expected_raw_metrics_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source registry"):
        exp27._validate_config(config)
