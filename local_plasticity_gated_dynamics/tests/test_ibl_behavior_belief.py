import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.common import load_json_config
from experiments.exp11_ibl_behavior_belief import _configured_session_specs, run_seed
from src.analysis.ibl_behavior_metrics import (
    binary_context_metrics,
    causal_behavior_features,
    fit_behavior_logistic,
    oracle_ceiling_beliefs,
)
from src.data.ibl_behavior import (
    ExponentialHistoryBelief,
    IBLBehaviorDataError,
    IBLBehaviorObservations,
    LearnedCategoricalHMM,
    NoMemoryBelief,
    _forward_backward,
    causal_exponential_trace,
    contiguous_block_split,
    load_ibl_behavior_table,
)


def _trial_table(*, n_blocks: int = 10, block_size: int = 12) -> pd.DataFrame:
    block_levels = np.concatenate(
        [np.array([0.5]), np.resize(np.array([0.2, 0.8]), n_blocks - 1)]
    )
    probabilities = np.repeat(block_levels, block_size)
    rng = np.random.default_rng(314159)
    left = rng.random(probabilities.size) < probabilities
    contrasts = np.resize(np.array([0.0, 0.25, 0.5, 1.0]), probabilities.size)
    # Deterministic alternation guarantees both response classes in every long fold.
    choice = np.where(np.arange(probabilities.size) % 2 == 0, -1, 1)
    return pd.DataFrame(
        {
            "contrastLeft": np.where(left, contrasts, np.nan),
            "contrastRight": np.where(~left, contrasts, np.nan),
            "choice": choice,
            "feedbackType": np.where(np.arange(probabilities.size) % 3, 1, -1),
            "probabilityLeft": probabilities,
        }
    )


def _session(*, n_blocks: int = 10, block_size: int = 12):
    return load_ibl_behavior_table(
        _trial_table(n_blocks=n_blocks, block_size=block_size),
        eid=f"eid-{n_blocks}",
        animal_id="mouse-test",
    )


def test_local_table_loader_separates_gate_observation_from_scoring_truth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trials.csv"
    _trial_table().to_csv(path, index=False)
    session = load_ibl_behavior_table(path, eid="eid-local", animal_id="mouse-local")
    observation = session.observations
    assert set(vars(observation)) == {"trial_ids", "stimulus_side"}
    assert not hasattr(observation, "probability_left")
    assert session.n_blocks == 10
    assert not observation.stimulus_side.flags.writeable
    assert np.array_equal(np.unique(session.context_labels), np.array([-1, 0, 1]))


def test_loader_rejects_non_ibl_context_and_ambiguous_stimulus_side() -> None:
    broken = _trial_table()
    broken.loc[0, "probabilityLeft"] = 0.65
    with pytest.raises(IBLBehaviorDataError, match="public IBL levels"):
        load_ibl_behavior_table(broken, eid="bad", animal_id="mouse")
    broken = _trial_table()
    broken.loc[0, "probabilityLeft"] = 0.2001
    with pytest.raises(IBLBehaviorDataError, match="public IBL levels"):
        load_ibl_behavior_table(broken, eid="bad", animal_id="mouse")
    with pytest.raises(IBLBehaviorDataError, match="feedbackType"):
        load_ibl_behavior_table(
            _trial_table().drop(columns="feedbackType"),
            eid="bad",
            animal_id="mouse",
        )
    broken = _trial_table()
    broken.loc[0, ["contrastLeft", "contrastRight"]] = [0.5, 0.5]
    with pytest.raises(IBLBehaviorDataError, match="exactly one"):
        load_ibl_behavior_table(broken, eid="bad", animal_id="mouse")


def test_loader_preserves_source_trials_and_fails_closed_on_gaps() -> None:
    frame = _trial_table()
    frame.insert(0, "source_trial_index", np.arange(len(frame)) + 100)
    official_mask = np.ones(len(frame), dtype=bool)
    official_mask[-3:] = False
    frame["official_bwm_mask"] = official_mask
    session = load_ibl_behavior_table(frame, eid="source", animal_id="mouse")
    np.testing.assert_array_equal(session.trial_ids, np.arange(len(frame)) + 100)
    np.testing.assert_array_equal(session.analysis_mask, official_mask)
    assert session.official_bwm_mask_present
    assert not np.any(session.context_score_mask[-3:])

    broken = frame.copy()
    broken.loc[5:, "source_trial_index"] += 1
    with pytest.raises(IBLBehaviorDataError, match="gaps"):
        load_ibl_behavior_table(broken, eid="gap", animal_id="mouse")


def test_public_ibl_choice_sign_maps_plus_one_to_left() -> None:
    frame = _trial_table(n_blocks=3, block_size=4)
    frame["choice"] = np.resize(np.array([1, -1]), len(frame))
    session = load_ibl_behavior_table(frame, eid="sign", animal_id="mouse")
    np.testing.assert_array_equal(
        session.choice_left, (frame["choice"].to_numpy() == 1).astype(int)
    )


def test_split_is_chronological_and_keeps_every_block_whole() -> None:
    session = _session()
    split = contiguous_block_split(
        session.block_ids,
        test_fraction=0.2,
        validation_fraction=0.2,
        min_blocks=5,
    )
    assert np.array_equal(split.train_indices, np.arange(split.train_indices.size))
    assert split.train_indices[-1] < split.dev_indices[0] < split.test_indices[0]
    assert not set(split.train_block_ids) & set(split.dev_block_ids)
    assert not set(split.train_block_ids) & set(split.test_block_ids)
    for block in np.unique(session.block_ids):
        holders = [
            np.any(session.block_ids[indices] == block)
            for indices in (split.train_indices, split.dev_indices, split.test_indices)
        ]
        assert sum(holders) == 1


def test_causal_trace_and_gate_prior_do_not_use_current_or_future_stimulus() -> None:
    session = _session()
    split = contiguous_block_split(session.block_ids)
    trial = split.test_indices[2]
    original = session.observations
    altered_sides = original.stimulus_side.copy()
    altered_sides[trial:] = 1 - altered_sides[trial:]
    altered = IBLBehaviorObservations(original.trial_ids, altered_sides)

    trace_original = causal_exponential_trace(original.stimulus_side, 0.9)
    trace_altered = causal_exponential_trace(altered.stimulus_side, 0.9)
    np.testing.assert_allclose(trace_original[: trial + 1], trace_altered[: trial + 1])

    exponential = ExponentialHistoryBelief(decays=(0.8, 0.9)).fit(
        original, split.train_indices
    )
    hmm = LearnedCategoricalHMM(max_iter=30, n_restarts=1, seed=4).fit(
        original, split.train_indices
    )
    for model in (NoMemoryBelief(), exponential, hmm):
        first = model.predict(original).beliefs
        second = model.predict(altered).beliefs
        np.testing.assert_allclose(first[: trial + 1], second[: trial + 1])


def test_unsupervised_hmm_fits_train_stimuli_only_and_orders_emissions() -> None:
    session = _session()
    split = contiguous_block_split(session.block_ids)
    model = LearnedCategoricalHMM(max_iter=40, n_restarts=2, seed=9).fit(
        session.observations, split.train_indices
    )
    prediction = model.predict(session.observations)
    np.testing.assert_array_equal(
        model.fit_trial_ids_, session.trial_ids[split.train_indices]
    )
    assert np.all(np.diff(model.emission_[:, 1]) >= 0.0)
    assert prediction.beliefs.shape == (session.trial_ids.size, 2)
    assert not prediction.beliefs.flags.writeable
    final_likelihood, _, _ = _forward_backward(
        session.stimulus_side[split.train_indices],
        model.initial_,
        model.transition_,
        model.emission_,
    )
    assert model.train_log_likelihood_ == pytest.approx(final_likelihood)
    with pytest.raises(TypeError, match="IBLBehaviorObservations"):
        model.fit(session, split.train_indices)  # type: ignore[arg-type]


def test_context_scoring_and_behavior_readout_keep_truth_and_fit_scopes_explicit() -> (
    None
):
    session = _session()
    split = contiguous_block_split(session.block_ids)
    prediction = NoMemoryBelief().predict(session.observations)
    context = binary_context_metrics(
        prediction.beliefs,
        session.context_labels,
        indices=split.test_indices[session.context_score_mask[split.test_indices]],
    )
    assert context.nll == pytest.approx(np.log(2.0))
    behavior = fit_behavior_logistic(session, prediction.beliefs, split, seed=2)
    assert set(behavior.fit_trial_ids).issubset(
        set(session.trial_ids[split.train_indices])
    )
    assert not set(behavior.fit_trial_ids) & set(session.trial_ids[split.test_indices])
    features = causal_behavior_features(session, prediction.beliefs)
    assert features.shape == (session.trial_ids.size, 7)
    oracle = oracle_ceiling_beliefs(session.context_labels)
    assert binary_context_metrics(
        oracle,
        session.context_labels,
        indices=split.test_indices[session.context_score_mask[split.test_indices]],
    ).nll == pytest.approx(0.0)


def test_experiment_records_four_session_level_conditions(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp11_ibl_behavior_belief.json")
    config["learned_hmm"] = {
        "max_iter": 30,
        "tolerance": 1e-5,
        "pseudocount": 0.1,
        "n_restarts": 1,
    }
    path = run_seed(config, 0, str(tmp_path), sessions=[_session()])
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert {record["condition"] for record in records} == {
        "no_memory",
        "exponential_history",
        "learned_categorical_hmm",
        "oracle_ceiling",
    }
    assert all(record["status"] == "complete" for record in records)
    assert all(record["statistics_unit"] == "session" for record in records)
    assert all(record["preprocessing_fit_train_only"] for record in records)
    learned = next(
        item for item in records if item["condition"] == "learned_categorical_hmm"
    )
    assert learned["gate_fit_accessed_probabilityLeft"] is False
    assert learned["gate_reset_at_true_boundaries"] is False
    assert learned["known_context_rate_initialization_used"] is True
    assert learned["gate_fit_supervision"] == (
        "task_informed_unsupervised_stimulus_only"
    )
    assert learned["algorithmic_seed_is_statistical_unit"] is False
    assert learned["online_state_updates_from_dev_and_past_test_observations"] is True
    no_memory = next(item for item in records if item["condition"] == "no_memory")
    assert no_memory["no_memory_scope"] == "belief_gate_only_not_behavior_history"
    assert no_memory["eligible_for_context_inference_support"] is False
    oracle = next(item for item in records if item["condition"] == "oracle_ceiling")
    assert oracle["eligible_for_context_inference_support"] is False
    assert oracle["oracle_is_evaluation_only"] is True


def test_cohort_manifest_enforces_session_animal_thresholds(tmp_path: Path) -> None:
    session_paths = []
    rows = []
    for index in range(4):
        path = tmp_path / f"trials_{index}.csv"
        _trial_table().to_csv(path, index=False)
        compact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        session_paths.append(path)
        rows.append(
            {
                "eid": f"eid-{index}",
                "subject": f"mouse-{index // 2}",
                "status": "eligible",
                "compact_table": path.name,
                "cohort_id": "test-cohort",
                "compact_table_sha256": compact_hash,
                "dataset_uuid": f"dataset-{index}",
                "dataset_revision": "2025-03-03",
                "dataset_hash": f"hash-{index}",
                "dataset_qc": "PASS",
                "bwm_repository_commit": "a" * 40,
            }
        )
    manifest = tmp_path / "cohort_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = {
        "cohort_manifest": str(manifest),
        "sessions": [],
        "cohort_requirements": {"minimum_sessions": 4, "minimum_animals": 2},
    }
    specs = _configured_session_specs(config)
    assert len(specs) == 4
    assert {item["animal_id"] for item in specs} == {"mouse-0", "mouse-1"}
    config["cohort_requirements"]["minimum_animals"] = 3
    with pytest.raises(ValueError, match="eligible animals"):
        _configured_session_specs(config)


def test_formal_manifest_provenance_survives_gate_evaluation(
    tmp_path: Path,
) -> None:
    frame = _trial_table()
    frame.insert(0, "source_trial_index", np.arange(len(frame)))
    frame["official_bwm_mask"] = True
    compact_table = tmp_path / "official_trials.csv"
    frame.to_csv(compact_table, index=False)
    compact_hash = hashlib.sha256(compact_table.read_bytes()).hexdigest()
    provenance = {
        "cohort_id": "frozen-test-cohort",
        "compact_table_sha256": compact_hash,
        "dataset_uuid": "dataset-uuid-test",
        "dataset_revision": "2025-03-03",
        "dataset_hash": "dataset-hash-test",
        "dataset_qc": "PASS",
        "bwm_repository_commit": "b" * 40,
    }
    manifest = tmp_path / "cohort_manifest.csv"
    pd.DataFrame(
        [
            {
                "eid": "eid-formal",
                "subject": "mouse-formal",
                "status": "eligible",
                "compact_table": compact_table.name,
                **provenance,
            }
        ]
    ).to_csv(manifest, index=False)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    config = load_json_config("configs/smoke/exp11_ibl_behavior_belief.json")
    config.update(
        profile="formal",
        cohort_manifest=str(manifest),
        sessions=[],
        cohort_requirements={"minimum_sessions": 1, "minimum_animals": 1},
    )
    config["learned_hmm"] = {
        "max_iter": 30,
        "tolerance": 1e-5,
        "pseudocount": 0.1,
        "n_restarts": 1,
    }
    path = run_seed(config, 0, str(tmp_path / "runs"))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_conditions = {
        "no_memory",
        "exponential_history",
        "learned_categorical_hmm",
        "oracle_ceiling",
    }
    assert len(records) == len(expected_conditions)
    assert {record["condition"] for record in records} == expected_conditions
    assert all(record["status"] == "complete" for record in records)
    for record in records:
        assert record["cohort_manifest_sha256"] == manifest_hash
        for name, expected in provenance.items():
            assert record[name] == expected


def test_experiment_preserves_one_failure_row_per_planned_condition(
    tmp_path: Path,
) -> None:
    config = load_json_config("configs/smoke/exp11_ibl_behavior_belief.json")
    path = run_seed(
        config,
        1,
        str(tmp_path),
        sessions=[_session(n_blocks=3, block_size=10)],
    )
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert all(record["status"] == "failed" for record in records)
    assert all(record["statistics_unit"] == "session" for record in records)
    assert {record["condition"] for record in records} == {
        "no_memory",
        "exponential_history",
        "learned_categorical_hmm",
        "oracle_ceiling",
    }
