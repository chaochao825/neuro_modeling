import json
import numpy as np
import pandas as pd
import pytest

from experiments.common import load_json_config
from experiments.exp40_ibl_state_utility import run_seed
from src.analysis.ibl_state_utility import (
    FACTORIZED_STATE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    MEAN_STATE_FEATURE_NAMES,
    belief_mean_features,
    causal_history_features,
    evaluate_choice_readout,
    factorized_clamp_features,
    factorized_state_features,
)
from src.data.ibl_behavior import contiguous_block_split, load_ibl_behavior_table
from src.models.ibl_run_length_observer import SemiMarkovBlockObserver


def _session(*, seed: int = 19):
    rng = np.random.default_rng(seed)
    levels = np.concatenate([np.array([0.5]), np.resize(np.array([0.2, 0.8]), 11)])
    probability_left = np.repeat(levels, 30)
    left = rng.random(probability_left.size) < probability_left
    contrast = np.resize(np.array([0.0, 0.0625, 0.125, 0.25]), probability_left.size)
    signed = np.where(left, contrast, -contrast)
    choice_probability = 1.0 / (
        1.0 + np.exp(-np.clip(5.0 * signed + 1.2 * (probability_left - 0.5), -20, 20))
    )
    choice_left = rng.random(probability_left.size) < choice_probability
    table = pd.DataFrame(
        {
            "contrastLeft": np.where(left, contrast, np.nan),
            "contrastRight": np.where(~left, contrast, np.nan),
            "choice": np.where(choice_left, 1, -1),
            "feedbackType": np.where(rng.random(probability_left.size) < 0.8, 1, -1),
            "probabilityLeft": probability_left,
        }
    )
    return load_ibl_behavior_table(table, eid="synthetic", animal_id="mouse")


def _factorized(session):
    split = contiguous_block_split(session.block_ids)
    prediction = (
        SemiMarkovBlockObserver()
        .fit(session.observations, split.train_indices)
        .predict(session.observations)
    )
    return split, prediction, factorized_state_features(session, prediction)


def test_feature_contract_is_causal_and_three_state_expansion_is_fixed() -> None:
    session = _session()
    split, prediction, factorized = _factorized(session)
    history = causal_history_features(session)
    mean = belief_mean_features(session, prediction.beliefs)

    assert history.shape == (session.trial_ids.size, len(HISTORY_FEATURE_NAMES))
    assert mean.shape == (session.trial_ids.size, len(MEAN_STATE_FEATURE_NAMES))
    assert factorized.shape == (
        session.trial_ids.size,
        len(FACTORIZED_STATE_FEATURE_NAMES),
    )
    np.testing.assert_allclose(mean[:, :-1], history)
    np.testing.assert_allclose(factorized[:, : mean.shape[1]], mean)
    assert not factorized.flags.writeable

    trial = split.test_indices[5]
    altered_choice = session.choice_left.copy()
    altered_choice[trial] = 1 - altered_choice[trial]
    object.__setattr__(session, "choice_left", altered_choice)
    altered_history = causal_history_features(session)
    np.testing.assert_allclose(history[: trial + 1], altered_history[: trial + 1])


def test_choice_readout_selects_on_dev_and_never_fits_test_rows() -> None:
    session = _session()
    split, _, features = _factorized(session)
    fit = np.concatenate([split.train_indices, split.dev_indices])
    interventions = {
        "clamp_release": factorized_clamp_features(
            features, fit_indices=fit, clamp="release"
        ),
        "clamp_concentration": factorized_clamp_features(
            features, fit_indices=fit, clamp="concentration"
        ),
    }
    evaluation = evaluate_choice_readout(
        session,
        split,
        features,
        condition="factorized_state",
        feature_names=FACTORIZED_STATE_FEATURE_NAMES,
        seed=4,
        interventions=interventions,
    )

    assert set(evaluation.fit_trial_ids).isdisjoint(evaluation.test_trial_ids)
    np.testing.assert_array_equal(evaluation.fit_trial_ids, session.trial_ids[fit])
    assert evaluation.dev_selection_trial_count >= 8
    assert evaluation.dev_selection_scope == "low_contrast"
    assert evaluation.metrics["low_contrast"].n_trials >= 8
    assert set(evaluation.intervention_metrics) == {
        "clamp_release",
        "clamp_concentration",
    }

    changed = features.copy()
    changed[split.test_indices] += 1000.0
    changed_evaluation = evaluate_choice_readout(
        session,
        split,
        changed,
        condition="factorized_state",
        feature_names=FACTORIZED_STATE_FEATURE_NAMES,
        seed=4,
    )
    assert changed_evaluation.selected_c == evaluation.selected_c
    assert changed_evaluation.dev_selection_nll == pytest.approx(
        evaluation.dev_selection_nll
    )

    all_dev = evaluate_choice_readout(
        session,
        split,
        features,
        condition="factorized_state",
        feature_names=FACTORIZED_STATE_FEATURE_NAMES,
        dev_selection_scope="all",
        seed=4,
    )
    assert all_dev.dev_selection_scope == "all"
    assert all_dev.dev_selection_trial_count > evaluation.dev_selection_trial_count


def test_clamps_use_fit_state_and_recompute_directional_interaction() -> None:
    session = _session()
    split, _, features = _factorized(session)
    fit = np.concatenate([split.train_indices, split.dev_indices])
    release = FACTORIZED_STATE_FEATURE_NAMES.index("release_probability")
    prior = FACTORIZED_STATE_FEATURE_NAMES.index("prior_log_odds")
    interaction = FACTORIZED_STATE_FEATURE_NAMES.index("prior_x_release")
    clamped = factorized_clamp_features(features, fit_indices=fit, clamp="release")
    np.testing.assert_allclose(clamped[:, release], np.mean(features[fit, release]))
    np.testing.assert_allclose(
        clamped[:, interaction], clamped[:, prior] * clamped[:, release]
    )
    with pytest.raises(ValueError, match="clamp"):
        factorized_clamp_features(features, fit_indices=fit, clamp="belief")


def test_choice_readout_rejects_feature_contract_and_sparse_dev_endpoint() -> None:
    session = _session()
    split = contiguous_block_split(session.block_ids)
    history = causal_history_features(session)
    with pytest.raises(ValueError, match="feature_names"):
        evaluate_choice_readout(
            session,
            split,
            history,
            condition="history",
            feature_names=("one",),
        )
    with pytest.raises(ValueError, match="c_grid"):
        evaluate_choice_readout(
            session,
            split,
            history,
            condition="history",
            feature_names=HISTORY_FEATURE_NAMES,
            c_grid=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="low_contrast_threshold"):
        evaluate_choice_readout(
            session,
            split,
            history,
            condition="history",
            feature_names=HISTORY_FEATURE_NAMES,
            low_contrast_threshold=-1.0,
        )
    with pytest.raises(ValueError, match="dev_selection_scope"):
        evaluate_choice_readout(
            session,
            split,
            history,
            condition="history",
            feature_names=HISTORY_FEATURE_NAMES,
            dev_selection_scope="test",
        )


def test_exp40_preserves_complete_paired_panel_and_raw_predictions(tmp_path) -> None:
    session = _session()
    config = load_json_config("configs/smoke/exp40_ibl_state_utility.json")
    path = run_seed(config, 0, str(tmp_path), sessions=[session])
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 7
    assert all(record["status"] == "complete" for record in records)
    assert all(record["statistics_unit"] == "animal" for record in records)
    assert all(record["test_used_for_selection"] is False for record in records)
    assert {record["condition"] for record in records} == {
        "history_only",
        "learned_hmm_mean",
        "semimarkov_mean",
        "semimarkov_release",
        "semimarkov_concentration",
        "factorized_state",
        "oracle_context_mean",
    }
    factorized = next(
        record for record in records if record["condition"] == "factorized_state"
    )
    assert factorized["gate_accessed_probabilityLeft"] is False
    assert factorized["gate_uses_current_trial_stimulus"] is False
    assert factorized["state_dimension"] == 3
    assert "clamp_release_low_contrast_nll_harm" in factorized
    assert (path / factorized["raw_prediction_artifact"]).is_file()
    oracle = next(
        record for record in records if record["condition"] == "oracle_context_mean"
    )
    assert oracle["oracle_is_evaluation_only"] is True
    assert oracle["gate_accessed_probabilityLeft"] is True
