from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import src.baselines.tuned_recurrent as tuned
from src.baselines.tuned_recurrent import (
    AllCandidatesFailedError,
    RecurrentCandidate,
    RecurrentSequenceData,
    block_safe_inner_split,
    build_candidate_grid,
    evaluate_masked_mse,
    predict_recurrent,
    refit_selected_recurrent_baseline,
    tune_recurrent_baseline,
)


def _development_data(seed: int = 0) -> RecurrentSequenceData:
    rng = np.random.default_rng(seed)
    n_blocks, trials_per_block, time = 8, 3, 6
    inputs = rng.normal(size=(n_blocks * trials_per_block, time, 2))
    targets = np.cumsum(inputs[..., :1], axis=1) / np.sqrt(time)
    loss_mask = np.zeros(inputs.shape[:2], dtype=bool)
    loss_mask[:, -2:] = True
    block_ids = np.repeat(np.arange(n_blocks), trials_per_block)
    return RecurrentSequenceData(inputs, targets, loss_mask, block_ids, "development")


def _small_candidates() -> tuple[RecurrentCandidate, ...]:
    return build_candidate_grid(
        cell_types=("rate_rnn", "gru"),
        hidden_sizes=(4,),
        learning_rates=(0.02,),
        max_epochs=4,
        batch_size=8,
        grad_clip=1.0,
        patience=2,
    )


def test_inner_split_is_deterministic_and_never_splits_blocks() -> None:
    development = _development_data()
    first_train, first_validation = block_safe_inner_split(
        development, validation_fraction=0.25, seed=11
    )
    second_train, second_validation = block_safe_inner_split(
        development, validation_fraction=0.25, seed=11
    )

    assert set(first_train.block_tokens).isdisjoint(first_validation.block_tokens)
    assert (
        first_train.trial_count + first_validation.trial_count
        == development.trial_count
    )
    assert first_validation.block_count == 2
    np.testing.assert_array_equal(first_train.inputs, second_train.inputs)
    np.testing.assert_array_equal(
        first_validation.block_ids, second_validation.block_ids
    )
    for block in np.unique(development.block_ids):
        in_train = block in first_train.block_ids
        in_validation = block in first_validation.block_ids
        assert in_train != in_validation


def test_rate_rnn_and_gru_tuning_is_deterministic_auditable_and_restores_best() -> None:
    training, validation = block_safe_inner_split(
        _development_data(), validation_fraction=0.25, seed=7
    )
    candidates = _small_candidates()
    first = tune_recurrent_baseline(training, validation, candidates, seed=19)
    second = tune_recurrent_baseline(training, validation, candidates, seed=19)

    assert first.selected_candidate_id == second.selected_candidate_id
    assert {audit.config.cell_type for audit in first.candidate_audits} == {
        "rate_rnn",
        "gru",
    }
    assert {audit.status for audit in first.candidate_audits} == {"complete"}
    assert sum(audit.selected for audit in first.candidate_audits) == 1
    for audit in first.candidate_audits:
        assert audit.parameter_count is not None and audit.parameter_count > 0
        assert audit.epochs_ran == len(audit.train_loss_history)
        assert audit.epochs_ran == len(audit.validation_loss_history)
        assert audit.best_epoch is not None
        assert audit.checkpoint_sha256 is not None

    selected = next(audit for audit in first.candidate_audits if audit.selected)
    restored_loss = evaluate_masked_mse(first.model, validation)
    assert restored_loss == pytest.approx(selected.best_validation_loss, rel=1e-6)
    first_prediction, first_states = predict_recurrent(first.model, validation.inputs)
    second_prediction, second_states = predict_recurrent(
        second.model, validation.inputs
    )
    np.testing.assert_array_equal(first_prediction, second_prediction)
    np.testing.assert_array_equal(first_states, second_states)

    metadata = first.audit_metadata()
    assert metadata["selection_data_scope"] == "inner_validation_blocks_only"
    assert metadata["test_data_used_for_selection"] is False
    assert metadata["candidate_count"] == 2
    assert metadata["candidate_failure_count"] == 0
    assert (
        metadata["train_block_fingerprint"] != metadata["validation_block_fingerprint"]
    )
    assert (
        first.model.checkpoint_metadata()["eligible_for_local_initialization"] is False
    )
    json.dumps(metadata)


def test_candidate_failure_is_retained_while_remaining_grid_runs(monkeypatch) -> None:
    training, validation = block_safe_inner_split(
        _development_data(), validation_fraction=0.25, seed=3
    )
    candidates = (
        RecurrentCandidate("rate_rnn", 3, 0.01, max_epochs=2, patience=1),
        RecurrentCandidate("rate_rnn", 4, 0.01, max_epochs=2, patience=1),
    )
    original = tuned._fit_candidate

    def fail_one(training, validation, config, **kwargs):
        if config.hidden_size == 3:
            raise RuntimeError("intentional candidate failure")
        return original(training, validation, config, **kwargs)

    monkeypatch.setattr(tuned, "_fit_candidate", fail_one)
    result = tune_recurrent_baseline(training, validation, candidates, seed=23)

    failed = next(
        audit for audit in result.candidate_audits if audit.status == "failed"
    )
    assert failed.error_type == "RuntimeError"
    assert failed.error == "intentional candidate failure"
    assert failed.train_loss_history == ()
    assert result.selection_metadata["candidate_failure_count"] == 1
    assert (
        next(audit for audit in result.candidate_audits if audit.selected).status
        == "complete"
    )


def test_all_candidate_failures_remain_available_on_exception(monkeypatch) -> None:
    training, validation = block_safe_inner_split(
        _development_data(), validation_fraction=0.25, seed=5
    )

    def always_fail(*args, **kwargs):
        raise FloatingPointError("intentional numerical failure")

    monkeypatch.setattr(tuned, "_fit_candidate", always_fail)
    with pytest.raises(AllCandidatesFailedError) as captured:
        tune_recurrent_baseline(
            training,
            validation,
            (RecurrentCandidate("gru", 4, 0.01, max_epochs=2, patience=1),),
            seed=29,
        )
    audits = captured.value.candidate_audits
    assert len(audits) == 1
    assert audits[0].status == "failed"
    assert audits[0].error_type == "FloatingPointError"
    json.dumps(captured.value.audit_metadata())


def test_tuning_rejects_block_overlap_and_has_no_test_argument() -> None:
    training, validation = block_safe_inner_split(
        _development_data(), validation_fraction=0.25, seed=13
    )
    overlapping_validation = RecurrentSequenceData(
        validation.inputs,
        validation.targets,
        validation.loss_mask,
        np.full(validation.trial_count, training.block_ids[0]),
        "invalid_validation",
    )
    with pytest.raises(ValueError, match="blocks overlap"):
        tune_recurrent_baseline(
            training,
            overlapping_validation,
            (RecurrentCandidate("rate_rnn", 4, 0.01, max_epochs=2, patience=1),),
            seed=31,
        )
    assert "test" not in inspect.signature(tune_recurrent_baseline).parameters


def test_candidate_grid_is_complete_and_rejects_duplicates() -> None:
    grid = build_candidate_grid(
        cell_types=("rate_rnn", "gru"),
        hidden_sizes=(4, 8),
        learning_rates=(0.01, 0.001),
        rate_leaks=(0.5, 1.0),
        max_epochs=2,
        patience=1,
    )
    assert len(grid) == 12
    assert sum(item.cell_type == "rate_rnn" for item in grid) == 8
    assert sum(item.cell_type == "gru" for item in grid) == 4

    training, validation = block_safe_inner_split(
        _development_data(), validation_fraction=0.25, seed=17
    )
    duplicate = RecurrentCandidate("gru", 4, 0.01, max_epochs=2, patience=1)
    with pytest.raises(ValueError, match="must be unique"):
        tune_recurrent_baseline(training, validation, (duplicate, duplicate), seed=37)


def test_hyperparameters_share_initialization_within_architecture_family() -> None:
    training, validation = block_safe_inner_split(
        _development_data(), validation_fraction=0.25, seed=59
    )
    grid = build_candidate_grid(
        cell_types=("rate_rnn", "gru"),
        hidden_sizes=(4,),
        learning_rates=(0.01, 0.003),
        rate_leaks=(0.5, 1.0),
        max_epochs=2,
        patience=1,
    )
    tuned_result = tune_recurrent_baseline(training, validation, grid, seed=61)
    seeds_by_family: dict[tuple[str, int], set[int]] = {}
    for audit in tuned_result.candidate_audits:
        family = (audit.config.cell_type, audit.config.hidden_size)
        seeds_by_family.setdefault(family, set()).add(audit.candidate_seed)
    assert all(len(seeds) == 1 for seeds in seeds_by_family.values())
    assert len({next(iter(seeds)) for seeds in seeds_by_family.values()}) == 2
    assert tuned_result.selection_metadata["candidate_initialization_policy"] == (
        "shared_within_cell_type_and_hidden_size"
    )


def test_selected_candidate_is_deterministically_refit_on_all_development_blocks() -> (
    None
):
    development = _development_data()
    training, validation = block_safe_inner_split(
        development, validation_fraction=0.25, seed=41
    )
    tuning = tune_recurrent_baseline(training, validation, _small_candidates(), seed=43)
    first = refit_selected_recurrent_baseline(development, tuning)
    second = refit_selected_recurrent_baseline(development, tuning)

    selected = next(item for item in tuning.candidate_audits if item.selected)
    assert first.audit.status == "complete"
    assert first.audit.initialization_rule.startswith("fresh_parameters")
    assert first.audit.epoch_rule == "exact_selected_inner_best_epoch"
    assert first.audit.data_scope == "exact_inner_train_validation_trial_union"
    assert first.audit.test_data_used_for_refit is False
    assert first.audit.planned_epochs == selected.best_epoch
    assert first.audit.epochs_ran == first.audit.planned_epochs
    assert len(first.audit.train_loss_history) == first.audit.planned_epochs
    assert first.audit.development_trial_count == development.trial_count
    assert first.audit.development_block_count == development.block_count
    assert first.audit.development_trial_count == (
        training.trial_count + validation.trial_count
    )
    assert first.audit.checkpoint_sha256 == second.audit.checkpoint_sha256
    assert first.audit.refit_seed == second.audit.refit_seed
    first_prediction, _ = predict_recurrent(first.model, development.inputs)
    second_prediction, _ = predict_recurrent(second.model, development.inputs)
    np.testing.assert_array_equal(first_prediction, second_prediction)
    json.dumps(first.audit_metadata())
    assert "test" not in inspect.signature(refit_selected_recurrent_baseline).parameters


def test_refit_rejects_any_dataset_other_than_exact_inner_union() -> None:
    development = _development_data()
    training, validation = block_safe_inner_split(
        development, validation_fraction=0.25, seed=47
    )
    tuning = tune_recurrent_baseline(
        training,
        validation,
        (RecurrentCandidate("rate_rnn", 4, 0.01, max_epochs=2, patience=1),),
        seed=53,
    )
    changed_targets = np.array(development.targets, copy=True)
    changed_targets[0, -1, 0] += 1.0
    changed = RecurrentSequenceData(
        development.inputs,
        changed_targets,
        development.loss_mask,
        development.block_ids,
        "changed_development",
    )
    with pytest.raises(ValueError, match="exactly equal"):
        refit_selected_recurrent_baseline(changed, tuning)
