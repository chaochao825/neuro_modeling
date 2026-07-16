from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from src.models.belief_controlled_plds import (
    DEFAULT_LATENT_DIMENSIONS,
    BeliefControlledCountSession,
    BeliefControlledPLDS,
    CausalBeliefReceipt,
    TrialFold,
    UnalignedSessionLatentCoordinatesError,
    poisson_log_likelihood,
    select_latent_dimension,
)


def _sessions(
    seed: int = 11,
) -> tuple[
    tuple[BeliefControlledCountSession, ...],
    dict[str, TrialFold],
]:
    rng = np.random.default_rng(seed)
    n_trials, n_time, input_dim, latent_dim = 18, 7, 2, 2
    labels = ("task-a", "task-b")
    operators = (
        np.array([[0.72, 0.22], [-0.16, 0.74]]),
        np.array([[0.70, -0.24], [0.20, 0.72]]),
    )
    input_operators = (
        np.array([[0.32, -0.08], [0.06, 0.16]]),
        np.array([[-0.28, 0.12], [0.04, 0.20]]),
    )
    sessions = []
    folds: dict[str, TrialFold] = {}
    for session_index, n_units in enumerate((7, 8)):
        session_id = f"session-{session_index}"
        trial_ids = np.asarray(
            [f"{session_id}:trial-{index:02d}" for index in range(n_trials)],
            dtype=object,
        )
        task_index = np.arange(n_trials) % 2
        task_ids = np.asarray([labels[index] for index in task_index], dtype=object)
        beliefs = np.asarray(
            [(0.93, 0.07) if index == 0 else (0.07, 0.93) for index in task_index]
        )
        inputs = rng.normal(0.0, 0.7, size=(n_trials, n_time, input_dim))
        loading = rng.normal(0.0, 0.24, size=(latent_dim, n_units))
        bias = rng.normal(1.55, 0.08, size=n_units)
        counts = np.empty((n_trials, n_time, n_units), dtype=np.int64)
        for trial in range(n_trials):
            state = rng.normal(0.0, 0.35, size=latent_dim)
            task = task_index[trial]
            for time in range(n_time):
                log_rate = bias + state @ loading
                counts[trial, time] = rng.poisson(np.exp(log_rate))
                state = (
                    state @ operators[task]
                    + inputs[trial, time] @ input_operators[task]
                    + rng.normal(0.0, 0.025, latent_dim)
                )
        receipt = CausalBeliefReceipt(
            evaluated_trial_keys=tuple(
                (session_id, str(trial_id)) for trial_id in trial_ids
            ),
            source_columns=("cue_lag1", "choice_lag1"),
        )
        session = BeliefControlledCountSession(
            session_id=session_id,
            animal_id=f"animal-{session_index}",
            counts=counts,
            inputs=inputs,
            beliefs=beliefs,
            belief_labels=labels,
            trial_ids=trial_ids,
            belief_receipt=receipt,
            task_ids=task_ids,
        )
        sessions.append(session)
        folds[session_id] = TrialFold(
            train_trial_ids=tuple(trial_ids[:12]),
            test_trial_ids=tuple(trial_ids[12:]),
        )
    return tuple(sessions), folds


def test_all_five_families_score_exact_poisson_and_count_parameters() -> None:
    all_sessions, all_folds = _sessions()
    sessions = all_sessions[:1]
    folds = {sessions[0].session_id: all_folds[sessions[0].session_id]}
    models = {
        family: BeliefControlledPLDS(
            family, 2, gate_rank=1, max_irls=15
        ).fit(sessions, folds)
        for family in (
            "common",
            "input-gated",
            "state-gated",
            "fully-gated",
            "separate-task",
        )
    }
    for family, model in models.items():
        prediction = model.predict(sessions, folds)
        score = model.score(sessions, folds)
        assert set(prediction) == {"session-0"}
        assert np.isfinite(score.log_likelihood)
        assert score.n_observations > 0
        assert score.parameter_count == model.parameter_count()
        assert score.likelihood_kind.startswith("one_step_conditional_poisson")
        assert not score.full_marginal_plds
        assert not score.heldout_truth_used
        assert all(
            item.rates.shape == item.observed.shape
            for item in prediction.values()
        )
        if family in {"state-gated", "fully-gated"}:
            operators = model.effective_state_operators()
            delta = operators["task-b"] - operators["task-a"]
            assert np.linalg.matrix_rank(delta, tol=1e-8) <= 1
    assert models["common"].parameter_count() < models[
        "input-gated"
    ].parameter_count()
    assert models["common"].parameter_count() < models[
        "state-gated"
    ].parameter_count()
    assert models["fully-gated"].parameter_count() < models[
        "separate-task"
    ].parameter_count()


def test_separate_task_truth_is_fit_only_and_not_required_for_scoring() -> None:
    all_sessions, all_folds = _sessions(13)
    sessions = all_sessions[:1]
    folds = {sessions[0].session_id: all_folds[sessions[0].session_id]}
    model = BeliefControlledPLDS("separate_task", 2, max_irls=12).fit(
        sessions, folds
    )
    baseline = model.score(sessions, folds)
    truth_free = tuple(replace(session, task_ids=None) for session in sessions)
    without_truth = model.score(truth_free, folds)
    assert without_truth == baseline
    predict_parameters = set(inspect.signature(model.predict).parameters)
    score_parameters = set(inspect.signature(model.score).parameters)
    assert not (
        {"task", "tasks", "task_ids", "context", "contexts"}
        & (predict_parameters | score_parameters)
    )


def test_all_preprocessing_and_observations_are_fit_on_train_only() -> None:
    all_sessions, all_folds = _sessions(21)
    sessions = all_sessions[:1]
    folds = {sessions[0].session_id: all_folds[sessions[0].session_id]}
    baseline = BeliefControlledPLDS("fully-gated", 2, max_irls=12).fit(
        sessions, folds
    )
    baseline_prediction = baseline.predict(sessions, folds)["session-0"].rates.copy()
    changed_counts = sessions[0].counts.copy()
    changed_counts[12:, -1] += 25
    changed_first = replace(sessions[0], counts=changed_counts)
    changed_sessions = (changed_first,)
    refit = BeliefControlledPLDS("fully-gated", 2, max_irls=12).fit(
        changed_sessions, folds
    )
    assert refit.fit_fingerprint_ == baseline.fit_fingerprint_
    np.testing.assert_array_equal(
        refit.predict(changed_sessions, folds)["session-0"].rates,
        baseline_prediction,
    )
    assert (
        refit.score(changed_sessions, folds).log_likelihood
        != baseline.score(sessions, folds).log_likelihood
    )

    changed_inputs = sessions[0].inputs.copy()
    changed_inputs[12:] += 3.0
    changed_beliefs = sessions[0].beliefs.copy()
    changed_beliefs[12:, :, 0] = 0.35
    changed_beliefs[12:, :, 1] = 0.65
    heldout_covariates = replace(
        sessions[0], inputs=changed_inputs, beliefs=changed_beliefs
    )
    covariate_sessions = (heldout_covariates,)
    covariate_refit = BeliefControlledPLDS(
        "fully-gated", 2, max_irls=12
    ).fit(covariate_sessions, folds)
    assert covariate_refit.fit_fingerprint_ == baseline.fit_fingerprint_
    assert not np.allclose(
        covariate_refit.predict(covariate_sessions, folds)["session-0"].rates,
        baseline_prediction,
    )


def test_belief_receipt_and_soft_probability_contract_fail_closed() -> None:
    sessions, _ = _sessions(4)
    session = sessions[0]
    hard = session.beliefs.copy()
    hard[0, 0] = (1.0, 0.0)
    with pytest.raises(ValueError, match="strictly soft"):
        replace(session, beliefs=hard)
    leaked = CausalBeliefReceipt(
        evaluated_trial_keys=tuple(
            (session.session_id, str(trial_id)) for trial_id in session.trial_ids
        ),
        source_columns=("true_context",),
        accessed_test_truth=True,
    )
    with pytest.raises(ValueError, match="leakage"):
        replace(session, belief_receipt=leaked)
    dishonest = CausalBeliefReceipt(
        evaluated_trial_keys=tuple(
            (session.session_id, str(trial_id)) for trial_id in session.trial_ids
        ),
        source_columns=("true_context",),
    )
    with pytest.raises(ValueError, match="privileged"):
        replace(session, belief_receipt=dishonest)
    misaligned = CausalBeliefReceipt(
        evaluated_trial_keys=tuple(
            (session.session_id, str(trial_id))
            for trial_id in session.trial_ids[::-1]
        ),
        source_columns=("cue_lag1",),
    )
    with pytest.raises(ValueError, match="trial order"):
        replace(session, belief_receipt=misaligned)


def test_nested_dimension_selection_retains_failed_dimensions_and_outer_scope() -> None:
    all_sessions, all_outer = _sessions(31)
    sessions = all_sessions[:1]
    outer = {sessions[0].session_id: all_outer[sessions[0].session_id]}
    inner = {}
    for session in sessions:
        allowed = outer[session.session_id].train_trial_ids
        inner[session.session_id] = TrialFold(
            train_trial_ids=tuple(allowed[:8]),
            test_trial_ids=tuple(allowed[8:]),
        )
    selection = select_latent_dimension(
        "common",
        sessions,
        [inner],
        candidate_dimensions=(2, 8),
        allowed_trial_ids={
            session_id: fold.train_trial_ids for session_id, fold in outer.items()
        },
        max_irls=8,
    )
    assert selection.candidate_dimensions == (2, 8)
    assert selection.selected_dimension == 2
    assert selection.candidates[0].eligible
    assert not selection.candidates[1].eligible
    assert selection.candidates[1].fold_errors[0] is not None
    assert DEFAULT_LATENT_DIMENSIONS == (2, 4, 8, 16)

    leaking = dict(inner)
    first = sessions[0]
    leaking[first.session_id] = TrialFold(
        inner[first.session_id].train_trial_ids,
        (first.trial_ids[-1],),
    )
    with pytest.raises(ValueError, match="outside outer training"):
        select_latent_dimension(
            "common",
            sessions,
            [leaking],
            candidate_dimensions=(2,),
            allowed_trial_ids={
                session_id: fold.train_trial_ids
                for session_id, fold in outer.items()
            },
            max_irls=5,
        )


def test_multi_session_independent_pca_coordinates_fail_closed() -> None:
    sessions, folds = _sessions(44)
    with pytest.raises(
        UnalignedSessionLatentCoordinatesError,
        match="no train-only shared basis or identifiable session alignment",
    ):
        BeliefControlledPLDS("common", 2, max_irls=5).fit(sessions, folds)


def test_poisson_likelihood_includes_factorial_term() -> None:
    counts = np.asarray([0, 1, 3, 5], dtype=int)
    rates = np.asarray([0.5, 1.5, 2.5, 4.0])
    expected = sum(
        count * math.log(rate) - rate - math.lgamma(count + 1)
        for count, rate in zip(counts, rates, strict=True)
    )
    assert poisson_log_likelihood(counts, rates) == pytest.approx(expected)
    with pytest.raises(ValueError, match="positive"):
        poisson_log_likelihood(counts, np.asarray([0.5, 0.0, 2.5, 4.0]))
