"""Leakage, likelihood, and hierarchy tests for conditional count dynamics."""

from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from src.models.hierarchical_count_dynamics import (
    BeliefFitReceipt,
    HierarchicalCountDynamics,
    NeuralCountSession,
    TrialBlockSplit,
    poisson_log_likelihood,
)


_REGIONS = ("visual", "motor", "frontal")


def _synthetic_sessions(
    seed: int = 4,
) -> tuple[tuple[NeuralCountSession, ...], dict[str, TrialBlockSplit]]:
    rng = np.random.default_rng(seed)
    n_sessions, n_trials, n_time = 3, 72, 11
    units_per_region = 3
    regions = tuple(region for region in _REGIONS for _ in range(units_per_region))
    transitions = (
        np.array([[0.76, 0.34], [-0.24, 0.72]]),
        np.array([[0.72, -0.36], [0.28, 0.74]]),
    )
    region_loading = np.array([[0.62, 0.08], [0.05, 0.60], [-0.42, 0.38]])
    sessions: list[NeuralCountSession] = []
    splits: dict[str, TrialBlockSplit] = {}
    for session_index in range(n_sessions):
        unit_loading = np.repeat(region_loading, units_per_region, axis=0)
        unit_loading += rng.normal(0.0, 0.025, size=unit_loading.shape)
        unit_bias = rng.normal(3.45 + 0.08 * session_index, 0.06, len(regions))
        nuisance_loading = rng.normal(0.0, 0.04, len(regions))
        beliefs = np.empty((n_trials, 2), dtype=float)
        controls = rng.normal(0.0, 1.0, size=(n_trials, 1))
        counts = np.empty((n_trials, n_time, len(regions)), dtype=np.int64)
        for trial in range(n_trials):
            state = (trial // 3) % 2
            beliefs[trial] = (0.94, 0.06) if state == 0 else (0.06, 0.94)
            latent = rng.normal(0.0, 0.55, size=2)
            for time in range(n_time):
                log_rate = (
                    unit_bias
                    + unit_loading @ latent
                    + nuisance_loading * controls[trial, 0]
                )
                counts[trial, time] = rng.poisson(np.exp(log_rate))
                latent = transitions[state] @ latent + rng.normal(0.0, 0.025, 2)
        trial_ids = np.asarray(
            [f"session-{session_index}:trial-{index:03d}" for index in range(n_trials)],
            dtype=object,
        )
        n_train = 51
        receipt = BeliefFitReceipt.bind(
            beliefs,
            method="synthetic_past_only_fixture",
            fit_trial_ids=trial_ids[:n_train],
            checkpoint_payload={"fixture_seed": seed, "session": session_index},
        )
        session = NeuralCountSession(
            session_id=f"session-{session_index}",
            animal_id=f"animal-{session_index // 2}",
            counts=counts,
            unit_regions=regions,
            beliefs=beliefs,
            trial_ids=trial_ids,
            belief_receipt=receipt,
            controls=controls,
        )
        sessions.append(session)
        splits[session.session_id] = TrialBlockSplit.chronological_holdout(
            session,
            block_ids=np.arange(n_trials) // 3,
            n_train=n_train,
        )
    return tuple(sessions), splits


def _fit_family(
    family: str,
    sessions: tuple[NeuralCountSession, ...],
    splits: dict[str, TrialBlockSplit],
) -> HierarchicalCountDynamics:
    return HierarchicalCountDynamics(
        family,
        common_regions=_REGIONS,
        latent_dim=2,
        ridge=2e-2,
        seed=17,
    ).fit(sessions, splits)


def _disjoint_region_sessions(
    seed: int = 4,
) -> tuple[tuple[NeuralCountSession, ...], dict[str, TrialBlockSplit]]:
    sessions, splits = _synthetic_sessions(seed)
    disjoint = []
    units_per_region = 3
    for index, session in enumerate(sessions):
        start = index * units_per_region
        stop = start + units_per_region
        disjoint.append(
            replace(
                session,
                counts=session.counts[:, :, start:stop],
                unit_regions=(_REGIONS[index],) * units_per_region,
            )
        )
    return tuple(disjoint), splits


def test_shared_soft_belief_dynamics_beats_common_and_is_compact() -> None:
    sessions, splits = _synthetic_sessions()
    common = _fit_family("common", sessions, splits)
    shared = _fit_family("shared", sessions, splits)
    full = _fit_family("full", sessions, splits)
    common_score = common.score(sessions, splits)
    shared_score = shared.score(sessions, splits)
    full_score = full.score(sessions, splits)

    assert shared_score.nll_per_count < common_score.nll_per_count
    assert shared_score.closure_mse < common_score.closure_mse
    # The true operators are shared.  The shared model should preserve the
    # flexible model's held-out fit within a small conditional-NLL tolerance.
    assert shared_score.nll_per_count <= full_score.nll_per_count + 0.03
    assert shared.parameter_count() < full.parameter_count()
    assert common.parameter_count() < shared.parameter_count()
    assert shared_score.parameter_count == shared.parameter_count()
    assert shared_score.likelihood_kind == "one_step_conditional_poisson"
    assert not shared_score.full_latent_lds
    assert len(shared_score.per_session) == len(sessions)
    assert all(np.isfinite(item.closure_mse) for item in shared_score.per_session)


def test_parameter_count_includes_preprocessing_observations_and_controls() -> None:
    sessions, splits = _synthetic_sessions(6)
    shared = _fit_family("shared", sessions, splits)
    n_regions, latent_dim = 3, 2
    preprocessing = 3 * n_regions + latent_dim * n_regions
    observations = sum(
        session.n_units * (latent_dim + session.control_dim + 1) for session in sessions
    )
    dynamics = 2 * latent_dim * (latent_dim + 1)
    assert shared.parameter_count() == preprocessing + observations + dynamics


def test_disjoint_region_sessions_use_train_only_imputation_without_nan() -> None:
    sessions, splits = _disjoint_region_sessions(16)
    baseline = _fit_family("shared", sessions, splits)
    assert baseline.region_imputation_strategy == "pooled_training_fold_region_mean"
    assert baseline.session_region_presence_ == {
        "session-0": (True, False, False),
        "session-1": (False, True, False),
        "session-2": (False, False, True),
    }
    assert baseline.region_train_observation_counts_ is not None
    np.testing.assert_array_equal(
        baseline.region_train_observation_counts_, np.full(3, 2 * 51 * 10)
    )
    assert (
        baseline.scaler_mean_ is not None and np.isfinite(baseline.scaler_mean_).all()
    )
    assert (
        baseline.scaler_scale_ is not None and np.isfinite(baseline.scaler_scale_).all()
    )
    assert (
        baseline.pca_components_ is not None
        and np.isfinite(baseline.pca_components_).all()
    )
    for prediction in baseline.predict(sessions, splits).values():
        assert np.isfinite(prediction.latent_prediction).all()
        assert np.isfinite(prediction.rates).all()

    changed_counts = sessions[0].counts.copy()
    changed_counts[51:] += 10_000
    changed_sessions = (replace(sessions[0], counts=changed_counts), *sessions[1:])
    refit = _fit_family("shared", changed_sessions, splits)
    np.testing.assert_array_equal(refit.scaler_mean_, baseline.scaler_mean_)
    np.testing.assert_array_equal(refit.scaler_scale_, baseline.scaler_scale_)
    np.testing.assert_array_equal(refit.pca_components_, baseline.pca_components_)
    assert refit.fit_fingerprint_ == baseline.fit_fingerprint_

    n_regions, latent_dim = 3, 2
    preprocessing = 3 * n_regions + latent_dim * n_regions
    observations = sum(
        session.n_units * (latent_dim + session.control_dim + 1) for session in sessions
    )
    dynamics = 2 * latent_dim * (latent_dim + 1)
    assert baseline.parameter_count() == preprocessing + observations + dynamics

    repeated = _fit_family("shared", sessions, splits)
    assert repeated.fit_fingerprint_ == baseline.fit_fingerprint_


def test_train_only_fit_is_invariant_to_heldout_targets_inputs_and_controls() -> None:
    sessions, splits = _synthetic_sessions(9)
    baseline = _fit_family("shared", sessions, splits)
    baseline_predictions = baseline.predict(sessions, splits)
    first = sessions[0]
    test_positions = np.arange(51, first.n_trials)

    target_counts = first.counts.copy()
    # Final bins are targets but never current inputs for any one-step pair.
    target_counts[test_positions, -1] += 17
    target_changed = replace(first, counts=target_counts)
    target_sessions = (target_changed, *sessions[1:])
    target_refit = _fit_family("shared", target_sessions, splits)
    assert target_refit.fit_fingerprint_ == baseline.fit_fingerprint_
    np.testing.assert_allclose(
        target_refit.predict(target_sessions, splits)[first.session_id].rates,
        baseline_predictions[first.session_id].rates,
    )
    assert (
        target_refit.score(target_sessions, splits).nll_per_count
        != baseline.score(sessions, splits).nll_per_count
    )

    current_counts = first.counts.copy()
    current_counts[test_positions, 0] += 25
    current_changed = replace(first, counts=current_counts)
    current_sessions = (current_changed, *sessions[1:])
    current_refit = _fit_family("shared", current_sessions, splits)
    assert current_refit.fit_fingerprint_ == baseline.fit_fingerprint_
    assert not np.allclose(
        current_refit.predict(current_sessions, splits)[first.session_id].rates,
        baseline_predictions[first.session_id].rates,
    )

    changed_controls = first.controls.copy()
    changed_controls[test_positions, 0] += 4.0
    control_changed = replace(first, controls=changed_controls)
    control_sessions = (control_changed, *sessions[1:])
    control_refit = _fit_family("shared", control_sessions, splits)
    assert control_refit.fit_fingerprint_ == baseline.fit_fingerprint_
    assert not np.allclose(
        control_refit.predict(control_sessions, splits)[first.session_id].rates,
        baseline_predictions[first.session_id].rates,
    )


def test_exact_poisson_likelihood_includes_factorial_term() -> None:
    counts = np.array([0, 1, 2, 4], dtype=int)
    rates = np.array([0.5, 1.5, 2.5, 3.5])
    expected = sum(
        count * math.log(rate) - rate - math.lgamma(count + 1)
        for count, rate in zip(counts, rates, strict=True)
    )
    assert poisson_log_likelihood(counts, rates) == pytest.approx(expected)
    with pytest.raises(ValueError, match="positive"):
        poisson_log_likelihood(counts, np.array([0.5, 0.0, 2.5, 3.5]))


def test_data_are_immutable_and_api_has_no_privileged_state_or_target_fields() -> None:
    sessions, _ = _synthetic_sessions(3)
    session = sessions[0]
    assert not session.counts.flags.writeable
    assert not session.beliefs.flags.writeable
    assert not session.trial_ids.flags.writeable
    assert session.controls is not None and not session.controls.flags.writeable
    with pytest.raises(ValueError):
        session.counts[0, 0, 0] = 9
    fields = set(inspect.signature(NeuralCountSession).parameters)
    assert not ({"target", "targets", "context", "contexts"} & fields)
    fit_fields = set(inspect.signature(HierarchicalCountDynamics.fit).parameters)
    assert not ({"target", "targets", "context", "contexts"} & fit_fields)
    assert session.beliefs.shape[1] == 2
    assert np.all((session.beliefs > 0.0) & (session.beliefs < 1.0))
    assert session.belief_receipt.input_columns == ("stimulus_side_lag1",)
    assert not session.belief_receipt.accessed_true_context


def test_invalid_splits_regions_beliefs_and_unseen_sessions_fail_closed() -> None:
    sessions, splits = _synthetic_sessions(10)
    first = sessions[0]
    with pytest.raises(ValueError, match="overlap"):
        TrialBlockSplit(
            (first.trial_ids[0],),
            (first.trial_ids[0],),
            tuple(first.trial_ids[:2]),
            (0, 1),
        )
    with pytest.raises(ValueError, match="chronological"):
        TrialBlockSplit(
            tuple(first.trial_ids[1::-1]),
            tuple(first.trial_ids[2:]),
            tuple(first.trial_ids),
            tuple(np.arange(first.n_trials) // 3),
        )

    hard_beliefs = first.beliefs.copy()
    hard_beliefs[0] = (1.0, 0.0)
    with pytest.raises(ValueError, match="strictly soft"):
        replace(first, beliefs=hard_beliefs)

    with pytest.raises(ValueError, match="true context"):
        BeliefFitReceipt(
            method="oracle",
            fit_trial_ids=splits[first.session_id].train_trial_ids,
            observation_fit_trial_ids=splits[first.session_id].train_trial_ids,
            input_columns=("stimulus_side_lag1",),
            uses_current_trial_stimulus=False,
            uses_future_trials=False,
            accessed_true_context=True,
            checkpoint_sha256="a" * 64,
            belief_sha256="b" * 64,
        )

    tampered_beliefs = first.beliefs.copy()
    tampered_beliefs[-1] = tampered_beliefs[-1, ::-1]
    with pytest.raises(ValueError, match="does not bind"):
        replace(first, beliefs=tampered_beliefs)

    gap_train = tuple(first.trial_ids[:48])
    gap_test = tuple(first.trial_ids[51:])
    gap_ids = gap_train + gap_test
    gap_blocks = tuple(np.concatenate((np.arange(48) // 3, np.arange(51, 72) // 3)))
    gap_split = TrialBlockSplit(gap_train, gap_test, gap_ids, gap_blocks)
    bad_splits = dict(splits)
    bad_splits[first.session_id] = gap_split
    with pytest.raises(ValueError, match="exactly bind"):
        _fit_family("shared", sessions, bad_splits)

    with pytest.raises(ValueError, match="whole block"):
        TrialBlockSplit.chronological_holdout(
            first,
            block_ids=np.arange(first.n_trials) // 3,
            n_train=52,
        )

    sparse_regions = tuple(
        "visual" if region == "frontal" else region for region in first.unit_regions
    )
    sparse = replace(first, unit_regions=sparse_regions)
    sparse_fit = _fit_family("shared", (sparse, *sessions[1:]), splits)
    assert sparse_fit.session_region_presence_[first.session_id] == (True, True, False)
    assert np.isfinite(sparse_fit.score((sparse, *sessions[1:]), splits).nll_per_count)

    no_anchor = replace(first, unit_regions=("off-basis",) * first.n_units)
    with pytest.raises(ValueError, match="no configured anchor region"):
        _fit_family("shared", (no_anchor, *sessions[1:]), splits)

    fitted = _fit_family("shared", sessions, splits)
    unknown = replace(first, session_id="new-session")
    unknown_split = {"new-session": splits[first.session_id]}
    with pytest.raises(ValueError, match="unseen session"):
        fitted.score((unknown,), unknown_split)


def test_fits_are_deterministic() -> None:
    sessions, splits = _synthetic_sessions(12)
    first = _fit_family("shared", sessions, splits)
    second = _fit_family("shared", sessions, splits)
    assert first.fit_fingerprint_ == second.fit_fingerprint_
    first_predictions = first.predict(sessions, splits)
    second_predictions = second.predict(sessions, splits)
    for session_id in first_predictions:
        np.testing.assert_array_equal(
            first_predictions[session_id].rates, second_predictions[session_id].rates
        )
