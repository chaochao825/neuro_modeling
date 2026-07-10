import numpy as np
import pytest

from src.models.reduced_dynamics import (
    CommonDynamics,
    FullSwitchingDynamics,
    SharedBasisSwitchingDynamics,
    TransitionDataset,
    retained_switching_gain,
    LDSSequenceDataset,
    SwitchingLDS,
)


def _low_rank_trials(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_trials, n_time, n_features, latent_dim = 24, 18, 6, 2
    basis, _ = np.linalg.qr(rng.normal(size=(n_features, latent_dim)))
    matrices = np.array(
        [
            [[0.85, -0.15], [0.15, 0.8]],
            [[0.8, 0.18], [-0.18, 0.82]],
        ]
    )
    contexts = np.repeat([0, 1], n_trials // 2)
    activity = np.empty((n_trials, n_time, n_features))
    for trial in range(n_trials):
        latent = rng.normal(size=latent_dim)
        for time in range(n_time):
            activity[trial, time] = latent @ basis.T + 0.01 * rng.normal(size=n_features)
            latent = latent @ matrices[contexts[trial]] + 0.02 * rng.normal(size=latent_dim)
    return activity, contexts, np.arange(n_trials) // 2


def test_transitions_never_cross_trial_and_split_whole_groups() -> None:
    activity, contexts, groups = _low_rank_trials()
    dataset = TransitionDataset.from_trials(
        activity, conditions=contexts, groups=groups
    )
    assert dataset.n_samples == activity.shape[0] * (activity.shape[1] - 1)
    assert np.all(np.diff(dataset.time_indices.reshape(activity.shape[0], -1), axis=1) == 1)
    train, test = dataset.train_test_split(test_fraction=0.25, seed=4)
    assert set(train.groups).isdisjoint(set(test.groups))


def test_shared_model_recovers_predictive_low_rank_dynamics() -> None:
    activity, contexts, groups = _low_rank_trials()
    dataset = TransitionDataset.from_trials(activity, conditions=contexts, groups=groups)
    train, test = dataset.train_test_split(test_fraction=0.25, seed=8)
    shared = SharedBasisSwitchingDynamics(2, ridge=1e-3).fit(train)
    score = shared.score(test)
    assert score.r2 > 0.7
    assert np.isfinite(score.log_likelihood)
    assert shared.basis_.shape == (activity.shape[-1], 2)
    assert shared.parameter_count() < FullSwitchingDynamics(ridge=1e-3).fit(train).parameter_count()


def test_all_model_families_predict_and_count_parameters() -> None:
    activity, contexts, groups = _low_rank_trials(2)
    dataset = TransitionDataset.from_trials(activity, conditions=contexts, groups=groups)
    train, test = dataset.train_test_split(test_fraction=0.25, seed=1)
    models = [
        CommonDynamics().fit(train),
        SharedBasisSwitchingDynamics(2).fit(train),
        FullSwitchingDynamics().fit(train),
    ]
    for model in models:
        assert model.predict(test).shape == test.following.shape
        assert model.parameter_count() > 0
        assert set(model.transition_matrices()) == {0, 1}


def test_unseen_condition_is_rejected() -> None:
    activity, contexts, groups = _low_rank_trials()
    dataset = TransitionDataset.from_trials(activity, conditions=contexts, groups=groups)
    model = CommonDynamics().fit(dataset)
    unseen = TransitionDataset(
        dataset.current[:3],
        dataset.following[:3],
        np.array([9, 9, 9]),
        dataset.groups[:3],
    )
    with pytest.raises(ValueError, match="unseen"):
        model.predict(unseen)


def test_retained_gain_definition() -> None:
    assert retained_switching_gain(2.0, 1.25, 1.0) == pytest.approx(0.75)
    assert np.isnan(retained_switching_gain(1.0, 1.0, 1.0))


def test_dataset_copies_and_freezes_inputs_and_keeps_condition_types_distinct() -> None:
    current = [[0.0], [1.0], [2.0], [3.0]]
    following = [[1.0], [2.0], [3.0], [4.0]]
    dataset = TransitionDataset(current, following, np.array([1, "1", 1, "1"], dtype=object), [0, 0, 1, 1])
    assert isinstance(dataset.current, np.ndarray)
    assert not dataset.current.flags.writeable
    model = FullSwitchingDynamics().fit(dataset)
    assert set(model.transition_matrices()) == {1, "1"}
    with pytest.raises(TypeError, match="integers"):
        dataset.subset([0.5, 1.5])


def test_non_boolean_valid_mask_and_missing_condition_coverage_are_rejected() -> None:
    activity = np.zeros((2, 3, 1))
    with pytest.raises(ValueError, match="boolean"):
        TransitionDataset.from_trials(
            activity,
            conditions=np.array([0, 1]),
            groups=[0, 1],
            valid_mask=np.array([[1, 2, 1], [1, 1, 1]]),
        )
    dataset = TransitionDataset.from_trials(
        activity, conditions=np.array([0, 1]), groups=[0, 1]
    )
    with pytest.raises(ValueError, match="every condition"):
        dataset.train_test_split(seed=0)


def test_switching_lds_reports_marginal_kalman_likelihood() -> None:
    rng = np.random.default_rng(4)
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    transitions = [np.array([[0.85, -0.2], [0.2, 0.8]]), np.array([[0.8, 0.2], [-0.2, 0.85]])]
    observations = []
    conditions = []
    groups = []
    for trial in range(12):
        condition = trial % 2
        latent = rng.normal(size=2)
        sequence = []
        for _ in range(20):
            sequence.append(basis @ latent + 0.05 * rng.normal(size=5))
            latent = transitions[condition] @ latent + 0.05 * rng.normal(size=2)
        observations.append(np.asarray(sequence))
        conditions.append(np.repeat(condition, 20))
        groups.append(trial)
    train = LDSSequenceDataset(tuple(observations[:8]), tuple(conditions[:8]), np.asarray(groups[:8]))
    test = LDSSequenceDataset(tuple(observations[8:]), tuple(conditions[8:]), np.asarray(groups[8:]))
    common = SwitchingLDS("common", 2).fit(train)
    shared = SwitchingLDS("shared", 2).fit(train)
    full = SwitchingLDS("full", 2).fit(train)
    for model in (common, shared, full):
        score = model.score(test)
        assert np.isfinite(score.log_likelihood)
        assert score.n_sequences == 4
        _, posterior = model.filter_sequence(test.observations[0], test.conditions[0])
        assert posterior.shape == (20, 2)
    assert shared.parameter_count() < full.parameter_count()
    n, d, k = shared.n_features_, shared.latent_dim, len(shared.conditions_)
    basis = n * d - d * (d + 1) // 2
    initial = d + d * (d + 1) // 2
    assert common.parameter_count() == 2 * n + basis + n + d * d + d + initial
    assert shared.parameter_count() == (
        2 * n + basis + n + k * (d * d + d) + initial
    )
    assert full.parameter_count() == (
        2 * n
        + k * (basis + n + d * d + d)
        + (k - 1) * n
        + initial
    )
    hidden = shared.filter_hidden_context_sequence(test.observations[0])
    assert hidden.context_probability.shape == (20, 2)
    assert np.allclose(hidden.context_probability.sum(axis=1), 1.0)
    singleton = shared.filter_hidden_context_sequence(test.observations[0][:1])
    assert singleton.context_probability.shape == (1, 2)
    assert np.allclose(singleton.context_probability.sum(axis=1), 1.0)
    assert np.isfinite(shared.score_hidden_context(test).log_likelihood)


def test_model_hyperparameters_and_missing_labels_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="integer latent_dim"):
        SharedBasisSwitchingDynamics(2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        CommonDynamics(ridge=np.nan)
    with pytest.raises(ValueError, match="finite"):
        TransitionDataset(
            [[0.0], [1.0]],
            [[1.0], [2.0]],
            [0.0, np.nan],
            [0, 1],
        )


def test_likelihood_in_original_coordinates_includes_scale_jacobian() -> None:
    rng = np.random.default_rng(19)
    current = rng.normal(size=(80, 2)) * np.array([2.0, 5.0])
    following = 0.5 * current + rng.normal(scale=[0.2, 0.4], size=(80, 2))
    data = TransitionDataset(current, following, np.zeros(80, dtype=int), np.arange(80))
    model = CommonDynamics(variance_floor=1e-12).fit(data)
    score = model.score(data)
    standardized_residual = (
        (following - model.mean_) / model.scale_ - model._predict_standardized(data)
    )
    expected = -0.5 * np.sum(
        np.log(2.0 * np.pi * model.noise_variance_)[None, :]
        + standardized_residual**2 / model.noise_variance_[None, :]
    ) - data.n_samples * np.sum(np.log(model.scale_))
    assert score.log_likelihood == pytest.approx(expected)


def test_tuple_valued_rule_epoch_conditions_remain_scalar_labels() -> None:
    current = np.arange(12, dtype=float).reshape(6, 2)
    following = current + 1.0
    conditions = [
        ("forward", "maintain"),
        ("forward", "maintain"),
        ("forward", "sort"),
        ("forward", "sort"),
        ("forward", "maintain"),
        ("forward", "sort"),
    ]
    dataset = TransitionDataset(current, following, conditions, [0, 0, 1, 1, 2, 2])
    model = FullSwitchingDynamics().fit(dataset)
    assert set(model.transition_matrices()) == {
        ("forward", "maintain"),
        ("forward", "sort"),
    }


def test_dynamics_and_correlated_controls_are_fit_jointly() -> None:
    rng = np.random.default_rng(23)
    current = rng.normal(size=(120, 2))
    controls = current[:, [0]] + 0.2 * rng.normal(size=(120, 1))
    dynamic = np.array([[0.7, -0.2], [0.1, 0.5]])
    control_coef = np.array([[1.3, -0.8]])
    following = current @ dynamic + controls @ control_coef + np.array([0.2, -0.1])
    data = TransitionDataset(
        current,
        following,
        np.zeros(120, dtype=int),
        np.arange(120),
        controls=controls,
    )
    model = CommonDynamics(ridge=0.0, variance_floor=1e-12).fit(data)
    np.testing.assert_allclose(model.predict(data), following, atol=1e-10)


def test_full_dimensional_shared_joint_fit_matches_full_family() -> None:
    activity, contexts, groups = _low_rank_trials(11)
    dataset = TransitionDataset.from_trials(activity, conditions=contexts, groups=groups)
    full = FullSwitchingDynamics(ridge=1e-6).fit(dataset)
    shared = SharedBasisSwitchingDynamics(activity.shape[-1], ridge=1e-6).fit(dataset)
    np.testing.assert_allclose(shared.predict(dataset), full.predict(dataset), atol=2e-6)
