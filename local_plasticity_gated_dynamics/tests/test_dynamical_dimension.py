from __future__ import annotations

import numpy as np
import pytest

from src.analysis.dynamical_dimension import (
    empirical_hankel_summary,
    fit_hankel_noise_floor,
    fit_hankel_preprocessor,
    jacobian_outlier_summary,
)


def test_paired_bulk_outlier_is_distinct_from_instability() -> None:
    bulk = np.diag([-1.0, -1.0, -1.0, -1.0])
    target = np.diag([-0.2, -1.0, -1.0, -1.0])

    summary = jacobian_outlier_summary(target, bulk, edge_quantile=1.0, tolerance=1e-12)

    assert summary.bulk_right_edge == pytest.approx(-1.0)
    assert summary.outlier_count == 1
    assert summary.bulk_tail_count == 0
    assert summary.excess_outlier_count == 1
    assert np.max(np.real(summary.target_eigenvalues)) < 0.0


def _linear_trials(initial_states: np.ndarray, *, steps: int = 8) -> np.ndarray:
    transition = np.diag([0.8, 0.45])
    observation = np.array([[1.0, 0.2], [0.3, 1.0], [1.2, -0.7]])
    trials = []
    for initial in initial_states:
        state = initial.copy()
        values = []
        for _ in range(steps):
            values.append(observation @ state)
            state = transition @ state
        trials.append(values)
    return np.asarray(trials, dtype=float)


def test_trial_safe_hankel_recovers_a_known_two_state_system() -> None:
    train_initial = np.array([[1.0, 0.5], [-1.0, -0.5], [0.4, -0.8], [-0.4, 0.8]])
    test_initial = np.array([[0.7, 1.1], [-0.7, -1.1], [1.3, -0.2], [-1.3, 0.2]])
    train = _linear_trials(train_initial)
    test = _linear_trials(test_initial)
    preprocessor = fit_hankel_preprocessor(train, normalize=True)

    summary = empirical_hankel_summary(
        test,
        past_lags=2,
        future_lags=2,
        preprocessor=preprocessor,
        rtol=1e-9,
        window_chunk_size=2,
    )

    assert summary.raw_numerical_rank == 2
    assert summary.noise_adjusted_dimension is None
    assert summary.threshold_source == "fixed_numeric_raw_only"
    assert "not_system_order" in summary.dimension_interpretation
    assert summary.n_trials == 4
    assert summary.n_windows == 4 * (8 - 2 - 2 + 1)
    assert summary.n_features == 3
    assert summary.preprocessing == "train_fitted_center_scale"
    assert summary.moment_kind == "train_centered_cross_moment"
    assert summary.preprocessor_train_observations == 32

    floor = fit_hankel_noise_floor(
        train,
        past_lags=2,
        future_lags=2,
        preprocessor=preprocessor,
        n_permutations=100,
        quantile=0.95,
        seed=2,
    )
    adjusted = empirical_hankel_summary(
        test,
        past_lags=2,
        future_lags=2,
        preprocessor=preprocessor,
        noise_floor=floor,
        rtol=1e-9,
    )
    assert adjusted.noise_adjusted_dimension == 2


def test_hankel_preprocessing_is_frozen_from_training_trials() -> None:
    train = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )
    preprocessor = fit_hankel_preprocessor(train, normalize=True)
    mean_before = preprocessor.mean_.copy()

    transformed = preprocessor.transform(np.full((3, 2), 1_000.0))

    np.testing.assert_array_equal(preprocessor.mean_, mean_before)
    assert np.all(transformed > 0.0)
    assert not preprocessor.mean_.flags.writeable
    assert not preprocessor.scale_.flags.writeable


def test_training_permutation_floor_rejects_full_raw_rank_from_noise() -> None:
    train_rng = np.random.default_rng(120)
    test_rng = np.random.default_rng(121)
    train = train_rng.normal(size=(40, 30, 3))
    test = test_rng.normal(size=(40, 30, 3))
    preprocessor = fit_hankel_preprocessor(train, normalize=True)
    floor = fit_hankel_noise_floor(
        train,
        past_lags=1,
        future_lags=1,
        preprocessor=preprocessor,
        n_permutations=100,
        quantile=0.99,
        seed=122,
    )
    summary = empirical_hankel_summary(
        test,
        past_lags=1,
        future_lags=1,
        preprocessor=preprocessor,
        noise_floor=floor,
    )

    assert summary.raw_numerical_rank == 3
    assert summary.noise_adjusted_dimension == 0
    np.testing.assert_allclose(
        summary.dimension_thresholds,
        np.maximum(summary.raw_numeric_threshold, floor.singular_value_thresholds),
    )
    assert summary.threshold_source == "train_fitted_within_trial_permutation"
    assert floor.method == "within_trial_future_window_permutation"
    assert floor.n_train_windows == 40 * 29
    assert not floor.null_max_singular_values.flags.writeable
    assert not floor.null_singular_values.flags.writeable


def test_hankel_windows_never_join_separate_trials() -> None:
    too_short = [np.ones((2, 1)), -np.ones((2, 1))]

    with pytest.raises(ValueError, match="no trial is long enough"):
        empirical_hankel_summary(too_short, past_lags=2, future_lags=1)


def test_hankel_accepts_single_list_trials_and_variable_trial_lengths() -> None:
    single = [[1.0], [2.0], [3.0]]
    one_trial = empirical_hankel_summary(single, past_lags=1, future_lags=1)
    assert one_trial.n_trials == 1
    assert one_trial.n_windows == 2

    variable = [np.arange(4.0)[:, None], np.arange(5.0)[:, None]]
    two_trials = empirical_hankel_summary(variable, past_lags=1, future_lags=1)
    assert two_trials.n_trials == 2
    assert two_trials.n_windows == 3 + 4


def test_dynamical_dimension_input_validation() -> None:
    with pytest.raises(ValueError, match="square"):
        jacobian_outlier_summary(np.ones((2, 3)), np.eye(2))
    with pytest.raises(ValueError, match="edge_quantile"):
        jacobian_outlier_summary(np.eye(2), np.eye(2), edge_quantile=0.0)
    with pytest.raises(ValueError, match="feature count"):
        empirical_hankel_summary(
            np.ones((2, 4, 3)),
            past_lags=1,
            future_lags=1,
            preprocessor=fit_hankel_preprocessor(np.ones((2, 4, 2))),
        )
