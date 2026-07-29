from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.models.piray_daw_qr_controller import (
    VarianceBounds,
    average_traces,
    run_autocovariance_qr,
    run_factorized_local_em,
    run_fixed_gain,
    run_hierarchical_particle,
    run_kalman_schedule,
    run_total_uncertainty,
)


def _stream(seed: int = 7, trials: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = 60.0 + np.cumsum(rng.normal(0.0, 2.0, size=trials))
    return (state + rng.normal(0.0, 4.0, size=trials))[:, None]


def test_fixed_gain_matches_delta_rule_and_resets_blocks() -> None:
    observations = np.array([[62.0, 55.0], [64.0, 57.0], [63.0, 60.0]])
    trace = run_fixed_gain(observations, gain=0.25, initial_mean=60.0)
    assert trace.predictive_mean[0].tolist() == [60.0, 60.0]
    assert trace.posterior_mean[0].tolist() == [60.5, 58.75]
    assert np.all(trace.gain == 0.25)
    assert trace.process_variance is None
    assert not trace.gain.flags.writeable


def test_oracle_schedule_has_opposite_q_and_r_gain_effects() -> None:
    observations = np.full((20, 4), 60.0)
    q = np.broadcast_to(np.array([4.0, 49.0, 4.0, 49.0]), observations.shape)
    r = np.broadcast_to(np.array([16.0, 16.0, 64.0, 64.0]), observations.shape)
    trace = run_kalman_schedule(
        observations,
        process_variance=q,
        observation_variance=r,
        bounds=VarianceBounds(0.25, 256.0),
    )
    mean_gain = trace.gain.mean(axis=0)
    assert mean_gain[1] > mean_gain[0]
    assert mean_gain[0] > mean_gain[2]
    assert mean_gain[3] > mean_gain[2]


@pytest.mark.parametrize(
    "runner,kwargs",
    [
        (
            run_factorized_local_em,
            {
                "initial_process_variance": 8.0,
                "initial_observation_variance": 8.0,
                "process_rate": 0.1,
                "observation_rate": 0.1,
            },
        ),
        (
            run_total_uncertainty,
            {
                "initial_total_variance": 16.0,
                "q_fraction": 0.3,
                "adaptation_rate": 0.1,
            },
        ),
        (
            run_autocovariance_qr,
            {
                "initial_process_variance": 8.0,
                "initial_observation_variance": 8.0,
                "statistic_decay": 0.95,
                "prior_mass": 8.0,
            },
        ),
    ],
)
def test_local_controllers_are_prefix_causal(runner, kwargs) -> None:
    observations = _stream(trials=80)
    full = runner(observations, **kwargs)
    prefix = runner(observations[:35], **kwargs)
    for name in (
        "predictive_mean",
        "posterior_mean",
        "gain",
        "process_variance",
        "observation_variance",
    ):
        np.testing.assert_allclose(getattr(full, name)[:35], getattr(prefix, name))


def test_total_uncertainty_keeps_one_fixed_allocation() -> None:
    trace = run_total_uncertainty(
        _stream(),
        initial_total_variance=16.0,
        q_fraction=0.3,
        adaptation_rate=0.5,
    )
    fraction = trace.process_variance / (
        trace.process_variance + trace.observation_variance
    )
    np.testing.assert_allclose(fraction, 0.3)


def test_autocovariance_recovers_long_run_qr_moments() -> None:
    rng = np.random.default_rng(19)
    q_true = 4.0
    r_true = 16.0
    state = 60.0 + np.cumsum(rng.normal(0.0, np.sqrt(q_true), size=5000))
    observations = (state + rng.normal(0.0, np.sqrt(r_true), size=5000))[:, None]
    trace = run_autocovariance_qr(
        observations,
        initial_process_variance=8.0,
        initial_observation_variance=8.0,
        statistic_decay=1.0,
        prior_mass=2.0,
    )
    assert trace.process_variance[-1, 0] == pytest.approx(q_true, rel=0.35)
    assert trace.observation_variance[-1, 0] == pytest.approx(r_true, rel=0.2)


def test_particle_comparator_is_seeded_and_prefix_causal() -> None:
    observations = _stream(trials=50)
    kwargs = {
        "change_probability_q": 0.1,
        "change_probability_r": 0.1,
        "log_step_scale": 0.2,
        "particle_count": 128,
        "seed": 31,
    }
    first = run_hierarchical_particle(observations, **kwargs)
    second = run_hierarchical_particle(observations, **kwargs)
    prefix = run_hierarchical_particle(observations[:25], **kwargs)
    np.testing.assert_array_equal(first.gain, second.gain)
    np.testing.assert_allclose(first.gain[:25], prefix.gain)
    assert np.all((first.gain >= 0.0) & (first.gain <= 1.0))


def test_particle_seeds_are_averaged_before_inference() -> None:
    observations = _stream(trials=30)
    traces = [
        run_hierarchical_particle(
            observations,
            change_probability_q=0.1,
            change_probability_r=0.1,
            log_step_scale=0.2,
            particle_count=64,
            seed=seed,
        )
        for seed in (1, 2)
    ]
    averaged = average_traces(traces)
    np.testing.assert_allclose(averaged.gain, (traces[0].gain + traces[1].gain) / 2)


def test_deployable_signatures_have_no_true_qr_or_future_inputs() -> None:
    for function in (
        run_factorized_local_em,
        run_total_uncertainty,
        run_autocovariance_qr,
        run_hierarchical_particle,
    ):
        names = set(inspect.signature(function).parameters)
        assert "true_process_variance" not in names
        assert "true_observation_variance" not in names
        assert "future_observations" not in names


def test_invalid_variances_and_rates_fail_closed() -> None:
    with pytest.raises(ValueError):
        VarianceBounds(1.0, 1.0)
    with pytest.raises(ValueError):
        run_factorized_local_em(
            _stream(),
            initial_process_variance=1.0,
            initial_observation_variance=1.0,
            process_rate=1.1,
            observation_rate=0.1,
        )
