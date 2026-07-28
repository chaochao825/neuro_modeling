from __future__ import annotations

import numpy as np
import pytest

from src.models.factorized_uncertainty_filter import (
    AdaptationRates,
    JumpFilterParameters,
    run_factorized_filter,
)
from src.models.fast_slow_uncertainty_audit import (
    run_fast_slow_exchange_filter,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, JumpFilterParameters, AdaptationRates]:
    observations = np.asarray([0.1, 0.2, 2.5, 2.2, -0.1, 0.0, 0.3, 0.2])
    sequence_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    initial = JumpFilterParameters(0.02, 0.03, 0.08, 4.0)
    adaptation = AdaptationRates(0.02, 0.1, 0.2)
    return observations, sequence_ids, initial, adaptation


def test_exchange_filter_matches_locked_exp39_path_without_overrides() -> None:
    observations, sequence_ids, initial, adaptation = _inputs()
    expected = run_factorized_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
    )
    actual = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
    )

    for name in (
        "predictive_nll",
        "predictive_mean",
        "filtered_mean",
        "filtered_variance",
        "hazard",
        "process_variance",
        "observation_variance",
        "jump_probability",
    ):
        assert np.allclose(getattr(actual, name), getattr(expected, name), atol=1e-12)
    assert np.array_equal(actual.release_probability, actual.jump_probability)
    assert np.all((actual.write_gain > 0.0) & (actual.write_gain < 1.0))


def test_oracle_release_is_post_observation_and_prefix_causal() -> None:
    observations, sequence_ids, initial, adaptation = _inputs()
    learned = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
    )
    release = learned.jump_probability.copy()
    release[2] = 1.0
    forced = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
        release_override=release,
    )

    assert forced.predictive_nll[2] == pytest.approx(learned.predictive_nll[2])
    assert forced.filtered_mean[2] != pytest.approx(learned.filtered_mean[2])
    assert np.allclose(forced.predictive_nll[:3], learned.predictive_nll[:3])

    future_changed = release.copy()
    future_changed[6:] = 1.0 - future_changed[6:]
    second = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
        release_override=future_changed,
    )
    for name in forced.__dict__:
        assert np.allclose(getattr(forced, name)[:6], getattr(second, name)[:6])


def test_qr_override_is_aligned_positive_and_prefix_causal() -> None:
    observations, sequence_ids, initial, adaptation = _inputs()
    q_values = np.full(len(observations), 0.04)
    r_values = np.full(len(observations), 0.12)
    first = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
        process_override=q_values,
        observation_override=r_values,
    )
    q_future = q_values.copy()
    r_future = r_values.copy()
    q_future[5:] = 0.2
    r_future[5:] = 0.4
    second = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
        process_override=q_future,
        observation_override=r_future,
    )
    for name in first.__dict__:
        assert np.allclose(getattr(first, name)[:5], getattr(second, name)[:5])

    with pytest.raises(ValueError, match="supplied together"):
        run_fast_slow_exchange_filter(
            observations,
            sequence_ids=sequence_ids,
            initial=initial,
            adaptation=adaptation,
            process_override=q_values,
        )
    with pytest.raises(ValueError, match="positive"):
        run_fast_slow_exchange_filter(
            observations,
            sequence_ids=sequence_ids,
            initial=initial,
            adaptation=adaptation,
            process_override=np.zeros(len(observations)),
            observation_override=r_values,
        )


def test_sequence_boundary_resets_remove_cross_sequence_oracle_effects() -> None:
    observations, sequence_ids, initial, adaptation = _inputs()
    release_a = np.zeros(len(observations))
    release_b = release_a.copy()
    release_b[:4] = 1.0
    first = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
        release_override=release_a,
    )
    second = run_fast_slow_exchange_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=initial,
        adaptation=adaptation,
        release_override=release_b,
    )
    for name in first.__dict__:
        assert np.allclose(getattr(first, name)[4:], getattr(second, name)[4:])
