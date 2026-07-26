from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.models.autocovariance_uncertainty_filter import (
    AutocovarianceFilterTrace,
    AutocovarianceUpdateConfig,
    derive_qr_from_increment_covariance,
    qr_from_total_variance,
    run_autocovariance_filter,
    run_total_variance_filter,
)
from src.models.factorized_uncertainty_filter import (
    JumpFilterParameters,
    ParameterBounds,
)
from src.tasks.matched_uncertainty import (
    MATCHED_QR_REGIMES,
    MatchedUncertaintyConfig,
    generate_matched_uncertainty_tape,
)


def _short_tape():
    return generate_matched_uncertainty_tape(
        seed=411,
        split="filter-unit",
        config=MatchedUncertaintyConfig(
            block_length=24, blocks_per_sequence=4, n_sequences=2
        ),
    )


def _run(observations: np.ndarray, sequence_ids: np.ndarray):
    return run_autocovariance_filter(
        observations,
        sequence_ids=sequence_ids,
        initial=JumpFilterParameters(1e-4, 0.02, 0.02, 4.0),
        update=AutocovarianceUpdateConfig(
            statistic_decay=0.995,
            prior_mass=4.0,
        ),
    )


def test_qr_derivation_uses_lag_one_information_and_applies_bounds() -> None:
    q_value, r_value = derive_qr_from_increment_covariance(0.06, -0.01)
    assert q_value == pytest.approx(0.04)
    assert r_value == pytest.approx(0.01)

    bounds = ParameterBounds(
        process_variance=(0.005, 0.05),
        observation_variance=(0.002, 0.04),
    )
    clipped_q, clipped_r = derive_qr_from_increment_covariance(0.0, 1.0, bounds=bounds)
    assert clipped_q == bounds.process_variance[1]
    assert clipped_r == bounds.observation_variance[0]


def test_filter_api_cannot_receive_true_regime_or_block_metadata() -> None:
    parameters = set(inspect.signature(run_autocovariance_filter).parameters)
    assert parameters == {"observations", "sequence_ids", "initial", "update", "bounds"}
    assert not parameters & {
        "block_ids",
        "regimes",
        "true_process_variance",
        "true_observation_variance",
    }


def test_filter_is_deterministic_finite_bounded_and_records_local_statistics() -> None:
    tape = _short_tape()
    first = _run(tape.observations, tape.sequence_ids)
    replay = _run(tape.observations, tape.sequence_ids)

    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, field), getattr(replay, field))
    assert np.all(np.isfinite(first.predictive_nll))
    assert np.all((first.process_variance >= 1e-4) & (first.process_variance <= 0.5))
    assert np.all(
        (first.observation_variance >= 1e-4) & (first.observation_variance <= 1.0)
    )
    np.testing.assert_allclose(first.gamma0, first.s0 / first.n0)
    np.testing.assert_allclose(first.gamma1, first.s1 / first.n1)
    assert np.all(first.n0 > 0.0)
    assert np.all(first.n1 > 0.0)

    sequence_starts = np.flatnonzero(
        np.r_[True, tape.sequence_ids[1:] != tape.sequence_ids[:-1]]
    )
    assert not np.any(first.has_previous_increment[sequence_starts])


def test_trace_defensively_copies_every_caller_array() -> None:
    original = _run(*_short_tape().method_inputs())
    caller_arrays = [
        np.array(getattr(original, field), copy=True)
        for field in original.__dataclass_fields__
    ]
    copied = AutocovarianceFilterTrace(*caller_arrays)
    caller_arrays[0][0] += 100.0
    caller_arrays[-1][0] = not caller_arrays[-1][0]
    assert copied.predictive_nll[0] != caller_arrays[0][0]
    assert copied.has_previous_increment[0] != caller_arrays[-1][0]
    assert not copied.predictive_nll.flags.writeable


def test_future_suffix_cannot_change_any_prefix_state_or_prediction() -> None:
    tape = _short_tape()
    prefix = len(tape.observations) // 2
    original = _run(tape.observations, tape.sequence_ids)
    changed_observations = tape.observations.copy()
    changed_observations[prefix:] += np.linspace(
        10.0, 100.0, len(changed_observations) - prefix
    )
    changed = _run(changed_observations, tape.sequence_ids)

    for field in original.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(original, field)[:prefix], getattr(changed, field)[:prefix]
        )


@pytest.mark.parametrize("regime", MATCHED_QR_REGIMES)
def test_long_stationary_sequence_recovers_q_and_r(regime) -> None:
    tape = generate_matched_uncertainty_tape(
        seed=410,
        split=f"recovery-{regime.name}",
        config=MatchedUncertaintyConfig(
            block_length=16_000,
            blocks_per_sequence=1,
            n_sequences=1,
        ),
        regimes=(regime,),
    )
    trace = run_autocovariance_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        initial=JumpFilterParameters(1e-4, 0.02, 0.02, 4.0),
        update=AutocovarianceUpdateConfig(
            statistic_decay=1.0,
            prior_mass=1.0,
        ),
    )
    assert trace.process_variance[-1] == pytest.approx(
        regime.process_variance, abs=0.003
    )
    assert trace.observation_variance[-1] == pytest.approx(
        regime.observation_variance, abs=0.003
    )


def test_update_configuration_validation() -> None:
    with pytest.raises(ValueError, match="statistic_decay"):
        AutocovarianceUpdateConfig(statistic_decay=0.0)
    with pytest.raises(ValueError, match="prior_mass"):
        AutocovarianceUpdateConfig(prior_mass=0.0)
    with pytest.raises(ValueError, match="within configured bounds"):
        run_autocovariance_filter(
            np.asarray([0.0]),
            sequence_ids=np.asarray([0]),
            initial=JumpFilterParameters(0.3, 0.02, 0.02, 4.0),
        )


def test_total_variance_is_the_single_tied_qr_coordinate_and_is_causal() -> None:
    q_value, r_value = qr_from_total_variance(0.06, 2.0 / 3.0)
    assert q_value == pytest.approx(0.04)
    assert r_value == pytest.approx(0.01)
    tape = _short_tape()
    prefix = 80
    kwargs = {
        "sequence_ids": tape.sequence_ids,
        "initial": JumpFilterParameters(1e-4, 0.02, 0.02, 4.0),
        "q_fraction": 0.5,
        "update": AutocovarianceUpdateConfig(statistic_decay=0.99, prior_mass=2.0),
    }
    original = run_total_variance_filter(tape.observations, **kwargs)
    changed_observations = tape.observations.copy()
    changed_observations[prefix:] += 50.0
    changed = run_total_variance_filter(changed_observations, **kwargs)
    np.testing.assert_array_equal(
        original.predictive_nll[:prefix], changed.predictive_nll[:prefix]
    )
    np.testing.assert_allclose(
        original.process_variance
        / (original.process_variance + 2.0 * original.observation_variance),
        0.5,
        atol=1e-12,
    )
    assert not {
        "block_ids",
        "regimes",
        "true_process_variance",
        "true_observation_variance",
    } & set(inspect.signature(run_total_variance_filter).parameters)
