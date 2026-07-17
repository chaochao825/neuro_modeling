from __future__ import annotations

import numpy as np
import pytest

from src.models.task_matched_actuators import (
    ActuatorFitError,
    TaskMatchedActuatorConfig,
    fit_task_matched_actuator,
    fit_task_matched_family,
)


def _simulate(
    *,
    a0: np.ndarray,
    b0: np.ndarray,
    inputs: np.ndarray,
    contexts: np.ndarray,
    initial: np.ndarray,
    delta_a: np.ndarray | None = None,
    delta_b: np.ndarray | None = None,
    gain: np.ndarray | None = None,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    batched = inputs.ndim == 3
    u = inputs if batched else inputs[None, ...]
    s = contexts if batched else contexts[None, ...]
    x0 = initial if batched else initial[None, ...]
    eps = np.zeros((*u.shape[:2], a0.shape[0])) if noise is None else noise
    if not batched and eps.ndim == 2:
        eps = eps[None, ...]
    da = np.zeros_like(a0) if delta_a is None else delta_a
    db = np.zeros_like(b0) if delta_b is None else delta_b
    g = np.zeros(a0.shape[0]) if gain is None else gain
    states = np.empty((u.shape[0], u.shape[1] + 1, a0.shape[0]))
    states[:, 0] = x0
    for time_index in range(u.shape[1]):
        base = states[:, time_index] @ a0.T + u[:, time_index] @ b0.T
        correction = s[:, time_index, None] * (
            states[:, time_index] @ da.T
            + u[:, time_index] @ db.T
            + g[None, :] * base
        )
        states[:, time_index + 1] = base + correction + eps[:, time_index]
    return states if batched else states[0]


def _base() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [
                [0.45, 0.05, 0.0],
                [0.0, 0.35, -0.04],
                [0.03, 0.0, 0.25],
            ]
        ),
        np.array(
            [
                [0.4, -0.1],
                [0.2, 0.3],
                [-0.15, 0.25],
            ]
        ),
    )


def test_pure_input_operator_is_recovered_by_routing_and_beats_low_rank() -> None:
    rng = np.random.default_rng(1)
    a0, b0 = _base()
    delta_b = np.array([[0.3, -0.1], [0.0, 0.2], [-0.15, 0.05]])
    inputs = rng.normal(size=(12, 20, 2))
    contexts = np.tile(np.array([-1.0, 1.0]), (12, 10))
    initial = rng.normal(scale=0.3, size=(12, 3))
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        delta_b=delta_b,
    )
    config = TaskMatchedActuatorConfig(rank_a=2, rank_b=2, ridge=0.0)

    routing = fit_task_matched_actuator(
        states, inputs, contexts, a0, b0, mode="routing", config=config
    )
    recurrent = fit_task_matched_actuator(
        states, inputs, contexts, a0, b0, mode="low_rank", config=config
    )

    np.testing.assert_allclose(routing.delta_b, delta_b, atol=1e-11)
    np.testing.assert_allclose(routing.delta_a, 0.0)
    assert routing.receipt.teacher_forced_error_rms < 1e-12
    assert (
        routing.receipt.teacher_forced_error_rms
        < recurrent.receipt.teacher_forced_error_rms
    )
    assert routing.receipt.raw_input_rank == 2
    assert routing.receipt.input_rank == 2


def test_pure_rank_one_recurrent_operator_is_recovered_exactly() -> None:
    rng = np.random.default_rng(2)
    a0, b0 = _base()
    left = np.array([0.2, -0.1, 0.15])
    right = np.array([0.5, 0.2, -0.3])
    delta_a = np.outer(left, right)
    inputs = rng.normal(size=(16, 14, 2))
    contexts = np.tile(np.array([-1.0, 1.0]), (16, 7))
    initial = rng.normal(scale=0.5, size=(16, 3))
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        delta_a=delta_a,
    )
    fitted = fit_task_matched_actuator(
        states,
        inputs,
        contexts,
        a0,
        b0,
        mode="low_rank",
        config=TaskMatchedActuatorConfig(rank_a=1, rank_b=1, ridge=0.0),
    )

    np.testing.assert_allclose(fitted.delta_a, delta_a, atol=1e-11)
    np.testing.assert_allclose(fitted.delta_b, 0.0)
    assert fitted.receipt.recurrent_rank == 1
    assert fitted.receipt.teacher_forced_error_rms < 1e-12


def test_recurrent_fit_records_raw_rank_before_registered_truncation() -> None:
    rng = np.random.default_rng(3)
    a0, b0 = _base()
    delta_a = np.diag([0.18, -0.12, 0.07])
    inputs = rng.normal(size=(20, 16, 2))
    contexts = np.tile(np.array([-1.0, 1.0]), (20, 8))
    initial = rng.normal(scale=0.4, size=(20, 3))
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        delta_a=delta_a,
    )
    fitted = fit_task_matched_actuator(
        states,
        inputs,
        contexts,
        a0,
        b0,
        mode="low_rank",
        config=TaskMatchedActuatorConfig(rank_a=1, rank_b=1, ridge=0.0),
    )

    assert fitted.receipt.raw_recurrent_rank == 3
    assert fitted.receipt.recurrent_rank == 1
    assert np.linalg.matrix_rank(fitted.delta_a) == 1
    assert fitted.receipt.teacher_forced_error_rms > 0.0


def test_population_gain_family_is_recovered_exactly() -> None:
    rng = np.random.default_rng(4)
    a0, b0 = _base()
    gain = np.array([0.15, -0.08, 0.11])
    inputs = rng.normal(size=(14, 18, 2))
    contexts = np.tile(np.array([-1.0, 1.0]), (14, 9))
    initial = rng.normal(scale=0.3, size=(14, 3))
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        gain=gain,
    )
    fitted = fit_task_matched_actuator(
        states,
        inputs,
        contexts,
        a0,
        b0,
        mode="gain",
        config=TaskMatchedActuatorConfig(rank_a=1, rank_b=1, ridge=0.0),
    )

    np.testing.assert_allclose(fitted.gain, gain, atol=1e-11)
    np.testing.assert_allclose(fitted.delta_a, 0.0)
    np.testing.assert_allclose(fitted.delta_b, 0.0)
    assert fitted.receipt.gain_rank == 3
    assert fitted.receipt.teacher_forced_error_rms < 1e-12


def test_all_active_modes_match_the_same_training_functional_l2_budget() -> None:
    rng = np.random.default_rng(5)
    a0, b0 = _base()
    delta_a = np.outer([0.12, -0.04, 0.08], [0.2, -0.1, 0.3])
    delta_b = np.array([[0.1, 0.0], [-0.03, 0.08], [0.0, -0.06]])
    inputs = rng.normal(size=(18, 12, 2))
    contexts = np.tile(np.array([-1.0, 1.0]), (18, 6))
    initial = rng.normal(scale=0.3, size=(18, 3))
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        delta_a=delta_a,
        delta_b=delta_b,
        gain=np.array([0.03, -0.02, 0.04]),
    )
    config = TaskMatchedActuatorConfig(rank_a=1, rank_b=2, ridge=1e-10)
    family = fit_task_matched_family(
        states,
        inputs,
        contexts,
        a0,
        b0,
        config=config,
        modes=("routing", "gain", "low_rank", "rgl"),
    )

    targets = {item.receipt.target_l2_rms for item in family.values()}
    training_ids = {item.receipt.training_fingerprint for item in family.values()}
    assert len(targets) == 1
    assert len(training_ids) == 1
    for item in family.values():
        assert item.receipt.budget_l2_relative_error <= 1e-10
        assert item.receipt.matched_current_l2_rms == pytest.approx(
            item.receipt.target_l2_rms, rel=1e-12
        )
        assert item.receipt.matched_current_l1_mean > 0.0


def test_frozen_is_intentional_zero_current_control() -> None:
    rng = np.random.default_rng(6)
    a0, b0 = _base()
    inputs = rng.normal(size=(10, 2))
    contexts = np.tile([-1.0, 1.0], 5)
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=np.array([0.2, -0.1, 0.3]),
        delta_b=np.array([[0.1, 0.0], [0.0, 0.1], [-0.05, 0.03]]),
    )
    frozen = fit_task_matched_actuator(
        states, inputs, contexts, a0, b0, mode="frozen"
    )

    np.testing.assert_array_equal(frozen.delta_a, 0.0)
    np.testing.assert_array_equal(frozen.delta_b, 0.0)
    np.testing.assert_array_equal(frozen.gain, 0.0)
    assert frozen.receipt.budget_scale == 0.0
    assert frozen.receipt.matched_current_l2_rms == 0.0
    assert frozen.receipt.teacher_forced_explained_fraction == 0.0


def test_rollout_uses_shared_noise_once_and_audits_the_shared_tape() -> None:
    rng = np.random.default_rng(7)
    a0, b0 = _base()
    delta_b = np.array([[0.12, -0.03], [0.02, 0.09], [-0.04, 0.07]])
    inputs = rng.normal(size=(20, 2))
    contexts = np.tile([-1.0, 1.0], 10)
    noise = rng.normal(scale=0.002, size=(20, 3))
    initial = np.array([0.15, -0.2, 0.05])
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        delta_b=delta_b,
        noise=noise,
    )
    config = TaskMatchedActuatorConfig(rank_a=1, rank_b=2, ridge=0.0)
    routing = fit_task_matched_actuator(
        states,
        inputs,
        contexts,
        a0,
        b0,
        mode="routing",
        process_noise=noise,
        config=config,
    )
    frozen = fit_task_matched_actuator(
        states,
        inputs,
        contexts,
        a0,
        b0,
        mode="frozen",
        process_noise=noise,
        config=config,
    )

    controlled_rollout = routing.rollout(
        initial, inputs, contexts, process_noise=noise
    )
    frozen_rollout = frozen.rollout(initial, inputs, contexts, process_noise=noise)

    assert controlled_rollout.tape_fingerprint == frozen_rollout.tape_fingerprint
    np.testing.assert_allclose(controlled_rollout.states, states, atol=1e-11)
    np.testing.assert_array_equal(controlled_rollout.process_noise, noise)
    np.testing.assert_allclose(
        controlled_rollout.total_correction_current[0],
        contexts[0] * (delta_b @ inputs[0]),
        atol=1e-11,
    )
    expected_frozen_first = a0 @ initial + b0 @ inputs[0] + noise[0]
    np.testing.assert_allclose(frozen_rollout.states[1], expected_frozen_first)
    np.testing.assert_allclose(controlled_rollout.event_proxy_by_step,
                               np.sum(np.abs(controlled_rollout.input_correction_current), axis=-1))


def test_fitted_arrays_and_rollout_arrays_are_immutable() -> None:
    rng = np.random.default_rng(8)
    a0, b0 = _base()
    inputs = rng.normal(size=(8, 2))
    contexts = np.tile([-1.0, 1.0], 4)
    initial = np.array([0.1, 0.2, -0.1])
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=initial,
        delta_b=np.array([[0.1, 0.0], [0.0, 0.1], [0.02, -0.03]]),
    )
    fitted = fit_task_matched_actuator(
        states,
        inputs,
        contexts,
        a0,
        b0,
        mode="routing",
        config=TaskMatchedActuatorConfig(rank_b=2),
    )
    rollout = fitted.rollout(initial, inputs, contexts)

    for array in (
        fitted.baseline_a,
        fitted.baseline_b,
        fitted.delta_a,
        fitted.delta_b,
        fitted.gain,
        rollout.states,
        rollout.total_correction_current,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99.0


def test_fitting_fails_closed_for_uncentered_degenerate_and_over_scale_data() -> None:
    rng = np.random.default_rng(9)
    a0, b0 = _base()
    inputs = rng.normal(size=(10, 2))
    centered = np.tile([-1.0, 1.0], 5)
    initial = np.array([0.2, -0.1, 0.05])
    delta_b = np.array([[0.1, 0.0], [0.0, 0.1], [-0.02, 0.03]])
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=centered,
        initial=initial,
        delta_b=delta_b,
    )

    with pytest.raises(ValueError, match="centered"):
        fit_task_matched_actuator(
            states, inputs, np.ones(10), a0, b0, mode="routing"
        )
    baseline_states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=centered,
        initial=initial,
    )
    with pytest.raises(ActuatorFitError, match="task residual"):
        fit_task_matched_actuator(
            baseline_states, inputs, centered, a0, b0, mode="routing"
        )
    with pytest.raises(ActuatorFitError, match="max_scale"):
        fit_task_matched_actuator(
            states,
            inputs,
            centered,
            a0,
            b0,
            mode="routing",
            config=TaskMatchedActuatorConfig(rank_b=2, max_scale=0.5),
        )
    zero_inputs = np.zeros_like(inputs)
    recurrent_states = _simulate(
        a0=a0,
        b0=b0,
        inputs=zero_inputs,
        contexts=centered,
        initial=initial,
        delta_a=np.outer([0.1, 0.05, -0.03], [0.2, -0.1, 0.15]),
    )
    with pytest.raises(ActuatorFitError, match="raw actuator direction"):
        fit_task_matched_actuator(
            recurrent_states,
            zero_inputs,
            centered,
            a0,
            b0,
            mode="routing",
        )


def test_shapes_modes_and_rank_limits_are_strictly_validated() -> None:
    a0, b0 = _base()
    inputs = np.ones((4, 2))
    contexts = np.array([-1.0, 1.0, -1.0, 1.0])
    states = _simulate(
        a0=a0,
        b0=b0,
        inputs=inputs,
        contexts=contexts,
        initial=np.zeros(3),
        delta_b=np.ones((3, 2)) * 0.01,
    )
    with pytest.raises(ValueError, match="one more"):
        fit_task_matched_actuator(
            states[:-1], inputs, contexts, a0, b0, mode="routing"
        )
    with pytest.raises(ValueError, match="rank_b"):
        fit_task_matched_actuator(
            states,
            inputs,
            contexts,
            a0,
            b0,
            mode="routing",
            config=TaskMatchedActuatorConfig(rank_b=3),
        )
    with pytest.raises(ValueError, match="one of"):
        fit_task_matched_actuator(
            states, inputs, contexts, a0, b0, mode="full"
        )
