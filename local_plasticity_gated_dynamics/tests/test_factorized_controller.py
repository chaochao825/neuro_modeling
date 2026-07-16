from __future__ import annotations

import numpy as np
import pytest

from src.models.factorized_controller import (
    ActuatorMode,
    FactorizedController,
    FactorizedControllerConfig,
)


def _controller() -> FactorizedController:
    return FactorizedController(
        base_recurrent=np.diag([0.2, 0.3, 0.4]),
        input_weights=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, -0.5],
            ]
        ),
        routing_axes=np.eye(2),
        gain_axes=np.array(
            [
                [0.5, 0.0],
                [0.0, 0.5],
                [0.25, -0.25],
            ]
        ),
        low_rank_left=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        low_rank_right=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        bias=np.zeros(3),
        config=FactorizedControllerConfig(
            activation="identity",
            gain_max=3.0,
            routing_max=3.0,
        ),
    )


def test_factorized_actuators_share_two_dimensional_control() -> None:
    controller = _controller()
    state = np.array([1.0, 2.0, 3.0])
    input_value = np.array([4.0, 5.0])
    control = np.array([0.2, -0.1])

    frozen = controller.step(
        state,
        input_value,
        mode="frozen",
        control=control,
    )
    routing = controller.step(
        state,
        input_value,
        mode="routing",
        control=control,
    )
    gain = controller.step(state, input_value, mode="gain", control=control)
    low_rank = controller.step(
        state,
        input_value,
        mode="low_rank",
        control=control,
    )
    combined = controller.step(state, input_value, mode="rgl", control=control)

    np.testing.assert_allclose(frozen.routing_scale, 1.0)
    np.testing.assert_allclose(frozen.gain, 1.0)
    np.testing.assert_allclose(
        routing.routing_scale,
        [1.2, 0.9],
    )
    np.testing.assert_allclose(routing.effective_recurrent, controller.base_recurrent)
    np.testing.assert_allclose(gain.routing_scale, 1.0)
    np.testing.assert_allclose(gain.gain, [1.1, 0.95, 1.075])
    assert np.linalg.matrix_rank(low_rank.recurrent_update) == 2
    assert (
        np.linalg.matrix_rank(low_rank.effective_recurrent - controller.base_recurrent)
        == 2
    )
    np.testing.assert_allclose(combined.routing_scale, routing.routing_scale)
    np.testing.assert_allclose(combined.gain, gain.gain)
    np.testing.assert_allclose(
        combined.recurrent_update,
        low_rank.recurrent_update,
    )
    assert not np.allclose(combined.next_state, frozen.next_state)
    assert controller.control_dim == 2


def test_rollout_returns_complete_audit_and_frozen_base_is_immutable() -> None:
    controller = _controller()
    original = controller.base_recurrent.copy()
    inputs = np.array(
        [
            [1.0, -0.5],
            [0.25, 0.75],
            [-0.5, 0.5],
        ]
    )
    controls = np.array(
        [
            [0.2, -0.1],
            [0.1, 0.3],
            [-0.2, 0.2],
        ]
    )
    frozen = controller.rollout(
        np.zeros(3),
        inputs,
        mode=ActuatorMode.FROZEN,
    )
    controlled = controller.rollout(
        np.zeros(3),
        inputs,
        mode=ActuatorMode.RGL,
        controls=controls,
    )
    audit = controller.audit_rollout(
        controlled,
        frozen_states=frozen.states,
    )

    assert controlled.states.shape == (4, 3)
    assert controlled.rates.shape == (3, 3)
    assert controlled.effective_recurrent_history.shape == (3, 3, 3)
    assert controlled.gain_history.shape == (3, 3)
    assert audit.mode == "rgl"
    assert audit.control_dim == 2
    assert audit.routing_control_dim == 2
    assert audit.gain_control_dim == 2
    assert audit.low_rank_control_dim == 2
    assert audit.base_recurrent_rank == 3
    assert audit.low_rank_update_rank <= 2
    assert audit.functional_displacement is not None
    assert audit.functional_displacement > 0.0
    assert audit.mean_gain > 0.0
    assert audit.max_gain >= audit.mean_gain
    assert audit.synaptic_event_proxy > 0.0
    assert audit.control_cost > 0.0
    np.testing.assert_array_equal(controller.base_recurrent, original)
    assert not controller.base_recurrent.flags.writeable
    assert not controlled.states.flags.writeable
    with pytest.raises(ValueError):
        controlled.states[0, 0] = 1.0


def test_random_factorized_controller_is_seed_deterministic() -> None:
    first = FactorizedController.random(
        n_units=8,
        input_dim=2,
        seed=17,
    )
    second = FactorizedController.random(
        n_units=8,
        input_dim=2,
        seed=17,
    )
    third = FactorizedController.random(
        n_units=8,
        input_dim=2,
        seed=18,
    )
    np.testing.assert_array_equal(first.base_recurrent, second.base_recurrent)
    np.testing.assert_array_equal(first.gain_axes, second.gain_axes)
    assert not np.array_equal(first.base_recurrent, third.base_recurrent)
    assert first.base_recurrent_rank > first.control_dim


def test_factorized_controller_validates_modes_shapes_and_control_dimension() -> None:
    with pytest.raises(ValueError, match="control_dim"):
        FactorizedControllerConfig(control_dim=3)
    with pytest.raises(ValueError, match="one of"):
        ActuatorMode.coerce("full")
    assert ActuatorMode.coerce("recurrent") is ActuatorMode.LOW_RANK

    controller = _controller()
    with pytest.raises(ValueError, match="cannot be combined"):
        controller.step(
            np.zeros(3),
            np.zeros(2),
            mode="rgl",
            control=np.zeros(2),
            gain_control=np.zeros(2),
        )
    with pytest.raises(ValueError, match="time, control_dim"):
        controller.rollout(
            np.zeros(3),
            np.zeros((3, 2)),
            mode="gain",
            gain_controls=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="rank must exceed"):
        FactorizedController(
            base_recurrent=np.diag([1.0, 1.0, 0.0]),
            input_weights=np.ones((3, 2)),
            routing_axes=np.ones((2, 2)),
            gain_axes=np.ones((3, 2)),
            low_rank_left=np.ones((3, 2)),
            low_rank_right=np.ones((3, 2)),
            bias=np.zeros(3),
        )
