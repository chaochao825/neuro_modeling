from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.functional_budget_matching import (
    FunctionalBudgetError,
    audit_joint_functional_budget,
    functional_observables,
    functional_state_displacement,
    match_functional_state_displacement,
)


def test_functional_state_displacement_uses_state_norm_not_parameter_norm() -> None:
    frozen = np.zeros((3, 2))
    controlled = np.array(
        [
            [9.0, 9.0],
            [3.0, 4.0],
            [0.0, 5.0],
        ]
    )
    assert functional_state_displacement(controlled, frozen) == pytest.approx(25.0)
    mask = np.array([True, False])
    assert functional_state_displacement(
        controlled,
        frozen,
        sample_mask=mask,
    ) == pytest.approx(25.0)
    assert functional_state_displacement(
        controlled,
        frozen,
        exclude_initial=False,
    ) == pytest.approx((162.0 + 25.0 + 25.0) / 3.0)


def test_matcher_finds_closed_loop_scale_and_keeps_search_receipt() -> None:
    frozen = np.zeros((5, 2))
    direction = np.array([3.0, 4.0])

    def rollout(scale: float) -> SimpleNamespace:
        states = frozen.copy()
        states[1:] = scale * direction
        return SimpleNamespace(states=states)

    match = match_functional_state_displacement(
        rollout,
        frozen,
        target_displacement=6.25,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
    )
    # Mean squared state norm is 25 * scale^2.
    assert match.scale == pytest.approx(0.5, abs=1e-8)
    assert match.achieved_displacement == pytest.approx(6.25, abs=1e-8)
    assert match.converged
    assert match.n_evaluations >= 3
    assert match.evaluated_scales[0] == 0.0
    assert match.evaluated_displacements[0] == 0.0
    assert not match.controlled_states.flags.writeable


def test_matcher_fails_closed_for_mismatched_tape_or_unreachable_budget() -> None:
    frozen = np.zeros((4, 2))

    with pytest.raises(ValueError, match="does not reproduce"):
        match_functional_state_displacement(
            lambda scale: np.ones_like(frozen),
            frozen,
            target_displacement=1.0,
        )

    def bounded_rollout(scale: float) -> np.ndarray:
        states = frozen.copy()
        states[1:, 0] = min(scale, 1.0)
        return states

    with pytest.raises(FunctionalBudgetError, match="unreachable"):
        match_functional_state_displacement(
            bounded_rollout,
            frozen,
            target_displacement=4.0,
            max_scale=2.0,
        )
    retained = match_functional_state_displacement(
        bounded_rollout,
        frozen,
        target_displacement=4.0,
        max_scale=2.0,
        raise_on_unreachable=False,
    )
    assert not retained.converged
    assert retained.achieved_displacement == pytest.approx(1.0)


def test_functional_observables_report_nonparameter_costs() -> None:
    frozen = np.zeros((3, 2))
    controlled = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    observables = functional_observables(
        controlled,
        frozen,
        rates=controlled[1:],
        gains=np.array([[1.0, 1.2], [0.8, 1.0]]),
        synaptic_event_proxy_by_step=np.array([2.0, 4.0]),
    )
    assert observables.state_displacement == pytest.approx(2.5)
    assert observables.mean_rate == pytest.approx(0.75)
    assert observables.max_absolute_rate == pytest.approx(2.0)
    assert observables.mean_gain == pytest.approx(1.0)
    assert observables.max_gain == pytest.approx(1.2)
    assert observables.synaptic_event_proxy == pytest.approx(3.0)


def test_joint_functional_budget_requires_every_registered_term() -> None:
    frozen_states = np.zeros((2, 3, 2))
    controlled_states = frozen_states.copy()
    controlled_states[:, 1:, 0] = 0.5
    frozen_rates = np.full((2, 2, 2), 0.5)
    controlled_rates = frozen_rates + 0.05
    frozen_gains = np.ones_like(frozen_rates)
    controlled_gains = frozen_gains.copy()
    controlled_gains[..., 0] += 0.2
    frozen_events = np.full((2, 2), 4.0)
    controlled_events = np.full((2, 2), 4.4)

    audit = audit_joint_functional_budget(
        controlled_states,
        frozen_states,
        controlled_rates=controlled_rates,
        frozen_rates=frozen_rates,
        controlled_gains=controlled_gains,
        frozen_gains=frozen_gains,
        controlled_event_proxy_by_step=controlled_events,
        frozen_event_proxy_by_step=frozen_events,
        target_state_displacement=0.25,
        target_mean_absolute_rate_change=0.05,
        state_relative_tolerance=0.01,
        rate_change_relative_tolerance=0.11,
        gain_envelope_limit=0.25,
        event_change_relative_tolerance=0.11,
    )
    assert audit.state_valid
    assert audit.rate_valid
    assert audit.gain_valid
    assert audit.event_valid
    assert audit.joint_valid
    assert audit.mean_absolute_rate_change == pytest.approx(0.05)
    assert audit.rate_change_relative_error == pytest.approx(0.0)
    assert audit.rate_change_relative_to_frozen == pytest.approx(0.1)
    assert audit.gain_envelope == pytest.approx(0.2)
    assert audit.event_change_relative_to_frozen == pytest.approx(0.1)

    failed = audit_joint_functional_budget(
        controlled_states,
        frozen_states,
        controlled_rates=controlled_rates,
        frozen_rates=frozen_rates,
        controlled_gains=controlled_gains,
        frozen_gains=frozen_gains,
        controlled_event_proxy_by_step=controlled_events,
        frozen_event_proxy_by_step=frozen_events,
        target_state_displacement=0.25,
        target_mean_absolute_rate_change=0.04,
        state_relative_tolerance=0.01,
        rate_change_relative_tolerance=0.20,
        gain_envelope_limit=0.25,
        event_change_relative_tolerance=0.11,
    )
    assert failed.state_valid
    assert not failed.rate_valid
    assert not failed.joint_valid
