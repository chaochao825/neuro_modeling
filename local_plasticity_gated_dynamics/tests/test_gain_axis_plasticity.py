"""Contracts for locally learned scalar-belief gain-axis updates."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.models.belief_gain import belief_gain_trajectory
from src.plasticity.gain_axis import (
    GainAxisThreeFactorRule,
    SignedPermutationTransform,
    apply_gain_axis_budget,
    apply_gain_axis_path_budget,
    make_deranged_shuffle,
    make_signed_permutation,
    orthogonal_component,
)
from src.plasticity.update_budget import UpdateBudgetController


PROJECT = Path(__file__).resolve().parents[1]


def test_exact_local_trace_and_scalar_three_factor_proposal() -> None:
    rule = GainAxisThreeFactorRule(
        learning_rate=0.2,
        tau_eligibility=2.0,
        dt=1.0,
        error_clip=0.5,
    )
    first_drive = np.array([1.0, -2.0, 0.5])
    retention = np.exp(-0.5)
    first_trace = (1.0 - retention) * first_drive
    np.testing.assert_allclose(rule.update_eligibility(first_drive), first_trace)

    second_drive = np.array([-0.5, 0.25, 1.0])
    second_trace = retention * first_trace + (1.0 - retention) * second_drive
    np.testing.assert_allclose(rule.update_eligibility(second_drive), second_trace)
    update = rule.propose(task_error=2.0, third_factor=-0.75)
    expected = 0.2 * second_trace * 0.5 * -0.75
    np.testing.assert_allclose(update.raw_update, expected)
    np.testing.assert_allclose(update.eligibility_trace, second_trace)
    assert update.task_error == 2.0
    assert update.clipped_task_error == 0.5
    assert update.scalar_modulator == pytest.approx(-0.375)
    assert update.costs.raw_l1 == pytest.approx(np.sum(np.abs(expected)))
    assert update.costs.raw_l2 == pytest.approx(np.linalg.norm(expected))
    assert not update.raw_update.flags.writeable
    assert not update.eligibility_trace.flags.writeable

    returned_trace = rule.eligibility_trace
    assert returned_trace is not None
    returned_trace[:] = 0.0
    np.testing.assert_allclose(rule.eligibility_trace, second_trace)


def test_step_is_atomic_when_scalar_validation_fails() -> None:
    rule = GainAxisThreeFactorRule(
        learning_rate=0.1,
        tau_eligibility=3.0,
    )
    rule.update_eligibility(np.array([0.2, 0.4]))
    before = rule.eligibility_trace
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        rule.step(
            np.array([1.0, 1.0]),
            task_error=1.0,
            third_factor=1.1,
        )
    np.testing.assert_array_equal(rule.eligibility_trace, before)


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_vector_budget_adapter_matches_the_selected_norm(norm: str) -> None:
    raw = np.array([3.0, -4.0, 1.0])
    target = 2.0
    controller = UpdateBudgetController(target, norm, planned_events=1)
    original = raw.copy()
    application = apply_gain_axis_budget(controller, raw)

    np.testing.assert_array_equal(raw, original)
    selected = (
        np.sum(np.abs(application.applied_update))
        if norm == "l1"
        else np.linalg.norm(application.applied_update)
    )
    assert selected == pytest.approx(target)
    assert application.selected_norm == norm
    assert 0.0 < application.scale_factor <= 1.0
    assert application.processed_events == 1
    assert controller.summary().attained
    assert application.fingerprint
    assert not application.raw_update.flags.writeable
    assert not application.applied_update.flags.writeable


def test_zero_vector_budget_event_is_retained_without_amplification() -> None:
    controller = UpdateBudgetController(1.0, "l1", planned_events=1)
    application = apply_gain_axis_budget(controller, np.zeros(4))
    np.testing.assert_array_equal(application.applied_update, np.zeros(4))
    assert application.scale_factor == 0.0
    summary = controller.summary()
    assert summary.zero_proposal_events == 1
    assert summary.final_shortfall == pytest.approx(1.0)
    assert not summary.attained


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_global_path_budget_preserves_relative_event_magnitudes(norm: str) -> None:
    raw = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.25, 3.0, -1.0],
            [-2.0, 0.5, 4.0],
        ]
    )
    raw_path = (
        float(np.sum(np.abs(raw)))
        if norm == "l1"
        else float(np.sum(np.linalg.norm(raw, axis=1)))
    )
    application = apply_gain_axis_path_budget(
        raw,
        total_budget=0.4 * raw_path,
        norm=norm,
    )

    np.testing.assert_allclose(application.applied_updates, 0.4 * raw)
    assert application.scale_factor == pytest.approx(0.4)
    assert application.selected_applied == pytest.approx(0.4 * raw_path)
    assert application.attained
    assert application.final_shortfall == pytest.approx(0.0)
    assert application.raw_nonzero_events == 3
    assert application.applied_nonzero_events == 3
    assert application.zero_proposal_events == 0
    assert not application.raw_updates.flags.writeable
    assert not application.applied_updates.flags.writeable


def test_global_path_budget_never_amplifies_an_insufficient_tape() -> None:
    raw = np.array([[0.1, -0.2], [0.0, 0.0]])
    application = apply_gain_axis_path_budget(
        raw,
        total_budget=1.0,
        norm="l1",
    )
    np.testing.assert_array_equal(application.applied_updates, raw)
    assert application.scale_factor == 1.0
    assert not application.attained
    assert application.final_shortfall == pytest.approx(0.7)
    assert application.zero_proposal_events == 1


def test_signed_permutation_is_deterministic_group_preserving_and_norm_exact() -> None:
    groups = np.array([0, 0, 0, 1, 1, 1])
    first = make_signed_permutation(
        6,
        seed=9,
        group_labels=groups,
        deranged=False,
        sign_flips=True,
    )
    second = make_signed_permutation(
        6,
        seed=9,
        group_labels=groups,
        deranged=False,
        sign_flips=True,
    )
    assert isinstance(first, SignedPermutationTransform)
    np.testing.assert_array_equal(first.permutation, second.permutation)
    np.testing.assert_array_equal(first.signs, second.signs)
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(
        first.group_labels[first.permutation], first.group_labels
    )

    vector = np.array([0.5, -1.0, 2.0, -3.0, 4.0, -5.0])
    original = vector.copy()
    transformed = first.apply(vector)
    np.testing.assert_array_equal(vector, original)
    assert np.sum(np.abs(transformed)) == pytest.approx(np.sum(np.abs(vector)))
    assert np.linalg.norm(transformed) == pytest.approx(np.linalg.norm(vector))
    assert not transformed.flags.writeable


def test_deranged_shuffle_has_no_fixed_points_or_sign_flips() -> None:
    groups = np.array([False, False, False, True, True, True])
    shuffle = make_deranged_shuffle(6, seed=13, group_labels=groups)
    assert shuffle.deranged
    assert np.all(shuffle.permutation != np.arange(6))
    np.testing.assert_array_equal(shuffle.signs, np.ones(6))
    np.testing.assert_array_equal(
        shuffle.group_labels[shuffle.permutation], shuffle.group_labels
    )
    vector = np.arange(1.0, 7.0)
    np.testing.assert_array_equal(shuffle.apply(vector), vector[shuffle.permutation])


def test_orthogonal_component_removes_only_reference_projection() -> None:
    vector = np.array([2.0, -1.0, 3.0, 0.5])
    reference = np.array([1.0, 2.0, -1.0, 0.0])
    projected = orthogonal_component(vector, reference)
    assert float(reference @ projected) == pytest.approx(0.0, abs=1e-12)
    expected = vector - reference * ((reference @ vector) / (reference @ reference))
    np.testing.assert_allclose(projected, expected)
    assert not projected.flags.writeable
    with pytest.raises(ValueError, match="non-zero norm"):
        orthogonal_component(vector, np.zeros_like(reference))


def test_budget_bound_maps_directly_to_positive_belief_gain_coefficients() -> None:
    total_budget = 0.6
    controller = UpdateBudgetController(total_budget, "l1", planned_events=2)
    coefficient = np.zeros(4)
    for proposal in (
        np.array([1.0, -2.0, 0.5, 0.25]),
        np.array([-0.25, 1.0, -1.5, 0.75]),
    ):
        coefficient += apply_gain_axis_budget(controller, proposal).applied_update
    assert np.max(np.abs(coefficient)) <= total_budget + 1e-12

    axis = coefficient / total_budget
    posterior = np.array([0.0, 0.5, 1.0])
    epochs = np.array(["cue", "sensory"])
    trajectory = belief_gain_trajectory(
        posterior,
        epochs,
        axis,
        strength=total_budget,
    )
    expected = np.ones_like(trajectory.gains)
    expected[:, 1] += (2.0 * posterior - 1.0)[:, None] * coefficient[None, :]
    np.testing.assert_allclose(trajectory.gains, expected)
    assert np.min(trajectory.gains) >= 1.0 - total_budget - 1e-12
    assert np.min(trajectory.gains) > 0.0


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (
            lambda: GainAxisThreeFactorRule(learning_rate=-1.0, tau_eligibility=1.0),
            ValueError,
        ),
        (
            lambda: GainAxisThreeFactorRule(learning_rate=0.1, tau_eligibility=0.0),
            ValueError,
        ),
        (
            lambda: GainAxisThreeFactorRule(
                learning_rate=0.1, tau_eligibility=1.0, error_clip=0.0
            ),
            ValueError,
        ),
        (
            lambda: make_signed_permutation(
                3, seed=0, group_labels=np.array([0.0, 0.0, 1.0])
            ),
            TypeError,
        ),
        (
            lambda: make_deranged_shuffle(3, seed=0, group_labels=np.array([0, 0, 1])),
            ValueError,
        ),
    ],
)
def test_strict_configuration_validation(
    factory: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        factory()


def test_gain_axis_module_has_no_torch_autograd_or_bptt_path() -> None:
    path = PROJECT / "src" / "plasticity" / "gain_axis.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert ".backward(" not in source
    assert "autograd" not in source.lower()
    assert "bptt" not in source.lower()
