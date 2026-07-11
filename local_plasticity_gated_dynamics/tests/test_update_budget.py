"""Contracts for selected-norm online plasticity-budget matching."""

from __future__ import annotations

import numpy as np
import pytest

from src.plasticity.update_budget import UpdateBudgetController


def test_l1_panel_reaches_target_without_overshoot() -> None:
    controller = UpdateBudgetController(6.0, "l1", 3, tolerance=1e-12)
    proposals = [
        np.array([[3.0, -2.0], [1.0, 0.0]]),
        np.array([[4.0, 0.0], [0.0, 2.0]]),
        np.array([[1.0, -1.0], [1.0, -1.0]]),
    ]

    cumulative = 0.0
    for proposal in proposals:
        original = proposal.copy()
        applied = controller.scale(proposal)
        np.testing.assert_array_equal(proposal, original)
        cumulative += float(np.sum(np.abs(applied)))
        assert cumulative <= 6.0 + 1e-12

    summary = controller.summary()
    assert summary.selected_norm == "l1"
    assert summary.secondary_norm == "l2"
    assert summary.cumulative_applied_l1 == pytest.approx(6.0)
    assert summary.selected_applied == pytest.approx(6.0)
    assert summary.remaining == pytest.approx(0.0, abs=1e-12)
    assert summary.complete is True
    assert summary.attained is True
    assert summary.final_shortfall == pytest.approx(0.0, abs=1e-12)
    assert summary.simultaneous_dual_norm_match is False
    assert summary.secondary_norm_is_diagnostic_only is True


def test_l2_panel_matches_only_l2_and_records_l1_diagnostic() -> None:
    controller = UpdateBudgetController(2.0, "l2", 2)
    proposal = np.array([[3.0, 4.0], [0.0, 0.0]])
    first = controller.scale(proposal)
    second = controller.scale(proposal)

    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert np.linalg.norm(second) == pytest.approx(1.0)
    summary = controller.summary()
    assert summary.cumulative_applied_l2 == pytest.approx(2.0)
    assert summary.cumulative_applied_l1 == pytest.approx(2.8)
    assert summary.cumulative_applied_l1 != pytest.approx(summary.total_budget)
    assert summary.attained is True
    assert summary.as_dict()["simultaneous_dual_norm_match"] is False


def test_zero_proposal_consumes_slot_and_retains_final_shortfall() -> None:
    controller = UpdateBudgetController(3.0, "l1", 2)
    first = controller.scale(np.full((2, 2), 2.0))
    second = controller.scale(np.zeros((2, 2)))

    assert np.sum(np.abs(first)) == pytest.approx(1.5)
    np.testing.assert_array_equal(second, np.zeros((2, 2)))
    summary = controller.summary()
    assert summary.zero_proposal_events == 1
    assert summary.raw_nonzero_events == 1
    assert summary.applied_nonzero_events == 1
    assert summary.processed_events == 2
    assert summary.remaining == pytest.approx(1.5)
    assert summary.final_shortfall == pytest.approx(1.5)
    assert summary.attained is False


def test_early_zero_budget_can_be_reallocated_to_later_events() -> None:
    controller = UpdateBudgetController(4.0, "l1", 3)
    controller.scale(np.zeros((1, 2)))
    second = controller.scale(np.array([[5.0, -5.0]]))
    third = controller.scale(np.array([[5.0, -5.0]]))

    assert np.sum(np.abs(second)) == pytest.approx(2.0)
    assert np.sum(np.abs(third)) == pytest.approx(2.0)
    summary = controller.summary()
    assert summary.zero_proposal_events == 1
    assert summary.cumulative_applied_l1 == pytest.approx(4.0)
    assert summary.attained is True


def test_insufficient_proposals_are_not_amplified_and_report_shortfall() -> None:
    controller = UpdateBudgetController(4.0, "l1", 2)
    first = controller.scale(np.array([[0.25, -0.25]]))
    second = controller.scale(np.array([[1.0, 0.0]]))

    np.testing.assert_array_equal(first, np.array([[0.25, -0.25]]))
    np.testing.assert_array_equal(second, np.array([[1.0, 0.0]]))
    summary = controller.summary()
    assert summary.cumulative_raw_l1 == pytest.approx(1.5)
    assert summary.cumulative_applied_l1 == pytest.approx(1.5)
    assert summary.final_shortfall == pytest.approx(2.5)
    assert summary.attained is False


def test_scaling_preserves_a_sparse_dale_feasible_segment() -> None:
    current = np.array([[0.4, -0.5], [0.3, -0.2]])
    mask = np.array([[True, True], [True, False]])
    current = np.where(mask, current, 0.0)
    # The full proposal reaches another sparse Dale-valid matrix.
    dale_applied = np.array([[-0.2, 0.2], [0.1, 0.0]])
    controller = UpdateBudgetController(0.25, "l1", 1)
    budgeted = controller.scale(dale_applied)
    candidate = current + budgeted

    assert np.all(candidate[:, 0] >= 0.0)
    assert np.all(candidate[:, 1] <= 0.0)
    assert np.all(candidate[~mask] == 0.0)
    assert np.sum(np.abs(budgeted)) == pytest.approx(0.25)


def test_summary_is_deterministic_and_incomplete_shortfall_is_undefined() -> None:
    left = UpdateBudgetController(1.0, "l2", 2)
    right = UpdateBudgetController(1.0, "l2", 2)
    proposal = np.array([[1.0, 2.0], [-3.0, 4.0]])

    np.testing.assert_array_equal(left.scale(proposal), right.scale(proposal))
    assert left.summary() == right.summary()
    assert left.summary().complete is False
    assert left.summary().attained is False
    assert left.summary().final_shortfall is None


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((-1.0, "l1", 1), ValueError),
        ((np.inf, "l1", 1), ValueError),
        ((True, "l1", 1), TypeError),
        ((1.0, "bad", 1), ValueError),
        ((1.0, 1, 1), TypeError),
        ((1.0, "l1", 0), ValueError),
        ((1.0, "l1", True), TypeError),
        ((1.0, "l1", 1, -1.0), ValueError),
    ],
)
def test_configuration_validation(
    args: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        UpdateBudgetController(*args)


def test_proposal_validation_and_event_limit() -> None:
    controller = UpdateBudgetController(1.0, "l1", 1)
    with pytest.raises(ValueError, match="two-dimensional"):
        controller.scale(np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        controller.scale(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="real-valued"):
        controller.scale(np.array([[1.0 + 2.0j]]))
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="norms must be finite"):
            controller.scale(np.full((2, 2), np.finfo(float).max))

    controller.scale(np.ones((2, 2)))
    with pytest.raises(RuntimeError, match="already been processed"):
        controller.scale(np.ones((2, 2)))


def test_proposal_shape_cannot_change_across_events() -> None:
    controller = UpdateBudgetController(1.0, "l1", 2)
    controller.scale(np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape changed"):
        controller.scale(np.ones((1, 4)))


def test_zero_budget_returns_zero_without_hiding_raw_cost() -> None:
    controller = UpdateBudgetController(0.0, "l1", 1)
    proposal = np.array([[1.0, -2.0]])
    np.testing.assert_array_equal(controller.scale(proposal), np.zeros_like(proposal))
    summary = controller.summary()
    assert summary.cumulative_raw_l1 == pytest.approx(3.0)
    assert summary.cumulative_applied_l1 == 0.0
    assert summary.attained is True
