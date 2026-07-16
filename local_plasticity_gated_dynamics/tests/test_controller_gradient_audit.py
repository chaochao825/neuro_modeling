"""Controller direction and realized-improvement audit contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.controller_gradient_audit import (
    audit_controller_update,
    directional_cosine,
    evaluate_update_improvement,
    summarize_improvement_trials,
)


def test_directional_cosine_reports_aligned_opposed_and_ineligible_updates() -> None:
    reference = np.array([1.0, -2.0, 0.5])
    aligned = directional_cosine(3.0 * reference, reference)
    opposed = directional_cosine(-reference, reference)
    zero = directional_cosine(np.zeros(3), reference)

    assert aligned.eligible
    assert aligned.cosine == pytest.approx(1.0)
    assert opposed.eligible
    assert opposed.cosine == pytest.approx(-1.0)
    assert not zero.eligible
    assert np.isnan(zero.cosine)
    assert zero.reason == "proposed_update_norm_below_tolerance"


def test_realized_improvement_reruns_objective_without_mutating_inputs() -> None:
    parameters = np.array([1.0, -2.0, 0.5])
    update = np.array([-0.5, 0.5, -0.25])
    before_parameters = parameters.copy()
    before_update = update.copy()

    def objective(value: np.ndarray) -> float:
        assert not value.flags.writeable
        return float(0.5 * value @ value)

    result = evaluate_update_improvement(parameters, update, objective)
    assert result.candidate_loss < result.baseline_loss
    assert result.improved
    assert result.absolute_improvement == pytest.approx(
        result.baseline_loss - result.candidate_loss
    )
    np.testing.assert_array_equal(result.candidate_parameters, parameters + update)
    np.testing.assert_array_equal(parameters, before_parameters)
    np.testing.assert_array_equal(update, before_update)
    assert not result.candidate_parameters.flags.writeable


def test_improvement_probability_uses_paired_experimental_units() -> None:
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([0.5, 2.2, 2.5, 4.0])
    audit = summarize_improvement_trials(baseline, candidate)
    assert audit.n_units == 4
    assert audit.n_improved == 2
    assert audit.probability_improved == pytest.approx(0.5)
    np.testing.assert_allclose(audit.improvements, baseline - candidate)
    assert audit.mean_improvement == pytest.approx(np.mean(baseline - candidate))


def test_joint_controller_audit_compares_update_sign_not_gradient_sign() -> None:
    parameters = np.array([1.0, -1.0])
    exact_update = np.array([-1.0, 1.0])
    local_update = 0.4 * exact_update
    audit = audit_controller_update(
        parameters,
        local_update,
        exact_update,
        lambda value: float(0.5 * value @ value),
    )
    assert audit.direction.cosine == pytest.approx(1.0)
    assert audit.improvement.improved
    assert audit.improvement.absolute_improvement > 0.0
    assert not audit.local_update.flags.writeable
    assert not audit.reference_update.flags.writeable


def test_audits_reject_shape_mismatch_and_nonfinite_objective() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        directional_cosine(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        evaluate_update_improvement(
            np.ones(2),
            np.zeros(2),
            lambda _: float("nan"),
        )
