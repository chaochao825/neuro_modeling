from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import src.plasticity.parameterizations as parameterization_module
from src.plasticity.parameterizations import (
    DirectAdditivePlasticity,
    FullPerSynapsePlasticity,
    SignPreservingMultiplicativePlasticity,
)


def _dale_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signs = np.array([1.0, 1.0, -1.0, -1.0])
    magnitudes = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 3.0, 2.0, 5.0],
            [4.0, 1.0, 3.0, 2.0],
        ]
    )
    mask = np.ones((4, 4), dtype=bool)
    current = magnitudes * signs[None, :]
    return current, mask, signs


def test_direct_parameterization_matches_the_current_outer_product_rule() -> None:
    current, mask, signs = _dale_state()
    post = np.array([0.2, -0.1, 0.4, -0.3])
    eligibility = np.array([1.0, 0.5, -0.25, 0.75])
    before = current.copy()

    update = DirectAdditivePlasticity(learning_rate=0.05, credit_dimension=2).propose(
        post,
        eligibility,
        current_weights=current,
        connectivity_mask=mask,
        presynaptic_signs=signs,
    )
    expected = 0.05 * np.outer(post, eligibility)

    np.testing.assert_allclose(update.raw_control_update, expected)
    np.testing.assert_allclose(update.masked_control_update, expected)
    assert update.parameterization == "direct_additive"
    assert update.control_space == "weight"
    assert update.credit_dimension == 2
    assert update.costs.raw_weight_l1 == pytest.approx(np.sum(np.abs(expected)))
    assert update.costs.raw_weight_l2 == pytest.approx(np.linalg.norm(expected))
    assert update.costs.masked_control_l2 == pytest.approx(np.linalg.norm(expected))
    assert update.costs.applied_control_l1 == pytest.approx(
        np.sum(np.abs(update.applied_control_update))
    )
    assert update.costs.applied_weight_l2 == pytest.approx(
        np.linalg.norm(update.dale_applied_update)
    )
    assert update.control_scale == 1.0
    assert not update.control_bound_active
    np.testing.assert_array_equal(current, before)
    assert not update.dale_applied_update.flags.writeable


def test_multiplicative_parameterization_preserves_sparse_dale_signs() -> None:
    current, mask, signs = _dale_state()
    post = np.array([0.6, -0.3, 0.2, -0.5])
    eligibility = np.array([0.4, -0.2, 0.7, -0.6])
    rule = SignPreservingMultiplicativePlasticity(
        learning_rate=0.5, credit_dimension=3, max_abs_log_step=None
    )

    update = rule.propose(
        post,
        eligibility,
        current_weights=current,
        connectivity_mask=mask,
        presynaptic_signs=signs,
    )
    after = current + update.dale_applied_update
    expected = current * np.exp(update.applied_control_update)

    np.testing.assert_allclose(after, expected)
    np.testing.assert_allclose(
        update.raw_weight_update,
        current * np.expm1(update.raw_control_update),
    )
    np.testing.assert_allclose(
        update.masked_weight_update,
        current * np.expm1(update.masked_control_update),
    )
    assert np.all(after[:, signs > 0.0] > 0.0)
    assert np.all(after[:, signs < 0.0] < 0.0)
    assert update.control_space == "log_magnitude"
    assert update.credit_dimension == 3
    assert np.linalg.matrix_rank(update.raw_control_update, tol=1e-12) == 1
    assert np.linalg.matrix_rank(update.dale_applied_update, tol=1e-12) > 1
    for stage, values in (
        ("raw_control", update.raw_control_update),
        ("masked_control", update.masked_control_update),
        ("applied_control", update.applied_control_update),
        ("raw_weight", update.raw_weight_update),
        ("masked_weight", update.masked_weight_update),
        ("applied_weight", update.dale_applied_update),
    ):
        assert getattr(update.costs, f"{stage}_l1") == pytest.approx(
            np.sum(np.abs(values))
        )
        assert getattr(update.costs, f"{stage}_l2") == pytest.approx(
            np.linalg.norm(values)
        )


def test_direct_parameterization_projects_the_candidate_not_the_update_sign() -> None:
    current, mask, signs = _dale_state()
    update = DirectAdditivePlasticity(learning_rate=100.0, credit_dimension=2).propose(
        -np.ones(4),
        np.ones(4),
        current_weights=current,
        connectivity_mask=mask,
        presynaptic_signs=signs,
    )
    after = current + update.dale_applied_update

    assert np.all(after[:, signs > 0.0] >= 0.0)
    assert np.all(after[:, signs < 0.0] <= 0.0)
    assert np.any(update.raw_weight_update[:, signs > 0.0] < 0.0)


def test_multiplicative_rule_respects_mask_and_records_control_clipping() -> None:
    current, mask, signs = _dale_state()
    mask[0, 0] = False
    current[0, 0] = 0.0
    update = SignPreservingMultiplicativePlasticity(
        learning_rate=10.0, credit_dimension=2, max_abs_log_step=0.05
    ).propose(
        np.ones(4),
        np.ones(4),
        current_weights=current,
        connectivity_mask=mask,
        presynaptic_signs=signs,
    )

    assert np.max(np.abs(update.applied_control_update)) == pytest.approx(0.05)
    assert update.control_bound_active
    assert update.control_scale == pytest.approx(0.005)
    assert update.pre_scale_exceedance_fraction == pytest.approx(1.0)
    np.testing.assert_allclose(
        update.applied_control_update,
        update.control_scale * update.masked_control_update,
    )
    assert np.linalg.matrix_rank(
        update.applied_control_update, tol=1e-12
    ) == np.linalg.matrix_rank(update.masked_control_update, tol=1e-12)
    assert update.dale_applied_update[0, 0] == 0.0
    assert np.all(update.dale_applied_update[~mask] == 0.0)


def test_full_per_synapse_control_has_one_independent_channel_per_edge() -> None:
    current, mask, signs = _dale_state()
    mask[0, 0] = False
    current[0, 0] = 0.0
    third_factor = np.arange(1.0, 17.0).reshape(4, 4)
    eligibility = np.array([0.5, 1.0, -0.5, 2.0])

    update = FullPerSynapsePlasticity(learning_rate=0.01).propose(
        third_factor,
        eligibility,
        current_weights=current,
        connectivity_mask=mask,
        presynaptic_signs=signs,
    )
    expected_raw = 0.01 * third_factor * eligibility[None, :]

    np.testing.assert_allclose(update.raw_control_update, expected_raw)
    np.testing.assert_allclose(
        update.masked_control_update, np.where(mask, expected_raw, 0.0)
    )
    assert update.credit_dimension == np.count_nonzero(mask)
    assert update.parameterization == "full_per_synapse"
    assert np.linalg.matrix_rank(update.raw_control_update, tol=1e-12) > 1
    assert np.all(update.dale_applied_update[~mask] == 0.0)
    after = current + update.dale_applied_update
    assert np.all(after[:, signs > 0.0] >= 0.0)
    assert np.all(after[:, signs < 0.0] <= 0.0)


def test_parameterizations_reject_invalid_weight_state_and_inputs() -> None:
    current, mask, signs = _dale_state()
    invalid = current.copy()
    invalid[0, 0] *= -1.0
    with pytest.raises(ValueError, match="Dale"):
        DirectAdditivePlasticity(learning_rate=0.1, credit_dimension=2).propose(
            np.ones(4),
            np.ones(4),
            current_weights=invalid,
            connectivity_mask=mask,
            presynaptic_signs=signs,
        )
    with pytest.raises(ValueError, match="must match"):
        FullPerSynapsePlasticity(learning_rate=0.1).propose(
            np.ones((3, 3)),
            np.ones(4),
            current_weights=current,
            connectivity_mask=mask,
            presynaptic_signs=signs,
        )
    with pytest.raises(ValueError, match="credit_dimension"):
        DirectAdditivePlasticity(learning_rate=0.1, credit_dimension=0)


def test_parameterization_module_has_no_torch_or_autograd_path() -> None:
    source = inspect.getsource(parameterization_module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert "autograd" not in source.lower()
    assert ".backward(" not in source
