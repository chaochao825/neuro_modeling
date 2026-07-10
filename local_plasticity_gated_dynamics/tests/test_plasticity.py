"""Analytic contracts for three-factor and inhibitory local plasticity."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from src.plasticity.inhibitory_homeostasis import InhibitoryHomeostasis
from src.plasticity.three_factor import ThreeFactorRule, project_dale_weights
import src.models.ei_rate_network as ei_rate_network_module
import src.models.md_gate as md_gate_module
import src.plasticity.inhibitory_homeostasis as inhibitory_homeostasis_module
import src.plasticity.three_factor as three_factor_module


def _assert_three_factor_updates_equal(left: object, right: object) -> None:
    array_fields = (
        "eligibility_trace",
        "post_factor",
        "hebbian_update",
        "decay_update",
        "raw_update",
        "masked_update",
        "dale_applied_update",
    )
    for field in array_fields:
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert getattr(left, "costs") == getattr(right, "costs")


def _assert_homeostatic_updates_equal(left: object, right: object) -> None:
    array_fields = (
        "post_error",
        "raw_update",
        "masked_update",
        "dale_applied_update",
    )
    for field in array_fields:
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert getattr(left, "costs") == getattr(right, "costs")


def test_eligibility_uses_exact_causal_trace_and_reset() -> None:
    rule = ThreeFactorRule(learning_rate=0.1, tau_eligibility=2.0, dt=1.0)
    pre = np.array([1.0, 0.5, 0.0])
    first = rule.update_eligibility(pre)
    expected_first = (1.0 - np.exp(-0.5)) * pre
    np.testing.assert_allclose(first, expected_first)

    second = rule.update_eligibility(np.zeros(3))
    np.testing.assert_allclose(second, np.exp(-0.5) * expected_first)
    rule.reset(3)
    np.testing.assert_array_equal(rule.eligibility_trace, np.zeros(3))


def test_three_factor_returns_raw_masked_dale_applied_stages_and_costs() -> None:
    rule = ThreeFactorRule(learning_rate=1.0, tau_eligibility=1.0, dt=1.0)
    pre = np.array([1.0, 2.0])
    modulator = np.array([-1.0, 1.0])
    derivative = np.array([1.0, 0.5])
    mask = np.array([[1, 1], [0, 1]], dtype=bool)
    signs = np.array([1.0, -1.0])
    current = np.array([[0.2, -0.2], [0.0, -0.1]])

    update = rule.propose(
        pre,
        modulator,
        post_derivative=derivative,
        connectivity_mask=mask,
        presynaptic_signs=signs,
        current_weights=current,
    )
    trace = (1.0 - np.exp(-1.0)) * pre
    expected_raw = np.outer(modulator * derivative, trace)
    expected_masked = expected_raw * mask
    np.testing.assert_allclose(update.eligibility_trace, trace)
    np.testing.assert_allclose(update.raw_update, expected_raw)
    np.testing.assert_allclose(update.masked_update, expected_masked)

    candidate = current + expected_masked
    candidate[:, 0] = np.maximum(candidate[:, 0], 0.0)
    candidate[:, 1] = np.minimum(candidate[:, 1], 0.0)
    candidate *= mask
    expected_applied = candidate - current
    np.testing.assert_allclose(update.dale_applied_update, expected_applied)
    assert update.costs.raw_l1 == pytest.approx(np.sum(np.abs(expected_raw)))
    assert update.costs.masked_l1 == pytest.approx(np.sum(np.abs(expected_masked)))
    assert update.costs.applied_l1 == pytest.approx(np.sum(np.abs(expected_applied)))
    assert update.costs.applied_l2 == pytest.approx(np.linalg.norm(expected_applied))


@pytest.mark.parametrize(
    ("use_dale", "weight_decay"),
    ((True, 0.03), (False, 0.0)),
)
def test_three_factor_trusted_path_is_elementwise_public_equivalent(
    use_dale: bool,
    weight_decay: float,
) -> None:
    rng = np.random.default_rng(503)
    n_post, n_pre = 4, 5
    mask = rng.random((n_post, n_pre)) > 0.25
    signs = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
    current = project_dale_weights(rng.normal(size=(n_post, n_pre)), signs, mask)
    task = rng.normal(size=(n_post, n_pre))
    public_rule = ThreeFactorRule(
        learning_rate=0.07,
        tau_eligibility=2.3,
        dt=0.4,
        weight_decay=weight_decay,
    )
    trusted_rule = ThreeFactorRule(
        learning_rate=0.07,
        tau_eligibility=2.3,
        dt=0.4,
        weight_decay=weight_decay,
    )

    for _ in range(3):
        pre = rng.random(n_pre)
        modulator = rng.normal(size=n_post)
        derivative = rng.random(n_post)
        kwargs = {
            "post_derivative": derivative,
            "connectivity_mask": mask,
            "presynaptic_signs": signs if use_dale else None,
            "current_weights": current,
            "current_task_weights": task if weight_decay else None,
        }
        public = public_rule.propose(pre, modulator, **kwargs)
        trusted = trusted_rule._propose_trusted(pre, modulator, **kwargs)
        _assert_three_factor_updates_equal(public, trusted)
        np.testing.assert_array_equal(
            public_rule.eligibility_trace,
            trusted_rule.eligibility_trace,
        )


def test_three_factor_trusted_path_omits_redundant_validation() -> None:
    source = inspect.getsource(ThreeFactorRule._propose_trusted)
    assert "np.isin" not in source
    assert "np.allclose" not in source
    assert "project_dale_weights(" not in source


def test_weight_decay_is_separate_and_requires_task_component() -> None:
    rule = ThreeFactorRule(
        learning_rate=0.5,
        tau_eligibility=1.0,
        dt=1.0,
        weight_decay=0.2,
    )
    with pytest.raises(ValueError, match="current_task_weights"):
        rule.propose(np.ones(2), np.ones(2))

    task = np.array([[1.0, 0.0], [0.0, -2.0]])
    update = rule.propose(np.zeros(2), np.zeros(2), current_task_weights=task)
    np.testing.assert_array_equal(update.hebbian_update, np.zeros((2, 2)))
    np.testing.assert_allclose(update.decay_update, -0.1 * task)
    np.testing.assert_allclose(update.raw_update, update.decay_update)


def test_homeostasis_strengthens_and_weakens_inhibition_in_correct_direction() -> None:
    rule = InhibitoryHomeostasis(learning_rate=0.5, target_rate=0.5, dt=1.0)
    rates = np.array([0.9, 0.1, 0.8])
    excitatory = np.array([True, True, False])
    inhibitory = ~excitatory
    mask = np.ones((3, 3), dtype=bool)
    current = np.zeros((3, 3))
    current[:, 2] = -0.2

    update = rule.propose(
        rates,
        excitatory_mask=excitatory,
        inhibitory_mask=inhibitory,
        current_weights=current,
        connectivity_mask=mask,
    )
    # Above-target E cell receives more-negative inhibition.
    assert update.raw_update[0, 2] < 0.0
    assert update.dale_applied_update[0, 2] < 0.0
    # Below-target E cell weakens its inhibitory input toward zero.
    assert update.raw_update[1, 2] > 0.0
    assert update.dale_applied_update[1, 2] > 0.0
    assert current[1, 2] + update.dale_applied_update[1, 2] <= 0.0
    assert np.all(update.raw_update[2] == 0.0)
    assert np.all(update.raw_update[:, :2] == 0.0)


def test_homeostasis_never_crosses_inhibitory_weight_through_zero() -> None:
    rule = InhibitoryHomeostasis(learning_rate=10.0, target_rate=1.0)
    rates = np.array([0.0, 1.0])
    current = np.array([[0.0, -0.1], [0.0, -0.1]])
    update = rule.propose(
        rates,
        excitatory_mask=np.array([True, False]),
        inhibitory_mask=np.array([False, True]),
        current_weights=current,
    )
    assert current[0, 1] + update.dale_applied_update[0, 1] == pytest.approx(0.0)


def test_homeostasis_trusted_path_is_elementwise_public_equivalent() -> None:
    rng = np.random.default_rng(907)
    n_units = 7
    excitatory = np.array([True, True, True, True, True, False, False])
    inhibitory = ~excitatory
    sparse_mask = rng.random((n_units, n_units)) > 0.2
    current = rng.random((n_units, n_units))
    current[:, inhibitory] *= -1.0
    current = np.where(sparse_mask, current, 0.0)
    rates = rng.random(n_units)
    rule = InhibitoryHomeostasis(
        learning_rate=0.03,
        target_rate=0.4,
        dt=0.2,
        max_abs_update=0.002,
    )

    public = rule.propose(
        rates,
        excitatory_mask=excitatory,
        inhibitory_mask=inhibitory,
        current_weights=current,
        connectivity_mask=sparse_mask,
    )
    trusted = rule._propose_trusted(
        rates,
        excitatory_mask=excitatory,
        inhibitory_mask=inhibitory,
        current_weights=current,
        connectivity_mask=sparse_mask,
    )
    _assert_homeostatic_updates_equal(public, trusted)


def test_homeostasis_trusted_path_omits_redundant_validation() -> None:
    source = inspect.getsource(InhibitoryHomeostasis._propose_trusted)
    assert "np.isin" not in source
    assert "np.allclose" not in source


def test_rule_to_network_stages_keep_local_cost_separate_from_normalization() -> None:
    network = ei_rate_network_module.EIRateNetwork(
        8,
        connection_probability=1.0,
        allow_self_connections=True,
        seed=31,
    )
    plasticity = ThreeFactorRule(
        learning_rate=1e-3,
        tau_eligibility=2.0,
        dt=1.0,
    )
    rule_update = plasticity.propose(
        np.linspace(0.1, 0.8, 8),
        np.linspace(-0.2, 0.2, 8),
        connectivity_mask=network.connectivity_mask,
        presynaptic_signs=network.presynaptic_signs,
        current_weights=network.recurrent_weights,
    )
    application = network.apply_task_update(rule_update.dale_applied_update)
    np.testing.assert_allclose(application.local_update, rule_update.dale_applied_update)
    assert application.local_l1_cost == pytest.approx(rule_update.costs.applied_l1)
    np.testing.assert_allclose(network.W_task, rule_update.dale_applied_update)

    homeostasis = InhibitoryHomeostasis(
        learning_rate=1e-3,
        target_rate=0.2,
    )
    homeostatic_rule_update = homeostasis.propose(
        np.linspace(0.1, 0.8, 8),
        excitatory_mask=network.excitatory_mask,
        inhibitory_mask=network.inhibitory_mask,
        current_weights=network.recurrent_weights,
        connectivity_mask=network.connectivity_mask,
    )
    homeostatic_application = network.apply_homeostatic_update(
        homeostatic_rule_update.dale_applied_update
    )
    np.testing.assert_allclose(
        homeostatic_application.local_update,
        homeostatic_rule_update.dale_applied_update,
    )
    assert homeostatic_application.local_l1_cost == pytest.approx(
        homeostatic_rule_update.costs.applied_l1
    )
    assert np.all(network.W_homeo[:, network.excitatory_mask] == 0.0)


def test_plasticity_input_validation() -> None:
    rule = ThreeFactorRule(learning_rate=0.1, tau_eligibility=2.0)
    rule.reset(2)
    trace_before = rule.eligibility_trace
    with pytest.raises(ValueError):
        rule.propose(np.ones(2), np.ones(3), connectivity_mask=np.ones((2, 2)))
    np.testing.assert_array_equal(rule.eligibility_trace, trace_before)
    with pytest.raises(ValueError):
        rule.propose(np.ones(2), np.ones(2), presynaptic_signs=np.array([1.0, 0.0]))
    np.testing.assert_array_equal(rule.eligibility_trace, trace_before)
    with pytest.raises(ValueError, match="already satisfy"):
        rule.propose(
            np.ones(2),
            np.ones(2),
            connectivity_mask=np.ones((2, 2), dtype=bool),
            presynaptic_signs=np.array([1.0, -1.0]),
            current_weights=np.ones((2, 2)),
        )
    np.testing.assert_array_equal(rule.eligibility_trace, trace_before)
    with pytest.raises(ValueError, match="non-empty"):
        rule.propose(np.empty(0), np.ones(2))
    assert rule.eligibility_trace is not None
    with pytest.raises(ValueError, match="non-empty"):
        rule.propose(np.ones(2), np.empty(0))

    homeostasis = InhibitoryHomeostasis(learning_rate=0.1, target_rate=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        homeostasis.propose(
            np.array([0.5, -0.1]),
            excitatory_mask=np.array([True, False]),
            inhibitory_mask=np.array([False, True]),
            current_weights=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="partition"):
        homeostasis.propose(
            np.array([0.5, 0.1]),
            excitatory_mask=np.array([True, False]),
            inhibitory_mask=np.array([True, False]),
            current_weights=np.zeros((2, 2)),
        )

    with pytest.raises(TypeError, match="current_weights"):
        homeostasis.propose(
            np.array([0.5, 0.1]),
            excitatory_mask=np.array([True, False]),
            inhibitory_mask=np.array([False, True]),
        )


def test_phase_two_local_modules_have_no_torch_or_autograd_path() -> None:
    modules = (
        ei_rate_network_module,
        md_gate_module,
        three_factor_module,
        inhibitory_homeostasis_module,
    )
    for module in modules:
        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(name == "torch" or name.startswith("torch.") for name in imported)
        assert ".backward(" not in source
        assert "autograd" not in source.lower()
