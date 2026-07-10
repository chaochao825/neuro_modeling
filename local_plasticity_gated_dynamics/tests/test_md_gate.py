"""Fast contracts for Hebbian MD context inference and PFC gain gating."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.md_gate import MDGate


def test_wta_is_one_hot_and_gain_is_winner_specific() -> None:
    gain_weights = np.zeros((6, 4))
    gain_weights[:, 0] = np.linspace(0.0, 1.0, 6)
    gain_weights[:, 1] = np.linspace(1.0, 0.0, 6)
    gate = MDGate(
        6,
        n_md=4,
        tau_trace=1.0,
        dt=1.0,
        gain_strength=0.5,
        gain_weights=gain_weights,
        seed=2,
    )
    # Bias makes the expected winner independent of random initialization.
    output = gate.step(np.ones(6), md_bias=np.array([0.0, 10.0, 0.0, 0.0]))

    assert output.winner == 1
    np.testing.assert_array_equal(output.md_activity, np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(output.pfc_gain, 1.0 + 0.5 * gain_weights[:, 1])
    assert np.all(output.pfc_gain > 0.0)


def test_presynaptic_trace_and_competitive_hebbian_update_are_local() -> None:
    gate = MDGate(
        5,
        n_md=4,
        learning_rate=0.2,
        tau_trace=2.0,
        dt=1.0,
        seed=5,
    )
    pfc = np.array([1.0, 0.5, 0.0, 0.25, 0.75])
    before = gate.hebbian_weights
    output = gate.step(
        pfc,
        learn=True,
        modulatory_signal=1.0,
        md_bias=np.array([20.0, 0.0, 0.0, 0.0]),
    )

    expected_trace = (1.0 - np.exp(-0.5)) * pfc
    np.testing.assert_allclose(output.pfc_trace, expected_trace)
    changed_rows = np.flatnonzero(np.any(np.abs(gate.hebbian_weights - before) > 1e-12, axis=1))
    np.testing.assert_array_equal(changed_rows, np.array([0]))
    np.testing.assert_allclose(gate.hebbian_weights - before, output.hebbian_update)
    np.testing.assert_allclose(np.linalg.norm(gate.hebbian_weights, axis=1), 1.0)


def test_zero_modulator_and_inference_do_not_change_weights() -> None:
    gate = MDGate(5, n_md=4, learning_rate=0.2, seed=8)
    before = gate.hebbian_weights
    no_learning = gate.step(np.ones(5), learn=False)
    np.testing.assert_allclose(gate.hebbian_weights, before)
    assert np.all(no_learning.hebbian_update == 0.0)

    zero_modulator = gate.step(np.ones(5), learn=True, modulatory_signal=0.0)
    np.testing.assert_allclose(gate.hebbian_weights, before)
    assert np.all(zero_modulator.hebbian_update == 0.0)


def test_reset_clears_trace_but_retains_learned_weights() -> None:
    gate = MDGate(4, n_md=4, tau_trace=2.0, seed=9)
    gate.step(np.ones(4), learn=True)
    weights = gate.hebbian_weights
    assert np.any(gate.pfc_trace)
    gate.reset()
    np.testing.assert_array_equal(gate.pfc_trace, np.zeros(4))
    np.testing.assert_allclose(gate.hebbian_weights, weights)


def test_unbiased_wta_separates_two_pfc_context_patterns() -> None:
    gate = MDGate(4, n_md=4, learning_rate=0.1, tau_trace=1.0, seed=10)
    # Deterministic competing prototypes; no md_bias or oracle label is used.
    gate.pfc_to_md[:] = np.eye(4)
    first_pattern = np.array([1.0, 0.0, 0.0, 0.0])
    second_pattern = np.array([0.0, 1.0, 0.0, 0.0])

    gate.reset()
    first = gate.step(first_pattern, learn=True)
    gate.reset()
    second = gate.step(second_pattern, learn=True)
    assert first.winner == 0
    assert second.winner == 1
    assert first.winner != second.winner


@pytest.mark.parametrize("n_md", [3, 9])
def test_md_population_is_restricted_to_four_through_eight(n_md: int) -> None:
    with pytest.raises(ValueError, match="n_md"):
        MDGate(10, n_md=n_md)


def test_md_input_validation() -> None:
    with pytest.raises(ValueError):
        MDGate(0)
    with pytest.raises(ValueError):
        MDGate(4, n_md=4, gain_weights=np.zeros((4, 3)))
    gate = MDGate(4, n_md=4)
    trace_before = gate.pfc_trace
    with pytest.raises(ValueError):
        gate.step(np.ones(3))
    with pytest.raises(ValueError):
        gate.step(np.ones(4), md_bias=np.ones(3))
    np.testing.assert_array_equal(gate.pfc_trace, trace_before)
    with pytest.raises(ValueError):
        gate.step(np.ones(4), learn="yes")
    with pytest.raises(ValueError, match="scalar"):
        gate.step(np.ones(4), modulatory_signal=np.ones(2))
