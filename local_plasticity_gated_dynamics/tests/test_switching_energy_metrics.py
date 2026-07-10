import numpy as np
import pytest

from src.analysis.energy_metrics import (
    activity_energy_proxy,
    energy_proxy_summary,
    firing_rate_energy_proxy,
    plasticity_cost,
    plasticity_update_energy_proxy,
    recurrent_current_energy_proxy,
    synaptic_event_energy_proxy,
    synaptic_energy_proxy,
)
from src.analysis.switching_metrics import (
    forgetting,
    forgetting_summary,
    jacobian_spectrum_summary,
    switch_cost,
    switch_cost_summary,
)


def test_switch_cost_uses_uncontaminated_context_windows() -> None:
    context = np.array([0] * 4 + [1] * 4 + [0] * 4)
    performance = np.array(
        [1.0, 1.0, 1.0, 1.0, 0.4, 0.6, 0.8, 0.8, 0.5, 0.5, 0.9, 0.9]
    )
    summary = switch_cost_summary(performance, context, pre_window=2, post_window=2)

    assert np.array_equal(summary.switch_indices, [4, 8])
    assert np.allclose(summary.per_switch_cost, [0.5, 0.3])
    assert switch_cost(performance, context, pre_window=2, post_window=2) == pytest.approx(
        0.4
    )


def test_forgetting_sign_always_means_deterioration() -> None:
    summary = forgetting_summary([0.9, 0.8], [0.7, 0.9])
    assert np.allclose(summary.per_unit_forgetting, [0.2, -0.1])
    assert forgetting([0.9, 0.8], [0.7, 0.9]) == pytest.approx(0.05)
    assert forgetting([0.1, 0.2], [0.4, 0.1], higher_is_better=False) == pytest.approx(
        0.1
    )


def test_jacobian_summary_distinguishes_time_conventions() -> None:
    continuous = jacobian_spectrum_summary(np.diag([-1.0, 0.2]), dynamics="continuous")
    assert continuous.max_real_part == pytest.approx(0.2)
    assert continuous.unstable_count == 1
    assert continuous.stability_margin == pytest.approx(-0.2)

    discrete = jacobian_spectrum_summary(np.diag([0.5, 1.2]), dynamics="discrete")
    assert discrete.spectral_radius == pytest.approx(1.2)
    assert discrete.unstable_count == 1
    assert discrete.stability_margin == pytest.approx(-0.2)


def test_three_energy_proxies_have_explicit_normalization() -> None:
    activity = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.eye(2)
    updates = np.array([[1.0, -1.0], [0.0, 2.0]])

    assert activity_energy_proxy(activity) == pytest.approx(7.5)
    assert activity_energy_proxy(activity, normalize_neurons=False) == pytest.approx(15.0)
    assert firing_rate_energy_proxy(activity) == pytest.approx(2.5)
    assert recurrent_current_energy_proxy(activity, weights) == pytest.approx(7.5)
    assert synaptic_energy_proxy(activity, weights) == pytest.approx(2.5)
    assert plasticity_cost(updates) == pytest.approx(4.0)
    assert plasticity_update_energy_proxy(updates) == pytest.approx(4.0)
    assert plasticity_cost(updates, normalize=True) == pytest.approx(1.0)
    summary = energy_proxy_summary(activity, weights, updates)
    assert summary.firing_rate == pytest.approx(2.5)
    assert summary.synaptic_event == pytest.approx(2.5)
    assert summary.plasticity_update == pytest.approx(1.0)


def test_synaptic_events_do_not_cancel_between_excitation_and_inhibition() -> None:
    activity = np.array([[1.0, 1.0]])
    weights = np.array([[1.0, -1.0]])

    assert recurrent_current_energy_proxy(activity, weights) == pytest.approx(0.0)
    assert synaptic_event_energy_proxy(
        activity, weights, normalize_connections=False
    ) == pytest.approx(2.0)
    assert synaptic_event_energy_proxy(activity, weights) == pytest.approx(1.0)


def test_switching_and_energy_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="no switches"):
        switch_cost([1.0, 1.0], [0, 0], pre_window=1, post_window=1)
    with pytest.raises(ValueError, match="square"):
        jacobian_spectrum_summary(np.ones((2, 3)))
    with pytest.raises(ValueError, match="match activity"):
        synaptic_energy_proxy(np.ones((3, 2)), np.eye(3))
    with pytest.raises(ValueError, match="norm"):
        plasticity_cost(np.eye(2), norm="bad")
