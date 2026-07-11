"""Transparent end-to-end compute proxies for belief-gated receivers.

The quantities here are unitless operation/event accounting terms.  They are
never interpreted as ATP or physical energy.  Unlike a posterior-amplitude
penalty, the gate term charges the declared inference algorithm independently
of whether its posterior is informative or neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


@dataclass(frozen=True)
class EndToEndComputeSummary:
    """Per-trial event components under one explicit counting convention."""

    input_weighted_events: float
    recurrent_weighted_events: float
    receiver_firing_magnitude: float
    readout_weighted_events: float
    gate_primitive_operations: float
    gate_state_updates: float
    total_compute_proxy: float
    n_trials: int
    interpretation: str


def end_to_end_compute_proxy(
    *,
    n_trials: int,
    input_event_sum: float,
    recurrent_event_sum: float,
    firing_sum: float,
    readout_features: ArrayLike,
    readout_weights: ArrayLike,
    gate_operations_per_trial: float,
    gate_state_updates_per_trial: float,
) -> EndToEndComputeSummary:
    """Combine receiver, readout, and declared gate event counts.

    Receiver firing magnitude is reported but excluded from the total because
    it is already the source of recurrent weighted-event accounting.  This
    avoids double charging the same population activity.
    """

    if (
        isinstance(n_trials, (bool, np.bool_))
        or not isinstance(n_trials, (int, np.integer))
        or int(n_trials) < 1
    ):
        raise ValueError("n_trials must be a positive integer")
    count = int(n_trials)
    input_total = _nonnegative_scalar(input_event_sum, name="input_event_sum")
    recurrent_total = _nonnegative_scalar(
        recurrent_event_sum, name="recurrent_event_sum"
    )
    firing_total = _nonnegative_scalar(firing_sum, name="firing_sum")
    gate_ops = _nonnegative_scalar(
        gate_operations_per_trial, name="gate_operations_per_trial"
    )
    gate_states = _nonnegative_scalar(
        gate_state_updates_per_trial, name="gate_state_updates_per_trial"
    )

    raw_features = np.asarray(readout_features)
    raw_weights = np.asarray(readout_weights)
    if raw_features.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("readout_features must be real numeric")
    if raw_weights.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("readout_weights must be real numeric")
    features = np.asarray(raw_features, dtype=np.float64)
    weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    if (
        features.ndim != 2
        or features.shape != (count, weights.size)
        or weights.size == 0
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError(
            "readout_features and readout_weights must have finite matching "
            "shapes [n_trials, feature] and [feature]"
        )
    readout_events = float(np.mean(np.abs(features) @ np.abs(weights)))
    per_trial_input = input_total / count
    per_trial_recurrent = recurrent_total / count
    per_trial_firing = firing_total / count
    total = (
        per_trial_input + per_trial_recurrent + readout_events + gate_ops + gate_states
    )
    return EndToEndComputeSummary(
        input_weighted_events=per_trial_input,
        recurrent_weighted_events=per_trial_recurrent,
        receiver_firing_magnitude=per_trial_firing,
        readout_weighted_events=readout_events,
        gate_primitive_operations=gate_ops,
        gate_state_updates=gate_states,
        total_compute_proxy=total,
        n_trials=count,
        interpretation="unitless_event_and_declared_primitive_count_not_atp",
    )


__all__ = ["EndToEndComputeSummary", "end_to_end_compute_proxy"]
