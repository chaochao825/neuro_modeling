"""Transparent, unitless energy proxies for rate-network comparisons.

These metrics are not claims about biological ATP consumption.  The requested
three proxies are absolute firing rate, absolute weighted synaptic events, and
weight-update magnitude.  Squared activity and squared net recurrent current
are retained as separately named diagnostics; importantly, the latter is not
used as the synaptic-event proxy because excitation/inhibition can cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _finite_real_array(values: ArrayLike, *, name: str, min_ndim: int) -> FloatArray:
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim < min_ndim or array.size == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty with at least {min_ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_normalize(normalize_neurons: bool) -> bool:
    if not isinstance(normalize_neurons, (bool, np.bool_)):
        raise TypeError("normalize_neurons must be boolean")
    return bool(normalize_neurons)


def activity_energy_proxy(
    activity: ArrayLike, *, normalize_neurons: bool = True
) -> float:
    """Return time/trial-averaged squared activity.

    ``activity`` is shaped ``(..., neurons)``.  With the default normalization
    the result is mean squared activity per neuron, allowing comparisons across
    network sizes.  Without normalization it is mean total squared activity
    per sample.
    """

    rates = _finite_real_array(activity, name="activity", min_ndim=2)
    normalize = _validate_normalize(normalize_neurons)
    per_sample = np.sum(rates * rates, axis=-1)
    result = float(np.mean(per_sample))
    return result / rates.shape[-1] if normalize else result


def firing_rate_energy_proxy(
    activity: ArrayLike, *, normalize_neurons: bool = True
) -> float:
    """Return mean absolute firing-rate magnitude.

    Negative rates from centered/tanh models are treated by magnitude.  The
    normalized result is the mean absolute rate per neuron; the unnormalized
    result is the mean total absolute rate per sample.
    """

    rates = _finite_real_array(activity, name="activity", min_ndim=2)
    normalize = _validate_normalize(normalize_neurons)
    per_sample = np.sum(np.abs(rates), axis=-1)
    result = float(np.mean(per_sample))
    return result / rates.shape[-1] if normalize else result


def recurrent_current_energy_proxy(
    activity: ArrayLike,
    weights: ArrayLike,
    *,
    normalize_neurons: bool = True,
) -> float:
    """Return mean squared *net* recurrent current ``activity @ weights.T``.

    ``weights`` is shaped ``(post_neurons, pre_neurons)`` and the final
    activity axis indexes pre-synaptic neurons.  This diagnostic permits E/I
    cancellation and therefore must not be interpreted as synaptic events.
    """

    rates = _finite_real_array(activity, name="activity", min_ndim=2)
    matrix = _finite_real_array(weights, name="weights", min_ndim=2)
    if matrix.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    if matrix.shape[1] != rates.shape[-1]:
        raise ValueError("weights pre-synaptic dimension must match activity")
    normalize = _validate_normalize(normalize_neurons)
    currents = rates @ matrix.T
    per_sample = np.sum(currents * currents, axis=-1)
    result = float(np.mean(per_sample))
    return result / matrix.shape[0] if normalize else result


def synaptic_event_energy_proxy(
    activity: ArrayLike,
    weights: ArrayLike,
    *,
    normalize_connections: bool = True,
) -> float:
    """Return ``mean_t sum_ij |W_ij| |r_j(t)|``.

    Excitatory and inhibitory transmissions are counted before summation, so
    opposite-signed currents cannot cancel.  With normalization, the total is
    divided by the number of nonzero connections, yielding mean weighted event
    magnitude per active connection.  A matrix with no connections has proxy
    zero.
    """

    rates = _finite_real_array(activity, name="activity", min_ndim=2)
    matrix = _finite_real_array(weights, name="weights", min_ndim=2)
    if matrix.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    if matrix.shape[1] != rates.shape[-1]:
        raise ValueError("weights pre-synaptic dimension must match activity")
    if not isinstance(normalize_connections, (bool, np.bool_)):
        raise TypeError("normalize_connections must be boolean")
    weighted_events = np.abs(rates) @ np.abs(matrix).T
    total_per_sample = np.sum(weighted_events, axis=-1)
    result = float(np.mean(total_per_sample))
    if not normalize_connections:
        return result
    connection_count = int(np.count_nonzero(matrix))
    return result / connection_count if connection_count else 0.0


def synaptic_energy_proxy(
    activity: ArrayLike,
    weights: ArrayLike,
    *,
    normalize_connections: bool = True,
) -> float:
    """Alias for :func:`synaptic_event_energy_proxy`."""

    return synaptic_event_energy_proxy(
        activity, weights, normalize_connections=normalize_connections
    )


def plasticity_cost(
    weight_updates: ArrayLike,
    *,
    norm: Literal["l1", "l2", "squared_l2"] = "l1",
    normalize: bool = False,
) -> float:
    """Return magnitude of all local weight updates.

    Leading dimensions may index steps/trials/seeds, while the final two axes
    index post- and pre-synaptic neurons.  When ``normalize`` is true, L1 and
    squared-L2 costs become elementwise means, and L2 becomes RMS magnitude.
    """

    updates = _finite_real_array(weight_updates, name="weight_updates", min_ndim=2)
    if norm not in {"l1", "l2", "squared_l2"}:
        raise ValueError("norm must be 'l1', 'l2', or 'squared_l2'")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    absolute = np.abs(updates)
    if norm == "l1":
        return float(np.mean(absolute) if normalize else np.sum(absolute))
    squared = updates * updates
    if norm == "squared_l2":
        return float(np.mean(squared) if normalize else np.sum(squared))
    return float(np.sqrt(np.mean(squared) if normalize else np.sum(squared)))


def plasticity_update_energy_proxy(
    weight_updates: ArrayLike,
    *,
    norm: Literal["l1", "l2", "squared_l2"] = "l1",
    normalize: bool = False,
) -> float:
    """Alias naming :func:`plasticity_cost` as the update-energy proxy."""

    return plasticity_cost(weight_updates, norm=norm, normalize=normalize)


@dataclass(frozen=True)
class EnergyProxySummary:
    """The three energy proxies evaluated with one normalization convention."""

    firing_rate: float
    synaptic_event: float
    plasticity_update: float
    normalized: bool
    plasticity_norm: str


def energy_proxy_summary(
    activity: ArrayLike,
    weights: ArrayLike,
    weight_updates: ArrayLike,
    *,
    normalize: bool = True,
    plasticity_norm: Literal["l1", "l2", "squared_l2"] = "l1",
) -> EnergyProxySummary:
    """Compute activity, recurrent-current, and update-cost proxies together."""

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    return EnergyProxySummary(
        firing_rate=firing_rate_energy_proxy(activity, normalize_neurons=normalize),
        synaptic_event=synaptic_event_energy_proxy(
            activity, weights, normalize_connections=normalize
        ),
        plasticity_update=plasticity_update_energy_proxy(
            weight_updates, norm=plasticity_norm, normalize=normalize
        ),
        normalized=bool(normalize),
        plasticity_norm=plasticity_norm,
    )
