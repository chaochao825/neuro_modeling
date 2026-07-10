"""Local inhibitory homeostasis for Dale-constrained E/I networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HomeostaticCosts:
    """Per-step plasticity path lengths."""

    raw_l1: float
    masked_l1: float
    applied_l1: float
    applied_l2: float


@dataclass(frozen=True)
class HomeostaticUpdate:
    """Raw, sparse, and Dale-applied I-to-E updates."""

    post_error: Array
    raw_update: Array
    masked_update: Array
    dale_applied_update: Array
    costs: HomeostaticCosts


def _population_mask(value: Array, *, name: str, n_units: int) -> Array:
    mask = np.asarray(value)
    if mask.shape != (n_units,) or not np.all(np.isin(mask, (False, True, 0, 1))):
        raise ValueError(f"{name} must be a binary vector with shape ({n_units},)")
    return mask.astype(bool)


class InhibitoryHomeostasis:
    """Vogels-style local homeostasis restricted to I-to-E synapses.

    For postsynaptic excitatory unit ``i`` and presynaptic inhibitory unit
    ``j``, the raw update is

    ``-learning_rate * dt * (rate_i - target_rate) * rate_j``.

    Hence activity above target makes an inhibitory weight more negative;
    activity below target weakens it toward zero.  The update is finally
    projected against the current inhibitory weights so it cannot cross zero.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        target_rate: float,
        dt: float = 1.0,
        max_abs_update: float | None = None,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative and finite")
        if not np.isfinite(target_rate) or target_rate < 0.0:
            raise ValueError("target_rate must be non-negative and finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if max_abs_update is not None and (
            not np.isfinite(max_abs_update) or max_abs_update <= 0.0
        ):
            raise ValueError("max_abs_update must be positive and finite")
        self.learning_rate = float(learning_rate)
        self.target_rate = float(target_rate)
        self.dt = float(dt)
        self.max_abs_update = None if max_abs_update is None else float(max_abs_update)

    def _propose_trusted(
        self,
        rates: Array,
        *,
        excitatory_mask: Array,
        inhibitory_mask: Array,
        current_weights: Array,
        connectivity_mask: Array | None = None,
    ) -> HomeostaticUpdate:
        """Return an update from arrays whose invariants were checked upstream.

        This private fast path deliberately skips shape, finite-value,
        population-partition, binary-mask, sparsity, and Dale validation.
        """

        n_units = rates.size
        shape = (n_units, n_units)
        sparse_mask = (
            np.ones(shape, dtype=bool)
            if connectivity_mask is None
            else connectivity_mask
        )
        i_to_e = excitatory_mask[:, np.newaxis] & inhibitory_mask[np.newaxis, :]
        applicable_mask = sparse_mask & i_to_e

        post_error = rates - self.target_rate
        raw = np.zeros(shape, dtype=float)
        raw[np.ix_(excitatory_mask, inhibitory_mask)] = (
            -self.learning_rate
            * self.dt
            * np.outer(post_error[excitatory_mask], rates[inhibitory_mask])
        )
        if self.max_abs_update is not None:
            raw = np.clip(raw, -self.max_abs_update, self.max_abs_update)
        masked = np.where(applicable_mask, raw, 0.0)

        candidate = current_weights + masked
        candidate[:, inhibitory_mask] = np.minimum(
            candidate[:, inhibitory_mask],
            0.0,
        )
        candidate = np.where(sparse_mask, candidate, 0.0)
        applied = candidate - current_weights

        costs = HomeostaticCosts(
            raw_l1=float(np.sum(np.abs(raw))),
            masked_l1=float(np.sum(np.abs(masked))),
            applied_l1=float(np.sum(np.abs(applied))),
            applied_l2=float(np.linalg.norm(applied)),
        )
        return HomeostaticUpdate(
            post_error=post_error,
            raw_update=raw,
            masked_update=masked,
            dale_applied_update=applied,
            costs=costs,
        )

    def propose(
        self,
        rates: Array,
        *,
        excitatory_mask: Array,
        inhibitory_mask: Array,
        current_weights: Array,
        connectivity_mask: Array | None = None,
    ) -> HomeostaticUpdate:
        """Return a local update without mutating a network."""

        rate_array = np.asarray(rates, dtype=float)
        if rate_array.ndim != 1 or rate_array.size < 2:
            raise ValueError("rates must be a one-dimensional population vector")
        if not np.all(np.isfinite(rate_array)) or np.any(rate_array < 0.0):
            raise ValueError("rates must be finite and non-negative")
        n_units = rate_array.size
        excitatory = _population_mask(
            excitatory_mask, name="excitatory_mask", n_units=n_units
        )
        inhibitory = _population_mask(
            inhibitory_mask, name="inhibitory_mask", n_units=n_units
        )
        if np.any(excitatory & inhibitory) or np.any(~(excitatory | inhibitory)):
            raise ValueError("excitatory_mask and inhibitory_mask must partition all units")
        if not np.any(excitatory) or not np.any(inhibitory):
            raise ValueError("both excitatory and inhibitory populations are required")

        shape = (n_units, n_units)
        if connectivity_mask is None:
            sparse_mask = np.ones(shape, dtype=bool)
        else:
            raw_mask = np.asarray(connectivity_mask)
            if raw_mask.shape != shape or not np.all(np.isin(raw_mask, (False, True, 0, 1))):
                raise ValueError("connectivity_mask must be a matching binary matrix")
            sparse_mask = raw_mask.astype(bool)
        before = np.asarray(current_weights, dtype=float)
        if before.shape != shape or not np.all(np.isfinite(before)):
            raise ValueError("current_weights must be a finite matching square matrix")
        if np.any(before[:, excitatory] < -1e-12):
            raise ValueError("current excitatory outgoing weights must be non-negative")
        if np.any(before[:, inhibitory] > 1e-12):
            raise ValueError("current inhibitory outgoing weights must be non-positive")
        if np.any(np.abs(before[~sparse_mask]) > 1e-12):
            raise ValueError("current_weights must respect connectivity_mask")
        return self._propose_trusted(
            rate_array,
            excitatory_mask=excitatory,
            inhibitory_mask=inhibitory,
            current_weights=before,
            connectivity_mask=sparse_mask,
        )

    step = propose


InhibitoryHomeostasisRule = InhibitoryHomeostasis
