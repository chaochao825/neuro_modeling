"""Explicit local three-factor plasticity with auditable update stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def _vector(value: Array, *, name: str, length: int | None = None) -> Array:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or array.size == 0
        or (length is not None and array.shape != (length,))
    ):
        expected = "one-dimensional" if length is None else f"shape ({length},)"
        raise ValueError(f"{name} must be non-empty with {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matrix(value: Array, *, name: str, shape: tuple[int, int]) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def project_dale_weights(weights: Array, presynaptic_signs: Array, mask: Array) -> Array:
    """Return sparse weights projected onto outgoing-column Dale signs."""

    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("weights must be two-dimensional")
    signs = _vector(presynaptic_signs, name="presynaptic_signs", length=matrix.shape[1])
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("presynaptic_signs must contain only -1 and +1")
    mask_array = np.asarray(mask)
    if mask_array.shape != matrix.shape:
        raise ValueError("mask must match weights")
    if not np.all(np.isin(mask_array, (False, True, 0, 1))):
        raise ValueError("mask must be boolean or binary")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights must contain only finite values")

    projected = np.where(mask_array.astype(bool), matrix, 0.0)
    excitatory = signs > 0
    projected[:, excitatory] = np.maximum(projected[:, excitatory], 0.0)
    projected[:, ~excitatory] = np.minimum(projected[:, ~excitatory], 0.0)
    return projected


def _project_dale_weights_trusted(
    weights: Array,
    presynaptic_signs: Array,
    mask: Array,
) -> Array:
    """Project a validated matrix without repeating public input checks."""

    projected = np.where(mask, weights, 0.0)
    excitatory = presynaptic_signs > 0.0
    projected[:, excitatory] = np.maximum(projected[:, excitatory], 0.0)
    projected[:, ~excitatory] = np.minimum(projected[:, ~excitatory], 0.0)
    return projected


@dataclass(frozen=True)
class PlasticityCosts:
    """Per-step path-length proxies for every transformation stage."""

    hebbian_l1: float
    decay_l1: float
    raw_l1: float
    masked_l1: float
    applied_l1: float
    applied_l2: float


@dataclass(frozen=True)
class ThreeFactorUpdate:
    """All stages of one local update, retained for rank/cost analysis."""

    eligibility_trace: Array
    post_factor: Array
    hebbian_update: Array
    decay_update: Array
    raw_update: Array
    masked_update: Array
    dale_applied_update: Array
    costs: PlasticityCosts


class ThreeFactorRule:
    """Stateful presynaptic eligibility and local three-factor outer product.

    The eligibility ODE ``tau * dp/dt = -p + r_pre`` is integrated exactly
    under a piecewise-constant presynaptic rate.  The weight proposal is
    ``learning_rate * dt * outer(modulator * derivative, eligibility)``.
    No trajectory, loss, or non-local gradient is retained.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        tau_eligibility: float,
        dt: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative and finite")
        if not np.isfinite(tau_eligibility) or tau_eligibility <= 0.0:
            raise ValueError("tau_eligibility must be positive and finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite")
        self.learning_rate = float(learning_rate)
        self.tau_eligibility = float(tau_eligibility)
        self.dt = float(dt)
        self.weight_decay = float(weight_decay)
        self._eligibility: Array | None = None

    @property
    def eligibility_trace(self) -> Array | None:
        return None if self._eligibility is None else self._eligibility.copy()

    def reset(self, n_pre: int | None = None) -> None:
        """Clear the trace, optionally allocating it for ``n_pre`` synapses."""

        if n_pre is None:
            self._eligibility = None
            return
        if not isinstance(n_pre, (int, np.integer)) or isinstance(n_pre, bool) or n_pre <= 0:
            raise ValueError("n_pre must be a positive integer")
        self._eligibility = np.zeros(int(n_pre), dtype=float)

    def update_eligibility(self, pre_activity: Array) -> Array:
        """Update and return a copy of the presynaptic eligibility trace."""

        pre = _vector(pre_activity, name="pre_activity")
        if self._eligibility is None:
            self._eligibility = np.zeros_like(pre)
        elif self._eligibility.shape != pre.shape:
            raise ValueError("pre_activity shape changed without resetting the rule")
        return self._update_eligibility_trusted(pre)

    def _update_eligibility_trusted(self, pre_activity: Array) -> Array:
        """Advance an eligibility trace whose shape and values are trusted."""

        if self._eligibility is None:
            self._eligibility = np.zeros_like(pre_activity)
        retention = np.exp(-self.dt / self.tau_eligibility)
        self._eligibility = (
            retention * self._eligibility + (1.0 - retention) * pre_activity
        )
        return self._eligibility.copy()

    def _propose_trusted(
        self,
        pre_activity: Array,
        modulatory_signal: Array,
        *,
        post_derivative: Array | None = None,
        connectivity_mask: Array | None = None,
        presynaptic_signs: Array | None = None,
        current_weights: Array | None = None,
        current_task_weights: Array | None = None,
    ) -> ThreeFactorUpdate:
        """Form an update from arrays whose invariants were checked by the caller.

        This private fast path intentionally omits shape, finite-value, binary-mask,
        and Dale validation.  In particular, ``current_weights`` is assumed to
        already obey ``connectivity_mask`` and ``presynaptic_signs``.
        """

        trace = self._update_eligibility_trusted(pre_activity)
        derivative = (
            np.ones_like(modulatory_signal)
            if post_derivative is None
            else post_derivative
        )
        post_factor = modulatory_signal * derivative
        shape = (post_factor.size, trace.size)

        hebbian = self.learning_rate * self.dt * np.outer(post_factor, trace)
        if self.weight_decay:
            decay = (
                -self.learning_rate
                * self.dt
                * self.weight_decay
                * current_task_weights
            )
        else:
            decay = np.zeros(shape, dtype=float)
        raw = hebbian + decay

        mask = (
            np.ones(shape, dtype=bool)
            if connectivity_mask is None
            else connectivity_mask
        )
        masked = np.where(mask, raw, 0.0)

        if presynaptic_signs is None:
            applied = masked.copy()
        else:
            before = (
                np.zeros(shape, dtype=float)
                if current_weights is None
                else current_weights
            )
            candidate = before + masked
            projected = _project_dale_weights_trusted(
                candidate,
                presynaptic_signs,
                mask,
            )
            applied = projected - before

        costs = PlasticityCosts(
            hebbian_l1=float(np.sum(np.abs(hebbian))),
            decay_l1=float(np.sum(np.abs(decay))),
            raw_l1=float(np.sum(np.abs(raw))),
            masked_l1=float(np.sum(np.abs(masked))),
            applied_l1=float(np.sum(np.abs(applied))),
            applied_l2=float(np.linalg.norm(applied)),
        )
        return ThreeFactorUpdate(
            eligibility_trace=trace,
            post_factor=post_factor.copy(),
            hebbian_update=hebbian,
            decay_update=decay,
            raw_update=raw,
            masked_update=masked,
            dale_applied_update=applied,
            costs=costs,
        )

    def propose(
        self,
        pre_activity: Array,
        modulatory_signal: Array,
        *,
        post_derivative: Array | None = None,
        connectivity_mask: Array | None = None,
        presynaptic_signs: Array | None = None,
        current_weights: Array | None = None,
        current_task_weights: Array | None = None,
    ) -> ThreeFactorUpdate:
        """Update eligibility and form raw, masked, and Dale-applied updates.

        ``current_weights`` is the effective recurrent matrix before this
        proposal.  Supplying it permits both potentiation and depression while
        projecting the *candidate weights*, rather than incorrectly forcing
        every update itself to have a Dale sign.
        """

        # Restore the causal trace if any downstream validation fails: rejected
        # calls must not silently change future plasticity.
        trace_before = None if self._eligibility is None else self._eligibility.copy()
        try:
            pre = _vector(pre_activity, name="pre_activity")
            if self._eligibility is not None and self._eligibility.shape != pre.shape:
                raise ValueError("pre_activity shape changed without resetting the rule")
            post = _vector(modulatory_signal, name="modulatory_signal")
            if post_derivative is None:
                derivative = np.ones_like(post)
            else:
                derivative = _vector(post_derivative, name="post_derivative", length=post.size)
            shape = (post.size, pre.size)
            if self.weight_decay:
                if current_task_weights is None:
                    raise ValueError("current_task_weights is required when weight_decay > 0")
                task_weights = _matrix(
                    current_task_weights, name="current_task_weights", shape=shape
                )
            else:
                if current_task_weights is not None:
                    task_weights = _matrix(
                        current_task_weights,
                        name="current_task_weights",
                        shape=shape,
                    )
                else:
                    task_weights = None

            if connectivity_mask is None:
                mask = np.ones(shape, dtype=bool)
            else:
                raw_mask = np.asarray(connectivity_mask)
                if raw_mask.shape != shape or not np.all(
                    np.isin(raw_mask, (False, True, 0, 1))
                ):
                    raise ValueError(
                        "connectivity_mask must be a binary matrix matching the update"
                    )
                mask = raw_mask.astype(bool)
            if presynaptic_signs is None:
                if current_weights is not None:
                    before = _matrix(current_weights, name="current_weights", shape=shape)
                else:
                    before = None
                signs = None
            else:
                signs = _vector(
                    presynaptic_signs, name="presynaptic_signs", length=pre.size
                )
                if not np.all(np.isin(signs, (-1.0, 1.0))):
                    raise ValueError("presynaptic_signs must contain only -1 and +1")
                before = (
                    np.zeros(shape, dtype=float)
                    if current_weights is None
                    else _matrix(current_weights, name="current_weights", shape=shape)
                )
                already_projected = project_dale_weights(before, signs, mask)
                if not np.allclose(before, already_projected, atol=1e-12, rtol=0.0):
                    raise ValueError(
                        "current_weights must already satisfy sparsity and Dale signs"
                    )
            return self._propose_trusted(
                pre,
                post,
                post_derivative=derivative,
                connectivity_mask=mask,
                presynaptic_signs=signs,
                current_weights=before,
                current_task_weights=task_weights,
            )
        except Exception:
            self._eligibility = trace_before
            raise

    step = propose


# Descriptive alias retained for callers that prefer the longer class name.
ThreeFactorPlasticity = ThreeFactorRule
