"""Local three-factor learning utilities for a scalar-belief gain axis.

The learned parameter is a vector ``a`` in

``gain_i(p) = 1 + a_i * (2 * p - 1)``.

This module is deliberately NumPy-only.  A neuron-local eligibility trace is
combined with a scalar task error and a scalar belief third factor.  It does
not retain trajectories, construct a loss graph, or propagate credit backward
through time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.plasticity.update_budget import UpdateBudgetController


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_vector(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_scalar(value: object, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_scalar(value: object, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _readonly_float(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_int(value: ArrayLike) -> IntArray:
    result = np.array(value, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(label: str, *values: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GainAxisUpdateCosts:
    """L1/L2 path-length proxies for one proposed vector update."""

    raw_l1: float
    raw_l2: float


@dataclass(frozen=True, slots=True)
class GainAxisUpdate:
    """One local eligibility-times-scalar-third-factor proposal."""

    eligibility_trace: FloatArray
    task_error: float
    clipped_task_error: float
    third_factor: float
    scalar_modulator: float
    raw_update: FloatArray
    costs: GainAxisUpdateCosts


class GainAxisThreeFactorRule:
    """Causal neuron-local eligibility followed by a scalar third factor.

    The exact trace update for a piecewise-constant local drive is

    ``e <- exp(-dt/tau) * e + (1-exp(-dt/tau)) * local_drive``.

    Once the trial-local trace is formed, :meth:`propose` returns

    ``eta * dt * e * clipped_task_error * third_factor``.

    ``third_factor`` is restricted to ``[-1, 1]`` so it can directly represent
    ``2 * p(z=1) - 1`` or an oracle ``{-1, +1}`` upper-bound signal.  The class
    has no hidden-state, loss-graph, or trajectory interface.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        tau_eligibility: float,
        dt: float = 1.0,
        error_clip: float | None = None,
    ) -> None:
        self.learning_rate = _nonnegative_scalar(learning_rate, name="learning_rate")
        self.tau_eligibility = _positive_scalar(tau_eligibility, name="tau_eligibility")
        self.dt = _positive_scalar(dt, name="dt")
        self.error_clip = (
            None
            if error_clip is None
            else _positive_scalar(error_clip, name="error_clip")
        )
        self._eligibility: FloatArray | None = None

    @property
    def eligibility_trace(self) -> FloatArray | None:
        """Return a copy so external callers cannot mutate the causal state."""

        return None if self._eligibility is None else self._eligibility.copy()

    def reset(self, n_units: int | None = None) -> None:
        """Clear the trace, optionally allocating a zero trace for ``n_units``."""

        if n_units is None:
            self._eligibility = None
            return
        self._eligibility = np.zeros(
            _positive_integer(n_units, name="n_units"), dtype=np.float64
        )

    def update_eligibility(self, local_drive: ArrayLike) -> FloatArray:
        """Advance the local trace and return an independent copy."""

        drive = _finite_vector(local_drive, name="local_drive")
        if self._eligibility is None:
            self._eligibility = np.zeros_like(drive)
        elif self._eligibility.shape != drive.shape:
            raise ValueError("local_drive shape changed without resetting the rule")
        retention = float(np.exp(-self.dt / self.tau_eligibility))
        self._eligibility = retention * self._eligibility + (1.0 - retention) * drive
        return self._eligibility.copy()

    def propose(
        self,
        *,
        task_error: float,
        third_factor: float,
    ) -> GainAxisUpdate:
        """Form an update from the already accumulated eligibility trace."""

        if self._eligibility is None:
            raise RuntimeError("eligibility trace must be initialized before propose")
        error = _finite_scalar(task_error, name="task_error")
        third = _finite_scalar(third_factor, name="third_factor")
        if not -1.0 <= third <= 1.0:
            raise ValueError("third_factor must lie in [-1, 1]")
        clipped_error = (
            error
            if self.error_clip is None
            else float(np.clip(error, -self.error_clip, self.error_clip))
        )
        modulator = clipped_error * third
        raw = self.learning_rate * self.dt * modulator * self._eligibility
        costs = GainAxisUpdateCosts(
            raw_l1=float(np.sum(np.abs(raw))),
            raw_l2=float(np.linalg.norm(raw)),
        )
        return GainAxisUpdate(
            eligibility_trace=_readonly_float(self._eligibility),
            task_error=error,
            clipped_task_error=clipped_error,
            third_factor=third,
            scalar_modulator=float(modulator),
            raw_update=_readonly_float(raw),
            costs=costs,
        )

    def step(
        self,
        local_drive: ArrayLike,
        *,
        task_error: float,
        third_factor: float,
    ) -> GainAxisUpdate:
        """Advance eligibility once and immediately form a proposal."""

        trace_before = None if self._eligibility is None else self._eligibility.copy()
        try:
            self.update_eligibility(local_drive)
            return self.propose(
                task_error=task_error,
                third_factor=third_factor,
            )
        except Exception:
            self._eligibility = trace_before
            raise


@dataclass(frozen=True, slots=True)
class GainAxisBudgetApplication:
    """One vector proposal after selected-norm budget scaling."""

    raw_update: FloatArray
    applied_update: FloatArray
    selected_norm: str
    scale_factor: float
    raw_l1: float
    raw_l2: float
    applied_l1: float
    applied_l2: float
    processed_events: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class GainAxisPathBudgetApplication:
    """One globally rescaled proposal tape with relative events preserved."""

    raw_updates: FloatArray
    applied_updates: FloatArray
    selected_norm: str
    total_budget: float
    scale_factor: float
    raw_path_l1: float
    raw_path_l2: float
    applied_path_l1: float
    applied_path_l2: float
    selected_applied: float
    attained: bool
    final_shortfall: float
    raw_nonzero_events: int
    applied_nonzero_events: int
    zero_proposal_events: int
    fingerprint: str


def apply_gain_axis_budget(
    controller: UpdateBudgetController,
    proposed_update: ArrayLike,
) -> GainAxisBudgetApplication:
    """Adapt a vector proposal to the existing two-dimensional controller.

    The vector is represented as an ``N x 1`` matrix only at the controller
    boundary.  Matrix L1 and Frobenius L2 then equal the vector L1 and L2
    exactly.  The controller is never allowed to amplify the proposal.
    """

    if not isinstance(controller, UpdateBudgetController):
        raise TypeError("controller must be an UpdateBudgetController")
    raw = _finite_vector(proposed_update, name="proposed_update")
    raw_copy = raw.copy()
    applied_matrix = controller.scale(raw_copy[:, None])
    if not np.array_equal(raw, raw_copy):
        raise RuntimeError("budget controller mutated the proposed vector")
    if applied_matrix.shape != (raw.size, 1):
        raise RuntimeError("budget controller returned an unexpected shape")
    applied = np.asarray(applied_matrix[:, 0], dtype=np.float64)
    raw_l1 = float(np.sum(np.abs(raw)))
    raw_l2 = float(np.linalg.norm(raw))
    applied_l1 = float(np.sum(np.abs(applied)))
    applied_l2 = float(np.linalg.norm(applied))
    denominator = raw_l1 if controller.norm == "l1" else raw_l2
    numerator = applied_l1 if controller.norm == "l1" else applied_l2
    scale = 0.0 if denominator == 0.0 else numerator / denominator
    if scale > 1.0 + 1e-12:
        raise RuntimeError("budget controller amplified a gain-axis proposal")
    return GainAxisBudgetApplication(
        raw_update=_readonly_float(raw),
        applied_update=_readonly_float(applied),
        selected_norm=controller.norm,
        scale_factor=float(min(scale, 1.0)),
        raw_l1=raw_l1,
        raw_l2=raw_l2,
        applied_l1=applied_l1,
        applied_l2=applied_l2,
        processed_events=controller.processed_events,
        fingerprint=_fingerprint(
            "gain-axis-budget-application-v1",
            controller.norm,
            controller.total_budget,
            controller.planned_events,
            controller.processed_events,
            raw,
            applied,
        ),
    )


def apply_gain_axis_path_budget(
    proposed_updates: ArrayLike,
    *,
    total_budget: float,
    norm: str,
    tolerance: float = 1e-9,
) -> GainAxisPathBudgetApplication:
    """Downscale one full proposal tape with a single global factor.

    Unlike equal-per-event online allocation, this audit helper preserves every
    event's relative magnitude and direction.  It never amplifies a proposal:
    if the raw selected-norm path is shorter than ``total_budget``, the returned
    record is explicitly unattained.
    """

    raw_value = np.asarray(proposed_updates)
    if raw_value.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("proposed_updates must be real numeric")
    raw = np.asarray(raw_value, dtype=np.float64)
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError("proposed_updates must be a non-empty 2D array")
    if not np.all(np.isfinite(raw)):
        raise ValueError("proposed_updates must contain only finite values")
    budget = _nonnegative_scalar(total_budget, name="total_budget")
    tol = _nonnegative_scalar(tolerance, name="tolerance")
    if not isinstance(norm, str):
        raise TypeError("norm must be a string")
    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'")

    event_l1 = np.sum(np.abs(raw), axis=1)
    event_l2 = np.linalg.norm(raw, axis=1)
    raw_path_l1 = float(np.sum(event_l1))
    raw_path_l2 = float(np.sum(event_l2))
    selected_raw = raw_path_l1 if norm == "l1" else raw_path_l2
    if selected_raw == 0.0:
        scale = 1.0 if budget == 0.0 else 0.0
    else:
        scale = min(1.0, budget / selected_raw)
    applied = raw * scale
    applied_event_l1 = np.sum(np.abs(applied), axis=1)
    applied_event_l2 = np.linalg.norm(applied, axis=1)
    applied_path_l1 = float(np.sum(applied_event_l1))
    applied_path_l2 = float(np.sum(applied_event_l2))
    selected_applied = applied_path_l1 if norm == "l1" else applied_path_l2
    if selected_applied > budget and selected_applied > 0.0:
        applied *= budget / selected_applied
        applied_event_l1 = np.sum(np.abs(applied), axis=1)
        applied_event_l2 = np.linalg.norm(applied, axis=1)
        applied_path_l1 = float(np.sum(applied_event_l1))
        applied_path_l2 = float(np.sum(applied_event_l2))
        selected_applied = applied_path_l1 if norm == "l1" else applied_path_l2
        scale = selected_applied / selected_raw
    shortfall = max(0.0, budget - selected_applied)
    attained = shortfall <= tol
    raw_nonzero = event_l1 > 0.0
    applied_nonzero = applied_event_l1 > 0.0
    frozen_raw = np.array(raw, dtype=np.float64, copy=True)
    frozen_raw.setflags(write=False)
    frozen_applied = np.array(applied, dtype=np.float64, copy=True)
    frozen_applied.setflags(write=False)
    return GainAxisPathBudgetApplication(
        raw_updates=frozen_raw,
        applied_updates=frozen_applied,
        selected_norm=norm,
        total_budget=budget,
        scale_factor=float(scale),
        raw_path_l1=raw_path_l1,
        raw_path_l2=raw_path_l2,
        applied_path_l1=applied_path_l1,
        applied_path_l2=applied_path_l2,
        selected_applied=selected_applied,
        attained=attained,
        final_shortfall=shortfall,
        raw_nonzero_events=int(np.count_nonzero(raw_nonzero)),
        applied_nonzero_events=int(np.count_nonzero(applied_nonzero)),
        zero_proposal_events=int(np.count_nonzero(~raw_nonzero)),
        fingerprint=_fingerprint(
            "gain-axis-path-budget-application-v1",
            norm,
            budget,
            tol,
            raw,
            applied,
            float(scale),
        ),
    )


def _validated_group_labels(group_labels: ArrayLike | None, *, length: int) -> IntArray:
    if group_labels is None:
        return np.zeros(length, dtype=np.int64)
    raw = np.asarray(group_labels)
    if raw.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("group_labels must be a boolean or integer vector")
    if raw.shape != (length,):
        raise ValueError(f"group_labels must have shape ({length},)")
    _, inverse = np.unique(raw, return_inverse=True)
    return np.asarray(inverse, dtype=np.int64)


def _validated_permutation(value: ArrayLike, *, length: int) -> IntArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u"} or raw.shape != (length,):
        raise ValueError(f"permutation must be an integer vector of shape ({length},)")
    permutation = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(np.sort(permutation), np.arange(length)):
        raise ValueError("permutation must contain every index exactly once")
    return permutation


@dataclass(frozen=True, slots=True)
class SignedPermutationTransform:
    """Fixed norm-preserving signed permutation of gain-axis coordinates."""

    permutation: IntArray
    signs: FloatArray
    group_labels: IntArray
    seed: int
    deranged: bool
    fingerprint: str

    def __post_init__(self) -> None:
        permutation = _validated_permutation(
            self.permutation, length=np.asarray(self.permutation).size
        )
        signs = _finite_vector(self.signs, name="signs")
        if signs.shape != permutation.shape or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("signs must contain one -1/+1 value per coordinate")
        labels = _validated_group_labels(self.group_labels, length=permutation.size)
        if not np.array_equal(labels[permutation], labels):
            raise ValueError("permutation must remain within every group")
        if not isinstance(self.deranged, (bool, np.bool_)):
            raise TypeError("deranged must be boolean")
        seed = _nonnegative_integer(self.seed, name="seed")
        if bool(self.deranged) and np.any(permutation == np.arange(permutation.size)):
            raise ValueError("deranged transform cannot contain fixed points")
        expected = _fingerprint(
            "gain-axis-signed-permutation-v1",
            permutation,
            signs,
            labels,
            seed,
            bool(self.deranged),
        )
        if self.fingerprint != expected:
            raise ValueError("transform fingerprint does not match its contents")
        object.__setattr__(self, "permutation", _readonly_int(permutation))
        object.__setattr__(self, "signs", _readonly_float(signs))
        object.__setattr__(self, "group_labels", _readonly_int(labels))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "deranged", bool(self.deranged))

    def apply(self, vector: ArrayLike) -> FloatArray:
        """Apply the transform without modifying its input."""

        values = _finite_vector(vector, name="vector")
        if values.shape != self.permutation.shape:
            raise ValueError(f"vector must have shape ({self.permutation.size},)")
        transformed = self.signs * values[self.permutation]
        if not np.isclose(
            np.sum(np.abs(transformed)),
            np.sum(np.abs(values)),
            atol=1e-12,
            rtol=1e-12,
        ) or not np.isclose(
            np.linalg.norm(transformed),
            np.linalg.norm(values),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError("signed permutation failed to preserve vector norms")
        return _readonly_float(transformed)


def make_signed_permutation(
    n_units: int,
    *,
    seed: int,
    group_labels: ArrayLike | None = None,
    deranged: bool = False,
    sign_flips: bool = True,
) -> SignedPermutationTransform:
    """Create a deterministic group-preserving signed permutation."""

    length = _positive_integer(n_units, name="n_units")
    transform_seed = _nonnegative_integer(seed, name="seed")
    if not isinstance(deranged, (bool, np.bool_)):
        raise TypeError("deranged must be boolean")
    if not isinstance(sign_flips, (bool, np.bool_)):
        raise TypeError("sign_flips must be boolean")
    labels = _validated_group_labels(group_labels, length=length)
    rng = np.random.default_rng(transform_seed)
    permutation = np.arange(length, dtype=np.int64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if bool(deranged) and indices.size < 2:
            raise ValueError("every deranged group must contain at least two units")
        if not bool(deranged):
            permutation[indices] = rng.permutation(indices)
            continue
        candidate = indices.copy()
        for _ in range(100):
            candidate = rng.permutation(indices)
            if not np.any(candidate == indices):
                break
        else:
            candidate = np.roll(indices, 1)
        permutation[indices] = candidate
    signs = (
        rng.choice(np.array([-1.0, 1.0]), size=length)
        if bool(sign_flips)
        else np.ones(length, dtype=np.float64)
    )
    fingerprint = _fingerprint(
        "gain-axis-signed-permutation-v1",
        permutation,
        signs,
        labels,
        transform_seed,
        bool(deranged),
    )
    return SignedPermutationTransform(
        permutation=permutation,
        signs=signs,
        group_labels=labels,
        seed=transform_seed,
        deranged=bool(deranged),
        fingerprint=fingerprint,
    )


def make_deranged_shuffle(
    n_units: int,
    *,
    seed: int,
    group_labels: ArrayLike | None = None,
) -> SignedPermutationTransform:
    """Create a sign-preserving, fixed-point-free coordinate shuffle."""

    return make_signed_permutation(
        n_units,
        seed=seed,
        group_labels=group_labels,
        deranged=True,
        sign_flips=False,
    )


def orthogonal_component(
    vector: ArrayLike,
    reference: ArrayLike,
    *,
    atol: float = 1e-12,
) -> FloatArray:
    """Project ``vector`` into the orthogonal complement of ``reference``.

    The projection occurs before budget matching because it generally changes
    both L1 and L2 proposal norms.
    """

    values = _finite_vector(vector, name="vector")
    anchor = _finite_vector(reference, name="reference")
    if values.shape != anchor.shape:
        raise ValueError("vector and reference must have the same shape")
    tolerance = _nonnegative_scalar(atol, name="atol")
    norm_squared = float(anchor @ anchor)
    if norm_squared <= tolerance * tolerance:
        raise ValueError("reference must have non-zero norm")
    projected = values - anchor * float((anchor @ values) / norm_squared)
    residual = float(abs(anchor @ projected))
    bound = max(
        tolerance,
        tolerance * float(np.linalg.norm(anchor)) * float(np.linalg.norm(projected)),
    )
    if residual > bound:
        raise RuntimeError("orthogonal projection failed its numerical audit")
    return _readonly_float(projected)


__all__ = [
    "GainAxisBudgetApplication",
    "GainAxisPathBudgetApplication",
    "GainAxisThreeFactorRule",
    "GainAxisUpdate",
    "GainAxisUpdateCosts",
    "SignedPermutationTransform",
    "apply_gain_axis_budget",
    "apply_gain_axis_path_budget",
    "make_deranged_shuffle",
    "make_signed_permutation",
    "orthogonal_component",
]
