"""Reachability-weighted demand metrics for actuator matching.

The raw Frobenius norms of task perturbations are coordinate dependent and
count state directions that a fixed carrier can neither reach nor observe.
This module instead measures state-transition and input demands through the
controllability and observability Gramians of a stable baseline system.

All quantities are computed from a registered task generator.  They are not
fitted to held-out behavior and therefore can be used as prospective
predictors in the Exp26 actuator phase diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov


FloatArray = NDArray[np.float64]


def _finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric matrix")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _finite_vector(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    vector = np.asarray(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer_or_none(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_system(
    transition: ArrayLike,
    input_matrix: ArrayLike,
    observation_matrix: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    transition_matrix = _finite_matrix(transition, name="transition")
    input_weights = _finite_matrix(input_matrix, name="input_matrix")
    observation = _finite_matrix(observation_matrix, name="observation_matrix")
    state_dim = transition_matrix.shape[0]
    if transition_matrix.shape != (state_dim, state_dim):
        raise ValueError("transition must be square")
    if input_weights.shape[0] != state_dim:
        raise ValueError("input_matrix must have one row per state dimension")
    if observation.shape[1] != state_dim:
        raise ValueError(
            "observation_matrix must have one column per state dimension"
        )
    radius = float(np.max(np.abs(np.linalg.eigvals(transition_matrix))))
    if radius >= 1.0 - 1e-12:
        raise ValueError("transition must be strictly stable")
    return transition_matrix, input_weights, observation


def _symmetrize_psd(value: ArrayLike, *, name: str) -> FloatArray:
    matrix = _finite_matrix(value, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = np.finfo(np.float64).eps * matrix.shape[0] * scale * 1_000.0
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    clipped = np.clip(eigenvalues, 0.0, None)
    return _readonly((eigenvectors * clipped[np.newaxis, :]) @ eigenvectors.T)


def _psd_matrix_or_identity(
    value: ArrayLike | None,
    *,
    dimension: int,
    name: str,
) -> FloatArray:
    if value is None:
        return _readonly(np.eye(dimension, dtype=np.float64))
    matrix = _finite_matrix(value, name=name)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape [{dimension}, {dimension}]")
    symmetry_scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12 * symmetry_scale):
        raise ValueError(f"{name} must be symmetric")
    return _symmetrize_psd(matrix, name=name)


def control_gramians(
    transition: ArrayLike,
    input_matrix: ArrayLike,
    observation_matrix: ArrayLike,
    *,
    input_second_moment: ArrayLike | None = None,
    output_weight: ArrayLike | None = None,
    horizon: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return controllability and observability Gramians.

    ``horizon=None`` uses the infinite-horizon discrete Lyapunov equations.
    A positive finite horizon uses explicit sums from lag zero through
    ``horizon - 1``.  The latter is a finite-horizon Gramian proxy; combining
    its two Gramians is not claimed to equal the exact energy of an entire
    trial with parameter perturbations at every step.  Strict stability is
    required in both cases so cells cannot silently rely on explosive carriers.
    """

    transition_matrix, input_weights, observation = _validate_system(
        transition, input_matrix, observation_matrix
    )
    resolved_horizon = _positive_integer_or_none(horizon, name="horizon")
    input_moment = _psd_matrix_or_identity(
        input_second_moment,
        dimension=input_weights.shape[1],
        name="input_second_moment",
    )
    output_metric = _psd_matrix_or_identity(
        output_weight,
        dimension=observation.shape[0],
        name="output_weight",
    )
    drive_covariance = input_weights @ input_moment @ input_weights.T
    output_covariance = observation.T @ output_metric @ observation
    if resolved_horizon is None:
        controllability = solve_discrete_lyapunov(
            transition_matrix, drive_covariance
        )
        observability = solve_discrete_lyapunov(
            transition_matrix.T, output_covariance
        )
    else:
        state_dim = transition_matrix.shape[0]
        controllability = np.zeros((state_dim, state_dim), dtype=np.float64)
        observability = np.zeros_like(controllability)
        forward = np.eye(state_dim, dtype=np.float64)
        for _ in range(resolved_horizon):
            controllability += forward @ drive_covariance @ forward.T
            observability += forward.T @ output_covariance @ forward
            forward = transition_matrix @ forward
    return (
        _symmetrize_psd(controllability, name="controllability_gramian"),
        _symmetrize_psd(observability, name="observability_gramian"),
    )


def _psd_sqrt_and_projector(
    value: ArrayLike,
    *,
    name: str,
    support_rtol: float,
) -> tuple[FloatArray, FloatArray, int]:
    matrix = _symmetrize_psd(value, name=name)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    maximum = float(np.max(eigenvalues))
    threshold = support_rtol * maximum if maximum > 0.0 else np.inf
    support = eigenvalues > threshold
    root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    if np.any(support):
        basis = eigenvectors[:, support]
        projector = basis @ basis.T
    else:
        projector = np.zeros_like(matrix)
    return _readonly(root), _readonly(projector), int(np.sum(support))


def projected_transition_rank(
    transition_demand: ArrayLike,
    controllability_gramian: ArrayLike,
    observability_gramian: ArrayLike,
    *,
    support_rtol: float = 1e-10,
    rank_rtol: float = 1e-10,
) -> int:
    """Return ``rank(P_o delta_A P_c)`` on reachable/observable supports.

    This is a necessary recurrent-actuator rank for exactly realizing the
    registered transition perturbation on zero-input intervals.  It is a
    lower bound, not a sufficient-performance claim.
    """

    support_rtol = _nonnegative_scalar(support_rtol, name="support_rtol")
    rank_rtol = _nonnegative_scalar(rank_rtol, name="rank_rtol")
    delta = _finite_matrix(transition_demand, name="transition_demand")
    controllability = _finite_matrix(
        controllability_gramian, name="controllability_gramian"
    )
    observability = _finite_matrix(
        observability_gramian, name="observability_gramian"
    )
    state_dim = delta.shape[0]
    if delta.shape != (state_dim, state_dim):
        raise ValueError("transition_demand must be square")
    if controllability.shape != delta.shape or observability.shape != delta.shape:
        raise ValueError("both Gramians must match transition_demand")
    _, controllable_projector, _ = _psd_sqrt_and_projector(
        controllability,
        name="controllability_gramian",
        support_rtol=support_rtol,
    )
    _, observable_projector, _ = _psd_sqrt_and_projector(
        observability,
        name="observability_gramian",
        support_rtol=support_rtol,
    )
    projected = observable_projector @ delta @ controllable_projector
    singular_values = np.linalg.svd(projected, compute_uv=False)
    if singular_values.size == 0 or float(singular_values[0]) == 0.0:
        return 0
    return int(np.sum(singular_values > rank_rtol * singular_values[0]))


@dataclass(frozen=True)
class ActuatorDemandReceipt:
    """Immutable receipt for a prospective actuator-demand calculation."""

    state_demand: float
    input_demand: float
    state_fraction: float
    state_energy_fraction: float
    projected_transition_rank: int
    controllable_dimension: int
    observable_dimension: int
    horizon: int | None
    controllability_gramian: FloatArray
    observability_gramian: FloatArray


@dataclass(frozen=True)
class FiniteHorizonDemandReceipt:
    """Event-local demand audit using train-fitted second moments."""

    state_demand: float
    input_demand: float
    state_fraction: float
    state_energy_fraction: float
    cross_energy: float
    cross_relative_magnitude: float
    marginal_decomposition_valid: bool
    horizon: int
    step_weights: FloatArray


@dataclass(frozen=True)
class TransitionRankReceipt:
    """Necessary-rank and irreducible weighted-tail audit."""

    raw_rank: int
    projected_rank: int
    weighted_singular_values: FloatArray
    energy_rank_99: int
    energy_rank_999: int
    candidate_ranks: tuple[int, ...]
    tail_energy_fractions: tuple[float, ...]


def _energy_rank(singular_values: FloatArray, threshold: float) -> int:
    energy = singular_values * singular_values
    total = float(np.sum(energy))
    if total == 0.0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def transition_rank_requirement(
    transition_demand: ArrayLike,
    controllability_gramian: ArrayLike,
    observability_gramian: ArrayLike,
    *,
    candidate_ranks: tuple[int, ...] = (1, 2, 4, 8),
    support_rtol: float = 1e-10,
    rank_rtol: float = 1e-10,
) -> TransitionRankReceipt:
    """Audit necessary rank and Eckart--Young weighted residual bounds."""

    delta = _finite_matrix(transition_demand, name="transition_demand")
    if delta.shape[0] != delta.shape[1]:
        raise ValueError("transition_demand must be square")
    controllability = _finite_matrix(
        controllability_gramian, name="controllability_gramian"
    )
    observability = _finite_matrix(
        observability_gramian, name="observability_gramian"
    )
    if controllability.shape != delta.shape or observability.shape != delta.shape:
        raise ValueError("both Gramians must match transition_demand")
    if not isinstance(candidate_ranks, tuple) or not candidate_ranks:
        raise TypeError("candidate_ranks must be a non-empty tuple")
    resolved_ranks: list[int] = []
    for value in candidate_ranks:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("candidate_ranks must contain integers")
        rank = int(value)
        if rank < 0:
            raise ValueError("candidate_ranks must be non-negative")
        resolved_ranks.append(rank)
    if resolved_ranks != sorted(set(resolved_ranks)):
        raise ValueError("candidate_ranks must be strictly increasing")
    support_rtol = _nonnegative_scalar(support_rtol, name="support_rtol")
    rank_rtol = _nonnegative_scalar(rank_rtol, name="rank_rtol")
    controllability_root, _, _ = _psd_sqrt_and_projector(
        controllability,
        name="controllability_gramian",
        support_rtol=support_rtol,
    )
    observability_root, _, _ = _psd_sqrt_and_projector(
        observability,
        name="observability_gramian",
        support_rtol=support_rtol,
    )
    weighted = observability_root @ delta @ controllability_root
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    total_energy = float(np.sum(singular_values * singular_values))
    tails = []
    for rank in resolved_ranks:
        tail = float(np.sum(singular_values[rank:] ** 2))
        tails.append(tail / total_energy if total_energy > 0.0 else 0.0)
    raw_singular = np.linalg.svd(delta, compute_uv=False)
    raw_rank = (
        0
        if float(raw_singular[0]) == 0.0
        else int(np.sum(raw_singular > rank_rtol * raw_singular[0]))
    )
    return TransitionRankReceipt(
        raw_rank=raw_rank,
        projected_rank=projected_transition_rank(
            delta,
            controllability,
            observability,
            support_rtol=support_rtol,
            rank_rtol=rank_rtol,
        ),
        weighted_singular_values=_readonly(singular_values),
        energy_rank_99=_energy_rank(singular_values, 0.99),
        energy_rank_999=_energy_rank(singular_values, 0.999),
        candidate_ranks=tuple(resolved_ranks),
        tail_energy_fractions=tuple(tails),
    )


def state_input_cross_energy(
    transition_demand: ArrayLike,
    input_demand: ArrayLike,
    observability_gramian: ArrayLike,
    state_input_cross_moment: ArrayLike,
) -> float:
    """Return the A/B cross term omitted by the marginal demand index.

    The value is ``2 tr(W_o delta_A S_xu delta_B.T)``.  A material value
    means the marginal ``state_fraction`` is descriptive rather than a full
    energy decomposition.
    """

    delta_a = _finite_matrix(transition_demand, name="transition_demand")
    delta_b = _finite_matrix(input_demand, name="input_demand")
    observability = _finite_matrix(
        observability_gramian, name="observability_gramian"
    )
    cross = _finite_matrix(
        state_input_cross_moment, name="state_input_cross_moment"
    )
    state_dim = delta_a.shape[0]
    if delta_a.shape != (state_dim, state_dim):
        raise ValueError("transition_demand must be square")
    if delta_b.shape[0] != state_dim:
        raise ValueError("input_demand must have one row per state dimension")
    if observability.shape != delta_a.shape:
        raise ValueError("observability_gramian must match transition_demand")
    if cross.shape != (state_dim, delta_b.shape[1]):
        raise ValueError("state_input_cross_moment has incompatible shape")
    return float(2.0 * np.trace(observability @ delta_a @ cross @ delta_b.T))


def _finite_moment_sequence(
    value: ArrayLike,
    *,
    name: str,
    horizon: int | None = None,
    rows: int | None = None,
    columns: int | None = None,
    symmetric_psd: bool,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    moments = np.asarray(raw, dtype=np.float64)
    if moments.ndim != 3 or moments.size == 0 or 0 in moments.shape:
        raise ValueError(f"{name} must have shape [time, row, column]")
    if not np.all(np.isfinite(moments)):
        raise ValueError(f"{name} must contain only finite values")
    if horizon is not None and moments.shape[0] != horizon:
        raise ValueError(f"{name} must have the registered horizon")
    if rows is not None and moments.shape[1] != rows:
        raise ValueError(f"{name} has the wrong row dimension")
    if columns is not None and moments.shape[2] != columns:
        raise ValueError(f"{name} has the wrong column dimension")
    if symmetric_psd:
        if moments.shape[1] != moments.shape[2]:
            raise ValueError(f"{name} matrices must be square")
        checked = np.empty_like(moments)
        for index, matrix in enumerate(moments):
            checked[index] = _symmetrize_psd(matrix, name=f"{name}[{index}]")
        return _readonly(checked)
    return _readonly(moments)


def finite_horizon_local_demand(
    transition: ArrayLike,
    observation_matrix: ArrayLike,
    transition_demand: ArrayLike,
    input_demand: ArrayLike,
    state_second_moments: ArrayLike,
    input_second_moments: ArrayLike,
    *,
    state_input_cross_moments: ArrayLike | None = None,
    output_weight: ArrayLike | None = None,
    step_weights: ArrayLike | None = None,
    cross_relative_tolerance: float = 0.05,
) -> FiniteHorizonDemandReceipt:
    """Compute finite-horizon local-injection demand from training moments.

    At each time ``t``, the transition and input corrections are treated as a
    local current injection and weighted by the remaining-horizon
    observability Gramian.  This avoids claiming that the product of two
    finite Gramians is the exact energy of a parameter change applied at every
    step.  Cross moments are audited explicitly; a material cross term marks
    the marginal ``chi`` decomposition as incomplete.
    """

    transition_matrix = _finite_matrix(transition, name="transition")
    observation = _finite_matrix(observation_matrix, name="observation_matrix")
    delta_a = _finite_matrix(transition_demand, name="transition_demand")
    delta_b = _finite_matrix(input_demand, name="input_demand")
    state_dim = transition_matrix.shape[0]
    if transition_matrix.shape != (state_dim, state_dim):
        raise ValueError("transition must be square")
    if observation.shape[1] != state_dim:
        raise ValueError("observation_matrix has the wrong state dimension")
    if delta_a.shape != transition_matrix.shape:
        raise ValueError("transition_demand must match transition")
    if delta_b.shape[0] != state_dim:
        raise ValueError("input_demand has the wrong state dimension")
    radius = float(np.max(np.abs(np.linalg.eigvals(transition_matrix))))
    if radius >= 1.0 - 1e-12:
        raise ValueError("transition must be strictly stable")
    state_moments = _finite_moment_sequence(
        state_second_moments,
        name="state_second_moments",
        rows=state_dim,
        columns=state_dim,
        symmetric_psd=True,
    )
    horizon = state_moments.shape[0]
    input_moments = _finite_moment_sequence(
        input_second_moments,
        name="input_second_moments",
        horizon=horizon,
        rows=delta_b.shape[1],
        columns=delta_b.shape[1],
        symmetric_psd=True,
    )
    if state_input_cross_moments is None:
        cross_moments = np.zeros(
            (horizon, state_dim, delta_b.shape[1]), dtype=np.float64
        )
    else:
        cross_moments = _finite_moment_sequence(
            state_input_cross_moments,
            name="state_input_cross_moments",
            horizon=horizon,
            rows=state_dim,
            columns=delta_b.shape[1],
            symmetric_psd=False,
        )
    output_metric = _psd_matrix_or_identity(
        output_weight,
        dimension=observation.shape[0],
        name="output_weight",
    )
    if step_weights is None:
        weights = np.full(horizon, 1.0 / horizon, dtype=np.float64)
    else:
        weights = _finite_vector(step_weights, name="step_weights")
        if weights.shape != (horizon,):
            raise ValueError("step_weights must match the registered horizon")
        if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("step_weights must be non-negative with positive sum")
        weights = weights / np.sum(weights)
    cross_tolerance = _nonnegative_scalar(
        cross_relative_tolerance, name="cross_relative_tolerance"
    )
    output_covariance = observation.T @ output_metric @ observation
    remaining: list[FloatArray] = [_readonly(np.zeros_like(transition_matrix))]
    for _ in range(horizon):
        remaining.append(
            _symmetrize_psd(
                output_covariance
                + transition_matrix.T @ remaining[-1] @ transition_matrix,
                name="remaining_observability",
            )
        )
    state_energy = 0.0
    input_energy = 0.0
    cross_energy = 0.0
    for time_index in range(horizon):
        future = remaining[horizon - time_index]
        weight = float(weights[time_index])
        state_energy += weight * float(
            np.trace(
                future
                @ delta_a
                @ state_moments[time_index]
                @ delta_a.T
            )
        )
        input_energy += weight * float(
            np.trace(
                future
                @ delta_b
                @ input_moments[time_index]
                @ delta_b.T
            )
        )
        cross_energy += weight * state_input_cross_energy(
            delta_a,
            delta_b,
            future,
            cross_moments[time_index],
        )
    scale = max(1.0, abs(state_energy), abs(input_energy))
    tolerance = np.finfo(np.float64).eps * state_dim * scale * 1_000.0
    if state_energy < -tolerance or input_energy < -tolerance:
        raise RuntimeError("local demand energy is numerically negative")
    state_energy = max(state_energy, 0.0)
    input_energy = max(input_energy, 0.0)
    state_value = float(np.sqrt(state_energy))
    input_value = float(np.sqrt(input_energy))
    denominator = state_value + input_value
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("transition and input demands are jointly zero")
    marginal_energy = state_energy + input_energy
    cross_relative = abs(cross_energy) / max(
        marginal_energy, np.finfo(np.float64).tiny
    )
    return FiniteHorizonDemandReceipt(
        state_demand=state_value,
        input_demand=input_value,
        state_fraction=float(state_value / denominator),
        state_energy_fraction=float(state_energy / marginal_energy),
        cross_energy=float(cross_energy),
        cross_relative_magnitude=float(cross_relative),
        marginal_decomposition_valid=bool(cross_relative <= cross_tolerance),
        horizon=horizon,
        step_weights=_readonly(weights),
    )


def actuator_demand_from_gramians(
    transition_demand: ArrayLike,
    input_demand: ArrayLike,
    controllability_gramian: ArrayLike,
    observability_gramian: ArrayLike,
    *,
    input_second_moment: ArrayLike | None = None,
    horizon: int | None = None,
    support_rtol: float = 1e-10,
    rank_rtol: float = 1e-10,
) -> ActuatorDemandReceipt:
    """Compute task demand from frozen, precomputed reference Gramians."""

    support_rtol = _nonnegative_scalar(support_rtol, name="support_rtol")
    rank_rtol = _nonnegative_scalar(rank_rtol, name="rank_rtol")
    delta_a = _finite_matrix(transition_demand, name="transition_demand")
    delta_b = _finite_matrix(input_demand, name="input_demand")
    state_dim = delta_a.shape[0]
    if delta_a.shape != (state_dim, state_dim):
        raise ValueError("transition_demand must be square")
    if delta_b.shape[0] != state_dim:
        raise ValueError("input_demand must have one row per state dimension")
    controllability = _finite_matrix(
        controllability_gramian, name="controllability_gramian"
    )
    observability = _finite_matrix(
        observability_gramian, name="observability_gramian"
    )
    if controllability.shape != delta_a.shape or observability.shape != delta_a.shape:
        raise ValueError("both Gramians must match transition_demand")
    input_moment = _psd_matrix_or_identity(
        input_second_moment,
        dimension=delta_b.shape[1],
        name="input_second_moment",
    )
    controllability_root, _, controllable_dimension = _psd_sqrt_and_projector(
        controllability,
        name="controllability_gramian",
        support_rtol=support_rtol,
    )
    observability_root, _, observable_dimension = _psd_sqrt_and_projector(
        observability,
        name="observability_gramian",
        support_rtol=support_rtol,
    )
    input_root, _, _ = _psd_sqrt_and_projector(
        input_moment,
        name="input_second_moment",
        support_rtol=support_rtol,
    )
    state_value = float(
        np.linalg.norm(
            observability_root @ delta_a @ controllability_root,
            ord="fro",
        )
    )
    input_value = float(
        np.linalg.norm(observability_root @ delta_b @ input_root, ord="fro")
    )
    denominator = state_value + input_value
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("transition and input demands are jointly zero")
    return ActuatorDemandReceipt(
        state_demand=state_value,
        input_demand=input_value,
        state_fraction=float(state_value / denominator),
        state_energy_fraction=float(
            state_value * state_value
            / (state_value * state_value + input_value * input_value)
        ),
        projected_transition_rank=projected_transition_rank(
            delta_a,
            controllability,
            observability,
            support_rtol=support_rtol,
            rank_rtol=rank_rtol,
        ),
        controllable_dimension=controllable_dimension,
        observable_dimension=observable_dimension,
        horizon=_positive_integer_or_none(horizon, name="horizon"),
        controllability_gramian=_readonly(controllability),
        observability_gramian=_readonly(observability),
    )


def actuator_demand_index(
    transition: ArrayLike,
    input_matrix: ArrayLike,
    observation_matrix: ArrayLike,
    transition_demand: ArrayLike,
    input_demand: ArrayLike,
    *,
    input_second_moment: ArrayLike | None = None,
    output_weight: ArrayLike | None = None,
    horizon: int | None = None,
    support_rtol: float = 1e-10,
    rank_rtol: float = 1e-10,
) -> ActuatorDemandReceipt:
    """Compute the prospective task--actuator matching index ``chi``.

    The registered demands are

    ``D_A = ||W_o^(1/2) delta_A W_c^(1/2)||_F`` and
    ``D_B = ||W_o^(1/2) delta_B Sigma_u^(1/2)||_F``.

    ``state_fraction = D_A / (D_A + D_B)`` is the phase-diagram coordinate:
    values near zero predict input routing/gain demand, while values near one
    predict recurrent-operator demand.  A zero/zero task is rejected rather
    than assigned an arbitrary phase.
    """

    support_rtol = _nonnegative_scalar(support_rtol, name="support_rtol")
    rank_rtol = _nonnegative_scalar(rank_rtol, name="rank_rtol")
    transition_matrix, input_weights, observation = _validate_system(
        transition, input_matrix, observation_matrix
    )
    delta_a = _finite_matrix(transition_demand, name="transition_demand")
    delta_b = _finite_matrix(input_demand, name="input_demand")
    state_dim = transition_matrix.shape[0]
    if delta_a.shape != (state_dim, state_dim):
        raise ValueError("transition_demand must match transition")
    if delta_b.shape != input_weights.shape:
        raise ValueError("input_demand must match input_matrix")
    input_moment = _psd_matrix_or_identity(
        input_second_moment,
        dimension=input_weights.shape[1],
        name="input_second_moment",
    )
    controllability, observability = control_gramians(
        transition_matrix,
        input_weights,
        observation,
        input_second_moment=input_moment,
        output_weight=output_weight,
        horizon=horizon,
    )
    return actuator_demand_from_gramians(
        delta_a,
        delta_b,
        controllability,
        observability,
        input_second_moment=input_moment,
        horizon=horizon,
        support_rtol=support_rtol,
        rank_rtol=rank_rtol,
    )


__all__ = [
    "ActuatorDemandReceipt",
    "TransitionRankReceipt",
    "actuator_demand_index",
    "actuator_demand_from_gramians",
    "control_gramians",
    "finite_horizon_local_demand",
    "projected_transition_rank",
    "state_input_cross_energy",
    "transition_rank_requirement",
]
