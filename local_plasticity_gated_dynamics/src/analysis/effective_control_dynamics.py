"""Train-only coarse effective-dynamics audits for belief-gated E/I receivers.

The receiver bridge currently exposes one feature vector per trial containing
the concatenated sensory, delay, and response epoch-mean rates.  Consequently
the reduced model in this module has exactly three coarse states per trial and
two within-trial transitions.  It is an *epoch-transition surrogate*, not a
trajectory LDS, a Kalman likelihood, or evidence that the full continuous
network is globally confined to the fitted basis.

All data-dependent quantities (centering/scaling, PCA basis, affine operators,
and closure-error normalization) are fitted from ``train_features`` only.
Held-out features are used exclusively by :meth:`FittedSoftEpochDynamics.score`.
The local Jacobian diagnostics use a caller-supplied operating state and the
fixed network checkpoint; they do not estimate an operating point from held-out
features.  No randomness is used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from src.analysis.manifold_metrics import FittedPCASubspace, fit_train_pca
from src.analysis.rank_stage_metrics import matrix_rank_summary
from src.models.ei_rate_network import EIRateNetwork


FloatArray = NDArray[np.float64]


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_vector(value: ArrayLike, *, name: str, length: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric vector")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return array


def _epoch_states(
    features: ArrayLike, *, n_units: int, name: str, minimum_trials: int
) -> FloatArray:
    raw = np.asarray(features)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric matrix")
    array = np.asarray(raw, dtype=np.float64)
    expected_features = 3 * int(n_units)
    if (
        array.ndim != 2
        or array.shape[0] < int(minimum_trials)
        or array.shape[1] != expected_features
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{name} must have finite shape [trial>={minimum_trials}, "
            f"3 * n_units={expected_features}]"
        )
    # ReceiverSimulation concatenates sensory, delay, and response blocks in
    # this exact order.  Reshaping never joins transitions across trials.
    return array.reshape(array.shape[0], 3, int(n_units))


def _state1_probability(value: ArrayLike, *, n_trials: int, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must contain real probabilities")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape == (n_trials,):
        probability = array
    elif array.shape == (n_trials, 2):
        if not np.allclose(np.sum(array, axis=1), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError(f"{name} two-state rows must sum to one")
        probability = array[:, 1]
    else:
        raise ValueError(f"{name} must have shape [trial] or [trial, 2]")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.asarray(probability, dtype=np.float64)


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_scalar(value: object, *, name: str) -> float:
    result = _nonnegative_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _soft_operator_design(latent: FloatArray, probability: FloatArray) -> FloatArray:
    if latent.ndim != 2 or probability.shape != (latent.shape[0],):
        raise ValueError("latent states and probabilities must align")
    base = np.column_stack((latent, np.ones(latent.shape[0], dtype=np.float64)))
    return np.concatenate(
        ((1.0 - probability[:, None]) * base, probability[:, None] * base),
        axis=1,
    )


def _latent_epoch_transitions(
    pca: FittedPCASubspace, states: FloatArray, probability: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n_trials = states.shape[0]
    latent_dim = pca.components_.shape[0]
    latent = pca.transform(states.reshape(-1, states.shape[-1])).reshape(
        n_trials, 3, latent_dim
    )
    current = latent[:, :-1].reshape(-1, latent_dim)
    following = latent[:, 1:].reshape(-1, latent_dim)
    transition_probability = np.repeat(probability, 2)
    return current, following, transition_probability


@dataclass(frozen=True, slots=True)
class SoftEpochClosureScore:
    """Held-out score for the frozen train-fitted epoch-transition surrogate."""

    normalized_closure_mse: float
    raw_latent_mse: float
    heldout_basis_residual_fraction: float
    n_trials: int
    n_transitions: int
    interpretation: str = (
        "heldout_three_epoch_soft_operator_prediction_not_full_trajectory_closure"
    )


@dataclass(frozen=True, slots=True)
class FittedSoftEpochDynamics:
    """Train-fitted PCA plus two softly mixed affine latent operators.

    ``operator_state0`` and ``operator_state1`` use row-vector convention and
    have shape ``[latent_dim + 1, latent_dim]``.  For belief ``p`` the one-step
    prediction is

    ``(1-p) * [z, 1] @ operator_state0 + p * [z, 1] @ operator_state1``.
    """

    pca: FittedPCASubspace
    operator_state0: FloatArray
    operator_state1: FloatArray
    closure_variance: FloatArray
    n_units: int
    n_train_trials: int
    ridge: float
    operator_design_rank: int
    operator_design_columns: int
    preprocessing_fit_scope: str = "train_features_only"

    @property
    def latent_dim(self) -> int:
        return int(self.pca.components_.shape[0])

    @property
    def raw_activity_basis(self) -> FloatArray:
        """Return an orthonormal raw-rate basis implied by train PCA.

        When train-only per-neuron normalization was requested, PCA basis
        vectors live in standardized coordinates.  Multiplication by the
        frozen training scale maps their tangent directions back to raw-rate
        coordinates before orthonormalization.
        """

        raw_directions = self.pca.scale_[:, None] * self.pca.basis_
        basis, _ = np.linalg.qr(raw_directions, mode="reduced")
        return _readonly(basis)

    def score(
        self, features: ArrayLike, state1_probability: ArrayLike
    ) -> SoftEpochClosureScore:
        """Score held-out trials without updating any fitted quantity."""

        states = _epoch_states(
            features, n_units=self.n_units, name="features", minimum_trials=1
        )
        probability = _state1_probability(
            state1_probability,
            n_trials=states.shape[0],
            name="state1_probability",
        )
        current, following, transition_probability = _latent_epoch_transitions(
            self.pca, states, probability
        )
        design = _soft_operator_design(current, transition_probability)
        coefficients = np.concatenate(
            (self.operator_state0, self.operator_state1), axis=0
        )
        prediction = design @ coefficients
        error = prediction - following
        raw_mse = float(np.mean(error * error))
        normalized = float(np.mean((error * error) / self.closure_variance[None, :]))

        flat = states.reshape(-1, self.n_units)
        standardized = (flat - self.pca.mean_) / self.pca.scale_
        projected = (standardized @ self.pca.components_.T) @ self.pca.components_
        residual_energy = float(np.sum((standardized - projected) ** 2))
        total_energy = float(np.sum(standardized**2))
        residual_fraction = (
            residual_energy / total_energy if total_energy > 0.0 else 0.0
        )
        return SoftEpochClosureScore(
            normalized_closure_mse=normalized,
            raw_latent_mse=raw_mse,
            heldout_basis_residual_fraction=float(residual_fraction),
            n_trials=int(states.shape[0]),
            n_transitions=int(current.shape[0]),
        )


def fit_soft_epoch_dynamics(
    train_features: ArrayLike,
    train_state1_probability: ArrayLike,
    *,
    n_units: int,
    latent_dim: int,
    ridge: float = 1e-4,
    normalize_activity: bool = False,
) -> FittedSoftEpochDynamics:
    """Fit the three-epoch PCA and soft affine operators on training trials."""

    if (
        isinstance(n_units, (bool, np.bool_))
        or not isinstance(n_units, (int, np.integer))
        or int(n_units) < 2
    ):
        raise ValueError("n_units must be an integer >= 2")
    if (
        isinstance(latent_dim, (bool, np.bool_))
        or not isinstance(latent_dim, (int, np.integer))
        or not 1 <= int(latent_dim) < int(n_units)
    ):
        raise ValueError("latent_dim must be an integer in [1, n_units - 1]")
    ridge_value = _nonnegative_scalar(ridge, name="ridge")
    if not isinstance(normalize_activity, (bool, np.bool_)):
        raise TypeError("normalize_activity must be boolean")

    states = _epoch_states(
        train_features,
        n_units=int(n_units),
        name="train_features",
        minimum_trials=2,
    )
    probability = _state1_probability(
        train_state1_probability,
        n_trials=states.shape[0],
        name="train_state1_probability",
    )
    flat_train = states.reshape(-1, int(n_units))
    pca = fit_train_pca(
        flat_train,
        int(latent_dim),
        normalize=bool(normalize_activity),
    )
    current, following, transition_probability = _latent_epoch_transitions(
        pca, states, probability
    )
    design = _soft_operator_design(current, transition_probability)
    gram = design.T @ design
    penalty = ridge_value * np.eye(gram.shape[0], dtype=np.float64)
    block = int(latent_dim) + 1
    # The two affine intercepts are not regularized.
    penalty[block - 1, block - 1] = 0.0
    penalty[2 * block - 1, 2 * block - 1] = 0.0
    rhs = design.T @ following
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + penalty) @ rhs

    variance = np.var(following, axis=0, ddof=0)
    closure_variance = np.where(variance > np.finfo(np.float64).eps, variance, 1.0)
    return FittedSoftEpochDynamics(
        pca=pca,
        operator_state0=_readonly(coefficients[:block]),
        operator_state1=_readonly(coefficients[block:]),
        closure_variance=_readonly(closure_variance),
        n_units=int(n_units),
        n_train_trials=int(states.shape[0]),
        ridge=ridge_value,
        operator_design_rank=int(np.linalg.matrix_rank(design)),
        operator_design_columns=int(design.shape[1]),
    )


def _gain_vector(
    gain_axis: ArrayLike,
    *,
    n_units: int,
    gain_strength: float,
    belief: float,
) -> tuple[FloatArray, FloatArray]:
    axis = _finite_vector(gain_axis, name="gain_axis", length=n_units)
    if np.max(np.abs(axis)) > 1.0 + 1e-12:
        raise ValueError("gain_axis maximum absolute value must not exceed one")
    strength = _nonnegative_scalar(gain_strength, name="gain_strength")
    if strength >= 1.0:
        raise ValueError("gain_strength must lie in [0, 1)")
    probability = _nonnegative_scalar(belief, name="belief")
    if probability > 1.0:
        raise ValueError("belief must lie in [0, 1]")
    gain = 1.0 + strength * (2.0 * probability - 1.0) * axis
    if np.any(gain <= 0.0):
        raise RuntimeError("validated scalar belief produced non-positive gain")
    derivative = 2.0 * strength * axis
    return gain, derivative


def _local_rate_derivative(
    network: EIRateNetwork, state: FloatArray, gain: FloatArray
) -> tuple[FloatArray, FloatArray]:
    activated = np.tanh(gain * state)
    derivative = gain * (1.0 - activated * activated)
    if network.activation_name == "rectified_tanh":
        derivative = np.where(activated > 0.0, derivative, 0.0)
    return activated, derivative


@dataclass(frozen=True, slots=True)
class LocalRateBasisPullback:
    """Least-squares pullback of a rate tangent into preactivation coordinates."""

    preactivation_basis: FloatArray
    tangent_dimension: int
    relative_rate_reconstruction_residual: float
    derivative_active_count: int
    eligible: bool
    interpretation: str = (
        "local_pseudoinverse_pullback_dr_equals_activation_derivative_times_dx"
    )


def pullback_rate_tangent_basis(
    network: EIRateNetwork,
    mean_state: ArrayLike,
    gain_axis: ArrayLike,
    rate_basis: ArrayLike,
    *,
    gain_strength: float,
    belief: float,
    derivative_tolerance: float = 1e-8,
    residual_tolerance: float = 0.05,
) -> LocalRateBasisPullback:
    """Map train-fitted rate directions into the Jacobian's ``x`` coordinates.

    Locally, ``dr = diag(phi'(g*x)) dx``.  The minimum-norm preactivation
    direction for every rate-basis column is therefore obtained with the
    diagonal Moore--Penrose pseudoinverse.  A rectified or saturated unit can
    make part of the rate basis locally unrepresentable; the relative residual
    and explicit eligibility flag prevent that approximation from being
    silently reported as exact normal stability.
    """

    if not isinstance(network, EIRateNetwork):
        raise TypeError("network must be an EIRateNetwork")
    state = _finite_vector(mean_state, name="mean_state", length=network.n_units)
    raw_basis = np.asarray(rate_basis)
    if raw_basis.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("rate_basis must be a real numeric matrix")
    basis = np.asarray(raw_basis, dtype=np.float64)
    if (
        basis.ndim != 2
        or basis.shape[0] != network.n_units
        or not 1 <= basis.shape[1] < network.n_units
        or not np.all(np.isfinite(basis))
        or np.linalg.matrix_rank(basis) != basis.shape[1]
    ):
        raise ValueError(
            "rate_basis must contain independent columns and leave a normal space"
        )
    derivative_floor = _positive_scalar(
        derivative_tolerance, name="derivative_tolerance"
    )
    residual_limit = _nonnegative_scalar(residual_tolerance, name="residual_tolerance")
    gain, _ = _gain_vector(
        gain_axis,
        n_units=network.n_units,
        gain_strength=gain_strength,
        belief=belief,
    )
    _, rate_derivative = _local_rate_derivative(network, state, gain)
    active = np.abs(rate_derivative) > derivative_floor
    candidate = np.zeros_like(basis)
    candidate[active] = basis[active] / rate_derivative[active, None]
    reconstructed_rate = rate_derivative[:, None] * candidate
    denominator = float(np.linalg.norm(basis))
    residual = float(np.linalg.norm(reconstructed_rate - basis) / denominator)

    left, singular, _ = np.linalg.svd(candidate, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = max(
        np.finfo(np.float64).eps * max(candidate.shape) * largest,
        derivative_floor * np.finfo(np.float64).eps,
    )
    tangent_dimension = int(np.count_nonzero(singular > threshold))
    if tangent_dimension < 1:
        raise ValueError(
            "rate basis has no locally representable preactivation tangent direction"
        )
    preactivation_basis = left[:, :tangent_dimension]
    eligible = bool(tangent_dimension == basis.shape[1] and residual <= residual_limit)
    return LocalRateBasisPullback(
        preactivation_basis=_readonly(preactivation_basis),
        tangent_dimension=tangent_dimension,
        relative_rate_reconstruction_residual=residual,
        derivative_active_count=int(np.count_nonzero(active)),
        eligible=eligible,
    )


def ei_continuous_jacobian(
    network: EIRateNetwork,
    mean_state: ArrayLike,
    gain_axis: ArrayLike,
    *,
    gain_strength: float,
    belief: float,
) -> FloatArray:
    """Continuous-time state Jacobian at one supplied, fixed operating state."""

    if not isinstance(network, EIRateNetwork):
        raise TypeError("network must be an EIRateNetwork")
    state = _finite_vector(mean_state, name="mean_state", length=network.n_units)
    gain, _ = _gain_vector(
        gain_axis,
        n_units=network.n_units,
        gain_strength=gain_strength,
        belief=belief,
    )
    _, activation_derivative = _local_rate_derivative(network, state, gain)
    jacobian = -np.eye(network.n_units) + (
        network.recurrent_weights * activation_derivative[None, :]
    )
    jacobian = jacobian / network.time_constants[:, None]
    return _readonly(jacobian)


def ei_scalar_gain_jacobian_tangent(
    network: EIRateNetwork,
    mean_state: ArrayLike,
    gain_axis: ArrayLike,
    *,
    gain_strength: float,
    belief: float = 0.5,
) -> FloatArray:
    """Return analytic ``dJ/dp`` for the scalar belief-to-gain controller.

    For ``rectified_tanh`` the derivative is defined as zero at its kink.  The
    returned matrix can be high rank even though the controller has one scalar
    input; matrix rank and control-input dimension must therefore be reported
    separately.
    """

    if not isinstance(network, EIRateNetwork):
        raise TypeError("network must be an EIRateNetwork")
    state = _finite_vector(mean_state, name="mean_state", length=network.n_units)
    gain, gain_derivative = _gain_vector(
        gain_axis,
        n_units=network.n_units,
        gain_strength=gain_strength,
        belief=belief,
    )
    activated = np.tanh(gain * state)
    sech_squared = 1.0 - activated * activated
    derivative_wrt_gain = sech_squared * (1.0 - 2.0 * gain * state * activated)
    activation_derivative_tangent = gain_derivative * derivative_wrt_gain
    if network.activation_name == "rectified_tanh":
        activation_derivative_tangent = np.where(
            activated > 0.0, activation_derivative_tangent, 0.0
        )
    tangent = (
        network.recurrent_weights * activation_derivative_tangent[None, :]
    ) / network.time_constants[:, None]
    return _readonly(tangent)


@dataclass(frozen=True, slots=True)
class ProjectedNormalLinearSummary:
    """Local linear contraction after projection onto a basis complement."""

    normal_dimension: int
    max_real_part: float
    decay_ratio: float
    normal_to_tangent_coupling_norm: float
    horizon: float
    interpretation: str = (
        "spectral_norm_of_expm_of_projected_normal_jacobian_local_linear_only"
    )


def projected_normal_linear_summary(
    jacobian: ArrayLike,
    tangent_basis: ArrayLike,
    *,
    horizon: float,
) -> ProjectedNormalLinearSummary:
    """Summarize the locally projected normal linear dynamics.

    ``decay_ratio`` is ``||exp(h * Qn.T @ J @ Qn)||_2``.  It captures possible
    non-normal transient amplification inside the projected normal system.  It
    deliberately excludes trajectories that leave the normal space and later
    re-enter, so the coupling norm is reported separately.
    """

    raw_jacobian = np.asarray(jacobian)
    raw_basis = np.asarray(tangent_basis)
    if raw_jacobian.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("jacobian must be a real numeric matrix")
    if raw_basis.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("tangent_basis must be a real numeric matrix")
    matrix = np.asarray(raw_jacobian, dtype=np.float64)
    basis = np.asarray(raw_basis, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("jacobian must be a finite non-empty square matrix")
    if (
        basis.ndim != 2
        or basis.shape[0] != matrix.shape[0]
        or not 1 <= basis.shape[1] < matrix.shape[0]
        or not np.all(np.isfinite(basis))
        or np.linalg.matrix_rank(basis) != basis.shape[1]
    ):
        raise ValueError(
            "tangent_basis must contain independent columns and leave a normal space"
        )
    horizon_value = _positive_scalar(horizon, name="horizon")
    complete_basis, _ = np.linalg.qr(basis, mode="complete")
    tangent = complete_basis[:, : basis.shape[1]]
    normal = complete_basis[:, basis.shape[1] :]
    restricted = normal.T @ matrix @ normal
    eigenvalues = np.linalg.eigvals(restricted)
    max_real_part = float(np.max(np.real(eigenvalues)))
    propagator = expm(horizon_value * restricted)
    decay_ratio = float(np.linalg.svd(propagator, compute_uv=False)[0])
    coupling = float(np.linalg.norm(tangent.T @ matrix @ normal, ord=2))
    return ProjectedNormalLinearSummary(
        normal_dimension=int(normal.shape[1]),
        max_real_part=max_real_part,
        decay_ratio=decay_ratio,
        normal_to_tangent_coupling_norm=coupling,
        horizon=horizon_value,
    )


@dataclass(frozen=True, slots=True)
class EffectiveControlDynamicsAudit:
    """Scalar, serialization-friendly summary of the coarse E/I audit."""

    physical_raw_rank: int
    physical_effective_rank: float
    physical_rank_threshold: float
    basis_dimension: int
    basis_explained_variance_fraction: float
    heldout_normalized_closure_mse: float
    heldout_raw_latent_mse: float
    heldout_basis_residual_fraction: float
    declared_scalar_control_dimension: int
    operator_control_dimension: int
    operator_control_effective_dimension: float
    operator_delta_frobenius_norm: float
    operator_design_rank: int
    operator_design_columns: int
    jacobian_tangent_raw_rank: int
    jacobian_tangent_effective_rank: float
    same_state_delta_jacobian_raw_rank: int
    same_state_delta_jacobian_effective_rank: float
    local_state_tangent_dimension: int
    rate_to_state_pullback_residual: float
    rate_to_state_derivative_active_count: int
    normal_stability_eligible: bool
    normal_dimension: int
    normal_local_decay_ratio: float
    normal_local_max_real_part: float
    normal_to_tangent_coupling_norm: float
    normal_horizon: float
    n_train_trials: int
    n_test_trials: int
    preprocessing_fit_train_only: bool = True
    heldout_used_for_fit: bool = False
    approximation_scope: str = (
        "three_epoch_mean_rate_soft_operator_surrogate_not_full_trajectory_lds"
    )
    jacobian_scope: str = (
        "continuous_time_same_supplied_state_scalar_belief_gain_linearization"
    )
    normal_basis_scope: str = (
        "local_pseudoinverse_preactivation_pullback_of_train_epoch_rate_basis"
    )


def effective_control_dynamics_audit(
    *,
    train_features: ArrayLike,
    test_features: ArrayLike,
    train_state1_probability: ArrayLike,
    test_state1_probability: ArrayLike,
    network: EIRateNetwork,
    mean_state: ArrayLike,
    gain_axis: ArrayLike,
    gain_strength: float,
    latent_dim: int,
    ridge: float = 1e-4,
    normalize_activity: bool = False,
    jacobian_reference_belief: float = 0.5,
    normal_horizon: float | None = None,
    rate_derivative_tolerance: float = 1e-8,
    rate_pullback_residual_tolerance: float = 0.05,
    rank_rtol: float = 1e-10,
    rank_atol: float = 1e-12,
) -> EffectiveControlDynamicsAudit:
    """Fit on train epochs and audit held-out closure plus local E/I geometry."""

    if not isinstance(network, EIRateNetwork):
        raise TypeError("network must be an EIRateNetwork")
    relative = _nonnegative_scalar(rank_rtol, name="rank_rtol")
    absolute = _nonnegative_scalar(rank_atol, name="rank_atol")
    fitted = fit_soft_epoch_dynamics(
        train_features,
        train_state1_probability,
        n_units=network.n_units,
        latent_dim=latent_dim,
        ridge=ridge,
        normalize_activity=normalize_activity,
    )
    closure = fitted.score(test_features, test_state1_probability)

    physical = matrix_rank_summary(
        network.recurrent_weights, rtol=relative, atol=absolute
    )
    operator_delta = fitted.operator_state1 - fitted.operator_state0
    # With two endpoints their affine-operator span has at most one direction.
    operator_control = matrix_rank_summary(
        operator_delta.reshape(1, -1), rtol=relative, atol=absolute
    )

    tangent = ei_scalar_gain_jacobian_tangent(
        network,
        mean_state,
        gain_axis,
        gain_strength=gain_strength,
        belief=jacobian_reference_belief,
    )
    tangent_rank = matrix_rank_summary(tangent, rtol=relative, atol=absolute)
    jacobian_low = ei_continuous_jacobian(
        network,
        mean_state,
        gain_axis,
        gain_strength=gain_strength,
        belief=0.0,
    )
    jacobian_high = ei_continuous_jacobian(
        network,
        mean_state,
        gain_axis,
        gain_strength=gain_strength,
        belief=1.0,
    )
    delta_jacobian = jacobian_high - jacobian_low
    delta_rank = matrix_rank_summary(delta_jacobian, rtol=relative, atol=absolute)
    reference_jacobian = ei_continuous_jacobian(
        network,
        mean_state,
        gain_axis,
        gain_strength=gain_strength,
        belief=jacobian_reference_belief,
    )
    pulled_back_basis = pullback_rate_tangent_basis(
        network,
        mean_state,
        gain_axis,
        fitted.raw_activity_basis,
        gain_strength=gain_strength,
        belief=jacobian_reference_belief,
        derivative_tolerance=rate_derivative_tolerance,
        residual_tolerance=rate_pullback_residual_tolerance,
    )
    horizon = network.dt if normal_horizon is None else normal_horizon
    normal = projected_normal_linear_summary(
        reference_jacobian,
        pulled_back_basis.preactivation_basis,
        horizon=horizon,
    )
    return EffectiveControlDynamicsAudit(
        physical_raw_rank=physical.numerical_rank,
        physical_effective_rank=physical.effective_rank,
        physical_rank_threshold=physical.threshold,
        basis_dimension=fitted.latent_dim,
        basis_explained_variance_fraction=float(
            np.sum(fitted.pca.explained_variance_ratio_)
        ),
        heldout_normalized_closure_mse=closure.normalized_closure_mse,
        heldout_raw_latent_mse=closure.raw_latent_mse,
        heldout_basis_residual_fraction=closure.heldout_basis_residual_fraction,
        declared_scalar_control_dimension=1,
        operator_control_dimension=operator_control.numerical_rank,
        operator_control_effective_dimension=operator_control.effective_rank,
        operator_delta_frobenius_norm=float(np.linalg.norm(operator_delta)),
        operator_design_rank=fitted.operator_design_rank,
        operator_design_columns=fitted.operator_design_columns,
        jacobian_tangent_raw_rank=tangent_rank.numerical_rank,
        jacobian_tangent_effective_rank=tangent_rank.effective_rank,
        same_state_delta_jacobian_raw_rank=delta_rank.numerical_rank,
        same_state_delta_jacobian_effective_rank=delta_rank.effective_rank,
        local_state_tangent_dimension=pulled_back_basis.tangent_dimension,
        rate_to_state_pullback_residual=(
            pulled_back_basis.relative_rate_reconstruction_residual
        ),
        rate_to_state_derivative_active_count=(
            pulled_back_basis.derivative_active_count
        ),
        normal_stability_eligible=pulled_back_basis.eligible,
        normal_dimension=normal.normal_dimension,
        normal_local_decay_ratio=normal.decay_ratio,
        normal_local_max_real_part=normal.max_real_part,
        normal_to_tangent_coupling_norm=normal.normal_to_tangent_coupling_norm,
        normal_horizon=normal.horizon,
        n_train_trials=fitted.n_train_trials,
        n_test_trials=closure.n_trials,
    )


__all__ = [
    "EffectiveControlDynamicsAudit",
    "FittedSoftEpochDynamics",
    "LocalRateBasisPullback",
    "ProjectedNormalLinearSummary",
    "SoftEpochClosureScore",
    "effective_control_dynamics_audit",
    "ei_continuous_jacobian",
    "ei_scalar_gain_jacobian_tangent",
    "fit_soft_epoch_dynamics",
    "projected_normal_linear_summary",
    "pullback_rate_tangent_basis",
]
