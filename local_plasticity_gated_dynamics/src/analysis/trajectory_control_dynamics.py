"""Train-only full-trajectory dynamics and nonlinear stability audits.

This module complements the coarse three-epoch surrogate used by Exp19.  It
fits a controlled, belief-conditioned affine Koopman model to every saved
Euler substep of whole training episodes and evaluates both one-step
prediction and free multi-step rollout on disjoint held-out episodes.

The nonlinear perturbation audit uses the frozen physical E/I checkpoint.  It
does not infer stability from PCA variance: perturbations are injected into
physical ``x`` coordinates tangent or normal to the x projection of the same
train-fitted joint ``[x, rates]`` basis, then replayed with identical future
inputs and gains.  PCA, normalization, operators, belief-conditioned
subspaces, and all reference scales are fitted from training trajectories
only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

from src.analysis.manifold_metrics import (
    FittedPCASubspace,
    fit_train_pca,
    principal_angles,
)
from src.models.ei_rate_network import EIRateNetwork, EIRateState


FloatArray = NDArray[np.float64]
EPOCH_NAMES = ("cue", "sensory", "delay", "response")
EPOCH_CONTRAST_NAMES = EPOCH_NAMES[:-1]


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _trajectory_fingerprint(
    x: FloatArray,
    rates: FloatArray,
    inputs: FloatArray,
    belief: FloatArray,
    epoch: NDArray[np.str_],
    integration_substeps: int,
    sequence_scope: str,
) -> str:
    digest = hashlib.sha256(b"trajectory-operator-train-receipt-v1\0")
    for value in (x, rates, inputs, belief, epoch):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        digest.update(b"\0")
    digest.update(str(int(integration_substeps)).encode("ascii"))
    digest.update(b"\0")
    digest.update(sequence_scope.encode("utf-8"))
    return digest.hexdigest()


def _state_preprocessing_fingerprint(
    x: FloatArray,
    rates: FloatArray,
    integration_substeps: int,
    sequence_scope: str,
) -> str:
    digest = hashlib.sha256(b"trajectory-state-preprocessing-receipt-v1\0")
    for value in (x, rates):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        digest.update(b"\0")
    digest.update(str(int(integration_substeps)).encode("ascii"))
    digest.update(b"\0")
    digest.update(sequence_scope.encode("utf-8"))
    return digest.hexdigest()


def _nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fit_trajectory_pca(
    samples: FloatArray,
    n_components: int,
    *,
    normalize: bool,
    solver: str,
    seed: int | None,
) -> FittedPCASubspace:
    if solver == "exact":
        return fit_train_pca(samples, n_components, normalize=normalize)
    if solver != "randomized":
        raise ValueError("pca_solver must be 'exact' or 'randomized'")
    if (
        seed is None
        or isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("randomized PCA requires a non-negative integer pca_seed")
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    if normalize:
        empirical_scale = np.std(centered, axis=0, ddof=0)
        scale = np.where(empirical_scale > 0.0, empirical_scale, 1.0)
    else:
        scale = np.ones(samples.shape[1], dtype=np.float64)
    standardized = centered / scale
    model = PCA(
        n_components=int(n_components),
        svd_solver="randomized",
        random_state=int(seed),
        iterated_power=4,
    ).fit(standardized)
    return FittedPCASubspace(
        mean_=_readonly(mean),
        scale_=_readonly(scale),
        components_=_readonly(model.components_),
        explained_variance_=_readonly(model.explained_variance_),
        explained_variance_ratio_=_readonly(model.explained_variance_ratio_),
        n_train_samples_=int(samples.shape[0]),
        normalized_=bool(normalize),
        fit_sample_ids_=None,
    )


def _trajectory_arrays(
    x: ArrayLike,
    rates: ArrayLike,
    exogenous_inputs: ArrayLike,
    belief: ArrayLike,
    epoch: ArrayLike,
    *,
    integration_substeps: int,
    n_units: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, NDArray[np.str_]]:
    raw_x = np.asarray(x)
    raw_rates = np.asarray(rates)
    raw_inputs = np.asarray(exogenous_inputs)
    raw_belief = np.asarray(belief)
    raw_epoch = np.asarray(epoch)
    if raw_x.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("x must be real numeric")
    if raw_rates.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("rates must be real numeric")
    if raw_inputs.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("exogenous_inputs must be real numeric")
    if raw_belief.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("belief must be real numeric")
    x_array = np.asarray(raw_x, dtype=np.float64)
    rate_array = np.asarray(raw_rates, dtype=np.float64)
    input_array = np.asarray(raw_inputs, dtype=np.float64)
    probability = np.asarray(raw_belief, dtype=np.float64)
    substeps = _positive_int(integration_substeps, name="integration_substeps")
    if (
        x_array.ndim != 3
        or rate_array.ndim != 3
        or input_array.ndim != 3
        or probability.ndim != 2
        or 0 in rate_array.shape
        or 0 in input_array.shape
    ):
        raise ValueError("trajectory arrays must be non-empty sequence-major arrays")
    if x_array.shape != rate_array.shape:
        raise ValueError("x and rates must have identical trajectory shapes")
    if n_units is not None and rate_array.shape[2] != int(n_units):
        raise ValueError("rates unit dimension differs from n_units")
    if input_array.shape[:2] != probability.shape:
        raise ValueError("exogenous_inputs and belief must align")
    if input_array.shape[0] != rate_array.shape[0]:
        raise ValueError("rates and controls must have the same sequence count")
    expected_history = input_array.shape[1] * substeps + 1
    if rate_array.shape[1] != expected_history:
        raise ValueError(
            "rates history must contain one initial state plus every Euler substep"
        )
    epochs = np.asarray(raw_epoch, dtype="U8")
    if (
        epochs.shape != (input_array.shape[1],)
        or not np.isin(epochs, EPOCH_NAMES).all()
    ):
        raise ValueError("epoch must align with coarse controls and use known labels")
    if (
        not np.all(np.isfinite(x_array))
        or not np.all(np.isfinite(rate_array))
        or not np.all(np.isfinite(input_array))
        or not np.all(np.isfinite(probability))
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise ValueError("trajectory arrays must be finite and beliefs lie in [0, 1]")
    return x_array, rate_array, input_array, probability, epochs


def _expanded_controls(
    exogenous_inputs: FloatArray,
    belief: FloatArray,
    epoch: NDArray[np.str_],
    *,
    integration_substeps: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    substeps = int(integration_substeps)
    fine_inputs = np.repeat(exogenous_inputs, substeps, axis=1)
    fine_belief = np.repeat(belief, substeps, axis=1)
    # Response is the reference level.  Three contrasts plus an intercept are
    # full-rank; four one-hot columns plus an intercept would be structurally
    # rank-deficient.
    coarse_onehot = np.column_stack(
        [epoch == item for item in EPOCH_CONTRAST_NAMES]
    ).astype(np.float64)
    fine_epoch = np.repeat(coarse_onehot, substeps, axis=0)
    fine_epoch = np.broadcast_to(
        fine_epoch[np.newaxis, :, :],
        (
            exogenous_inputs.shape[0],
            fine_inputs.shape[1],
            len(EPOCH_CONTRAST_NAMES),
        ),
    )
    return fine_inputs, fine_belief, np.asarray(fine_epoch, dtype=np.float64)


def _soft_design(
    latent: FloatArray,
    inputs: FloatArray,
    epoch_onehot: FloatArray,
    probability: FloatArray,
) -> FloatArray:
    if (
        latent.ndim != 2
        or inputs.ndim != 2
        or epoch_onehot.ndim != 2
        or probability.shape != (latent.shape[0],)
        or inputs.shape[0] != latent.shape[0]
        or epoch_onehot.shape[0] != latent.shape[0]
    ):
        raise ValueError("latent states and controls must align")
    base = np.column_stack(
        (
            latent,
            inputs,
            epoch_onehot,
            np.ones(latent.shape[0], dtype=np.float64),
        )
    )
    return np.concatenate(
        ((1.0 - probability[:, None]) * base, probability[:, None] * base),
        axis=1,
    )


def _shared_neutral_cue_expansion(*, block_columns: int, cue_column: int) -> FloatArray:
    """Map one shared cue coefficient into two endpoint operator blocks."""

    block = _positive_int(block_columns, name="block_columns")
    if (
        isinstance(cue_column, (bool, np.bool_))
        or not isinstance(cue_column, (int, np.integer))
        or not 0 <= int(cue_column) < block
    ):
        raise ValueError("cue_column must index one endpoint operator block")
    cue = int(cue_column)
    high_cue = block + cue
    expansion = np.zeros((2 * block, 2 * block - 1), dtype=np.float64)
    for unconstrained in range(2 * block):
        if unconstrained == high_cue:
            identifiable = cue
        elif unconstrained < high_cue:
            identifiable = unconstrained
        else:
            identifiable = unconstrained - 1
        expansion[unconstrained, identifiable] = 1.0
    return expansion


def _soft_design_shared_neutral_cue(
    latent: FloatArray,
    inputs: FloatArray,
    epoch_contrasts: FloatArray,
    probability: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Share one cue coefficient when cue control is exactly neutral.

    The physical gate is not applied during the cue.  Consequently, the two
    cue endpoint coefficients are not separately observable when cue belief is
    exactly 0.5.  This explicit expansion keeps one shared coefficient (19
    columns in registered Exp21) rather than fitting an unidentifiable
    two-block design.
    """

    if epoch_contrasts.shape[1] != len(EPOCH_CONTRAST_NAMES):
        raise ValueError("epoch contrasts do not match the registered epochs")
    cue_mask = epoch_contrasts[:, 0] == 1.0
    if not np.any(cue_mask):
        raise ValueError("shared-neutral-cue mode requires observed cue transitions")
    if not np.all(probability[cue_mask] == 0.5):
        raise ValueError("shared-neutral-cue mode requires exact cue belief 0.5")
    full = _soft_design(latent, inputs, epoch_contrasts, probability)
    block = latent.shape[1] + inputs.shape[1] + epoch_contrasts.shape[1] + 1
    cue_column = latent.shape[1] + inputs.shape[1]
    expansion = _shared_neutral_cue_expansion(
        block_columns=block,
        cue_column=cue_column,
    )
    return full @ expansion, expansion


def _state_only_design(
    latent: FloatArray,
    inputs: FloatArray,
    epoch_contrasts: FloatArray,
    probability: FloatArray,
) -> FloatArray:
    # State-transition coefficients and the affine bias may depend on belief.
    # Exogenous input and epoch coefficients remain shared.  The bias switch
    # is required because PCA centering turns a physical A switch into both an
    # A and intercept switch in latent coordinates.
    shared = np.column_stack((inputs, epoch_contrasts))
    return np.column_stack(
        (
            (1.0 - probability[:, None]) * latent,
            probability[:, None] * latent,
            shared,
            1.0 - probability,
            probability,
        )
    )


@dataclass(frozen=True, slots=True)
class TrajectoryKoopmanScore:
    """Held-out teacher-forced and free-rollout errors."""

    one_step_normalized_mse: float
    one_step_raw_latent_mse: float
    rollout_normalized_mse: float
    rollout_normalized_rmse: float
    rollout_endpoint_normalized_rmse: float
    rollout_per_horizon_normalized_rmse: FloatArray
    heldout_state_basis_residual_fraction: float
    heldout_rate_basis_residual_fraction: float
    n_sequences: int
    n_transitions: int
    n_rollout_windows: int
    interpretation: str = (
        "heldout_sequence_open_loop_rollout_from_initial_state_conditioned_on_"
        "observed_future_exogenous_controls"
    )


@dataclass(frozen=True, slots=True)
class PhysicalXTangentBasis:
    """Physical-x projection of the train-fitted joint latent basis."""

    basis: FloatArray
    singular_values: FloatArray
    rank: int
    condition_number: float
    x_block_energy_fraction: float


@dataclass(frozen=True, slots=True)
class FittedTrajectoryKoopman:
    """Train-fitted PCA and scalar-belief controlled affine operators."""

    state_pca: FittedPCASubspace
    rate_pca: FittedPCASubspace
    operator_state0: FloatArray
    operator_state1: FloatArray
    train_following_latent_variance: FloatArray
    n_units: int
    n_inputs: int
    integration_substeps: int
    ridge: float
    n_train_sequences: int
    n_train_transitions: int
    operator_design_rank: int
    operator_design_columns: int
    operator_unconstrained_columns: int
    operator_constraint: str
    belief_conditioned: bool
    operator_mode: str
    training_trajectory_fingerprint: str
    state_preprocessing_fingerprint: str
    trajectory_sequence_scope: str
    pca_solver: str
    pca_seed: int | None
    preprocessing_fit_scope: str = "whole_training_sequences_substeps_only"

    @property
    def latent_dim(self) -> int:
        return int(self.state_pca.components_.shape[0])

    @property
    def operator_identifiable_columns(self) -> int:
        """Number of free columns in the fitted operator parameterization."""

        return self.operator_design_columns

    @property
    def raw_rate_basis(self) -> FloatArray:
        raw_directions = self.rate_pca.scale_[:, None] * self.rate_pca.basis_
        basis, _ = np.linalg.qr(raw_directions, mode="reduced")
        return _readonly(basis)

    @property
    def physical_x_tangent_basis(self) -> PhysicalXTangentBasis:
        """Return the x block of the same joint basis used by the operator."""

        raw_joint = self.state_pca.scale_[:, None] * self.state_pca.components_.T
        x_block = raw_joint[: self.n_units]
        left, singular, _ = np.linalg.svd(x_block, full_matrices=False)
        tolerance = (
            np.finfo(np.float64).eps
            * max(x_block.shape)
            * (float(singular[0]) if singular.size else 0.0)
        )
        rank = int(np.count_nonzero(singular > tolerance))
        condition = (
            float(singular[0] / singular[self.latent_dim - 1])
            if rank >= self.latent_dim
            else float("inf")
        )
        joint_energy = float(np.sum(raw_joint * raw_joint))
        x_energy = float(np.sum(x_block * x_block))
        return PhysicalXTangentBasis(
            basis=_readonly(left[:, : min(rank, self.latent_dim)]),
            singular_values=_readonly(singular),
            rank=rank,
            condition_number=condition,
            x_block_energy_fraction=(
                x_energy / joint_energy if joint_energy > 0.0 else 0.0
            ),
        )

    @property
    def state_transition_delta_frobenius(self) -> float:
        return float(
            np.linalg.norm(
                self.operator_state1[: self.latent_dim]
                - self.operator_state0[: self.latent_dim]
            )
        )

    @property
    def exogenous_offset_delta_frobenius(self) -> float:
        return float(
            np.linalg.norm(
                self.operator_state1[self.latent_dim :]
                - self.operator_state0[self.latent_dim :]
            )
        )

    @property
    def exogenous_control_delta_frobenius(self) -> float:
        return float(
            np.linalg.norm(
                self.operator_state1[self.latent_dim : -1]
                - self.operator_state0[self.latent_dim : -1]
            )
        )

    @property
    def affine_bias_delta_norm(self) -> float:
        return float(
            np.linalg.norm(self.operator_state1[-1] - self.operator_state0[-1])
        )

    @property
    def parameter_count(self) -> int:
        """Number of fitted scalar coefficients in the operator design."""

        return int(self.operator_design_columns * self.latent_dim)

    def score(
        self,
        x: ArrayLike,
        rates: ArrayLike,
        raw_inputs: ArrayLike,
        belief: ArrayLike,
        epoch: ArrayLike,
        *,
        sequence_scope: str | None = None,
        rollout_horizon_steps: int | None = None,
        rollout_stride_steps: int | None = None,
    ) -> TrajectoryKoopmanScore:
        if (
            sequence_scope is not None
            and str(sequence_scope) != self.trajectory_sequence_scope
        ):
            raise ValueError("held-out trajectory sequence scope differs from fit")
        x_array, rate_array, input_array, probability, epochs = _trajectory_arrays(
            x,
            rates,
            raw_inputs,
            belief,
            epoch,
            integration_substeps=self.integration_substeps,
            n_units=self.n_units,
        )
        if input_array.shape[2] != self.n_inputs:
            raise ValueError("held-out input dimension differs from fitted model")
        fine_inputs, fine_probability, fine_epoch = _expanded_controls(
            input_array,
            probability,
            epochs,
            integration_substeps=self.integration_substeps,
        )
        n_sequences, n_states, _ = rate_array.shape
        state_array = np.concatenate((x_array, rate_array), axis=2)
        latent = self.state_pca.transform(
            state_array.reshape(-1, 2 * self.n_units)
        ).reshape(n_sequences, n_states, self.latent_dim)
        current = latent[:, :-1].reshape(-1, self.latent_dim)
        following = latent[:, 1:].reshape(-1, self.latent_dim)
        flat_inputs = fine_inputs.reshape(-1, self.n_inputs)
        flat_epoch = fine_epoch.reshape(-1, len(EPOCH_CONTRAST_NAMES))
        base = np.column_stack(
            (
                current,
                flat_inputs,
                flat_epoch,
                np.ones(current.shape[0], dtype=np.float64),
            )
        )
        if self.belief_conditioned:
            design = _soft_design(
                current,
                flat_inputs,
                flat_epoch,
                fine_probability.reshape(-1),
            )
            coefficients = np.concatenate(
                (self.operator_state0, self.operator_state1), axis=0
            )
            one_step = design @ coefficients
        else:
            one_step = base @ self.operator_state0
        one_error = one_step - following
        one_raw = float(np.mean(one_error * one_error))
        one_normalized = float(
            np.mean(
                (one_error * one_error) / self.train_following_latent_variance[None, :]
            )
        )

        maximum_horizon = n_states - 1
        horizon = (
            maximum_horizon
            if rollout_horizon_steps is None
            else _positive_int(rollout_horizon_steps, name="rollout_horizon_steps")
        )
        horizon = min(horizon, maximum_horizon)
        stride = (
            horizon
            if rollout_stride_steps is None
            else _positive_int(rollout_stride_steps, name="rollout_stride_steps")
        )
        starts = np.arange(
            0,
            maximum_horizon - horizon + 1,
            stride,
            dtype=int,
        )
        if starts.size == 0:
            starts = np.array([0], dtype=int)
        block = self.operator_state0.shape[0]
        rollout_errors: list[FloatArray] = []
        for start in starts:
            window = np.empty(
                (n_sequences, horizon + 1, self.latent_dim),
                dtype=np.float64,
            )
            window[:, 0] = latent[:, start]
            for offset in range(horizon):
                time = int(start + offset)
                base = np.column_stack(
                    (
                        window[:, offset],
                        fine_inputs[:, time],
                        fine_epoch[:, time],
                        np.ones(n_sequences, dtype=np.float64),
                    )
                )
                if base.shape[1] != block:
                    raise RuntimeError("rollout design differs from fitted operator")
                if self.belief_conditioned:
                    p = fine_probability[:, time, None]
                    window[:, offset + 1] = (1.0 - p) * (
                        base @ self.operator_state0
                    ) + p * (base @ self.operator_state1)
                else:
                    window[:, offset + 1] = base @ self.operator_state0
            rollout_errors.append(
                window[:, 1:] - latent[:, start + 1 : start + horizon + 1]
            )
        rollout_error = np.concatenate(rollout_errors, axis=0)
        normalized_squared = (
            rollout_error * rollout_error
        ) / self.train_following_latent_variance[None, None, :]
        per_horizon = np.sqrt(np.mean(normalized_squared, axis=(0, 2)))
        rollout_mse = float(np.mean(normalized_squared))

        flat_state = state_array[:, 1:].reshape(-1, 2 * self.n_units)
        standardized_state = (flat_state - self.state_pca.mean_) / self.state_pca.scale_
        projected_state = (
            standardized_state @ self.state_pca.components_.T
        ) @ self.state_pca.components_
        state_total = float(np.sum(standardized_state * standardized_state))
        state_residual = float(np.sum((standardized_state - projected_state) ** 2))
        flat_rates = rate_array[:, 1:].reshape(-1, self.n_units)
        standardized_rates = (flat_rates - self.rate_pca.mean_) / self.rate_pca.scale_
        projected_rates = (
            standardized_rates @ self.rate_pca.components_.T
        ) @ self.rate_pca.components_
        rate_total = float(np.sum(standardized_rates * standardized_rates))
        rate_residual = float(np.sum((standardized_rates - projected_rates) ** 2))
        return TrajectoryKoopmanScore(
            one_step_normalized_mse=one_normalized,
            one_step_raw_latent_mse=one_raw,
            rollout_normalized_mse=rollout_mse,
            rollout_normalized_rmse=float(np.sqrt(rollout_mse)),
            rollout_endpoint_normalized_rmse=float(per_horizon[-1]),
            rollout_per_horizon_normalized_rmse=_readonly(per_horizon),
            heldout_state_basis_residual_fraction=(
                state_residual / state_total if state_total > 0.0 else 0.0
            ),
            heldout_rate_basis_residual_fraction=(
                rate_residual / rate_total if rate_total > 0.0 else 0.0
            ),
            n_sequences=int(n_sequences),
            n_transitions=int((n_states - 1) * n_sequences),
            n_rollout_windows=int(starts.size * n_sequences),
        )


def fit_trajectory_koopman(
    train_x: ArrayLike,
    train_rates: ArrayLike,
    train_raw_inputs: ArrayLike,
    train_belief: ArrayLike,
    train_epoch: ArrayLike,
    *,
    integration_substeps: int,
    latent_dim: int,
    ridge: float = 1e-4,
    normalize_activity: bool = True,
    belief_conditioned: bool = True,
    operator_mode: str | None = None,
    shared_preprocessing: FittedTrajectoryKoopman | None = None,
    pca_solver: str | None = None,
    pca_seed: int | None = None,
    sequence_scope: str = "unspecified_sequence",
) -> FittedTrajectoryKoopman:
    """Fit trajectory preprocessing and operators from training episodes.

    ``shared_preprocessing`` lets a paired common-operator baseline reuse the
    exact state and rate PCA fitted by a belief-conditioned model on the same
    training trajectories.  This avoids giving either paired operator a
    different representation and avoids repeating an expensive high-
    dimensional PCA.  The caller remains responsible for passing the same
    training receipt; held-out trajectories must never be used here.
    """

    x, rates, inputs, probability, epochs = _trajectory_arrays(
        train_x,
        train_rates,
        train_raw_inputs,
        train_belief,
        train_epoch,
        integration_substeps=integration_substeps,
    )
    if (
        isinstance(latent_dim, (bool, np.bool_))
        or not isinstance(latent_dim, (int, np.integer))
        or not 1 <= int(latent_dim) < rates.shape[2]
    ):
        raise ValueError("latent_dim must be an integer in [1, n_units - 1]")
    ridge_value = _nonnegative(ridge, name="ridge")
    if not isinstance(normalize_activity, (bool, np.bool_)):
        raise TypeError("normalize_activity must be boolean")
    if not isinstance(belief_conditioned, (bool, np.bool_)):
        raise TypeError("belief_conditioned must be boolean")
    mode = (
        ("full" if bool(belief_conditioned) else "common")
        if operator_mode is None
        else str(operator_mode)
    )
    if mode not in {"common", "state_only", "full", "full_shared_neutral_cue"}:
        raise ValueError(
            "operator_mode must be 'common', 'state_only', 'full', or "
            "'full_shared_neutral_cue'"
        )
    if operator_mode is not None and bool(belief_conditioned) != (mode != "common"):
        raise ValueError("belief_conditioned and operator_mode disagree")
    if shared_preprocessing is not None and not isinstance(
        shared_preprocessing, FittedTrajectoryKoopman
    ):
        raise TypeError(
            "shared_preprocessing must be a FittedTrajectoryKoopman or None"
        )
    scope = str(sequence_scope)
    if not scope or scope.isspace():
        raise ValueError("sequence_scope must be a non-empty string")
    solver = (
        shared_preprocessing.pca_solver
        if shared_preprocessing is not None and pca_solver is None
        else "exact"
        if pca_solver is None
        else str(pca_solver)
    )
    if solver not in {"exact", "randomized"}:
        raise ValueError("pca_solver must be 'exact', 'randomized', or None")
    if pca_seed is not None and (
        isinstance(pca_seed, (bool, np.bool_))
        or not isinstance(pca_seed, (int, np.integer))
        or not 0 <= int(pca_seed) <= 2**32 - 2
    ):
        raise ValueError("pca_seed must be an integer in [0, 2**32 - 2] or None")
    if shared_preprocessing is None:
        if solver == "randomized" and pca_seed is None:
            raise ValueError("randomized PCA requires pca_seed")
        if solver == "exact" and pca_seed is not None:
            raise ValueError("exact PCA requires pca_seed=None")
    elif pca_seed is not None and shared_preprocessing.pca_seed != int(pca_seed):
        raise ValueError("explicit pca_seed differs from shared preprocessing")
    training_receipt = _trajectory_fingerprint(
        x,
        rates,
        inputs,
        probability,
        epochs,
        int(integration_substeps),
        scope,
    )
    preprocessing_receipt = _state_preprocessing_fingerprint(
        x,
        rates,
        int(integration_substeps),
        scope,
    )
    fine_inputs, fine_probability, fine_epoch = _expanded_controls(
        inputs,
        probability,
        epochs,
        integration_substeps=integration_substeps,
    )
    if shared_preprocessing is None:
        train_state_samples = np.concatenate((x[:, 1:], rates[:, 1:]), axis=2).reshape(
            -1, 2 * rates.shape[2]
        )
        state_pca = _fit_trajectory_pca(
            train_state_samples,
            int(latent_dim),
            normalize=bool(normalize_activity),
            solver=solver,
            seed=pca_seed,
        )
        rate_pca = _fit_trajectory_pca(
            rates[:, 1:].reshape(-1, rates.shape[2]),
            int(latent_dim),
            normalize=bool(normalize_activity),
            solver=solver,
            seed=(None if pca_seed is None else int(pca_seed) + 1),
        )
        preprocessing_scope = "whole_training_sequences_substeps_only"
    else:
        if (
            shared_preprocessing.n_units != rates.shape[2]
            or shared_preprocessing.n_inputs != inputs.shape[2]
            or shared_preprocessing.integration_substeps != int(integration_substeps)
            or shared_preprocessing.latent_dim != int(latent_dim)
            or shared_preprocessing.state_pca.normalized_ != bool(normalize_activity)
            or shared_preprocessing.rate_pca.normalized_ != bool(normalize_activity)
            or shared_preprocessing.pca_solver != solver
            or shared_preprocessing.state_preprocessing_fingerprint
            != preprocessing_receipt
            or shared_preprocessing.trajectory_sequence_scope != scope
        ):
            raise ValueError(
                "shared preprocessing is incompatible with training trajectories"
            )
        state_pca = shared_preprocessing.state_pca
        rate_pca = shared_preprocessing.rate_pca
        preprocessing_scope = "paired_reuse_of_whole_training_sequences_substeps_only"
    all_state = np.concatenate((x, rates), axis=2)
    all_latent = state_pca.transform(all_state.reshape(-1, 2 * rates.shape[2])).reshape(
        rates.shape[0], rates.shape[1], int(latent_dim)
    )
    current = all_latent[:, :-1].reshape(-1, int(latent_dim))
    following = all_latent[:, 1:].reshape(-1, int(latent_dim))
    flat_inputs = fine_inputs.reshape(-1, inputs.shape[2])
    flat_epoch = fine_epoch.reshape(-1, len(EPOCH_CONTRAST_NAMES))
    base_design = np.column_stack(
        (
            current,
            flat_inputs,
            flat_epoch,
            np.ones(current.shape[0], dtype=np.float64),
        )
    )
    expansion: FloatArray | None = None
    if mode == "full":
        design = _soft_design(
            current,
            flat_inputs,
            flat_epoch,
            fine_probability.reshape(-1),
        )
    elif mode == "full_shared_neutral_cue":
        design, expansion = _soft_design_shared_neutral_cue(
            current,
            flat_inputs,
            flat_epoch,
            fine_probability.reshape(-1),
        )
    elif mode == "state_only":
        design = _state_only_design(
            current,
            flat_inputs,
            flat_epoch,
            fine_probability.reshape(-1),
        )
    else:
        design = base_design
    block = int(latent_dim) + inputs.shape[2] + len(EPOCH_CONTRAST_NAMES) + 1
    if mode in {"full", "full_shared_neutral_cue"}:
        unconstrained_penalty = ridge_value * np.eye(2 * block, dtype=np.float64)
        unconstrained_penalty[block - 1, block - 1] = 0.0
        unconstrained_penalty[2 * block - 1, 2 * block - 1] = 0.0
        penalty = (
            unconstrained_penalty
            if expansion is None
            else expansion.T @ unconstrained_penalty @ expansion
        )
    elif mode == "state_only":
        penalty = ridge_value * np.eye(design.shape[1], dtype=np.float64)
        penalty[-2, -2] = 0.0
        penalty[-1, -1] = 0.0
    else:
        penalty = ridge_value * np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
    gram = design.T @ design
    rhs = design.T @ following
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + penalty) @ rhs
    variance = np.var(following, axis=0, ddof=0)
    following_latent_variance = np.where(
        variance > np.finfo(np.float64).eps, variance, 1.0
    )
    if mode in {"full", "full_shared_neutral_cue"}:
        unconstrained_coefficients = (
            coefficients if expansion is None else expansion @ coefficients
        )
        state0 = unconstrained_coefficients[:block]
        state1 = unconstrained_coefficients[block:]
    elif mode == "state_only":
        shared_start = 2 * int(latent_dim)
        shared_stop = shared_start + inputs.shape[2] + len(EPOCH_CONTRAST_NAMES)
        shared = coefficients[shared_start:shared_stop]
        state0 = np.vstack(
            (
                coefficients[: int(latent_dim)],
                shared,
                coefficients[-2],
            )
        )
        state1 = np.vstack(
            (
                coefficients[int(latent_dim) : 2 * int(latent_dim)],
                shared,
                coefficients[-1],
            )
        )
    else:
        state0 = coefficients
        state1 = state0
    return FittedTrajectoryKoopman(
        state_pca=state_pca,
        rate_pca=rate_pca,
        operator_state0=_readonly(state0),
        operator_state1=_readonly(state1),
        train_following_latent_variance=_readonly(following_latent_variance),
        n_units=int(rates.shape[2]),
        n_inputs=int(inputs.shape[2]),
        integration_substeps=int(integration_substeps),
        ridge=ridge_value,
        n_train_sequences=int(rates.shape[0]),
        n_train_transitions=int(current.shape[0]),
        operator_design_rank=int(np.linalg.matrix_rank(design)),
        operator_design_columns=int(design.shape[1]),
        operator_unconstrained_columns=(
            int(2 * block)
            if mode in {"full", "full_shared_neutral_cue"}
            else int(design.shape[1])
        ),
        operator_constraint=(
            "shared_neutral_cue_coefficient"
            if mode == "full_shared_neutral_cue"
            else "none"
        ),
        belief_conditioned=mode != "common",
        operator_mode=mode,
        training_trajectory_fingerprint=training_receipt,
        state_preprocessing_fingerprint=preprocessing_receipt,
        trajectory_sequence_scope=scope,
        pca_solver=solver,
        pca_seed=(
            int(pca_seed)
            if shared_preprocessing is None
            and solver == "randomized"
            and pca_seed is not None
            else shared_preprocessing.pca_seed
            if shared_preprocessing is not None
            else None
        ),
        preprocessing_fit_scope=preprocessing_scope,
    )


def persistence_trajectory_score(
    fitted: FittedTrajectoryKoopman,
    x: ArrayLike,
    rates: ArrayLike,
    raw_inputs: ArrayLike,
    belief: ArrayLike,
    epoch: ArrayLike,
    *,
    sequence_scope: str | None = None,
    rollout_horizon_steps: int | None = None,
    rollout_stride_steps: int | None = None,
) -> TrajectoryKoopmanScore:
    """Score a no-dynamics persistence baseline in the frozen train basis."""

    if (
        sequence_scope is not None
        and str(sequence_scope) != fitted.trajectory_sequence_scope
    ):
        raise ValueError("held-out trajectory sequence scope differs from fit")
    x_array, rate_array, input_array, _, _ = _trajectory_arrays(
        x,
        rates,
        raw_inputs,
        belief,
        epoch,
        integration_substeps=fitted.integration_substeps,
        n_units=fitted.n_units,
    )
    if input_array.shape[2] != fitted.n_inputs:
        raise ValueError("held-out input dimension differs from fitted model")
    states = np.concatenate((x_array, rate_array), axis=2)
    latent = fitted.state_pca.transform(states.reshape(-1, 2 * fitted.n_units)).reshape(
        states.shape[0], states.shape[1], fitted.latent_dim
    )
    one_error = latent[:, :-1] - latent[:, 1:]
    one_raw = float(np.mean(one_error * one_error))
    one_normalized = float(
        np.mean(
            (one_error * one_error)
            / fitted.train_following_latent_variance[None, None, :]
        )
    )
    maximum_horizon = latent.shape[1] - 1
    horizon = (
        maximum_horizon
        if rollout_horizon_steps is None
        else _positive_int(rollout_horizon_steps, name="rollout_horizon_steps")
    )
    horizon = min(horizon, maximum_horizon)
    stride = (
        horizon
        if rollout_stride_steps is None
        else _positive_int(rollout_stride_steps, name="rollout_stride_steps")
    )
    starts = np.arange(
        0,
        maximum_horizon - horizon + 1,
        stride,
        dtype=int,
    )
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    rollout_errors = [
        np.repeat(latent[:, start : start + 1], horizon, axis=1)
        - latent[:, start + 1 : start + horizon + 1]
        for start in starts
    ]
    rollout_error = np.concatenate(rollout_errors, axis=0)
    normalized_squared = (
        rollout_error * rollout_error
    ) / fitted.train_following_latent_variance[None, None, :]
    per_horizon = np.sqrt(np.mean(normalized_squared, axis=(0, 2)))

    flat_state = states[:, 1:].reshape(-1, 2 * fitted.n_units)
    standardized_state = (flat_state - fitted.state_pca.mean_) / fitted.state_pca.scale_
    projected_state = (
        standardized_state @ fitted.state_pca.components_.T
    ) @ fitted.state_pca.components_
    state_total = float(np.sum(standardized_state * standardized_state))
    state_residual = float(np.sum((standardized_state - projected_state) ** 2))
    flat_rates = rate_array[:, 1:].reshape(-1, fitted.n_units)
    standardized_rates = (flat_rates - fitted.rate_pca.mean_) / fitted.rate_pca.scale_
    projected_rates = (
        standardized_rates @ fitted.rate_pca.components_.T
    ) @ fitted.rate_pca.components_
    rate_total = float(np.sum(standardized_rates * standardized_rates))
    rate_residual = float(np.sum((standardized_rates - projected_rates) ** 2))
    rollout_mse = float(np.mean(normalized_squared))
    return TrajectoryKoopmanScore(
        one_step_normalized_mse=one_normalized,
        one_step_raw_latent_mse=one_raw,
        rollout_normalized_mse=rollout_mse,
        rollout_normalized_rmse=float(np.sqrt(rollout_mse)),
        rollout_endpoint_normalized_rmse=float(per_horizon[-1]),
        rollout_per_horizon_normalized_rmse=_readonly(per_horizon),
        heldout_state_basis_residual_fraction=(
            state_residual / state_total if state_total > 0.0 else 0.0
        ),
        heldout_rate_basis_residual_fraction=(
            rate_residual / rate_total if rate_total > 0.0 else 0.0
        ),
        n_sequences=int(states.shape[0]),
        n_transitions=int(states.shape[0] * (states.shape[1] - 1)),
        n_rollout_windows=int(starts.size * states.shape[0]),
        interpretation="heldout_sequence_persistence_baseline_in_train_state_basis",
    )


@dataclass(frozen=True, slots=True)
class BeliefManifoldGeometry:
    """Train-defined low/high-belief subspaces and held-out centroid audit."""

    eligible: bool
    ineligibility_reason: str | None
    subspace_dimension: int | None
    principal_angles_degrees: tuple[float, ...]
    mean_principal_angle_degrees: float | None
    maximum_principal_angle_degrees: float | None
    train_centroid_distance: float | None
    heldout_nearest_centroid_accuracy: float | None
    n_train_low: int
    n_train_high: int
    n_test_extreme: int
    interpretation: str = (
        "belief_binned_train_subspaces_and_centroids_not_proof_of_attractors"
    )


def belief_manifold_geometry(
    fitted: FittedTrajectoryKoopman,
    train_rates: ArrayLike,
    train_belief: ArrayLike,
    test_rates: ArrayLike,
    test_belief: ArrayLike,
    *,
    low_threshold: float = 0.25,
    high_threshold: float = 0.75,
    subspace_dim: int | None = None,
) -> BeliefManifoldGeometry:
    """Compare low/high-belief trajectory geometry using training fits only."""

    low = _nonnegative(low_threshold, name="low_threshold")
    high = _nonnegative(high_threshold, name="high_threshold")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("belief thresholds must satisfy 0 <= low < high <= 1")
    raw_train = np.asarray(train_rates)
    raw_test = np.asarray(test_rates)
    raw_train_p = np.asarray(train_belief)
    raw_test_p = np.asarray(test_belief)
    if raw_train.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("train_rates must be real numeric")
    if raw_test.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("test_rates must be real numeric")
    if raw_train_p.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("train_belief must be real numeric")
    if raw_test_p.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError("test_belief must be real numeric")
    train = np.asarray(raw_train, dtype=np.float64)
    test = np.asarray(raw_test, dtype=np.float64)
    train_p = np.asarray(raw_train_p, dtype=np.float64)
    test_p = np.asarray(raw_test_p, dtype=np.float64)
    if (
        train.ndim != 3
        or test.ndim != 3
        or train.shape[2] != fitted.n_units
        or test.shape[2] != fitted.n_units
        or train_p.ndim != 2
        or test_p.ndim != 2
    ):
        raise ValueError("geometry trajectories have incompatible shapes")
    if (
        train_p.shape[0] != train.shape[0]
        or test_p.shape[0] != test.shape[0]
        or not np.all(np.isfinite(train))
        or not np.all(np.isfinite(test))
        or not np.all(np.isfinite(train_p))
        or not np.all(np.isfinite(test_p))
        or np.any((train_p < 0.0) | (train_p > 1.0))
        or np.any((test_p < 0.0) | (test_p > 1.0))
    ):
        raise ValueError(
            "geometry trajectories must be finite, sequence-aligned, and "
            "beliefs must lie in [0, 1]"
        )
    if train.shape[1] - 1 != train_p.shape[1] * fitted.integration_substeps:
        raise ValueError("train belief does not align with substep trajectory")
    if test.shape[1] - 1 != test_p.shape[1] * fitted.integration_substeps:
        raise ValueError("test belief does not align with substep trajectory")
    fine_train_p = np.repeat(train_p, fitted.integration_substeps, axis=1).reshape(-1)
    fine_test_p = np.repeat(test_p, fitted.integration_substeps, axis=1).reshape(-1)
    train_flat = train[:, 1:].reshape(-1, fitted.n_units)
    test_flat = test[:, 1:].reshape(-1, fitted.n_units)
    train_standardized = (train_flat - fitted.rate_pca.mean_) / fitted.rate_pca.scale_
    test_latent = fitted.rate_pca.transform(test_flat)
    train_latent = fitted.rate_pca.transform(train_flat)
    train_low = fine_train_p <= low
    train_high = fine_train_p >= high
    test_extreme = (fine_test_p <= low) | (fine_test_p >= high)
    n_train_low = int(np.count_nonzero(train_low))
    n_train_high = int(np.count_nonzero(train_high))
    n_test_extreme = int(np.count_nonzero(test_extreme))
    if n_train_low < 2 or n_train_high < 2:
        return BeliefManifoldGeometry(
            eligible=False,
            ineligibility_reason=(
                "insufficient_or_rank_deficient_low_or_high_belief_training_samples"
            ),
            subspace_dimension=None,
            principal_angles_degrees=(),
            mean_principal_angle_degrees=None,
            maximum_principal_angle_degrees=None,
            train_centroid_distance=None,
            heldout_nearest_centroid_accuracy=None,
            n_train_low=n_train_low,
            n_train_high=n_train_high,
            n_test_extreme=n_test_extreme,
        )
    if subspace_dim is None:
        requested_dim = fitted.latent_dim
    elif (
        isinstance(subspace_dim, (bool, np.bool_))
        or not isinstance(subspace_dim, (int, np.integer))
        or int(subspace_dim) < 1
    ):
        raise ValueError("subspace_dim must be a positive integer or None")
    else:
        requested_dim = int(subspace_dim)
    low_centered = train_standardized[train_low] - np.mean(
        train_standardized[train_low], axis=0, keepdims=True
    )
    high_centered = train_standardized[train_high] - np.mean(
        train_standardized[train_high], axis=0, keepdims=True
    )
    _, low_singular, low_vt = np.linalg.svd(low_centered, full_matrices=False)
    _, high_singular, high_vt = np.linalg.svd(high_centered, full_matrices=False)
    low_tolerance = max(
        1e-12,
        np.finfo(np.float64).eps
        * max(low_centered.shape)
        * (float(low_singular[0]) if low_singular.size else 0.0),
    )
    high_tolerance = max(
        1e-12,
        np.finfo(np.float64).eps
        * max(high_centered.shape)
        * (float(high_singular[0]) if high_singular.size else 0.0),
    )
    available = min(
        requested_dim,
        n_train_low - 1,
        n_train_high - 1,
        fitted.n_units,
        int(np.count_nonzero(low_singular > low_tolerance)),
        int(np.count_nonzero(high_singular > high_tolerance)),
    )
    if available < 1:
        return BeliefManifoldGeometry(
            eligible=False,
            ineligibility_reason=(
                "insufficient_or_rank_deficient_low_or_high_belief_training_samples"
            ),
            subspace_dimension=None,
            principal_angles_degrees=(),
            mean_principal_angle_degrees=None,
            maximum_principal_angle_degrees=None,
            train_centroid_distance=None,
            heldout_nearest_centroid_accuracy=None,
            n_train_low=n_train_low,
            n_train_high=n_train_high,
            n_test_extreme=n_test_extreme,
        )
    angles = principal_angles(
        low_vt[:available].T,
        high_vt[:available].T,
        degrees=True,
    )
    low_centroid = np.mean(train_latent[train_low], axis=0)
    high_centroid = np.mean(train_latent[train_high], axis=0)
    pooled_scale = np.sqrt(
        np.mean(
            np.var(train_latent[train_low], axis=0)
            + np.var(train_latent[train_high], axis=0)
        )
    )
    centroid_distance = float(
        np.linalg.norm(high_centroid - low_centroid)
        / max(pooled_scale, np.finfo(np.float64).eps)
    )
    if np.any(test_extreme):
        extreme_latent = test_latent[test_extreme]
        low_distance = np.linalg.norm(extreme_latent - low_centroid, axis=1)
        high_distance = np.linalg.norm(extreme_latent - high_centroid, axis=1)
        predicted_high = high_distance < low_distance
        true_high = fine_test_p[test_extreme] >= high
        centroid_accuracy = float(np.mean(predicted_high == true_high))
    else:
        centroid_accuracy = None
    return BeliefManifoldGeometry(
        eligible=True,
        ineligibility_reason=None,
        subspace_dimension=int(available),
        principal_angles_degrees=tuple(float(item) for item in angles),
        mean_principal_angle_degrees=float(np.mean(angles)),
        maximum_principal_angle_degrees=float(np.max(angles)),
        train_centroid_distance=centroid_distance,
        heldout_nearest_centroid_accuracy=centroid_accuracy,
        n_train_low=n_train_low,
        n_train_high=n_train_high,
        n_test_extreme=n_test_extreme,
    )


class PerturbationEligibilityError(ValueError):
    """Expected scientific ineligibility for the nonlinear perturbation audit."""


@dataclass(frozen=True, slots=True)
class NonlinearPerturbationSummary:
    """Frozen-network physical-x tangent/normal recovery statistics."""

    tangent_basis_space: str
    tangent_basis_rank: int
    tangent_basis_singular_values: tuple[float, ...]
    tangent_basis_condition_number: float
    tangent_basis_x_block_energy_fraction: float
    sampled_reference_fraction: float
    eligible_sampled_reference_fraction: float
    eligible_reference_fraction: float
    eligible_reference_count: int
    planned_reference_count: int
    candidate_reference_count: int
    sampled_reference_count: int
    normal_perturbation_count: int
    normal_recovery_fraction: float | None
    normal_endpoint_ratio_median: float | None
    normal_endpoint_ratio_maximum: float | None
    maximum_projected_finite_time_normal_log_growth_rate: float | None
    tangent_endpoint_ratio_median: float | None
    normal_vs_tangent_log_ratio_median: float | None
    initial_normal_purity_median: float | None
    initial_tangent_purity_median: float | None
    amplitudes: tuple[float, ...]
    normal_endpoint_ratio_by_amplitude: tuple[float | None, ...]
    baseline_replay_max_abs_error: float
    baseline_replay_tolerance: float
    horizon_steps: int
    horizon_time: float
    interpretation: str = (
        "finite_amplitude_recovery_normal_to_physical_x_projection_of_train_"
        "joint_latent_subspace_with_identical_future_input_and_gain_not_an_"
        "asymptotic_lyapunov_exponent"
    )


@dataclass(frozen=True, slots=True)
class FixedDriveAttractorProbe:
    """Counterfactual fixed-drive convergence under two frozen beliefs."""

    beliefs: tuple[float, float]
    initial_state_count: int
    endpoint_dispersion_contraction: tuple[float, float]
    endpoint_centroid_separation: float
    normalized_endpoint_centroid_separation: float
    centroid_separation_over_initial_dispersion: float
    both_conditions_contract: bool
    separated_convergence: bool
    population_gain: bool
    pathway_gating: bool
    horizon_steps: int
    horizon_time: float
    raw_drive: tuple[float, float]
    interpretation: str = (
        "provided_anchor_finite_horizon_fixed_drive_convergence_probe_not_"
        "global_attractor_proof"
    )


def _mean_dispersion(values: FloatArray) -> float:
    centroid = np.mean(values, axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(values - centroid, axis=1)))


def fixed_drive_attractor_probe(
    network: EIRateNetwork,
    initial_x: ArrayLike,
    gain_axis: ArrayLike,
    *,
    gain_strength: float,
    beliefs: tuple[float, float] = (0.1, 0.9),
    raw_drive: tuple[float, float] = (0.25, 0.25),
    horizon_steps: int = 20,
    minimum_separation: float = 1.0,
    minimum_initial_scaled_separation: float = 0.05,
    population_gain: bool = True,
    pathway_gating: bool = True,
) -> FixedDriveAttractorProbe:
    """Replay common anchors under two constant control values."""

    if not isinstance(network, EIRateNetwork) or network.n_inputs != 2:
        raise TypeError("network must be a two-input EIRateNetwork")
    anchors = np.asarray(initial_x, dtype=np.float64)
    if (
        anchors.ndim != 2
        or anchors.shape[0] < 2
        or anchors.shape[1] != network.n_units
        or not np.all(np.isfinite(anchors))
    ):
        raise ValueError("initial_x must contain at least two finite network states")
    axis = np.asarray(gain_axis, dtype=np.float64)
    if (
        axis.shape != (network.n_units,)
        or not np.all(np.isfinite(axis))
        or np.max(np.abs(axis)) > 1.0 + 1e-12
    ):
        raise ValueError("gain_axis must be finite, unit-aligned, and bounded")
    strength = _nonnegative(gain_strength, name="gain_strength")
    if strength >= 1.0:
        raise ValueError("gain_strength must lie in [0, 1)")
    if (
        not isinstance(beliefs, tuple)
        or len(beliefs) != 2
        or not np.all(np.isfinite(beliefs))
        or not all(0.0 <= float(item) <= 1.0 for item in beliefs)
        or float(beliefs[0]) >= float(beliefs[1])
    ):
        raise ValueError("beliefs must contain two increasing probabilities")
    drive = np.asarray(raw_drive, dtype=np.float64)
    if drive.shape != (2,) or not np.all(np.isfinite(drive)):
        raise ValueError("raw_drive must contain two finite values")
    horizon = _positive_int(horizon_steps, name="horizon_steps")
    separation_threshold = _nonnegative(minimum_separation, name="minimum_separation")
    initial_scaled_threshold = _nonnegative(
        minimum_initial_scaled_separation,
        name="minimum_initial_scaled_separation",
    )
    if not isinstance(population_gain, (bool, np.bool_)):
        raise TypeError("population_gain must be boolean")
    if not isinstance(pathway_gating, (bool, np.bool_)):
        raise TypeError("pathway_gating must be boolean")

    initial_dispersion = _mean_dispersion(anchors)
    if initial_dispersion <= np.finfo(np.float64).eps:
        raise ValueError("initial_x must have non-zero anchor dispersion")
    endpoint_sets: list[FloatArray] = []
    contractions: list[float] = []
    for probability in beliefs:
        p = float(probability)
        gain = (
            1.0 + strength * (2.0 * p - 1.0) * axis
            if bool(population_gain)
            else np.ones(network.n_units, dtype=np.float64)
        )
        routed = (
            np.array([(1.0 - p) * drive[0], p * drive[1]])
            if bool(pathway_gating)
            else drive
        )
        endpoints = np.empty_like(anchors)
        for index, anchor in enumerate(anchors):
            state = network.initial_state(anchor, gain=gain)
            for _ in range(horizon):
                state = network.step(state, routed, gain=gain).state
            endpoints[index] = state.x
        endpoint_sets.append(endpoints)
        contractions.append(
            _mean_dispersion(endpoints)
            / max(initial_dispersion, np.finfo(np.float64).eps)
        )
    centroid_low = np.mean(endpoint_sets[0], axis=0)
    centroid_high = np.mean(endpoint_sets[1], axis=0)
    separation = float(np.linalg.norm(centroid_high - centroid_low))
    endpoint_scale = 0.5 * (
        _mean_dispersion(endpoint_sets[0]) + _mean_dispersion(endpoint_sets[1])
    )
    normalized_separation = separation / max(endpoint_scale, np.finfo(np.float64).eps)
    initial_scaled_separation = separation / initial_dispersion
    both_contract = bool(all(item < 1.0 for item in contractions))
    separated = bool(
        both_contract
        and normalized_separation >= separation_threshold
        and initial_scaled_separation >= initial_scaled_threshold
    )
    return FixedDriveAttractorProbe(
        beliefs=(float(beliefs[0]), float(beliefs[1])),
        initial_state_count=int(anchors.shape[0]),
        endpoint_dispersion_contraction=(
            float(contractions[0]),
            float(contractions[1]),
        ),
        endpoint_centroid_separation=separation,
        normalized_endpoint_centroid_separation=float(normalized_separation),
        centroid_separation_over_initial_dispersion=float(initial_scaled_separation),
        both_conditions_contract=both_contract,
        separated_convergence=separated,
        population_gain=bool(population_gain),
        pathway_gating=bool(pathway_gating),
        horizon_steps=horizon,
        horizon_time=float(horizon * network.dt),
        raw_drive=(float(drive[0]), float(drive[1])),
    )


def _future_states(
    network: EIRateNetwork,
    state: EIRateState,
    future_inputs: FloatArray,
    future_gains: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if future_inputs.shape[0] != future_gains.shape[0]:
        raise ValueError("future inputs and gains must align")
    x_history = np.empty(
        (future_inputs.shape[0] + 1, network.n_units), dtype=np.float64
    )
    rate_history = np.empty_like(x_history)
    x_history[0] = state.x
    rate_history[0] = state.rates
    current = state
    for index in range(future_inputs.shape[0]):
        current = network.step(
            current,
            future_inputs[index],
            gain=future_gains[index],
        ).state
        x_history[index + 1] = current.x
        rate_history[index + 1] = current.rates
    return x_history, rate_history


def nonlinear_perturbation_recovery(
    network: EIRateNetwork,
    fitted: FittedTrajectoryKoopman,
    x_trajectory: ArrayLike,
    rate_trajectory: ArrayLike,
    routed_inputs: ArrayLike,
    population_gain_belief: ArrayLike,
    epoch: ArrayLike,
    gain_axis: ArrayLike,
    *,
    gain_strength: float,
    integration_substeps: int,
    horizon_steps: int,
    amplitudes: Sequence[float],
    n_references: int,
    seed: int,
    baseline_replay_tolerance: float = 1e-10,
) -> NonlinearPerturbationSummary:
    """Replay physical-x perturbations around the train-fitted joint subspace.

    The tangent basis is the x-coordinate projection of the same train-only
    joint ``[x, rates]`` PCA used by the reduced operator.  This avoids an
    ill-defined inverse activation derivative at rectified-tanh boundaries.
    Endpoints are projected and scored in physical x coordinates.
    """

    if not isinstance(network, EIRateNetwork):
        raise TypeError("network must be an EIRateNetwork")
    if fitted.n_units != network.n_units or fitted.integration_substeps != int(
        integration_substeps
    ):
        raise ValueError("fitted dynamics and physical replay are incompatible")
    x, rates, inputs, probability, epochs = _trajectory_arrays(
        x_trajectory,
        rate_trajectory,
        routed_inputs,
        population_gain_belief,
        epoch,
        integration_substeps=integration_substeps,
        n_units=network.n_units,
    )
    horizon = _positive_int(horizon_steps, name="horizon_steps")
    references = _positive_int(n_references, name="n_references")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    axis = np.asarray(gain_axis, dtype=np.float64)
    if (
        axis.shape != (network.n_units,)
        or not np.all(np.isfinite(axis))
        or np.max(np.abs(axis)) > 1.0 + 1e-12
    ):
        raise ValueError("gain_axis must be finite, unit-aligned, and bounded by one")
    strength = _nonnegative(gain_strength, name="gain_strength")
    if strength >= 1.0:
        raise ValueError("gain_strength must lie in [0, 1)")
    replay_tolerance = _nonnegative(
        baseline_replay_tolerance, name="baseline_replay_tolerance"
    )
    amplitude_values = tuple(float(item) for item in amplitudes)
    if (
        not amplitude_values
        or not np.all(np.isfinite(amplitude_values))
        or any(item <= 0.0 for item in amplitude_values)
    ):
        raise ValueError("amplitudes must contain positive finite values")

    fine_inputs, fine_probability, _ = _expanded_controls(
        inputs,
        probability,
        epochs,
        integration_substeps=integration_substeps,
    )
    fine_epoch_labels = np.repeat(epochs, int(integration_substeps))
    active = fine_epoch_labels != "cue"
    signed = 2.0 * fine_probability - 1.0
    fine_gains = (
        1.0
        + strength * signed[:, :, None] * active[None, :, None] * axis[None, None, :]
    )
    if np.any(fine_gains <= 0.0):
        raise RuntimeError("validated gains became non-positive")

    tangent_geometry = fitted.physical_x_tangent_basis
    if tangent_geometry.rank != fitted.latent_dim:
        raise PerturbationEligibilityError(
            "physical-x projection of the joint latent basis is rank-deficient"
        )
    q_tangent = tangent_geometry.basis
    if q_tangent.shape != (network.n_units, fitted.latent_dim):
        raise RuntimeError("physical-x tangent basis has an invalid shape")
    projector = q_tangent @ q_tangent.T

    candidates = [
        (sequence, time)
        for sequence in range(rates.shape[0])
        for time in range(rates.shape[1] - horizon)
        if time > 0 and active[time - 1]
    ]
    if not candidates:
        raise PerturbationEligibilityError(
            "no active reference has the requested future horizon"
        )
    rng = np.random.default_rng(int(seed))
    chosen_indices = rng.choice(
        len(candidates), size=min(references, len(candidates)), replace=False
    )
    chosen = [candidates[int(index)] for index in np.atleast_1d(chosen_indices)]

    normal_ratios: list[float] = []
    normal_lyapunov: list[float] = []
    tangent_ratios: list[float] = []
    relative_log_ratios: list[float] = []
    purities: list[float] = []
    tangent_purities: list[float] = []
    reference_worst_ratios: list[float] = []
    ratios_by_amplitude: list[list[float]] = [[] for _ in range(len(amplitude_values))]
    eligible_references = 0
    baseline_replay_max_error = 0.0
    for sequence, time in chosen:
        state_gain = fine_gains[sequence, time - 1]
        random_normal = rng.normal(size=network.n_units)
        normal_direction = random_normal - q_tangent @ (q_tangent.T @ random_normal)
        normal_norm = float(np.linalg.norm(normal_direction))
        if normal_norm <= np.finfo(np.float64).eps:
            continue
        normal_direction /= normal_norm
        tangent_coefficients = rng.normal(size=fitted.latent_dim)
        tangent_direction = q_tangent @ tangent_coefficients
        tangent_norm = float(np.linalg.norm(tangent_direction))
        if tangent_norm <= np.finfo(np.float64).eps:
            continue
        tangent_direction /= tangent_norm

        baseline_state = EIRateState(
            x=np.array(x[sequence, time], dtype=np.float64, copy=True),
            rates=np.array(rates[sequence, time], dtype=np.float64, copy=True),
        )
        future_input = fine_inputs[sequence, time : time + horizon]
        future_gain = fine_gains[sequence, time : time + horizon]
        baseline_x, baseline_rates = _future_states(
            network, baseline_state, future_input, future_gain
        )
        replay_error = max(
            float(np.max(np.abs(baseline_x - x[sequence, time : time + horizon + 1]))),
            float(
                np.max(
                    np.abs(baseline_rates - rates[sequence, time : time + horizon + 1])
                )
            ),
        )
        baseline_replay_max_error = max(baseline_replay_max_error, replay_error)
        if replay_error > replay_tolerance:
            raise RuntimeError(
                "baseline replay differs from the recorded frozen trajectory"
            )
        reference_normal_ratios: list[float] = []
        reference_tangent_ratios: list[float] = []
        reference_relative_ratios: list[float] = []
        reference_normal_purities: list[float] = []
        reference_tangent_purities: list[float] = []
        reference_by_amplitude: list[list[float]] = [
            [] for _ in range(len(amplitude_values))
        ]
        for amplitude_index, amplitude in enumerate(amplitude_values):
            for sign in (-1.0, 1.0):
                normal_delta = sign * amplitude * normal_direction
                initial_normal = normal_delta - projector @ normal_delta
                initial_normal_norm = float(np.linalg.norm(initial_normal))
                initial_total_norm = float(np.linalg.norm(normal_delta))
                normal_purity = initial_normal_norm / max(
                    initial_total_norm, np.finfo(np.float64).eps
                )
                normal_x = baseline_state.x + normal_delta
                normal_rate = network.initial_state(normal_x, gain=state_gain).rates
                normal_history_x, _ = _future_states(
                    network,
                    EIRateState(x=normal_x, rates=normal_rate),
                    future_input,
                    future_gain,
                )
                endpoint_delta = normal_history_x[-1] - baseline_x[-1]
                endpoint_normal = endpoint_delta - projector @ endpoint_delta
                normal_ratio = float(
                    np.linalg.norm(endpoint_normal) / initial_normal_norm
                )
                tangent_delta = sign * amplitude * tangent_direction
                tangent_initial = projector @ tangent_delta
                tangent_initial_norm = float(np.linalg.norm(tangent_initial))
                tangent_total_norm = float(np.linalg.norm(tangent_delta))
                tangent_purity = tangent_initial_norm / max(
                    tangent_total_norm, np.finfo(np.float64).eps
                )
                tangent_x = baseline_state.x + tangent_delta
                tangent_rate = network.initial_state(tangent_x, gain=state_gain).rates
                tangent_history_x, _ = _future_states(
                    network,
                    EIRateState(x=tangent_x, rates=tangent_rate),
                    future_input,
                    future_gain,
                )
                tangent_endpoint_delta = tangent_history_x[-1] - baseline_x[-1]
                tangent_endpoint = projector @ tangent_endpoint_delta
                tangent_ratio = float(
                    np.linalg.norm(tangent_endpoint) / tangent_initial_norm
                )
                if not (
                    np.isfinite(normal_ratio)
                    and np.isfinite(tangent_ratio)
                    and initial_normal_norm > np.finfo(np.float64).eps
                    and tangent_initial_norm > np.finfo(np.float64).eps
                ):
                    continue
                reference_normal_ratios.append(normal_ratio)
                reference_tangent_ratios.append(tangent_ratio)
                reference_relative_ratios.append(
                    float(
                        np.log(
                            max(normal_ratio, np.finfo(np.float64).tiny)
                            / max(tangent_ratio, np.finfo(np.float64).tiny)
                        )
                    )
                )
                reference_normal_purities.append(normal_purity)
                reference_tangent_purities.append(tangent_purity)
                reference_by_amplitude[amplitude_index].append(normal_ratio)
        expected = 2 * len(amplitude_values)
        if (
            len(reference_normal_ratios) == expected
            and len(reference_tangent_ratios) == expected
        ):
            eligible_references += 1
            normal_ratios.extend(reference_normal_ratios)
            tangent_ratios.extend(reference_tangent_ratios)
            relative_log_ratios.extend(reference_relative_ratios)
            purities.extend(reference_normal_purities)
            tangent_purities.extend(reference_tangent_purities)
            reference_worst_ratios.append(max(reference_normal_ratios))
            normal_lyapunov.extend(
                float(
                    np.log(max(ratio, np.finfo(np.float64).tiny))
                    / (horizon * network.dt)
                )
                for ratio in reference_normal_ratios
            )
            for amplitude_index, values in enumerate(reference_by_amplitude):
                ratios_by_amplitude[amplitude_index].extend(values)

    sampled = len(chosen)
    sampled_fraction = sampled / references
    eligible_sampled_fraction = eligible_references / sampled if sampled else 0.0
    eligible_planned_fraction = eligible_references / references
    amplitude_medians = tuple(
        float(np.median(values)) if values else None for values in ratios_by_amplitude
    )
    if not normal_ratios:
        return NonlinearPerturbationSummary(
            tangent_basis_space="train_joint_state_pca_physical_x_projection",
            tangent_basis_rank=tangent_geometry.rank,
            tangent_basis_singular_values=tuple(
                float(item) for item in tangent_geometry.singular_values
            ),
            tangent_basis_condition_number=tangent_geometry.condition_number,
            tangent_basis_x_block_energy_fraction=(
                tangent_geometry.x_block_energy_fraction
            ),
            sampled_reference_fraction=sampled_fraction,
            eligible_sampled_reference_fraction=eligible_sampled_fraction,
            eligible_reference_fraction=eligible_planned_fraction,
            eligible_reference_count=eligible_references,
            planned_reference_count=references,
            candidate_reference_count=len(candidates),
            sampled_reference_count=sampled,
            normal_perturbation_count=0,
            normal_recovery_fraction=None,
            normal_endpoint_ratio_median=None,
            normal_endpoint_ratio_maximum=None,
            maximum_projected_finite_time_normal_log_growth_rate=None,
            tangent_endpoint_ratio_median=(
                float(np.median(tangent_ratios)) if tangent_ratios else None
            ),
            normal_vs_tangent_log_ratio_median=(
                float(np.median(relative_log_ratios)) if relative_log_ratios else None
            ),
            initial_normal_purity_median=None,
            initial_tangent_purity_median=(
                float(np.median(tangent_purities)) if tangent_purities else None
            ),
            amplitudes=amplitude_values,
            normal_endpoint_ratio_by_amplitude=amplitude_medians,
            baseline_replay_max_abs_error=baseline_replay_max_error,
            baseline_replay_tolerance=replay_tolerance,
            horizon_steps=horizon,
            horizon_time=float(horizon * network.dt),
        )
    return NonlinearPerturbationSummary(
        tangent_basis_space="train_joint_state_pca_physical_x_projection",
        tangent_basis_rank=tangent_geometry.rank,
        tangent_basis_singular_values=tuple(
            float(item) for item in tangent_geometry.singular_values
        ),
        tangent_basis_condition_number=tangent_geometry.condition_number,
        tangent_basis_x_block_energy_fraction=(
            tangent_geometry.x_block_energy_fraction
        ),
        sampled_reference_fraction=sampled_fraction,
        eligible_sampled_reference_fraction=eligible_sampled_fraction,
        eligible_reference_fraction=eligible_planned_fraction,
        eligible_reference_count=eligible_references,
        planned_reference_count=references,
        candidate_reference_count=len(candidates),
        sampled_reference_count=sampled,
        normal_perturbation_count=len(normal_ratios),
        normal_recovery_fraction=float(
            np.mean(np.asarray(reference_worst_ratios) < 1.0)
        ),
        normal_endpoint_ratio_median=float(np.median(reference_worst_ratios)),
        normal_endpoint_ratio_maximum=float(np.max(reference_worst_ratios)),
        maximum_projected_finite_time_normal_log_growth_rate=float(
            np.max(normal_lyapunov)
        ),
        tangent_endpoint_ratio_median=(
            float(np.median(tangent_ratios)) if tangent_ratios else None
        ),
        normal_vs_tangent_log_ratio_median=(
            float(np.median(relative_log_ratios)) if relative_log_ratios else None
        ),
        initial_normal_purity_median=float(np.median(purities)),
        initial_tangent_purity_median=(
            float(np.median(tangent_purities)) if tangent_purities else None
        ),
        amplitudes=amplitude_values,
        normal_endpoint_ratio_by_amplitude=amplitude_medians,
        baseline_replay_max_abs_error=baseline_replay_max_error,
        baseline_replay_tolerance=replay_tolerance,
        horizon_steps=horizon,
        horizon_time=float(horizon * network.dt),
    )


__all__ = [
    "BeliefManifoldGeometry",
    "FixedDriveAttractorProbe",
    "FittedTrajectoryKoopman",
    "NonlinearPerturbationSummary",
    "PerturbationEligibilityError",
    "PhysicalXTangentBasis",
    "TrajectoryKoopmanScore",
    "belief_manifold_geometry",
    "fixed_drive_attractor_probe",
    "fit_trajectory_koopman",
    "nonlinear_perturbation_recovery",
    "persistence_trajectory_score",
]
