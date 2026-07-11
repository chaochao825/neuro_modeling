"""Validate rank stages and low-dimensional credit on a shared E/I substrate."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.dynamical_dimension import (
    empirical_hankel_summary,
    fit_hankel_noise_floor,
    fit_hankel_preprocessor,
    jacobian_outlier_summary,
)
from src.analysis.rank_metrics import participation_ratio
from src.analysis.rank_stage_metrics import (
    CreditTangentSummary,
    MaskedOuterProductIdentitySummary,
    UpdateStageRankSummary,
    credit_tangent_summary,
    masked_outer_product_identity,
    matrix_rank_summary,
    update_stage_rank_summary,
)
from src.models.ei_rate_network import EIRateNetwork
from src.plasticity.parameterizations import (
    DirectAdditivePlasticity,
    FullPerSynapsePlasticity,
    ParameterizedPlasticityUpdate,
    SignPreservingMultiplicativePlasticity,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed, make_rng


Array = np.ndarray
PARAMETERIZATIONS = frozenset({"direct", "multiplicative", "full-per-synapse"})
STAGE_NAMES = (
    "hebbian",
    "decay",
    "raw",
    "masked",
    "dale_applied",
    "normalization_correction",
    "total",
)


def _fingerprint(*arrays: Array) -> str:
    hasher = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def _combined_fingerprint(*values: str) -> str:
    hasher = hashlib.sha256()
    for value in values:
        encoded = value.encode("ascii")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    return hasher.hexdigest()


@dataclass(frozen=True)
class RankStageCondition:
    """One parameterization, feedback dimension, and alignment-angle cell."""

    condition: str
    parameterization: str
    requested_feedback_dim: int | str
    feedback_angle_degrees: float
    geometry_valid: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dimension_label(value: int | str) -> str:
    return str(value) if isinstance(value, int) else value


def _angle_label(value: float) -> str:
    return (
        str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
    )


def build_rank_stage_conditions(config: Mapping[str, Any]) -> list[RankStageCondition]:
    """Build the complete stable P1 grid, including invalid full-angle cells."""

    raw_dimensions = list(config["feedback_dims"])
    if not raw_dimensions:
        raise ValueError("feedback_dims must be non-empty")
    dimensions: list[int | str] = []
    for value in raw_dimensions:
        if isinstance(value, bool):
            raise TypeError("feedback dimensions cannot be boolean")
        if isinstance(value, str):
            if value != "full":
                raise ValueError("string feedback dimensions must equal 'full'")
            normalized: int | str = "full"
        else:
            normalized = int(value)
            if normalized <= 0:
                raise ValueError("integer feedback dimensions must be positive")
        if normalized in dimensions:
            raise ValueError("feedback_dims cannot contain duplicates")
        dimensions.append(normalized)

    angles = [float(value) for value in config["feedback_angles_degrees"]]
    if not angles or len(set(angles)) != len(angles):
        raise ValueError("feedback_angles_degrees must be non-empty and unique")
    if any(not np.isfinite(value) or not 0.0 <= value <= 90.0 for value in angles):
        raise ValueError("feedback angles must lie in [0, 90]")

    parameterizations = [str(value) for value in config["parameterizations"]]
    if (
        not parameterizations
        or len(set(parameterizations)) != len(parameterizations)
        or any(value not in PARAMETERIZATIONS for value in parameterizations)
    ):
        raise ValueError(
            f"parameterizations must be unique members of {sorted(PARAMETERIZATIONS)}"
        )

    conditions = []
    for dimension in dimensions:
        for angle in angles:
            for parameterization in parameterizations:
                valid = not (dimension == "full" and angle != 0.0)
                conditions.append(
                    RankStageCondition(
                        condition=(
                            f"{parameterization}__feedback-{_dimension_label(dimension)}"
                            f"__angle-{_angle_label(angle)}"
                        ),
                        parameterization=parameterization,
                        requested_feedback_dim=dimension,
                        feedback_angle_degrees=angle,
                        geometry_valid=valid,
                    )
                )
    return conditions


@dataclass(frozen=True)
class SharedRankStageResources:
    """Seed-level resources reused unchanged by every valid grid cell."""

    network_seed: int
    initial_weights: Array
    connectivity_mask: Array
    presynaptic_signs: Array
    task_basis: Array
    orthogonal_complement: Array
    channel_values: Array
    update_state_x: Array
    post_derivative: Array
    eligibility_trace: Array
    edge_modulation: Array
    train_initial_states: Array
    train_inputs: Array
    train_noise: Array
    test_initial_states: Array
    test_inputs: Array
    test_noise: Array
    hankel_feature_indices: Array
    initialization_id: str
    mask_id: str
    state_id: str
    noise_id: str
    trajectory_tape_id: str
    feedback_frame_id: str
    edge_modulation_id: str
    hankel_feature_id: str
    shared_resource_id: str


def _network_options(architecture: Mapping[str, Any]) -> dict[str, Any]:
    if architecture.get("kind") != "ei":
        raise ValueError("exp08 requires an E/I architecture")
    allowed = {
        "n_units",
        "n_inputs",
        "excitatory_fraction",
        "connection_probability",
        "tau_e",
        "tau_i",
        "dt",
        "bulk_gain",
        "inhibitory_gain",
        "input_scale",
        "activation",
        "allow_self_connections",
        "normalize_fan_in_after_update",
    }
    return {key: architecture[key] for key in allowed if key in architecture}


def _make_network(
    architecture: Mapping[str, Any], *, network_seed: int
) -> EIRateNetwork:
    return EIRateNetwork(**_network_options(architecture), seed=network_seed)


def prepare_shared_resources(
    config: Mapping[str, Any], *, seed: int
) -> SharedRankStageResources:
    """Create the paired initialization, state, trajectory, and noise tapes."""

    architecture = dict(config["architecture"])
    trajectory = dict(config["trajectory"])
    network_seed = derive_seed(seed, "exp08-network")
    network = _make_network(architecture, network_seed=network_seed)
    n_units = network.n_units
    latent_dim = int(config["latent_dim"])
    if not 1 <= latent_dim <= n_units // 2:
        raise ValueError("latent_dim must lie in [1, n_units // 2]")

    rng = make_rng(seed, "exp08-shared-resources")
    frame, _ = np.linalg.qr(rng.normal(size=(n_units, n_units)))
    task_basis = frame[:, :latent_dim]
    complement = frame[:, latent_dim:]
    channel_values = rng.normal(size=n_units)
    update_state_x = rng.uniform(0.05, 0.5, size=n_units)
    post_derivative = 1.0 - np.tanh(update_state_x) ** 2
    if network.activation_name == "rectified_tanh":
        post_derivative *= update_state_x > 0.0
    state_rates = network.initial_state(update_state_x).rates
    eligibility = 0.05 + state_rates + np.abs(rng.normal(scale=0.02, size=n_units))
    edge_modulation = rng.normal(size=(n_units, n_units))
    connected_scale = float(np.std(edge_modulation[network.connectivity_mask]))
    if not np.isfinite(connected_scale) or connected_scale <= 0.0:
        raise FloatingPointError(
            "edge modulation requires at least two variable connected entries"
        )
    edge_modulation /= connected_scale

    n_inputs = network.n_inputs
    if n_inputs <= 0:
        raise ValueError("exp08 trajectory simulation requires n_inputs > 0")
    time_steps = int(trajectory["time_steps"])
    train_trials = int(trajectory["train_trials"])
    test_trials = int(trajectory["test_trials"])
    if min(time_steps, train_trials, test_trials) <= 0:
        raise ValueError("trajectory counts and time_steps must be positive")
    input_scale = float(trajectory.get("input_scale", 0.5))
    noise_std = float(trajectory.get("noise_std", 0.0))
    initial_scale = float(trajectory.get("initial_state_scale", 0.1))
    scales = np.asarray([input_scale, noise_std, initial_scale])
    if not np.all(np.isfinite(scales)) or np.any(scales < 0.0):
        raise ValueError("trajectory scales must be finite and non-negative")

    train_initial = rng.normal(scale=initial_scale, size=(train_trials, n_units))
    test_initial = rng.normal(scale=initial_scale, size=(test_trials, n_units))
    train_inputs = rng.normal(
        scale=input_scale, size=(train_trials, time_steps, n_inputs)
    )
    test_inputs = rng.normal(
        scale=input_scale, size=(test_trials, time_steps, n_inputs)
    )
    train_noise = rng.normal(scale=noise_std, size=(train_trials, time_steps, n_units))
    test_noise = rng.normal(scale=noise_std, size=(test_trials, time_steps, n_units))

    hankel_feature_count = int(config["hankel"]["feature_count"])
    if not 1 <= hankel_feature_count <= n_units:
        raise ValueError("hankel feature_count must lie in [1, n_units]")
    hankel_indices = np.sort(
        rng.choice(n_units, size=hankel_feature_count, replace=False)
    )

    initialization_id = _fingerprint(
        network.recurrent_weights,
        network.input_weights,
        network.time_constants,
        network.presynaptic_signs,
    )
    mask_id = _fingerprint(network.connectivity_mask.astype(np.uint8))
    state_id = _fingerprint(
        update_state_x, post_derivative, eligibility, channel_values
    )
    noise_id = _fingerprint(train_noise, test_noise)
    trajectory_tape_id = _fingerprint(
        train_initial, train_inputs, test_initial, test_inputs
    )
    feedback_frame_id = _fingerprint(task_basis, complement)
    edge_modulation_id = _fingerprint(edge_modulation)
    hankel_feature_id = _fingerprint(hankel_indices)
    shared_resource_id = _combined_fingerprint(
        initialization_id,
        mask_id,
        state_id,
        noise_id,
        trajectory_tape_id,
        feedback_frame_id,
        edge_modulation_id,
        hankel_feature_id,
    )
    return SharedRankStageResources(
        network_seed=network_seed,
        initial_weights=network.recurrent_weights,
        connectivity_mask=network.connectivity_mask.copy(),
        presynaptic_signs=network.presynaptic_signs.copy(),
        task_basis=task_basis,
        orthogonal_complement=complement,
        channel_values=channel_values,
        update_state_x=update_state_x,
        post_derivative=post_derivative,
        eligibility_trace=eligibility,
        edge_modulation=edge_modulation,
        train_initial_states=train_initial,
        train_inputs=train_inputs,
        train_noise=train_noise,
        test_initial_states=test_initial,
        test_inputs=test_inputs,
        test_noise=test_noise,
        hankel_feature_indices=hankel_indices,
        initialization_id=initialization_id,
        mask_id=mask_id,
        state_id=state_id,
        noise_id=noise_id,
        trajectory_tape_id=trajectory_tape_id,
        feedback_frame_id=feedback_frame_id,
        edge_modulation_id=edge_modulation_id,
        hankel_feature_id=hankel_feature_id,
        shared_resource_id=shared_resource_id,
    )


def construct_feedback_basis(
    task_basis: Array,
    orthogonal_complement: Array,
    requested_dim: int | str,
    angle_degrees: float,
) -> tuple[Array, dict[str, Any]]:
    """Construct an orthonormal feedback basis with explicit principal angle."""

    task = np.asarray(task_basis, dtype=float)
    complement = np.asarray(orthogonal_complement, dtype=float)
    n_units, latent_dim = task.shape
    if complement.ndim != 2 or complement.shape != (n_units, n_units - latent_dim):
        raise ValueError("orthogonal_complement shape does not match task_basis")
    angle = float(angle_degrees)
    if requested_dim == "full":
        if angle != 0.0:
            raise ValueError(
                "full feedback spans the ambient space; nonzero subspace angle is undefined"
            )
        # Keep the full condition in the same seed-level coordinate frame as
        # the reduced conditions.  Its first latent_dim channels are therefore
        # exactly the aligned task channels, followed by the fixed complement.
        basis = np.column_stack([task, complement])
        actual_dim = n_units
    else:
        actual_dim = int(requested_dim)
        if not 1 <= actual_dim <= complement.shape[1]:
            raise ValueError(
                "non-full feedback dimension must not exceed the orthogonal complement"
            )
        radians = np.radians(angle)
        rotated_count = min(actual_dim, latent_dim)
        rotated = (
            np.cos(radians) * task[:, :rotated_count]
            + np.sin(radians) * complement[:, :rotated_count]
        )
        extra_count = actual_dim - rotated_count
        extras = complement[:, rotated_count : rotated_count + extra_count]
        basis = np.column_stack([rotated, extras]) if extra_count else rotated
    if not np.allclose(basis.T @ basis, np.eye(actual_dim), atol=1e-10, rtol=0.0):
        raise FloatingPointError("constructed feedback basis is not orthonormal")
    cosines = np.linalg.svd(task.T @ basis, compute_uv=False)
    principal_angles = np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))
    alignment_fraction = float(np.sum((task.T @ basis) ** 2) / actual_dim)
    return basis, {
        "actual_feedback_dim": int(actual_dim),
        "feedback_alignment_fraction": alignment_fraction,
        "feedback_principal_angles_degrees": principal_angles.tolist(),
        "feedback_geometry_definition": (
            "task_axes_rotated_into_fixed_orthogonal_complement_with_extra_nuisance_axes"
            if requested_dim != "full"
            else "full_ambient_basis"
        ),
    }


def _parameterized_update(
    condition: RankStageCondition,
    *,
    post_factor: Array,
    resources: SharedRankStageResources,
    learning_rate: float,
    max_abs_log_step: float | None,
    actual_feedback_dim: int,
) -> ParameterizedPlasticityUpdate:
    common = {
        "current_weights": resources.initial_weights,
        "connectivity_mask": resources.connectivity_mask,
        "presynaptic_signs": resources.presynaptic_signs,
    }
    if condition.parameterization == "direct":
        return DirectAdditivePlasticity(
            learning_rate=learning_rate,
            credit_dimension=actual_feedback_dim,
        ).propose(post_factor, resources.eligibility_trace, **common)
    if condition.parameterization == "multiplicative":
        return SignPreservingMultiplicativePlasticity(
            learning_rate=learning_rate,
            credit_dimension=actual_feedback_dim,
            max_abs_log_step=max_abs_log_step,
        ).propose(post_factor, resources.eligibility_trace, **common)
    if condition.parameterization == "full-per-synapse":
        synaptic_third_factor = post_factor[:, None] * resources.edge_modulation
        return FullPerSynapsePlasticity(learning_rate=learning_rate).propose(
            synaptic_third_factor,
            resources.eligibility_trace,
            **common,
        )
    raise ValueError(f"unsupported parameterization: {condition.parameterization}")


def _applied_weight_before_dale(
    update: ParameterizedPlasticityUpdate,
    current_weights: Array,
) -> Array:
    """Map the actually applied control into weight space before Dale projection."""

    if update.control_space == "weight":
        return np.asarray(update.applied_control_update, dtype=float).copy()
    if update.control_space == "log_magnitude":
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(current_weights, dtype=float) * np.expm1(
                update.applied_control_update
            )
    raise ValueError(f"unsupported control space: {update.control_space}")


def _simulate_trials(
    network: EIRateNetwork,
    initial_states: Array,
    inputs: Array,
    noise: Array,
) -> tuple[Array, Array]:
    if (
        inputs.shape[:2] != noise.shape[:2]
        or initial_states.shape[0] != inputs.shape[0]
    ):
        raise ValueError("trajectory tape trial/time dimensions do not match")
    n_trials, n_steps = inputs.shape[:2]
    x_history = np.empty((n_trials, n_steps, network.n_units), dtype=float)
    rate_history = np.empty_like(x_history)
    for trial in range(n_trials):
        state = network.initial_state(initial_states[trial])
        for time in range(n_steps):
            state = network.step(
                state,
                inputs[trial, time],
                noise=noise[trial, time],
            ).state
            x_history[trial, time] = state.x
            rate_history[trial, time] = state.rates
    return x_history, rate_history


def _jacobian(network: EIRateNetwork, weights: Array, mean_x: Array) -> Array:
    derivative = 1.0 - np.tanh(mean_x) ** 2
    if network.activation_name == "rectified_tanh":
        derivative *= mean_x > 0.0
    jacobian = -np.eye(network.n_units) + weights @ np.diag(derivative)
    return jacobian / network.time_constants[:, None]


def _stage_rank_metrics(summary: UpdateStageRankSummary) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        item = getattr(summary, stage)
        metrics[f"{stage}_numerical_rank"] = item.numerical_rank
        metrics[f"{stage}_effective_rank"] = item.effective_rank
        details[stage] = {
            "numerical_rank": item.numerical_rank,
            "effective_rank": item.effective_rank,
            "threshold": item.threshold,
            "singular_values": item.singular_values.tolist(),
        }
    metrics["stage_rank_details"] = details
    return metrics


def _identity_metrics(summary: MaskedOuterProductIdentitySummary) -> dict[str, Any]:
    details = {}
    for name, item in {
        "raw_outer": summary.raw_outer,
        "mask": summary.mask,
        "masked_outer": summary.masked_outer,
        "diagonal_form": summary.diagonal_form,
    }.items():
        details[name] = {
            "numerical_rank": item.numerical_rank,
            "effective_rank": item.effective_rank,
            "threshold": item.threshold,
            "singular_values": item.singular_values.tolist(),
        }
    return {
        "masked_identity_equal": summary.equal,
        "masked_identity_max_abs_residual": summary.max_abs_residual,
        "masked_identity_exact_rank_preservation_expected": (
            summary.exact_rank_preservation_expected
        ),
        "masked_identity_numerically_preserves_mask_rank": (
            summary.numerically_preserves_mask_rank
        ),
        "masked_identity_left_condition_number": (
            summary.left_diagonal_condition_number
        ),
        "masked_identity_right_condition_number": (
            summary.right_diagonal_condition_number
        ),
        "identity_raw_outer_numerical_rank": summary.raw_outer.numerical_rank,
        "identity_mask_numerical_rank": summary.mask.numerical_rank,
        "identity_masked_numerical_rank": summary.masked_outer.numerical_rank,
        "identity_diagonal_form_numerical_rank": (summary.diagonal_form.numerical_rank),
        "identity_masked_effective_rank": summary.masked_outer.effective_rank,
        "masked_identity_rank_details": details,
        "masked_identity_scope": "shared_lowdim_outer_credit_reference",
    }


def _tangent_metrics(summary: CreditTangentSummary) -> dict[str, Any]:
    return {
        "lowdim_credit_tangent_dimension": summary.numerical_dimension,
        "lowdim_credit_tangent_effective_dimension": summary.effective_dimension,
        "lowdim_credit_tangent_threshold": summary.threshold,
        "lowdim_credit_tangent_singular_values": summary.singular_values.tolist(),
        "lowdim_credit_tangent_feedback_dim": summary.feedback_dim,
        "lowdim_credit_tangent_active_synapses": summary.n_active_synapses,
        "lowdim_credit_tangent_stage": summary.stage,
    }


def _parameterization_rank_metrics(
    update: ParameterizedPlasticityUpdate,
    applied_weight_pre_dale: Array,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Report control-coordinate and physical-weight ranks side by side."""

    arrays = {
        "raw_control": update.raw_control_update,
        "masked_control": update.masked_control_update,
        "applied_control": update.applied_control_update,
        "raw_weight": update.raw_weight_update,
        "masked_weight": update.masked_weight_update,
        "applied_weight_pre_dale": applied_weight_pre_dale,
        "dale_applied_weight": update.dale_applied_update,
    }
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for name, array in arrays.items():
        rank = matrix_rank_summary(array, rtol=rtol, atol=atol)
        metrics[f"parameterization_{name}_numerical_rank"] = rank.numerical_rank
        metrics[f"parameterization_{name}_effective_rank"] = rank.effective_rank
        details[name] = {
            "numerical_rank": rank.numerical_rank,
            "effective_rank": rank.effective_rank,
            "threshold": rank.threshold,
            "singular_values": rank.singular_values.tolist(),
        }
    metrics["parameterization_rank_details"] = details
    return metrics


def evaluate_rank_stage_condition(
    config: Mapping[str, Any],
    resources: SharedRankStageResources,
    condition: RankStageCondition,
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one valid P1 grid cell against the shared seed-level tape."""

    if not condition.geometry_valid:
        raise ValueError("invalid geometry must be retained before evaluation")
    architecture = dict(config["architecture"])
    update_options = dict(config["update"])
    rank_options = dict(config.get("rank", {}))
    hankel_options = dict(config["hankel"])
    network = _make_network(architecture, network_seed=resources.network_seed)
    if not np.array_equal(network.recurrent_weights, resources.initial_weights):
        raise RuntimeError("cell initialization differs from shared W0")
    if not np.array_equal(network.connectivity_mask, resources.connectivity_mask):
        raise RuntimeError("cell connectivity mask differs from shared mask")

    basis, geometry = construct_feedback_basis(
        resources.task_basis,
        resources.orthogonal_complement,
        condition.requested_feedback_dim,
        condition.feedback_angle_degrees,
    )
    actual_feedback_dim = int(geometry["actual_feedback_dim"])
    channels = resources.channel_values[:actual_feedback_dim] / np.sqrt(
        actual_feedback_dim
    )
    modulator = basis @ channels
    post_factor = resources.post_derivative * modulator
    feedback_basis_id = _fingerprint(basis)
    feedback_channel_prefix_id = _fingerprint(
        resources.channel_values[:actual_feedback_dim]
    )
    feedback_modulator_id = _fingerprint(modulator)
    learning_rate = float(update_options["learning_rate"])
    max_abs_log_step = update_options.get("max_abs_log_step", 0.1)
    if max_abs_log_step is not None:
        max_abs_log_step = float(max_abs_log_step)
    parameterized = _parameterized_update(
        condition,
        post_factor=post_factor,
        resources=resources,
        learning_rate=learning_rate,
        max_abs_log_step=max_abs_log_step,
        actual_feedback_dim=actual_feedback_dim,
    )
    rank_rtol = float(rank_options.get("rtol", 1e-8))
    rank_atol = float(rank_options.get("atol", 1e-12))
    applied_weight_pre_dale = _applied_weight_before_dale(
        parameterized,
        resources.initial_weights,
    )
    application = network.apply_task_update(parameterized.dale_applied_update)
    if not np.allclose(
        application.local_update,
        parameterized.dale_applied_update,
        rtol=rank_rtol,
        atol=rank_atol,
    ):
        raise RuntimeError("network Dale application changed the preprojected update")
    control_bound_correction = (
        applied_weight_pre_dale - parameterized.masked_weight_update
    )
    control_bound_rank = matrix_rank_summary(
        control_bound_correction,
        rtol=rank_rtol,
        atol=rank_atol,
    )
    dale_projection_correction = application.local_update - applied_weight_pre_dale
    dale_projection_rank = matrix_rank_summary(
        dale_projection_correction,
        rtol=rank_rtol,
        atol=rank_atol,
    )
    if not np.allclose(
        application.local_update,
        applied_weight_pre_dale + dale_projection_correction,
        rtol=rank_rtol,
        atol=rank_atol,
    ):
        raise RuntimeError(
            "applied pre-Dale weight and Dale correction do not reconstruct local update"
        )
    control_bound_changed_count = int(
        np.count_nonzero(
            resources.connectivity_mask & (np.abs(control_bound_correction) > rank_atol)
        )
    )
    connected_count = int(np.count_nonzero(resources.connectivity_mask))
    dale_changed_count = int(
        np.count_nonzero(
            resources.connectivity_mask
            & (np.abs(dale_projection_correction) > rank_atol)
        )
    )
    dale_candidate = resources.initial_weights + application.local_update
    dale_boundary_count = int(
        np.count_nonzero(
            resources.connectivity_mask & (np.abs(dale_candidate) <= rank_atol)
        )
    )
    zero_decay = np.zeros_like(resources.initial_weights)
    stage_summary = update_stage_rank_summary(
        hebbian_update=parameterized.raw_weight_update,
        decay_update=zero_decay,
        raw_update=parameterized.raw_weight_update,
        masked_update=parameterized.masked_weight_update,
        dale_applied_update=application.local_update,
        normalization_correction=application.normalization_correction,
        total_update=application.total_update,
        rtol=rank_rtol,
        atol=rank_atol,
    )
    identity = masked_outer_product_identity(
        resources.connectivity_mask,
        post_factor,
        resources.eligibility_trace,
        rtol=rank_rtol,
        atol=rank_atol,
    )
    tangent_scale: Array | None
    if condition.parameterization == "direct":
        tangent_scale = None
        tangent_stage = "masked_additive"
    elif condition.parameterization == "multiplicative":
        tangent_scale = resources.initial_weights
        tangent_stage = "masked_log_magnitude_linearization"
    else:
        tangent_scale = resources.edge_modulation
        tangent_stage = "lowdim_factor_through_full_synapse_gain"
    tangent = credit_tangent_summary(
        basis,
        resources.eligibility_trace,
        post_derivative=resources.post_derivative,
        connectivity_mask=resources.connectivity_mask,
        synaptic_scale=tangent_scale,
        stage=tangent_stage,
        rtol=float(rank_options.get("tangent_rtol", 1e-8)),
        atol=rank_atol,
        edge_chunk_size=int(rank_options.get("edge_chunk_size", 8192)),
    )

    train_x, train_rates = _simulate_trials(
        network,
        resources.train_initial_states,
        resources.train_inputs,
        resources.train_noise,
    )
    test_x, test_rates = _simulate_trials(
        network,
        resources.test_initial_states,
        resources.test_inputs,
        resources.test_noise,
    )
    mean_test_x = np.mean(test_x, axis=(0, 1))
    full_jacobian = _jacobian(network, network.recurrent_weights, mean_test_x)
    paired_bulk_jacobian = _jacobian(network, resources.initial_weights, mean_test_x)
    outliers = jacobian_outlier_summary(
        full_jacobian,
        paired_bulk_jacobian,
        edge_quantile=float(config.get("jacobian_edge_quantile", 0.99)),
        tolerance=float(config.get("jacobian_tolerance", 1e-9)),
    )

    feature_indices = resources.hankel_feature_indices
    train_hankel_activity = train_rates[:, :, feature_indices]
    test_hankel_activity = test_rates[:, :, feature_indices]
    preprocessor = fit_hankel_preprocessor(
        train_hankel_activity,
        normalize=bool(hankel_options.get("normalize", True)),
    )
    noise_floor = fit_hankel_noise_floor(
        train_hankel_activity,
        past_lags=int(hankel_options["past_lags"]),
        future_lags=int(hankel_options["future_lags"]),
        preprocessor=preprocessor,
        n_permutations=int(hankel_options.get("n_permutations", 100)),
        quantile=float(hankel_options.get("null_quantile", 0.95)),
        seed=derive_seed(seed, "exp08-hankel-null"),
    )
    hankel = empirical_hankel_summary(
        test_hankel_activity,
        past_lags=int(hankel_options["past_lags"]),
        future_lags=int(hankel_options["future_lags"]),
        preprocessor=preprocessor,
        noise_floor=noise_floor,
        rtol=float(hankel_options.get("rtol", 1e-8)),
        atol=float(hankel_options.get("atol", 1e-12)),
        window_chunk_size=int(hankel_options.get("window_chunk_size", 256)),
    )
    train_activity_mean = np.mean(train_rates, axis=(0, 1), keepdims=True)
    heldout_centered_activity = (test_rates - train_activity_mean).reshape(
        -1, network.n_units
    )

    metrics = {
        "status": "complete",
        "profile": config.get("profile", "unspecified"),
        "training_algorithm": "one_step_local_rank_stage_audit",
        "used_autograd": False,
        "statistics_unit": "seed",
        **geometry,
        "feedback_basis_id": feedback_basis_id,
        "feedback_channel_prefix_id": feedback_channel_prefix_id,
        "feedback_modulator_id": feedback_modulator_id,
        **_identity_metrics(identity),
        **_stage_rank_metrics(stage_summary),
        **_parameterization_rank_metrics(
            parameterized,
            applied_weight_pre_dale,
            rtol=rank_rtol,
            atol=rank_atol,
        ),
        **_tangent_metrics(tangent),
        "parameterization_credit_dimension": parameterized.credit_dimension,
        "parameterization_implementation": parameterized.parameterization,
        "parameterization_control_tangent_dimension": (
            parameterized.credit_dimension
            if condition.parameterization == "full-per-synapse"
            else tangent.numerical_dimension
        ),
        "parameterization_control_tangent_definition": (
            "independent_connected_edge_controls_before_Dale_projection"
            if condition.parameterization == "full-per-synapse"
            else "fixed_state_lowdim_feedback_image_before_Dale_projection"
        ),
        "lowdim_tangent_is_full_parameterization_tangent": (
            condition.parameterization != "full-per-synapse"
        ),
        "realized_control_slice_dimension": actual_feedback_dim,
        "full_per_synapse_control_space_exhaustively_sampled": (
            False if condition.parameterization == "full-per-synapse" else None
        ),
        "full_per_synapse_scope": (
            "one_lowdim_slice_through_nominal_independent_edge_control_space"
            if condition.parameterization == "full-per-synapse"
            else "not_applicable"
        ),
        "parameterization_control_space": parameterized.control_space,
        "stage_rank_coordinate": "physical_weight_change",
        "parameterization_rank_coordinates": (
            "control_change_and_physical_weight_change_reported_separately"
        ),
        "hebbian_stage_definition": {
            "direct": "raw_additive_outer_product_weight_update",
            "multiplicative": "physical_weight_change_from_raw_log_outer_control",
            "full-per-synapse": (
                "physical_weight_change_from_independent_per_edge_third_factor"
            ),
        }[condition.parameterization],
        "parameterization_costs": asdict(parameterized.costs),
        "cross_parameterization_budget_matched": False,
        "cross_parameterization_dynamic_metrics_scope": (
            "descriptive_only_until_applied_update_budget_is_matched"
        ),
        "parameterization_applied_weight_pre_dale_l1": float(
            np.sum(np.abs(applied_weight_pre_dale))
        ),
        "parameterization_applied_weight_pre_dale_l2": float(
            np.linalg.norm(applied_weight_pre_dale)
        ),
        "parameterization_control_scale": parameterized.control_scale,
        "parameterization_control_bound_active": (parameterized.control_bound_active),
        "parameterization_pre_scale_exceedance_fraction": (
            parameterized.pre_scale_exceedance_fraction
        ),
        "control_bound_correction_numerical_rank": (control_bound_rank.numerical_rank),
        "control_bound_correction_effective_rank": (control_bound_rank.effective_rank),
        "control_bound_changed_synapse_count": control_bound_changed_count,
        "control_bound_changed_synapse_fraction": (
            control_bound_changed_count / connected_count
        ),
        "control_bound_correction_l1": float(np.sum(np.abs(control_bound_correction))),
        "control_bound_correction_l2": float(np.linalg.norm(control_bound_correction)),
        "control_bound_stage_definition": (
            "applied_weight_pre_Dale_minus_unbounded_masked_weight"
        ),
        "decay_definition": "zero_initial_task_component_one_step_audit",
        "normalization_enabled": network.normalize_fan_in_after_update,
        "dale_valid_after_update": network.validate_dale(),
        "dale_projection_correction_numerical_rank": (
            dale_projection_rank.numerical_rank
        ),
        "dale_projection_correction_effective_rank": (
            dale_projection_rank.effective_rank
        ),
        "dale_projection_changed_synapse_count": dale_changed_count,
        "dale_projection_changed_synapse_fraction": (
            dale_changed_count / connected_count
        ),
        "dale_projection_correction_l1": float(
            np.sum(np.abs(dale_projection_correction))
        ),
        "dale_projection_correction_l2": float(
            np.linalg.norm(dale_projection_correction)
        ),
        "dale_projection_stage_definition": (
            "Dale_applied_weight_minus_applied_weight_pre_Dale"
        ),
        "dale_boundary_synapse_count": dale_boundary_count,
        "dale_boundary_synapse_fraction": dale_boundary_count / connected_count,
        "dale_boundary_tolerance": rank_atol,
        "jacobian_outlier_count": outliers.outlier_count,
        "jacobian_bulk_tail_count": outliers.bulk_tail_count,
        "jacobian_excess_outlier_count": outliers.excess_outlier_count,
        "jacobian_bulk_right_edge": outliers.bulk_right_edge,
        "jacobian_edge_quantile": outliers.edge_quantile,
        "jacobian_target_eigenvalues_real": np.real(
            outliers.target_eigenvalues
        ).tolist(),
        "jacobian_target_eigenvalues_imag": np.imag(
            outliers.target_eigenvalues
        ).tolist(),
        "jacobian_bulk_eigenvalues_real": np.real(outliers.bulk_eigenvalues).tolist(),
        "jacobian_bulk_eigenvalues_imag": np.imag(outliers.bulk_eigenvalues).tolist(),
        "jacobian_pairing_scope": "same_state_derivative_W0_reference",
        "jacobian_state_scope": "mean_heldout_trial_time_preactivation",
        "activity_participation_ratio": participation_ratio(
            heldout_centered_activity,
            center=False,
        ),
        "activity_scope": "heldout_shared_trajectory_tape_after_one_update",
        "activity_centering_scope": "train_fitted_mean_applied_to_heldout",
        "hankel_raw_numerical_rank": hankel.raw_numerical_rank,
        "hankel_raw_effective_rank": hankel.raw_effective_rank,
        "hankel_noise_adjusted_dimension": hankel.noise_adjusted_dimension,
        "hankel_raw_numeric_threshold": hankel.raw_numeric_threshold,
        "hankel_dimension_thresholds": hankel.dimension_thresholds.tolist(),
        "hankel_singular_values": hankel.singular_values.tolist(),
        "hankel_threshold_source": hankel.threshold_source,
        "hankel_dimension_interpretation": hankel.dimension_interpretation,
        "hankel_null_singular_value_thresholds": (
            noise_floor.singular_value_thresholds.tolist()
        ),
        "hankel_null_max_singular_values": (
            noise_floor.null_max_singular_values.tolist()
        ),
        "hankel_train_window_count": noise_floor.n_train_windows,
        "hankel_test_window_count": hankel.n_windows,
        "hankel_train_trial_count": int(train_rates.shape[0]),
        "hankel_test_trial_count": int(test_rates.shape[0]),
        "hankel_feature_indices": feature_indices.tolist(),
        "hankel_feature_scope": "seed_fixed_before_activity_simulation",
        "hankel_preprocessor_train_only": True,
        "hankel_test_activity_used_for_fit": False,
        "initialization_id": resources.initialization_id,
        "mask_id": resources.mask_id,
        "state_id": resources.state_id,
        "noise_id": resources.noise_id,
        "trajectory_tape_id": resources.trajectory_tape_id,
        "feedback_frame_id": resources.feedback_frame_id,
        "edge_modulation_id": resources.edge_modulation_id,
        "hankel_feature_id": resources.hankel_feature_id,
        "shared_resource_id": resources.shared_resource_id,
        "train_activity_shape": list(train_rates.shape),
        "test_activity_shape": list(test_rates.shape),
    }
    return metrics


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "one_step_local_rank_stage_audit",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun(
        "exp08_rank_stage_validation",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        try:
            conditions = build_rank_stage_conditions(config)
            architecture = dict(config["architecture"])
            dimensions = {
                "architecture": str(architecture["name"]),
                "model_kind": "ei",
                "n_units": int(architecture["n_units"]),
            }
            planned = [
                {**condition.as_dict(), **dimensions} for condition in conditions
            ]
            run.register_conditions(planned)
        except Exception as error:
            run.register_conditions([{"condition": "setup"}])
            run.mark_condition_failure(error, condition="setup")
            return run.path

        try:
            resources = prepare_shared_resources(config, seed=seed)
        except Exception as error:
            for condition, item in zip(conditions, planned, strict=True):
                if condition.geometry_valid:
                    run.mark_condition_failure(error, **item)
                else:
                    run.mark_condition_invalid(
                        "full feedback spans the ambient space; nonzero principal angle is undefined",
                        **item,
                    )
            return run.path

        for condition, item in zip(conditions, planned, strict=True):
            if not condition.geometry_valid:
                run.mark_condition_invalid(
                    "full feedback spans the ambient space; nonzero principal angle is undefined",
                    **item,
                )
                continue
            try:
                metrics = evaluate_rank_stage_condition(
                    config,
                    resources,
                    condition,
                    seed=seed,
                )
                run.record(metrics, **item)
            except Exception as error:
                run.mark_condition_failure(error, **item)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "P1 rank-stage validation",
        "configs/formal/exp08_rank_stage_validation.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
