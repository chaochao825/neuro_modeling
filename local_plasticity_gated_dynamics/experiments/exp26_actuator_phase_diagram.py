"""Continuous task--actuator matching phase diagram on a high-rank E/I base.

Exp26 replaces the two manually aligned Exp24 endpoints with independently
rotated, rank-controlled linear task generators.  Every actuator family is
fit on the same training blocks and matched to the same training functional
current budget.  Validation and test blocks are complete held-out trials;
time points are never randomly reassigned.

The registered prospective coordinate is a finite-horizon local-injection
demand fraction, ``chi``.  It uses the actual context differences
``alpha * delta_A`` and ``(1-alpha) * delta_B`` and explicitly audits the
state--input cross term.  RGL is retained only as a combined-actuator ceiling;
the primary label compares the best single input family with low-rank
recurrent control.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.actuator_demand import (
    control_gramians,
    finite_horizon_local_demand,
    transition_rank_requirement,
)
from src.analysis.actuator_manifest import (
    GeneratorCell,
    manifest_hash,
    select_generator_manifest,
)
from src.models.factorized_controller import ActuatorMode
from src.models.task_matched_actuators import (
    TaskMatchedActuator,
    TaskMatchedActuatorConfig,
    TaskMatchedRollout,
    fit_task_matched_actuator,
)
from src.tasks.actuator_matching import (
    ActuatorCarrier,
    ActuatorDatasetConfig,
    ActuatorMatchingDataset,
    ActuatorTaskSplit,
    CarrierConfig,
    make_carrier,
    make_dataset,
    make_task_spec,
)
from src.utils.artifacts import ExperimentRun


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EXPERIMENT = "exp26_actuator_phase_diagram"
PROTOCOL_VERSION = "exp26_preregistered_v1"
MODES = ("frozen", "routing", "gain", "low_rank", "rgl")
PRIMARY_SINGLE_FAMILY_MODES = ("routing", "gain", "low_rank")


@dataclass(frozen=True)
class SharedReadout:
    """One train-fitted readout shared by all paired actuator modes."""

    mean: FloatArray
    scale: FloatArray
    model: Ridge

    def predict(self, features: FloatArray) -> IntArray:
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        decision = np.asarray(self.model.predict(normalized), dtype=np.float64)
        return np.where(decision >= 0.0, 1, -1).astype(np.int64)


@dataclass(frozen=True)
class GeneratorSetup:
    cell: GeneratorCell
    dataset: ActuatorMatchingDataset
    readout: SharedReadout
    frozen_train: TaskMatchedRollout
    frozen_validation: TaskMatchedRollout
    frozen_test: TaskMatchedRollout
    target_train_scale: float
    demand_metrics: Mapping[str, object]


def _manifest(config: Mapping[str, Any]) -> tuple[GeneratorCell, ...]:
    options = dict(config["manifest"])
    cells = select_generator_manifest(
        options["grid"],
        per_alpha_per_split=int(options["per_alpha_per_split"]),
        selection_seed=int(options["selection_seed"]),
    )
    observed = manifest_hash(cells)
    expected = str(options["expected_hash"])
    if observed != expected:
        raise RuntimeError(
            f"Exp26 manifest hash mismatch: expected {expected}, observed {observed}"
        )
    return cells


def _planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    cells = _manifest(config)
    receipt = manifest_hash(cells)
    return [
        {
            "generator_id": cell.generator_id,
            "generator_split": cell.generator_split,
            "alpha": cell.alpha,
            "transition_rank": cell.transition_rank,
            "input_rank": cell.input_rank,
            "delay": cell.delay,
            "noise_std": cell.noise_std,
            "rotation_seed": cell.rotation_seed,
            "actuator_mode": mode,
            "condition": mode,
            "manifest_hash": receipt,
        }
        for cell in cells
        for mode in MODES
    ]


def _carrier_config(config: Mapping[str, Any]) -> CarrierConfig:
    return CarrierConfig(**dict(config["carrier"]))


def _dataset_config(config: Mapping[str, Any]) -> ActuatorDatasetConfig:
    options = dict(config["task"])
    return ActuatorDatasetConfig(
        n_train_blocks=int(options["n_train_blocks"]),
        n_validation_blocks=int(options["n_validation_blocks"]),
        n_test_blocks=int(options["n_test_blocks"]),
        trials_per_block=int(options["trials_per_block"]),
        input_steps=int(options["input_steps"]),
        input_std=float(options["input_std"]),
    )


def _actuator_config(config: Mapping[str, Any]) -> TaskMatchedActuatorConfig:
    options = dict(config["actuator"])
    return TaskMatchedActuatorConfig(
        rank_a=int(options["rank_a_capacity"]),
        rank_b=int(options["rank_b_capacity"]),
        ridge=float(options["ridge"]),
        max_scale=float(options["max_scale"]),
        degeneracy_tolerance=float(options["degeneracy_tolerance"]),
        budget_relative_tolerance=float(options["budget_relative_tolerance"]),
        context_center_tolerance=float(options["context_center_tolerance"]),
    )


def _control_observable(
    target_states: FloatArray,
    frozen_states: FloatArray,
    observation: FloatArray,
) -> FloatArray:
    return (target_states[:, -1] - frozen_states[:, -1]) @ observation.T


def _control_labels(features: FloatArray) -> IntArray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("control features must have shape [trial, output]")
    if np.any(values[:, 0] == 0.0):
        raise RuntimeError("control-induced target behavior has an exact zero margin")
    return np.where(values[:, 0] > 0.0, 1, -1).astype(np.int64)


def _fit_shared_readout(
    split: ActuatorTaskSplit,
    frozen: TaskMatchedRollout,
    observation: FloatArray,
    *,
    ridge: float,
) -> SharedReadout:
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("analysis.readout_ridge must be finite and non-negative")
    features = _control_observable(
        split.target_states,
        frozen.states,
        observation,
    )
    labels = _control_labels(features)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    model = Ridge(alpha=ridge).fit((features - mean) / scale, labels)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return SharedReadout(mean=mean, scale=scale, model=model)


def _balanced_accuracy(labels: IntArray, predictions: IntArray) -> float:
    return float(balanced_accuracy_score(labels, predictions))


def _context_tape(split: ActuatorTaskSplit) -> FloatArray:
    return np.broadcast_to(
        split.contexts[:, np.newaxis],
        split.inputs.shape[:2],
    ).astype(np.float64, copy=True)


def _rollout(
    actuator: TaskMatchedActuator,
    split: ActuatorTaskSplit,
) -> TaskMatchedRollout:
    return actuator.rollout(
        split.target_states[:, 0],
        split.inputs,
        _context_tape(split),
        process_noise=split.noise,
    )


def _registered_second_moments(
    carrier: ActuatorCarrier,
    dataset_config: ActuatorDatasetConfig,
    *,
    delay: int,
    noise_std: float,
) -> tuple[FloatArray, FloatArray]:
    """Return analytic baseline moments under the registered white tapes."""

    horizon = dataset_config.input_steps + int(delay)
    if horizon < 1:
        raise ValueError("registered task horizon must be positive")
    n_state = carrier.config.n_neurons
    n_input = carrier.config.n_inputs
    state_moments = np.empty((horizon, n_state, n_state), dtype=np.float64)
    input_moments = np.zeros((horizon, n_input, n_input), dtype=np.float64)
    occupancy = np.zeros((n_state, n_state), dtype=np.float64)
    noise_covariance = float(noise_std) ** 2 * np.eye(n_state)
    for step in range(horizon):
        state_moments[step] = occupancy
        if step < dataset_config.input_steps:
            input_moments[step] = (
                dataset_config.input_std**2 * np.eye(n_input)
            )
        occupancy = (
            carrier.a0 @ occupancy @ carrier.a0.T
            + carrier.b0 @ input_moments[step] @ carrier.b0.T
            + noise_covariance
        )
        occupancy = 0.5 * (occupancy + occupancy.T)
    return state_moments, input_moments


def _empirical_cross_moments(split: ActuatorTaskSplit) -> FloatArray:
    states = np.asarray(split.target_states[:, :-1], dtype=np.float64)
    inputs = np.asarray(split.inputs, dtype=np.float64)
    return np.einsum("bti,btj->tij", states, inputs) / states.shape[0]


def _demand_metrics(
    config: Mapping[str, Any],
    carrier: ActuatorCarrier,
    dataset: ActuatorMatchingDataset,
    moments: tuple[FloatArray, FloatArray],
) -> dict[str, object]:
    analysis = dict(config["analysis"])
    spec = dataset.spec
    actual_delta_a = spec.alpha * spec.delta_a
    actual_delta_b = (1.0 - spec.alpha) * spec.delta_b
    state_moments, input_moments = moments
    demand = finite_horizon_local_demand(
        carrier.a0,
        carrier.c,
        actual_delta_a,
        actual_delta_b,
        state_moments,
        input_moments,
        cross_relative_tolerance=float(analysis["cross_relative_tolerance"]),
    )
    empirical_cross = finite_horizon_local_demand(
        carrier.a0,
        carrier.c,
        actual_delta_a,
        actual_delta_b,
        state_moments,
        input_moments,
        state_input_cross_moments=_empirical_cross_moments(dataset.train),
        cross_relative_tolerance=float(analysis["cross_relative_tolerance"]),
    )
    average_input_moment = np.mean(input_moments, axis=0)
    controllability, observability = control_gramians(
        carrier.a0,
        carrier.b0,
        carrier.c,
        input_second_moment=average_input_moment,
        horizon=dataset.train.n_steps,
    )
    rank = transition_rank_requirement(
        actual_delta_a,
        controllability,
        observability,
        candidate_ranks=(0, 1, 2, 4, 8),
        support_rtol=float(analysis["support_rtol"]),
        rank_rtol=float(analysis["rank_rtol"]),
    )
    return {
        "chi": demand.state_fraction,
        "chi_energy": demand.state_energy_fraction,
        "state_demand": demand.state_demand,
        "input_demand": demand.input_demand,
        "demand_horizon": demand.horizon,
        "demand_definition": "finite_horizon_local_injection_train_law",
        "demand_uses_actual_context_difference": True,
        "demand_input_second_moment_included": True,
        "demand_cross_energy": demand.cross_energy,
        "demand_cross_relative_magnitude": demand.cross_relative_magnitude,
        "demand_marginal_decomposition_valid": (
            demand.marginal_decomposition_valid
        ),
        "generator_state_input_cross_moment_zero_by_construction": True,
        "empirical_train_cross_energy": empirical_cross.cross_energy,
        "empirical_train_cross_relative_magnitude": (
            empirical_cross.cross_relative_magnitude
        ),
        "empirical_train_cross_within_tolerance": (
            empirical_cross.marginal_decomposition_valid
        ),
        "chi_minus_alpha": demand.state_fraction - spec.alpha,
        "chi_is_not_defined_by_alpha": True,
        "delta_a_independent_amplitude": spec.delta_a_amplitude,
        "delta_b_independent_amplitude": spec.delta_b_amplitude,
        "amplitudes_equalized_by_demand": False,
        "transition_rank_raw": rank.raw_rank,
        "transition_rank_projected": rank.projected_rank,
        "transition_energy_rank_99": rank.energy_rank_99,
        "transition_energy_rank_999": rank.energy_rank_999,
        "transition_rank_candidates": list(rank.candidate_ranks),
        "transition_rank_tail_energy_fractions": list(
            rank.tail_energy_fractions
        ),
    }


def _target_train_scale(dataset: ActuatorMatchingDataset) -> float:
    states = np.asarray(dataset.train.target_states[:, 1:], dtype=np.float64)
    centered = states - np.mean(states, axis=(0, 1), keepdims=True)
    scale = float(np.sqrt(np.mean(centered * centered)))
    if scale <= 1e-12:
        raise RuntimeError("training target state scale is degenerate")
    return scale


def _setup_generator(
    config: Mapping[str, Any],
    carrier: ActuatorCarrier,
    cell: GeneratorCell,
    *,
    seed: int,
    moment_cache: dict[tuple[int, float], tuple[FloatArray, FloatArray]],
) -> GeneratorSetup:
    task = dict(config["task"])
    spec = make_task_spec(
        carrier,
        alpha=cell.alpha,
        rA=cell.transition_rank,
        rB=cell.input_rank,
        delay=cell.delay,
        noise=cell.noise_std,
        rotation_seed=cell.rotation_seed,
        generator_id=cell.generator_id,
        delta_a_log10_range=tuple(task["delta_a_log10_range"]),
        delta_b_log10_range=tuple(task["delta_b_log10_range"]),
        stability_limit=float(task["stability_limit"]),
    )
    dataset_config = _dataset_config(config)
    dataset = make_dataset(spec, dataset_config, seed=seed)
    cache_key = (cell.delay, cell.noise_std)
    if cache_key not in moment_cache:
        moment_cache[cache_key] = _registered_second_moments(
            carrier,
            dataset_config,
            delay=cell.delay,
            noise_std=cell.noise_std,
        )
    frozen_actuator = fit_task_matched_actuator(
        dataset.train.target_states,
        dataset.train.inputs,
        _context_tape(dataset.train),
        carrier.a0,
        carrier.b0,
        mode=ActuatorMode.FROZEN,
        process_noise=dataset.train.noise,
        config=_actuator_config(config),
    )
    frozen_train = _rollout(frozen_actuator, dataset.train)
    frozen_validation = _rollout(frozen_actuator, dataset.validation)
    frozen_test = _rollout(frozen_actuator, dataset.test)
    readout = _fit_shared_readout(
        dataset.train,
        frozen_train,
        carrier.c,
        ridge=float(dict(config["analysis"])["readout_ridge"]),
    )
    return GeneratorSetup(
        cell=cell,
        dataset=dataset,
        readout=readout,
        frozen_train=frozen_train,
        frozen_validation=frozen_validation,
        frozen_test=frozen_test,
        target_train_scale=_target_train_scale(dataset),
        demand_metrics=_demand_metrics(
            config, carrier, dataset, moment_cache[cache_key]
        ),
    )


def _cosine(first: FloatArray, second: FloatArray) -> float:
    first_flat = np.asarray(first, dtype=np.float64).ravel()
    second_flat = np.asarray(second, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(first_flat, second_flat) / denominator)


def _normalized_state_error(
    target: FloatArray,
    prediction: FloatArray,
    *,
    train_scale: float,
) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)) / train_scale)


def _output_r2(
    target: FloatArray,
    prediction: FloatArray,
    observation: FloatArray,
) -> float:
    target_output = np.einsum("bti,oi->bto", target, observation).reshape(-1)
    predicted_output = np.einsum(
        "bti,oi->bto", prediction, observation
    ).reshape(-1)
    return float(r2_score(target_output, predicted_output))


def _effective_spectral_radii(actuator: TaskMatchedActuator) -> tuple[float, float]:
    gain_a = actuator.gain[:, np.newaxis] * actuator.baseline_a
    radii = []
    for context in (-1.0, 1.0):
        effective = actuator.baseline_a + context * (
            actuator.delta_a + gain_a
        )
        radii.append(float(np.max(np.abs(np.linalg.eigvals(effective)))))
    return float(radii[0]), float(radii[1])


def _split_metrics(
    setup: GeneratorSetup,
    split: ActuatorTaskSplit,
    rollout: TaskMatchedRollout,
    frozen: TaskMatchedRollout,
    *,
    prefix: str,
) -> dict[str, object]:
    target = np.asarray(split.target_states, dtype=np.float64)
    prediction = np.asarray(rollout.states, dtype=np.float64)
    target_control = _control_observable(
        target,
        frozen.states,
        setup.dataset.spec.carrier.c,
    )
    predicted_control = _control_observable(
        prediction,
        frozen.states,
        setup.dataset.spec.carrier.c,
    )
    target_labels = _control_labels(target_control)
    labels = setup.readout.predict(predicted_control)
    absolute_predictions = np.where(
        prediction[:, -1] @ setup.dataset.spec.carrier.c[0] >= 0.0,
        1,
        -1,
    ).astype(np.int64)
    delay_start = split.input_steps + 1
    delay_target = target[:, delay_start:]
    delay_prediction = prediction[:, delay_start:]
    if delay_target.shape[1] == 0:
        delay_error: float | None = None
    else:
        delay_error = _normalized_state_error(
            delay_target,
            delay_prediction,
            train_scale=setup.target_train_scale,
        )
    return {
        f"{prefix}_balanced_accuracy": _balanced_accuracy(target_labels, labels),
        f"{prefix}_behavior_endpoint": "control_induced_observable_sign",
        f"{prefix}_absolute_observation_balanced_accuracy": _balanced_accuracy(
            split.labels, absolute_predictions
        ),
        f"{prefix}_state_normalized_rmse": _normalized_state_error(
            target,
            prediction,
            train_scale=setup.target_train_scale,
        ),
        f"{prefix}_zero_input_normalized_rmse": delay_error,
        f"{prefix}_output_r2": _output_r2(
            target,
            prediction,
            setup.dataset.spec.carrier.c,
        ),
        f"{prefix}_correction_event_proxy_mean": float(
            np.mean(rollout.event_proxy_by_step)
        ),
        f"{prefix}_tape_fingerprint": rollout.tape_fingerprint,
    }


def _condition_metrics(
    config: Mapping[str, Any],
    setup: GeneratorSetup,
    *,
    mode: str,
) -> tuple[dict[str, object], bool]:
    dataset = setup.dataset
    contexts = _context_tape(dataset.train)
    actuator = fit_task_matched_actuator(
        dataset.train.target_states,
        dataset.train.inputs,
        contexts,
        dataset.spec.carrier.a0,
        dataset.spec.carrier.b0,
        mode=mode,
        process_noise=dataset.train.noise,
        config=_actuator_config(config),
    )
    train = _rollout(actuator, dataset.train)
    validation = _rollout(actuator, dataset.validation)
    test = _rollout(actuator, dataset.test)
    radii = _effective_spectral_radii(actuator)
    budget_applicable = mode != ActuatorMode.FROZEN.value
    budget_valid = bool(
        not budget_applicable
        or actuator.receipt.budget_l2_relative_error
        <= _actuator_config(config).budget_relative_tolerance
    )
    stable = bool(max(radii) < 1.0)
    target_train_ba = _balanced_accuracy(
        _control_labels(
            _control_observable(
                dataset.train.target_states,
                setup.frozen_train.states,
                dataset.spec.carrier.c,
            )
        ),
        setup.readout.predict(
            _control_observable(
                dataset.train.target_states,
                setup.frozen_train.states,
                dataset.spec.carrier.c,
            )
        ),
    )
    target_validation_ba = _balanced_accuracy(
        _control_labels(
            _control_observable(
                dataset.validation.target_states,
                setup.frozen_validation.states,
                dataset.spec.carrier.c,
            )
        ),
        setup.readout.predict(
            _control_observable(
                dataset.validation.target_states,
                setup.frozen_validation.states,
                dataset.spec.carrier.c,
            )
        ),
    )
    target_test_ba = _balanced_accuracy(
        _control_labels(
            _control_observable(
                dataset.test.target_states,
                setup.frozen_test.states,
                dataset.spec.carrier.c,
            )
        ),
        setup.readout.predict(
            _control_observable(
                dataset.test.target_states,
                setup.frozen_test.states,
                dataset.spec.carrier.c,
            )
        ),
    )
    metrics: dict[str, object] = {
        "status": "complete",
        "experiment_protocol_version": PROTOCOL_VERSION,
        "statistics_unit": "seed",
        "split_unit": "block",
        "time_points_randomly_split": False,
        "profile": str(config["profile"]),
        "dev_only": bool(config.get("dev_only", False)),
        "training_algorithm": str(config["training_algorithm"]),
        "used_autograd": False,
        "used_bptt": False,
        "local_learning_enabled": False,
        "oracle_task_matched_actuator_ceiling": True,
        "selector_learning_enabled": False,
        "rgl_is_combined_ceiling_only": True,
        "optimal_label_uses_single_family_only": True,
        "shared_scalar_centered_context_control": True,
        "belief_dimension": 1,
        "operator_rank_separate_from_belief_dimension": True,
        "frozen_high_rank_dale_compatible_base": True,
        "effective_corrections_dale_constrained": False,
        "base_recurrent_rank": int(
            np.linalg.matrix_rank(dataset.spec.carrier.a0)
        ),
        "base_recurrent_spectral_radius": dataset.spec.carrier.spectral_radius,
        "base_recurrent_fingerprint": dataset.spec.carrier.fingerprint,
        "task_spec_fingerprint": dataset.spec.fingerprint,
        "dataset_fingerprint": dataset.fingerprint,
        "train_split_fingerprint": dataset.train.fingerprint,
        "validation_split_fingerprint": dataset.validation.fingerprint,
        "test_split_fingerprint": dataset.test.fingerprint,
        "paired_base_across_modes": True,
        "paired_data_across_modes": True,
        "paired_initial_state_across_modes": True,
        "paired_context_across_modes": True,
        "paired_noise_across_modes": True,
        "readout_fit_train_only": True,
        "readout_shared_across_modes": True,
        "primary_behavior_is_control_induced_causal_contrast": True,
        "absolute_behavior_reported_secondary": True,
        "target_train_balanced_accuracy": target_train_ba,
        "target_validation_balanced_accuracy": target_validation_ba,
        "target_test_balanced_accuracy": target_test_ba,
        "functional_budget_type": "train_teacher_forced_correction_current_l2_rms",
        "functional_budget_fit_scope": "training_blocks_only",
        "functional_budget_target_uses_behavior": False,
        "functional_budget_applicable": budget_applicable,
        "functional_budget_valid": budget_valid,
        "functional_budget_target_l2_rms": actuator.receipt.target_l2_rms,
        "functional_budget_matched_l2_rms": (
            actuator.receipt.matched_current_l2_rms
        ),
        "functional_budget_l2_relative_error": (
            actuator.receipt.budget_l2_relative_error
        ),
        "functional_current_l1_reported_not_matched": True,
        "functional_current_l1_mean": actuator.receipt.matched_current_l1_mean,
        "budget_scale": actuator.receipt.budget_scale,
        "teacher_forced_error_rms": actuator.receipt.teacher_forced_error_rms,
        "teacher_forced_explained_fraction": (
            actuator.receipt.teacher_forced_explained_fraction
        ),
        "actuator_rank_a_capacity": actuator.receipt.rank_a_limit,
        "actuator_rank_b_capacity": actuator.receipt.rank_b_limit,
        "fitted_recurrent_rank": actuator.receipt.recurrent_rank,
        "fitted_input_rank": actuator.receipt.input_rank,
        "fitted_gain_rank": actuator.receipt.gain_rank,
        "raw_recurrent_fit_rank": actuator.receipt.raw_recurrent_rank,
        "raw_input_fit_rank": actuator.receipt.raw_input_rank,
        "recurrent_task_alignment_cosine": _cosine(
            actuator.delta_a,
            0.5 * dataset.spec.alpha * dataset.spec.delta_a,
        ),
        "input_task_alignment_cosine": _cosine(
            actuator.delta_b,
            0.5 * (1.0 - dataset.spec.alpha) * dataset.spec.delta_b,
        ),
        "effective_context_minus_spectral_radius": radii[0],
        "effective_context_plus_spectral_radius": radii[1],
        "effective_dynamics_strictly_stable": stable,
        "training_fingerprint": actuator.receipt.training_fingerprint,
        "training_noise_fingerprint": actuator.receipt.process_noise_fingerprint,
        "correction_fingerprint": actuator.receipt.correction_fingerprint,
        **setup.demand_metrics,
        **_split_metrics(
            setup,
            dataset.train,
            train,
            setup.frozen_train,
            prefix="train",
        ),
        **_split_metrics(
            setup,
            dataset.validation,
            validation,
            setup.frozen_validation,
            prefix="validation",
        ),
        **_split_metrics(
            setup,
            dataset.test,
            test,
            setup.frozen_test,
            prefix="test",
        ),
    }
    return metrics, bool(budget_valid and stable)


def _dimensions(
    cell: GeneratorCell,
    *,
    mode: str,
    receipt: str,
) -> dict[str, object]:
    return {
        "generator_id": cell.generator_id,
        "generator_split": cell.generator_split,
        "alpha": cell.alpha,
        "transition_rank": cell.transition_rank,
        "input_rank": cell.input_rank,
        "delay": cell.delay,
        "noise_std": cell.noise_std,
        "rotation_seed": cell.rotation_seed,
        "actuator_mode": mode,
        "condition": mode,
        "manifest_hash": receipt,
    }


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
) -> Path:
    initialize_seed(seed)
    cells = _manifest(config)
    receipt = manifest_hash(cells)
    carrier = make_carrier(_carrier_config(config), seed)
    moment_cache: dict[tuple[int, float], tuple[FloatArray, FloatArray]] = {}
    with ExperimentRun(
        EXPERIMENT,
        seed,
        config,
        results_root=results_root,
    ) as run:
        run.register_conditions(_planned_conditions(config))
        for cell in cells:
            try:
                setup = _setup_generator(
                    config,
                    carrier,
                    cell,
                    seed=seed,
                    moment_cache=moment_cache,
                )
            except Exception as error:
                for mode in MODES:
                    run.mark_condition_failure(
                        error,
                        **_dimensions(cell, mode=mode, receipt=receipt),
                    )
                continue
            for mode in MODES:
                dimensions = _dimensions(cell, mode=mode, receipt=receipt)
                try:
                    metrics, valid = _condition_metrics(
                        config,
                        setup,
                        mode=mode,
                    )
                    if valid:
                        run.record(metrics, **dimensions)
                    else:
                        failed = []
                        if not metrics["functional_budget_valid"]:
                            failed.append("functional_budget")
                        if not metrics["effective_dynamics_strictly_stable"]:
                            failed.append("effective_stability")
                        metrics["failure_reason"] = ",".join(failed)
                        run.record_failed_condition(metrics, **dimensions)
                except Exception as error:
                    run.mark_condition_failure(error, **dimensions)
        return run.path


def _selected_seeds(config: dict[str, Any], override: str | None) -> Iterable[int]:
    return seed_list(override if override is not None else config["seeds"])


def main() -> None:
    parser = basic_parser(
        "Exp26 preregistered task-actuator matching phase diagram",
        "configs/formal/exp26_actuator_phase_diagram.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    for seed in _selected_seeds(config, args.seeds):
        path = run_seed(config, seed, args.results_root)
        print(path)


if __name__ == "__main__":
    main()
