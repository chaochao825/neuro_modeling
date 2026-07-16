"""Oracle benchmark for routing, gain, and low-rank effective control.

Exp24 deliberately isolates actuator capability from controller learning.
Every mode in a task receives the same two-dimensional oracle context signal,
frozen high-rank Dale-compatible base, inputs, block split, initial states, and
readout protocol.  Non-frozen modes are matched using *training-only* caused
state displacement; parameter norms are neither matched nor reported.

The two registered tasks target different computations:

``routing_dominant``
    Two simultaneous evidence streams are present and context selects which
    stream should control the response.
``dynamics_dominant``
    The same input distribution is used in both contexts, while the target
    follows either positive integration or an alternating negative recurrence.

All dynamics forecasts are controlled rollouts supplied with held-out future
inputs and oracle controls.  They are not autonomous forecasts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.communication_subspace import (
    compare_communication_subspaces,
    fit_train_communication_subspace,
)
from src.analysis.covariance_geometry import (
    compare_covariance_geometries,
    fit_conditional_covariance_geometry,
)
from src.analysis.functional_budget_matching import (
    audit_joint_functional_budget,
    functional_observables,
    functional_state_displacement,
    match_functional_state_displacement,
)
from src.models.factorized_controller import (
    ActuatorMode,
    FactorizedController,
    FactorizedControllerConfig,
    FactorizedRollout,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EXPERIMENT = "exp24_factorized_control_benchmark"
PROTOCOL_VERSION = "exp24_v2"
TASKS = ("routing_dominant", "dynamics_dominant")
MODES = ("frozen", "routing", "gain", "low_rank", "rgl")
CONTROLLER_SOURCE = "oracle_true_context_actuator_isolation"


@dataclass(frozen=True)
class BenchmarkSplit:
    """One block-level split; no time point is independently reassigned."""

    inputs: FloatArray
    contexts: IntArray
    labels: IntArray
    instantaneous_labels: IntArray
    block_ids: IntArray
    switch_steps: IntArray


@dataclass(frozen=True)
class BenchmarkDataset:
    task: str
    train: BenchmarkSplit
    test: BenchmarkSplit
    ood: BenchmarkSplit


@dataclass(frozen=True)
class RolloutBundle:
    states: FloatArray
    rates: FloatArray
    preactivations: FloatArray
    gains: FloatArray
    routing_scales: FloatArray
    event_proxy: FloatArray
    controls: FloatArray


@dataclass(frozen=True)
class BinaryReadout:
    mean: FloatArray
    scale: FloatArray
    model: Ridge

    def decision(self, features: FloatArray) -> FloatArray:
        values = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return np.asarray(self.model.predict(values), dtype=np.float64)

    def predict(self, features: FloatArray) -> IntArray:
        return np.where(self.decision(features) >= 0.0, 1, -1).astype(np.int64)


@dataclass(frozen=True)
class LinearContextDynamics:
    a: FloatArray
    b: FloatArray
    c: FloatArray


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
            digest.update(
                json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _protocol_id(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_path"}
    return _fingerprint("exp24-factorized-control-v2", payload)


def _planned_conditions() -> list[dict[str, object]]:
    return [
        {
            "task": task,
            "condition": mode,
            "actuator_mode": mode,
            "controller_source": CONTROLLER_SOURCE,
            "control_dim": 2,
        }
        for task in TASKS
        for mode in MODES
    ]


def _sign(values: FloatArray) -> IntArray:
    return np.where(np.asarray(values) >= 0.0, 1, -1).astype(np.int64)


def _validate_task_config(config: dict[str, Any]) -> dict[str, Any]:
    options = dict(config["task"])
    for key in ("n_train_blocks", "n_test_blocks"):
        value = int(options[key])
        if value < 2 or value % 2:
            raise ValueError(f"task.{key} must be an even integer of at least two")
    for key in ("trials_per_block", "n_steps"):
        if int(options[key]) < 4:
            raise ValueError(f"task.{key} must be at least four")
    routing_steps = int(options.get("routing_sensory_steps", 1))
    if not 1 <= routing_steps < int(options["n_steps"]):
        raise ValueError(
            "task.routing_sensory_steps must be positive and shorter than n_steps"
        )
    if int(options["n_ood_blocks"]) < 1:
        raise ValueError("task.n_ood_blocks must be positive")
    rho = float(options["dynamics_rho"])
    if not 0.0 < rho < 1.0:
        raise ValueError("task.dynamics_rho must lie strictly between zero and one")
    return options


def _make_split(
    task: str,
    *,
    n_blocks: int,
    trials_per_block: int,
    n_steps: int,
    block_offset: int,
    rng: np.random.Generator,
    sensory_noise_std: float,
    evidence_min: float,
    evidence_max: float,
    dynamics_rho: float,
    routing_sensory_steps: int,
    mixed_contexts: bool,
) -> BenchmarkSplit:
    n_trials = n_blocks * trials_per_block
    inputs = np.empty((n_trials, n_steps, 2), dtype=np.float64)
    contexts = np.empty((n_trials, n_steps), dtype=np.int64)
    labels = np.empty(n_trials, dtype=np.int64)
    instantaneous = np.empty((n_trials, n_steps), dtype=np.int64)
    block_ids = np.empty(n_trials, dtype=np.int64)
    switch_steps = np.full(n_trials, -1, dtype=np.int64)
    routing_tape: dict[tuple[int, int], FloatArray] = {}
    dynamics_tape: dict[tuple[int, int], tuple[FloatArray, FloatArray]] = {}
    row = 0
    for block_index in range(n_blocks):
        block_id = block_offset + block_index
        base_context = block_index % 2
        for trial_index in range(trials_per_block):
            context_path = np.full(n_steps, base_context, dtype=np.int64)
            if mixed_contexts:
                jitter = (trial_index % 3) - 1
                switch = int(np.clip(n_steps // 2 + jitter, 2, n_steps - 2))
                context_path[switch:] = 1 - base_context
                switch_steps[row] = switch
            contexts[row] = context_path
            block_ids[row] = block_id
            if task == "routing_dominant":
                tape_key = (block_index // 2, trial_index)
                if tape_key not in routing_tape:
                    sign_pairs = np.array(
                        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
                    )
                    signs = sign_pairs[trial_index % sign_pairs.shape[0]]
                    strengths = rng.uniform(evidence_min, evidence_max, size=2)
                    streams = (
                        0.1
                        * sensory_noise_std
                        * rng.normal(size=(n_steps, 2))
                    )
                    sensory = slice(n_steps - routing_sensory_steps, n_steps)
                    streams[sensory] = (
                        signs[np.newaxis, :] * strengths[np.newaxis, :]
                        + sensory_noise_std
                        * rng.normal(size=(routing_sensory_steps, 2))
                    )
                    routing_tape[tape_key] = streams
                streams = routing_tape[tape_key].copy()
                inputs[row] = streams
                selected = streams[np.arange(n_steps), context_path]
                cumulative = np.cumsum(selected)
                instantaneous[row] = _sign(cumulative)
                labels[row] = int(instantaneous[row, -1])
            elif task == "dynamics_dominant":
                tape_key = (block_index // 2, trial_index)
                if tape_key not in dynamics_tape:
                    drive = rng.normal(size=n_steps)
                    drive += 0.15 * (1 if trial_index % 2 == 0 else -1)
                    nuisance = rng.normal(size=n_steps)
                    dynamics_tape[tape_key] = (drive, nuisance)
                drive, nuisance = dynamics_tape[tape_key]
                inputs[row, :, 0] = drive
                inputs[row, :, 1] = nuisance
                latent = 0.0
                for step in range(n_steps):
                    rho = dynamics_rho if context_path[step] == 0 else -dynamics_rho
                    latent = rho * latent + drive[step]
                    instantaneous[row, step] = 1 if latent >= 0.0 else -1
                labels[row] = int(instantaneous[row, -1])
            else:
                raise ValueError(f"unsupported Exp24 task: {task}")
            row += 1
    return BenchmarkSplit(
        inputs=inputs,
        contexts=contexts,
        labels=labels,
        instantaneous_labels=instantaneous,
        block_ids=block_ids,
        switch_steps=switch_steps,
    )


def _dataset(config: dict[str, Any], task: str, seed: int) -> BenchmarkDataset:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    options = _validate_task_config(config)
    rng = np.random.default_rng(derive_seed(seed, "exp24", task, "data"))
    common = {
        "trials_per_block": int(options["trials_per_block"]),
        "n_steps": int(options["n_steps"]),
        "rng": rng,
        "sensory_noise_std": float(options["sensory_noise_std"]),
        "evidence_min": float(options["evidence_min"]),
        "evidence_max": float(options["evidence_max"]),
        "dynamics_rho": float(options["dynamics_rho"]),
        "routing_sensory_steps": int(options.get("routing_sensory_steps", 1)),
    }
    n_train = int(options["n_train_blocks"])
    n_test = int(options["n_test_blocks"])
    train = _make_split(
        task,
        n_blocks=n_train,
        block_offset=0,
        mixed_contexts=False,
        **common,
    )
    test = _make_split(
        task,
        n_blocks=n_test,
        block_offset=n_train,
        mixed_contexts=False,
        **common,
    )
    ood = _make_split(
        task,
        n_blocks=int(options["n_ood_blocks"]),
        block_offset=n_train + n_test,
        mixed_contexts=True,
        **common,
    )
    return BenchmarkDataset(task=task, train=train, test=test, ood=ood)


def _dale_base(
    n_units: int,
    *,
    seed: int,
    inhibitory_fraction: float,
    spectral_radius: float,
) -> tuple[FloatArray, IntArray]:
    if not 0.0 < inhibitory_fraction < 1.0:
        raise ValueError("network.inhibitory_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    n_inhibitory = max(1, int(round(n_units * inhibitory_fraction)))
    n_excitatory = n_units - n_inhibitory
    if n_excitatory < 1:
        raise ValueError("network must contain excitatory and inhibitory units")
    signs = np.ones(n_units, dtype=np.int64)
    signs[n_excitatory:] = -1
    magnitudes = rng.lognormal(mean=-1.0, sigma=0.45, size=(n_units, n_units))
    mask = rng.random((n_units, n_units)) < 0.75
    base = magnitudes * mask * signs[np.newaxis, :]
    np.fill_diagonal(base, 0.0)
    radius = float(np.max(np.abs(np.linalg.eigvals(base))))
    if radius <= 0.0:
        raise RuntimeError("generated Dale base is degenerate")
    base *= spectral_radius / radius
    if np.linalg.matrix_rank(base) <= 2:
        raise RuntimeError("generated Dale base is not high rank")
    return base, signs


def _controller(
    config: dict[str, Any],
    task: str,
    seed: int,
) -> tuple[FactorizedController, IntArray]:
    options = dict(config["network"])
    n_units = int(options["n_units"])
    if n_units < 6:
        raise ValueError("network.n_units must be at least six")
    base, dale_signs = _dale_base(
        n_units,
        seed=derive_seed(seed, "exp24", task, "dale-base"),
        inhibitory_fraction=float(options["inhibitory_fraction"]),
        spectral_radius=float(options["spectral_radius"]),
    )
    rng = np.random.default_rng(derive_seed(seed, "exp24", task, "actuators"))
    half = n_units // 2
    input_weights = 0.05 * rng.normal(size=(n_units, 2))
    routing_axes = np.array([[0.9, -0.9], [-0.9, 0.9]], dtype=np.float64)
    gain_strength = float(options["gain_axis_strength"])
    gain_axes = np.empty((n_units, 2), dtype=np.float64)
    gain_axes[:half, 0] = gain_strength
    gain_axes[:half, 1] = -gain_strength
    gain_axes[half:, 0] = -gain_strength
    gain_axes[half:, 1] = gain_strength
    if task == "routing_dominant":
        # The two streams enter disjoint receiver populations. Routing can
        # select a physical stream and population gain can select its receiver
        # population before the late evidence is observed. The recurrent
        # actuator acts on the previous state, so it cannot multiplicatively
        # select evidence that arrives only on the final sensory step.
        input_scale = float(options["input_scale"])
        input_weights = np.zeros((n_units, 2), dtype=np.float64)
        input_weights[:half, 0] = input_scale * rng.uniform(0.75, 1.25, half)
        input_weights[half:, 1] = input_scale * rng.uniform(
            0.75,
            1.25,
            n_units - half,
        )
        left = rng.normal(size=(n_units, 2))
        right = rng.normal(size=(n_units, 2))
        left /= np.maximum(np.linalg.norm(left, axis=0, keepdims=True), 1e-12)
        right /= np.maximum(np.linalg.norm(right, axis=0, keepdims=True), 1e-12)
        left *= float(options["routing_task_low_rank_strength"])
    elif task == "dynamics_dominant":
        direction = rng.normal(size=n_units)
        direction /= np.linalg.norm(direction)
        input_weights[:, 0] += float(options["input_scale"]) * direction
        input_weights[:, 1] *= 0.1
        strength = float(options["dynamics_low_rank_strength"])
        left = np.column_stack((strength * direction, strength * direction))
        right = np.column_stack((direction, -direction))
        gain_axes += 0.05 * rng.normal(size=gain_axes.shape)
    else:
        raise ValueError(f"unsupported Exp24 task: {task}")
    controller = FactorizedController(
        base_recurrent=base,
        input_weights=input_weights,
        routing_axes=routing_axes,
        gain_axes=gain_axes,
        low_rank_left=left,
        low_rank_right=right,
        bias=np.zeros(n_units, dtype=np.float64),
        config=FactorizedControllerConfig(
            control_dim=2,
            activation="tanh",
            gain_min=float(options["gain_min"]),
            gain_max=float(options["gain_max"]),
            routing_min=float(options["routing_min"]),
            routing_max=float(options["routing_max"]),
        ),
    )
    return controller, dale_signs


def _oracle_controls(contexts: IntArray) -> FloatArray:
    context = np.asarray(contexts, dtype=np.int64)
    if context.ndim != 2 or not np.all(np.isin(context, (0, 1))):
        raise ValueError("contexts must have shape [trial, time] and values 0/1")
    controls = np.zeros((*context.shape, 2), dtype=np.float64)
    trial, time = np.indices(context.shape)
    controls[trial, time, context] = 1.0
    return controls


def _rollout_split(
    controller: FactorizedController,
    split: BenchmarkSplit,
    mode: str,
    *,
    scale: float,
) -> RolloutBundle:
    selected = ActuatorMode.coerce(mode)
    controls = scale * _oracle_controls(split.contexts)
    n_trials, n_steps, _ = split.inputs.shape
    states = np.zeros((n_trials, n_steps + 1, controller.n_units))
    rates = np.empty((n_trials, n_steps, controller.n_units))
    preactivations = np.empty_like(rates)
    gains = np.empty_like(rates)
    routing = np.empty((n_trials, n_steps, controller.input_dim))
    events = np.empty((n_trials, n_steps))
    for step in range(n_steps):
        control = controls[:, step]
        if selected.uses_routing:
            routing_scale = np.clip(
                1.0 + control @ controller.routing_axes.T,
                controller.config.routing_min,
                controller.config.routing_max,
            )
        else:
            routing_scale = np.ones((n_trials, controller.input_dim))
        if selected.uses_gain:
            gain = np.clip(
                1.0 + control @ controller.gain_axes.T,
                controller.config.gain_min,
                controller.config.gain_max,
            )
        else:
            gain = np.ones((n_trials, controller.n_units))
        state = states[:, step]
        recurrent_current = state @ controller.base_recurrent.T
        if selected.uses_low_rank:
            recurrent_current += (
                (state @ controller.low_rank_right) * control
            ) @ controller.low_rank_left.T
        routed_input = routing_scale * split.inputs[:, step]
        input_current = (routed_input @ controller.input_weights.T) * gain
        preactivation = recurrent_current + input_current + controller.bias
        if controller.config.activation == "tanh":
            rate = np.tanh(preactivation)
        elif controller.config.activation == "relu":
            rate = np.maximum(preactivation, 0.0)
        else:
            rate = preactivation.copy()
        states[:, step + 1] = rate
        rates[:, step] = rate
        preactivations[:, step] = preactivation
        gains[:, step] = gain
        routing[:, step] = routing_scale
        events[:, step] = np.sum(np.abs(recurrent_current), axis=1) + np.sum(
            np.abs(input_current), axis=1
        )
    return RolloutBundle(
        states=states,
        rates=rates,
        preactivations=preactivations,
        gains=gains,
        routing_scales=routing,
        event_proxy=events,
        controls=controls,
    )


def _features(states: FloatArray) -> FloatArray:
    rates = np.asarray(states[:, 1:], dtype=np.float64)
    midpoint = max(1, rates.shape[1] // 2)
    return np.concatenate(
        (
            np.mean(rates[:, :midpoint], axis=1),
            np.mean(rates[:, midpoint:], axis=1),
            rates[:, -1],
        ),
        axis=1,
    )


def _fit_readout(
    features: FloatArray,
    labels: IntArray,
    *,
    alpha: float,
) -> BinaryReadout:
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    scale = np.where(std > 1e-10, std, 1.0)
    model = Ridge(alpha=alpha).fit((values - mean) / scale, target)
    return BinaryReadout(mean=mean, scale=scale, model=model)


def _balanced_accuracy(labels: IntArray, prediction: IntArray) -> float:
    return float(balanced_accuracy_score(labels, prediction))


def _time_readout(
    train: BenchmarkSplit,
    train_rollout: RolloutBundle,
    *,
    alpha: float,
) -> BinaryReadout:
    return _fit_readout(
        train_rollout.states[:, 1:].reshape(-1, train_rollout.states.shape[-1]),
        train.instantaneous_labels.reshape(-1),
        alpha=alpha,
    )


def _switch_latency(
    readout: BinaryReadout,
    split: BenchmarkSplit,
    rollout: RolloutBundle,
    *,
    stable_steps: int,
) -> float:
    prediction = readout.predict(
        rollout.states[:, 1:].reshape(-1, rollout.states.shape[-1])
    ).reshape(split.instantaneous_labels.shape)
    latencies: list[int] = []
    for trial, switch in enumerate(split.switch_steps):
        if switch < 0:
            continue
        latency = split.inputs.shape[1] - int(switch)
        for step in range(int(switch), split.inputs.shape[1]):
            stop = min(split.inputs.shape[1], step + stable_steps)
            if stop - step < stable_steps:
                break
            if np.all(
                prediction[trial, step:stop]
                == split.instantaneous_labels[trial, step:stop]
            ):
                latency = step - int(switch)
                break
        latencies.append(latency)
    if not latencies:
        raise ValueError("switch latency requires OOD trials with a context switch")
    return float(np.mean(latencies))


def _readout_interference(
    train: BenchmarkSplit,
    test: BenchmarkSplit,
    train_rollout: RolloutBundle,
    test_rollout: RolloutBundle,
    *,
    alpha: float,
    shared_ba: float,
) -> tuple[float, float]:
    train_features = _features(train_rollout.states)
    test_features = _features(test_rollout.states)
    predictions = np.empty_like(test.labels)
    train_context = train.contexts[:, -1]
    test_context = test.contexts[:, -1]
    for context in (0, 1):
        train_mask = train_context == context
        test_mask = test_context == context
        readout = _fit_readout(
            train_features[train_mask],
            train.labels[train_mask],
            alpha=alpha,
        )
        predictions[test_mask] = readout.predict(test_features[test_mask])
    separate_ba = _balanced_accuracy(test.labels, predictions)
    return separate_ba, separate_ba - shared_ba


def _transition_rows(
    split: BenchmarkSplit,
    rollout: RolloutBundle,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    current = rollout.states[:, :-1].reshape(-1, rollout.states.shape[-1])
    following = rollout.states[:, 1:].reshape(-1, rollout.states.shape[-1])
    inputs = split.inputs.reshape(-1, split.inputs.shape[-1])
    contexts = split.contexts.reshape(-1)
    return current, following, inputs, contexts


def _fit_context_dynamics(
    split: BenchmarkSplit,
    rollout: RolloutBundle,
    *,
    alpha: float,
) -> tuple[LinearContextDynamics, LinearContextDynamics]:
    current, following, inputs, contexts = _transition_rows(split, rollout)
    fitted: list[LinearContextDynamics] = []
    n_units = current.shape[1]
    input_dim = inputs.shape[1]
    for context in (0, 1):
        mask = contexts == context
        design = np.column_stack(
            (current[mask], inputs[mask], np.ones(np.sum(mask)))
        )
        model = Ridge(alpha=alpha, fit_intercept=False).fit(design, following[mask])
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        fitted.append(
            LinearContextDynamics(
                a=coefficients[:, :n_units],
                b=coefficients[:, n_units : n_units + input_dim],
                c=coefficients[:, -1],
            )
        )
    return fitted[0], fitted[1]


def _controlled_rollout_rmse(
    models: tuple[LinearContextDynamics, LinearContextDynamics],
    train_rollout: RolloutBundle,
    split: BenchmarkSplit,
    heldout: RolloutBundle,
) -> tuple[float, float]:
    predictions = np.empty_like(heldout.states)
    predictions[:, 0] = heldout.states[:, 0]
    for trial in range(predictions.shape[0]):
        for step in range(split.inputs.shape[1]):
            model = models[int(split.contexts[trial, step])]
            predictions[trial, step + 1] = (
                model.a @ predictions[trial, step]
                + model.b @ split.inputs[trial, step]
                + model.c
            )
    error = predictions[:, 1:] - heldout.states[:, 1:]
    rmse = float(np.sqrt(np.mean(error**2)))
    train_scale = float(np.std(train_rollout.states[:, 1:]))
    normalized = rmse / max(train_scale, 1e-12)
    return rmse, normalized


def _dynamics_deltas(
    models: tuple[LinearContextDynamics, LinearContextDynamics],
) -> tuple[float, float, float]:
    first, second = models
    return (
        float(np.linalg.norm(second.a - first.a, ord="fro")),
        float(np.linalg.norm(second.b - first.b, ord="fro")),
        float(np.linalg.norm(second.c - first.c)),
    )


def _jacobian_metrics(
    controller: FactorizedController,
    rollout: RolloutBundle,
    *,
    mode: str,
    max_samples: int,
    outlier_radius: float,
) -> dict[str, float]:
    rates = rollout.rates.reshape(-1, rollout.rates.shape[-1])
    controls = rollout.controls.reshape(-1, rollout.controls.shape[-1])
    indices = np.linspace(
        0, rates.shape[0] - 1, min(max_samples, rates.shape[0]), dtype=np.int64
    )
    outliers: list[float] = []
    radii: list[float] = []
    reactivities: list[float] = []
    for index in indices:
        derivative = 1.0 - rates[index] ** 2
        effective = controller.effective_recurrent(controls[index], mode=mode)
        jacobian = derivative[:, np.newaxis] * effective
        eigenvalues = np.linalg.eigvals(jacobian)
        radius = float(np.max(np.abs(eigenvalues)))
        singular_max = float(np.linalg.svd(jacobian, compute_uv=False)[0])
        radii.append(radius)
        outliers.append(float(np.sum(np.abs(eigenvalues) > outlier_radius)))
        reactivities.append(singular_max - radius)
    return {
        "jacobian_outlier_count_mean": float(np.mean(outliers)),
        "jacobian_outlier_count_max": float(np.max(outliers)),
        "jacobian_spectral_radius_mean": float(np.mean(radii)),
        "jacobian_spectral_radius_max": float(np.max(radii)),
        "nonnormal_reactivity_mean": float(np.mean(reactivities)),
        "nonnormal_reactivity_max": float(np.max(reactivities)),
    }


def _normal_direction(states: FloatArray, rank: int) -> tuple[FloatArray, int]:
    values = states.reshape(-1, states.shape[-1])
    centered = values - np.mean(values, axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = singular[0] * np.finfo(np.float64).eps * max(centered.shape)
    numerical_rank = int(np.sum(singular > tolerance))
    tangent_rank = min(rank, numerical_rank, values.shape[1] - 1)
    if tangent_rank < 1:
        raise ValueError("training activity has no non-trivial tangent basis")
    basis = vt[:tangent_rank].T
    for coordinate in range(values.shape[1]):
        candidate = np.zeros(values.shape[1])
        candidate[coordinate] = 1.0
        candidate -= basis @ (basis.T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-8:
            return candidate / norm, tangent_rank
    raise ValueError("could not construct a training-fit normal direction")


def _normal_perturbation_recovery(
    controller: FactorizedController,
    mode: str,
    scale: float,
    split: BenchmarkSplit,
    baseline: RolloutBundle,
    *,
    train_states: FloatArray,
    tangent_rank: int,
    amplitude: float,
    n_trials: int,
) -> dict[str, float | int]:
    normal, fitted_rank = _normal_direction(train_states, tangent_rank)
    perturb_step = max(1, split.inputs.shape[1] // 3)
    endpoint: list[float] = []
    auc: list[float] = []
    controls = scale * _oracle_controls(split.contexts)
    for trial in range(min(n_trials, split.inputs.shape[0])):
        initial = baseline.states[trial, perturb_step] + amplitude * normal
        replay = controller.rollout(
            initial,
            split.inputs[trial, perturb_step:],
            mode=mode,
            controls=controls[trial, perturb_step:],
        )
        reference = baseline.states[trial, perturb_step:]
        distance = np.linalg.norm(replay.states - reference, axis=1) / amplitude
        endpoint.append(float(distance[-1]))
        auc.append(float(np.mean(distance[1:])))
    return {
        "normal_tangent_basis_rank": fitted_rank,
        "normal_perturbation_endpoint_ratio": float(np.mean(endpoint)),
        "normal_perturbation_auc_ratio": float(np.mean(auc)),
        "normal_perturbation_amplitude": amplitude,
    }


def _conditional_covariance_metrics(
    split: BenchmarkSplit,
    rollout: RolloutBundle,
    *,
    n_components: int,
) -> dict[str, object]:
    states = rollout.states[:, 1:].reshape(-1, rollout.states.shape[-1])
    contexts = split.contexts.reshape(-1)
    inputs = split.inputs.reshape(-1, split.inputs.shape[-1])
    labels = np.repeat(split.labels, split.inputs.shape[1])
    time = np.tile(
        np.linspace(-1.0, 1.0, split.inputs.shape[1]), split.inputs.shape[0]
    )
    sample_ids = np.asarray(
        [
            f"{int(block)}:{trial}:{step}"
            for trial, block in enumerate(split.block_ids)
            for step in range(split.inputs.shape[1])
        ],
        dtype=object,
    )
    fitted = []
    for context in (0, 1):
        mask = contexts == context
        covariates = np.column_stack(
            (inputs[mask], time[mask], labels[mask].astype(np.float64))
        )
        fitted.append(
            fit_conditional_covariance_geometry(
                states[mask],
                covariates,
                n_components=min(n_components, states.shape[1]),
                sample_ids=sample_ids[mask],
            )
        )
    comparison = compare_covariance_geometries(
        fitted[0].geometry_,
        fitted[1].geometry_,
    )
    return {
        "conditional_covariance_angles_degrees": (
            comparison.principal_angles_degrees.tolist()
        ),
        "conditional_covariance_angle_mean_degrees": float(
            np.mean(comparison.principal_angles_degrees)
        ),
        "conditional_covariance_subspace_overlap": comparison.subspace_overlap,
        "conditional_covariance_bures_distance": comparison.bures_distance,
        "conditional_covariance_participation_ratio_context0": (
            comparison.participation_ratio_a
        ),
        "conditional_covariance_participation_ratio_context1": (
            comparison.participation_ratio_b
        ),
        "conditional_covariance_spectrum_context0": (
            comparison.normalized_eigenspectrum_a.tolist()
        ),
        "conditional_covariance_spectrum_context1": (
            comparison.normalized_eigenspectrum_b.tolist()
        ),
    }


def _fit_communication_with_fallback(
    source: FloatArray,
    target: FloatArray,
    *,
    rank: int,
    ridge: float,
    sample_ids: NDArray[np.object_],
):
    last_error: ValueError | None = None
    for candidate in range(rank, 0, -1):
        try:
            return fit_train_communication_subspace(
                source,
                target,
                rank=candidate,
                ridge=ridge,
                normalize=True,
                sample_ids=sample_ids,
            )
        except ValueError as error:
            last_error = error
    if last_error is None:
        raise RuntimeError("communication rank fallback was not evaluated")
    raise last_error


def _communication_metrics(
    train: BenchmarkSplit,
    test: BenchmarkSplit,
    train_rollout: RolloutBundle,
    test_rollout: RolloutBundle,
    *,
    rank: int,
    ridge: float,
) -> dict[str, float | int | list[float]]:
    n_units = train_rollout.states.shape[-1]
    half = n_units // 2
    train_current, train_next, _, train_context = _transition_rows(
        train, train_rollout
    )
    test_current, test_next, _, test_context = _transition_rows(test, test_rollout)
    train_ids = np.asarray(
        [
            f"{int(block)}:{trial}:{step}"
            for trial, block in enumerate(train.block_ids)
            for step in range(train.inputs.shape[1])
        ],
        dtype=object,
    )
    fitted = []
    heldout_r2 = []
    for context in (0, 1):
        train_mask = train_context == context
        test_mask = test_context == context
        model = _fit_communication_with_fallback(
            train_current[train_mask, :half],
            train_next[train_mask, half:],
            rank=min(rank, half, n_units - half),
            ridge=ridge,
            sample_ids=train_ids[train_mask],
        )
        fitted.append(model)
        heldout_r2.append(
            model.heldout_r2(
                test_current[test_mask, :half],
                test_next[test_mask, half:],
            )
        )
    source = compare_communication_subspaces(fitted[0], fitted[1], side="source")
    target = compare_communication_subspaces(fitted[0], fitted[1], side="target")
    return {
        "communication_rank_context0": fitted[0].rank_,
        "communication_rank_context1": fitted[1].rank_,
        "communication_source_overlap": source.overlap,
        "communication_target_overlap": target.overlap,
        "communication_source_angles_degrees": (
            source.principal_angles_degrees.tolist()
        ),
        "communication_target_angles_degrees": (
            target.principal_angles_degrees.tolist()
        ),
        "communication_heldout_r2_context0": float(heldout_r2[0]),
        "communication_heldout_r2_context1": float(heldout_r2[1]),
        "communication_heldout_r2_mean": float(np.mean(heldout_r2)),
    }


def _dataset_fingerprint(dataset: BenchmarkDataset) -> str:
    return _fingerprint(
        "exp24-dataset",
        dataset.train.inputs,
        dataset.train.contexts,
        dataset.train.labels,
        dataset.test.inputs,
        dataset.test.contexts,
        dataset.test.labels,
        dataset.ood.inputs,
        dataset.ood.contexts,
        dataset.ood.labels,
    )


def _split_fingerprint(dataset: BenchmarkDataset) -> str:
    return _fingerprint(
        "exp24-block-split",
        dataset.train.block_ids,
        dataset.test.block_ids,
        dataset.ood.block_ids,
    )


def _controller_fingerprint(controller: FactorizedController) -> str:
    return _fingerprint(
        "exp24-controller-init",
        controller.base_recurrent,
        controller.input_weights,
        controller.routing_axes,
        controller.gain_axes,
        controller.low_rank_left,
        controller.low_rank_right,
    )


def _mode_scale_and_rollout(
    controller: FactorizedController,
    split: BenchmarkSplit,
    mode: str,
    frozen: RolloutBundle,
    *,
    target_displacement: float,
    budget: dict[str, Any],
) -> tuple[float, RolloutBundle, bool, int, float]:
    if mode == ActuatorMode.FROZEN.value:
        return 0.0, frozen, True, 1, 0.0

    def rollout_at_scale(scale: float) -> FactorizedRollout | RolloutBundle:
        return _rollout_split(controller, split, mode, scale=scale)

    match = match_functional_state_displacement(
        rollout_at_scale,
        frozen.states,
        target_displacement=target_displacement,
        initial_scale=float(budget["initial_scale"]),
        max_scale=float(budget["max_scale"]),
        relative_tolerance=float(budget["relative_tolerance"]),
        absolute_tolerance=float(budget["absolute_tolerance"]),
        max_iterations=int(budget["max_iterations"]),
        raise_on_unreachable=False,
    )
    rollout = _rollout_split(controller, split, mode, scale=match.scale)
    return (
        match.scale,
        rollout,
        match.converged,
        match.n_evaluations,
        match.relative_error,
    )


def _envelope_safe_scale(
    controller: FactorizedController,
    split: BenchmarkSplit,
    mode: str,
    frozen: RolloutBundle,
    *,
    budget: Mapping[str, Any],
) -> float:
    """Find the first gain/event-envelope boundary on training trajectories."""

    maximum = float(budget["max_scale"])
    gain_limit = float(budget["gain_envelope_limit"])
    event_limit = float(budget["event_change_relative_tolerance"])
    frozen_event = max(float(np.mean(np.abs(frozen.event_proxy))), 1e-12)

    def valid(scale: float) -> bool:
        rollout = _rollout_split(controller, split, mode, scale=scale)
        gain_envelope = float(np.max(np.abs(rollout.gains - frozen.gains)))
        event_relative = (
            abs(float(np.mean(rollout.event_proxy) - np.mean(frozen.event_proxy)))
            / frozen_event
        )
        return gain_envelope <= gain_limit + 1e-12 and event_relative <= event_limit

    if not valid(0.0):
        raise RuntimeError("zero control scale violates its paired frozen envelope")
    low = 0.0
    high: float | None = None
    for candidate in np.linspace(0.0, maximum, 65)[1:]:
        if valid(float(candidate)):
            low = float(candidate)
        else:
            high = float(candidate)
            break
    if high is None:
        return maximum
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        if valid(midpoint):
            low = midpoint
        else:
            high = midpoint
    return low


def _task_setup(
    config: dict[str, Any],
    task: str,
    seed: int,
) -> tuple[
    BenchmarkDataset,
    FactorizedController,
    IntArray,
    RolloutBundle,
    float,
    float,
    dict[str, float],
]:
    dataset = _dataset(config, task, seed)
    controller, dale_signs = _controller(config, task, seed)
    frozen = _rollout_split(
        controller,
        dataset.train,
        ActuatorMode.FROZEN.value,
        scale=0.0,
    )
    raw_displacements: dict[str, float] = {}
    for mode in MODES[1:]:
        raw = _rollout_split(controller, dataset.train, mode, scale=1.0)
        raw_displacements[mode] = functional_state_displacement(
            raw.states, frozen.states
        )
    positive = [value for value in raw_displacements.values() if value > 1e-14]
    if len(positive) != len(MODES) - 1:
        raise RuntimeError("every active actuator must cause positive train displacement")
    budget = dict(config["functional_budget"])
    target_fraction = float(budget["envelope_target_fraction"])
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("envelope_target_fraction must lie strictly between 0 and 1")
    envelope_safe_displacements = []
    for mode in MODES[1:]:
        safe_scale = _envelope_safe_scale(
            controller,
            dataset.train,
            mode,
            frozen,
            budget=budget,
        )
        safe_rollout = _rollout_split(
            controller,
            dataset.train,
            mode,
            scale=target_fraction * safe_scale,
        )
        envelope_safe_displacements.append(
            functional_state_displacement(safe_rollout.states, frozen.states)
        )
    target = float(min(envelope_safe_displacements))
    if target <= 1e-14:
        raise RuntimeError(
            "joint gain/event envelopes leave no non-degenerate state budget"
        )
    state_matched_rate_changes: list[float] = []
    for mode in MODES[1:]:
        _, rollout, converged, _, _ = _mode_scale_and_rollout(
            controller,
            dataset.train,
            mode,
            frozen,
            target_displacement=target,
            budget=budget,
        )
        if not converged:
            raise RuntimeError(
                "common rate target requires every active actuator to first "
                "match the registered state budget"
            )
        state_matched_rate_changes.append(
            float(np.mean(np.abs(rollout.rates - frozen.rates)))
        )
    target_rate_change = 0.5 * float(
        min(state_matched_rate_changes) + max(state_matched_rate_changes)
    )
    if target_rate_change <= 0.0:
        raise RuntimeError("common state-matched rate-change target is degenerate")
    return (
        dataset,
        controller,
        dale_signs,
        frozen,
        target,
        target_rate_change,
        raw_displacements,
    )


def _condition_metrics(
    config: dict[str, Any],
    *,
    task: str,
    mode: str,
    seed: int,
    dataset: BenchmarkDataset,
    controller: FactorizedController,
    dale_signs: IntArray,
    frozen_train: RolloutBundle,
    target_displacement: float,
    target_rate_change: float,
    raw_displacements: dict[str, float],
) -> tuple[dict[str, object], bool]:
    analysis = dict(config["analysis"])
    budget = dict(config["functional_budget"])
    scale, train, budget_converged, evaluations, relative_error = (
        _mode_scale_and_rollout(
            controller,
            dataset.train,
            mode,
            frozen_train,
            target_displacement=target_displacement,
            budget=budget,
        )
    )
    test = _rollout_split(controller, dataset.test, mode, scale=scale)
    ood = _rollout_split(controller, dataset.ood, mode, scale=scale)
    frozen_test = _rollout_split(
        controller, dataset.test, ActuatorMode.FROZEN.value, scale=0.0
    )
    readout_alpha = float(analysis["readout_alpha"])
    readout = _fit_readout(
        _features(train.states),
        dataset.train.labels,
        alpha=readout_alpha,
    )
    train_ba = _balanced_accuracy(
        dataset.train.labels, readout.predict(_features(train.states))
    )
    test_ba = _balanced_accuracy(
        dataset.test.labels, readout.predict(_features(test.states))
    )
    ood_ba = _balanced_accuracy(
        dataset.ood.labels, readout.predict(_features(ood.states))
    )
    time_readout = _time_readout(
        dataset.train,
        train,
        alpha=float(analysis["time_readout_alpha"]),
    )
    switch_latency = _switch_latency(
        time_readout,
        dataset.ood,
        ood,
        stable_steps=int(analysis["switch_stable_steps"]),
    )
    separate_ba, interference = _readout_interference(
        dataset.train,
        dataset.test,
        train,
        test,
        alpha=readout_alpha,
        shared_ba=test_ba,
    )
    dynamics = _fit_context_dynamics(
        dataset.train,
        train,
        alpha=float(analysis["dynamics_ridge"]),
    )
    rollout_rmse, rollout_nrmse = _controlled_rollout_rmse(
        dynamics, train, dataset.test, test
    )
    delta_a, delta_b, delta_c = _dynamics_deltas(dynamics)
    jacobian = _jacobian_metrics(
        controller,
        train,
        mode=mode,
        max_samples=int(analysis["jacobian_samples"]),
        outlier_radius=float(analysis["jacobian_outlier_radius"]),
    )
    perturbation = _normal_perturbation_recovery(
        controller,
        mode,
        scale,
        dataset.test,
        test,
        train_states=train.states,
        tangent_rank=int(analysis["tangent_rank"]),
        amplitude=float(analysis["perturbation_amplitude"]),
        n_trials=int(analysis["perturbation_trials"]),
    )
    covariance = _conditional_covariance_metrics(
        dataset.train,
        train,
        n_components=int(analysis["covariance_components"]),
    )
    communication = _communication_metrics(
        dataset.train,
        dataset.test,
        train,
        test,
        rank=int(analysis["communication_rank"]),
        ridge=float(analysis["communication_ridge"]),
    )
    observables = functional_observables(
        train.states,
        frozen_train.states,
        rates=train.rates,
        gains=train.gains,
        synaptic_event_proxy_by_step=train.event_proxy,
    )
    joint_audit = audit_joint_functional_budget(
        train.states,
        frozen_train.states,
        controlled_rates=train.rates,
        frozen_rates=frozen_train.rates,
        controlled_gains=train.gains,
        frozen_gains=frozen_train.gains,
        controlled_event_proxy_by_step=train.event_proxy,
        frozen_event_proxy_by_step=frozen_train.event_proxy,
        target_state_displacement=(
            0.0 if mode == ActuatorMode.FROZEN.value else target_displacement
        ),
        target_mean_absolute_rate_change=(
            0.0 if mode == ActuatorMode.FROZEN.value else target_rate_change
        ),
        state_relative_tolerance=float(budget["relative_tolerance"]),
        rate_change_relative_tolerance=float(
            budget["rate_change_relative_tolerance"]
        ),
        gain_envelope_limit=float(budget["gain_envelope_limit"]),
        event_change_relative_tolerance=float(
            budget["event_change_relative_tolerance"]
        ),
    )
    joint_budget_valid = bool(budget_converged and joint_audit.joint_valid)
    test_displacement = functional_state_displacement(
        test.states, frozen_test.states
    )
    base_fingerprint = _fingerprint("exp24-base", controller.base_recurrent)
    metrics: dict[str, object] = {
        "status": "complete",
        "experiment_protocol_version": PROTOCOL_VERSION,
        "experiment_protocol_id": _protocol_id(config),
        "statistics_unit": "seed",
        "split_unit": "block",
        "time_points_randomly_split": False,
        "task": task,
        "actuator_mode": mode,
        "controller_source": CONTROLLER_SOURCE,
        "oracle_controller": True,
        "oracle_controls_use_true_context": True,
        "local_learning_enabled": False,
        "used_bptt": False,
        "used_autograd": False,
        "exp23_prerequisite_required_for_learned_controller": True,
        "exp23_prerequisite_applied_to_oracle_benchmark": False,
        "oracle_actuator_isolation": True,
        "control_dim": controller.control_dim,
        "shared_control_dim_across_actuators": True,
        "control_step_count": int(dataset.train.inputs.shape[1]),
        "control_step_count_matched_across_modes": True,
        "functional_budget_type": (
            "train_state_target_plus_rate_gain_event_envelopes"
        ),
        "functional_budget_fit_scope": "training_blocks_only",
        "parameter_norm_budget_used": False,
        "functional_budget_equality_targets": [
            "mean_squared_state_displacement",
            "mean_absolute_rate_change",
        ],
        "functional_budget_rate_target_rule": (
            "training_only_minimax_midrange_across_active_state_matched_actuators"
        ),
        "functional_budget_targets_use_behavior": False,
        "functional_budget_state_target_rule": (
            "minimum_training_displacement_at_fraction_of_first_"
            "gain_or_event_envelope_boundary"
        ),
        "functional_budget_envelope_target_fraction": float(
            budget["envelope_target_fraction"]
        ),
        "functional_budget_rate_target_fit_scope": "training_blocks_only",
        "functional_budget_upper_bound_constraints": [
            "population_gain_envelope",
            "synaptic_event_proxy_change_relative_to_frozen",
        ],
        "all_functional_budget_terms_preregistered": True,
        "rate_gain_event_observables_reported_not_jointly_matched": False,
        "functional_budget_target": (
            0.0 if mode == ActuatorMode.FROZEN.value else target_displacement
        ),
        "functional_budget_raw_scale_one_displacement": (
            0.0 if mode == ActuatorMode.FROZEN.value else raw_displacements[mode]
        ),
        "functional_budget_control_scale": scale,
        "functional_budget_achieved_displacement": observables.state_displacement,
        "functional_budget_relative_error": relative_error,
        "functional_budget_converged": budget_converged,
        "functional_budget_state_valid": joint_audit.state_valid,
        "functional_budget_rate_valid": joint_audit.rate_valid,
        "functional_budget_gain_valid": joint_audit.gain_valid,
        "functional_budget_event_valid": joint_audit.event_valid,
        "joint_functional_budget_valid": joint_budget_valid,
        "functional_budget_search_evaluations": evaluations,
        "functional_budget_state_relative_tolerance": (
            joint_audit.state_relative_tolerance
        ),
        "functional_budget_mean_absolute_rate_change": (
            joint_audit.mean_absolute_rate_change
        ),
        "functional_budget_target_mean_absolute_rate_change": (
            joint_audit.target_mean_absolute_rate_change
        ),
        "functional_budget_rate_change_relative_error": (
            joint_audit.rate_change_relative_error
        ),
        "functional_budget_rate_change_relative_to_frozen": (
            joint_audit.rate_change_relative_to_frozen
        ),
        "functional_budget_rate_change_relative_tolerance": (
            joint_audit.rate_change_relative_tolerance
        ),
        "functional_budget_gain_envelope": joint_audit.gain_envelope,
        "functional_budget_gain_envelope_limit": joint_audit.gain_envelope_limit,
        "functional_budget_gain_envelope_fraction": (
            joint_audit.gain_envelope_fraction
        ),
        "functional_budget_synaptic_event_proxy_change": (
            joint_audit.synaptic_event_proxy_change
        ),
        "functional_budget_event_change_relative_to_frozen": (
            joint_audit.event_change_relative_to_frozen
        ),
        "functional_budget_event_change_relative_tolerance": (
            joint_audit.event_change_relative_tolerance
        ),
        "test_functional_state_displacement": test_displacement,
        "mean_rate": float(np.mean(train.rates)),
        "mean_absolute_rate": float(np.mean(np.abs(train.rates))),
        "max_absolute_rate": observables.max_absolute_rate,
        "mean_gain": observables.mean_gain,
        "max_gain": observables.max_gain,
        "mean_routing_scale": float(np.mean(train.routing_scales)),
        "max_routing_scale": float(np.max(train.routing_scales)),
        "synaptic_event_proxy": observables.synaptic_event_proxy,
        "synaptic_event_proxy_definition": (
            "mean absolute recurrent current plus absolute input current"
        ),
        "train_balanced_accuracy": train_ba,
        "test_balanced_accuracy": test_ba,
        "ood_mixed_context_balanced_accuracy": ood_ba,
        "switch_latency_steps": switch_latency,
        "separate_context_readout_balanced_accuracy": separate_ba,
        "readout_interference": interference,
        "delta_A_frobenius": delta_a,
        "delta_B_frobenius": delta_b,
        "delta_c_l2": delta_c,
        "controlled_rollout_rmse": rollout_rmse,
        "controlled_rollout_normalized_rmse": rollout_nrmse,
        "controlled_rollout_uses_future_inputs": True,
        "controlled_rollout_uses_future_oracle_controls": True,
        "autonomous_rollout": False,
        "readout_fit_train_only": True,
        "readout_protocol_shared_across_modes": True,
        "readout_weights_shared_across_modes": False,
        "dynamics_fit_train_only": True,
        "covariance_preprocessing_fit_train_only": True,
        "communication_subspace_fit_train_only": True,
        "normal_tangent_basis_fit_train_only": True,
        "normal_recovery_is_finite_amplitude_probe": True,
        "pca_variance_called_stable_manifold": False,
        "n_train_blocks": int(np.unique(dataset.train.block_ids).size),
        "n_test_blocks": int(np.unique(dataset.test.block_ids).size),
        "n_ood_blocks": int(np.unique(dataset.ood.block_ids).size),
        "n_train_trials": int(dataset.train.labels.size),
        "n_test_trials": int(dataset.test.labels.size),
        "n_ood_trials": int(dataset.ood.labels.size),
        "n_steps": int(dataset.train.inputs.shape[1]),
        "base_recurrent_rank": controller.base_recurrent_rank,
        "base_is_high_rank": controller.base_recurrent_rank > controller.control_dim,
        "base_dale_columns": bool(
            np.all(
                np.sign(controller.base_recurrent[:, dale_signs > 0][
                    controller.base_recurrent[:, dale_signs > 0] != 0
                ])
                > 0
            )
            and np.all(
                np.sign(controller.base_recurrent[:, dale_signs < 0][
                    controller.base_recurrent[:, dale_signs < 0] != 0
                ])
                < 0
            )
        ),
        "inhibitory_fraction": float(np.mean(dale_signs < 0)),
        "base_recurrent_fingerprint": base_fingerprint,
        "network_initialization_fingerprint": _controller_fingerprint(controller),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "block_split_fingerprint": _split_fingerprint(dataset),
        "oracle_control_fingerprint": _fingerprint(
            "exp24-oracle-control",
            _oracle_controls(dataset.train.contexts),
            _oracle_controls(dataset.test.contexts),
            _oracle_controls(dataset.ood.contexts),
        ),
        "paired_base_initialization_across_modes": True,
        "paired_data_across_modes": True,
        "paired_block_split_across_modes": True,
        "paired_initial_state_across_modes": True,
        "paired_noise_across_modes": True,
        "routing_input_projection_columns_identical": bool(
            np.array_equal(
                controller.input_weights[:, 0],
                controller.input_weights[:, 1],
            )
        ),
        "routing_identity_available_only_before_input_projection": bool(
            task == "routing_dominant"
        ),
        "low_rank_operator_receives_post_projection_state_only": True,
        **jacobian,
        **perturbation,
        **covariance,
        **communication,
    }
    return metrics, joint_budget_valid


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
) -> Path:
    initialize_seed(seed)
    with ExperimentRun(
        EXPERIMENT,
        seed,
        config,
        results_root=results_root,
    ) as run:
        run.register_conditions(_planned_conditions())
        for task in TASKS:
            try:
                (
                    dataset,
                    controller,
                    dale_signs,
                    frozen_train,
                    target,
                    target_rate_change,
                    raw_displacements,
                ) = _task_setup(config, task, seed)
            except Exception as error:
                for mode in MODES:
                    run.mark_condition_failure(
                        error,
                        task=task,
                        condition=mode,
                        actuator_mode=mode,
                        controller_source=CONTROLLER_SOURCE,
                        control_dim=2,
                    )
                continue
            for mode in MODES:
                dimensions = {
                    "task": task,
                    "condition": mode,
                    "actuator_mode": mode,
                    "controller_source": CONTROLLER_SOURCE,
                    "control_dim": 2,
                }
                try:
                    metrics, joint_budget_valid = _condition_metrics(
                        config,
                        task=task,
                        mode=mode,
                        seed=seed,
                        dataset=dataset,
                        controller=controller,
                        dale_signs=dale_signs,
                        frozen_train=frozen_train,
                        target_displacement=target,
                        target_rate_change=target_rate_change,
                        raw_displacements=raw_displacements,
                    )
                    metrics.pop("task")
                    metrics.pop("actuator_mode")
                    metrics.pop("controller_source")
                    metrics.pop("control_dim")
                    if (
                        mode != ActuatorMode.FROZEN.value
                        and not joint_budget_valid
                    ):
                        failed_terms = [
                            name
                            for name in ("state", "rate", "gain", "event")
                            if not metrics[f"functional_budget_{name}_valid"]
                        ]
                        metrics["failure_reason"] = (
                            "training-only joint functional budget invalid: "
                            + ",".join(failed_terms)
                        )
                        run.record_failed_condition(metrics, **dimensions)
                    else:
                        run.record(metrics, **dimensions)
                except Exception as error:
                    run.mark_condition_failure(error, **dimensions)
        return run.path


def _selected_seeds(config: dict[str, Any], override: str | None) -> Iterable[int]:
    return seed_list(override if override is not None else config["seeds"])


def main() -> None:
    parser = basic_parser(
        "Exp24 oracle factorized-control functional benchmark",
        "configs/formal/exp24_factorized_control_benchmark.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    for seed in _selected_seeds(config, args.seeds):
        path = run_seed(config, seed, args.results_root)
        print(path)


if __name__ == "__main__":
    main()
