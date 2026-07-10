"""Phase-2 causal local-learning and context-gating experiment runner.

The local path in this module is NumPy-only.  The isolated BPTT ceiling is
imported lazily, and only when the explicit ``bptt`` condition is requested.
Trials remain intact throughout training and evaluation; the public split
helper allocates complete scheduling blocks within each context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from src.analysis.energy_metrics import (
    firing_rate_energy_proxy,
    plasticity_update_energy_proxy,
    synaptic_event_energy_proxy,
)
from src.analysis.manifold_metrics import (
    fit_train_pca,
    latent_r2,
    principal_angles,
    subspace_overlap,
)
from src.analysis.rank_metrics import effective_rank, participation_ratio
from src.analysis.switching_metrics import (
    forgetting,
    jacobian_spectrum_summary,
    switch_cost_summary,
)
from src.models.ei_rate_network import (
    AppliedWeightUpdate,
    EIRateNetwork,
    EIRateState,
    EIRateStep,
)
from src.models.md_gate import MDGate
from src.plasticity.inhibitory_homeostasis import InhibitoryHomeostasis
from src.plasticity.three_factor import ThreeFactorRule, ThreeFactorUpdate
from src.tasks.context_integration import (
    ContextIntegrationBatch,
    ContextIntegrationConfig,
    generate_context_integration,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed, make_rng


Array = np.ndarray
GateMode = Literal["oracle", "learned", "none"]


@dataclass(frozen=True)
class Phase2Condition:
    """One required Phase-2 model/control condition."""

    name: str
    algorithm: Literal["local", "bptt"]
    recurrent_learning: bool
    gate_mode: GateMode
    homeostasis: bool
    feedback_mode: Literal["low_dimensional", "full", "shuffled"]
    separate_network: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("condition name must be non-empty")
        if self.algorithm not in {"local", "bptt"}:
            raise ValueError("algorithm must be 'local' or 'bptt'")
        if self.gate_mode not in {"oracle", "learned", "none"}:
            raise ValueError("gate_mode is invalid")
        if self.feedback_mode not in {"low_dimensional", "full", "shuffled"}:
            raise ValueError("feedback_mode is invalid")
        for name in ("recurrent_learning", "homeostasis", "separate_network"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
        if self.algorithm == "bptt" and (
            self.recurrent_learning or self.homeostasis or self.separate_network
        ):
            raise ValueError("bptt condition cannot enable local-only mechanisms")
        if self.separate_network and self.gate_mode != "none":
            raise ValueError("separate networks select context directly and cannot use a gate")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_phase2_conditions(base_gate: Literal["oracle", "learned"]) -> list[Phase2Condition]:
    """Return the complete required condition matrix in stable report order."""

    if base_gate not in {"oracle", "learned"}:
        raise ValueError("base_gate must be 'oracle' or 'learned'")
    return [
        Phase2Condition("local", "local", True, base_gate, True, "low_dimensional"),
        Phase2Condition("bptt", "bptt", False, "none", False, "full"),
        Phase2Condition(
            "readout-only", "local", False, base_gate, False, "low_dimensional"
        ),
        Phase2Condition("no-gate", "local", True, "none", True, "low_dimensional"),
        Phase2Condition(
            "no-homeostasis", "local", True, base_gate, False, "low_dimensional"
        ),
        Phase2Condition("full-feedback", "local", True, base_gate, True, "full"),
        Phase2Condition(
            "shuffled-feedback", "local", True, base_gate, True, "shuffled"
        ),
        Phase2Condition(
            "separate-network", "local", True, "none", True, "low_dimensional", True
        ),
    ]


def balanced_block_split(
    batch: ContextIntegrationBatch,
    *,
    test_fraction: float,
    seed: int,
    switch_window: int = 1,
) -> tuple[ContextIntegrationBatch, ContextIntegrationBatch]:
    """Split paired adjacent blocks while retaining an estimable test switch.

    Adjacent scheduling blocks form indivisible evaluation episodes.  This
    preserves at least one genuine held-out context boundary while still
    keeping every original block wholly on one side of the split.
    """

    if not isinstance(batch, ContextIntegrationBatch):
        raise TypeError("batch must be a ContextIntegrationBatch")
    if not np.isfinite(test_fraction) or not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    switch_window = _positive_int(switch_window, "switch_window")
    for block in np.unique(batch.block_ids):
        if np.unique(batch.contexts[batch.block_ids == block]).size != 1:
            raise ValueError("each scheduling block must contain exactly one context")
    ordered_blocks = list(dict.fromkeys(batch.block_ids.tolist()))
    episodes = [
        ordered_blocks[index : index + 2]
        for index in range(0, len(ordered_blocks), 2)
    ]
    if len(episodes) < 2:
        raise ValueError("at least four scheduling blocks are required for a split")

    def has_estimable_switch(indices: Array) -> bool:
        trial_ids = batch.trial_ids[indices]
        contexts = batch.contexts[indices]
        boundaries = np.concatenate(
            [
                np.array([0]),
                np.flatnonzero(np.diff(trial_ids) != 1) + 1,
                np.array([trial_ids.size]),
            ]
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            if stop - start < 2 * switch_window:
                continue
            try:
                switch_cost_summary(
                    np.ones(stop - start),
                    contexts[start:stop],
                    pre_window=switch_window,
                    post_window=switch_window,
                )
            except ValueError:
                continue
            return True
        return False

    rng = make_rng(seed, "phase2_block_split")
    episode_sizes = np.asarray(
        [np.sum(np.isin(batch.block_ids, episode)) for episode in episodes],
        dtype=int,
    )
    target_trials = float(test_fraction * batch.inputs.shape[0])
    best: tuple[float, Array, Array] | None = None
    for _ in range(5000):
        selected: list[int] = []
        selected_trials = 0
        for raw_index in rng.permutation(len(episodes)):
            episode_index = int(raw_index)
            proposed = selected_trials + int(episode_sizes[episode_index])
            current_error = abs(selected_trials - target_trials)
            proposed_error = abs(proposed - target_trials)
            if (
                not selected
                or proposed_error < current_error
                or (proposed_error == current_error and selected_trials < target_trials)
            ):
                selected.append(episode_index)
                selected_trials = proposed
        selected_episodes = np.asarray(selected, dtype=int)
        if selected_episodes.size == 0 or selected_episodes.size == len(episodes):
            continue
        test_blocks = {
            block
            for episode_index in selected_episodes
            for block in episodes[int(episode_index)]
        }
        test_mask = np.isin(batch.block_ids, list(test_blocks))
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)
        if set(batch.contexts[train_indices].tolist()) != {0, 1}:
            continue
        if set(batch.contexts[test_indices].tolist()) != {0, 1}:
            continue
        if not has_estimable_switch(test_indices):
            continue
        if np.intersect1d(
            batch.block_ids[train_indices], batch.block_ids[test_indices]
        ).size:
            raise RuntimeError("block leakage detected")
        error = abs(test_indices.size - target_trials)
        if best is None or error < best[0]:
            best = (error, train_indices, test_indices)
            if error == 0.0:
                break
    if best is not None:
        return batch.subset(best[1]), batch.subset(best[2])
    raise ValueError(
        "could not split paired blocks with both contexts and an estimable test switch"
    )


class _NonDaleRateNetwork:
    """Explicit dense/sparse NumPy rate network for the N=256 non-Dale stage."""

    activation_name = "tanh"
    is_ei = False

    def __init__(self, architecture: Mapping[str, Any], *, n_inputs: int, seed: int) -> None:
        self.n_units = _positive_int(architecture.get("n_units"), "n_units")
        self.n_inputs = int(n_inputs)
        self.dt = _positive_float(architecture.get("dt", 1.0), "dt")
        tau = _positive_float(architecture.get("tau", 20.0), "tau")
        if self.dt > tau:
            raise ValueError("non-Dale dt cannot exceed tau")
        self.time_constants = np.full(self.n_units, tau, dtype=float)
        probability = float(architecture.get("connection_probability", 1.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("connection_probability must lie in [0, 1]")
        bulk_gain = _nonnegative_float(architecture.get("bulk_gain", 0.8), "bulk_gain")
        input_scale = _nonnegative_float(
            architecture.get("input_scale", 1.0), "input_scale"
        )
        allow_self = architecture.get("allow_self_connections", False)
        if not isinstance(allow_self, (bool, np.bool_)):
            raise TypeError("allow_self_connections must be boolean")
        rng = np.random.default_rng(seed)
        self.connectivity_mask = rng.random((self.n_units, self.n_units)) < probability
        if not allow_self:
            np.fill_diagonal(self.connectivity_mask, False)
        self._W_bulk = (
            rng.normal(size=(self.n_units, self.n_units))
            * self.connectivity_mask
            * bulk_gain
            / np.sqrt(max(1, self.n_units))
        )
        self._W_task = np.zeros_like(self._W_bulk)
        self._W_homeo = np.zeros_like(self._W_bulk)
        self._W_normalization = np.zeros_like(self._W_bulk)
        self._effective_recurrent = self._W_bulk.copy()
        self.input_weights = rng.normal(
            scale=input_scale / np.sqrt(max(1, n_inputs)),
            size=(self.n_units, n_inputs),
        )
        zero = np.zeros_like(self._W_bulk)
        self.last_task_application = AppliedWeightUpdate(
            zero.copy(), zero.copy(), zero.copy(), 0.0, 0.0, 0.0
        )

    @property
    def recurrent_weights(self) -> Array:
        return self._effective_recurrent.copy()

    @staticmethod
    def _readonly_component(component: Array) -> Array:
        snapshot = component.copy()
        snapshot.setflags(write=False)
        return snapshot

    @property
    def W_bulk(self) -> Array:
        return self._readonly_component(self._W_bulk)

    @property
    def W_task(self) -> Array:
        return self._readonly_component(self._W_task)

    @property
    def W_homeo(self) -> Array:
        return self._readonly_component(self._W_homeo)

    @property
    def W_normalization(self) -> Array:
        return self._readonly_component(self._W_normalization)

    def _task_weights_for_learning(self) -> Array:
        return self._W_task

    def _effective_weights_for_learning(self) -> Array:
        return self._effective_recurrent

    @property
    def weights(self) -> Array:
        return self.recurrent_weights

    def initial_state(self) -> EIRateState:
        zeros = np.zeros(self.n_units, dtype=float)
        return EIRateState(x=zeros.copy(), rates=zeros.copy())

    def step(
        self, state: EIRateState, input_t: Array, *, gain: Array | float = 1.0
    ) -> EIRateStep:
        x = np.asarray(state.x, dtype=float)
        rates = np.asarray(state.rates, dtype=float)
        inputs = np.asarray(input_t, dtype=float)
        if x.shape != (self.n_units,) or rates.shape != (self.n_units,):
            raise ValueError("state dimensions do not match non-Dale network")
        if inputs.shape != (self.n_inputs,) or not np.all(np.isfinite(inputs)):
            raise ValueError("input_t dimensions do not match non-Dale network")
        if np.isscalar(gain):
            gains = np.full(self.n_units, float(gain), dtype=float)
        else:
            gains = np.asarray(gain, dtype=float)
        if gains.shape != (self.n_units,) or not np.all(np.isfinite(gains)) or np.any(
            gains <= 0.0
        ):
            raise ValueError("gain must be positive and match n_units")
        recurrent = self._effective_recurrent @ rates
        input_drive = self.input_weights @ inputs
        dx = (-x + recurrent + input_drive) / self.time_constants
        next_x = x + self.dt * dx
        next_rates = np.tanh(gains * next_x)
        return EIRateStep(
            state=EIRateState(x=next_x, rates=next_rates),
            recurrent_drive=recurrent,
            input_drive=input_drive,
        )

    def apply_task_update(self, update: Array) -> AppliedWeightUpdate:
        proposed = np.asarray(update, dtype=float)
        if proposed.shape != self._W_task.shape or not np.all(np.isfinite(proposed)):
            raise ValueError("task update must be a finite matching matrix")
        local = np.where(self.connectivity_mask, proposed, 0.0)
        return self._apply_projected_task_update(local)

    def _apply_projected_task_update(self, local: Array) -> AppliedWeightUpdate:
        """Trusted inner-loop path for an already masked non-Dale update."""

        self._W_task += local
        self._effective_recurrent += local
        zeros = np.zeros_like(local)
        result = AppliedWeightUpdate(
            local_update=local.copy(),
            normalization_correction=zeros,
            total_update=local.copy(),
            local_l1_cost=float(np.sum(np.abs(local))),
            normalization_l1_cost=0.0,
            total_l1_cost=float(np.sum(np.abs(local))),
        )
        self.last_task_application = result
        return result


@dataclass
class _UpdateAccumulator:
    raw: Array
    masked: Array
    applied: Array
    total: Array
    raw_l1: float = 0.0
    masked_l1: float = 0.0
    applied_l1: float = 0.0
    total_l1: float = 0.0
    normalization_l1: float = 0.0
    homeostasis_local_l1: float = 0.0
    homeostasis_total_l1: float = 0.0
    task_update_count: int = 0
    homeostasis_update_count: int = 0

    @classmethod
    def zeros(cls, n_units: int) -> "_UpdateAccumulator":
        zero = np.zeros((n_units, n_units), dtype=float)
        return cls(zero.copy(), zero.copy(), zero.copy(), zero.copy())

    def add_task(self, proposal: ThreeFactorUpdate, applied: AppliedWeightUpdate) -> None:
        self.raw += proposal.raw_update
        self.masked += proposal.masked_update
        self.applied += applied.local_update
        self.total += applied.total_update
        self.raw_l1 += proposal.costs.raw_l1
        self.masked_l1 += proposal.costs.masked_l1
        self.applied_l1 += applied.local_l1_cost
        self.total_l1 += applied.total_l1_cost
        self.normalization_l1 += applied.normalization_l1_cost
        self.task_update_count += 1


@dataclass
class _LocalBundle:
    network: EIRateNetwork | _NonDaleRateNetwork
    rule: ThreeFactorRule
    homeostasis_rule: InhibitoryHomeostasis | None
    readout: Array
    readout_bias: float
    oracle_gains: Array
    feedback_basis: Array
    feedback_encoder: Array
    md_gate: MDGate | None
    updates: _UpdateAccumulator


@dataclass(frozen=True)
class ContextConditionResult:
    """Metrics and held-out arrays for one seed/architecture/condition."""

    metrics: dict[str, Any]
    predictions: Array
    activity: Array
    trial_correct: Array


@dataclass(frozen=True)
class _Evaluation:
    predictions: Array
    x: Array
    activity: Array
    gains: Array
    md_winners: Array


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def architecture_dimensions(architecture: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable artifact dimensions for an architecture configuration."""

    name = architecture.get("name")
    kind = architecture.get("kind")
    if not isinstance(name, str) or not name:
        raise ValueError("architecture name must be non-empty")
    if kind not in {"ei", "non_dale"}:
        raise ValueError("architecture kind must be 'ei' or 'non_dale'")
    n_units = _positive_int(architecture.get("n_units"), "n_units")
    if kind == "ei":
        excitatory_fraction = float(architecture.get("excitatory_fraction", 0.8))
        inhibitory_fraction: float | None = 1.0 - excitatory_fraction
        inhibitory_gain: float | None = float(architecture.get("inhibitory_gain", 1.0))
    else:
        inhibitory_fraction = None
        inhibitory_gain = None
    return {
        "architecture": name,
        "model_kind": kind,
        "n_units": n_units,
        "inhibitory_fraction": inhibitory_fraction,
        "inhibitory_gain": inhibitory_gain,
    }


def _make_network(
    architecture: Mapping[str, Any], *, n_inputs: int, seed: int
) -> EIRateNetwork | _NonDaleRateNetwork:
    dimensions = architecture_dimensions(architecture)
    if dimensions["model_kind"] == "non_dale":
        return _NonDaleRateNetwork(architecture, n_inputs=n_inputs, seed=seed)
    return EIRateNetwork(
        dimensions["n_units"],
        n_inputs=n_inputs,
        excitatory_fraction=float(architecture.get("excitatory_fraction", 0.8)),
        connection_probability=float(architecture.get("connection_probability", 0.1)),
        tau_e=float(architecture.get("tau_e", 20.0)),
        tau_i=float(architecture.get("tau_i", 10.0)),
        dt=float(architecture.get("dt", 1.0)),
        bulk_gain=float(architecture.get("bulk_gain", 0.8)),
        inhibitory_gain=float(architecture.get("inhibitory_gain", 1.0)),
        input_scale=float(architecture.get("input_scale", 1.0)),
        activation=str(architecture.get("activation", "rectified_tanh")),
        allow_self_connections=architecture.get("allow_self_connections", False),
        normalize_fan_in_after_update=architecture.get(
            "normalize_fan_in_after_update", True
        ),
        seed=seed,
    )


def _oracle_patterns(n_units: int) -> Array:
    indices = np.arange(n_units)
    patterns = np.zeros((n_units, 2), dtype=float)
    patterns[:, 0] = (indices % 2 == 0).astype(float)
    patterns[:, 1] = (indices % 2 == 1).astype(float)
    return patterns


def _make_bundle(
    architecture: Mapping[str, Any],
    condition: Phase2Condition,
    training: Mapping[str, Any],
    *,
    seed: int,
) -> _LocalBundle:
    network = _make_network(architecture, n_inputs=4, seed=derive_seed(seed, "network"))
    n_units = network.n_units
    if (
        isinstance(network, EIRateNetwork)
        and condition.homeostasis
        and network.activation_name != "rectified_tanh"
    ):
        raise ValueError(
            "inhibitory homeostasis requires rectified_tanh non-negative rates"
        )
    feedback_dim = (
        n_units
        if condition.feedback_mode == "full"
        else min(_positive_int(training.get("feedback_dim", 4), "feedback_dim"), n_units)
    )
    # Keep the task-independent initialization streams stable across every
    # paired control.  Mechanism-specific randomness (for example shuffled
    # feedback targets) is derived separately by the condition runner.
    feedback_rng = np.random.default_rng(derive_seed(seed, "feedback"))
    readout_rng = np.random.default_rng(derive_seed(seed, "readout"))
    if condition.feedback_mode == "full":
        feedback_basis = np.eye(n_units, dtype=float)
    else:
        aligned = np.asarray(network.input_weights, dtype=float)
        if aligned.shape[1] < feedback_dim:
            aligned = np.column_stack(
                [
                    aligned,
                    feedback_rng.normal(
                        size=(n_units, feedback_dim - aligned.shape[1])
                    ),
                ]
            )
        feedback_basis, _ = np.linalg.qr(aligned[:, :feedback_dim], mode="reduced")
    feature_dim = 7  # constant, four inputs, two context indicators
    feedback_encoder = feedback_rng.normal(
        scale=1.0 / np.sqrt(feature_dim), size=(feedback_dim, feature_dim)
    )
    gate_strength = _nonnegative_float(training.get("gate_strength", 0.25), "gate_strength")
    patterns = _oracle_patterns(n_units)
    oracle_gains = 1.0 + gate_strength * patterns
    md_gate: MDGate | None = None
    if condition.gate_mode == "learned":
        n_md = _positive_int(training.get("n_md", 4), "n_md")
        gain_weights = np.zeros((n_units, n_md), dtype=float)
        gain_weights[:, :2] = patterns
        md_gate = MDGate(
            n_units,
            n_md=n_md,
            learning_rate=float(training.get("md_learning_rate", 0.05)),
            tau_trace=float(training.get("md_tau_trace", 5.0)),
            dt=1.0,
            gain_strength=gate_strength,
            gain_weights=gain_weights,
            seed=derive_seed(seed, "md"),
        )
    rule = ThreeFactorRule(
        learning_rate=float(training.get("recurrent_learning_rate", 1e-3)),
        tau_eligibility=float(training.get("tau_eligibility", 5.0)),
        dt=1.0,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    homeostasis_rule: InhibitoryHomeostasis | None = None
    if isinstance(network, EIRateNetwork) and condition.homeostasis:
        homeostasis_rule = InhibitoryHomeostasis(
            learning_rate=float(training.get("homeostasis_learning_rate", 1e-4)),
            target_rate=float(training.get("target_rate", 0.1)),
            dt=1.0,
            max_abs_update=training.get("homeostasis_max_abs_update", 1e-3),
        )
    return _LocalBundle(
        network=network,
        rule=rule,
        homeostasis_rule=homeostasis_rule,
        readout=readout_rng.normal(scale=0.01 / np.sqrt(n_units), size=n_units),
        readout_bias=0.0,
        oracle_gains=oracle_gains,
        feedback_basis=feedback_basis,
        feedback_encoder=feedback_encoder,
        md_gate=md_gate,
        updates=_UpdateAccumulator.zeros(n_units),
    )


def _select_bundle(
    bundles: Sequence[_LocalBundle], condition: Phase2Condition, context: int
) -> _LocalBundle:
    return bundles[context] if condition.separate_network else bundles[0]


def _gate_gain(
    bundle: _LocalBundle,
    *,
    context: int,
    pre_activity: Array,
    gate_mode: GateMode,
) -> tuple[Array, int]:
    if gate_mode == "oracle":
        return bundle.oracle_gains[:, context], context
    if gate_mode == "learned":
        if bundle.md_gate is None:
            raise RuntimeError("learned gate requested without an MDGate")
        output = bundle.md_gate.step(pre_activity, learn=False)
        return output.pfc_gain, output.winner
    return np.ones(bundle.network.n_units, dtype=float), -1


def _feedback_modulator(
    bundle: _LocalBundle,
    *,
    error: float,
    input_t: Array,
    context: int,
    pre_activity: Array,
    mode: str,
    scale: float,
) -> Array:
    clipped_error = float(np.clip(error, -2.0, 2.0))
    if mode == "full":
        centered = pre_activity - np.mean(pre_activity)
        signal = centered + 1.0 / np.sqrt(pre_activity.size)
        norm = np.linalg.norm(signal)
        if norm > 1.0:
            signal = signal / norm
        return scale * clipped_error * signal
    features = np.concatenate(
        [np.ones(1), np.asarray(input_t, dtype=float), np.eye(2, dtype=float)[context]]
    )
    channels = bundle.feedback_encoder @ features
    norm = np.linalg.norm(channels)
    if norm > 1.0:
        channels = channels / norm
    return scale * clipped_error * (bundle.feedback_basis @ channels)


def _post_derivative(bundle: _LocalBundle, state: EIRateState, gain: Array) -> Array:
    derivative = gain * (1.0 - np.tanh(gain * state.x) ** 2)
    if bundle.network.activation_name == "rectified_tanh":
        derivative = derivative * (state.x > 0.0)
    return derivative


def _train_local_epochs(
    bundles: Sequence[_LocalBundle],
    batch: ContextIntegrationBatch,
    condition: Phase2Condition,
    training: Mapping[str, Any],
    *,
    epochs: int,
    gate_mode: GateMode,
    teacher_targets: Array,
) -> None:
    if epochs < 0:
        raise ValueError("epochs cannot be negative")
    update_interval = _positive_int(training.get("update_interval", 1), "update_interval")
    homeostasis_interval = _positive_int(
        training.get("homeostasis_interval", 1), "homeostasis_interval"
    )
    feedback_scale = _positive_float(training.get("feedback_scale", 0.1), "feedback_scale")
    readout_lr = _positive_float(
        training.get("readout_learning_rate", 0.05), "readout_learning_rate"
    )
    global_step = 0
    for _ in range(epochs):
        for trial in range(batch.inputs.shape[0]):
            context = int(batch.contexts[trial])
            bundle = _select_bundle(bundles, condition, context)
            state = bundle.network.initial_state()
            bundle.rule.reset(bundle.network.n_units)
            if bundle.md_gate is not None:
                bundle.md_gate.reset()
            for time in range(batch.inputs.shape[1]):
                pre_activity = state.rates.copy()
                gain, _ = _gate_gain(
                    bundle,
                    context=context,
                    pre_activity=pre_activity,
                    gate_mode=gate_mode,
                )
                state = bundle.network.step(
                    state, batch.inputs[trial, time], gain=gain
                ).state
                prediction = float(bundle.readout @ state.rates + bundle.readout_bias)
                readout_error = (
                    float(batch.targets[trial, time, 0]) - prediction
                    if batch.loss_mask[trial, time]
                    else 0.0
                )
                feedback_error = (
                    float(teacher_targets[trial, time, 0]) - prediction
                    if batch.loss_mask[trial, time]
                    else 0.0
                )
                if batch.loss_mask[trial, time]:
                    normalized_step = readout_lr * np.clip(readout_error, -2.0, 2.0) / (
                        1.0 + float(state.rates @ state.rates)
                    )
                    bundle.readout += normalized_step * state.rates
                    bundle.readout_bias += 0.1 * readout_lr * float(
                        np.clip(readout_error, -2.0, 2.0)
                    )

                if condition.recurrent_learning:
                    if global_step % update_interval == 0:
                        modulator = _feedback_modulator(
                            bundle,
                            error=feedback_error,
                            input_t=batch.inputs[trial, time],
                            context=context,
                            pre_activity=pre_activity,
                            mode=condition.feedback_mode,
                            scale=feedback_scale,
                        )
                        is_ei = isinstance(bundle.network, EIRateNetwork)
                        proposal = bundle.rule._propose_trusted(
                            pre_activity,
                            modulator,
                            post_derivative=_post_derivative(bundle, state, gain),
                            connectivity_mask=bundle.network.connectivity_mask,
                            presynaptic_signs=(
                                bundle.network.presynaptic_signs if is_ei else None
                            ),
                            current_weights=bundle.network._effective_weights_for_learning(),
                            current_task_weights=bundle.network._task_weights_for_learning(),
                        )
                        application = bundle.network._apply_projected_task_update(
                            proposal.dale_applied_update
                        )
                        bundle.updates.add_task(proposal, application)
                    else:
                        bundle.rule.update_eligibility(pre_activity)

                if (
                    bundle.homeostasis_rule is not None
                    and global_step % homeostasis_interval == 0
                ):
                    homeostatic = bundle.homeostasis_rule._propose_trusted(
                        state.rates,
                        excitatory_mask=bundle.network.excitatory_mask,
                        inhibitory_mask=bundle.network.inhibitory_mask,
                        current_weights=bundle.network._effective_weights_for_learning(),
                        connectivity_mask=bundle.network.connectivity_mask,
                    )
                    application = bundle.network._apply_projected_homeostatic_update(
                        homeostatic.dale_applied_update
                    )
                    bundle.updates.homeostasis_local_l1 += application.local_l1_cost
                    bundle.updates.homeostasis_total_l1 += application.total_l1_cost
                    bundle.updates.homeostasis_update_count += 1
                global_step += 1


def _evaluate_local(
    bundles: Sequence[_LocalBundle],
    batch: ContextIntegrationBatch,
    condition: Phase2Condition,
    *,
    gate_mode: GateMode,
) -> _Evaluation:
    n_trials, n_steps, _ = batch.inputs.shape
    n_units = bundles[0].network.n_units
    predictions = np.empty((n_trials, n_steps), dtype=float)
    x = np.empty((n_trials, n_steps, n_units), dtype=float)
    activity = np.empty_like(x)
    gains = np.empty_like(x)
    winners = np.full((n_trials, n_steps), -1, dtype=int)
    for trial in range(n_trials):
        context = int(batch.contexts[trial])
        bundle = _select_bundle(bundles, condition, context)
        state = bundle.network.initial_state()
        if bundle.md_gate is not None:
            bundle.md_gate.reset()
        for time in range(n_steps):
            gain, winner = _gate_gain(
                bundle,
                context=context,
                pre_activity=state.rates,
                gate_mode=gate_mode,
            )
            state = bundle.network.step(state, batch.inputs[trial, time], gain=gain).state
            predictions[trial, time] = bundle.readout @ state.rates + bundle.readout_bias
            x[trial, time] = state.x
            activity[trial, time] = state.rates
            gains[trial, time] = gain
            winners[trial, time] = winner
    return _Evaluation(predictions, x, activity, gains, winners)


def _fit_md_from_oracle_activity(
    bundles: Sequence[_LocalBundle],
    batch: ContextIntegrationBatch,
    condition: Phase2Condition,
    oracle_evaluation: _Evaluation,
    training: Mapping[str, Any],
) -> None:
    passes = _positive_int(training.get("md_fit_passes", 2), "md_fit_passes")
    bias_strength = _positive_float(
        training.get("md_bias_strength", 5.0), "md_bias_strength"
    )
    cue_indices = np.flatnonzero(batch.epoch == "cue")
    for _ in range(passes):
        for trial in range(batch.inputs.shape[0]):
            context = int(batch.contexts[trial])
            bundle = _select_bundle(bundles, condition, context)
            if bundle.md_gate is None:
                continue
            bundle.md_gate.reset()
            bias = np.zeros(bundle.md_gate.n_md, dtype=float)
            bias[context] = bias_strength
            for time in cue_indices:
                bundle.md_gate.step(
                    oracle_evaluation.activity[trial, time],
                    learn=True,
                    modulatory_signal=1.0,
                    md_bias=bias,
                )


def _trial_behavior(
    batch: ContextIntegrationBatch, predictions: Array, *, switch_window: int
) -> tuple[dict[str, Any], Array]:
    response = batch.epoch == "response"
    logits = np.mean(predictions[:, response], axis=1)
    predicted_choices = np.where(logits >= 0.0, 1, -1)
    correct = (predicted_choices == batch.choices).astype(float)
    metrics: dict[str, Any] = {
        "accuracy": float(np.mean(correct)),
        "accuracy_by_context": [
            float(np.mean(correct[batch.contexts == context])) for context in (0, 1)
        ],
        "test_trial_ids": batch.trial_ids.tolist(),
        "test_block_ids": batch.block_ids.tolist(),
        "test_contexts": batch.contexts.tolist(),
        "response_logits": logits.tolist(),
        "predicted_choices": predicted_choices.tolist(),
        "trial_correct": correct.tolist(),
    }
    switch_values: list[float] = []
    boundaries = np.concatenate(
        [
            np.array([0]),
            np.flatnonzero(np.diff(batch.trial_ids) != 1) + 1,
            np.array([batch.trial_ids.size]),
        ]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop - start < 2:
            continue
        try:
            switching = switch_cost_summary(
                correct[start:stop],
                batch.contexts[start:stop],
                pre_window=switch_window,
                post_window=switch_window,
            )
            switch_values.extend(switching.per_switch_cost.tolist())
        except ValueError:
            continue
    if switch_values:
        values = np.asarray(switch_values, dtype=float)
        metrics.update(
            switch_cost=float(np.mean(values)),
            switch_cost_median=float(np.median(values)),
            n_valid_switches=int(values.size),
            switch_cost_estimable=True,
            switch_cost_contiguous_trials_only=True,
        )
    else:
        metrics.update(
            switch_cost=None,
            switch_cost_median=None,
            n_valid_switches=0,
            switch_cost_estimable=False,
            switch_cost_contiguous_trials_only=True,
        )
    return metrics, correct


def _accuracy_by_context(batch: ContextIntegrationBatch, predictions: Array) -> list[float]:
    response = batch.epoch == "response"
    choices = np.where(np.mean(predictions[:, response], axis=1) >= 0.0, 1, -1)
    correct = choices == batch.choices
    return [float(np.mean(correct[batch.contexts == context])) for context in (0, 1)]


def _block_half_indices(batch: ContextIntegrationBatch) -> tuple[Array, Array]:
    ordered_blocks = np.unique(batch.block_ids)
    if ordered_blocks.size < 2:
        raise ValueError("sequential forgetting requires at least two training blocks")
    midpoint = max(1, ordered_blocks.size // 2)
    first_blocks = ordered_blocks[:midpoint]
    second_blocks = ordered_blocks[midpoint:]
    if second_blocks.size == 0:
        raise ValueError("sequential forgetting requires a non-empty second phase")
    return (
        np.flatnonzero(np.isin(batch.block_ids, first_blocks)),
        np.flatnonzero(np.isin(batch.block_ids, second_blocks)),
    )


def _activity_representation_metrics(
    train_activity: Array,
    test_activity: Array,
    train_contexts: Array,
    test_contexts: Array,
    *,
    reduced_dim: int,
    ridge: float,
) -> dict[str, Any]:
    n_units = train_activity.shape[-1]
    train_flat = train_activity.reshape(-1, n_units)
    test_flat = test_activity.reshape(-1, n_units)
    metrics: dict[str, Any] = {
        "activity_participation_ratio": participation_ratio(test_flat),
    }
    bases: list[Array] = []
    for context in (0, 1):
        context_activity = train_activity[train_contexts == context].reshape(-1, n_units)
        dimension = min(reduced_dim, n_units, context_activity.shape[0] - 1)
        if dimension < 1:
            raise ValueError("insufficient training activity for context subspace")
        bases.append(
            fit_train_pca(
                context_activity,
                dimension,
                normalize=False,
                sample_ids=np.arange(context_activity.shape[0]),
            ).basis_
        )
    common_dim = min(basis.shape[1] for basis in bases)
    first, second = bases[0][:, :common_dim], bases[1][:, :common_dim]
    angles = principal_angles(first, second, degrees=True)
    metrics.update(
        context_subspace_overlap=subspace_overlap(first, second),
        context_principal_angles_degrees=angles.tolist(),
        context_subspace_dim=int(common_dim),
    )

    dimension = min(reduced_dim, n_units, train_flat.shape[0] - 1)
    pca = fit_train_pca(
        train_flat,
        dimension,
        normalize=False,
        sample_ids=np.arange(train_flat.shape[0]),
    )
    train_z = pca.transform(train_flat).reshape(*train_activity.shape[:2], dimension)
    test_z = pca.transform(test_flat).reshape(*test_activity.shape[:2], dimension)
    predictions: list[Array] = []
    targets: list[Array] = []
    for context in (0, 1):
        train_rows = train_contexts == context
        current = train_z[train_rows, :-1].reshape(-1, dimension)
        following = train_z[train_rows, 1:].reshape(-1, dimension)
        design = np.column_stack([current, np.ones(current.shape[0])])
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ following)
        test_rows = test_contexts == context
        test_current = test_z[test_rows, :-1].reshape(-1, dimension)
        test_following = test_z[test_rows, 1:].reshape(-1, dimension)
        test_design = np.column_stack([test_current, np.ones(test_current.shape[0])])
        predictions.append(test_design @ coefficients)
        targets.append(test_following)
    metrics["reduced_heldout_r2"] = float(
        latent_r2(np.vstack(targets), np.vstack(predictions))
    )
    metrics["reduced_dim"] = int(dimension)
    return metrics


def _separate_network_representation_metrics(
    test_activity: Array, test_contexts: Array
) -> dict[str, Any]:
    """Summarize dimensions without equating coordinates across networks."""

    dimensions: list[float] = []
    for context in (0, 1):
        rows = test_contexts == context
        if not np.any(rows):
            raise ValueError("each separate network requires held-out trials")
        activity = test_activity[rows].reshape(-1, test_activity.shape[-1])
        dimensions.append(participation_ratio(activity))
    return {
        "activity_participation_ratio": float(np.mean(dimensions)),
        "activity_participation_ratio_by_network": dimensions,
        "activity_dimension_scope": "mean_of_within_network_context_dimensions",
        "context_subspace_overlap": None,
        "context_principal_angles_degrees": None,
        "context_subspace_dim": None,
        "reduced_heldout_r2": None,
        "reduced_dim": None,
        "shared_coordinate_metrics_applicable": False,
        "shared_coordinate_metrics_reason": (
            "independently initialized separate networks have no shared neuron coordinates"
        ),
    }


def _jacobian_for_bundle(
    bundle: _LocalBundle, x: Array, gain: Array
) -> tuple[Array, dict[str, Any]]:
    mean_x = np.mean(x, axis=0)
    mean_gain = np.mean(gain, axis=0)
    derivative = mean_gain * (1.0 - np.tanh(mean_gain * mean_x) ** 2)
    if bundle.network.activation_name == "rectified_tanh":
        derivative *= mean_x > 0.0
    weights = bundle.network.recurrent_weights
    jacobian = -np.eye(weights.shape[0]) + weights @ np.diag(derivative)
    jacobian = jacobian / bundle.network.time_constants[:, None]
    summary = jacobian_spectrum_summary(jacobian, dynamics="continuous")
    return summary.eigenvalues, {
        "jacobian_spectral_radius": summary.spectral_radius,
        "jacobian_max_real_part": summary.max_real_part,
        "jacobian_stability_margin": summary.stability_margin,
        "jacobian_unstable_fraction": summary.unstable_fraction,
    }


def _local_metrics(
    bundles: Sequence[_LocalBundle],
    condition: Phase2Condition,
    train_batch: ContextIntegrationBatch,
    test_batch: ContextIntegrationBatch,
    train_evaluation: _Evaluation,
    test_evaluation: _Evaluation,
    training: Mapping[str, Any],
    retention_before: Sequence[float],
    retention_after_sequential: Sequence[float],
) -> tuple[dict[str, Any], Array]:
    behavior, correct = _trial_behavior(
        test_batch,
        test_evaluation.predictions,
        switch_window=_positive_int(training.get("switch_window", 1), "switch_window"),
    )
    reduced_dim = _positive_int(training.get("reduced_dim", 4), "reduced_dim")
    ridge = _positive_float(training.get("reduced_ridge", 1e-4), "reduced_ridge")
    if condition.separate_network:
        representation = _separate_network_representation_metrics(
            test_evaluation.activity, test_batch.contexts
        )
    else:
        representation = _activity_representation_metrics(
            train_evaluation.activity,
            test_evaluation.activity,
            train_batch.contexts,
            test_batch.contexts,
            reduced_dim=reduced_dim,
            ridge=ridge,
        )
        representation.update(
            activity_participation_ratio_by_network=[
                representation["activity_participation_ratio"]
            ],
            activity_dimension_scope="shared_network_heldout_activity",
            shared_coordinate_metrics_applicable=True,
        )
    stage_names = ("raw", "masked", "applied", "total")
    rank_by_stage: dict[str, list[float]] = {
        stage: [effective_rank(getattr(bundle.updates, stage)) for bundle in bundles]
        for stage in stage_names
    }
    metrics: dict[str, Any] = {
        **behavior,
        **representation,
        "status": "complete",
        "training_algorithm": "causal_online_three_factor"
        if condition.recurrent_learning
        else "causal_online_readout_only",
        "used_autograd": False,
        "recurrent_learning_enabled": condition.recurrent_learning,
        "homeostasis_enabled": any(
            bundle.homeostasis_rule is not None for bundle in bundles
        ),
        "gate_mode": condition.gate_mode,
        "feedback_mode": condition.feedback_mode,
        "separate_network": condition.separate_network,
        "feedback_dim": int(bundles[0].feedback_basis.shape[1]),
        "homeostasis_applicable": isinstance(bundles[0].network, EIRateNetwork),
        "raw_update_effective_rank": float(np.mean(rank_by_stage["raw"])),
        "masked_update_effective_rank": float(np.mean(rank_by_stage["masked"])),
        "applied_update_effective_rank": float(np.mean(rank_by_stage["applied"])),
        "total_update_effective_rank": float(np.mean(rank_by_stage["total"])),
        "raw_update_effective_rank_by_network": rank_by_stage["raw"],
        "masked_update_effective_rank_by_network": rank_by_stage["masked"],
        "applied_update_effective_rank_by_network": rank_by_stage["applied"],
        "total_update_effective_rank_by_network": rank_by_stage["total"],
        "rank_scope": "cumulative_task_three_factor_stages_excluding_homeostasis",
        "raw_plasticity_l1": float(sum(bundle.updates.raw_l1 for bundle in bundles)),
        "masked_plasticity_l1": float(
            sum(bundle.updates.masked_l1 for bundle in bundles)
        ),
        "applied_plasticity_l1": float(
            sum(bundle.updates.applied_l1 for bundle in bundles)
        ),
        "total_plasticity_l1": float(
            sum(bundle.updates.total_l1 for bundle in bundles)
        ),
        "normalization_plasticity_l1": float(
            sum(bundle.updates.normalization_l1 for bundle in bundles)
        ),
        "homeostasis_local_l1": float(
            sum(bundle.updates.homeostasis_local_l1 for bundle in bundles)
        ),
        "homeostasis_total_l1": float(
            sum(bundle.updates.homeostasis_total_l1 for bundle in bundles)
        ),
        "firing_rate_energy": firing_rate_energy_proxy(
            test_evaluation.activity.reshape(-1, test_evaluation.activity.shape[-1])
        ),
    }
    retention_after = [float(value) for value in retention_after_sequential]
    final_retention = [float(value) for value in behavior["accuracy_by_context"]]
    metrics["forgetting"] = forgetting(retention_before, retention_after)
    metrics["retention_before_by_context"] = [
        float(value) for value in retention_before
    ]
    metrics["retention_after_by_context"] = retention_after
    metrics["retention_after_sequential_by_context"] = retention_after
    metrics["retention_after_replay_by_context"] = final_retention
    metrics["forgetting_after_replay"] = forgetting(
        retention_before, final_retention
    )
    metrics["forgetting_definition"] = (
        "heldout context accuracy after first training-block half minus accuracy "
        "immediately after the second half, before any joint replay"
    )
    synaptic_values: list[float] = []
    synaptic_weights: list[int] = []
    for index, bundle in enumerate(bundles):
        if condition.separate_network:
            rows = test_batch.contexts == index
            activity = test_evaluation.activity[rows]
        else:
            activity = test_evaluation.activity
        flattened = activity.reshape(-1, activity.shape[-1])
        synaptic_values.append(
            synaptic_event_energy_proxy(flattened, bundle.network.recurrent_weights)
        )
        synaptic_weights.append(flattened.shape[0])
    metrics["synaptic_event_energy"] = float(
        np.average(synaptic_values, weights=synaptic_weights)
    )
    plasticity_path = float(
        sum(
            bundle.updates.total_l1 + bundle.updates.homeostasis_total_l1
            for bundle in bundles
        )
    )
    task_weight_events = 0
    homeostasis_local_weight_events = 0
    homeostasis_total_weight_events = 0
    for bundle in bundles:
        task_support = int(np.count_nonzero(bundle.network.connectivity_mask))
        task_weight_events += bundle.updates.task_update_count * task_support
        if isinstance(bundle.network, EIRateNetwork):
            excitatory_rows = np.flatnonzero(bundle.network.excitatory_mask)
            inhibitory_columns = np.flatnonzero(bundle.network.inhibitory_mask)
            homeostasis_local_support = int(
                np.count_nonzero(
                    bundle.network.connectivity_mask[
                        np.ix_(excitatory_rows, inhibitory_columns)
                    ]
                )
            )
            # Fan-in normalization rescales every connected E- and I-source
            # synapse on an affected excitatory postsynaptic row.  Count that
            # broader support when the numerator includes total corrections.
            homeostasis_total_support = (
                int(
                    np.count_nonzero(
                        bundle.network.connectivity_mask[excitatory_rows]
                    )
                )
                if bundle.network.normalize_fan_in_after_update
                else homeostasis_local_support
            )
            homeostasis_local_weight_events += (
                bundle.updates.homeostasis_update_count * homeostasis_local_support
            )
            homeostasis_total_weight_events += (
                bundle.updates.homeostasis_update_count * homeostasis_total_support
            )
    update_elements = task_weight_events + homeostasis_total_weight_events
    metrics["plasticity_update_energy"] = plasticity_update_energy_proxy(
        np.asarray([[plasticity_path]], dtype=float), normalize=False
    )
    metrics["plasticity_update_energy_per_weight_event"] = (
        plasticity_path / update_elements if update_elements else 0.0
    )
    metrics["task_plasticity_weight_event_count"] = int(task_weight_events)
    metrics["homeostasis_plasticity_weight_event_count"] = int(
        homeostasis_total_weight_events
    )
    metrics["homeostasis_local_plasticity_weight_event_count"] = int(
        homeostasis_local_weight_events
    )
    metrics["plasticity_weight_event_count"] = int(update_elements)
    homeostasis_local_path = float(
        sum(bundle.updates.homeostasis_local_l1 for bundle in bundles)
    )
    metrics["homeostasis_local_update_energy_per_weight_event"] = (
        homeostasis_local_path / homeostasis_local_weight_events
        if homeostasis_local_weight_events
        else 0.0
    )
    metrics["plasticity_weight_event_scope"] = (
        "task sparse support plus homeostasis total support including fan-in "
        "normalization on excitatory postsynaptic rows"
    )
    metrics["task_update_count"] = int(
        sum(bundle.updates.task_update_count for bundle in bundles)
    )
    metrics["homeostasis_update_count"] = int(
        sum(bundle.updates.homeostasis_update_count for bundle in bundles)
    )

    jacobian_metrics: list[dict[str, Any]] = []
    eigenvalues: list[Array] = []
    for index, bundle in enumerate(bundles):
        rows = test_batch.contexts == index if condition.separate_network else np.ones(
            test_batch.inputs.shape[0], dtype=bool
        )
        values, summary = _jacobian_for_bundle(
            bundle,
            test_evaluation.x[rows].reshape(-1, bundle.network.n_units),
            test_evaluation.gains[rows].reshape(-1, bundle.network.n_units),
        )
        eigenvalues.append(values)
        jacobian_metrics.append(summary)
    for key in jacobian_metrics[0]:
        metrics[key] = float(np.mean([item[key] for item in jacobian_metrics]))
    metrics["jacobian_eigenvalues_real_by_network"] = [
        np.real(values).tolist() for values in eigenvalues
    ]
    metrics["jacobian_eigenvalues_imag_by_network"] = [
        np.imag(values).tolist() for values in eigenvalues
    ]
    if condition.gate_mode == "learned":
        cue_end = int(train_batch.config.epoch_steps["cue"] - 1)
        winners = test_evaluation.md_winners[:, cue_end]
        metrics["gate_context_accuracy"] = float(np.mean(winners == test_batch.contexts))
        metrics["gate_stage"] = "hebbian_pfc_to_md_then_frozen_inference"
        metrics["md_fit_used_context_bias"] = True
    elif condition.gate_mode == "oracle":
        metrics["gate_context_accuracy"] = 1.0
        metrics["gate_stage"] = "oracle"
        metrics["md_fit_used_context_bias"] = False
    else:
        metrics["gate_context_accuracy"] = None
        metrics["gate_stage"] = "none"
        metrics["md_fit_used_context_bias"] = False
    return metrics, correct


def _shuffled_teacher(batch: ContextIntegrationBatch, seed: int) -> Array:
    rng = make_rng(seed, "shuffled_feedback_teacher")
    order = rng.permutation(batch.inputs.shape[0])
    if np.array_equal(order, np.arange(order.size)) and order.size > 1:
        order = np.roll(order, 1)
    return batch.targets[order].copy()


def _run_local_condition(
    train_batch: ContextIntegrationBatch,
    test_batch: ContextIntegrationBatch,
    condition: Phase2Condition,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    seed: int,
) -> ContextConditionResult:
    count = 2 if condition.separate_network else 1
    bundles = [
        _make_bundle(
            architecture,
            condition,
            training,
            seed=derive_seed(seed, "bundle", index),
        )
        for index in range(count)
    ]
    teacher = (
        _shuffled_teacher(train_batch, seed)
        if condition.feedback_mode == "shuffled"
        else train_batch.targets
    )
    first_indices, second_indices = _block_half_indices(train_batch)
    first_batch = train_batch.subset(first_indices)
    second_batch = train_batch.subset(second_indices)
    retention_before: list[float]
    retention_after_sequential: list[float]
    if condition.gate_mode == "learned":
        pretrain_epochs = _positive_int(
            training.get("oracle_pretrain_epochs", 1), "oracle_pretrain_epochs"
        )
        _train_local_epochs(
            bundles,
            train_batch,
            condition,
            training,
            epochs=pretrain_epochs,
            gate_mode="oracle",
            teacher_targets=teacher,
        )
        oracle_activity = _evaluate_local(
            bundles, train_batch, condition, gate_mode="oracle"
        )
        _fit_md_from_oracle_activity(
            bundles, train_batch, condition, oracle_activity, training
        )
        learned_epochs = _positive_int(
            training.get("learned_epochs", 1), "learned_epochs"
        )
        _train_local_epochs(
            bundles,
            first_batch,
            condition,
            training,
            epochs=1,
            gate_mode="learned",
            teacher_targets=teacher[first_indices],
        )
        midpoint_evaluation = _evaluate_local(
            bundles, test_batch, condition, gate_mode="learned"
        )
        retention_before = _accuracy_by_context(
            test_batch, midpoint_evaluation.predictions
        )
        _train_local_epochs(
            bundles,
            second_batch,
            condition,
            training,
            epochs=1,
            gate_mode="learned",
            teacher_targets=teacher[second_indices],
        )
        sequential_evaluation = _evaluate_local(
            bundles, test_batch, condition, gate_mode="learned"
        )
        retention_after_sequential = _accuracy_by_context(
            test_batch, sequential_evaluation.predictions
        )
        if learned_epochs > 1:
            _train_local_epochs(
                bundles,
                train_batch,
                condition,
                training,
                epochs=learned_epochs - 1,
                gate_mode="learned",
                teacher_targets=teacher,
            )
    else:
        epochs = _positive_int(training.get("train_epochs", 1), "train_epochs")
        _train_local_epochs(
            bundles,
            first_batch,
            condition,
            training,
            epochs=1,
            gate_mode=condition.gate_mode,
            teacher_targets=teacher[first_indices],
        )
        midpoint_evaluation = _evaluate_local(
            bundles, test_batch, condition, gate_mode=condition.gate_mode
        )
        retention_before = _accuracy_by_context(
            test_batch, midpoint_evaluation.predictions
        )
        _train_local_epochs(
            bundles,
            second_batch,
            condition,
            training,
            epochs=1,
            gate_mode=condition.gate_mode,
            teacher_targets=teacher[second_indices],
        )
        sequential_evaluation = _evaluate_local(
            bundles, test_batch, condition, gate_mode=condition.gate_mode
        )
        retention_after_sequential = _accuracy_by_context(
            test_batch, sequential_evaluation.predictions
        )
        if epochs > 1:
            _train_local_epochs(
                bundles,
                train_batch,
                condition,
                training,
                epochs=epochs - 1,
                gate_mode=condition.gate_mode,
                teacher_targets=teacher,
            )
    train_evaluation = _evaluate_local(
        bundles, train_batch, condition, gate_mode=condition.gate_mode
    )
    test_evaluation = _evaluate_local(
        bundles, test_batch, condition, gate_mode=condition.gate_mode
    )
    metrics, correct = _local_metrics(
        bundles,
        condition,
        train_batch,
        test_batch,
        train_evaluation,
        test_evaluation,
        training,
        retention_before,
        retention_after_sequential,
    )
    return ContextConditionResult(
        metrics=metrics,
        predictions=test_evaluation.predictions,
        activity=test_evaluation.activity,
        trial_correct=correct,
    )


def _run_bptt_condition(
    train_batch: ContextIntegrationBatch,
    test_batch: ContextIntegrationBatch,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    seed: int,
) -> ContextConditionResult:
    # Lazy import is the isolation boundary: no local condition imports torch.
    from src.baselines import bptt as bptt_module

    options = dict(training.get("bptt", {}))
    options.update(
        hidden_size=_positive_int(architecture.get("n_units"), "n_units"),
        seed=seed,
    )
    bptt_config = bptt_module.BPTTConfig(**options)
    model, losses = bptt_module.train_bptt_baseline(
        np.array(train_batch.inputs, dtype=np.float32, copy=True),
        np.array(train_batch.targets, dtype=np.float32, copy=True),
        np.array(train_batch.loss_mask, dtype=bool, copy=True),
        bptt_config,
    )
    _, train_activity = bptt_module.predict_bptt(
        model, np.array(train_batch.inputs, dtype=np.float32, copy=True)
    )
    test_prediction, test_activity = bptt_module.predict_bptt(
        model, np.array(test_batch.inputs, dtype=np.float32, copy=True)
    )
    predictions = test_prediction[..., 0]
    behavior, correct = _trial_behavior(
        test_batch,
        predictions,
        switch_window=_positive_int(training.get("switch_window", 1), "switch_window"),
    )
    representation = _activity_representation_metrics(
        train_activity,
        test_activity,
        train_batch.contexts,
        test_batch.contexts,
        reduced_dim=_positive_int(training.get("reduced_dim", 4), "reduced_dim"),
        ridge=_positive_float(training.get("reduced_ridge", 1e-4), "reduced_ridge"),
    )
    weights = model.recurrent_layer.weight.detach().cpu().numpy().copy()
    derivative = np.mean(1.0 - test_activity**2, axis=(0, 1))
    jacobian = weights @ np.diag(derivative)
    jacobian_summary = jacobian_spectrum_summary(jacobian, dynamics="discrete")
    metrics: dict[str, Any] = {
        **behavior,
        **representation,
        "status": "complete",
        "training_algorithm": "bptt_baseline",
        "used_autograd": True,
        "recurrent_learning_enabled": True,
        "homeostasis_enabled": False,
        "gate_mode": "none",
        "feedback_mode": "not_applicable",
        "separate_network": False,
        "feedback_dim": int(weights.shape[0]),
        "homeostasis_applicable": False,
        "raw_update_effective_rank": None,
        "masked_update_effective_rank": None,
        "applied_update_effective_rank": None,
        "total_update_effective_rank": None,
        "recurrent_effective_rank": effective_rank(weights),
        "raw_plasticity_l1": None,
        "masked_plasticity_l1": None,
        "applied_plasticity_l1": None,
        "total_plasticity_l1": None,
        "normalization_plasticity_l1": None,
        "homeostasis_local_l1": None,
        "homeostasis_total_l1": None,
        "firing_rate_energy": firing_rate_energy_proxy(
            test_activity.reshape(-1, test_activity.shape[-1])
        ),
        "synaptic_event_energy": synaptic_event_energy_proxy(
            test_activity.reshape(-1, test_activity.shape[-1]), weights
        ),
        "plasticity_update_energy": None,
        "plasticity_update_energy_per_weight_event": None,
        "task_plasticity_weight_event_count": None,
        "homeostasis_plasticity_weight_event_count": None,
        "homeostasis_local_plasticity_weight_event_count": None,
        "plasticity_weight_event_count": None,
        "homeostasis_local_update_energy_per_weight_event": None,
        "plasticity_weight_event_scope": "not_applicable_to_bptt_baseline",
        "plasticity_update_energy_reason": (
            "isolated baseline API does not expose pre-training recurrent parameters"
        ),
        "task_update_count": None,
        "homeostasis_update_count": 0,
        "rank_scope": "not_applicable_to_nonlocal_bptt_stages",
        "jacobian_spectral_radius": jacobian_summary.spectral_radius,
        "jacobian_max_real_part": jacobian_summary.max_real_part,
        "jacobian_stability_margin": jacobian_summary.stability_margin,
        "jacobian_unstable_fraction": jacobian_summary.unstable_fraction,
        "jacobian_eigenvalues_real_by_network": [
            np.real(jacobian_summary.eigenvalues).tolist()
        ],
        "jacobian_eigenvalues_imag_by_network": [
            np.imag(jacobian_summary.eigenvalues).tolist()
        ],
        "gate_context_accuracy": None,
        "gate_stage": "not_applicable",
        "md_fit_used_context_bias": False,
        "shared_coordinate_metrics_applicable": True,
        "bptt_final_loss": float(losses[-1]),
        "bptt_checkpoint_metadata": model.checkpoint_metadata(),
        "forgetting": None,
        "forgetting_after_replay": None,
        "retention_before_by_context": None,
        "retention_after_by_context": None,
        "retention_after_sequential_by_context": None,
        "retention_after_replay_by_context": behavior["accuracy_by_context"],
        "forgetting_definition": "not_applicable_to_joint_bptt_training",
    }
    return ContextConditionResult(metrics, predictions, test_activity, correct)


def run_context_condition(
    train_batch: ContextIntegrationBatch,
    test_batch: ContextIntegrationBatch,
    condition: Phase2Condition,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    seed: int,
) -> ContextConditionResult:
    """Train and evaluate one complete Phase-2 condition."""

    if not isinstance(condition, Phase2Condition):
        raise TypeError("condition must be a Phase2Condition")
    if not isinstance(architecture, Mapping) or not isinstance(training, Mapping):
        raise TypeError("architecture and training must be mappings")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if train_batch.config != test_batch.config:
        raise ValueError("train and test batches must share a task config")
    if np.intersect1d(train_batch.block_ids, test_batch.block_ids).size:
        raise ValueError("train/test block leakage detected")
    dimensions = architecture_dimensions(architecture)
    if condition.name == "no-homeostasis" and dimensions["model_kind"] != "ei":
        raise ValueError(
            "no-homeostasis is only an interpretable control for an E/I architecture"
        )
    if condition.algorithm == "bptt":
        result = _run_bptt_condition(
            train_batch, test_batch, architecture, training, seed=seed
        )
    else:
        result = _run_local_condition(
            train_batch,
            test_batch,
            condition,
            architecture,
            training,
            seed=seed,
        )
    metrics = dict(result.metrics)
    metrics.update(
        initialization_id=f"phase2:{dimensions['architecture']}:{int(seed)}",
        initialization_seed=int(seed),
        paired_initialization_scope=(
            "shared_network_input_readout_seed"
            if not condition.separate_network and condition.algorithm == "local"
            else (
                "shared_pairing_seed_but_distinct_bptt_model_family"
                if condition.algorithm == "bptt"
                else "base_seed_with_context_specific_network_substreams"
            )
        ),
        homeostasis_control_applicable=dimensions["model_kind"] == "ei",
        homeostasis_control_interpretation=(
            "applicable_ei_ablation"
            if dimensions["model_kind"] == "ei"
            else "not_applicable_non_dale_architecture"
        ),
    )
    return ContextConditionResult(
        metrics=metrics,
        predictions=result.predictions,
        activity=result.activity,
        trial_correct=result.trial_correct,
    )


def run_phase2_experiment(
    config: Mapping[str, Any],
    *,
    seed: int,
    results_root: str | Path,
    experiment_name: str,
    base_gate: Literal["oracle", "learned"],
) -> Path:
    """Run every architecture/condition cell and preserve all failures."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    run_config = {
        **dict(config),
        "training_algorithm": "phase2_condition_grid",
        "used_autograd": True,
        "autograd_scope": "bptt_condition_only",
        "parent_checkpoint": None,
        "base_gate": base_gate,
    }
    with ExperimentRun(
        experiment_name, seed, run_config, results_root=results_root
    ) as run:
        try:
            raw_architectures = config["architectures"]
            if isinstance(raw_architectures, (str, bytes, Mapping)):
                raise TypeError("architectures must be a sequence of mappings")
            architectures = list(raw_architectures)
            if not architectures:
                raise ValueError("architectures cannot be empty")
            if not all(isinstance(item, Mapping) for item in architectures):
                raise TypeError("every architecture must be a mapping")
            dimensions = [architecture_dimensions(item) for item in architectures]
            if len({item["architecture"] for item in dimensions}) != len(dimensions):
                raise ValueError("architecture names must be unique")
            raw_training = config["training"]
            if not isinstance(raw_training, Mapping):
                raise TypeError("training must be a mapping")
            training_config = dict(raw_training)
            conditions = build_phase2_conditions(base_gate)
            lookup = {condition.name: condition for condition in conditions}
            selected_conditions: list[list[Phase2Condition]] = []
            for architecture, architecture_dimension in zip(
                architectures, dimensions, strict=True
            ):
                requested = architecture.get("conditions")
                if requested is None:
                    selected = [
                        condition
                        for condition in conditions
                        if not (
                            condition.name == "no-homeostasis"
                            and architecture_dimension["model_kind"] != "ei"
                        )
                    ]
                else:
                    if isinstance(requested, str) or not isinstance(
                        requested, Sequence
                    ):
                        raise TypeError(
                            "architecture conditions must be a sequence of names"
                        )
                    names = list(requested)
                    if len(set(names)) != len(names):
                        raise ValueError("architecture condition names must be unique")
                    unknown = set(names) - set(lookup)
                    if unknown:
                        raise ValueError(
                            f"unknown architecture conditions: {sorted(unknown)}"
                        )
                    if (
                        architecture_dimension["model_kind"] != "ei"
                        and "no-homeostasis" in names
                    ):
                        raise ValueError(
                            "no-homeostasis can only be scheduled for an E/I architecture"
                        )
                    selected = [lookup[name] for name in names]
                if not selected:
                    raise ValueError("every architecture needs at least one condition")
                selected_conditions.append(selected)
            if {
                condition.name
                for group in selected_conditions
                for condition in group
            } != {condition.name for condition in conditions}:
                raise ValueError("the experiment grid must include every required condition")
            planned = [
                {"condition": condition.name, **architecture_dimension}
                for architecture_dimension, architecture_conditions in zip(
                    dimensions, selected_conditions, strict=True
                )
                for condition in architecture_conditions
            ]
            run.register_conditions(planned)
        except Exception as error:
            run.mark_condition_failure(
                error,
                condition="setup",
                architecture="setup",
                model_kind="setup",
                n_units=None,
                inhibitory_fraction=None,
                inhibitory_gain=None,
            )
            return run.path
        try:
            task_config = ContextIntegrationConfig(**dict(config["task"]))
            batch = generate_context_integration(task_config, seed=seed)
            train_batch, test_batch = balanced_block_split(
                batch,
                test_fraction=float(config.get("test_fraction", 0.25)),
                seed=derive_seed(seed, "phase2_split"),
                switch_window=_positive_int(
                    training_config.get("switch_window", 1), "switch_window"
                ),
            )
        except Exception as error:
            for record_dimensions in planned:
                run.mark_condition_failure(error, **record_dimensions)
            return run.path
        for architecture, architecture_dimension, architecture_conditions in zip(
            architectures, dimensions, selected_conditions, strict=True
        ):
            for condition in architecture_conditions:
                record_dimensions = {
                    "condition": condition.name,
                    **architecture_dimension,
                }
                try:
                    result = run_context_condition(
                        train_batch,
                        test_batch,
                        condition,
                        architecture,
                        training_config,
                        seed=derive_seed(
                            seed,
                            "phase2_paired_initialization",
                            architecture_dimension["architecture"],
                        ),
                    )
                    run.record(
                        {
                            **result.metrics,
                            "split_unit": "paired_adjacent_scheduling_blocks",
                            "train_trial_count": int(train_batch.inputs.shape[0]),
                            "test_trial_count": int(test_batch.inputs.shape[0]),
                            "actual_test_fraction": float(
                                test_batch.inputs.shape[0] / batch.inputs.shape[0]
                            ),
                        },
                        **record_dimensions,
                    )
                except Exception as error:
                    run.mark_condition_failure(error, **record_dimensions)
        return run.path
