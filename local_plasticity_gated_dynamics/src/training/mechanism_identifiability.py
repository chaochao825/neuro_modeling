"""Strictly paired mechanism-identifiability experiments.

This module is deliberately NumPy-only.  It creates every stochastic and
learned nuisance component once per seed, then replays that contract while
changing only recurrent-plasticity geometry or the explicitly named
plasticity component.  BPTT and GRU baselines live in :mod:`src.baselines`
and are invoked only by the experiment driver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.analysis.rank_metrics import participation_ratio, rank_summary
from src.analysis.switching_metrics import jacobian_spectrum_summary
from src.models.ei_rate_network import EIRateNetwork
from src.plasticity.inhibitory_homeostasis import InhibitoryHomeostasis
from src.plasticity.three_factor import ThreeFactorRule
from src.plasticity.update_budget import UpdateBudgetController, UpdateBudgetSummary
from src.tasks.context_integration import ContextIntegrationBatch
from src.training.context_local import _activity_representation_metrics, _trial_behavior
from src.utils.reproducibility import derive_seed, make_rng


Array = np.ndarray
FeedbackMode = Literal["aligned", "random", "orthogonal", "shuffled", "full"]
BudgetNorm = Literal["l1", "l2"]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _digest(*values: object) -> str:
    """Return a stable SHA-256 digest including array shape and dtype."""

    digest = sha256()
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


def _batch_data_id(batch: ContextIntegrationBatch) -> str:
    """Fingerprint every trial-level value consumed by a causal branch."""

    return _digest(
        batch.inputs,
        batch.targets,
        batch.loss_mask,
        batch.contexts,
        batch.choices,
        batch.coherences,
        batch.trial_ids,
        batch.block_ids,
        batch.epoch,
        batch.time_ms,
        batch.config,
    )


@dataclass(frozen=True)
class MechanismCondition:
    """One causal P0 cell with independent mechanism switches."""

    name: str
    feedback_mode: FeedbackMode
    task_plasticity_enabled: bool
    homeostasis_enabled: bool
    normalization_enabled: bool
    budget_norm: BudgetNorm

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("condition name must be non-empty")
        if self.feedback_mode not in {
            "aligned",
            "random",
            "orthogonal",
            "shuffled",
            "full",
        }:
            raise ValueError("invalid feedback_mode")
        if self.budget_norm not in {"l1", "l2"}:
            raise ValueError("budget_norm must be 'l1' or 'l2'")
        for field in (
            "task_plasticity_enabled",
            "homeostasis_enabled",
            "normalization_enabled",
        ):
            if not isinstance(getattr(self, field), (bool, np.bool_)):
                raise TypeError(f"{field} must be boolean")
        if not self.task_plasticity_enabled and self.feedback_mode != "aligned":
            raise ValueError(
                "feedback geometry is inapplicable without task plasticity"
            )

    @property
    def mechanism(self) -> str:
        components = []
        if self.task_plasticity_enabled:
            components.append("task")
        if self.homeostasis_enabled:
            components.append("homeostasis")
        if self.normalization_enabled:
            components.append("normalization")
        if not components:
            return "frozen-recurrent"
        if len(components) == 1:
            return f"{components[0]}-only"
        return "+".join(components)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "mechanism": self.mechanism}


def build_mechanism_conditions(
    budget_norms: Sequence[BudgetNorm] = ("l1", "l2"),
) -> list[MechanismCondition]:
    """Build geometry and component panels without duplicate causal cells."""

    normalized = tuple(budget_norms)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("budget_norms must be a non-empty unique sequence")
    if any(item not in {"l1", "l2"} for item in normalized):
        raise ValueError("budget_norms may contain only 'l1' and 'l2'")
    conditions: list[MechanismCondition] = []
    for norm in normalized:
        # Primary geometry panel: every branch shares the exact same yoked
        # homeostatic tape and fan-in normalization policy.  The only changed
        # factor is the task-credit projector (or the shuffled correspondence).
        for feedback in ("aligned", "random", "orthogonal", "shuffled", "full"):
            conditions.append(
                MechanismCondition(
                    f"task-homeostasis-normalization__{feedback}__{norm}",
                    feedback,
                    True,
                    True,
                    True,
                    norm,
                )
            )
        for feedback in ("aligned", "random", "orthogonal", "shuffled", "full"):
            conditions.append(
                MechanismCondition(
                    f"task-only__{feedback}__{norm}",
                    feedback,
                    True,
                    False,
                    False,
                    norm,
                )
            )
        # Explicit component panel.  The paired cells expose normalization
        # instead of hiding it inside the task/homeostasis labels.
        conditions.extend(
            [
                MechanismCondition(
                    f"task-homeostasis__aligned__{norm}",
                    "aligned",
                    True,
                    True,
                    False,
                    norm,
                ),
                MechanismCondition(
                    f"task-normalization__aligned__{norm}",
                    "aligned",
                    True,
                    False,
                    True,
                    norm,
                ),
                MechanismCondition(
                    f"homeostasis-only__aligned__{norm}",
                    "aligned",
                    False,
                    True,
                    False,
                    norm,
                ),
                MechanismCondition(
                    f"homeostasis-normalization__aligned__{norm}",
                    "aligned",
                    False,
                    True,
                    True,
                    norm,
                ),
                MechanismCondition(
                    f"normalization-only__aligned__{norm}",
                    "aligned",
                    False,
                    False,
                    True,
                    norm,
                ),
                MechanismCondition(
                    f"frozen-recurrent__aligned__{norm}",
                    "aligned",
                    False,
                    False,
                    False,
                    norm,
                ),
            ]
        )
    return conditions


@dataclass(frozen=True)
class PairedTrialTape:
    """Compact deterministic replay contract for orders, noise, and shuffling."""

    seed: int
    trial_orders: tuple[Array, ...]
    shuffled_source_indices: Array
    train_trial_ids: Array
    test_trial_ids: Array
    train_data_id: str
    test_data_id: str
    noise_seed: int
    evaluation_noise_seed: int

    @classmethod
    def create(
        cls,
        train: ContextIntegrationBatch,
        test: ContextIntegrationBatch,
        *,
        epochs: int,
        seed: int,
        shuffle_trial_order: bool,
    ) -> "PairedTrialTape":
        epochs = _positive_int(epochs, "epochs")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(shuffle_trial_order, (bool, np.bool_)):
            raise TypeError("shuffle_trial_order must be boolean")
        rng = make_rng(int(seed), "p0-trial-order")
        base = np.arange(train.inputs.shape[0], dtype=int)
        orders = tuple(
            (rng.permutation(base) if shuffle_trial_order else base.copy())
            for _ in range(epochs)
        )
        source = base.copy()
        # Shuffle only within a block/context stratum.  This preserves target
        # marginals and changes temporal credit correspondence alone.
        shuffle_rng = make_rng(int(seed), "p0-feedback-shuffle")
        for block in np.unique(train.block_ids):
            block_mask = train.block_ids == block
            for context in np.unique(train.contexts[block_mask]):
                indices = np.flatnonzero(block_mask & (train.contexts == context))
                if indices.size < 2:
                    raise ValueError(
                        "every block/context shuffle stratum must contain at least two trials"
                    )
                permutation = indices.copy()
                for _ in range(100):
                    permutation = shuffle_rng.permutation(indices)
                    if not np.any(permutation == indices):
                        break
                else:
                    shift = int(shuffle_rng.integers(1, indices.size))
                    permutation = np.roll(indices, shift)
                source[indices] = permutation
        arrays = [*orders, source]
        for array in arrays:
            array.setflags(write=False)
        train_ids = np.array(train.trial_ids, dtype=int, copy=True)
        test_ids = np.array(test.trial_ids, dtype=int, copy=True)
        train_ids.setflags(write=False)
        test_ids.setflags(write=False)
        return cls(
            seed=int(seed),
            trial_orders=orders,
            shuffled_source_indices=source,
            train_trial_ids=train_ids,
            test_trial_ids=test_ids,
            train_data_id=_batch_data_id(train),
            test_data_id=_batch_data_id(test),
            noise_seed=derive_seed(int(seed), "p0-shared-training-noise"),
            evaluation_noise_seed=derive_seed(int(seed), "p0-shared-evaluation-noise"),
        )

    @property
    def trial_order_id(self) -> str:
        return _digest(*self.trial_orders, self.train_data_id)

    @property
    def shuffled_source_id(self) -> str:
        return _digest(self.shuffled_source_indices)

    @property
    def split_id(self) -> str:
        return _digest(self.train_data_id, self.test_data_id)

    @property
    def replay_contract_id(self) -> str:
        return _digest(
            self.trial_order_id,
            self.shuffled_source_id,
            self.split_id,
            self.noise_id,
        )

    @property
    def noise_id(self) -> str:
        return _digest(self.noise_seed, self.evaluation_noise_seed)

    def training_noise(
        self, epoch: int, trial_index: int, n_steps: int, n_units: int, std: float
    ) -> Array:
        std = _nonnegative_float(std, "noise_std")
        if std == 0.0:
            return np.zeros((n_steps, n_units), dtype=float)
        rng = make_rng(self.noise_seed, "epoch", int(epoch), "trial", int(trial_index))
        return rng.normal(scale=std, size=(n_steps, n_units))

    def evaluation_noise(
        self, trial_index: int, n_steps: int, n_units: int, std: float
    ) -> Array:
        std = _nonnegative_float(std, "evaluation_noise_std")
        if std == 0.0:
            return np.zeros((n_steps, n_units), dtype=float)
        rng = make_rng(self.evaluation_noise_seed, "trial", int(trial_index))
        return rng.normal(scale=std, size=(n_steps, n_units))


@dataclass(frozen=True)
class FeedbackDesign:
    """One shared high-span encoder and deterministic credit projectors."""

    shared_encoder: Array
    task_basis: Array
    aligned_basis: Array
    random_basis: Array
    orthogonal_basis: Array

    @property
    def encoder_id(self) -> str:
        return _digest(self.shared_encoder)

    @property
    def design_id(self) -> str:
        return _digest(
            self.shared_encoder,
            self.task_basis,
            self.aligned_basis,
            self.random_basis,
            self.orthogonal_basis,
        )

    def basis(self, mode: FeedbackMode) -> Array | None:
        if mode in {"aligned", "shuffled"}:
            return self.aligned_basis
        if mode == "random":
            return self.random_basis
        if mode == "orthogonal":
            return self.orthogonal_basis
        if mode == "full":
            return None
        raise ValueError(f"unknown feedback mode: {mode}")

    def project(self, source: Array, mode: FeedbackMode) -> Array:
        vector = np.asarray(source, dtype=float)
        if vector.shape != (self.shared_encoder.shape[0],):
            raise ValueError("source has the wrong neural dimension")
        basis = self.basis(mode)
        return vector.copy() if basis is None else basis @ (basis.T @ vector)


@dataclass(frozen=True)
class PairedResources:
    """Everything that must be identical before a causal branch."""

    architecture: dict[str, Any]
    network_seed: int
    readout: Array
    readout_bias: float
    readout_fit_trial_ids: Array
    oracle_gains: Array
    feedback: FeedbackDesign
    tape: PairedTrialTape
    reference_rate_tape: Array
    feedback_signal_tape: Array
    shuffled_feedback_signal_tape: Array
    feedback_signal_tape_id: str
    shuffled_feedback_signal_tape_id: str
    feedback_encoder_rank: int
    feedback_signal_span: int
    shuffled_feedback_signal_span: int
    homeostasis_reference_weights: Array
    training_noise_std: float
    evaluation_noise_std: float
    feedback_scale: float
    homeostasis_target_rate: float
    noise_contract_id: str
    replay_contract_id: str
    initialization_id: str
    readout_training_id: str
    gate_id: str
    homeostasis_signal_id: str


@dataclass
class _StageAccumulator:
    raw: Array
    masked: Array
    dale: Array
    normalization: Array
    task: Array
    homeostasis: Array
    task_path_l1: float = 0.0
    task_path_l2: float = 0.0
    homeostasis_path_l1: float = 0.0
    homeostasis_path_l2: float = 0.0
    normalization_path_l1: float = 0.0
    normalization_path_l2: float = 0.0
    task_events: int = 0
    homeostasis_events: int = 0

    @classmethod
    def zeros(cls, n_units: int) -> "_StageAccumulator":
        zero = np.zeros((n_units, n_units), dtype=float)
        return cls(*(zero.copy() for _ in range(6)))

    def add_task(
        self,
        *,
        raw: Array,
        masked: Array,
        dale: Array,
        normalization: Array,
    ) -> None:
        self.raw += raw
        self.masked += masked
        self.dale += dale
        self.normalization += normalization
        self.task += dale
        self.task_path_l1 += float(np.sum(np.abs(dale)))
        self.task_path_l2 += float(np.linalg.norm(dale))
        self.normalization_path_l1 += float(np.sum(np.abs(normalization)))
        self.normalization_path_l2 += float(np.linalg.norm(normalization))
        self.task_events += 1

    def add_homeostasis(self, *, dale: Array, normalization: Array) -> None:
        self.homeostasis += dale
        self.normalization += normalization
        self.homeostasis_path_l1 += float(np.sum(np.abs(dale)))
        self.homeostasis_path_l2 += float(np.linalg.norm(dale))
        self.normalization_path_l1 += float(np.sum(np.abs(normalization)))
        self.normalization_path_l2 += float(np.linalg.norm(normalization))
        self.homeostasis_events += 1


@dataclass(frozen=True)
class MechanismResult:
    metrics: dict[str, Any]
    predictions: Array
    activity: Array
    trial_correct: Array


def _make_network(architecture: Mapping[str, Any], *, seed: int) -> EIRateNetwork:
    if architecture.get("kind") != "ei":
        raise ValueError(
            "P0 mechanism-identifiability currently requires an E/I architecture"
        )
    return EIRateNetwork(
        _positive_int(architecture.get("n_units"), "n_units"),
        n_inputs=4,
        excitatory_fraction=float(architecture.get("excitatory_fraction", 0.8)),
        connection_probability=float(architecture.get("connection_probability", 0.1)),
        tau_e=float(architecture.get("tau_e", 20.0)),
        tau_i=float(architecture.get("tau_i", 10.0)),
        dt=float(architecture.get("dt", 1.0)),
        bulk_gain=float(architecture.get("bulk_gain", 0.8)),
        inhibitory_gain=float(architecture.get("inhibitory_gain", 1.0)),
        input_scale=float(architecture.get("input_scale", 1.0)),
        activation=str(architecture.get("activation", "rectified_tanh")),
        allow_self_connections=bool(architecture.get("allow_self_connections", False)),
        normalize_fan_in_after_update=bool(
            architecture.get("normalize_fan_in_after_update", True)
        ),
        seed=seed,
    )


def _oracle_gain_patterns(n_units: int, strength: float) -> Array:
    indices = np.arange(n_units)
    patterns = np.column_stack((indices % 2 == 0, indices % 2 == 1)).astype(float)
    return 1.0 + strength * patterns


def _run_frozen_activity(
    network: EIRateNetwork,
    batch: ContextIntegrationBatch,
    gains: Array,
    *,
    noise_getter: Any | None = None,
) -> tuple[Array, Array]:
    n_trials, n_steps = batch.inputs.shape[:2]
    activity = np.empty((n_trials, n_steps, network.n_units), dtype=float)
    x = np.empty_like(activity)
    for trial in range(n_trials):
        state = network.initial_state()
        noise = (
            np.zeros((n_steps, network.n_units), dtype=float)
            if noise_getter is None
            else noise_getter(trial)
        )
        gain = gains[:, int(batch.contexts[trial])]
        for time in range(n_steps):
            state = network.step(
                state,
                batch.inputs[trial, time],
                gain=gain,
                noise=noise[time],
            ).state
            x[trial, time] = state.x
            activity[trial, time] = state.rates
    return x, activity


def _fit_common_readout(
    activity: Array,
    batch: ContextIntegrationBatch,
    *,
    ridge: float,
) -> tuple[Array, float]:
    mask = np.asarray(batch.loss_mask, dtype=bool)
    design = activity[mask]
    target = batch.targets[..., 0][mask]
    if design.shape[0] < 2:
        raise ValueError("readout fitting requires at least two training samples")
    model = Ridge(alpha=ridge, fit_intercept=True)
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=float).copy(), float(model.intercept_)


def _orthonormal_columns(matrix: Array, n_columns: int) -> Array:
    if matrix.ndim != 2 or matrix.shape[0] < n_columns:
        raise ValueError("cannot construct the requested orthonormal basis")
    q, _ = np.linalg.qr(matrix, mode="reduced")
    if q.shape[1] < n_columns:
        raise ValueError("basis proposal has insufficient rank")
    return q[:, :n_columns]


def _make_feedback_design(
    network: EIRateNetwork,
    readout: Array,
    *,
    feedback_dim: int,
    seed: int,
) -> FeedbackDesign:
    n_units = network.n_units
    feedback_dim = min(_positive_int(feedback_dim, "feedback_dim"), n_units)
    if feedback_dim > n_units - 1:
        raise ValueError("feedback_dim must leave at least one orthogonal direction")
    rng = make_rng(seed, "p0-feedback-design")
    proposal = np.column_stack(
        [readout, network.input_weights, rng.normal(size=(n_units, feedback_dim))]
    )
    task_basis = _orthonormal_columns(proposal, feedback_dim)
    if float(task_basis[:, 0] @ readout) < 0.0:
        task_basis[:, 0] *= -1.0
    complement_proposal = rng.normal(size=(n_units, max(feedback_dim, 2)))
    complement_proposal -= task_basis @ (task_basis.T @ complement_proposal)
    orthogonal_basis = _orthonormal_columns(complement_proposal, feedback_dim)
    random_basis = _orthonormal_columns(
        rng.normal(size=(n_units, feedback_dim)), feedback_dim
    )
    # The first aligned channel is the exact transpose-readout direction: for
    # scalar squared error this is the immediate task-gradient direction at
    # the population output.  Remaining channels carry weaker input-feature
    # credit.  Orthogonal feedback removes this component by construction.
    task_encoder = np.zeros((feedback_dim, 5), dtype=float)
    task_encoder[0, 0] = 1.0
    for index in range(1, feedback_dim):
        task_encoder[index, 1 + (index - 1) % 4] = 0.25
    low_dimensional_encoder = task_basis @ task_encoder
    # Add a shared state-dependent channel outside the aligned task basis.
    # This makes identity (``full``) feedback capable of expressing a genuinely
    # high-dimensional signal, while every geometry still receives the exact
    # same unprojected source before its projector is applied.
    state_encoder = 0.10 * (np.eye(n_units, dtype=float) - task_basis @ task_basis.T)
    shared_encoder = np.column_stack((low_dimensional_encoder, state_encoder))
    for array in (
        shared_encoder,
        task_basis,
        random_basis,
        orthogonal_basis,
    ):
        array.setflags(write=False)
    return FeedbackDesign(
        shared_encoder=shared_encoder,
        task_basis=task_basis,
        aligned_basis=task_basis,
        random_basis=random_basis,
        orthogonal_basis=orthogonal_basis,
    )


def _normalized_feedback_source(
    feedback: FeedbackDesign,
    *,
    error: float,
    input_t: Array,
    reference_rates: Array,
    scale: float,
) -> Array:
    features = np.concatenate(
        [
            np.ones(1),
            np.asarray(input_t, dtype=float),
            np.asarray(reference_rates, dtype=float),
        ]
    )
    if features.shape != (feedback.shared_encoder.shape[1],):
        raise ValueError("feedback features do not match the shared encoder")
    source = feedback.shared_encoder @ features
    norm = float(np.linalg.norm(source))
    if norm > 1.0:
        source = source / norm
    return scale * float(np.clip(error, -2.0, 2.0)) * source


def _precompute_reference_tapes(
    reference: EIRateNetwork,
    train: ContextIntegrationBatch,
    gains: Array,
    tape: PairedTrialTape,
    feedback: FeedbackDesign,
    readout: Array,
    bias: float,
    *,
    noise_std: float,
    feedback_scale: float,
) -> tuple[Array, Array, Array]:
    """Materialize the frozen reference branch once for all causal cells."""

    epochs = len(tape.trial_orders)
    n_trials, n_steps = train.inputs.shape[:2]
    n_units = reference.n_units
    rates = np.empty((epochs, n_trials, n_steps, n_units), dtype=float)
    signals = np.empty_like(rates)
    for epoch in range(epochs):
        for trial in range(n_trials):
            state = reference.initial_state()
            gain = gains[:, int(train.contexts[trial])]
            noise = tape.training_noise(epoch, trial, n_steps, n_units, noise_std)
            for time in range(n_steps):
                state = reference.step(
                    state,
                    train.inputs[trial, time],
                    gain=gain,
                    noise=noise[time],
                ).state
                rates[epoch, trial, time] = state.rates
                prediction = float(readout @ state.rates + bias)
                error = float(train.targets[trial, time, 0]) - prediction
                signals[epoch, trial, time] = _normalized_feedback_source(
                    feedback,
                    error=error,
                    input_t=train.inputs[trial, time],
                    reference_rates=state.rates,
                    scale=feedback_scale,
                )
    # Permute the *entire* precomputed third-factor signal within each
    # block/context stratum.  This preserves the empirical signal marginal
    # exactly and destroys only its correspondence to the current trial.
    shuffled_signals = np.array(
        signals[:, tape.shuffled_source_indices, :, :], dtype=float, copy=True
    )
    for array in (rates, signals, shuffled_signals):
        array.setflags(write=False)
    return rates, signals, shuffled_signals


def prepare_paired_resources(
    train: ContextIntegrationBatch,
    test: ContextIntegrationBatch,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    seed: int,
) -> PairedResources:
    """Create the branch point once and fingerprint real array contents."""

    if train.config != test.config:
        raise ValueError("train and test must come from the same task configuration")
    if np.intersect1d(train.block_ids, test.block_ids).size:
        raise ValueError("train/test block leakage detected")
    epochs = _positive_int(training.get("train_epochs", 1), "train_epochs")
    tape = PairedTrialTape.create(
        train,
        test,
        epochs=epochs,
        seed=derive_seed(seed, "p0-tape"),
        shuffle_trial_order=bool(training.get("shuffle_trial_order", False)),
    )
    network_seed = derive_seed(seed, "p0-network")
    reference = _make_network(architecture, seed=network_seed)
    gain_strength = _nonnegative_float(
        training.get("gate_strength", 0.25), "gate_strength"
    )
    gains = _oracle_gain_patterns(reference.n_units, gain_strength)
    ordered_blocks = list(dict.fromkeys(train.block_ids.tolist()))
    readout_fit_blocks = _positive_int(
        training.get("readout_fit_blocks", len(ordered_blocks)),
        "readout_fit_blocks",
    )
    if readout_fit_blocks > len(ordered_blocks):
        raise ValueError("readout_fit_blocks exceeds the available training blocks")
    calibration_blocks = ordered_blocks[:readout_fit_blocks]
    calibration_indices = np.flatnonzero(np.isin(train.block_ids, calibration_blocks))
    calibration = train.subset(calibration_indices)
    if set(calibration.contexts.tolist()) != {0, 1}:
        raise ValueError("readout calibration blocks must contain both contexts")
    _, readout_activity = _run_frozen_activity(reference, calibration, gains)
    readout, bias = _fit_common_readout(
        readout_activity,
        calibration,
        ridge=_positive_float(training.get("readout_ridge", 1e-3), "readout_ridge"),
    )
    feedback = _make_feedback_design(
        reference,
        readout,
        feedback_dim=_positive_int(training.get("feedback_dim", 4), "feedback_dim"),
        seed=derive_seed(seed, "p0-feedback"),
    )
    training_noise_std = _nonnegative_float(
        training.get("network_noise_std", 0.0), "network_noise_std"
    )
    evaluation_noise_std = _nonnegative_float(
        training.get("evaluation_noise_std", training_noise_std),
        "evaluation_noise_std",
    )
    feedback_scale = _positive_float(
        training.get("feedback_scale", 0.1), "feedback_scale"
    )
    homeostasis_target_rate = _nonnegative_float(
        training.get("target_rate", 0.1), "target_rate"
    )
    reference_rates, feedback_signals, shuffled_feedback_signals = (
        _precompute_reference_tapes(
            reference,
            train,
            gains,
            tape,
            feedback,
            readout,
            bias,
            noise_std=training_noise_std,
            feedback_scale=feedback_scale,
        )
    )
    repeated_loss_mask = np.broadcast_to(train.loss_mask, feedback_signals.shape[:-1])
    signal_samples = feedback_signals[repeated_loss_mask]
    shuffled_signal_samples = shuffled_feedback_signals[repeated_loss_mask]
    feedback_signal_span = int(np.linalg.matrix_rank(signal_samples))
    shuffled_feedback_signal_span = int(np.linalg.matrix_rank(shuffled_signal_samples))
    homeostasis_reference_weights = np.array(
        reference._effective_weights_for_learning(), dtype=float, copy=True
    )
    homeostasis_reference_weights.setflags(write=False)
    architecture_copy = dict(architecture)
    initialization_id = _digest(
        reference.W_bulk,
        reference.input_weights,
        reference.connectivity_mask,
        reference.presynaptic_signs,
    )
    readout_training_id = _digest(
        readout,
        bias,
        calibration.trial_ids,
        calibration.loss_mask,
        training.get("readout_ridge", 1e-3),
    )
    gate_id = _digest(gains, train.contexts, test.contexts)
    homeostasis_signal_id = _digest(
        reference_rates,
        homeostasis_reference_weights,
        homeostasis_target_rate,
    )
    feedback_signal_tape_id = _digest(feedback_signals)
    shuffled_feedback_signal_tape_id = _digest(shuffled_feedback_signals)
    noise_contract_id = _digest(tape.noise_id, training_noise_std, evaluation_noise_std)
    replay_contract_id = _digest(
        tape.replay_contract_id,
        noise_contract_id,
        feedback_signal_tape_id,
        shuffled_feedback_signal_tape_id,
        homeostasis_signal_id,
    )
    for array in (readout, gains):
        array.setflags(write=False)
    readout_fit_trial_ids = np.array(calibration.trial_ids, dtype=int, copy=True)
    readout_fit_trial_ids.setflags(write=False)
    return PairedResources(
        architecture=architecture_copy,
        network_seed=network_seed,
        readout=readout,
        readout_bias=bias,
        readout_fit_trial_ids=readout_fit_trial_ids,
        oracle_gains=gains,
        feedback=feedback,
        tape=tape,
        reference_rate_tape=reference_rates,
        feedback_signal_tape=feedback_signals,
        shuffled_feedback_signal_tape=shuffled_feedback_signals,
        feedback_signal_tape_id=feedback_signal_tape_id,
        shuffled_feedback_signal_tape_id=shuffled_feedback_signal_tape_id,
        feedback_encoder_rank=int(np.linalg.matrix_rank(feedback.shared_encoder)),
        feedback_signal_span=feedback_signal_span,
        shuffled_feedback_signal_span=shuffled_feedback_signal_span,
        homeostasis_reference_weights=homeostasis_reference_weights,
        training_noise_std=training_noise_std,
        evaluation_noise_std=evaluation_noise_std,
        feedback_scale=feedback_scale,
        homeostasis_target_rate=homeostasis_target_rate,
        noise_contract_id=noise_contract_id,
        replay_contract_id=replay_contract_id,
        initialization_id=initialization_id,
        readout_training_id=readout_training_id,
        gate_id=gate_id,
        homeostasis_signal_id=homeostasis_signal_id,
    )


def _planned_update_events(
    batch: ContextIntegrationBatch, *, epochs: int, interval: int
) -> int:
    """Count causal update slots under the exact global-step schedule."""

    interval = _positive_int(interval, "update interval")
    n_steps = batch.inputs.shape[1]
    count = 0
    global_step = 0
    for _ in range(epochs):
        for _trial in range(batch.inputs.shape[0]):
            for time in range(n_steps):
                if batch.loss_mask[_trial, time] and global_step % interval == 0:
                    count += 1
                global_step += 1
    if count < 1:
        raise ValueError("the update schedule contains no loss-bearing event")
    return count


def _controller_metrics(
    prefix: str, summary: UpdateBudgetSummary | None
) -> dict[str, Any]:
    if summary is None:
        return {
            f"{prefix}_budget_enabled": False,
            f"{prefix}_budget_selected_norm": None,
            f"{prefix}_budget_target": 0.0,
            f"{prefix}_budget_selected_applied": 0.0,
            f"{prefix}_budget_attained": True,
            f"{prefix}_budget_final_shortfall": 0.0,
            f"{prefix}_budget_planned_events": 0,
            f"{prefix}_budget_processed_events": 0,
            f"{prefix}_budget_secondary_norm_diagnostic_only": True,
        }
    values = summary.as_dict()
    return {
        f"{prefix}_budget_enabled": True,
        **{f"{prefix}_budget_{key}": value for key, value in values.items()},
    }


def _scaled_stage(proposed: Array, budgeted: Array, stage: Array) -> Array:
    denominator = float(np.linalg.norm(proposed))
    if denominator == 0.0:
        return np.zeros_like(stage)
    factor = float(np.linalg.norm(budgeted)) / denominator
    # UpdateBudgetController never amplifies.  Keep the invariant explicit at
    # this second audit boundary because the stage arrays feed rank claims.
    if factor > 1.0 + 1e-12:
        raise RuntimeError("budget controller amplified a Dale-applied update")
    return stage * min(factor, 1.0)


def _post_derivative(network: EIRateNetwork, state_x: Array, gain: Array) -> Array:
    derivative = gain * (1.0 - np.tanh(gain * state_x) ** 2)
    if network.activation_name == "rectified_tanh":
        derivative *= state_x > 0.0
    return derivative


def _evaluate_network(
    network: EIRateNetwork,
    batch: ContextIntegrationBatch,
    resources: PairedResources,
    *,
    evaluation_noise_std: float,
) -> tuple[Array, Array, Array, Array]:
    n_trials, n_steps = batch.inputs.shape[:2]
    predictions = np.empty((n_trials, n_steps), dtype=float)
    x = np.empty((n_trials, n_steps, network.n_units), dtype=float)
    activity = np.empty_like(x)
    gains = np.empty_like(x)
    for trial in range(n_trials):
        state = network.initial_state()
        gain = resources.oracle_gains[:, int(batch.contexts[trial])]
        noise = resources.tape.evaluation_noise(
            trial,
            n_steps,
            network.n_units,
            evaluation_noise_std,
        )
        for time in range(n_steps):
            state = network.step(
                state,
                batch.inputs[trial, time],
                gain=gain,
                noise=noise[time],
            ).state
            predictions[trial, time] = (
                resources.readout @ state.rates + resources.readout_bias
            )
            x[trial, time] = state.x
            activity[trial, time] = state.rates
            gains[trial, time] = gain
    return predictions, x, activity, gains


def _jacobian_metrics(network: EIRateNetwork, x: Array, gains: Array) -> dict[str, Any]:
    mean_x = np.mean(x.reshape(-1, network.n_units), axis=0)
    mean_gain = np.mean(gains.reshape(-1, network.n_units), axis=0)
    derivative = _post_derivative(network, mean_x, mean_gain)
    jacobian = -np.eye(network.n_units) + network.recurrent_weights @ np.diag(
        derivative
    )
    jacobian = jacobian / network.time_constants[:, None]
    summary = jacobian_spectrum_summary(jacobian, dynamics="continuous")
    return {
        "jacobian_spectral_radius": summary.spectral_radius,
        "jacobian_max_real_part": summary.max_real_part,
        "jacobian_stability_margin": summary.stability_margin,
        "jacobian_unstable_fraction": summary.unstable_fraction,
        "jacobian_eigenvalues_real": np.real(summary.eigenvalues).tolist(),
        "jacobian_eigenvalues_imag": np.imag(summary.eigenvalues).tolist(),
    }


def _rank_metrics(stages: _StageAccumulator, feedback_dim: int) -> dict[str, Any]:
    matrices = {
        "raw": stages.raw,
        "masked": stages.masked,
        "dale_applied": stages.dale,
        "normalization_correction": stages.normalization,
        "task_component": stages.task,
        "homeostasis_component": stages.homeostasis,
    }
    metrics: dict[str, Any] = {}
    for name, matrix in matrices.items():
        summary = rank_summary(matrix, k=min(feedback_dim, min(matrix.shape)))
        metrics.update(
            {
                f"{name}_numerical_rank": summary.numerical_rank,
                f"{name}_effective_rank": summary.effective_rank,
                f"{name}_top_k_singular_energy": summary.top_k_singular_energy,
            }
        )
    return metrics


def run_mechanism_condition(
    train: ContextIntegrationBatch,
    test: ContextIntegrationBatch,
    resources: PairedResources,
    condition: MechanismCondition,
    training: Mapping[str, Any],
) -> MechanismResult:
    """Run one causal branch under the immutable paired resource contract."""

    if not isinstance(condition, MechanismCondition):
        raise TypeError("condition must be a MechanismCondition")
    if train.config != test.config:
        raise ValueError("train and test configurations differ")
    if _batch_data_id(train) != resources.tape.train_data_id:
        raise ValueError("train data do not match the paired tape")
    if _batch_data_id(test) != resources.tape.test_data_id:
        raise ValueError("test data do not match the paired tape")
    if _digest(resources.feedback_signal_tape) != resources.feedback_signal_tape_id:
        raise RuntimeError("feedback signal tape was mutated after preparation")
    if (
        _digest(resources.shuffled_feedback_signal_tape)
        != resources.shuffled_feedback_signal_tape_id
    ):
        raise RuntimeError(
            "shuffled feedback signal tape was mutated after preparation"
        )

    architecture = {
        **resources.architecture,
        "normalize_fan_in_after_update": condition.normalization_enabled,
    }
    network = _make_network(architecture, seed=resources.network_seed)
    initial_weights = network.recurrent_weights
    initial_id = _digest(
        network.W_bulk,
        network.input_weights,
        network.connectivity_mask,
        network.presynaptic_signs,
    )
    if initial_id != resources.initialization_id:
        raise RuntimeError(
            "condition branch did not reproduce the paired initialization"
        )
    epochs = len(resources.tape.trial_orders)
    update_interval = _positive_int(
        training.get("update_interval", 1), "update_interval"
    )
    homeostasis_interval = _positive_int(
        training.get("homeostasis_interval", update_interval), "homeostasis_interval"
    )
    task_events = _planned_update_events(train, epochs=epochs, interval=update_interval)
    homeostasis_events = _planned_update_events(
        train, epochs=epochs, interval=homeostasis_interval
    )
    budgets_by_norm = training.get("total_update_budget_by_norm")
    if budgets_by_norm is not None:
        if not isinstance(budgets_by_norm, Mapping):
            raise TypeError("total_update_budget_by_norm must be a mapping")
        if set(budgets_by_norm) != {"l1", "l2"}:
            raise ValueError(
                "total_update_budget_by_norm must contain exactly l1 and l2"
            )
        total_budget = _positive_float(
            budgets_by_norm[condition.budget_norm],
            f"total_update_budget_by_norm[{condition.budget_norm}]",
        )
    else:
        total_budget = _positive_float(
            training.get("total_update_budget", 1e-3), "total_update_budget"
        )
    # ``total_budget`` is the per-mechanism target.  Consequently task-only
    # and homeostasis-only are norm matched, while task+homeostasis replays the
    # exact same homeostatic budget and adds one equally sized task budget.
    # This permits both strict on/off attribution and a matched task-vs-homeo
    # comparison without silently halving either mechanism.
    task_target = total_budget if condition.task_plasticity_enabled else 0.0
    homeostasis_target = total_budget if condition.homeostasis_enabled else 0.0
    budget_tolerance = _nonnegative_float(
        training.get("budget_tolerance", 1e-9), "budget_tolerance"
    )
    task_budget = (
        UpdateBudgetController(
            task_target,
            condition.budget_norm,
            task_events,
            tolerance=budget_tolerance,
        )
        if condition.task_plasticity_enabled
        else None
    )
    homeostasis_budget = (
        UpdateBudgetController(
            homeostasis_target,
            condition.budget_norm,
            homeostasis_events,
            tolerance=budget_tolerance,
        )
        if condition.homeostasis_enabled
        else None
    )
    rule = ThreeFactorRule(
        learning_rate=_positive_float(
            training.get("recurrent_learning_rate", 1e-3),
            "recurrent_learning_rate",
        ),
        tau_eligibility=_positive_float(
            training.get("tau_eligibility", 5.0), "tau_eligibility"
        ),
        dt=1.0,
        weight_decay=_nonnegative_float(
            training.get("weight_decay", 0.0), "weight_decay"
        ),
    )
    homeostasis_rule = InhibitoryHomeostasis(
        learning_rate=_positive_float(
            training.get("homeostasis_learning_rate", 1e-4),
            "homeostasis_learning_rate",
        ),
        target_rate=_nonnegative_float(training.get("target_rate", 0.1), "target_rate"),
        dt=1.0,
        max_abs_update=training.get("homeostasis_max_abs_update", None),
    )
    noise_std = _nonnegative_float(
        training.get("network_noise_std", 0.0), "network_noise_std"
    )
    feedback_scale = _positive_float(
        training.get("feedback_scale", 0.1), "feedback_scale"
    )
    target_rate = _nonnegative_float(training.get("target_rate", 0.1), "target_rate")
    if noise_std != resources.training_noise_std:
        raise ValueError("network_noise_std differs from the prepared replay tape")
    if feedback_scale != resources.feedback_scale:
        raise ValueError("feedback_scale differs from the prepared replay tape")
    if target_rate != resources.homeostasis_target_rate:
        raise ValueError("target_rate differs from the prepared replay tape")
    evaluation_noise_std = _nonnegative_float(
        training.get("evaluation_noise_std", noise_std), "evaluation_noise_std"
    )
    if evaluation_noise_std != resources.evaluation_noise_std:
        raise ValueError("evaluation_noise_std differs from the prepared replay tape")
    stages = _StageAccumulator.zeros(network.n_units)
    global_step = 0
    for epoch, order in enumerate(resources.tape.trial_orders):
        for raw_trial in order:
            trial = int(raw_trial)
            state = network.initial_state()
            rule.reset(network.n_units)
            gain = resources.oracle_gains[:, int(train.contexts[trial])]
            noise = resources.tape.training_noise(
                epoch,
                trial,
                train.inputs.shape[1],
                network.n_units,
                noise_std,
            )
            for time in range(train.inputs.shape[1]):
                pre_activity = state.rates.copy()
                state = network.step(
                    state,
                    train.inputs[trial, time],
                    gain=gain,
                    noise=noise[time],
                ).state
                eligible = bool(train.loss_mask[trial, time])
                if condition.task_plasticity_enabled:
                    if eligible and global_step % update_interval == 0:
                        signal_tape = (
                            resources.shuffled_feedback_signal_tape
                            if condition.feedback_mode == "shuffled"
                            else resources.feedback_signal_tape
                        )
                        shared_source = signal_tape[epoch, trial, time]
                        modulator = resources.feedback.project(
                            shared_source, condition.feedback_mode
                        )
                        proposal = rule._propose_trusted(
                            pre_activity,
                            modulator,
                            post_derivative=_post_derivative(network, state.x, gain),
                            connectivity_mask=network.connectivity_mask,
                            presynaptic_signs=network.presynaptic_signs,
                            current_weights=network._effective_weights_for_learning(),
                            current_task_weights=network._task_weights_for_learning(),
                        )
                        if task_budget is None:
                            raise RuntimeError("task budget was not initialized")
                        budgeted = task_budget.scale(proposal.dale_applied_update)
                        raw = _scaled_stage(
                            proposal.dale_applied_update, budgeted, proposal.raw_update
                        )
                        masked = _scaled_stage(
                            proposal.dale_applied_update,
                            budgeted,
                            proposal.masked_update,
                        )
                        applied = network._apply_projected_task_update(budgeted)
                        stages.add_task(
                            raw=raw,
                            masked=masked,
                            dale=applied.local_update,
                            normalization=applied.normalization_correction,
                        )
                    else:
                        rule.update_eligibility(pre_activity)
                if (
                    condition.homeostasis_enabled
                    and eligible
                    and global_step % homeostasis_interval == 0
                ):
                    proposal = homeostasis_rule._propose_trusted(
                        resources.reference_rate_tape[epoch, trial, time],
                        excitatory_mask=network.excitatory_mask,
                        inhibitory_mask=network.inhibitory_mask,
                        # Use the shared frozen reference weights so the local
                        # homeostatic proposal itself is bitwise identical
                        # across feedback branches.  Condition-specific fan-in
                        # normalization remains separately audited.
                        current_weights=resources.homeostasis_reference_weights,
                        connectivity_mask=network.connectivity_mask,
                    )
                    if homeostasis_budget is None:
                        raise RuntimeError("homeostasis budget was not initialized")
                    # Exact replay across task-plastic branches must remain
                    # Dale-safe for every possible current inhibitory weight.
                    # Retain only the strengthening (more-negative) I-to-E
                    # component; weakening proposals are audited as unavailable
                    # rather than made branch-dependent by boundary clipping.
                    shared_homeostatic = np.minimum(proposal.dale_applied_update, 0.0)
                    budgeted = homeostasis_budget.scale(shared_homeostatic)
                    applied = network._apply_projected_homeostatic_update(budgeted)
                    stages.add_homeostasis(
                        dale=applied.local_update,
                        normalization=applied.normalization_correction,
                    )
                elif (
                    condition.normalization_enabled
                    and not condition.task_plasticity_enabled
                    and not condition.homeostasis_enabled
                    and eligible
                    and global_step % update_interval == 0
                ):
                    # A pure renormalization of the already normalized initial
                    # matrix is a deliberately exact-zero negative control.
                    applied = network._apply_projected_task_update(
                        np.zeros_like(initial_weights)
                    )
                    stages.normalization += applied.normalization_correction
                    stages.normalization_path_l1 += float(
                        np.sum(np.abs(applied.normalization_correction))
                    )
                    stages.normalization_path_l2 += float(
                        np.linalg.norm(applied.normalization_correction)
                    )
                global_step += 1

    if task_budget is not None and not task_budget.complete:
        raise RuntimeError("task budget schedule did not consume every planned event")
    if homeostasis_budget is not None and not homeostasis_budget.complete:
        raise RuntimeError(
            "homeostasis budget schedule did not consume every planned event"
        )
    task_summary = None if task_budget is None else task_budget.summary()
    homeostasis_summary = (
        None if homeostasis_budget is None else homeostasis_budget.summary()
    )
    predictions, test_x, test_activity, test_gains = _evaluate_network(
        network,
        test,
        resources,
        evaluation_noise_std=evaluation_noise_std,
    )
    _, _, train_activity, _ = _evaluate_network(
        network,
        train,
        resources,
        evaluation_noise_std=0.0,
    )
    behavior, correct = _trial_behavior(
        test,
        predictions,
        switch_window=_positive_int(training.get("switch_window", 1), "switch_window"),
    )
    representation = _activity_representation_metrics(
        train_activity,
        test_activity,
        train.contexts,
        test.contexts,
        reduced_dim=_positive_int(training.get("reduced_dim", 4), "reduced_dim"),
        ridge=_positive_float(training.get("reduced_ridge", 1e-3), "reduced_ridge"),
    )
    final_weights = network.recurrent_weights
    if not network.validate_dale():
        raise RuntimeError("paired replay left the sparse Dale-feasible set")
    selected_test_error = (
        predictions[test.loss_mask] - test.targets[..., 0][test.loss_mask]
    )
    response_mask = test.epoch == "response"
    response_error = predictions[:, response_mask] - test.targets[:, response_mask, 0]
    task_attained = task_summary is None or task_summary.attained
    homeostasis_attained = homeostasis_summary is None or homeostasis_summary.attained
    selected_total = sum(
        summary.selected_applied
        for summary in (task_summary, homeostasis_summary)
        if summary is not None
    )
    target_total = task_target + homeostasis_target
    budget_match_valid = bool(
        task_attained
        and homeostasis_attained
        and abs(selected_total - target_total) <= max(budget_tolerance, 1e-12)
    )
    signal_tape = (
        resources.shuffled_feedback_signal_tape
        if condition.feedback_mode == "shuffled"
        else resources.feedback_signal_tape
    )
    signal_tape_id = (
        resources.shuffled_feedback_signal_tape_id
        if condition.feedback_mode == "shuffled"
        else resources.feedback_signal_tape_id
    )
    credit_source_span = (
        resources.shuffled_feedback_signal_span
        if condition.feedback_mode == "shuffled"
        else resources.feedback_signal_span
    )
    projector_basis = resources.feedback.basis(condition.feedback_mode)
    projector_rank = (
        network.n_units if projector_basis is None else projector_basis.shape[1]
    )
    if projector_basis is None:
        projected_signal_span = credit_source_span
    else:
        repeated_loss_mask = np.broadcast_to(train.loss_mask, signal_tape.shape[:-1])
        projected_signal_span = int(
            np.linalg.matrix_rank(signal_tape[repeated_loss_mask] @ projector_basis)
        )
    metrics: dict[str, Any] = {
        **behavior,
        **representation,
        **_jacobian_metrics(network, test_x, test_gains),
        **_rank_metrics(stages, resources.feedback.aligned_basis.shape[1]),
        **_controller_metrics("task", task_summary),
        **_controller_metrics("homeostasis", homeostasis_summary),
        "status": "complete" if budget_match_valid else "invalid",
        "training_algorithm": "strictly_paired_causal_three_factor",
        "used_autograd": False,
        "recurrent_learning_enabled": condition.task_plasticity_enabled,
        "task_plasticity_enabled": condition.task_plasticity_enabled,
        "homeostasis_enabled": condition.homeostasis_enabled,
        "normalization_enabled": condition.normalization_enabled,
        "mechanism": condition.mechanism,
        "feedback_mode": condition.feedback_mode,
        "feedback_dim": projector_rank,
        "feedback_projector_rank": projector_rank,
        "feedback_encoder_rank": resources.feedback_encoder_rank,
        "feedback_credit_source_span": credit_source_span,
        "feedback_projected_signal_span": projected_signal_span,
        "budget_norm": condition.budget_norm,
        "selected_total_budget_target": target_total,
        "selected_total_budget_applied": selected_total,
        "budget_attained": bool(task_attained and homeostasis_attained),
        "budget_match_valid": budget_match_valid,
        "failure_reason": None if budget_match_valid else "update_budget_shortfall",
        "simultaneous_dual_norm_match": False,
        "secondary_norm_is_diagnostic_only": True,
        "homeostasis_control_kind": "yoked_inhibitory_strengthening_control",
        "homeostasis_replay_projection": "shared_inhibitory_strengthening_only",
        "homeostasis_is_closed_loop_ei_stability_evidence": False,
        "task_path_l1": stages.task_path_l1,
        "task_path_l2": stages.task_path_l2,
        "homeostasis_path_l1": stages.homeostasis_path_l1,
        "homeostasis_path_l2": stages.homeostasis_path_l2,
        "normalization_path_l1": stages.normalization_path_l1,
        "normalization_path_l2": stages.normalization_path_l2,
        "activity_participation_ratio_direct": participation_ratio(
            test_activity.reshape(-1, network.n_units)
        ),
        "heldout_masked_mse": float(np.mean(selected_test_error**2)),
        "heldout_response_mse": float(np.mean(response_error**2)),
        "initialization_id": resources.initialization_id,
        "actual_initialization_id": initial_id,
        "readout_training_id": resources.readout_training_id,
        "readout_fit_trial_ids": resources.readout_fit_trial_ids.tolist(),
        "readout_fit_trial_count": int(resources.readout_fit_trial_ids.size),
        "gate_id": resources.gate_id,
        "trial_order_id": resources.tape.trial_order_id,
        "noise_id": resources.noise_contract_id,
        "noise_seed_id": resources.tape.noise_id,
        "split_id": resources.tape.split_id,
        "train_data_id": resources.tape.train_data_id,
        "test_data_id": resources.tape.test_data_id,
        "replay_contract_id": resources.replay_contract_id,
        "tape_replay_contract_id": resources.tape.replay_contract_id,
        "homeostasis_signal_id": resources.homeostasis_signal_id,
        "feedback_encoder_id": resources.feedback.encoder_id,
        "feedback_signal_tape_id": signal_tape_id,
        "feedback_design_id": resources.feedback.design_id,
        "shuffled_source_id": resources.tape.shuffled_source_id,
        "shuffled_fixed_point_fraction": float(
            np.mean(
                resources.tape.shuffled_source_indices
                == np.arange(resources.tape.shuffled_source_indices.size)
            )
        ),
        "initial_recurrent_id": _digest(initial_weights),
        "final_recurrent_id": _digest(final_weights),
        "recurrent_changed": not np.array_equal(initial_weights, final_weights),
        "task_component_l1": float(np.sum(np.abs(network.W_task))),
        "homeostasis_component_l1": float(np.sum(np.abs(network.W_homeo))),
        "normalization_component_l1": float(np.sum(np.abs(network.W_normalization))),
        "task_component_id": _digest(network.W_task),
        "homeostasis_component_id": _digest(network.W_homeo),
        "normalization_component_id": _digest(network.W_normalization),
        "dale_valid": True,
    }
    if condition.mechanism in {"frozen-recurrent", "normalization-only"}:
        metrics["frozen_exact"] = bool(np.array_equal(initial_weights, final_weights))
    else:
        metrics["frozen_exact"] = False
    return MechanismResult(
        metrics=metrics,
        predictions=predictions,
        activity=test_activity,
        trial_correct=correct,
    )


__all__ = [
    "FeedbackDesign",
    "MechanismCondition",
    "MechanismResult",
    "PairedResources",
    "PairedTrialTape",
    "build_mechanism_conditions",
    "prepare_paired_resources",
    "run_mechanism_condition",
]
