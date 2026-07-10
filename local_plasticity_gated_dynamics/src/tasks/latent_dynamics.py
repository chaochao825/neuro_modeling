"""Leakage-safe synthetic latent dynamics used by the Phase-1 experiments.

The generator creates independent trajectory blocks for training and testing.
Transitions are materialized *within* blocks, so no target can cross a block
boundary.  Feedback matrices are represented by orthonormal bases and are
applied without constructing a dense ``N x N`` projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from src.utils.reproducibility import make_rng


FeedbackMode = Literal["aligned", "random", "orthogonal", "shuffled"]
_FEEDBACK_MODES = frozenset({"aligned", "random", "orthogonal", "shuffled"})


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_finite_float(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    upper: float | None = None,
    strict_lower: bool = False,
    strict_upper: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None and (result <= lower if strict_lower else result < lower):
        relation = ">" if strict_lower else ">="
        raise ValueError(f"{name} must be {relation} {lower}")
    if upper is not None and (result >= upper if strict_upper else result > upper):
        relation = "<" if strict_upper else "<="
        raise ValueError(f"{name} must be {relation} {upper}")
    return result


def _readonly_copy(array: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(array, dtype=dtype, order="C", copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class LatentDynamicsConfig:
    """Configuration with the formal Phase-1 dimensions as defaults.

    The default training set contains 10,000 transitions per context, split
    over five independently initialized blocks.  The held-out set is generated
    from a separate random-number stream and is never a slice of the training
    trajectories.
    """

    n_neurons: int = 128
    latent_dim: int = 4
    n_contexts: int = 2
    train_steps_per_context: int = 10_000
    test_steps_per_context: int = 2_000
    n_train_blocks_per_context: int = 5
    n_test_blocks_per_context: int = 1
    dynamics_decay: float = 0.985
    context_angles: tuple[float, float] = (0.12, -0.09)
    process_noise_std: float = 0.04
    activity_noise_std: float = 0.02
    initial_state_std: float = 0.75
    embedding_gain: float = 2.0
    seed: int = 0

    def __post_init__(self) -> None:
        n_neurons = _require_int("n_neurons", self.n_neurons, minimum=2)
        latent_dim = _require_int("latent_dim", self.latent_dim, minimum=1)
        n_contexts = _require_int("n_contexts", self.n_contexts, minimum=1)
        train_steps = _require_int(
            "train_steps_per_context", self.train_steps_per_context, minimum=1
        )
        test_steps = _require_int(
            "test_steps_per_context", self.test_steps_per_context, minimum=1
        )
        n_train_blocks = _require_int(
            "n_train_blocks_per_context", self.n_train_blocks_per_context, minimum=1
        )
        n_test_blocks = _require_int(
            "n_test_blocks_per_context", self.n_test_blocks_per_context, minimum=1
        )
        _require_int("seed", self.seed, minimum=0)
        if latent_dim >= n_neurons:
            raise ValueError("latent_dim must be smaller than n_neurons")
        if n_contexts != 2:
            raise ValueError("Phase-1 requires exactly two contexts")
        if train_steps % n_train_blocks:
            raise ValueError(
                "train_steps_per_context must be divisible by "
                "n_train_blocks_per_context"
            )
        if test_steps % n_test_blocks:
            raise ValueError(
                "test_steps_per_context must be divisible by "
                "n_test_blocks_per_context"
            )
        _require_finite_float(
            "dynamics_decay",
            self.dynamics_decay,
            lower=0.0,
            upper=1.0,
            strict_lower=True,
            strict_upper=True,
        )
        for name in (
            "process_noise_std",
            "activity_noise_std",
            "initial_state_std",
        ):
            _require_finite_float(name, getattr(self, name), lower=0.0)
        _require_finite_float(
            "embedding_gain", self.embedding_gain, lower=0.0, strict_lower=True
        )
        if not isinstance(self.context_angles, Sequence) or len(self.context_angles) != 2:
            raise ValueError("context_angles must contain exactly two angles")
        angles = tuple(
            _require_finite_float(f"context_angles[{index}]", angle)
            for index, angle in enumerate(self.context_angles)
        )
        if np.isclose(angles[0], angles[1]):
            raise ValueError("the two context angles must differ")
        object.__setattr__(self, "context_angles", angles)


@dataclass(frozen=True)
class TransitionBatch:
    """Flattened within-block transitions and their provenance."""

    activity_t: np.ndarray
    activity_tp1: np.ndarray
    latent_t: np.ndarray
    latent_tp1: np.ndarray
    contexts: np.ndarray
    block_ids: np.ndarray

    def __post_init__(self) -> None:
        activity_t = np.asarray(self.activity_t)
        activity_tp1 = np.asarray(self.activity_tp1)
        latent_t = np.asarray(self.latent_t)
        latent_tp1 = np.asarray(self.latent_tp1)
        contexts = np.asarray(self.contexts)
        block_ids = np.asarray(self.block_ids)
        if activity_t.ndim != 2 or activity_tp1.shape != activity_t.shape:
            raise ValueError("activity transition arrays must have equal 2-D shapes")
        if latent_t.ndim != 2 or latent_tp1.shape != latent_t.shape:
            raise ValueError("latent transition arrays must have equal 2-D shapes")
        n_samples = activity_t.shape[0]
        if latent_t.shape[0] != n_samples:
            raise ValueError("activity and latent transitions must have equal lengths")
        if contexts.shape != (n_samples,) or block_ids.shape != (n_samples,):
            raise ValueError("contexts and block_ids must contain one value per transition")
        if not np.issubdtype(contexts.dtype, np.integer) or not np.issubdtype(
            block_ids.dtype, np.integer
        ):
            raise TypeError("contexts and block_ids must contain integers")
        for name, values in (
            ("activity_t", activity_t),
            ("activity_tp1", activity_tp1),
            ("latent_t", latent_t),
            ("latent_tp1", latent_tp1),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")

    @property
    def inputs(self) -> np.ndarray:
        return self.activity_t

    @property
    def targets(self) -> np.ndarray:
        return self.activity_tp1

    def __len__(self) -> int:
        return self.activity_t.shape[0]


@dataclass(frozen=True)
class TrajectorySplit:
    """Uniform-length trajectory blocks for one leakage-safe split."""

    latent_states: np.ndarray
    activities: np.ndarray
    block_contexts: np.ndarray
    block_ids: np.ndarray
    name: str

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent_states, dtype=float)
        activity = np.asarray(self.activities, dtype=float)
        contexts = np.asarray(self.block_contexts)
        block_ids = np.asarray(self.block_ids)
        if latent.ndim != 3 or activity.ndim != 3:
            raise ValueError("latent_states and activities must be 3-D block arrays")
        if latent.shape[:2] != activity.shape[:2]:
            raise ValueError("latent and activity blocks must share block/time dimensions")
        if latent.shape[1] < 2:
            raise ValueError("each block must contain at least one transition")
        n_blocks = latent.shape[0]
        if contexts.shape != (n_blocks,) or block_ids.shape != (n_blocks,):
            raise ValueError("block_contexts and block_ids must have one entry per block")
        if not np.issubdtype(contexts.dtype, np.integer) or not np.issubdtype(
            block_ids.dtype, np.integer
        ):
            raise TypeError("block_contexts and block_ids must contain integers")
        if np.any(contexts < 0):
            raise ValueError("block_contexts must be non-negative")
        if np.unique(block_ids).size != n_blocks:
            raise ValueError("block_ids must be unique within a split")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(activity)):
            raise ValueError("trajectory arrays contain non-finite values")
        object.__setattr__(self, "latent_states", _readonly_copy(latent, dtype=float))
        object.__setattr__(self, "activities", _readonly_copy(activity, dtype=float))
        object.__setattr__(self, "block_contexts", _readonly_copy(contexts, dtype=int))
        object.__setattr__(self, "block_ids", _readonly_copy(block_ids, dtype=int))

    @property
    def n_blocks(self) -> int:
        return self.latent_states.shape[0]

    @property
    def steps_per_block(self) -> int:
        return self.latent_states.shape[1] - 1

    @property
    def n_transitions(self) -> int:
        return self.n_blocks * self.steps_per_block

    def transitions(self) -> TransitionBatch:
        """Return only adjacent samples from the same trajectory block."""

        n_blocks, n_states, _ = self.latent_states.shape
        steps = n_states - 1
        return TransitionBatch(
            activity_t=self.activities[:, :-1].reshape(-1, self.activities.shape[-1]),
            activity_tp1=self.activities[:, 1:].reshape(-1, self.activities.shape[-1]),
            latent_t=self.latent_states[:, :-1].reshape(-1, self.latent_states.shape[-1]),
            latent_tp1=self.latent_states[:, 1:].reshape(-1, self.latent_states.shape[-1]),
            contexts=np.repeat(self.block_contexts, steps),
            block_ids=np.repeat(self.block_ids, steps),
        )


@dataclass(frozen=True)
class SyntheticLatentDataset:
    """Generated task, including shared ground-truth parameters and splits."""

    config: LatentDynamicsConfig
    embedding: np.ndarray
    context_dynamics: np.ndarray
    train: TrajectorySplit
    test: TrajectorySplit

    def __post_init__(self) -> None:
        embedding = np.asarray(self.embedding, dtype=float)
        dynamics = np.asarray(self.context_dynamics, dtype=float)
        expected_embedding = (self.config.n_neurons, self.config.latent_dim)
        expected_dynamics = (
            self.config.n_contexts,
            self.config.latent_dim,
            self.config.latent_dim,
        )
        if embedding.shape != expected_embedding:
            raise ValueError(f"embedding must have shape {expected_embedding}")
        if dynamics.shape != expected_dynamics:
            raise ValueError(f"context_dynamics must have shape {expected_dynamics}")
        for split, expected_blocks, expected_steps in (
            (
                self.train,
                self.config.n_contexts * self.config.n_train_blocks_per_context,
                self.config.train_steps_per_context
                // self.config.n_train_blocks_per_context,
            ),
            (
                self.test,
                self.config.n_contexts * self.config.n_test_blocks_per_context,
                self.config.test_steps_per_context
                // self.config.n_test_blocks_per_context,
            ),
        ):
            if split.n_blocks != expected_blocks or split.steps_per_block != expected_steps:
                raise ValueError("trajectory split dimensions do not match config")
            if split.activities.shape[-1] != self.config.n_neurons:
                raise ValueError("trajectory activity dimension does not match config")
            if split.latent_states.shape[-1] != self.config.latent_dim:
                raise ValueError("trajectory latent dimension does not match config")
            if np.any(split.block_contexts >= self.config.n_contexts):
                raise ValueError("trajectory contains an out-of-range context label")
        if np.intersect1d(self.train.block_ids, self.test.block_ids).size:
            raise ValueError("train and test block identifiers must be disjoint")
        object.__setattr__(self, "embedding", _readonly_copy(embedding, dtype=float))
        object.__setattr__(self, "context_dynamics", _readonly_copy(dynamics, dtype=float))


def _canonicalize_qr(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return q * signs


def make_context_dynamics(config: LatentDynamicsConfig) -> np.ndarray:
    """Construct two stable rotation-decay systems in a shared latent basis."""

    rng = make_rng(config.seed, "latent-context-dynamics")
    mixing = _canonicalize_qr(rng.normal(size=(config.latent_dim, config.latent_dim)))
    systems: list[np.ndarray] = []
    for context, base_angle in enumerate(config.context_angles):
        canonical = np.zeros((config.latent_dim, config.latent_dim), dtype=float)
        for pair_index, start in enumerate(range(0, config.latent_dim - 1, 2)):
            angle = float(base_angle) * (1.0 + 0.2 * pair_index)
            cosine, sine = np.cos(angle), np.sin(angle)
            canonical[start : start + 2, start : start + 2] = config.dynamics_decay * np.array(
                [[cosine, -sine], [sine, cosine]], dtype=float
            )
        if config.latent_dim % 2:
            canonical[-1, -1] = config.dynamics_decay * (0.97 - 0.02 * context)
        systems.append(mixing @ canonical @ mixing.T)
    return np.stack(systems)


def _generate_split(
    config: LatentDynamicsConfig,
    *,
    embedding: np.ndarray,
    dynamics: np.ndarray,
    name: Literal["train", "test"],
    block_id_offset: int,
) -> TrajectorySplit:
    if name == "train":
        steps_per_context = config.train_steps_per_context
        blocks_per_context = config.n_train_blocks_per_context
    else:
        steps_per_context = config.test_steps_per_context
        blocks_per_context = config.n_test_blocks_per_context
    steps_per_block = steps_per_context // blocks_per_context
    n_blocks = config.n_contexts * blocks_per_context
    latent = np.empty((n_blocks, steps_per_block + 1, config.latent_dim), dtype=float)
    activity = np.empty((n_blocks, steps_per_block + 1, config.n_neurons), dtype=float)
    contexts = np.repeat(np.arange(config.n_contexts, dtype=int), blocks_per_context)
    rng = make_rng(config.seed, "latent-trajectories", name)

    for block_index, context in enumerate(contexts):
        states = latent[block_index]
        states[0] = rng.normal(scale=config.initial_state_std, size=config.latent_dim)
        innovations = rng.normal(
            scale=config.process_noise_std,
            size=(steps_per_block, config.latent_dim),
        )
        for step in range(steps_per_block):
            states[step + 1] = dynamics[context] @ states[step] + innovations[step]
        embedded = states @ embedding.T
        rates = np.tanh(embedded)
        if config.activity_noise_std:
            rates = rates + rng.normal(scale=config.activity_noise_std, size=rates.shape)
        activity[block_index] = rates

    return TrajectorySplit(
        latent_states=latent,
        activities=activity,
        block_contexts=contexts,
        block_ids=np.arange(block_id_offset, block_id_offset + n_blocks, dtype=int),
        name=name,
    )


def generate_latent_dynamics(
    config: LatentDynamicsConfig | None = None,
) -> SyntheticLatentDataset:
    """Generate deterministic, independently blocked train and test data."""

    config = LatentDynamicsConfig() if config is None else config
    if not isinstance(config, LatentDynamicsConfig):
        raise TypeError("config must be a LatentDynamicsConfig")
    embedding_rng = make_rng(config.seed, "latent-embedding")
    embedding_basis = _canonicalize_qr(
        embedding_rng.normal(size=(config.n_neurons, config.latent_dim))
    )
    embedding = config.embedding_gain * embedding_basis
    dynamics = make_context_dynamics(config)
    train = _generate_split(
        config,
        embedding=embedding,
        dynamics=dynamics,
        name="train",
        block_id_offset=0,
    )
    train_block_count = config.n_contexts * config.n_train_blocks_per_context
    test = _generate_split(
        config,
        embedding=embedding,
        dynamics=dynamics,
        name="test",
        block_id_offset=train_block_count,
    )
    return SyntheticLatentDataset(config, embedding, dynamics, train, test)


def _validate_embedding(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(embedding, dtype=float)
    if array.ndim != 2:
        raise ValueError("embedding must be a 2-D array")
    n_neurons, latent_dim = array.shape
    if latent_dim < 1 or n_neurons <= latent_dim:
        raise ValueError("embedding must have shape (N, d) with 1 <= d < N")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding contains non-finite values")
    if np.linalg.matrix_rank(array) != latent_dim:
        raise ValueError("embedding must have full column rank")
    return array, _canonicalize_qr(array)


def _haar_columns(size: int, n_columns: int, rng: np.random.Generator) -> np.ndarray:
    if n_columns == 0:
        return np.empty((size, 0), dtype=float)
    return _canonicalize_qr(rng.normal(size=(size, n_columns)))


def _orthogonal_complement(task_basis: np.ndarray) -> np.ndarray:
    # Full SVD gives a deterministic, complete null-space basis even when all
    # N-d complement directions are requested.
    _, _, vh = np.linalg.svd(task_basis.T, full_matrices=True)
    return vh[task_basis.shape[1] :].T


def _sample_from_complement(
    task_basis: np.ndarray,
    n_columns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    complement = _orthogonal_complement(task_basis)
    if n_columns > complement.shape[1]:
        raise ValueError(
            f"cannot draw {n_columns} orthogonal directions; only "
            f"{complement.shape[1]} are available"
        )
    rotation = _haar_columns(complement.shape[1], n_columns, rng)
    return complement @ rotation


def _aligned_basis(
    task_basis: np.ndarray,
    feedback_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    latent_dim = task_basis.shape[1]
    if feedback_dim <= latent_dim:
        return task_basis[:, :feedback_dim].copy()
    extra = _sample_from_complement(task_basis, feedback_dim - latent_dim, rng)
    return np.concatenate((task_basis, extra), axis=1)


def _sattolo_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a single-cycle permutation, hence no unchanged positions."""

    if size < 2:
        raise ValueError("a shuffled feedback block requires at least two transitions")
    permutation = np.arange(size, dtype=int)
    for index in range(size - 1, 0, -1):
        swap = int(rng.integers(0, index))
        permutation[index], permutation[swap] = permutation[swap], permutation[index]
    return permutation


def make_blockwise_feedback_permutation(
    block_ids: Sequence[int] | np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Create a deterministic temporal feedback shuffle within each block.

    The returned array maps each current transition to the transition whose
    prediction error will be broadcast at that time.  Every block is shuffled
    independently with a single-cycle permutation: there are no fixed points,
    no cross-block sources, and therefore no train/test crossing when called on
    one split's transition batch.
    """

    seed = _require_int("seed", seed, minimum=0)
    if isinstance(block_ids, np.ndarray):
        blocks = block_ids
    else:
        try:
            blocks = np.asarray(list(block_ids))
        except TypeError as exc:
            raise TypeError("block_ids must be a one-dimensional integer sequence") from exc
    if blocks.ndim != 1 or blocks.size < 1:
        raise ValueError("block_ids must be a non-empty one-dimensional sequence")
    if np.issubdtype(blocks.dtype, np.bool_) or not np.issubdtype(
        blocks.dtype, np.integer
    ):
        raise TypeError("block_ids must contain integers")

    permutation = np.empty(blocks.size, dtype=int)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        if indices.size < 2:
            raise ValueError(
                f"block {int(block)} has fewer than two transitions and cannot be shuffled"
            )
        local = _sattolo_permutation(
            indices.size,
            make_rng(seed, "temporal-feedback-shuffle", int(block), indices.size),
        )
        permutation[indices] = indices[local]
    permutation.setflags(write=False)
    return permutation


@dataclass(frozen=True)
class FeedbackSubspace:
    """Orthonormal feedback basis plus its task-alignment provenance."""

    basis: np.ndarray
    task_basis: np.ndarray
    mode: FeedbackMode
    seed: int

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        task_basis = np.asarray(self.task_basis, dtype=float)
        if not isinstance(self.mode, str) or self.mode not in _FEEDBACK_MODES:
            raise ValueError(f"unknown feedback mode: {self.mode!r}")
        _require_int("seed", self.seed, minimum=0)
        if basis.ndim != 2 or task_basis.ndim != 2:
            raise ValueError("basis and task_basis must be 2-D")
        if basis.shape[0] != task_basis.shape[0] or basis.shape[1] < 1:
            raise ValueError("feedback and task bases have incompatible shapes")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(task_basis)):
            raise ValueError("feedback bases contain non-finite values")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
            raise ValueError("feedback basis columns must be orthonormal")
        if not np.allclose(
            task_basis.T @ task_basis, np.eye(task_basis.shape[1]), atol=1e-10
        ):
            raise ValueError("task basis columns must be orthonormal")
        object.__setattr__(self, "basis", _readonly_copy(basis, dtype=float))
        object.__setattr__(self, "task_basis", _readonly_copy(task_basis, dtype=float))

    @property
    def n_neurons(self) -> int:
        return self.basis.shape[0]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    @property
    def alignment_fraction(self) -> float:
        """Fraction of feedback-basis energy lying in the true task subspace."""

        overlap = self.task_basis.T @ self.basis
        return float(np.square(overlap).sum() / self.dimension)

    @property
    def projector(self) -> np.ndarray:
        return self.basis @ self.basis.T

    def project(self, errors: np.ndarray) -> np.ndarray:
        """Project one error vector or a batch using the low-rank factorization."""

        array = np.asarray(errors, dtype=float)
        if array.ndim not in (1, 2) or array.shape[-1] != self.n_neurons:
            raise ValueError(
                f"errors must have trailing dimension {self.n_neurons} and rank 1 or 2"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("errors contain non-finite values")
        return (array @ self.basis) @ self.basis.T


def make_feedback_subspace(
    embedding: np.ndarray,
    feedback_dim: int,
    mode: FeedbackMode,
    *,
    seed: int,
) -> FeedbackSubspace:
    """Construct a deterministic feedback control with explicit geometry.

    ``aligned`` contains task directions first and fills any excess dimension
    from their orthogonal complement. ``orthogonal`` is entirely confined to
    that complement. ``random`` is Haar-distributed in neural space.
    ``shuffled`` uses the same aligned basis as ``aligned``.  Its control is a
    *temporal* error shuffle, constructed by
    :func:`make_blockwise_feedback_permutation` and supplied to model fitting.
    Keeping the basis identical isolates temporal correspondence from feedback
    rank and geometry.

    For the formal ``N=128, d=4`` setup, orthogonal feedback dimensions above
    ``124`` are mathematically impossible and raise ``ValueError``.
    """

    _, task_basis = _validate_embedding(embedding)
    feedback_dim = _require_int("feedback_dim", feedback_dim, minimum=1)
    seed = _require_int("seed", seed, minimum=0)
    if feedback_dim > task_basis.shape[0]:
        raise ValueError("feedback_dim cannot exceed n_neurons")
    if not isinstance(mode, str) or mode not in _FEEDBACK_MODES:
        raise ValueError(f"unknown feedback mode: {mode!r}")

    n_neurons, latent_dim = task_basis.shape
    if mode == "orthogonal" and feedback_dim > n_neurons - latent_dim:
        raise ValueError(
            f"orthogonal feedback_dim={feedback_dim} is invalid for N={n_neurons}, "
            f"latent_dim={latent_dim}; maximum is {n_neurons - latent_dim}"
        )

    if mode in {"aligned", "shuffled"}:
        aligned_rng = make_rng(
            seed, "feedback", "aligned-core", n_neurons, latent_dim, feedback_dim
        )
        basis = _aligned_basis(task_basis, feedback_dim, aligned_rng)
    elif mode == "orthogonal":
        basis = _sample_from_complement(
            task_basis,
            feedback_dim,
            make_rng(seed, "feedback", "orthogonal", n_neurons, latent_dim, feedback_dim),
        )
    else:
        basis = _haar_columns(
            n_neurons,
            feedback_dim,
            make_rng(seed, "feedback", "random", n_neurons, latent_dim, feedback_dim),
        )

    return FeedbackSubspace(
        basis=basis,
        task_basis=task_basis,
        mode=mode,
        seed=seed,
    )


__all__ = [
    "FeedbackMode",
    "FeedbackSubspace",
    "LatentDynamicsConfig",
    "SyntheticLatentDataset",
    "TrajectorySplit",
    "TransitionBatch",
    "generate_latent_dynamics",
    "make_blockwise_feedback_permutation",
    "make_context_dynamics",
    "make_feedback_subspace",
]
