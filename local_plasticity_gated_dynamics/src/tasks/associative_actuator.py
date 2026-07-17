"""Paired contextual-association tasks for the Exp30 trend panel.

The task deliberately separates a query-time direct cue from a trial-local
key--value binding.  A demand coordinate ``mu`` mixes the two sources in the
target.  All actuator conditions receive the same block split, keys, values,
queries, direct cues, distractors, and target-noise tape.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.reproducibility import derive_seed


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("task arrays must contain only finite values")
    result.setflags(write=False)
    return result


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AssociativeActuatorTaskConfig:
    """Configuration shared by every demand cell in one seed."""

    n_train_blocks: int = 8
    n_test_blocks: int = 4
    trials_per_block: int = 32
    key_dim: int = 8
    n_pairs: int = 4
    delay: int = 4
    target_noise_std: float = 0.05

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_train_blocks", 2),
            ("n_test_blocks", 2),
            ("trials_per_block", 4),
            ("key_dim", 2),
            ("n_pairs", 2),
            ("delay", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            object.__setattr__(self, name, int(value))
        if self.n_pairs > self.key_dim:
            raise ValueError("n_pairs cannot exceed key_dim")
        noise = float(self.target_noise_std)
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError("target_noise_std must be finite and non-negative")
        object.__setattr__(self, "target_noise_std", noise)


@dataclass(frozen=True, slots=True, eq=False)
class AssociativeActuatorSplit:
    """One whole-block split with no time-point reassignment."""

    write_keys: FloatArray
    write_values: FloatArray
    query_keys: FloatArray
    direct_cues: FloatArray
    distractors: FloatArray
    retrieval_targets: FloatArray
    target_noise: FloatArray
    block_ids: IntArray
    fingerprint: str

    def __post_init__(self) -> None:
        keys = np.asarray(self.write_keys)
        if keys.ndim != 3:
            raise ValueError("write_keys must have shape [trial, pair, key]")
        n_trials, n_pairs, key_dim = keys.shape
        expected = {
            "write_values": (n_trials, n_pairs),
            "query_keys": (n_trials, key_dim),
            "direct_cues": (n_trials,),
            "retrieval_targets": (n_trials,),
            "target_noise": (n_trials,),
            "block_ids": (n_trials,),
        }
        distractors = np.asarray(self.distractors)
        if distractors.ndim != 2 or distractors.shape[0] != n_trials:
            raise ValueError("distractors must have shape [trial, delay]")
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64:
            raise ValueError("fingerprint must be a SHA-256 hex string")
        for name in (
            "write_keys",
            "write_values",
            "query_keys",
            "direct_cues",
            "distractors",
            "retrieval_targets",
            "target_noise",
        ):
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), dtype=np.float64),
            )
        object.__setattr__(
            self, "block_ids", _readonly(self.block_ids, dtype=np.int64)
        )

    @property
    def n_trials(self) -> int:
        return int(self.write_keys.shape[0])

    def target(self, mu: float) -> FloatArray:
        """Return the registered noisy target for one memory-demand value."""

        demand = float(mu)
        if not np.isfinite(demand) or not 0.0 <= demand <= 1.0:
            raise ValueError("mu must lie in [0, 1]")
        target = (
            np.sqrt(1.0 - demand) * self.direct_cues
            + np.sqrt(demand) * self.retrieval_targets
            + self.target_noise
        )
        return _readonly(target, dtype=np.float64)  # type: ignore[return-value]

    def noiseless_target(self, mu: float) -> FloatArray:
        demand = float(mu)
        if not np.isfinite(demand) or not 0.0 <= demand <= 1.0:
            raise ValueError("mu must lie in [0, 1]")
        target = (
            np.sqrt(1.0 - demand) * self.direct_cues
            + np.sqrt(demand) * self.retrieval_targets
        )
        return _readonly(target, dtype=np.float64)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class AssociativeActuatorDataset:
    config: AssociativeActuatorTaskConfig
    train: AssociativeActuatorSplit
    test: AssociativeActuatorSplit

    def __post_init__(self) -> None:
        if set(self.train.block_ids.tolist()) & set(self.test.block_ids.tolist()):
            raise ValueError("train and test block IDs must be disjoint")


def _make_split(
    config: AssociativeActuatorTaskConfig,
    *,
    seed: int,
    split_name: str,
    n_blocks: int,
    block_offset: int,
) -> AssociativeActuatorSplit:
    rng = np.random.default_rng(
        derive_seed(seed, "exp30-associative-actuator", split_name)
    )
    n_trials = n_blocks * config.trials_per_block
    keys = np.zeros((n_trials, config.n_pairs, config.key_dim), dtype=np.float64)
    values = np.empty((n_trials, config.n_pairs), dtype=np.float64)
    queries = np.zeros((n_trials, config.key_dim), dtype=np.float64)
    retrieval = np.empty(n_trials, dtype=np.float64)
    direct = np.empty(n_trials, dtype=np.float64)
    block_ids = np.repeat(
        np.arange(block_offset, block_offset + n_blocks, dtype=np.int64),
        config.trials_per_block,
    )
    for trial in range(n_trials):
        key_indices = rng.permutation(config.key_dim)[: config.n_pairs]
        keys[trial, np.arange(config.n_pairs), key_indices] = 1.0
        trial_values = rng.choice(np.array([-1.0, 1.0]), size=config.n_pairs)
        values[trial] = trial_values
        query_slot = int(rng.integers(config.n_pairs))
        queries[trial] = keys[trial, query_slot]
        retrieval[trial] = trial_values[query_slot]
        direct[trial] = float(rng.choice(np.array([-1.0, 1.0])))
    distractors = rng.choice(
        np.array([-1.0, 1.0]), size=(n_trials, config.delay)
    ).astype(np.float64)
    noise = config.target_noise_std * rng.normal(size=n_trials)
    fingerprint = _fingerprint(
        keys, values, queries, direct, distractors, retrieval, noise, block_ids
    )
    return AssociativeActuatorSplit(
        write_keys=keys,
        write_values=values,
        query_keys=queries,
        direct_cues=direct,
        distractors=distractors,
        retrieval_targets=retrieval,
        target_noise=noise,
        block_ids=block_ids,
        fingerprint=fingerprint,
    )


def make_associative_actuator_dataset(
    config: AssociativeActuatorTaskConfig,
    seed: int,
) -> AssociativeActuatorDataset:
    """Generate paired train/test blocks for all Exp30 actuator conditions."""

    if not isinstance(config, AssociativeActuatorTaskConfig):
        raise TypeError("config must be an AssociativeActuatorTaskConfig")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    train = _make_split(
        config,
        seed=seed,
        split_name="train",
        n_blocks=config.n_train_blocks,
        block_offset=0,
    )
    test = _make_split(
        config,
        seed=seed,
        split_name="test",
        n_blocks=config.n_test_blocks,
        block_offset=config.n_train_blocks,
    )
    return AssociativeActuatorDataset(config=config, train=train, test=test)


__all__ = [
    "AssociativeActuatorDataset",
    "AssociativeActuatorSplit",
    "AssociativeActuatorTaskConfig",
    "make_associative_actuator_dataset",
]
