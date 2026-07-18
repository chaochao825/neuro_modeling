"""Hidden-reliability associative blocks for reward-only actuator selection.

The target is one queried value, not an explicit mixture of actuator outputs.
Input routing receives a noisy copy of that value.  A fixed associative motif
must retrieve it from dense bipolar bindings and therefore develops natural
capacity and distractor-write limits.  The cue reliability is constant within
one block but is never part of the selector input.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.reproducibility import derive_seed


FloatArray = NDArray[np.float64]


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _readonly(value: object, *, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"array must be {ndim}-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError("arrays must contain only finite values")
    frozen = np.array(result, dtype=np.float64, order="C", copy=True)
    frozen.setflags(write=False)
    return frozen


def _fingerprint(*arrays: np.ndarray, labels: tuple[object, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(labels).encode("utf-8"))
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HiddenReliabilityTaskConfig:
    n_train_blocks_per_cell: int = 6
    n_test_blocks_per_cell: int = 6
    trials_per_block: int = 128
    probe_trials: int = 64
    key_dim: int = 16
    load_values: tuple[int, ...] = (8, 16, 24)
    distractor_write_values: tuple[int, ...] = (8, 24, 48)
    direct_reliabilities: tuple[float, ...] = (0.55, 0.95)
    distractor_strength: float = 1.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_train_blocks_per_cell", 2),
            ("n_test_blocks_per_cell", 2),
            ("trials_per_block", 8),
            ("probe_trials", 2),
            ("key_dim", 2),
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name, minimum=minimum),
            )
        if self.probe_trials >= self.trials_per_block:
            raise ValueError("probe_trials must be smaller than trials_per_block")
        if self.probe_trials % 2:
            raise ValueError("probe_trials must be even for balanced exploration")

        loads = tuple(
            _positive_int(value, name="load_values element", minimum=2)
            for value in self.load_values
        )
        distractors = tuple(
            _positive_int(
                value, name="distractor_write_values element", minimum=0
            )
            for value in self.distractor_write_values
        )
        reliabilities = tuple(float(value) for value in self.direct_reliabilities)
        if len(loads) < 2 or len(set(loads)) != len(loads):
            raise ValueError("load_values must contain at least two unique values")
        if len(distractors) < 2 or len(set(distractors)) != len(distractors):
            raise ValueError(
                "distractor_write_values must contain at least two unique values"
            )
        if len(reliabilities) < 2 or len(set(reliabilities)) != len(reliabilities):
            raise ValueError(
                "direct_reliabilities must contain at least two unique values"
            )
        if loads != tuple(sorted(loads)) or distractors != tuple(sorted(distractors)):
            raise ValueError("load and distractor values must be strictly increasing")
        if reliabilities != tuple(sorted(reliabilities)):
            raise ValueError("direct_reliabilities must be strictly increasing")
        if any(not np.isfinite(value) or not 0.5 < value <= 1.0 for value in reliabilities):
            raise ValueError("direct reliabilities must lie in (0.5, 1]")
        strength = float(self.distractor_strength)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("distractor_strength must be finite and non-negative")
        object.__setattr__(self, "load_values", loads)
        object.__setattr__(self, "distractor_write_values", distractors)
        object.__setattr__(self, "direct_reliabilities", reliabilities)
        object.__setattr__(self, "distractor_strength", strength)


@dataclass(frozen=True, slots=True)
class HiddenReliabilityBlockSpec:
    root_seed: int
    split: str
    block_id: int
    cell_rep: int
    direct_reliability: float
    association_load: int
    distractor_writes: int
    block_seed: int

    @property
    def cell_key(self) -> tuple[float, int, int]:
        return (
            float(self.direct_reliability),
            int(self.association_load),
            int(self.distractor_writes),
        )


@dataclass(frozen=True, slots=True, eq=False)
class HiddenReliabilityBlock:
    spec: HiddenReliabilityBlockSpec
    targets: FloatArray
    direct_cues: FloatArray
    write_keys: FloatArray
    write_values: FloatArray
    query_keys: FloatArray
    distractor_keys: FloatArray
    distractor_values: FloatArray
    fingerprint: str

    def __post_init__(self) -> None:
        n_trials, load, key_dim = np.asarray(self.write_keys).shape
        distractors = int(self.spec.distractor_writes)
        expected = {
            "targets": (n_trials,),
            "direct_cues": (n_trials,),
            "write_values": (n_trials, load),
            "query_keys": (n_trials, key_dim),
            "distractor_keys": (n_trials, distractors, key_dim),
            "distractor_values": (n_trials, distractors),
        }
        if load != self.spec.association_load:
            raise ValueError("write-key load differs from the block specification")
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64:
            raise ValueError("fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "write_keys", _readonly(self.write_keys, ndim=3))
        object.__setattr__(self, "write_values", _readonly(self.write_values, ndim=2))
        object.__setattr__(self, "query_keys", _readonly(self.query_keys, ndim=2))
        object.__setattr__(
            self, "distractor_keys", _readonly(self.distractor_keys, ndim=3)
        )
        object.__setattr__(
            self, "distractor_values", _readonly(self.distractor_values, ndim=2)
        )
        object.__setattr__(self, "targets", _readonly(self.targets, ndim=1))
        object.__setattr__(self, "direct_cues", _readonly(self.direct_cues, ndim=1))

    @property
    def n_trials(self) -> int:
        return int(self.targets.size)


def make_hidden_reliability_block_specs(
    config: HiddenReliabilityTaskConfig,
    seed: int,
) -> tuple[HiddenReliabilityBlockSpec, ...]:
    """Create a deterministic whole-block train/test registry."""

    if not isinstance(config, HiddenReliabilityTaskConfig):
        raise TypeError("config must be a HiddenReliabilityTaskConfig")
    seed = _positive_int(seed, name="seed", minimum=0)
    specs: list[HiddenReliabilityBlockSpec] = []
    block_id = 0
    for split, n_reps in (
        ("train", config.n_train_blocks_per_cell),
        ("test", config.n_test_blocks_per_cell),
    ):
        for reliability in config.direct_reliabilities:
            for load in config.load_values:
                for distractors in config.distractor_write_values:
                    for rep in range(n_reps):
                        specs.append(
                            HiddenReliabilityBlockSpec(
                                root_seed=seed,
                                split=split,
                                block_id=block_id,
                                cell_rep=rep,
                                direct_reliability=reliability,
                                association_load=load,
                                distractor_writes=distractors,
                                block_seed=derive_seed(
                                    seed,
                                    "exp31-hidden-reliability",
                                    split,
                                    reliability,
                                    load,
                                    distractors,
                                    rep,
                                ),
                            )
                        )
                        block_id += 1
    return tuple(specs)


def materialize_hidden_reliability_block(
    config: HiddenReliabilityTaskConfig,
    spec: HiddenReliabilityBlockSpec,
) -> HiddenReliabilityBlock:
    """Materialize one registered tape without exposing context to a selector."""

    if not isinstance(config, HiddenReliabilityTaskConfig):
        raise TypeError("config must be a HiddenReliabilityTaskConfig")
    if not isinstance(spec, HiddenReliabilityBlockSpec):
        raise TypeError("spec must be a HiddenReliabilityBlockSpec")
    if (
        spec.direct_reliability not in config.direct_reliabilities
        or spec.association_load not in config.load_values
        or spec.distractor_writes not in config.distractor_write_values
    ):
        raise ValueError("block specification is not registered by the task config")
    rng = np.random.default_rng(spec.block_seed)
    n_trials = config.trials_per_block
    key_dim = config.key_dim
    scale = 1.0 / np.sqrt(float(key_dim))
    write_keys = rng.choice(
        np.array([-scale, scale]),
        size=(n_trials, spec.association_load, key_dim),
    )
    write_values = rng.choice(
        np.array([-1.0, 1.0]), size=(n_trials, spec.association_load)
    )
    query_slot = rng.integers(spec.association_load, size=n_trials)
    row = np.arange(n_trials)
    query_keys = write_keys[row, query_slot]
    targets = write_values[row, query_slot]
    correct_direct = rng.random(n_trials) < spec.direct_reliability
    direct_cues = np.where(correct_direct, targets, -targets)
    distractor_keys = rng.choice(
        np.array([-scale, scale]),
        size=(n_trials, spec.distractor_writes, key_dim),
    )
    distractor_values = rng.choice(
        np.array([-1.0, 1.0]), size=(n_trials, spec.distractor_writes)
    )
    fingerprint = _fingerprint(
        write_keys,
        write_values,
        query_keys,
        targets,
        direct_cues,
        distractor_keys,
        distractor_values,
        labels=(
            spec.root_seed,
            spec.split,
            spec.block_id,
            spec.cell_rep,
            spec.direct_reliability,
            spec.association_load,
            spec.distractor_writes,
        ),
    )
    return HiddenReliabilityBlock(
        spec=spec,
        targets=targets,
        direct_cues=direct_cues,
        write_keys=write_keys,
        write_values=write_values,
        query_keys=query_keys,
        distractor_keys=distractor_keys,
        distractor_values=distractor_values,
        fingerprint=fingerprint,
    )


__all__ = [
    "HiddenReliabilityBlock",
    "HiddenReliabilityBlockSpec",
    "HiddenReliabilityTaskConfig",
    "make_hidden_reliability_block_specs",
    "materialize_hidden_reliability_block",
]
