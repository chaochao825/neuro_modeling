"""Fixed associative and compressive actuators on a frozen carrier.

The associative state is updated by a local outer product and reset between
trials.  A paired shuffled condition permutes values across keys without
changing any write-update L1/L2 budget.  The carrier bridge uses the frozen
high-rank Dale matrix from Exp26 and a single fixed observable injection axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.tasks.actuator_matching import ActuatorCarrier
from src.tasks.associative_actuator import AssociativeActuatorSplit
from src.utils.reproducibility import derive_seed


FloatArray = NDArray[np.float64]


def _finite(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric array")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class WriteBudgetAudit:
    mean_l1: float
    mean_l2: float
    max_update_rank: int
    n_trials: int
    n_pairs: int


@dataclass(frozen=True, slots=True, eq=False)
class AssociativeMemoryActuator:
    """Trial-local scalar-value associative memory with a rank-one update."""

    key_dim: int
    compression_axis: FloatArray
    compression_decay: float = 0.98
    distractor_gain: float = 0.08

    def __post_init__(self) -> None:
        if isinstance(self.key_dim, (bool, np.bool_)) or not isinstance(
            self.key_dim, (int, np.integer)
        ):
            raise TypeError("key_dim must be an integer")
        key_dim = int(self.key_dim)
        if key_dim < 2:
            raise ValueError("key_dim must be at least two")
        axis = _finite(self.compression_axis, name="compression_axis", ndim=1)
        if axis.shape != (key_dim,):
            raise ValueError("compression_axis must have shape [key_dim]")
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("compression_axis must be non-zero")
        decay = float(self.compression_decay)
        gain = float(self.distractor_gain)
        if not np.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError("compression_decay must lie in [0, 1]")
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("distractor_gain must be finite and non-negative")
        object.__setattr__(self, "key_dim", key_dim)
        object.__setattr__(self, "compression_axis", _readonly(axis / norm))
        object.__setattr__(self, "compression_decay", decay)
        object.__setattr__(self, "distractor_gain", gain)

    @classmethod
    def random(
        cls,
        *,
        key_dim: int,
        seed: int,
        compression_decay: float = 0.98,
        distractor_gain: float = 0.08,
    ) -> "AssociativeMemoryActuator":
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        rng = np.random.default_rng(
            derive_seed(int(seed), "exp30-associative", "compression-axis")
        )
        axis = rng.choice(np.array([-1.0, 1.0]), size=int(key_dim))
        return cls(
            key_dim=int(key_dim),
            compression_axis=axis,
            compression_decay=compression_decay,
            distractor_gain=distractor_gain,
        )

    def _validate_split(self, split: AssociativeActuatorSplit) -> None:
        if not isinstance(split, AssociativeActuatorSplit):
            raise TypeError("split must be an AssociativeActuatorSplit")
        if split.write_keys.shape[2] != self.key_dim:
            raise ValueError("split key dimension differs from the actuator")

    def retrieve(self, split: AssociativeActuatorSplit) -> FloatArray:
        """Write all local bindings and content-address the query."""

        self._validate_split(split)
        memory = np.einsum(
            "np,npi->ni", split.write_values, split.write_keys, optimize=True
        )
        result = np.einsum("ni,ni->n", memory, split.query_keys, optimize=True)
        return _readonly(result)

    def retrieve_shuffled(self, split: AssociativeActuatorSplit) -> FloatArray:
        """Break key--value correspondence while preserving write budgets."""

        self._validate_split(split)
        shifted_values = np.roll(split.write_values, shift=1, axis=1)
        memory = np.einsum(
            "np,npi->ni", shifted_values, split.write_keys, optimize=True
        )
        result = np.einsum("ni,ni->n", memory, split.query_keys, optimize=True)
        return _readonly(result)

    def compressive_retrieval(self, split: AssociativeActuatorSplit) -> FloatArray:
        """One-dimensional non-content-addressable history state.

        This is a deliberately small internal-dynamics control: it stores one
        fixed projection of every binding, decays through the delay, and can
        only gate that scalar by the projected query.  It is not trained and is
        not intended as a strong sequence-model baseline.
        """

        self._validate_split(split)
        projected_keys = np.einsum(
            "npi,i->np", split.write_keys, self.compression_axis, optimize=True
        )
        state = np.sum(split.write_values * projected_keys, axis=1)
        query_projection = split.query_keys @ self.compression_axis
        for step in range(split.distractors.shape[1]):
            state = (
                self.compression_decay * state
                + self.distractor_gain * split.distractors[:, step]
            )
        return _readonly(state * query_projection)

    def write_budget(
        self,
        split: AssociativeActuatorSplit,
        *,
        shuffled: bool = False,
    ) -> WriteBudgetAudit:
        """Audit per-trial cumulative local write-update norms."""

        self._validate_split(split)
        values = (
            np.roll(split.write_values, shift=1, axis=1)
            if shuffled
            else split.write_values
        )
        updates = values[:, :, None] * split.write_keys
        cumulative = np.sum(updates, axis=1)
        l1 = np.sum(np.abs(updates), axis=(1, 2))
        l2 = np.linalg.norm(cumulative, axis=1)
        ranks = [
            int(np.linalg.matrix_rank(update[np.newaxis, :]))
            for trial in updates
            for update in trial
        ]
        return WriteBudgetAudit(
            mean_l1=float(np.mean(l1)),
            mean_l2=float(np.mean(l2)),
            max_update_rank=max(ranks, default=0),
            n_trials=int(split.n_trials),
            n_pairs=int(split.write_keys.shape[1]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class FrozenCarrierBridge:
    """Fixed scalar actuator axis transmitted through a high-rank carrier."""

    carrier: ActuatorCarrier
    injection_axis: FloatArray
    response_axis: FloatArray
    reconstruction_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.carrier, ActuatorCarrier):
            raise TypeError("carrier must be an ActuatorCarrier")
        n = self.carrier.config.n_neurons
        injection = _finite(self.injection_axis, name="injection_axis", ndim=1)
        response = _finite(self.response_axis, name="response_axis", ndim=1)
        if injection.shape != (n,) or response.shape != (n,):
            raise ValueError("carrier axes must have shape [n_neurons]")
        error = float(self.reconstruction_error)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("reconstruction_error must be finite and non-negative")
        object.__setattr__(self, "injection_axis", _readonly(injection))
        object.__setattr__(self, "response_axis", _readonly(response))
        object.__setattr__(self, "reconstruction_error", error)

    @classmethod
    def from_carrier(cls, carrier: ActuatorCarrier) -> "FrozenCarrierBridge":
        if not isinstance(carrier, ActuatorCarrier):
            raise TypeError("carrier must be an ActuatorCarrier")
        observation = carrier.c[0]
        normalization = float(observation @ observation)
        desired_response = observation / normalization
        injection = (np.eye(carrier.config.n_neurons) - carrier.a0) @ desired_response
        response = np.linalg.solve(
            np.eye(carrier.config.n_neurons) - carrier.a0, injection
        )
        reconstructed = float(carrier.c[0] @ response)
        return cls(
            carrier=carrier,
            injection_axis=injection,
            response_axis=response,
            reconstruction_error=abs(reconstructed - 1.0),
        )

    def transmit(self, controls: ArrayLike) -> FloatArray:
        values = _finite(controls, name="controls", ndim=1)
        states = values[:, None] * self.response_axis[None, :]
        outputs = states @ self.carrier.c[0]
        return _readonly(outputs)


__all__ = [
    "AssociativeMemoryActuator",
    "FrozenCarrierBridge",
    "WriteBudgetAudit",
]
