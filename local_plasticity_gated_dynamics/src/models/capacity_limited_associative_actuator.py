"""Fixed dense-key associative actuator with an exact write-budget control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.hidden_reliability_association import HiddenReliabilityBlock


FloatArray = NDArray[np.float64]


def _frozen(value: object) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("actuator outputs must be finite")
    frozen = np.array(result, dtype=np.float64, order="C", copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class DenseWriteBudget:
    mean_update_l1: float
    mean_update_l2: float
    mean_final_state_l2: float
    writes_per_trial: int
    per_write_rank: int


@dataclass(frozen=True, slots=True, eq=False)
class CapacityLimitedOutputs:
    routing: FloatArray
    associative: FloatArray
    associative_query_shuffled: FloatArray
    associative_score: FloatArray
    shuffled_score: FloatArray
    associative_budget: DenseWriteBudget
    shuffled_budget: DenseWriteBudget
    update_budget_exact: bool

    def __post_init__(self) -> None:
        for name in (
            "routing",
            "associative",
            "associative_query_shuffled",
            "associative_score",
            "shuffled_score",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        shapes = {
            np.asarray(getattr(self, name)).shape
            for name in (
                "routing",
                "associative",
                "associative_query_shuffled",
                "associative_score",
                "shuffled_score",
            )
        }
        if len(shapes) != 1:
            raise ValueError("all actuator outputs must have identical shape")


class CapacityLimitedAssociativeActuator:
    """Outer-product writes followed by dense content-addressed retrieval.

    The query-shuffled control reads the *same final memory states* with a
    cyclically shifted query tape.  It therefore preserves every write, the
    cumulative update L1/L2, and final-state norm exactly while breaking the
    trial-specific content address.
    """

    used_autograd = False
    used_bptt = False

    def __init__(self, *, key_dim: int, distractor_strength: float = 1.0) -> None:
        if isinstance(key_dim, (bool, np.bool_)) or not isinstance(
            key_dim, (int, np.integer)
        ):
            raise TypeError("key_dim must be an integer")
        self.key_dim = int(key_dim)
        if self.key_dim < 2:
            raise ValueError("key_dim must be at least two")
        self.distractor_strength = float(distractor_strength)
        if (
            not np.isfinite(self.distractor_strength)
            or self.distractor_strength < 0.0
        ):
            raise ValueError("distractor_strength must be finite and non-negative")

    @staticmethod
    def _sign(score: np.ndarray) -> np.ndarray:
        return np.where(score >= 0.0, 1.0, -1.0)

    def evaluate(self, block: HiddenReliabilityBlock) -> CapacityLimitedOutputs:
        if not isinstance(block, HiddenReliabilityBlock):
            raise TypeError("block must be a HiddenReliabilityBlock")
        if block.write_keys.shape[2] != self.key_dim:
            raise ValueError("block key dimension differs from the actuator")
        binding_updates = block.write_values[:, :, None] * block.write_keys
        distractor_updates = self.distractor_strength * (
            block.distractor_values[:, :, None] * block.distractor_keys
        )
        memory = np.sum(binding_updates, axis=1)
        if distractor_updates.shape[1]:
            memory = memory + np.sum(distractor_updates, axis=1)
        score = np.einsum("ni,ni->n", memory, block.query_keys, optimize=True)
        shuffled_query = np.roll(block.query_keys, shift=1, axis=0)
        shuffled_score = np.einsum(
            "ni,ni->n", memory, shuffled_query, optimize=True
        )

        update_l1 = np.sum(np.abs(binding_updates), axis=(1, 2))
        update_squared_l2 = np.sum(np.square(binding_updates), axis=(1, 2))
        if distractor_updates.shape[1]:
            update_l1 = update_l1 + np.sum(
                np.abs(distractor_updates), axis=(1, 2)
            )
            update_squared_l2 = update_squared_l2 + np.sum(
                np.square(distractor_updates), axis=(1, 2)
            )
        budget = DenseWriteBudget(
            mean_update_l1=float(np.mean(update_l1)),
            mean_update_l2=float(np.mean(np.sqrt(update_squared_l2))),
            mean_final_state_l2=float(np.mean(np.linalg.norm(memory, axis=1))),
            writes_per_trial=int(
                block.spec.association_load + block.spec.distractor_writes
            ),
            per_write_rank=1,
        )
        shuffled_budget = budget
        exact = budget == shuffled_budget
        return CapacityLimitedOutputs(
            routing=block.direct_cues,
            associative=self._sign(score),
            associative_query_shuffled=self._sign(shuffled_score),
            associative_score=score,
            shuffled_score=shuffled_score,
            associative_budget=budget,
            shuffled_budget=shuffled_budget,
            update_budget_exact=exact,
        )


__all__ = [
    "CapacityLimitedAssociativeActuator",
    "CapacityLimitedOutputs",
    "DenseWriteBudget",
]
