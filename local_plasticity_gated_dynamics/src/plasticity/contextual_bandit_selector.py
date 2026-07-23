"""Executed-reward-only local learning for multi-actuator selection.

The controller is a small bank of linear reward predictors.  It updates only
the eligibility assigned to the executed action (or to an explicitly supplied
wrong-credit action for a causal control).  ``reward_provider`` is invoked
exactly once, after action selection, so counterfactual action rewards are not
part of the main learning API.  No autograd or BPTT is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.streaming_fewshot_actuators import StreamingActuatorTrace


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: ArrayLike, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _sigmoid(value: FloatArray) -> FloatArray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True, slots=True)
class ContextualBanditConfig:
    learning_rate: float = 0.05
    eligibility_retention: float = 0.0
    belief_retention: float = 0.5
    epsilon: float = 0.1
    weight_decay: float = 0.0
    initial_reward_logit: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "eligibility_retention",
            "belief_retention",
            "epsilon",
            "weight_decay",
            "initial_reward_logit",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.eligibility_retention <= 1.0:
            raise ValueError("eligibility_retention must lie in [0, 1]")
        if not 0.0 <= self.belief_retention < 1.0:
            raise ValueError("belief_retention must lie in [0, 1)")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")


@dataclass(frozen=True, slots=True, eq=False)
class ContextualBanditReceipt:
    weights: FloatArray
    eligibilities: FloatArray
    belief: FloatArray
    action_counts: IntArray
    reward_sums: FloatArray
    n_reward_queries: int
    update_l1_cost: float
    update_l2_cost: float
    used_autograd: bool
    used_bptt: bool
    fingerprint: str


@dataclass(frozen=True, slots=True, eq=False)
class SelectorTrace:
    actions: IntArray
    predictions: IntArray
    action_values: FloatArray
    beliefs: FloatArray
    selected_event_l1: FloatArray
    video_ids: NDArray[np.str_]

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.int64)
        predictions = np.asarray(self.predictions, dtype=np.int64)
        values = np.asarray(self.action_values, dtype=np.float64)
        beliefs = np.asarray(self.beliefs, dtype=np.float64)
        costs = np.asarray(self.selected_event_l1, dtype=np.float64)
        video_ids = np.asarray(self.video_ids, dtype=str)
        if actions.ndim != 1 or actions.size == 0:
            raise ValueError("selector actions must be a non-empty vector")
        n = actions.size
        if predictions.shape != (n,) or costs.shape != (n,) or video_ids.shape != (n,):
            raise ValueError("selector trace arrays must share the frame dimension")
        if values.ndim != 2 or values.shape[0] != n or beliefs.shape != values.shape:
            raise ValueError("selector value and belief arrays have invalid shapes")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(beliefs)):
            raise ValueError("selector values and beliefs must be finite")
        if np.any(actions < 0) or np.any(actions >= values.shape[1]):
            raise ValueError("selector actions fall outside the action range")
        if np.any(costs < 0.0):
            raise ValueError("selector event costs must be non-negative")
        object.__setattr__(self, "actions", _readonly(actions, dtype=np.int64))
        object.__setattr__(self, "predictions", _readonly(predictions, dtype=np.int64))
        object.__setattr__(self, "action_values", _readonly(values))
        object.__setattr__(self, "beliefs", _readonly(beliefs))
        object.__setattr__(self, "selected_event_l1", _readonly(costs))
        object.__setattr__(self, "video_ids", _readonly(video_ids, dtype=str))


class RewardOnlyContextualController:
    """A local contextual bandit with a low-dimensional persistent belief."""

    def __init__(
        self,
        n_actions: int,
        n_features: int,
        *,
        config: ContextualBanditConfig | None = None,
        seed: int = 0,
        initial_weights: ArrayLike | None = None,
    ) -> None:
        for name, value, minimum in (
            ("n_actions", n_actions, 2),
            ("n_features", n_features, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.config = config or ContextualBanditConfig()
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        if initial_weights is None:
            self.weights = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
            self.weights[:, 0] = self.config.initial_reward_logit
        else:
            weights = np.asarray(initial_weights, dtype=np.float64)
            if weights.shape != (self.n_actions, self.n_features):
                raise ValueError("initial_weights have the wrong shape")
            if not np.all(np.isfinite(weights)):
                raise ValueError("initial_weights must be finite")
            self.weights = weights.copy()
        self.eligibilities = np.zeros_like(self.weights)
        self.belief = np.full(self.n_actions, 0.5, dtype=np.float64)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.reward_sums = np.zeros(self.n_actions, dtype=np.float64)
        self.n_reward_queries = 0
        self.update_l1_cost = 0.0
        self._update_l2_sq = 0.0

    def _context(self, value: ArrayLike) -> FloatArray:
        raw = np.asarray(value)
        if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
            raise TypeError("context must be a real numeric vector")
        context = np.asarray(raw, dtype=np.float64)
        if context.shape != (self.n_features,):
            raise ValueError(f"context must have shape ({self.n_features},)")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must be finite")
        return context

    def reset_sequence(self) -> None:
        """Reset observable state at a query-video boundary, not learned weights."""

        self.belief.fill(0.5)

    def action_values(self, context: ArrayLike) -> FloatArray:
        feature = self._context(context)
        return _sigmoid(self.weights @ feature)

    def select(
        self, context: ArrayLike, *, explore: bool
    ) -> tuple[int, FloatArray, FloatArray]:
        values = self.action_values(context)
        retention = self.config.belief_retention
        self.belief = retention * self.belief + (1.0 - retention) * values
        if explore and self._rng.random() < self.config.epsilon:
            action = int(self._rng.integers(self.n_actions))
        else:
            maximum = np.max(self.belief)
            tied = np.flatnonzero(
                np.isclose(self.belief, maximum, rtol=0.0, atol=1e-12)
            )
            action = int(tied[0])
        return action, _readonly(values), _readonly(self.belief)

    def observe(
        self,
        context: ArrayLike,
        *,
        executed_action: int,
        reward: float,
        credit_action: int | None = None,
    ) -> None:
        feature = self._context(context)
        if (
            isinstance(executed_action, bool)
            or not 0 <= int(executed_action) < self.n_actions
        ):
            raise ValueError("executed_action is outside the action range")
        executed_action = int(executed_action)
        credit = executed_action if credit_action is None else int(credit_action)
        if not 0 <= credit < self.n_actions:
            raise ValueError("credit_action is outside the action range")
        reward_value = float(reward)
        if not np.isfinite(reward_value) or not 0.0 <= reward_value <= 1.0:
            raise ValueError("reward must lie in [0, 1]")
        self.eligibilities *= self.config.eligibility_retention
        self.eligibilities[credit] += feature
        prediction = float(_sigmoid(np.asarray([self.weights[credit] @ feature]))[0])
        error = reward_value - prediction
        update = self.config.learning_rate * error * self.eligibilities[credit]
        if self.config.weight_decay:
            update -= (
                self.config.learning_rate
                * self.config.weight_decay
                * self.weights[credit]
            )
        self.weights[credit] += update
        self.action_counts[executed_action] += 1
        self.reward_sums[executed_action] += reward_value
        self.n_reward_queries += 1
        self.update_l1_cost += float(np.sum(np.abs(update)))
        self._update_l2_sq += float(np.sum(update**2))

    def train_step(
        self,
        context: ArrayLike,
        reward_provider: Callable[[int], float],
        *,
        credit_transform: Callable[[int], int] | None = None,
    ) -> tuple[int, float]:
        """Select, execute, query one reward, and apply one local update."""

        action, _, _ = self.select(context, explore=True)
        reward = float(reward_provider(action))
        credit = action if credit_transform is None else int(credit_transform(action))
        self.observe(
            context,
            executed_action=action,
            reward=reward,
            credit_action=credit,
        )
        return action, reward

    def predict_trace(self, trace: StreamingActuatorTrace) -> SelectorTrace:
        if trace.contexts.shape[1] != self.n_features:
            raise ValueError(
                "actuator trace context dimension does not match controller"
            )
        if trace.predictions.shape[1] != self.n_actions:
            raise ValueError(
                "actuator trace action dimension does not match controller"
            )
        n = trace.contexts.shape[0]
        actions = np.empty(n, dtype=np.int64)
        predictions = np.empty(n, dtype=np.int64)
        values = np.empty((n, self.n_actions), dtype=np.float64)
        beliefs = np.empty_like(values)
        costs = np.empty(n, dtype=np.float64)
        previous_video = ""
        for index in range(n):
            video_id = str(trace.video_ids[index])
            if video_id != previous_video:
                self.reset_sequence()
            action, action_values, belief = self.select(
                trace.contexts[index], explore=False
            )
            actions[index] = action
            predictions[index] = trace.predictions[index, action]
            values[index] = action_values
            beliefs[index] = belief
            costs[index] = trace.action_event_l1[index, action]
            previous_video = video_id
        return SelectorTrace(
            actions=actions,
            predictions=predictions,
            action_values=values,
            beliefs=beliefs,
            selected_event_l1=costs,
            video_ids=trace.video_ids,
        )

    def receipt(self) -> ContextualBanditReceipt:
        digest = hashlib.sha256(b"reward-only-contextual-controller-v1")
        for value in (
            self.weights,
            self.eligibilities,
            self.belief,
            self.action_counts,
            self.reward_sums,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(str(self.n_reward_queries).encode("ascii"))
        return ContextualBanditReceipt(
            weights=_readonly(self.weights),
            eligibilities=_readonly(self.eligibilities),
            belief=_readonly(self.belief),
            action_counts=_readonly(self.action_counts, dtype=np.int64),
            reward_sums=_readonly(self.reward_sums),
            n_reward_queries=int(self.n_reward_queries),
            update_l1_cost=float(self.update_l1_cost),
            update_l2_cost=float(np.sqrt(self._update_l2_sq)),
            used_autograd=False,
            used_bptt=False,
            fingerprint=digest.hexdigest(),
        )

    @classmethod
    def from_receipt(
        cls,
        receipt: ContextualBanditReceipt,
        *,
        config: ContextualBanditConfig,
        seed: int,
    ) -> "RewardOnlyContextualController":
        if receipt.used_autograd or receipt.used_bptt:
            raise ValueError("local controller receipt cannot use autograd or BPTT")
        return cls(
            receipt.weights.shape[0],
            receipt.weights.shape[1],
            config=config,
            seed=seed,
            initial_weights=receipt.weights,
        )
