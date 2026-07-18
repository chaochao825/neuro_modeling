"""Persistent local and Bayesian reward beliefs for sparse delayed feedback.

The local controller exposes no API for true context or counterfactual reward.
Its internal state contains two retained action values; their difference is
the one-dimensional action-control coordinate.  Each reward update is the
eligibility of one credited action multiplied by one scalar reward prediction
error evaluated when the delayed reward is delivered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _readonly(value: object, *, dtype: np.dtype, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"array must be {ndim}-dimensional")
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("floating arrays must be finite")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _softmax_binary(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) / temperature
    logits = logits - float(np.max(logits))
    probability = np.exp(logits)
    probability /= float(np.sum(probability))
    return probability


@dataclass(frozen=True, slots=True)
class _PendingReward:
    source_trial: int
    due_trial: int
    executed_action: int
    credited_action: int
    reward: float


@dataclass(frozen=True, slots=True, eq=False)
class RewardBeliefReceipt:
    method: str
    actions: IntArray
    action_one_probabilities: FloatArray
    belief_probabilities: FloatArray
    delivered_source_trials: IntArray
    delivered_executed_actions: IntArray
    delivered_credited_actions: IntArray
    delivered_rewards: FloatArray
    q_values: FloatArray
    cumulative_update_l1: float
    cumulative_update_l2: float
    cumulative_retention_l1: float
    cumulative_retention_l2: float
    cumulative_total_state_update_l1: float
    cumulative_total_state_update_l2: float
    internal_state_dimension: int
    control_dimension: int
    belief_semantics: str
    rpe_reference: str
    pending_feedback_count: int
    observation_fingerprint: str
    used_true_context: bool
    used_counterfactual_reward: bool
    used_autograd: bool
    used_bptt: bool

    def __post_init__(self) -> None:
        actions = _readonly(self.actions, dtype=np.dtype(np.int64), ndim=1)
        action_p = _readonly(
            self.action_one_probabilities, dtype=np.dtype(np.float64), ndim=1
        )
        belief_p = _readonly(
            self.belief_probabilities, dtype=np.dtype(np.float64), ndim=1
        )
        source = _readonly(
            self.delivered_source_trials, dtype=np.dtype(np.int64), ndim=1
        )
        executed = _readonly(
            self.delivered_executed_actions, dtype=np.dtype(np.int64), ndim=1
        )
        credited = _readonly(
            self.delivered_credited_actions, dtype=np.dtype(np.int64), ndim=1
        )
        rewards = _readonly(self.delivered_rewards, dtype=np.dtype(np.float64), ndim=1)
        q_values = _readonly(self.q_values, dtype=np.dtype(np.float64), ndim=1)
        if action_p.shape != actions.shape or belief_p.shape != actions.shape:
            raise ValueError("per-trial receipt arrays must have identical shape")
        if not (source.shape == executed.shape == credited.shape == rewards.shape):
            raise ValueError("delivered reward arrays must have identical shape")
        if q_values.shape != (2,):
            raise ValueError("q_values must have length two")
        for value in (action_p, belief_p, rewards, q_values):
            if np.any((value < 0.0) | (value > 1.0)):
                raise ValueError("probability and reward values must lie in [0, 1]")
        if np.any((actions < 0) | (actions > 1)):
            raise ValueError("actions must be binary")
        if len(self.observation_fingerprint) != 64:
            raise ValueError("observation_fingerprint must be a SHA-256 digest")
        for name in (
            "cumulative_update_l1",
            "cumulative_update_l2",
            "cumulative_retention_l1",
            "cumulative_retention_l2",
            "cumulative_total_state_update_l1",
            "cumulative_total_state_update_l2",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.internal_state_dimension < 1 or self.control_dimension < 1:
            raise ValueError("state and control dimensions must be positive")
        if not self.belief_semantics or not self.rpe_reference:
            raise ValueError("belief and RPE semantics must be explicit")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "action_one_probabilities", action_p)
        object.__setattr__(self, "belief_probabilities", belief_p)
        object.__setattr__(self, "delivered_source_trials", source)
        object.__setattr__(self, "delivered_executed_actions", executed)
        object.__setattr__(self, "delivered_credited_actions", credited)
        object.__setattr__(self, "delivered_rewards", rewards)
        object.__setattr__(self, "q_values", q_values)


class PersistentRewardBeliefSelector:
    """Two-action local RPE controller with retention and delayed eligibility."""

    used_true_context = False
    used_counterfactual_reward = False
    used_autograd = False
    used_bptt = False

    def __init__(
        self,
        *,
        alpha: float = 0.30,
        retention: float = 0.98,
        temperature: float = 0.08,
        q_prior: float = 0.5,
        seed: int = 0,
        update_mode: str = "fixed_rpe",
        credit_assignment: str = "executed",
    ) -> None:
        self.alpha = _unit_interval(alpha, name="alpha")
        if self.alpha == 0.0:
            raise ValueError("alpha must be positive")
        self.retention = _unit_interval(retention, name="retention")
        self.temperature = _positive(temperature, name="temperature")
        self.q_prior = _unit_interval(q_prior, name="q_prior")
        self.seed = _integer(seed, name="seed")
        if update_mode not in {"fixed_rpe", "sample_average"}:
            raise ValueError("update_mode must be fixed_rpe or sample_average")
        if credit_assignment not in {"executed", "opposite"}:
            raise ValueError("credit_assignment must be executed or opposite")
        if update_mode == "sample_average" and retention != 1.0:
            raise ValueError("sample-average mode requires retention=1")
        self.update_mode = update_mode
        self.credit_assignment = credit_assignment
        self._rng = np.random.default_rng(self.seed)
        self._q = np.full(2, self.q_prior, dtype=np.float64)
        self._counts = np.zeros(2, dtype=np.int64)
        self._pending: list[_PendingReward] = []
        self._actions: list[int] = []
        self._action_p: list[float] = []
        self._belief_p: list[float] = []
        self._delivered: list[_PendingReward] = []
        self._update_l1 = 0.0
        self._update_squared_l2 = 0.0
        self._retention_l1 = 0.0
        self._retention_squared_l2 = 0.0
        self._total_state_update_l1 = 0.0
        self._total_state_update_squared_l2 = 0.0
        self._last_advanced = -1
        self._choose_open = False

    @property
    def q_values(self) -> FloatArray:
        result = np.array(self._q, copy=True)
        result.setflags(write=False)
        return result

    def _apply(self, event: _PendingReward) -> None:
        action = event.credited_action
        self._counts[action] += 1
        step = (
            1.0 / float(self._counts[action])
            if self.update_mode == "sample_average"
            else self.alpha
        )
        update = step * (event.reward - self._q[action])
        self._q[action] += update
        self._update_l1 += abs(update)
        self._update_squared_l2 += update * update
        self._total_state_update_l1 += abs(update)
        self._total_state_update_squared_l2 += update * update
        self._delivered.append(event)

    def advance(self, trial: int) -> None:
        """Advance one trial, decay the belief, and deliver due rewards."""

        trial = _integer(trial, name="trial")
        if trial != self._last_advanced + 1:
            raise ValueError("trials must advance consecutively")
        if self._choose_open:
            raise RuntimeError("the previous trial did not choose an action")
        retained = self.q_prior + self.retention * (self._q - self.q_prior)
        drift = retained - self._q
        self._q = retained
        self._retention_l1 += float(np.sum(np.abs(drift)))
        self._retention_squared_l2 += float(np.sum(drift * drift))
        self._total_state_update_l1 += float(np.sum(np.abs(drift)))
        self._total_state_update_squared_l2 += float(np.sum(drift * drift))
        due = [event for event in self._pending if event.due_trial <= trial]
        self._pending = [event for event in self._pending if event.due_trial > trial]
        for event in sorted(due, key=lambda item: (item.due_trial, item.source_trial)):
            self._apply(event)
        self._last_advanced = trial
        self._choose_open = True

    def choose(self, uniform: float | None = None) -> int:
        if not self._choose_open:
            raise RuntimeError("advance must be called exactly once before choose")
        probability = _softmax_binary(self._q, self.temperature)
        if uniform is None:
            action = int(self._rng.choice(2, p=probability))
        else:
            draw = _unit_interval(uniform, name="uniform")
            action = int(draw < float(probability[1]))
        self._actions.append(action)
        self._action_p.append(float(probability[1]))
        self._belief_p.append(float(probability[1]))
        self._choose_open = False
        return action

    def schedule_feedback(
        self,
        *,
        source_trial: int,
        executed_action: int,
        reward: float,
        available: bool,
        delay: int,
    ) -> None:
        """Schedule only the scalar reward of the executed action."""

        source_trial = _integer(source_trial, name="source_trial")
        executed_action = _integer(executed_action, name="executed_action")
        delay = _integer(delay, name="delay")
        if executed_action not in (0, 1):
            raise ValueError("executed_action must be binary")
        if source_trial != self._last_advanced or self._choose_open:
            raise RuntimeError("feedback must follow the action from the current trial")
        if not isinstance(available, (bool, np.bool_)):
            raise TypeError("available must be boolean")
        reward = _unit_interval(reward, name="reward")
        if not bool(available):
            return
        credited = (
            executed_action
            if self.credit_assignment == "executed"
            else 1 - executed_action
        )
        self._pending.append(
            _PendingReward(
                source_trial=source_trial,
                due_trial=source_trial + delay + 1,
                executed_action=executed_action,
                credited_action=credited,
                reward=reward,
            )
        )

    def receipt(self) -> RewardBeliefReceipt:
        if self._choose_open:
            raise RuntimeError(
                "cannot create a receipt before choosing the current action"
            )
        delivered = tuple(self._delivered)
        source = np.asarray([item.source_trial for item in delivered], dtype=np.int64)
        executed = np.asarray(
            [item.executed_action for item in delivered], dtype=np.int64
        )
        credited = np.asarray(
            [item.credited_action for item in delivered], dtype=np.int64
        )
        rewards = np.asarray([item.reward for item in delivered], dtype=np.float64)
        digest = hashlib.sha256()
        for array in (source, executed, credited, rewards):
            digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(b"\0")
        method = (
            "cumulative_sample_average"
            if self.update_mode == "sample_average"
            else (
                "opposite_eligibility_local"
                if self.credit_assignment == "opposite"
                else "persistent_rpe_local"
            )
        )
        return RewardBeliefReceipt(
            method=method,
            actions=np.asarray(self._actions, dtype=np.int64),
            action_one_probabilities=np.asarray(self._action_p, dtype=np.float64),
            belief_probabilities=np.asarray(self._belief_p, dtype=np.float64),
            delivered_source_trials=source,
            delivered_executed_actions=executed,
            delivered_credited_actions=credited,
            delivered_rewards=rewards,
            q_values=self._q,
            cumulative_update_l1=float(self._update_l1),
            cumulative_update_l2=float(np.sqrt(self._update_squared_l2)),
            cumulative_retention_l1=float(self._retention_l1),
            cumulative_retention_l2=float(np.sqrt(self._retention_squared_l2)),
            cumulative_total_state_update_l1=float(self._total_state_update_l1),
            cumulative_total_state_update_l2=float(
                np.sqrt(self._total_state_update_squared_l2)
            ),
            internal_state_dimension=2,
            control_dimension=1,
            belief_semantics="action_one_policy_probability_proxy",
            rpe_reference="delivery_time_current_credited_action_value",
            pending_feedback_count=len(self._pending),
            observation_fingerprint=digest.hexdigest(),
            used_true_context=False,
            used_counterfactual_reward=False,
            used_autograd=False,
            used_bptt=False,
        )


class BayesianDelayedRewardFilter:
    """Known-hazard HMM comparator fitted only from train reward emissions."""

    used_true_context = False
    used_counterfactual_reward = False
    used_autograd = False
    used_bptt = False

    def __init__(
        self,
        emission_probabilities: object,
        *,
        hazard: float,
        temperature: float = 0.08,
        seed: int = 0,
    ) -> None:
        emission = np.asarray(emission_probabilities, dtype=np.float64)
        if emission.shape != (2, 2) or not np.all(np.isfinite(emission)):
            raise ValueError("emission_probabilities must be a finite 2x2 matrix")
        if np.any((emission <= 0.0) | (emission >= 1.0)):
            raise ValueError("emission probabilities must lie strictly in (0, 1)")
        self.emission = np.array(emission, copy=True)
        self.hazard = _unit_interval(hazard, name="hazard")
        if self.hazard == 0.0:
            raise ValueError("hazard must be positive")
        self.temperature = _positive(temperature, name="temperature")
        self.seed = _integer(seed, name="seed")
        self._rng = np.random.default_rng(self.seed)
        self._transition = np.array(
            [[1.0 - self.hazard, self.hazard], [self.hazard, 1.0 - self.hazard]],
            dtype=np.float64,
        )
        self._pending: list[_PendingReward] = []
        self._observations: dict[int, tuple[int, float]] = {}
        self._posterior: list[np.ndarray] = []
        self._current_prior = np.array([0.5, 0.5], dtype=np.float64)
        self._actions: list[int] = []
        self._action_p: list[float] = []
        self._belief_p: list[float] = []
        self._delivered: list[_PendingReward] = []
        self._last_advanced = -1
        self._choose_open = False

    def _recompute_from(self, start_trial: int, end_trial: int) -> None:
        if start_trial < 0 or end_trial < start_trial:
            raise ValueError("invalid Bayesian recomputation interval")
        if start_trial == 0:
            prior = np.array([0.5, 0.5], dtype=np.float64)
            prefix: list[np.ndarray] = []
        else:
            if start_trial - 1 >= len(self._posterior):
                raise RuntimeError("Bayesian posterior prefix is incomplete")
            prior = self._posterior[start_trial - 1] @ self._transition
            prefix = self._posterior[:start_trial]
        posterior = list(prefix)
        for trial in range(start_trial, end_trial + 1):
            if trial > start_trial:
                prior = posterior[-1] @ self._transition
            observation = self._observations.get(trial)
            if observation is None:
                current = prior
            else:
                action, reward = observation
                likelihood = (
                    self.emission[:, action]
                    if reward >= 0.5
                    else 1.0 - self.emission[:, action]
                )
                current = prior * likelihood
                total = float(np.sum(current))
                if total <= 0.0 or not np.isfinite(total):
                    raise FloatingPointError("Bayesian reward posterior collapsed")
                current = current / total
            posterior.append(np.array(current, copy=True))
        self._posterior = posterior

    def advance(self, trial: int) -> None:
        trial = _integer(trial, name="trial")
        if trial != self._last_advanced + 1:
            raise ValueError("trials must advance consecutively")
        if self._choose_open:
            raise RuntimeError("the previous trial did not choose an action")
        due = [event for event in self._pending if event.due_trial <= trial]
        self._pending = [event for event in self._pending if event.due_trial > trial]
        for event in sorted(due, key=lambda item: (item.due_trial, item.source_trial)):
            self._observations[event.source_trial] = (
                event.executed_action,
                event.reward,
            )
            self._delivered.append(event)
        if trial:
            start = (
                min(event.source_trial for event in due)
                if due
                else len(self._posterior)
            )
            if start <= trial - 1:
                self._recompute_from(start, trial - 1)
            self._current_prior = self._posterior[-1] @ self._transition
        else:
            self._posterior = []
            self._current_prior = np.array([0.5, 0.5], dtype=np.float64)
        self._last_advanced = trial
        self._choose_open = True

    def choose(self, uniform: float | None = None) -> int:
        if not self._choose_open:
            raise RuntimeError("advance must be called exactly once before choose")
        expected_reward = self._current_prior @ self.emission
        probability = _softmax_binary(expected_reward, self.temperature)
        if uniform is None:
            action = int(self._rng.choice(2, p=probability))
        else:
            draw = _unit_interval(uniform, name="uniform")
            action = int(draw < float(probability[1]))
        self._actions.append(action)
        self._action_p.append(float(probability[1]))
        self._belief_p.append(float(self._current_prior[1]))
        self._choose_open = False
        return action

    def schedule_feedback(
        self,
        *,
        source_trial: int,
        executed_action: int,
        reward: float,
        available: bool,
        delay: int,
    ) -> None:
        source_trial = _integer(source_trial, name="source_trial")
        executed_action = _integer(executed_action, name="executed_action")
        delay = _integer(delay, name="delay")
        if executed_action not in (0, 1):
            raise ValueError("executed_action must be binary")
        if source_trial != self._last_advanced or self._choose_open:
            raise RuntimeError("feedback must follow the action from the current trial")
        if not isinstance(available, (bool, np.bool_)):
            raise TypeError("available must be boolean")
        reward = _unit_interval(reward, name="reward")
        if bool(available):
            self._pending.append(
                _PendingReward(
                    source_trial=source_trial,
                    due_trial=source_trial + delay + 1,
                    executed_action=executed_action,
                    credited_action=executed_action,
                    reward=reward,
                )
            )

    def receipt(self) -> RewardBeliefReceipt:
        delivered = tuple(self._delivered)
        source = np.asarray([item.source_trial for item in delivered], dtype=np.int64)
        executed = np.asarray(
            [item.executed_action for item in delivered], dtype=np.int64
        )
        rewards = np.asarray([item.reward for item in delivered], dtype=np.float64)
        digest = hashlib.sha256()
        for array in (source, executed, rewards):
            digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(b"\0")
        return RewardBeliefReceipt(
            method="bayes_reward_filter",
            actions=np.asarray(self._actions, dtype=np.int64),
            action_one_probabilities=np.asarray(self._action_p, dtype=np.float64),
            belief_probabilities=np.asarray(self._belief_p, dtype=np.float64),
            delivered_source_trials=source,
            delivered_executed_actions=executed,
            delivered_credited_actions=executed,
            delivered_rewards=rewards,
            q_values=self._current_prior,
            cumulative_update_l1=0.0,
            cumulative_update_l2=0.0,
            cumulative_retention_l1=0.0,
            cumulative_retention_l2=0.0,
            cumulative_total_state_update_l1=0.0,
            cumulative_total_state_update_l2=0.0,
            internal_state_dimension=1,
            control_dimension=1,
            belief_semantics="hmm_state_one_posterior",
            rpe_reference="not_applicable_exact_bayesian_filter",
            pending_feedback_count=len(self._pending),
            observation_fingerprint=digest.hexdigest(),
            used_true_context=False,
            used_counterfactual_reward=False,
            used_autograd=False,
            used_bptt=False,
        )


__all__ = [
    "BayesianDelayedRewardFilter",
    "PersistentRewardBeliefSelector",
    "RewardBeliefReceipt",
]
