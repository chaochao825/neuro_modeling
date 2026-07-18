"""Two-action local delta selector with executed-reward-only access."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ACTIONS = ("routing", "associative")
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def balanced_probe_schedule(n_probe_trials: int, *, seed: int) -> IntArray:
    """Return a randomized, exactly balanced forced-exploration schedule."""

    if isinstance(n_probe_trials, (bool, np.bool_)) or not isinstance(
        n_probe_trials, (int, np.integer)
    ):
        raise TypeError("n_probe_trials must be an integer")
    n_probe_trials = int(n_probe_trials)
    if n_probe_trials < 2 or n_probe_trials % 2:
        raise ValueError("n_probe_trials must be a positive even integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    schedule = np.tile(np.arange(len(ACTIONS), dtype=np.int64), n_probe_trials // 2)
    rng.shuffle(schedule)
    schedule.setflags(write=False)
    return schedule


@dataclass(frozen=True, slots=True, eq=False)
class RewardOnlySelectorReceipt:
    q_values: FloatArray
    action_counts: IntArray
    observed_actions: IntArray
    observed_rewards: FloatArray
    selected_action: int
    cumulative_update_l1: float
    cumulative_update_l2: float
    observation_fingerprint: str
    used_true_context: bool = False
    used_counterfactual_reward: bool = False
    used_autograd: bool = False
    used_bptt: bool = False

    def __post_init__(self) -> None:
        q = np.array(self.q_values, dtype=np.float64, copy=True)
        counts = np.array(self.action_counts, dtype=np.int64, copy=True)
        actions = np.array(self.observed_actions, dtype=np.int64, copy=True)
        rewards = np.array(self.observed_rewards, dtype=np.float64, copy=True)
        if q.shape != (2,) or counts.shape != (2,):
            raise ValueError("q_values and action_counts must have length two")
        if actions.ndim != 1 or rewards.shape != actions.shape:
            raise ValueError("observed actions and rewards must be paired vectors")
        if np.any((actions < 0) | (actions >= 2)):
            raise ValueError("observed action index is invalid")
        if np.any((rewards < 0.0) | (rewards > 1.0)):
            raise ValueError("observed rewards must lie in [0, 1]")
        for value in (q, counts, actions, rewards):
            value.setflags(write=False)
        object.__setattr__(self, "q_values", q)
        object.__setattr__(self, "action_counts", counts)
        object.__setattr__(self, "observed_actions", actions)
        object.__setattr__(self, "observed_rewards", rewards)


class RewardOnlyDeltaSelector:
    """Sample-average local rule: action eligibility times scalar RPE.

    ``observe`` accepts exactly one action index and one scalar reward.  There
    is no method argument for context descriptors or unexecuted utilities.
    """

    used_true_context = False
    used_counterfactual_reward = False
    used_autograd = False
    used_bptt = False

    def __init__(self, *, q_prior: float = 0.5, tie_break_action: int = 0) -> None:
        prior = float(q_prior)
        if not np.isfinite(prior) or not 0.0 <= prior <= 1.0:
            raise ValueError("q_prior must lie in [0, 1]")
        if isinstance(tie_break_action, (bool, np.bool_)) or not isinstance(
            tie_break_action, (int, np.integer)
        ):
            raise TypeError("tie_break_action must be an integer")
        self.tie_break_action = int(tie_break_action)
        if self.tie_break_action not in range(len(ACTIONS)):
            raise ValueError("tie_break_action is invalid")
        self._q = np.full(len(ACTIONS), prior, dtype=np.float64)
        self._counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._update_l1 = 0.0
        self._update_squared_l2 = 0.0

    @property
    def q_values(self) -> FloatArray:
        result = np.array(self._q, copy=True)
        result.setflags(write=False)
        return result

    def observe(self, action: int, reward: float) -> None:
        """Apply one local update from the reward of the executed action only."""

        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise TypeError("action must be an integer index")
        action = int(action)
        if action not in range(len(ACTIONS)):
            raise ValueError("action index is invalid")
        if isinstance(reward, (bool, np.bool_)):
            reward = float(reward)
        if not isinstance(reward, (int, float, np.integer, np.floating)):
            raise TypeError("reward must be one scalar")
        reward = float(reward)
        if not np.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be a finite scalar in [0, 1]")
        self._counts[action] += 1
        step_size = 1.0 / float(self._counts[action])
        update = step_size * (reward - self._q[action])
        self._q[action] += update
        self._update_l1 += abs(update)
        self._update_squared_l2 += update * update
        self._actions.append(action)
        self._rewards.append(reward)

    def select(self) -> int:
        if np.any(self._counts == 0):
            raise RuntimeError("every action must be probed before deployment")
        maximum = float(np.max(self._q))
        winners = np.flatnonzero(np.isclose(self._q, maximum, atol=0.0, rtol=0.0))
        if self.tie_break_action in winners:
            return self.tie_break_action
        return int(winners[0])

    def receipt(self) -> RewardOnlySelectorReceipt:
        selected = self.select()
        actions = np.asarray(self._actions, dtype=np.int64)
        rewards = np.asarray(self._rewards, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(actions.astype("<i8", copy=False).tobytes())
        digest.update(rewards.astype("<f8", copy=False).tobytes())
        return RewardOnlySelectorReceipt(
            q_values=self._q,
            action_counts=self._counts,
            observed_actions=actions,
            observed_rewards=rewards,
            selected_action=selected,
            cumulative_update_l1=float(self._update_l1),
            cumulative_update_l2=float(np.sqrt(self._update_squared_l2)),
            observation_fingerprint=digest.hexdigest(),
        )


__all__ = [
    "ACTIONS",
    "RewardOnlyDeltaSelector",
    "RewardOnlySelectorReceipt",
    "balanced_probe_schedule",
]
