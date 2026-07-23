from __future__ import annotations

import numpy as np

from src.plasticity.contextual_bandit_selector import (
    ContextualBanditConfig,
    RewardOnlyContextualController,
)


def test_reward_callback_is_queried_exactly_once() -> None:
    controller = RewardOnlyContextualController(
        2,
        2,
        config=ContextualBanditConfig(epsilon=0.0),
        seed=3,
    )
    queried: list[int] = []

    def reward_provider(action: int) -> float:
        queried.append(action)
        return float(action == 0)

    action, reward = controller.train_step([1.0, -1.0], reward_provider)
    assert queried == [action]
    assert reward == 1.0
    receipt = controller.receipt()
    assert receipt.n_reward_queries == 1
    assert receipt.used_autograd is False
    assert receipt.used_bptt is False


def test_local_controller_learns_context_action_matching() -> None:
    config = ContextualBanditConfig(
        learning_rate=0.08,
        epsilon=0.2,
        belief_retention=0.0,
    )
    controller = RewardOnlyContextualController(2, 2, config=config, seed=11)
    rng = np.random.default_rng(91)
    for _ in range(3000):
        sign = float(rng.choice((-1.0, 1.0)))
        context = np.asarray([1.0, sign])
        target_action = int(sign > 0.0)
        controller.reset_sequence()
        controller.train_step(
            context, lambda action, target=target_action: float(action == target)
        )

    for sign, expected in ((-1.0, 0), (1.0, 1)):
        controller.reset_sequence()
        action, _, _ = controller.select([1.0, sign], explore=False)
        assert action == expected
    assert controller.receipt().update_l1_cost > 0.0


def test_credit_shuffle_breaks_the_action_credit_mapping() -> None:
    config = ContextualBanditConfig(
        learning_rate=0.05,
        epsilon=0.25,
        belief_retention=0.0,
    )
    correct = RewardOnlyContextualController(2, 2, config=config, seed=7)
    shuffled = RewardOnlyContextualController(2, 2, config=config, seed=7)
    rng = np.random.default_rng(13)
    contexts = [np.asarray([1.0, float(rng.choice((-1.0, 1.0)))]) for _ in range(2000)]
    for context in contexts:
        target = int(context[1] > 0.0)
        correct.reset_sequence()
        shuffled.reset_sequence()

        def provider(action: int, target: int = target) -> float:
            return float(action == target)

        correct.train_step(context, provider)
        shuffled.train_step(
            context, provider, credit_transform=lambda action: 1 - action
        )

    def accuracy(controller: RewardOnlyContextualController) -> float:
        output = []
        for sign in (-1.0, 1.0) * 100:
            controller.reset_sequence()
            action, _, _ = controller.select([1.0, sign], explore=False)
            output.append(action == int(sign > 0.0))
        return float(np.mean(output))

    assert accuracy(correct) >= 0.99
    assert accuracy(correct) > accuracy(shuffled)
