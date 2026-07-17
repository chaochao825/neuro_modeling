from __future__ import annotations

import numpy as np

from src.models.associative_memory_actuator import (
    AssociativeMemoryActuator,
    FrozenCarrierBridge,
)
from src.tasks.actuator_matching import CarrierConfig, make_carrier
from src.tasks.associative_actuator import (
    AssociativeActuatorTaskConfig,
    make_associative_actuator_dataset,
)


def _split():
    config = AssociativeActuatorTaskConfig(
        n_train_blocks=2,
        n_test_blocks=2,
        trials_per_block=16,
        key_dim=8,
        n_pairs=4,
        delay=3,
        target_noise_std=0.0,
    )
    return make_associative_actuator_dataset(config, 33).train


def test_local_outer_product_retrieves_and_shuffled_budget_is_exact() -> None:
    split = _split()
    actuator = AssociativeMemoryActuator.random(key_dim=8, seed=7)
    np.testing.assert_array_equal(actuator.retrieve(split), split.retrieval_targets)
    shuffled = actuator.retrieve_shuffled(split)
    assert float(np.mean(shuffled == split.retrieval_targets)) < 0.8
    direct_budget = actuator.write_budget(split)
    shuffled_budget = actuator.write_budget(split, shuffled=True)
    assert direct_budget == shuffled_budget
    assert direct_budget.mean_l1 == 4.0
    assert direct_budget.mean_l2 == 2.0
    assert direct_budget.max_update_rank == 1


def test_low_rank_history_is_one_dimensional_and_not_exact_content_addressing() -> None:
    split = _split()
    actuator = AssociativeMemoryActuator.random(key_dim=8, seed=8)
    compressed = actuator.compressive_retrieval(split)
    assert compressed.shape == split.retrieval_targets.shape
    assert np.corrcoef(compressed, split.retrieval_targets)[0, 1] < 0.9


def test_frozen_dale_carrier_bridge_transmits_one_shared_scalar_axis() -> None:
    carrier = make_carrier(
        CarrierConfig(n_neurons=16, n_inputs=4, n_outputs=1), seed=91
    )
    bridge = FrozenCarrierBridge.from_carrier(carrier)
    controls = np.linspace(-2.0, 2.0, 13)
    np.testing.assert_allclose(bridge.transmit(controls), controls, atol=1e-12)
    assert np.linalg.matrix_rank(carrier.a0) == 16
    assert carrier.spectral_radius < 1.0
    assert bridge.reconstruction_error < 1e-12
