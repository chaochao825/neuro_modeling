from __future__ import annotations

import numpy as np
import pytest

from src.models.capacity_limited_associative_actuator import (
    CapacityLimitedAssociativeActuator,
)
from src.tasks.hidden_reliability_association import (
    HiddenReliabilityTaskConfig,
    make_hidden_reliability_block_specs,
    materialize_hidden_reliability_block,
)


def _config() -> HiddenReliabilityTaskConfig:
    return HiddenReliabilityTaskConfig(
        n_train_blocks_per_cell=2,
        n_test_blocks_per_cell=2,
        trials_per_block=16,
        probe_trials=4,
        key_dim=4,
        load_values=(2, 4),
        distractor_write_values=(0, 2),
        direct_reliabilities=(0.6, 0.9),
    )


def test_hidden_reliability_task_is_deterministic_dense_and_block_split() -> None:
    config = _config()
    specs = make_hidden_reliability_block_specs(config, 31)
    assert len(specs) == 2 * 2 * 2 * (2 + 2)
    train_ids = {spec.block_id for spec in specs if spec.split == "train"}
    test_ids = {spec.block_id for spec in specs if spec.split == "test"}
    assert train_ids.isdisjoint(test_ids)
    first = materialize_hidden_reliability_block(config, specs[0])
    repeated = materialize_hidden_reliability_block(config, specs[0])
    assert first.fingerprint == repeated.fingerprint
    assert np.array_equal(first.write_keys, repeated.write_keys)
    assert set(np.unique(np.abs(first.write_keys))) == {0.5}
    assert np.allclose(np.linalg.norm(first.write_keys, axis=2), 1.0)
    assert set(np.unique(first.targets)) <= {-1.0, 1.0}
    assert set(np.unique(first.direct_cues)) <= {-1.0, 1.0}
    assert first.write_keys.flags.writeable is False


def test_capacity_limited_actuator_preserves_query_shuffle_write_budget() -> None:
    config = _config()
    spec = next(
        item
        for item in make_hidden_reliability_block_specs(config, 44)
        if item.split == "test" and item.distractor_writes == 2
    )
    block = materialize_hidden_reliability_block(config, spec)
    actuator = CapacityLimitedAssociativeActuator(
        key_dim=config.key_dim,
        distractor_strength=config.distractor_strength,
    )
    outputs = actuator.evaluate(block)
    assert outputs.update_budget_exact
    assert outputs.associative_budget == outputs.shuffled_budget
    assert outputs.associative_budget.per_write_rank == 1
    assert outputs.associative_budget.writes_per_trial == (
        spec.association_load + spec.distractor_writes
    )
    assert set(np.unique(outputs.routing)) <= {-1.0, 1.0}
    assert set(np.unique(outputs.associative)) <= {-1.0, 1.0}
    assert set(np.unique(outputs.associative_query_shuffled)) <= {-1.0, 1.0}
    assert outputs.associative.flags.writeable is False
    with pytest.raises(ValueError, match="dimension"):
        CapacityLimitedAssociativeActuator(key_dim=5).evaluate(block)


def test_task_rejects_unbalanced_probe_and_nonhidden_reliability() -> None:
    with pytest.raises(ValueError, match="even"):
        HiddenReliabilityTaskConfig(probe_trials=3)
    with pytest.raises(ValueError, match=r"\(0.5, 1\]"):
        HiddenReliabilityTaskConfig(direct_reliabilities=(0.5, 0.9))
