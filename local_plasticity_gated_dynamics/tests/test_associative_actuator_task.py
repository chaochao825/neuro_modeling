from __future__ import annotations

import numpy as np
import pytest

from src.tasks.associative_actuator import (
    AssociativeActuatorTaskConfig,
    make_associative_actuator_dataset,
)


def _config() -> AssociativeActuatorTaskConfig:
    return AssociativeActuatorTaskConfig(
        n_train_blocks=4,
        n_test_blocks=2,
        trials_per_block=12,
        key_dim=6,
        n_pairs=3,
        delay=2,
        target_noise_std=0.05,
    )


def test_associative_task_is_deterministic_block_split_and_immutable() -> None:
    first = make_associative_actuator_dataset(_config(), 17)
    second = make_associative_actuator_dataset(_config(), 17)
    other = make_associative_actuator_dataset(_config(), 18)
    assert first.train.fingerprint == second.train.fingerprint
    assert first.test.fingerprint == second.test.fingerprint
    assert first.train.fingerprint != other.train.fingerprint
    assert set(first.train.block_ids.tolist()).isdisjoint(first.test.block_ids.tolist())
    assert not first.train.write_keys.flags.writeable
    assert not first.train.write_values.flags.writeable
    assert not first.test.query_keys.flags.writeable
    with pytest.raises(ValueError):
        first.train.write_values[0, 0] = 0.0


def test_query_contains_no_value_and_registered_target_is_exact() -> None:
    dataset = make_associative_actuator_dataset(_config(), 21)
    split = dataset.train
    retrieved = np.einsum(
        "np,npi,ni->n",
        split.write_values,
        split.write_keys,
        split.query_keys,
        optimize=True,
    )
    np.testing.assert_array_equal(retrieved, split.retrieval_targets)
    # The same visible query key occurs with both target signs, so the query
    # alone cannot leak the trial-specific value.
    for key_index in range(dataset.config.key_dim):
        rows = np.flatnonzero(split.query_keys[:, key_index] == 1.0)
        if rows.size >= 4:
            assert set(split.retrieval_targets[rows].tolist()) == {-1.0, 1.0}
    mu = 0.75
    expected = (
        np.sqrt(1.0 - mu) * split.direct_cues
        + np.sqrt(mu) * split.retrieval_targets
        + split.target_noise
    )
    np.testing.assert_allclose(split.target(mu), expected)
    assert not split.target(mu).flags.writeable


def test_associative_task_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        AssociativeActuatorTaskConfig(key_dim=3, n_pairs=4)
    with pytest.raises(ValueError, match="non-negative"):
        AssociativeActuatorTaskConfig(target_noise_std=-0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_associative_actuator_dataset(_config(), 1).train.target(1.1)
