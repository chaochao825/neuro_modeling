from __future__ import annotations

import numpy as np
import pytest

from src.tasks.nonstationary_actuator_stream import (
    NonstationaryActuatorStreamConfig,
    make_hidden_state_tape,
    make_nested_feedback_tapes,
    make_stream_tape,
)


def _config() -> NonstationaryActuatorStreamConfig:
    return NonstationaryActuatorStreamConfig(
        n_trials=128,
        key_dim=4,
        direct_reliabilities=(0.6, 0.9),
        load_values=(2, 4),
        distractor_write_values=(0, 2),
        hazards=(0.05, 0.2),
        feedback_fractions=(0.5, 0.125),
        feedback_delays=(0, 4),
    )


def test_hidden_state_and_nested_feedback_tapes_are_reproducible() -> None:
    config = _config()
    states, switches, digest = make_hidden_state_tape(config, seed=3, hazard=0.05)
    again = make_hidden_state_tape(config, seed=3, hazard=0.05)
    np.testing.assert_array_equal(states, again[0])
    np.testing.assert_array_equal(switches, again[1])
    assert digest == again[2]
    np.testing.assert_array_equal(switches[1:], states[1:] != states[:-1])
    assert not states.flags.writeable
    schedules = make_nested_feedback_tapes(config, seed=3)
    dense = schedules[0.5][0]
    sparse = schedules[0.125][0]
    assert np.all(~sparse | dense)
    assert dense.sum() == round(0.5 * config.feedback_eligible_trials)
    assert sparse.sum() == round(0.125 * config.feedback_eligible_trials)
    assert not np.any(dense[config.feedback_eligible_trials :])


def test_stream_tape_does_not_change_external_tapes_with_delay() -> None:
    config = _config()
    first = make_stream_tape(
        config,
        seed=7,
        hazard=0.2,
        feedback_fraction=0.125,
        feedback_delay=0,
    )
    delayed = make_stream_tape(
        config,
        seed=7,
        hazard=0.2,
        feedback_fraction=0.125,
        feedback_delay=4,
    )
    assert first.state_fingerprint == delayed.state_fingerprint
    assert first.feedback_fingerprint == delayed.feedback_fingerprint
    np.testing.assert_array_equal(first.feedback_available, delayed.feedback_available)
    with pytest.raises(ValueError, match="hazard"):
        make_stream_tape(
            config,
            seed=7,
            hazard=0.1,
            feedback_fraction=0.125,
            feedback_delay=0,
        )
