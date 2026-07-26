from __future__ import annotations

import numpy as np
import pytest

from src.models.factorized_uncertainty_filter import (
    AdaptationRates,
    JumpFilterParameters,
    run_factorized_filter,
    run_fixed_jump_filter,
    run_imm_filter,
    run_oracle_filter,
)
from src.tasks.factorized_uncertainty import (
    FactorialStreamConfig,
    UncertaintyLevels,
    all_factorial_cells,
    generate_uncertainty_tape,
)


def _tape():
    return generate_uncertainty_tape(
        seed=390,
        split="filter-unit",
        cells=all_factorial_cells(),
        config=FactorialStreamConfig(
            block_length=16, blocks_per_sequence=8, n_sequences=2
        ),
    )


def test_zero_adaptation_matches_fixed_filter_exactly() -> None:
    tape = _tape()
    parameters = JumpFilterParameters(0.02, 0.01, 0.05, 4.0)
    adaptive = run_factorized_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        initial=parameters,
        adaptation=AdaptationRates(0.0, 0.0, 0.0),
    )
    fixed = run_fixed_jump_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        parameters=parameters,
    )
    for field in adaptive.__dict__:
        np.testing.assert_allclose(
            getattr(adaptive, field), getattr(fixed, field), rtol=0.0, atol=0.0
        )


def test_factorized_filter_is_causal_and_clamps_only_named_parameter() -> None:
    tape = _tape()
    initial = JumpFilterParameters(0.02, 0.01, 0.05, 4.0)
    adaptation = AdaptationRates(0.1, 0.1, 0.1)
    prefix = 100
    full = run_factorized_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        initial=initial,
        adaptation=adaptation,
    )
    altered = tape.observations.copy()
    altered[prefix:] += 100.0
    changed = run_factorized_filter(
        altered,
        sequence_ids=tape.sequence_ids,
        initial=initial,
        adaptation=adaptation,
    )
    np.testing.assert_allclose(full.predictive_nll[:prefix], changed.predictive_nll[:prefix])
    clamped = run_factorized_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        initial=initial,
        adaptation=adaptation,
        clamp="q",
    )
    np.testing.assert_allclose(clamped.process_variance, initial.process_variance)
    assert np.std(clamped.hazard) > 0.0
    assert np.std(clamped.observation_variance) > 0.0


def test_imm_single_mode_is_rejected_and_oracle_arrays_must_align() -> None:
    tape = _tape()
    mode = JumpFilterParameters(0.02, 0.01, 0.05, 4.0)
    with pytest.raises(ValueError, match="at least two"):
        run_imm_filter(
            tape.observations,
            sequence_ids=tape.sequence_ids,
            modes=(mode,),
            mode_switch_probability=0.01,
        )
    with pytest.raises(ValueError, match="align"):
        run_oracle_filter(
            tape.observations,
            sequence_ids=tape.sequence_ids,
            hazard=tape.hazard[:-1],
            process_variance=tape.process_variance,
            observation_variance=tape.observation_variance,
            jump_variance=4.0,
        )


def test_oracle_and_imm_return_finite_complete_traces() -> None:
    tape = _tape()
    levels = UncertaintyLevels()
    modes = tuple(
        JumpFilterParameters(*levels.values(cell), 4.0)
        for cell in all_factorial_cells()
    )
    imm = run_imm_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        modes=modes,
        mode_switch_probability=1 / 64,
    )
    oracle = run_oracle_filter(
        tape.observations,
        sequence_ids=tape.sequence_ids,
        hazard=tape.hazard,
        process_variance=tape.process_variance,
        observation_variance=tape.observation_variance,
        jump_variance=4.0,
    )
    assert len(imm.predictive_nll) == len(tape.observations)
    assert np.all(np.isfinite(imm.predictive_nll))
    assert np.all(np.isfinite(oracle.predictive_nll))
    assert np.mean(oracle.predictive_nll) < 10.0


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="hazard"):
        JumpFilterParameters(0.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="adaptation"):
        AdaptationRates(1.1, 0.1, 0.1)
