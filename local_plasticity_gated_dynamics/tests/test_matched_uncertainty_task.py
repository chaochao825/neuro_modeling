from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tasks.matched_uncertainty import (
    MATCHED_QR_REGIMES,
    MatchedQRRegime,
    MatchedUncertaintyConfig,
    MatchedUncertaintyTape,
    generate_matched_uncertainty_tape,
    matched_qr_pairs,
)


def test_registered_pairs_match_q_plus_two_r_and_have_zero_hazard() -> None:
    expected = ((0.04, 0.01, 0.06), (0.0025, 0.02875, 0.06))
    observed = tuple(
        (
            regime.process_variance,
            regime.observation_variance,
            regime.increment_variance,
        )
        for regime in matched_qr_pairs()[0]
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)

    for first, second in matched_qr_pairs():
        assert first.process_variance != second.process_variance
        assert first.observation_variance != second.observation_variance
        assert first.increment_variance == pytest.approx(
            second.increment_variance, abs=1e-15
        )

    tape = generate_matched_uncertainty_tape(
        seed=41,
        split="zero-hazard",
        config=MatchedUncertaintyConfig(
            block_length=8, blocks_per_sequence=4, n_sequences=2
        ),
    )
    np.testing.assert_array_equal(tape.hazard, 0.0)


def test_matched_tape_is_deterministic_balanced_and_digest_bound() -> None:
    config = MatchedUncertaintyConfig(
        block_length=12, blocks_per_sequence=8, n_sequences=3
    )
    first = generate_matched_uncertainty_tape(seed=410, split="fit", config=config)
    replay = generate_matched_uncertainty_tape(seed=410, split="fit", config=config)
    heldout = generate_matched_uncertainty_tape(seed=410, split="test", config=config)

    assert first.digest == replay.digest
    assert first.digest != heldout.digest
    np.testing.assert_array_equal(first.observations, replay.observations)
    assert not np.array_equal(first.observations, heldout.observations)
    assert len(first.digest) == 64

    expected_per_regime = config.block_length * (
        config.blocks_per_sequence // len(MATCHED_QR_REGIMES)
    )
    for sequence_id in range(config.n_sequences):
        sequence_regimes = first.regimes[first.sequence_ids == sequence_id]
        names, counts = np.unique(sequence_regimes, return_counts=True)
        assert set(names) == {value.name for value in MATCHED_QR_REGIMES}
        assert set(counts) == {expected_per_regime}


def test_method_inputs_exclude_evaluation_only_metadata_and_are_immutable() -> None:
    tape = generate_matched_uncertainty_tape(
        seed=2,
        split="api",
        config=MatchedUncertaintyConfig(
            block_length=4, blocks_per_sequence=4, n_sequences=1
        ),
    )
    observations, sequence_ids = tape.method_inputs()
    assert observations is tape.observations
    assert sequence_ids is tape.sequence_ids
    assert not observations.flags.writeable
    assert not sequence_ids.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        observations[0] = 0.0


def test_tape_defensively_copies_arrays() -> None:
    tape = generate_matched_uncertainty_tape(
        seed=3,
        split="copy",
        config=MatchedUncertaintyConfig(
            block_length=4, blocks_per_sequence=4, n_sequences=1
        ),
    )
    source = np.array(tape.observations, copy=True)
    copied = MatchedUncertaintyTape(
        observations=source,
        latent=tape.latent,
        hazard=tape.hazard,
        process_variance=tape.process_variance,
        observation_variance=tape.observation_variance,
        sequence_ids=tape.sequence_ids,
        block_ids=tape.block_ids,
        regimes=tape.regimes,
        split=tape.split,
        digest=tape.digest,
    )
    source[0] += 1.0
    assert copied.observations[0] != source[0]


def test_matched_generator_rejects_unbalanced_or_invalid_design() -> None:
    with pytest.raises(ValueError, match="divisible"):
        generate_matched_uncertainty_tape(
            seed=0,
            split="bad",
            config=MatchedUncertaintyConfig(blocks_per_sequence=5),
        )
    with pytest.raises(ValueError, match="positive"):
        MatchedQRRegime("bad", 0.0, 0.1)
    with pytest.raises(ValueError, match="unique"):
        generate_matched_uncertainty_tape(
            seed=0,
            split="bad",
            config=MatchedUncertaintyConfig(blocks_per_sequence=2),
            regimes=(
                MatchedQRRegime("duplicate", 0.1, 0.1),
                MatchedQRRegime("duplicate", 0.2, 0.2),
            ),
        )


def test_dataclass_inputs_are_canonicalized_and_malformed_tapes_fail_closed() -> None:
    regime = MatchedQRRegime("numeric", np.float32(0.04), np.float64(0.01))
    assert type(regime.process_variance) is float
    assert type(regime.observation_variance) is float
    with pytest.raises(TypeError, match="numeric scalar"):
        MatchedQRRegime("string", "0.04", 0.01)  # type: ignore[arg-type]

    config = MatchedUncertaintyConfig(
        block_length=4.0, blocks_per_sequence=np.int64(4), n_sequences=1.0
    )
    assert type(config.block_length) is int
    assert type(config.blocks_per_sequence) is int
    assert type(config.n_sequences) is int
    assert type(config.initial_state_variance) is float

    tape = generate_matched_uncertainty_tape(seed=3, split="strict", config=config)
    with pytest.raises(ValueError, match="one-dimensional"):
        replace(tape, observations=tape.observations[:, None])
    with pytest.raises(TypeError, match="integer dtype"):
        replace(tape, sequence_ids=tape.sequence_ids.astype(float))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(tape, digest="z" * 64)
