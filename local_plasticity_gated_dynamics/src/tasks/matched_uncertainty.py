"""Matched-marginal random-walk streams for Q/R identifiability audits.

The four registered regimes form two pairs with identical one-step increment
variance, ``Q + 2 R``.  They therefore cannot be separated from the marginal
increment variance alone; the lag-one increment covariance is required.  The
generating hazard is exactly zero.  True regime labels and block boundaries are
metadata for evaluation only and are not arguments to the filtering API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from src.utils.reproducibility import make_rng


GENERATOR_VERSION = "exp41-matched-qr-tape-v1"


@dataclass(frozen=True)
class MatchedQRRegime:
    """One registered H=0 random-walk regime."""

    name: str
    process_variance: float
    observation_variance: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("regime name must be a non-empty string")
        for field in ("process_variance", "observation_variance"):
            raw = getattr(self, field)
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{field} must be a numeric scalar")
            value = float(raw)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)

    @property
    def increment_variance(self) -> float:
        """Marginal variance of ``y[t] - y[t-1]`` within a regime."""

        return self.process_variance + 2.0 * self.observation_variance


MATCHED_QR_REGIMES: tuple[MatchedQRRegime, ...] = (
    MatchedQRRegime("m06_q_dominant", 0.04, 0.01),
    MatchedQRRegime("m06_r_dominant", 0.0025, 0.02875),
    MatchedQRRegime("m12_q_dominant", 0.08, 0.02),
    MatchedQRRegime("m12_r_dominant", 0.01, 0.055),
)


def matched_qr_pairs() -> tuple[tuple[MatchedQRRegime, MatchedQRRegime], ...]:
    """Return the two preregistered equal-``Q + 2R`` regime pairs."""

    return (
        (MATCHED_QR_REGIMES[0], MATCHED_QR_REGIMES[1]),
        (MATCHED_QR_REGIMES[2], MATCHED_QR_REGIMES[3]),
    )


@dataclass(frozen=True)
class MatchedUncertaintyConfig:
    """Balanced block design; filtering is grouped only at sequence level."""

    block_length: int = 256
    blocks_per_sequence: int = 8
    n_sequences: int = 4
    initial_state_variance: float = 4.0

    def __post_init__(self) -> None:
        for field in ("block_length", "blocks_per_sequence", "n_sequences"):
            value = getattr(self, field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{field} must be a positive integer")
            if int(value) != value or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
            object.__setattr__(self, field, int(value))
        raw_variance = self.initial_state_variance
        if isinstance(raw_variance, (bool, np.bool_)) or not isinstance(
            raw_variance, (int, float, np.integer, np.floating)
        ):
            raise TypeError("initial_state_variance must be numeric")
        variance = float(raw_variance)
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("initial_state_variance must be positive")
        object.__setattr__(self, "initial_state_variance", variance)


@dataclass(frozen=True)
class MatchedUncertaintyTape:
    """Immutable observations plus evaluation-only generating metadata."""

    observations: np.ndarray
    latent: np.ndarray
    hazard: np.ndarray
    process_variance: np.ndarray
    observation_variance: np.ndarray
    sequence_ids: np.ndarray
    block_ids: np.ndarray
    regimes: np.ndarray
    split: str
    digest: str

    def __post_init__(self) -> None:
        raw_sequence_ids = np.asarray(self.sequence_ids)
        raw_block_ids = np.asarray(self.block_ids)
        if not np.issubdtype(raw_sequence_ids.dtype, np.integer) or np.issubdtype(
            raw_sequence_ids.dtype, np.bool_
        ):
            raise TypeError("sequence_ids must have an integer dtype")
        if not np.issubdtype(raw_block_ids.dtype, np.integer) or np.issubdtype(
            raw_block_ids.dtype, np.bool_
        ):
            raise TypeError("block_ids must have an integer dtype")
        arrays = (
            np.array(self.observations, dtype=np.float64, copy=True),
            np.array(self.latent, dtype=np.float64, copy=True),
            np.array(self.hazard, dtype=np.float64, copy=True),
            np.array(self.process_variance, dtype=np.float64, copy=True),
            np.array(self.observation_variance, dtype=np.float64, copy=True),
            np.array(raw_sequence_ids, dtype=np.int64, copy=True),
            np.array(raw_block_ids, dtype=np.int64, copy=True),
            np.array(self.regimes, dtype=str, copy=True),
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("tape arrays must be one-dimensional")
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("tape arrays must have one shared non-zero length")
        numeric = arrays[:7]
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("numeric tape arrays must be finite")
        if not np.all(arrays[2] == 0.0):
            raise ValueError("the matched-Q/R audit requires H=0")
        if np.any(arrays[3] <= 0.0) or np.any(arrays[4] <= 0.0):
            raise ValueError("Q/R metadata must be positive")
        if np.any(arrays[5] < 0) or np.any(arrays[6] < 0):
            raise ValueError("sequence_ids and block_ids must be non-negative")
        if np.any(arrays[5][1:] < arrays[5][:-1]):
            raise ValueError("sequence_ids must be ordered")
        if np.any(arrays[6][1:] < arrays[6][:-1]):
            raise ValueError("block_ids must be ordered")
        if np.any(np.char.str_len(arrays[7]) == 0):
            raise ValueError("regime labels must be non-empty")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("digest must be a SHA-256 hex digest")
        field_names = (
            "observations",
            "latent",
            "hazard",
            "process_variance",
            "observation_variance",
            "sequence_ids",
            "block_ids",
            "regimes",
        )
        for field, value in zip(field_names, arrays, strict=True):
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    def method_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the complete information available to causal methods.

        In particular, this excludes true Q/R values, regime labels, and block
        boundaries.  Sequence identifiers only mark independent sequences.
        """

        return self.observations, self.sequence_ids


def _validate_regimes(
    regimes: Sequence[MatchedQRRegime], *, blocks_per_sequence: int
) -> tuple[MatchedQRRegime, ...]:
    selected = tuple(regimes)
    if not selected:
        raise ValueError("at least one regime is required")
    names = tuple(value.name for value in selected)
    if len(set(names)) != len(names):
        raise ValueError("regime names must be unique")
    if blocks_per_sequence % len(selected):
        raise ValueError("blocks_per_sequence must be divisible by regime count")
    return selected


def _balanced_order(
    rng: np.random.Generator,
    regimes: tuple[MatchedQRRegime, ...],
    repeats: int,
) -> tuple[MatchedQRRegime, ...]:
    result: list[MatchedQRRegime] = []
    previous: str | None = None
    for _ in range(repeats):
        permutation = rng.permutation(len(regimes)).tolist()
        if (
            previous is not None
            and len(permutation) > 1
            and regimes[permutation[0]].name == previous
        ):
            replacement = next(
                index
                for index, regime_index in enumerate(permutation)
                if regimes[regime_index].name != previous
            )
            permutation[0], permutation[replacement] = (
                permutation[replacement],
                permutation[0],
            )
        batch = tuple(regimes[index] for index in permutation)
        result.extend(batch)
        previous = batch[-1].name
    return tuple(result)


def _digest(
    *arrays: np.ndarray,
    seed: int,
    split: str,
    regimes: tuple[MatchedQRRegime, ...],
    config: MatchedUncertaintyConfig,
) -> str:
    hasher = hashlib.sha256()
    labels = (GENERATOR_VERSION, seed, split, regimes, config)
    for label in labels:
        encoded = repr(label).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    for value in arrays:
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def generate_matched_uncertainty_tape(
    *,
    seed: int,
    split: str,
    config: MatchedUncertaintyConfig = MatchedUncertaintyConfig(),
    regimes: Sequence[MatchedQRRegime] = MATCHED_QR_REGIMES,
) -> MatchedUncertaintyTape:
    """Generate a balanced H=0 tape using domain-separated RNG streams.

    Ordering, initial state, process innovations, and observation noise use
    independently derived RNG domains.  Changing scheduling code therefore
    cannot silently consume innovations from another stochastic mechanism.
    """

    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    selected = _validate_regimes(
        regimes, blocks_per_sequence=config.blocks_per_sequence
    )
    rng_labels = (GENERATOR_VERSION, split)
    order_rng = make_rng(seed, *rng_labels, "block-order")
    initial_rng = make_rng(seed, *rng_labels, "initial-state")
    process_rng = make_rng(seed, *rng_labels, "process-innovation")
    observation_rng = make_rng(seed, *rng_labels, "observation-noise")

    total = config.n_sequences * config.blocks_per_sequence * config.block_length
    observations = np.empty(total, dtype=np.float64)
    latent = np.empty(total, dtype=np.float64)
    hazard = np.zeros(total, dtype=np.float64)
    process = np.empty(total, dtype=np.float64)
    observation = np.empty(total, dtype=np.float64)
    sequence_ids = np.empty(total, dtype=np.int64)
    block_ids = np.empty(total, dtype=np.int64)
    max_name_length = max(len(regime.name) for regime in selected)
    regime_names = np.empty(total, dtype=f"U{max_name_length}")

    cursor = 0
    global_block = 0
    repeats = config.blocks_per_sequence // len(selected)
    for sequence_id in range(config.n_sequences):
        order = _balanced_order(order_rng, selected, repeats)
        state = float(initial_rng.normal(0.0, np.sqrt(config.initial_state_variance)))
        for regime in order:
            process_standard = process_rng.normal(size=config.block_length)
            observation_standard = observation_rng.normal(size=config.block_length)
            for within_block in range(config.block_length):
                state += float(
                    np.sqrt(regime.process_variance) * process_standard[within_block]
                )
                observed = state + float(
                    np.sqrt(regime.observation_variance)
                    * observation_standard[within_block]
                )
                observations[cursor] = observed
                latent[cursor] = state
                process[cursor] = regime.process_variance
                observation[cursor] = regime.observation_variance
                sequence_ids[cursor] = sequence_id
                block_ids[cursor] = global_block
                regime_names[cursor] = regime.name
                cursor += 1
            global_block += 1

    digest = _digest(
        observations,
        latent,
        hazard,
        process,
        observation,
        sequence_ids,
        block_ids,
        regime_names,
        seed=seed,
        split=split,
        regimes=selected,
        config=config,
    )
    return MatchedUncertaintyTape(
        observations=observations,
        latent=latent,
        hazard=hazard,
        process_variance=process,
        observation_variance=observation,
        sequence_ids=sequence_ids,
        block_ids=block_ids,
        regimes=regime_names,
        split=split,
        digest=digest,
    )


__all__ = [
    "GENERATOR_VERSION",
    "MATCHED_QR_REGIMES",
    "MatchedQRRegime",
    "MatchedUncertaintyConfig",
    "MatchedUncertaintyTape",
    "generate_matched_uncertainty_tape",
    "matched_qr_pairs",
]
