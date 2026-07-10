"""Trial-level context-switch latency without neuron-level pseudoreplication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwitchLatencySummary:
    latent_latencies: np.ndarray
    behavior_latencies: np.ndarray
    latent_lead_trials: np.ndarray
    median_latent_lead_trials: float
    n_switches: int


def _crossing_latency(values: np.ndarray, previous_level: float, new_level: float) -> float:
    midpoint = 0.5 * (previous_level + new_level)
    if new_level >= previous_level:
        hits = np.flatnonzero(values >= midpoint)
    else:
        hits = np.flatnonzero(values <= midpoint)
    return float(hits[0]) if hits.size else float("nan")


def switch_latency_summary(
    latent_context_score: np.ndarray,
    behavior_bias: np.ndarray,
    block_ids: np.ndarray,
    *,
    reference_trials: int = 5,
) -> SwitchLatencySummary:
    """Compare half-transition latencies at contiguous block switches.

    Positive `latent_lead_trials` means the latent score crossed before the
    behavior bias. Each switch is descriptive; inferential aggregation must be
    performed at the session/animal level.
    """

    latent = np.asarray(latent_context_score, dtype=float)
    behavior = np.asarray(behavior_bias, dtype=float)
    blocks = np.asarray(block_ids)
    if latent.ndim != 1 or behavior.shape != latent.shape or blocks.shape != latent.shape:
        raise ValueError("latent, behavior, and block_ids must be matching vectors")
    if not np.isfinite(latent).all() or not np.isfinite(behavior).all():
        raise ValueError("latent and behavior scores must be finite")
    if reference_trials < 1:
        raise ValueError("reference_trials must be positive")
    starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
    latent_latencies = []
    behavior_latencies = []
    for block_index in range(1, len(starts)):
        previous_start = starts[block_index - 1]
        current_start = starts[block_index]
        current_stop = starts[block_index + 1] if block_index + 1 < len(starts) else len(blocks)
        previous_slice = slice(max(previous_start, current_start - reference_trials), current_start)
        # Define the new-state endpoint from the *late* part of the block.  The
        # initial post-switch samples remain exclusively in the crossing
        # search, avoiding a self-referential threshold that forces early hits.
        # A block no longer than the requested reference window has no
        # independent early sample, so its switch latency is not estimable.
        if current_stop - current_start <= reference_trials:
            continue
        new_reference = slice(current_stop - reference_trials, current_stop)
        if previous_slice.stop - previous_slice.start < 1 or new_reference.stop - new_reference.start < 1:
            continue
        latent_previous = float(np.mean(latent[previous_slice]))
        latent_new = float(np.mean(latent[new_reference]))
        behavior_previous = float(np.mean(behavior[previous_slice]))
        behavior_new = float(np.mean(behavior[new_reference]))
        latent_latencies.append(
            _crossing_latency(latent[current_start:current_stop], latent_previous, latent_new)
        )
        behavior_latencies.append(
            _crossing_latency(
                behavior[current_start:current_stop], behavior_previous, behavior_new
            )
        )
    latent_array = np.asarray(latent_latencies, dtype=float)
    behavior_array = np.asarray(behavior_latencies, dtype=float)
    finite = np.isfinite(latent_array) & np.isfinite(behavior_array)
    latent_array = latent_array[finite]
    behavior_array = behavior_array[finite]
    lead = behavior_array - latent_array
    median = float(np.median(lead)) if lead.size else float("nan")
    return SwitchLatencySummary(
        latent_latencies=latent_array,
        behavior_latencies=behavior_array,
        latent_lead_trials=lead,
        median_latent_lead_trials=median,
        n_switches=int(len(lead)),
    )


def causal_within_block_bias(choice: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
    """Cumulative choice bias reset at every block boundary."""

    choices = np.asarray(choice, dtype=float)
    blocks = np.asarray(block_ids)
    if choices.ndim != 1 or blocks.shape != choices.shape or not np.isfinite(choices).all():
        raise ValueError("choice and block_ids must be matching finite vectors")
    result = np.empty_like(choices)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        result[indices] = np.cumsum(choices[indices]) / np.arange(1, len(indices) + 1)
    return result
