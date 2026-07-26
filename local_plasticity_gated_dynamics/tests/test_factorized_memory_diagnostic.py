from __future__ import annotations

import numpy as np
import pytest

from src.analysis.factorized_memory_diagnostic import (
    direct_alpha_filter,
    fit_oracle_write_probe,
    grouped_binary_metrics,
    likelihood_hmm_filter,
    oracle_write_targets,
    source_video_belief_metrics,
    summarize_video_metrics,
)


def _evidence() -> np.ndarray:
    return np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )


def test_direct_alpha_one_tracks_current_evidence_and_reset_is_exact() -> None:
    evidence = _evidence()
    streams = np.repeat("s", len(evidence))
    trace = direct_alpha_filter(evidence, stream_ids=streams, alpha=1.0)
    assert np.allclose(trace.probabilities, evidence)
    slow = direct_alpha_filter(
        evidence,
        stream_ids=streams,
        alpha=0.1,
        reset_flags=np.array([False, False, True, False]),
    )
    assert np.allclose(slow.probabilities[2], evidence[2])
    assert not slow.probabilities.flags.writeable


def test_likelihood_hmm_accumulates_and_true_reset_clears_old_class() -> None:
    evidence = _evidence()
    log_likelihood = np.log(evidence)
    streams = np.repeat("s", len(evidence))
    cumulative = likelihood_hmm_filter(
        log_likelihood, stream_ids=streams, hazard=0.0, temperature=1.0
    )
    reset = likelihood_hmm_filter(
        log_likelihood,
        stream_ids=streams,
        hazard=0.0,
        temperature=1.0,
        reset_flags=np.array([False, False, True, False]),
    )
    assert cumulative.probabilities[1, 0] > evidence[1, 0]
    assert reset.predictions[2] == 1
    assert cumulative.predictions[2] == 0


def test_oracle_write_targets_include_memory_mass_without_changing_tape() -> None:
    evidence = _evidence()
    labels = np.array([0, 0, 1, 1])
    trace = oracle_write_targets(
        evidence, labels, stream_ids=np.repeat("s", 4), retention=0.5
    )
    assert trace.targets[0]
    assert trace.targets[2]
    assert np.all(np.diff(trace.log_memory_mass) > 0.0)
    assert np.all(np.isfinite(trace.keep_nll))


def test_video_metrics_keep_source_video_as_unit() -> None:
    evidence = _evidence()
    frame = source_video_belief_metrics(
        evidence,
        np.array([0, 0, 1, 1]),
        source_video_ids=np.array(["v0", "v0", "v1", "v1"]),
        switch_flags=np.array([False, False, True, False]),
        post_switch_window=2,
    )
    summary = summarize_video_metrics(frame)
    assert summary["n_videos"] == 2
    assert summary["n_frames"] == 4
    assert summary["video_equal_accuracy"] == pytest.approx(1.0)
    assert np.isfinite(summary["video_equal_post_switch_nll"])


def test_oracle_write_probe_and_grouped_metrics_are_deterministic() -> None:
    features = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.1],
            [1.0, 0.9],
            [2.0, 1.0],
            [-1.5, 0.2],
            [1.5, 0.8],
        ]
    )
    targets = np.array([False, False, True, True, False, True])
    first = fit_oracle_write_probe(features, targets, seed=7)
    second = fit_oracle_write_probe(features, targets, seed=7)
    scores = first.predict_proba(features)[:, 1]
    assert np.allclose(scores, second.predict_proba(features)[:, 1])
    metrics = grouped_binary_metrics(
        targets,
        scores,
        group_ids=np.array(["a", "a", "a", "b", "b", "b"]),
    )
    assert metrics["n_groups"] == 2
    assert metrics["n_auc_groups"] == 2
    assert metrics["video_equal_auc"] > 0.5


def test_diagnostic_rejects_noncausal_or_invalid_parameters() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError):
        direct_alpha_filter(evidence, stream_ids=np.repeat("s", 4), alpha=1.1)
    with pytest.raises(ValueError):
        likelihood_hmm_filter(
            np.log(evidence),
            stream_ids=np.repeat("s", 4),
            hazard=1.0,
            temperature=1.0,
        )
