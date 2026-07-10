import numpy as np
import pytest

import src.analysis.lead_lag as lead_lag
from src.analysis.lead_lag import causal_within_block_bias, switch_latency_summary


def test_switch_latency_reports_positive_latent_lead() -> None:
    blocks = np.repeat(np.arange(3), 6)
    latent = np.r_[np.zeros(6), [0.7, 1, 1, 1, 1, 1], [0.3, 0, 0, 0, 0, 0]]
    behavior = np.r_[np.zeros(6), [0, 0.2, 0.7, 1, 1, 1], [1, 0.8, 0.3, 0, 0, 0]]
    summary = switch_latency_summary(latent, behavior, blocks, reference_trials=3)
    assert summary.n_switches == 2
    assert summary.median_latent_lead_trials > 0


def test_causal_bias_resets_at_block_boundary() -> None:
    bias = causal_within_block_bias(np.array([1, -1, -1, -1]), np.array([0, 0, 1, 1]))
    assert np.allclose(bias, [1.0, 0.0, -1.0, -1.0])


def test_latency_endpoint_uses_late_block_reference_not_search_prefix() -> None:
    blocks = np.repeat([0, 1], 6)
    latent = np.r_[np.zeros(6), [0.1, 0.2, 0.3, 0.8, 1.0, 1.0]]
    behavior = np.r_[np.zeros(6), [0.0, 0.0, 0.2, 0.4, 0.9, 1.0]]
    summary = switch_latency_summary(latent, behavior, blocks, reference_trials=2)
    np.testing.assert_array_equal(summary.latent_latencies, [3.0])
    np.testing.assert_array_equal(summary.behavior_latencies, [4.0])


def test_latency_skips_block_when_late_reference_would_cover_it() -> None:
    blocks = np.repeat([0, 1], 3)
    latent = np.r_[np.zeros(3), [0.2, 0.7, 1.0]]
    behavior = np.r_[np.zeros(3), [0.0, 0.5, 1.0]]
    summary = switch_latency_summary(latent, behavior, blocks, reference_trials=3)
    assert summary.n_switches == 0
    assert summary.latent_latencies.size == 0
    assert summary.behavior_latencies.size == 0
    assert np.isnan(summary.median_latent_lead_trials)


def test_unestimable_crossing_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossing_results = iter([float("nan"), 2.0])
    monkeypatch.setattr(
        lead_lag,
        "_crossing_latency",
        lambda *_args: next(crossing_results),
    )
    blocks = np.repeat([0, 1], 4)
    latent = np.r_[np.zeros(4), np.ones(4)]
    behavior = latent.copy()
    summary = switch_latency_summary(latent, behavior, blocks, reference_trials=2)
    assert summary.n_switches == 0
    assert summary.latent_lead_trials.size == 0
    assert np.isnan(summary.median_latent_lead_trials)
