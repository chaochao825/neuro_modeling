from __future__ import annotations

import numpy as np
import pytest

from src.models.soft_memory_controller import (
    ControllerStandardizer,
    SoftMemoryConfig,
    accumulate_with_retention,
    causal_control_features,
    controller_retention,
    soft_memory_accumulator,
)


def _config(**overrides: float) -> SoftMemoryConfig:
    values = {
        "retention_floor": 0.1,
        "retention_ceiling": 0.99,
        "bias": -0.5,
        "surprise_weight": 1.0,
        "entropy_weight": 0.5,
        "disagreement_weight": 2.0,
        "fast_retention": 0.25,
        "slow_retention": 0.95,
        "evidence_weight_floor": 1.0,
    }
    values.update(overrides)
    return SoftMemoryConfig(**values)


def _switch_tape() -> tuple[np.ndarray, np.ndarray]:
    first = np.tile([0.92, 0.08], (8, 1))
    second = np.tile([0.05, 0.95], (8, 1))
    evidence = np.vstack([first, second])
    streams = np.repeat("stream", len(evidence))
    return evidence, streams


def test_causal_features_respond_to_switch_without_labels() -> None:
    evidence, streams = _switch_tape()
    trace = causal_control_features(
        evidence,
        stream_ids=streams,
        fast_retention=0.25,
        slow_retention=0.95,
    )
    assert trace.raw_features.shape == (16, 3)
    assert trace.surprise[0] == 0.0
    assert trace.surprise[8] > np.max(trace.surprise[2:8])
    assert trace.disagreement[8] > np.max(trace.disagreement[2:8])
    assert np.allclose(trace.fast_probability.sum(axis=1), 1.0)


def test_generative_surprise_uses_supplied_log_likelihood() -> None:
    evidence, streams = _switch_tape()
    log_likelihood = np.log(np.maximum(evidence, 1e-12))
    trace = causal_control_features(
        evidence,
        stream_ids=streams,
        fast_retention=0.25,
        slow_retention=0.95,
        observation_log_likelihood=log_likelihood,
    )
    assert trace.surprise[8] > trace.surprise[7]


def test_controller_retention_is_continuous_and_monotone() -> None:
    config = _config(
        bias=0.0,
        surprise_weight=1.0,
        entropy_weight=0.0,
        disagreement_weight=0.0,
    )
    features = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    retention, risk = controller_retention(features, config=config)
    assert np.all(np.diff(risk) > 0.0)
    assert np.all(np.diff(retention) < 0.0)
    assert np.all((retention > config.retention_floor) & (retention < config.retention_ceiling))


def test_hard_controller_preserves_same_control_score_boundary() -> None:
    config = _config(bias=0.0)
    features = np.zeros((2, 3))
    retention, risk = controller_retention(features, config=config, hard=True)
    assert np.array_equal(risk, np.ones(2))
    assert np.allclose(retention, config.retention_floor)


def test_soft_accumulator_is_causal_under_future_perturbation() -> None:
    evidence, streams = _switch_tape()
    standardizer = ControllerStandardizer.fit(
        causal_control_features(
            evidence,
            stream_ids=streams,
            fast_retention=0.25,
            slow_retention=0.95,
        ).raw_features
    )
    first = soft_memory_accumulator(
        evidence,
        stream_ids=streams,
        config=_config(),
        standardizer=standardizer,
    )
    perturbed = evidence.copy()
    perturbed[12:] = perturbed[12:, ::-1]
    second = soft_memory_accumulator(
        perturbed,
        stream_ids=streams,
        config=_config(),
        standardizer=standardizer,
    )
    assert np.allclose(first.probabilities[:12], second.probabilities[:12])
    assert np.allclose(first.retention[:12], second.retention[:12])


def test_accumulation_resets_only_at_observable_stream_boundary() -> None:
    evidence = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    retention = np.ones(3)
    predictions, probabilities, state_l1 = accumulate_with_retention(
        evidence,
        stream_ids=np.array(["a", "a", "b"]),
        retention_value=retention,
    )
    assert np.array_equal(predictions, [0, 0, 1])
    assert np.allclose(probabilities[2], evidence[2])
    assert np.allclose(state_l1, [1.0, 2.0, 1.0])


def test_entropy_weight_is_explicit_and_bounded() -> None:
    evidence = np.array([[0.5, 0.5], [0.99, 0.01]])
    streams = np.repeat("a", 2)
    raw = causal_control_features(
        evidence,
        stream_ids=streams,
        fast_retention=0.25,
        slow_retention=0.95,
    ).raw_features
    trace = soft_memory_accumulator(
        evidence,
        stream_ids=streams,
        config=_config(evidence_weight_floor=0.2),
        standardizer=ControllerStandardizer.fit(np.vstack([raw, raw + 1e-3])),
    )
    assert trace.evidence_weight[0] == pytest.approx(0.2)
    assert trace.evidence_weight[1] > trace.evidence_weight[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"retention_floor": 0.99},
        {"fast_retention": 0.95},
        {"evidence_weight_floor": -0.1},
        {"bias": np.nan},
    ],
)
def test_invalid_controller_configs_fail_closed(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def test_standardizer_rejects_degenerate_schema_but_stabilizes_constant_axis() -> None:
    with pytest.raises(ValueError):
        ControllerStandardizer.fit(np.ones((3, 2)))
    standardizer = ControllerStandardizer.fit(np.ones((3, 3)))
    assert np.allclose(standardizer.scale, 1e-8)
    assert np.allclose(standardizer.transform(np.ones((1, 3))), 0.0)

