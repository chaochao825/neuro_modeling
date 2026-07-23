from __future__ import annotations

import numpy as np

from src.data.orbit_streaming import OrbitQueryObservation, OrbitSupportSet
from src.models.streaming_fewshot_actuators import (
    ACTUATOR_NAMES,
    CONTEXT_FEATURE_NAMES,
    PersonalizedStreamingActuators,
    StreamingActuatorConfig,
)


def _support() -> OrbitSupportSet:
    return OrbitSupportSet(
        embeddings=np.asarray(
            [
                [1.0, 0.0, 0.1, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.1],
                [0.1, 0.9, 0.0, 0.0],
            ]
        ),
        labels=np.asarray([0, 0, 1, 1]),
        video_ids=np.asarray(["s0", "s0", "s1", "s1"]),
        frame_indices=np.asarray([0, 1, 0, 1]),
    )


def _query() -> OrbitQueryObservation:
    return OrbitQueryObservation(
        embeddings=np.asarray(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
            ]
        ),
        video_ids=np.asarray(["v0", "v0", "v1", "v1"]),
        frame_indices=np.asarray([2, 5, 1, 4]),
    )


def test_actuators_are_support_fitted_and_finite() -> None:
    fitted = PersonalizedStreamingActuators.fit(
        _support(), n_classes=2, config=StreamingActuatorConfig(delta_passes=2)
    )
    trace = fitted.trace(_query())
    assert trace.scores.shape == (4, len(ACTUATOR_NAMES), 2)
    assert trace.contexts.shape == (4, len(CONTEXT_FEATURE_NAMES))
    assert np.isfinite(trace.scores).all()
    assert fitted.write_l1_cost > 0.0
    assert fitted.write_l2_cost > 0.0
    assert not trace.predictions.flags.writeable

    changed_query = OrbitQueryObservation(
        embeddings=np.flip(_query().embeddings, axis=1),
        video_ids=_query().video_ids,
        frame_indices=_query().frame_indices,
    )
    fitted.trace(changed_query)
    # Query data cannot mutate or refit any support-derived quantity.
    assert (
        fitted.fit_fingerprint
        == PersonalizedStreamingActuators.fit(_support(), n_classes=2).fit_fingerprint
    )


def test_temporal_state_resets_at_video_boundary() -> None:
    config = StreamingActuatorConfig(temporal_retention=0.8)
    fitted = PersonalizedStreamingActuators.fit(_support(), n_classes=2, config=config)
    query = OrbitQueryObservation(
        embeddings=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]] * 2),
        video_ids=np.asarray(["v0", "v0", "v1", "v1"]),
        frame_indices=np.asarray([0, 1, 0, 1]),
    )
    trace = fitted.trace(query)
    temporal = ACTUATOR_NAMES.index("temporal")
    assert np.allclose(trace.scores[0, temporal], trace.scores[2, temporal])
    video_start = CONTEXT_FEATURE_NAMES.index("video_start")
    assert np.array_equal(trace.contexts[:, video_start], [1.0, 0.0, 1.0, 0.0])


def test_gain_is_support_only() -> None:
    fitted = PersonalizedStreamingActuators.fit(_support(), n_classes=2)
    assert fitted.feature_gain.shape == (_support().embeddings.shape[1],)
    assert np.all(fitted.feature_gain >= fitted.config.gain_min)
    assert np.all(fitted.feature_gain <= fitted.config.gain_max)
