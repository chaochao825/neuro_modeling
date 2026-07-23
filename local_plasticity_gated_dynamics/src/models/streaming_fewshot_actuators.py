"""Reusable few-shot actuator motifs over a frozen visual embedding.

The four actions intentionally share the same support/query tape:

``prototype``
    Direct cosine evidence from class prototypes.
``gain``
    A support-only diagonal Fisher-style feature gain before prototyping.
``delta``
    A bounded class-by-feature associative matrix written with a delta rule.
``temporal``
    A causal leaky evidence state reset at every query video boundary.

No action receives query labels.  Every action returns class evidence, and a
controller may select one action per frame from causal diagnostics derived
from the prototype stream and support statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.data.orbit_streaming import OrbitQueryObservation, OrbitSupportSet


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ACTUATOR_NAMES: Final[tuple[str, ...]] = (
    "prototype",
    "gain",
    "delta",
    "temporal",
)
CONTEXT_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "bias",
    "prototype_max",
    "prototype_margin",
    "prototype_novelty",
    "embedding_change",
    "video_start",
    "elapsed_fraction",
    "support_dispersion",
    "log_support_per_class",
)


def _readonly(value: ArrayLike, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise TypeError(f"{name} must be a real numeric matrix")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != 2 or result.size == 0 or 0 in result.shape:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _normalise_rows(value: ArrayLike, *, floor: float) -> FloatArray:
    matrix = _finite_matrix(value, name="value")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, floor)


def _softmax(value: FloatArray) -> FloatArray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _round_robin_class_order(labels: IntArray, frame_indices: IntArray) -> IntArray:
    queues: list[list[int]] = []
    for label in sorted(np.unique(labels).tolist()):
        positions = np.flatnonzero(labels == label)
        positions = positions[np.argsort(frame_indices[positions], kind="stable")]
        queues.append(positions.tolist())
    order: list[int] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return np.asarray(order, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class StreamingActuatorConfig:
    norm_floor: float = 1e-8
    variance_floor: float = 1e-4
    gain_min: float = 0.25
    gain_max: float = 4.0
    delta_beta: float = 0.5
    delta_passes: int = 2
    temporal_retention: float = 0.85
    score_temperature: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "norm_floor",
            "variance_floor",
            "gain_min",
            "gain_max",
            "delta_beta",
            "temporal_retention",
            "score_temperature",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.norm_floor <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("normalisation floors must be positive")
        if not 0.0 < self.gain_min <= self.gain_max:
            raise ValueError("gain bounds must satisfy 0 < min <= max")
        if not 0.0 < self.delta_beta <= 1.0:
            raise ValueError("delta_beta must lie in (0, 1]")
        if (
            isinstance(self.delta_passes, bool)
            or not isinstance(self.delta_passes, (int, np.integer))
            or int(self.delta_passes) < 1
        ):
            raise ValueError("delta_passes must be a positive integer")
        object.__setattr__(self, "delta_passes", int(self.delta_passes))
        if not 0.0 <= self.temporal_retention < 1.0:
            raise ValueError("temporal_retention must lie in [0, 1)")
        if self.score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")


@dataclass(frozen=True, slots=True, eq=False)
class StreamingActuatorTrace:
    scores: FloatArray
    predictions: IntArray
    contexts: FloatArray
    action_event_l1: FloatArray
    temporal_state_l2: FloatArray
    video_ids: NDArray[np.str_]
    frame_indices: IntArray
    fingerprint: str

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64)
        predictions = np.asarray(self.predictions, dtype=np.int64)
        contexts = np.asarray(self.contexts, dtype=np.float64)
        costs = np.asarray(self.action_event_l1, dtype=np.float64)
        state = np.asarray(self.temporal_state_l2, dtype=np.float64)
        video_ids = np.asarray(self.video_ids, dtype=str)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        if scores.ndim != 3 or scores.shape[1] != len(ACTUATOR_NAMES):
            raise ValueError("scores must have shape [frame, action, class]")
        n_frames = scores.shape[0]
        if predictions.shape != scores.shape[:2]:
            raise ValueError("predictions must have shape [frame, action]")
        if contexts.shape != (n_frames, len(CONTEXT_FEATURE_NAMES)):
            raise ValueError("contexts have the wrong feature dimension")
        if costs.shape != (n_frames, len(ACTUATOR_NAMES)):
            raise ValueError("action_event_l1 has the wrong shape")
        if state.shape != (n_frames,) or video_ids.shape != (n_frames,):
            raise ValueError("trace metadata has the wrong frame dimension")
        if frame_indices.shape != (n_frames,):
            raise ValueError("trace frame_indices have the wrong shape")
        if not all(
            np.all(np.isfinite(value)) for value in (scores, contexts, costs, state)
        ):
            raise ValueError("trace arrays must be finite")
        if np.any(costs < 0.0) or np.any(state < 0.0):
            raise ValueError("trace costs and norms must be non-negative")
        object.__setattr__(self, "scores", _readonly(scores))
        object.__setattr__(self, "predictions", _readonly(predictions, dtype=np.int64))
        object.__setattr__(self, "contexts", _readonly(contexts))
        object.__setattr__(self, "action_event_l1", _readonly(costs))
        object.__setattr__(self, "temporal_state_l2", _readonly(state))
        object.__setattr__(self, "video_ids", _readonly(video_ids, dtype=str))
        object.__setattr__(
            self, "frame_indices", _readonly(frame_indices, dtype=np.int64)
        )


@dataclass(frozen=True, slots=True, eq=False)
class PersonalizedStreamingActuators:
    config: StreamingActuatorConfig
    n_classes: int
    prototypes: FloatArray
    feature_gain: FloatArray
    gain_prototypes: FloatArray
    delta_memory: FloatArray
    support_dispersion: float
    support_per_class: FloatArray
    write_l1_cost: float
    write_l2_cost: float
    fit_fingerprint: str

    @classmethod
    def fit(
        cls,
        support: OrbitSupportSet,
        *,
        n_classes: int,
        config: StreamingActuatorConfig | None = None,
    ) -> "PersonalizedStreamingActuators":
        cfg = config or StreamingActuatorConfig()
        if (
            isinstance(n_classes, bool)
            or not isinstance(n_classes, (int, np.integer))
            or int(n_classes) < 2
        ):
            raise ValueError("n_classes must be an integer >= 2")
        n_classes = int(n_classes)
        labels = np.asarray(support.labels, dtype=np.int64)
        if np.any(labels >= n_classes):
            raise ValueError("support labels exceed n_classes")
        expected = set(range(n_classes))
        if set(np.unique(labels).tolist()) != expected:
            raise ValueError("every class must occur in support")
        keys = _normalise_rows(support.embeddings, floor=cfg.norm_floor)
        feature_dim = keys.shape[1]
        prototypes = np.empty((n_classes, feature_dim), dtype=np.float64)
        counts = np.empty(n_classes, dtype=np.float64)
        within = np.zeros(feature_dim, dtype=np.float64)
        for label in range(n_classes):
            class_keys = keys[labels == label]
            counts[label] = class_keys.shape[0]
            prototypes[label] = np.mean(class_keys, axis=0)
            within += np.sum((class_keys - prototypes[label][None, :]) ** 2, axis=0)
        prototypes = _normalise_rows(prototypes, floor=cfg.norm_floor)
        within /= max(1, keys.shape[0] - n_classes)
        overall = np.average(prototypes, axis=0, weights=counts)
        between = np.average(
            (prototypes - overall[None, :]) ** 2,
            axis=0,
            weights=counts,
        )
        gain = np.sqrt((between + cfg.variance_floor) / (within + cfg.variance_floor))
        gain = np.clip(gain, cfg.gain_min, cfg.gain_max)
        gain /= max(float(np.sqrt(np.mean(gain**2))), cfg.norm_floor)
        # RMS matching keeps the gain actuator on a comparable score scale;
        # the second projection makes the advertised hard bounds invariant to
        # that rescaling.
        gain = np.clip(gain, cfg.gain_min, cfg.gain_max)
        gained_keys = _normalise_rows(keys * gain[None, :], floor=cfg.norm_floor)
        gain_prototypes = np.empty_like(prototypes)
        for label in range(n_classes):
            gain_prototypes[label] = np.mean(gained_keys[labels == label], axis=0)
        gain_prototypes = _normalise_rows(gain_prototypes, floor=cfg.norm_floor)

        memory = np.zeros((n_classes, feature_dim), dtype=np.float64)
        write_l1 = 0.0
        write_l2_sq = 0.0
        order = _round_robin_class_order(labels, support.frame_indices)
        for _ in range(cfg.delta_passes):
            for index in order:
                key = keys[index]
                value = np.zeros(n_classes, dtype=np.float64)
                value[labels[index]] = 1.0
                residual = value - memory @ key
                update = cfg.delta_beta * residual[:, None] * key[None, :]
                memory += update
                write_l1 += float(np.sum(np.abs(update)))
                write_l2_sq += float(np.sum(update**2))
        memory = _normalise_rows(memory, floor=cfg.norm_floor)
        correct_similarity = np.sum(keys * prototypes[labels], axis=1)
        dispersion = float(np.mean(np.clip(1.0 - correct_similarity, 0.0, 2.0)))
        digest = hashlib.sha256(b"orbit-streaming-actuator-fit-v1")
        for value in (
            keys,
            labels,
            prototypes,
            gain,
            gain_prototypes,
            memory,
            counts,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return cls(
            config=cfg,
            n_classes=n_classes,
            prototypes=_readonly(prototypes),
            feature_gain=_readonly(gain),
            gain_prototypes=_readonly(gain_prototypes),
            delta_memory=_readonly(memory),
            support_dispersion=dispersion,
            support_per_class=_readonly(counts),
            write_l1_cost=write_l1,
            write_l2_cost=float(np.sqrt(write_l2_sq)),
            fit_fingerprint=digest.hexdigest(),
        )

    @property
    def feature_dim(self) -> int:
        return int(self.prototypes.shape[1])

    def trace(self, query: OrbitQueryObservation) -> StreamingActuatorTrace:
        if query.embeddings.shape[1] != self.feature_dim:
            raise ValueError("query feature dimension does not match fitted actuators")
        cfg = self.config
        keys = _normalise_rows(query.embeddings, floor=cfg.norm_floor)
        prototype_scores = keys @ self.prototypes.T
        gain_keys = _normalise_rows(
            keys * self.feature_gain[None, :], floor=cfg.norm_floor
        )
        gain_scores = gain_keys @ self.gain_prototypes.T
        delta_scores = keys @ self.delta_memory.T
        temporal_scores = np.empty_like(prototype_scores)
        temporal_norm = np.empty(keys.shape[0], dtype=np.float64)
        contexts = np.empty(
            (keys.shape[0], len(CONTEXT_FEATURE_NAMES)), dtype=np.float64
        )
        previous_key = np.zeros(self.feature_dim, dtype=np.float64)
        temporal_state = np.zeros(self.n_classes, dtype=np.float64)
        previous_video = ""
        within_video_index = 0
        video_lengths = {
            str(video_id): int(np.sum(query.video_ids == video_id))
            for video_id in np.unique(query.video_ids)
        }
        mean_support = float(np.mean(self.support_per_class))
        for index, video_id_raw in enumerate(query.video_ids):
            video_id = str(video_id_raw)
            video_start = video_id != previous_video
            if video_start:
                temporal_state.fill(0.0)
                previous_key.fill(0.0)
                within_video_index = 0
            current = prototype_scores[index]
            temporal_state = (
                cfg.temporal_retention * temporal_state
                + (1.0 - cfg.temporal_retention) * current
            )
            temporal_scores[index] = temporal_state
            temporal_norm[index] = float(np.linalg.norm(temporal_state))
            ordered = np.partition(current, -2)
            maximum = float(ordered[-1])
            margin = float(ordered[-1] - ordered[-2])
            change = (
                0.0
                if video_start
                else float(np.clip(1.0 - keys[index] @ previous_key, 0.0, 2.0))
            )
            denominator = max(1, video_lengths[video_id] - 1)
            contexts[index] = (
                1.0,
                maximum,
                margin,
                1.0 - maximum,
                change,
                float(video_start),
                within_video_index / denominator,
                self.support_dispersion,
                float(np.log1p(mean_support)),
            )
            previous_key = keys[index]
            previous_video = video_id
            within_video_index += 1
        scores = cfg.score_temperature * np.stack(
            (prototype_scores, gain_scores, delta_scores, temporal_scores), axis=1
        )
        predictions = np.argmax(scores, axis=-1).astype(np.int64)
        # Transparent arithmetic/event proxies.  They exclude the shared
        # encoder and charge the temporal action for both direct evidence and
        # state integration.
        base_cost = np.sum(np.abs(keys), axis=1)[:, None] * self.n_classes
        costs = np.concatenate(
            (
                base_cost,
                base_cost + np.sum(np.abs(gain_keys - keys), axis=1)[:, None],
                base_cost,
                base_cost + np.sum(np.abs(temporal_scores), axis=1)[:, None],
            ),
            axis=1,
        )
        digest = hashlib.sha256(self.fit_fingerprint.encode("ascii"))
        for value in (
            keys,
            query.video_ids,
            query.frame_indices,
            scores,
            contexts,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return StreamingActuatorTrace(
            scores=scores,
            predictions=predictions,
            contexts=contexts,
            action_event_l1=costs,
            temporal_state_l2=temporal_norm,
            video_ids=query.video_ids,
            frame_indices=query.frame_indices,
            fingerprint=digest.hexdigest(),
        )

    def probabilities(self, trace: StreamingActuatorTrace) -> FloatArray:
        """Return per-action class probabilities for calibration analyses."""

        if trace.scores.shape[2] != self.n_classes:
            raise ValueError("trace class dimension does not match fitted actuators")
        return _readonly(_softmax(trace.scores))
