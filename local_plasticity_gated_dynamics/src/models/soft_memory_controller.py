"""Causal low-dimensional controllers for continuously adaptive memory.

The controller never receives labels, future observations, or gradients at
deployment.  It maps three causal statistics -- predictive surprise,
observation entropy, and fast/slow disagreement -- to a scalar retention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from src.models.change_aware_prefix import jensen_shannon_divergence


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: ArrayLike, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _probability_rows(value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError("evidence must have shape [frame, class>=2]")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("evidence must be finite and non-negative")
    if not np.allclose(np.sum(result, axis=1), 1.0, atol=1e-6):
        raise ValueError("evidence rows must sum to one")
    return result


def _stream_vector(value: ArrayLike, *, n_frames: int) -> NDArray[np.str_]:
    streams = np.asarray(value, dtype=str)
    if streams.shape != (n_frames,) or np.any(np.char.str_len(streams) == 0):
        raise ValueError("stream_ids must align with frames and be non-empty")
    return streams


def _open_unit_interval(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must lie in [0, 1)")
    return result


def _closed_unit_interval(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ControllerStandardizer:
    """Train/development-fitted normalization for the three control inputs."""

    mean: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.shape != (3,) or scale.shape != (3,):
            raise ValueError("controller normalization must contain three features")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("controller normalization must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("controller scales must be positive")
        object.__setattr__(self, "mean", _readonly(mean, dtype=np.float64))
        object.__setattr__(self, "scale", _readonly(scale, dtype=np.float64))

    @classmethod
    def fit(cls, features_value: ArrayLike) -> ControllerStandardizer:
        """Fit only on a declared training/development feature tape."""

        features = np.asarray(features_value, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 3 or features.shape[0] < 2:
            raise ValueError("features must have shape [sample>=2, 3]")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must be finite")
        scale = np.std(features, axis=0, ddof=0)
        scale = np.maximum(scale, 1e-8)
        return cls(mean=np.mean(features, axis=0), scale=scale)

    def transform(self, features_value: ArrayLike) -> FloatArray:
        features = np.asarray(features_value, dtype=np.float64)
        if features.shape[-1:] != (3,) or not np.all(np.isfinite(features)):
            raise ValueError("features must be finite with final dimension three")
        return np.asarray((features - self.mean) / self.scale, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SoftMemoryConfig:
    """Frozen scalar controller and state parameters."""

    retention_floor: float
    retention_ceiling: float
    bias: float
    surprise_weight: float
    entropy_weight: float
    disagreement_weight: float
    fast_retention: float = 0.5
    slow_retention: float = 0.98
    evidence_weight_floor: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        floor = _closed_unit_interval(self.retention_floor, name="retention_floor")
        ceiling = _closed_unit_interval(
            self.retention_ceiling, name="retention_ceiling"
        )
        if floor >= ceiling:
            raise ValueError("retention_floor must be below retention_ceiling")
        fast = _open_unit_interval(self.fast_retention, name="fast_retention")
        slow = _open_unit_interval(self.slow_retention, name="slow_retention")
        if fast >= slow:
            raise ValueError("fast_retention must be below slow_retention")
        weight_floor = _closed_unit_interval(
            self.evidence_weight_floor, name="evidence_weight_floor"
        )
        scalar_names = (
            "bias",
            "surprise_weight",
            "entropy_weight",
            "disagreement_weight",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        epsilon = float(self.epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        object.__setattr__(self, "retention_floor", floor)
        object.__setattr__(self, "retention_ceiling", ceiling)
        object.__setattr__(self, "fast_retention", fast)
        object.__setattr__(self, "slow_retention", slow)
        object.__setattr__(self, "evidence_weight_floor", weight_floor)
        object.__setattr__(self, "epsilon", epsilon)

    @property
    def control_dimension(self) -> int:
        return 3


@dataclass(frozen=True, slots=True, eq=False)
class CausalFeatureTrace:
    """Label-free causal statistics used by every adaptive condition."""

    raw_features: FloatArray
    fast_probability: FloatArray
    slow_probability: FloatArray

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_features, dtype=np.float64)
        fast = np.asarray(self.fast_probability, dtype=np.float64)
        slow = np.asarray(self.slow_probability, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0:
            raise ValueError("raw_features must have shape [frame, 3]")
        if fast.shape != slow.shape or fast.shape[0] != raw.shape[0]:
            raise ValueError("fast and slow probabilities must align with features")
        if fast.ndim != 2 or fast.shape[1] < 2:
            raise ValueError("probability states must have at least two classes")
        for name, value in (("raw", raw), ("fast", fast), ("slow", slow)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} values must be finite")
        if np.any(fast < 0.0) or np.any(slow < 0.0):
            raise ValueError("probability states must be non-negative")
        if not np.allclose(fast.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("fast probability rows must sum to one")
        if not np.allclose(slow.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("slow probability rows must sum to one")
        object.__setattr__(self, "raw_features", _readonly(raw, dtype=np.float64))
        object.__setattr__(
            self, "fast_probability", _readonly(fast, dtype=np.float64)
        )
        object.__setattr__(
            self, "slow_probability", _readonly(slow, dtype=np.float64)
        )

    @property
    def surprise(self) -> FloatArray:
        return self.raw_features[:, 0]

    @property
    def entropy(self) -> FloatArray:
        return self.raw_features[:, 1]

    @property
    def disagreement(self) -> FloatArray:
        return self.raw_features[:, 2]


@dataclass(frozen=True, slots=True, eq=False)
class SoftMemoryTrace:
    """Framewise predictions, controller state, and auditable control signals."""

    predictions: IntArray
    probabilities: FloatArray
    retention: FloatArray
    evidence_weight: FloatArray
    change_risk: FloatArray
    standardized_features: FloatArray
    raw_features: FloatArray
    state_l1: FloatArray

    def __post_init__(self) -> None:
        predictions = np.asarray(self.predictions, dtype=np.int64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        retention = np.asarray(self.retention, dtype=np.float64)
        evidence_weight = np.asarray(self.evidence_weight, dtype=np.float64)
        change_risk = np.asarray(self.change_risk, dtype=np.float64)
        standardized = np.asarray(self.standardized_features, dtype=np.float64)
        raw = np.asarray(self.raw_features, dtype=np.float64)
        state_l1 = np.asarray(self.state_l1, dtype=np.float64)
        if predictions.ndim != 1 or predictions.size == 0:
            raise ValueError("predictions must be a non-empty vector")
        n_frames = predictions.size
        if probabilities.ndim != 2 or probabilities.shape[0] != n_frames:
            raise ValueError("probabilities must align with predictions")
        if standardized.shape != (n_frames, 3) or raw.shape != (n_frames, 3):
            raise ValueError("controller feature traces must have shape [frame, 3]")
        for name, value in (
            ("retention", retention),
            ("evidence_weight", evidence_weight),
            ("change_risk", change_risk),
            ("state_l1", state_l1),
        ):
            if value.shape != (n_frames,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite aligned vector")
        if not np.all(np.isfinite(probabilities)) or not np.all(
            np.isfinite(standardized)
        ):
            raise ValueError("trace arrays must be finite")
        if np.any((retention < 0.0) | (retention > 1.0)):
            raise ValueError("retention must lie in [0, 1]")
        if np.any((change_risk < 0.0) | (change_risk > 1.0)):
            raise ValueError("change risk must lie in [0, 1]")
        if np.any(evidence_weight <= 0.0) or np.any(state_l1 <= 0.0):
            raise ValueError("evidence weights and state norms must be positive")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probability rows must sum to one")
        if np.any(predictions < 0) or np.any(predictions >= probabilities.shape[1]):
            raise ValueError("predictions fall outside probability classes")
        for name, value, dtype in (
            ("predictions", predictions, np.int64),
            ("probabilities", probabilities, np.float64),
            ("retention", retention, np.float64),
            ("evidence_weight", evidence_weight, np.float64),
            ("change_risk", change_risk, np.float64),
            ("standardized_features", standardized, np.float64),
            ("raw_features", raw, np.float64),
            ("state_l1", state_l1, np.float64),
        ):
            object.__setattr__(self, name, _readonly(value, dtype=dtype))


def _normalize_state(state: FloatArray, *, epsilon: float) -> FloatArray:
    total = float(np.sum(state))
    if total <= epsilon:
        return np.full(state.size, 1.0 / state.size, dtype=np.float64)
    return np.asarray(state / total, dtype=np.float64)


def causal_control_features(
    evidence_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    fast_retention: float,
    slow_retention: float,
    observation_log_likelihood: ArrayLike | None = None,
    epsilon: float = 1e-12,
) -> CausalFeatureTrace:
    """Compute causal surprise, entropy, and fast/slow disagreement.

    When a class-conditional generative score is available, surprise is the
    negative log posterior-predictive mixture likelihood.  Otherwise it uses
    the calibrated class-evidence overlap.  Labels never enter either path.
    """

    evidence = _probability_rows(evidence_value)
    streams = _stream_vector(stream_ids, n_frames=evidence.shape[0])
    fast_keep = _open_unit_interval(fast_retention, name="fast_retention")
    slow_keep = _open_unit_interval(slow_retention, name="slow_retention")
    if fast_keep >= slow_keep:
        raise ValueError("fast_retention must be below slow_retention")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    log_likelihood: FloatArray | None = None
    if observation_log_likelihood is not None:
        log_likelihood = np.asarray(observation_log_likelihood, dtype=np.float64)
        if log_likelihood.shape != evidence.shape or not np.all(
            np.isfinite(log_likelihood)
        ):
            raise ValueError("observation_log_likelihood must match evidence")

    n_frames, n_classes = evidence.shape
    raw = np.empty((n_frames, 3), dtype=np.float64)
    fast_probabilities = np.empty_like(evidence)
    slow_probabilities = np.empty_like(evidence)
    fast_state = np.zeros(n_classes, dtype=np.float64)
    slow_state = np.zeros(n_classes, dtype=np.float64)
    previous_stream = ""
    run_length = 0
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    log_classes = np.log(float(n_classes))
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous_stream:
            fast_state.fill(0.0)
            slow_state.fill(0.0)
            run_length = 0
        previous_slow = (
            _normalize_state(slow_state, epsilon=epsilon)
            if run_length > 0
            else uniform
        )
        if run_length == 0:
            surprise = 0.0
        elif log_likelihood is None:
            surprise = -np.log(
                max(float(np.dot(previous_slow, evidence[index])), epsilon)
            )
        else:
            surprise = -float(
                logsumexp(np.log(np.maximum(previous_slow, epsilon)) + log_likelihood[index])
            )
        entropy = -float(
            np.sum(evidence[index] * np.log(np.maximum(evidence[index], epsilon)))
        ) / log_classes
        candidate_fast = fast_keep * fast_state + evidence[index]
        candidate_slow = slow_keep * slow_state + evidence[index]
        fast_probability = _normalize_state(candidate_fast, epsilon=epsilon)
        slow_probability = _normalize_state(candidate_slow, epsilon=epsilon)
        disagreement = jensen_shannon_divergence(
            fast_probability, slow_probability
        )
        raw[index] = (max(0.0, surprise), entropy, disagreement)
        fast_probabilities[index] = fast_probability
        slow_probabilities[index] = slow_probability
        fast_state = candidate_fast
        slow_state = candidate_slow
        run_length += 1
        previous_stream = stream
    return CausalFeatureTrace(
        raw_features=raw,
        fast_probability=fast_probabilities,
        slow_probability=slow_probabilities,
    )


def controller_retention(
    standardized_features_value: ArrayLike,
    *,
    config: SoftMemoryConfig,
    hard: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Map three standardized causal statistics to risk and retention."""

    features = np.asarray(standardized_features_value, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError("standardized_features must have shape [frame, 3]")
    if not np.all(np.isfinite(features)):
        raise ValueError("standardized_features must be finite")
    weights = np.array(
        [
            config.surprise_weight,
            config.entropy_weight,
            config.disagreement_weight,
        ],
        dtype=np.float64,
    )
    logits = np.clip(config.bias + features @ weights, -60.0, 60.0)
    risk = 1.0 / (1.0 + np.exp(-logits))
    if hard:
        risk = (risk >= 0.5).astype(np.float64)
    retention = config.retention_ceiling - (
        config.retention_ceiling - config.retention_floor
    ) * risk
    return (
        np.asarray(retention, dtype=np.float64),
        np.asarray(risk, dtype=np.float64),
    )


def accumulate_with_retention(
    evidence_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    retention_value: ArrayLike,
    evidence_weight_value: ArrayLike | None = None,
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Accumulate one evidence tape with a supplied causal retention tape."""

    evidence = _probability_rows(evidence_value)
    streams = _stream_vector(stream_ids, n_frames=evidence.shape[0])
    retention = np.asarray(retention_value, dtype=np.float64)
    if retention.shape != (evidence.shape[0],) or not np.all(
        np.isfinite(retention)
    ):
        raise ValueError("retention must be a finite frame vector")
    if np.any((retention < 0.0) | (retention > 1.0)):
        raise ValueError("retention must lie in [0, 1]")
    if evidence_weight_value is None:
        weights = np.ones(evidence.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(evidence_weight_value, dtype=np.float64)
        if weights.shape != (evidence.shape[0],) or not np.all(np.isfinite(weights)):
            raise ValueError("evidence weights must be a finite frame vector")
        if np.any(weights <= 0.0):
            raise ValueError("evidence weights must be positive")

    probabilities = np.empty_like(evidence)
    state_l1 = np.empty(evidence.shape[0], dtype=np.float64)
    state = np.zeros(evidence.shape[1], dtype=np.float64)
    previous_stream = ""
    for index, stream_raw in enumerate(streams):
        stream = str(stream_raw)
        if stream != previous_stream:
            state.fill(0.0)
        state = retention[index] * state + weights[index] * evidence[index]
        total = float(np.sum(state))
        if total <= 0.0:
            raise RuntimeError("memory state lost all mass")
        probabilities[index] = state / total
        state_l1[index] = total
        previous_stream = stream
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    return predictions, probabilities, state_l1


def soft_memory_accumulator(
    evidence_value: ArrayLike,
    *,
    stream_ids: ArrayLike,
    config: SoftMemoryConfig,
    standardizer: ControllerStandardizer,
    observation_log_likelihood: ArrayLike | None = None,
    hard: bool = False,
    retention_override: ArrayLike | None = None,
) -> SoftMemoryTrace:
    """Run the continuous controller or a matched hard/override condition."""

    evidence = _probability_rows(evidence_value)
    feature_trace = causal_control_features(
        evidence,
        stream_ids=stream_ids,
        fast_retention=config.fast_retention,
        slow_retention=config.slow_retention,
        observation_log_likelihood=observation_log_likelihood,
        epsilon=config.epsilon,
    )
    standardized = standardizer.transform(feature_trace.raw_features)
    retention, risk = controller_retention(
        standardized, config=config, hard=hard
    )
    if retention_override is not None:
        override = np.asarray(retention_override, dtype=np.float64)
        if override.shape != retention.shape or not np.all(np.isfinite(override)):
            raise ValueError("retention_override must be a finite frame vector")
        if np.any((override < 0.0) | (override > 1.0)):
            raise ValueError("retention_override must lie in [0, 1]")
        retention = override
    entropy = feature_trace.entropy
    weights = config.evidence_weight_floor + (
        1.0 - config.evidence_weight_floor
    ) * (1.0 - entropy)
    predictions, probabilities, state_l1 = accumulate_with_retention(
        evidence,
        stream_ids=stream_ids,
        retention_value=retention,
        evidence_weight_value=weights,
    )
    return SoftMemoryTrace(
        predictions=predictions,
        probabilities=probabilities,
        retention=retention,
        evidence_weight=weights,
        change_risk=risk,
        standardized_features=standardized,
        raw_features=feature_trace.raw_features,
        state_l1=state_l1,
    )


__all__ = [
    "CausalFeatureTrace",
    "ControllerStandardizer",
    "SoftMemoryConfig",
    "SoftMemoryTrace",
    "accumulate_with_retention",
    "causal_control_features",
    "controller_retention",
    "soft_memory_accumulator",
]
