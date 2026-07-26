"""Train-only class-conditional evidence models for frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _readonly(value: ArrayLike, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _feature_matrix(value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError("features must have shape [sample, dimension>=2]")
    if not np.all(np.isfinite(result)):
        raise ValueError("features must be finite")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("feature rows must have positive norm")
    return np.asarray(result / norms, dtype=np.float64)


def _class_labels(value: ArrayLike, *, n_samples: int, n_classes: int) -> NDArray[np.int64]:
    labels = np.asarray(value)
    if labels.dtype.kind not in {"i", "u"} or labels.shape != (n_samples,):
        raise ValueError("labels must be an aligned integer vector")
    result = np.asarray(labels, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= n_classes):
        raise ValueError("labels fall outside declared classes")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class VMFEvidenceModel:
    """Shared-concentration von Mises--Fisher model on unit embeddings.

    A shared concentration keeps the class normalization constants identical.
    The returned log likelihood is relative to perfect alignment, so it is
    finite, non-positive, and suitable for causal predictive surprise.
    """

    directions: FloatArray
    concentration: float
    class_counts: NDArray[np.int64]

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions, dtype=np.float64)
        counts = np.asarray(self.class_counts, dtype=np.int64)
        concentration = float(self.concentration)
        if directions.ndim != 2 or directions.shape[0] < 2 or directions.shape[1] < 2:
            raise ValueError("directions must have shape [class>=2, dimension>=2]")
        if counts.shape != (directions.shape[0],) or np.any(counts < 1):
            raise ValueError("class_counts must be positive and aligned")
        if not np.all(np.isfinite(directions)):
            raise ValueError("directions must be finite")
        if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6):
            raise ValueError("directions must have unit norm")
        if not np.isfinite(concentration) or concentration <= 0.0:
            raise ValueError("concentration must be finite and positive")
        object.__setattr__(
            self, "directions", _readonly(directions, dtype=np.float64)
        )
        object.__setattr__(self, "class_counts", _readonly(counts, dtype=np.int64))
        object.__setattr__(self, "concentration", concentration)

    @classmethod
    def fit(
        cls,
        features_value: ArrayLike,
        labels_value: ArrayLike,
        *,
        n_classes: int,
        min_concentration: float = 1.0,
        max_concentration: float = 1000.0,
    ) -> VMFEvidenceModel:
        """Fit directions and a shared concentration on training samples only."""

        if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
            raise TypeError("n_classes must be an integer")
        n_classes = int(n_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be at least two")
        features = _feature_matrix(features_value)
        labels = _class_labels(
            labels_value, n_samples=features.shape[0], n_classes=n_classes
        )
        lower = float(min_concentration)
        upper = float(max_concentration)
        if not np.isfinite(lower) or not np.isfinite(upper) or not 0.0 < lower <= upper:
            raise ValueError("invalid concentration bounds")
        directions = np.empty((n_classes, features.shape[1]), dtype=np.float64)
        counts = np.empty(n_classes, dtype=np.int64)
        resultant_lengths: list[float] = []
        for class_id in range(n_classes):
            class_features = features[labels == class_id]
            if class_features.shape[0] == 0:
                raise ValueError(f"class {class_id} has no training samples")
            resultant = np.sum(class_features, axis=0)
            resultant_norm = float(np.linalg.norm(resultant))
            if resultant_norm <= 0.0:
                raise ValueError(f"class {class_id} has a zero resultant")
            directions[class_id] = resultant / resultant_norm
            counts[class_id] = class_features.shape[0]
            resultant_lengths.append(resultant_norm / class_features.shape[0])
        mean_resultant = float(np.average(resultant_lengths, weights=counts))
        dimension = float(features.shape[1])
        denominator = max(1.0 - mean_resultant**2, 1e-8)
        estimate = mean_resultant * (dimension - mean_resultant**2) / denominator
        concentration = float(np.clip(estimate, lower, upper))
        return cls(
            directions=directions,
            concentration=concentration,
            class_counts=counts,
        )

    @property
    def n_classes(self) -> int:
        return int(self.directions.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.directions.shape[1])

    def relative_log_likelihood(self, features_value: ArrayLike) -> FloatArray:
        features = _feature_matrix(features_value)
        if features.shape[1] != self.feature_dim:
            raise ValueError("feature dimension does not match evidence model")
        cosine = np.clip(features @ self.directions.T, -1.0, 1.0)
        result = self.concentration * (cosine - 1.0)
        return np.asarray(result, dtype=np.float64)

    def probabilities(
        self, features_value: ArrayLike, *, temperature: float
    ) -> FloatArray:
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        logits = self.relative_log_likelihood(features_value) / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        result = np.exp(logits)
        result /= np.sum(result, axis=1, keepdims=True)
        return np.asarray(result, dtype=np.float64)


__all__ = ["VMFEvidenceModel"]
