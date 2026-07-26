from __future__ import annotations

import numpy as np
import pytest

from src.models.embedding_evidence import VMFEvidenceModel


def _separable_features() -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [
            [1.0, 0.05, 0.0],
            [1.0, -0.05, 0.0],
            [0.0, 1.0, 0.05],
            [0.0, 1.0, -0.05],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    return features, labels


def test_vmf_fit_and_probability_rows() -> None:
    features, labels = _separable_features()
    model = VMFEvidenceModel.fit(features, labels, n_classes=2)
    probabilities = model.probabilities(features, temperature=1.0)
    assert model.n_classes == 2
    assert model.feature_dim == 3
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(probabilities, axis=1), labels)


def test_vmf_relative_likelihood_is_nonpositive() -> None:
    features, labels = _separable_features()
    model = VMFEvidenceModel.fit(features, labels, n_classes=2)
    scores = model.relative_log_likelihood(features)
    assert np.all(scores <= 1e-12)
    assert np.all(scores[np.arange(4), labels] > scores[np.arange(4), 1 - labels])


def test_vmf_missing_class_and_bad_temperature_fail_closed() -> None:
    features, labels = _separable_features()
    with pytest.raises(ValueError, match="no training samples"):
        VMFEvidenceModel.fit(features, labels, n_classes=3)
    model = VMFEvidenceModel.fit(features, labels, n_classes=2)
    with pytest.raises(ValueError):
        model.probabilities(features, temperature=0.0)


def test_vmf_model_parameters_are_read_only() -> None:
    features, labels = _separable_features()
    model = VMFEvidenceModel.fit(features, labels, n_classes=2)
    with pytest.raises(ValueError):
        model.directions[0, 0] = 0.0
