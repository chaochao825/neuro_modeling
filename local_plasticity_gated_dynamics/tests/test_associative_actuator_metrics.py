from __future__ import annotations

import numpy as np
import pytest

from src.analysis.associative_actuator_metrics import (
    matched_rms_scale,
    sign_accuracy,
    train_normalized_score,
    train_reference_variance,
)


def test_train_normalized_metrics_and_rms_matching() -> None:
    train = np.array([-2.0, -1.0, 1.0, 2.0])
    variance = train_reference_variance(train)
    assert variance == pytest.approx(2.5)
    assert train_normalized_score(train, train, train_variance=variance) == 1.0
    raw = np.array([-1.0, 1.0, -1.0, 1.0])
    scale = matched_rms_scale(train, raw)
    assert np.sqrt(np.mean((scale * raw) ** 2)) == pytest.approx(
        np.sqrt(np.mean(train**2))
    )
    assert sign_accuracy(train, train) == 1.0


def test_metric_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive variance"):
        train_reference_variance(np.ones(4))
    with pytest.raises(ValueError, match="identical shape"):
        train_normalized_score([0.0, 1.0], [0.0, 1.0, 2.0], train_variance=1.0)
    with pytest.raises(ValueError, match="positive RMS"):
        matched_rms_scale([1.0, -1.0], [0.0, 0.0])
