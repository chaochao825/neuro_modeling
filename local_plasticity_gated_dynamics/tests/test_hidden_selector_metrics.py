from __future__ import annotations

import numpy as np
import pytest

from src.analysis.hidden_selector_metrics import (
    binary_accuracy,
    holm_adjust,
    paired_bootstrap_interval,
    sign_flip_pvalue,
)


def test_binary_accuracy_is_sign_based_and_paired() -> None:
    assert binary_accuracy([-1.0, 1.0, 1.0], [-2.0, 0.2, -0.1]) == pytest.approx(
        2.0 / 3.0
    )
    with pytest.raises(ValueError, match="paired"):
        binary_accuracy([1.0, -1.0], [1.0])


def test_seed_bootstrap_and_sign_flip_are_deterministic() -> None:
    differences = np.linspace(0.03, 0.08, 12)
    first = paired_bootstrap_interval(
        differences, n_resamples=2000, seed=17
    )
    second = paired_bootstrap_interval(
        differences, n_resamples=2000, seed=17
    )
    assert first == second
    assert 0.03 < first[0] < first[1] < 0.08
    pvalue = sign_flip_pvalue(
        differences - 0.02, n_resamples=5000, seed=19
    )
    assert pvalue < 0.01


def test_holm_adjustment_is_monotone_in_ordered_family() -> None:
    adjusted = holm_adjust({"first": 0.01, "second": 0.03, "third": 0.04})
    assert adjusted == pytest.approx(
        {"first": 0.03, "second": 0.06, "third": 0.06}
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        holm_adjust({})
