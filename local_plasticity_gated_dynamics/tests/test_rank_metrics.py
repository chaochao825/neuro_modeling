import numpy as np
import pytest

from src.analysis.rank_metrics import (
    effective_rank,
    participation_ratio,
    participation_ratio_from_spectrum,
    rank_summary,
    top_k_singular_energy,
)


def test_equal_singular_modes_have_expected_effective_rank_and_energy() -> None:
    matrix = np.diag([2.0, 2.0, 2.0, 2.0, 0.0])

    assert effective_rank(matrix) == pytest.approx(4.0)
    assert top_k_singular_energy(matrix, 2) == pytest.approx(0.5)
    summary = rank_summary(matrix, k=4)
    assert summary.numerical_rank == 4
    assert summary.top_k_singular_energy == pytest.approx(1.0)


def test_participation_ratio_uses_variance_spectrum() -> None:
    assert participation_ratio_from_spectrum([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    activity = np.array(
        [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    )
    assert participation_ratio(activity) == pytest.approx(2.0)


def test_zero_matrix_metrics_are_explicitly_zero() -> None:
    matrix = np.zeros((3, 2))
    assert effective_rank(matrix) == 0.0
    assert top_k_singular_energy(matrix, 1) == 0.0
    assert participation_ratio_from_spectrum([0.0, 0.0]) == 0.0


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (effective_rank, (np.array([[np.nan]]),)),
        (top_k_singular_energy, (np.eye(2), 0)),
        (top_k_singular_energy, (np.eye(2), True)),
        (participation_ratio_from_spectrum, ([1.0, -1.0],)),
        (participation_ratio, (np.ones((1, 2)),)),
    ],
)
def test_rank_metrics_reject_invalid_inputs(function, args) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(*args)
