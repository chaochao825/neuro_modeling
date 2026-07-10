import numpy as np
import pytest

from src.analysis.task_subspaces import (
    compare_condition_subspaces,
    fit_condition_subspace,
    fit_demixed_condition_subspace,
)


def test_condition_subspace_is_train_fit_and_recovers_known_axis() -> None:
    activity = np.array([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
    fitted = fit_condition_subspace(
        activity, ["a", "a", "b", "b"], n_components=1, sample_ids=[0, 1, 2, 3]
    )
    assert abs(fitted.basis[0, 0]) > 0.99
    assert fitted.fit_sample_ids == (0, 1, 2, 3)
    assert not fitted.basis.flags.writeable


def test_condition_subspace_comparison() -> None:
    first = fit_condition_subspace(
        np.array([[-1.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        [0, 0, 1, 1],
        n_components=1,
    )
    second = fit_condition_subspace(
        np.array([[0.0, -1.0], [0.0, -2.0], [0.0, 1.0], [0.0, 2.0]]),
        [0, 0, 1, 1],
        n_components=1,
    )
    comparison = compare_condition_subspaces(first, second)
    assert comparison["principal_angles_degrees"][0] == pytest.approx(90.0)
    assert comparison["overlap"] == pytest.approx(0.0)


def test_demixed_subspace_controls_correlated_nuisance_factor() -> None:
    target = np.tile([0, 0, 1, 1], 5)
    nuisance = np.tile([0, 1, 0, 1], 5)
    activity = np.column_stack([2 * target - 1, 3 * nuisance - 1.5])
    fitted = fit_demixed_condition_subspace(
        activity,
        target,
        nuisance_labels={"nuisance": nuisance},
        n_components=1,
    )
    assert abs(fitted.basis[0, 0]) > 0.99


def test_demixed_subspace_rejects_exact_target_nuisance_confounding() -> None:
    target = np.tile([0, 0, 1, 1], 5)
    activity = np.column_stack([2 * target - 1, np.zeros(target.size)])
    with pytest.raises(ValueError, match="not estimable"):
        fit_demixed_condition_subspace(
            activity,
            target,
            nuisance_labels={"duplicate": target.copy()},
            n_components=1,
        )
