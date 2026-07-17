from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pytest

import src.plasticity.selector_three_factor as local_module
from src.plasticity.selector_three_factor import LocalThreeFactorSelector


def _toy_data(n_samples: int = 18) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    cues = rng.normal(size=(n_samples, 3, 8))
    cues[:, 0, 3:] = 0.0
    cues[:, 1, :3] = 0.0
    cues[:, 1, 5:] = 0.0
    cues[:, 2, :5] = 0.0
    cues[:, 2, -1] = 1.0
    belief = 0.8**2 * cues[:, 0] + 0.8 * cues[:, 1] + cues[:, 2]
    scores = np.column_stack((belief[:, 0], belief[:, 3], -belief[:, 0] - belief[:, 3]))
    utilities = np.full_like(scores, 0.2)
    utilities[np.arange(n_samples), np.argmax(scores, axis=1)] = 0.9
    return cues, utilities


def test_one_step_update_is_prebelief_eligibility_times_k3_modulation() -> None:
    cues = np.zeros((1, 3, 8))
    cues[0, 0, 0] = 1.0
    cues[0, 1, 1] = 2.0
    cues[0, 2, 2] = -1.0
    cues[0, 2, -1] = 1.0
    utilities = np.array([[1.0, 0.2, 0.0]])
    model = LocalThreeFactorSelector(
        learning_rate=0.1,
        epochs=1,
        temperature=1.0,
        teacher_temperature=0.5,
        l2=0.0,
        eligibility_decay=0.8,
        belief_retention=0.8,
        shuffle_seed=3,
    )
    receipt = model.fit(cues, utilities)

    eligibility = 0.8**2 * cues[0, 0] + 0.8 * cues[0, 1] + cues[0, 2]
    shifted = utilities[0] / 0.5
    teacher = np.exp(shifted - shifted.max())
    teacher /= teacher.sum()
    expected = 0.1 * np.outer(teacher - 1.0 / 3.0, eligibility)
    np.testing.assert_allclose(receipt.weights, expected, atol=1e-14)
    assert receipt.modulatory_dimension == 3
    assert "pre_belief" in receipt.eligibility_definition


def test_local_selector_is_deterministic_immutable_and_json_auditable() -> None:
    cues, utilities = _toy_data()
    options = dict(epochs=40, shuffle_seed=91, teacher_temperature=0.05)
    first = LocalThreeFactorSelector(**options)
    second = LocalThreeFactorSelector(**options)
    first_receipt = first.fit(cues, utilities)
    second_receipt = second.fit(cues, utilities)

    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(
        first_receipt.train_probabilities, second_receipt.train_probabilities
    )
    assert first_receipt.shuffle_fingerprint == second_receipt.shuffle_fingerprint
    assert first_receipt.cumulative_update_l1 > 0.0
    assert first_receipt.cumulative_update_l2 > 0.0
    assert first_receipt.used_bptt is False
    assert first_receipt.used_autograd is False
    json.dumps(first_receipt.to_dict())
    assert not first.weights.flags.writeable
    assert not first.predict_proba(cues).flags.writeable


def test_local_module_has_no_gradient_engine_import_or_reverse_time_pass() -> None:
    source = inspect.getsource(local_module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert ".backward(" not in source
    assert "reversed(" not in source


def test_local_selector_rejects_bad_shapes_nonfinite_data_and_bad_hyperparameters() -> (
    None
):
    cues, utilities = _toy_data(4)
    model = LocalThreeFactorSelector(epochs=2)
    with pytest.raises(ValueError, match="shape"):
        model.fit(cues[:, :2], utilities)
    bad = cues.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad, utilities)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LocalThreeFactorSelector(eligibility_decay=1.1)
    with pytest.raises(RuntimeError, match="not been fitted"):
        LocalThreeFactorSelector().predict_proba(cues)

    fitted = LocalThreeFactorSelector(epochs=1)
    fitted.fit(cues, utilities)
    fitted._weights = np.full((3, 8), 1e308)  # noqa: SLF001 - fault injection
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="probabilities"):
            fitted.predict_proba(cues)
