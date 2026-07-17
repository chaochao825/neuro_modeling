from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.models.actuator_selector import GRUSelectorBaseline


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(44)
    cues = rng.normal(size=(12, 3, 8))
    cues[:, 2, -1] = 1.0
    summary = cues.sum(axis=1)
    scores = np.column_stack((summary[:, 0], summary[:, 3], summary[:, 5]))
    utilities = np.full_like(scores, 0.1)
    utilities[np.arange(len(cues)), np.argmax(scores, axis=1)] = 0.9
    return cues, utilities


def test_gru_baseline_is_deterministic_cpu_bptt_and_audits_updates() -> None:
    cues, utilities = _toy_data()
    options = dict(
        hidden_dim=8,
        epochs=20,
        learning_rate=0.02,
        weight_decay=1e-4,
        teacher_temperature=0.05,
        seed=123,
        device="cpu",
        deterministic=True,
    )
    first = GRUSelectorBaseline(**options)
    second = GRUSelectorBaseline(**options)
    first_receipt = first.fit(cues, utilities)
    second_receipt = second.fit(cues, utilities)

    np.testing.assert_array_equal(
        first_receipt.train_probabilities, second_receipt.train_probabilities
    )
    assert first_receipt.parameter_fingerprint == second_receipt.parameter_fingerprint
    assert first_receipt.used_bptt is True
    assert first_receipt.used_autograd is True
    assert first_receipt.autograd_engine == "torch.autograd"
    assert first_receipt.device == "cpu"
    assert first_receipt.cumulative_update_l1 > 0.0
    assert first_receipt.cumulative_update_l2 > 0.0
    assert first_receipt.parameter_count > 0
    json.dumps(first_receipt.to_dict())
    assert not first.predict_proba(cues).flags.writeable


def test_gru_baseline_rejects_non_cpu_nondeterministic_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="CPU"):
        GRUSelectorBaseline(device="cuda")
    with pytest.raises(ValueError, match="deterministic"):
        GRUSelectorBaseline(deterministic=False)
    cues, utilities = _toy_data()
    model = GRUSelectorBaseline(epochs=1)
    with pytest.raises(ValueError, match="shape"):
        model.fit(cues[:, :2], utilities)
    bad_utilities = utilities.copy()
    bad_utilities[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        model.fit(cues, bad_utilities)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_proba(cues)

    model.fit(cues, utilities)
    assert model._network is not None  # noqa: SLF001 - fault injection
    with torch.no_grad():
        model._network.readout.bias.fill_(float("inf"))  # noqa: SLF001
    with pytest.raises(FloatingPointError, match="probabilities"):
        model.predict_proba(cues)
