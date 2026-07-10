import ast
from pathlib import Path

import numpy as np
import torch

from src.baselines.bptt import BPTTConfig, train_bptt_baseline
from src.models.local_predictive import LocalPredictiveConfig, LocalPredictiveModel


PROJECT = Path(__file__).resolve().parents[1]


def test_bptt_baseline_is_tagged_and_trains_only_in_baseline_module() -> None:
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(8, 5, 2))
    targets = inputs[..., :1]
    mask = np.ones((8, 5), dtype=bool)
    model, losses = train_bptt_baseline(
        inputs, targets, mask, BPTTConfig(hidden_size=4, epochs=2, batch_size=4, seed=0)
    )
    assert len(losses) == 2
    assert model.checkpoint_metadata()["eligible_for_local_initialization"] is False


def test_local_modules_have_static_no_bptt_gate() -> None:
    paths = [
        PROJECT / "src/models/local_predictive.py",
        PROJECT / "src/models/ei_rate_network.py",
        PROJECT / "src/models/md_gate.py",
        PROJECT / "src/plasticity/three_factor.py",
        PROJECT / "src/plasticity/inhibitory_homeostasis.py",
    ]
    forbidden = {"torch.optim", "torch.autograd", "src.baselines", "backward"}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        attributes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Attribute):
                attributes.add(node.attr)
        assert not any(any(token in item for token in forbidden) for item in imported)
        assert "backward" not in attributes


def test_local_training_never_calls_torch_backward(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("local model attempted autograd")

    monkeypatch.setattr(torch.Tensor, "backward", fail)
    basis = np.eye(4)[:, :2]
    model = LocalPredictiveModel(
        basis,
        config=LocalPredictiveConfig(max_epochs=2, batch_size=None, seed=0),
    )
    x = np.eye(4)
    model.fit(x, np.roll(x, 1, axis=0))
    assert model.plasticity_cost >= 0.0
