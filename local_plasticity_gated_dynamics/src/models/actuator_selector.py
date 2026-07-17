"""Deterministic CPU GRU/BPTT baseline for Exp27 actuator selection.

This is intentionally a baseline module.  The local main model lives in
``src.plasticity.selector_three_factor`` and does not import this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.actuator_selector_dataset import CANDIDATE_MODES


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def _positive_float(value: object, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    invalid = resolved < 0.0 if allow_zero else resolved <= 0.0
    if not np.isfinite(resolved) or invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return resolved


def _readonly(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class _SelectorGRUNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dtype=torch.float64,
        )
        self.readout = nn.Linear(
            hidden_dim,
            len(CANDIDATE_MODES),
            dtype=torch.float64,
        )

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(cues)
        return self.readout(sequence[:, -1])


@dataclass(frozen=True)
class GRUTrainingReceipt:
    """JSON-convertible audit receipt for the isolated BPTT baseline."""

    algorithm: str
    objective: str
    used_bptt: bool
    used_autograd: bool
    autograd_engine: str
    candidate_modes: tuple[str, ...]
    input_dim: int
    hidden_dim: int
    learning_rate: float
    epochs: int
    weight_decay: float
    teacher_temperature: float
    seed: int
    device: str
    deterministic: bool
    n_samples: int
    parameter_count: int
    parameter_l1: float
    parameter_l2: float
    parameter_fingerprint: str
    final_loss: float
    cumulative_update_l1: float
    cumulative_update_l2: float
    train_probabilities: np.ndarray

    def __post_init__(self) -> None:
        if self.algorithm != "gru_bptt":
            raise ValueError("algorithm must be gru_bptt")
        if not self.used_bptt or not self.used_autograd:
            raise ValueError("GRU baseline must disclose BPTT and autograd")
        if self.autograd_engine != "torch.autograd" or self.device != "cpu":
            raise ValueError("GRU baseline must use torch autograd on CPU")
        if self.deterministic is not True:
            raise ValueError("GRU baseline must be deterministic")
        if tuple(self.candidate_modes) != CANDIDATE_MODES:
            raise ValueError("receipt candidate dictionary is invalid")
        if (
            not isinstance(self.parameter_fingerprint, str)
            or len(self.parameter_fingerprint) != 64
        ):
            raise ValueError("parameter_fingerprint must be a SHA-256 digest")
        probabilities = _readonly(
            self.train_probabilities, name="train_probabilities", ndim=2
        )
        if probabilities.shape != (int(self.n_samples), len(CANDIDATE_MODES)):
            raise ValueError("train_probabilities has an invalid shape")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("train probabilities must sum to one")
        object.__setattr__(self, "train_probabilities", probabilities)
        scalar_values = (
            self.learning_rate,
            self.weight_decay,
            self.teacher_temperature,
            self.parameter_l1,
            self.parameter_l2,
            self.final_loss,
            self.cumulative_update_l1,
            self.cumulative_update_l2,
        )
        if not np.all(np.isfinite(scalar_values)) or min(scalar_values) < 0.0:
            raise ValueError("receipt scalar metrics must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a directly JSON-serializable receipt."""

        return {
            field: (value.tolist() if isinstance(value, np.ndarray) else value)
            for field, value in self.__dict__.items()
        }


class GRUSelectorBaseline:
    """Small, deterministic, full-sequence BPTT selector baseline."""

    used_bptt = True
    used_autograd = True

    def __init__(
        self,
        input_dim: int = 8,
        *,
        hidden_dim: int = 8,
        learning_rate: float = 0.02,
        epochs: int = 200,
        weight_decay: float = 1e-4,
        teacher_temperature: float = 0.05,
        seed: int = 0,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        self.input_dim = _positive_int(input_dim, name="input_dim")
        self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
        self.learning_rate = _positive_float(learning_rate, name="learning_rate")
        self.epochs = _positive_int(epochs, name="epochs")
        self.weight_decay = _positive_float(
            weight_decay, name="weight_decay", allow_zero=True
        )
        self.teacher_temperature = _positive_float(
            teacher_temperature, name="teacher_temperature"
        )
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        self.seed = int(seed)
        if device != "cpu":
            raise ValueError("GRU selector baseline is registered for CPU only")
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a bool")
        if not deterministic:
            raise ValueError("GRU selector baseline must be deterministic")
        self.device = device
        self.deterministic = deterministic
        self._network: _SelectorGRUNetwork | None = None

    def _validate_inputs(
        self, cues: object, utilities: object | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        cue_array = _readonly(cues, name="cues", ndim=3)
        if cue_array.shape[0] < 1 or cue_array.shape[1] != 3:
            raise ValueError("cues must have shape [sample, 3, input_dim]")
        if cue_array.shape[2] != self.input_dim:
            raise ValueError("cue input dimension does not match selector")
        if utilities is None:
            return cue_array, None
        utility_array = _readonly(utilities, name="utilities", ndim=2)
        if utility_array.shape != (cue_array.shape[0], len(CANDIDATE_MODES)):
            raise ValueError("utilities must have shape [sample, 3]")
        if np.any((utility_array < 0.0) | (utility_array > 1.0)):
            raise ValueError("utility values must lie in [0, 1]")
        return cue_array, utility_array

    @staticmethod
    def _parameter_vector(network: nn.Module) -> np.ndarray:
        arrays = [
            parameter.detach().cpu().numpy().reshape(-1)
            for parameter in network.parameters()
        ]
        return np.concatenate(arrays).astype(np.float64, copy=False)

    def fit(self, cues: object, utilities: object) -> GRUTrainingReceipt:
        cue_array, utility_array = self._validate_inputs(cues, utilities)
        assert utility_array is not None
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        cumulative_l1 = 0.0
        cumulative_squared_l2 = 0.0
        try:
            torch.use_deterministic_algorithms(True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                network = _SelectorGRUNetwork(self.input_dim, self.hidden_dim).to("cpu")
                optimizer = torch.optim.AdamW(
                    network.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
                cue_tensor = torch.as_tensor(
                    np.array(cue_array, copy=True), dtype=torch.float64, device="cpu"
                )
                utility_tensor = torch.as_tensor(
                    np.array(utility_array, copy=True),
                    dtype=torch.float64,
                    device="cpu",
                )
                teacher = torch.softmax(
                    utility_tensor / self.teacher_temperature,
                    dim=1,
                ).detach()
                network.train()
                for _ in range(self.epochs):
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(cue_tensor)
                    log_probability = torch.log_softmax(logits, dim=1)
                    loss = -torch.mean(torch.sum(teacher * log_probability, dim=1))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("GRU selector loss became non-finite")
                    before = [
                        parameter.detach().clone() for parameter in network.parameters()
                    ]
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for old, parameter in zip(
                            before, network.parameters(), strict=True
                        ):
                            update = parameter - old
                            cumulative_l1 += float(torch.sum(torch.abs(update)).item())
                            cumulative_squared_l2 += float(
                                torch.sum(update * update).item()
                            )
                network.eval()
                with torch.no_grad():
                    final_logits = network(cue_tensor)
                    final_loss_tensor = -torch.mean(
                        torch.sum(
                            teacher * torch.log_softmax(final_logits, dim=1), dim=1
                        )
                    )
                self._network = network
                final_loss = float(final_loss_tensor.item())
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
        probabilities = self.predict_proba(cue_array)
        assert self._network is not None
        parameters = self._parameter_vector(self._network)
        parameter_digest = hashlib.sha256()
        parameter_digest.update(np.asarray(parameters.shape, dtype=np.int64).tobytes())
        parameter_digest.update(parameters.astype("<f8", copy=False).tobytes())
        return GRUTrainingReceipt(
            algorithm="gru_bptt",
            objective="soft_teacher_cross_entropy",
            used_bptt=True,
            used_autograd=True,
            autograd_engine="torch.autograd",
            candidate_modes=CANDIDATE_MODES,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            teacher_temperature=self.teacher_temperature,
            seed=self.seed,
            device=self.device,
            deterministic=self.deterministic,
            n_samples=cue_array.shape[0],
            parameter_count=int(parameters.size),
            parameter_l1=float(np.sum(np.abs(parameters))),
            parameter_l2=float(np.linalg.norm(parameters)),
            parameter_fingerprint=parameter_digest.hexdigest(),
            final_loss=final_loss,
            cumulative_update_l1=cumulative_l1,
            cumulative_update_l2=float(np.sqrt(cumulative_squared_l2)),
            train_probabilities=probabilities,
        )

    def predict_proba(self, cues: object) -> np.ndarray:
        cue_array, _ = self._validate_inputs(cues)
        if self._network is None:
            raise RuntimeError("selector has not been fitted")
        self._network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(
                np.array(cue_array, copy=True), dtype=torch.float64, device="cpu"
            )
            probabilities = torch.softmax(self._network(tensor), dim=1).cpu().numpy()
        result = np.array(probabilities, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(result)) or not np.allclose(
            np.sum(result, axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise FloatingPointError("GRU selector probabilities are invalid")
        result.setflags(write=False)
        return result

    def predict(self, cues: object) -> np.ndarray:
        choices = np.argmax(self.predict_proba(cues), axis=1).astype(np.int64)
        choices.setflags(write=False)
        return choices


__all__ = ["GRUSelectorBaseline", "GRUTrainingReceipt"]
