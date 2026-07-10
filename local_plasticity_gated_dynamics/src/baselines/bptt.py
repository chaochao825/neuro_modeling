"""BPTT performance ceiling for context integration.

This module is intentionally isolated from every local-learning model. Its
checkpoints are tagged and must never initialize a local-plasticity run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BPTTConfig:
    hidden_size: int
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    grad_clip: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("hidden_size, epochs, and batch_size must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.seed < 0:
            raise ValueError("learning_rate/grad_clip must be positive and seed non-negative")


class BPTTRateRNN(nn.Module):
    """Simple tanh RNN used only as a non-local baseline."""

    training_algorithm = "bptt_baseline"

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or output_size < 1:
            raise ValueError("network dimensions must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size, bias=True)
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape [batch, time, input_size]")
        hidden = torch.zeros(
            inputs.shape[0], self.hidden_size, dtype=inputs.dtype, device=inputs.device
        )
        history = []
        for time in range(inputs.shape[1]):
            hidden = torch.tanh(
                self.input_layer(inputs[:, time]) + self.recurrent_layer(hidden)
            )
            history.append(hidden)
        states = torch.stack(history, dim=1)
        return self.readout(states), states

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "training_algorithm": self.training_algorithm,
            "used_autograd": True,
            "eligible_for_local_initialization": False,
        }


def train_bptt_baseline(
    inputs: np.ndarray,
    targets: np.ndarray,
    loss_mask: np.ndarray,
    config: BPTTConfig,
) -> tuple[BPTTRateRNN, list[float]]:
    """Train the isolated ceiling model with full recurrent BPTT."""

    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    mask = np.asarray(loss_mask, dtype=bool)
    if x.ndim != 3 or y.shape != (*x.shape[:2], 1) or mask.shape != x.shape[:2]:
        raise ValueError("inputs/targets/loss_mask shapes are inconsistent")
    torch.manual_seed(config.seed)
    model = BPTTRateRNN(x.shape[-1], config.hidden_size, y.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    losses: list[float] = []
    tensor_x = torch.from_numpy(x)
    tensor_y = torch.from_numpy(y)
    tensor_mask = torch.from_numpy(mask)
    for _ in range(config.epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, x.shape[0], config.batch_size):
            batch = order[start : start + config.batch_size]
            prediction, _ = model(tensor_x[batch])
            selected = tensor_mask[batch]
            loss = torch.mean((prediction[selected] - tensor_y[batch][selected]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        losses.append(epoch_loss / max(1, n_batches))
    return model, losses


@torch.no_grad()
def predict_bptt(model: BPTTRateRNN, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction, states = model(torch.as_tensor(inputs, dtype=torch.float32))
    return prediction.cpu().numpy(), states.cpu().numpy()
