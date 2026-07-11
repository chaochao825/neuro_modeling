"""Deterministically tuned recurrent baselines with an auditable inner split.

This module is an explicit non-local baseline boundary.  Both model families
use BPTT, and neither their parameters nor their checkpoints are eligible to
initialize a local-learning model.  Hyperparameters are selected exclusively
from complete validation blocks; an outer test set is deliberately absent
from the tuning API.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import nn


CellType = Literal["rate_rnn", "gru"]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _block_token(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        raise TypeError("block identifiers must be scalar values")
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _fingerprint_parts(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _fingerprint_parts(
        [
            str(contiguous.dtype).encode("utf-8"),
            repr(contiguous.shape).encode("utf-8"),
            contiguous.tobytes(),
        ]
    )


def _block_fingerprint(block_ids: np.ndarray) -> str:
    return _fingerprint_parts(
        [_block_token(value).encode("utf-8") for value in block_ids.tolist()]
    )


def _trial_multiset_fingerprint(*datasets: RecurrentSequenceData) -> str:
    """Fingerprint trial contents independently of split concatenation order."""

    trial_digests: list[bytes] = []
    for data in datasets:
        for index in range(data.trial_count):
            digest = _fingerprint_parts(
                [
                    _block_token(data.block_ids[index]).encode("utf-8"),
                    _array_fingerprint(data.inputs[index]).encode("ascii"),
                    _array_fingerprint(data.targets[index]).encode("ascii"),
                    _array_fingerprint(data.loss_mask[index]).encode("ascii"),
                ]
            )
            trial_digests.append(bytes.fromhex(digest))
    return _fingerprint_parts(sorted(trial_digests))


@dataclass(frozen=True, eq=False)
class RecurrentSequenceData:
    """One trial-major sequence split with globally meaningful block IDs."""

    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    block_ids: np.ndarray
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        inputs = np.array(self.inputs, dtype=np.float32, copy=True)
        targets = np.array(self.targets, dtype=np.float32, copy=True)
        loss_mask = np.array(self.loss_mask, dtype=bool, copy=True)
        block_ids = np.array(self.block_ids, copy=True)
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [trial, time, input]")
        if targets.ndim != 3 or targets.shape[:2] != inputs.shape[:2]:
            raise ValueError("targets must have shape [trial, time, output]")
        if targets.shape[-1] < 1:
            raise ValueError("targets must have at least one output")
        if loss_mask.shape != inputs.shape[:2]:
            raise ValueError("loss_mask must have shape [trial, time]")
        if block_ids.ndim != 1 or block_ids.shape[0] != inputs.shape[0]:
            raise ValueError("block_ids must contain one identifier per trial")
        if inputs.shape[0] < 1 or inputs.shape[1] < 1 or inputs.shape[2] < 1:
            raise ValueError("sequence dimensions must be positive")
        if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(targets)):
            raise ValueError("inputs and targets must be finite")
        if not np.any(loss_mask):
            raise ValueError("loss_mask must select at least one point")
        # Validate identifiers early and make all stored arrays immutable.  A
        # tuning result can therefore be tied to stable data fingerprints.
        for value in block_ids.tolist():
            _block_token(value)
        for array in (inputs, targets, loss_mask, block_ids):
            array.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "loss_mask", loss_mask)
        object.__setattr__(self, "block_ids", block_ids)

    @property
    def trial_count(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def block_tokens(self) -> tuple[str, ...]:
        return tuple(_block_token(value) for value in self.block_ids.tolist())

    @property
    def block_count(self) -> int:
        return len(set(self.block_tokens))

    def audit_fingerprint(self) -> str:
        return _fingerprint_parts(
            [
                _array_fingerprint(self.inputs).encode("ascii"),
                _array_fingerprint(self.targets).encode("ascii"),
                _array_fingerprint(self.loss_mask).encode("ascii"),
                _block_fingerprint(self.block_ids).encode("ascii"),
            ]
        )


def block_safe_inner_split(
    development: RecurrentSequenceData,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[RecurrentSequenceData, RecurrentSequenceData]:
    """Split a development set by whole blocks for inner model selection."""

    if not isinstance(development, RecurrentSequenceData):
        raise TypeError("development must be RecurrentSequenceData")
    fraction = _finite_float(validation_fraction, "validation_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    seed = _nonnegative_integer(seed, "seed")
    tokens = np.asarray(development.block_tokens, dtype=object)
    unique_tokens = list(dict.fromkeys(tokens.tolist()))
    if len(unique_tokens) < 2:
        raise ValueError("at least two complete blocks are required")
    n_validation = int(round(fraction * len(unique_tokens)))
    n_validation = min(max(n_validation, 1), len(unique_tokens) - 1)
    rng = np.random.default_rng(seed)
    selected = {
        unique_tokens[index]
        for index in rng.permutation(len(unique_tokens))[:n_validation]
    }
    validation_indices = np.flatnonzero(np.isin(tokens, list(selected)))
    training_indices = np.flatnonzero(~np.isin(tokens, list(selected)))

    def subset(indices: np.ndarray, name: str) -> RecurrentSequenceData:
        return RecurrentSequenceData(
            development.inputs[indices],
            development.targets[indices],
            development.loss_mask[indices],
            development.block_ids[indices],
            name,
        )

    return subset(training_indices, "inner_train"), subset(
        validation_indices, "inner_validation"
    )


@dataclass(frozen=True)
class RecurrentCandidate:
    """One preregistered recurrent-baseline hyperparameter candidate."""

    cell_type: CellType
    hidden_size: int
    learning_rate: float
    weight_decay: float = 0.0
    max_epochs: int = 100
    batch_size: int = 32
    grad_clip: float = 1.0
    patience: int = 10
    min_delta: float = 0.0
    rate_leak: float = 1.0

    def __post_init__(self) -> None:
        if self.cell_type not in {"rate_rnn", "gru"}:
            raise ValueError("cell_type must be 'rate_rnn' or 'gru'")
        hidden_size = _positive_integer(self.hidden_size, "hidden_size")
        max_epochs = _positive_integer(self.max_epochs, "max_epochs")
        batch_size = _positive_integer(self.batch_size, "batch_size")
        patience = _positive_integer(self.patience, "patience")
        learning_rate = _finite_float(self.learning_rate, "learning_rate")
        weight_decay = _finite_float(self.weight_decay, "weight_decay")
        grad_clip = _finite_float(self.grad_clip, "grad_clip")
        min_delta = _finite_float(self.min_delta, "min_delta")
        rate_leak = _finite_float(self.rate_leak, "rate_leak")
        if learning_rate <= 0.0 or grad_clip <= 0.0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if weight_decay < 0.0 or min_delta < 0.0:
            raise ValueError("weight_decay and min_delta must be non-negative")
        if not 0.0 < rate_leak <= 1.0:
            raise ValueError("rate_leak must lie in (0, 1]")
        if self.cell_type == "gru" and rate_leak != 1.0:
            raise ValueError("rate_leak is only configurable for rate_rnn")
        for name, value in (
            ("hidden_size", hidden_size),
            ("max_epochs", max_epochs),
            ("batch_size", batch_size),
            ("patience", patience),
            ("learning_rate", learning_rate),
            ("weight_decay", weight_decay),
            ("grad_clip", grad_clip),
            ("min_delta", min_delta),
            ("rate_leak", rate_leak),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_candidate_grid(
    *,
    cell_types: Sequence[CellType],
    hidden_sizes: Sequence[int],
    learning_rates: Sequence[float],
    weight_decays: Sequence[float] = (0.0,),
    rate_leaks: Sequence[float] = (1.0,),
    max_epochs: int = 100,
    batch_size: int = 32,
    grad_clip: float = 1.0,
    patience: int = 10,
    min_delta: float = 0.0,
) -> tuple[RecurrentCandidate, ...]:
    """Build a deterministic Cartesian candidate grid for both families."""

    if not cell_types or not hidden_sizes or not learning_rates or not weight_decays:
        raise ValueError("candidate-grid axes must be non-empty")
    if not rate_leaks:
        raise ValueError("rate_leaks must be non-empty")
    candidates: list[RecurrentCandidate] = []
    for cell_type, hidden_size, learning_rate, weight_decay in product(
        cell_types, hidden_sizes, learning_rates, weight_decays
    ):
        leaks = rate_leaks if cell_type == "rate_rnn" else (1.0,)
        for rate_leak in leaks:
            candidates.append(
                RecurrentCandidate(
                    cell_type=cell_type,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    grad_clip=grad_clip,
                    patience=patience,
                    min_delta=min_delta,
                    rate_leak=rate_leak,
                )
            )
    encoded = [json.dumps(item.to_dict(), sort_keys=True) for item in candidates]
    if len(encoded) != len(set(encoded)):
        raise ValueError("candidate grid contains duplicate configurations")
    return tuple(candidates)


class RateRNNBaseline(nn.Module):
    """Leaky tanh rate RNN trained end-to-end with BPTT."""

    training_algorithm = "bptt_rate_rnn_baseline"
    cell_type = "rate_rnn"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        leak: float,
    ) -> None:
        super().__init__()
        self.input_size = _positive_integer(input_size, "input_size")
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.output_size = _positive_integer(output_size, "output_size")
        self.leak = _finite_float(leak, "leak")
        if not 0.0 < self.leak <= 1.0:
            raise ValueError("leak must lie in (0, 1]")
        self.input_layer = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.recurrent_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.readout = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape [batch, time, input_size]")
        hidden = torch.zeros(
            inputs.shape[0], self.hidden_size, dtype=inputs.dtype, device=inputs.device
        )
        history = []
        for time_index in range(inputs.shape[1]):
            proposal = torch.tanh(
                self.input_layer(inputs[:, time_index]) + self.recurrent_layer(hidden)
            )
            hidden = (1.0 - self.leak) * hidden + self.leak * proposal
            history.append(hidden)
        states = torch.stack(history, dim=1)
        return self.readout(states), states

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "cell_type": self.cell_type,
            "training_algorithm": self.training_algorithm,
            "used_autograd": True,
            "eligible_for_local_initialization": False,
        }


class GRUBaseline(nn.Module):
    """GRU sequence baseline trained end-to-end with BPTT."""

    training_algorithm = "bptt_gru_baseline"
    cell_type = "gru"

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = _positive_integer(input_size, "input_size")
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.output_size = _positive_integer(output_size, "output_size")
        self.recurrent = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.readout = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("inputs must have shape [batch, time, input_size]")
        states, _ = self.recurrent(inputs)
        return self.readout(states), states

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "cell_type": self.cell_type,
            "training_algorithm": self.training_algorithm,
            "used_autograd": True,
            "eligible_for_local_initialization": False,
        }


RecurrentModel = RateRNNBaseline | GRUBaseline


@dataclass(frozen=True)
class CandidateAudit:
    """Serializable outcome for one candidate, including failed candidates."""

    candidate_id: str
    config: RecurrentCandidate
    candidate_seed: int
    status: Literal["complete", "failed"]
    selected: bool
    parameter_count: int | None
    epochs_ran: int
    best_epoch: int | None
    best_validation_loss: float | None
    train_loss_history: tuple[float, ...]
    validation_loss_history: tuple[float, ...]
    stopped_early: bool
    checkpoint_sha256: str | None
    error_type: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["train_loss_history"] = list(self.train_loss_history)
        payload["validation_loss_history"] = list(self.validation_loss_history)
        return payload


@dataclass(frozen=True)
class TunedRecurrentResult:
    """Selected model plus the complete auditable candidate panel."""

    model: RecurrentModel
    selected_candidate_id: str
    selected_config: RecurrentCandidate
    candidate_audits: tuple[CandidateAudit, ...]
    selection_metadata: dict[str, object]

    def audit_metadata(self) -> dict[str, object]:
        return {
            **self.selection_metadata,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_config": self.selected_config.to_dict(),
            "candidate_audits": [item.to_dict() for item in self.candidate_audits],
        }


@dataclass(frozen=True)
class RefitAudit:
    """Audit record for the selected candidate's full-development refit."""

    candidate_id: str
    config: RecurrentCandidate
    refit_seed: int
    initialization_rule: str
    epoch_rule: str
    data_scope: str
    test_data_used_for_refit: bool
    planned_epochs: int
    epochs_ran: int
    status: Literal["complete", "failed"]
    parameter_count: int | None
    train_loss_history: tuple[float, ...]
    development_name: str
    development_trial_count: int
    development_block_count: int
    development_data_fingerprint: str
    development_trial_multiset_fingerprint: str
    checkpoint_sha256: str | None
    error_type: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["train_loss_history"] = list(self.train_loss_history)
        return payload


@dataclass(frozen=True)
class RefitRecurrentResult:
    """Fresh model fitted on every inner-development block after selection."""

    model: RecurrentModel
    audit: RefitAudit

    def audit_metadata(self) -> dict[str, object]:
        return self.audit.to_dict()


class RefitFailedError(RuntimeError):
    """Raised with a serializable audit if full-development refitting fails."""

    def __init__(self, audit: RefitAudit) -> None:
        super().__init__(f"selected recurrent baseline refit failed: {audit.error}")
        self.audit = audit

    def audit_metadata(self) -> dict[str, object]:
        """Expose the failed refit audit through the shared driver contract."""

        return self.audit.to_dict()


class AllCandidatesFailedError(RuntimeError):
    """Raised only after retaining an audit record for every failed candidate."""

    def __init__(self, candidate_audits: Sequence[CandidateAudit]) -> None:
        super().__init__("all recurrent baseline candidates failed")
        self.candidate_audits = tuple(candidate_audits)

    def audit_metadata(self) -> dict[str, object]:
        return {
            "status": "failed",
            "candidate_audits": [item.to_dict() for item in self.candidate_audits],
        }


def _candidate_identity(config: RecurrentCandidate, root_seed: int) -> tuple[str, int]:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2s(encoded.encode("utf-8"), digest_size=8).hexdigest()
    # Hyperparameters of the same architecture receive an identical parameter
    # initialization and minibatch-order stream.  Validation therefore selects
    # optimization settings rather than a lucky initialization draw.
    initialization_family = f"{config.cell_type}:{config.hidden_size}"
    seed_digest = hashlib.blake2s(
        f"{root_seed}:{initialization_family}".encode("utf-8"), digest_size=4
    ).digest()
    return f"{config.cell_type}_{digest}", int.from_bytes(seed_digest, "little")


def _configure_determinism(device: torch.device) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_model(
    config: RecurrentCandidate,
    *,
    input_size: int,
    output_size: int,
) -> RecurrentModel:
    if config.cell_type == "rate_rnn":
        return RateRNNBaseline(
            input_size,
            config.hidden_size,
            output_size,
            leak=config.rate_leak,
        )
    return GRUBaseline(input_size, config.hidden_size, output_size)


def parameter_count(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def _masked_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if not torch.any(mask):
        raise ValueError("a training batch has no selected loss points")
    return torch.mean((prediction[mask] - target[mask]) ** 2)


@torch.no_grad()
def _dataset_loss(
    model: RecurrentModel,
    data: RecurrentSequenceData,
    device: torch.device,
) -> float:
    model.eval()
    inputs = torch.tensor(data.inputs, dtype=torch.float32, device=device)
    targets = torch.tensor(data.targets, dtype=torch.float32, device=device)
    mask = torch.tensor(data.loss_mask, dtype=torch.bool, device=device)
    prediction, _ = model(inputs)
    value = float(_masked_loss(prediction, targets, mask).detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError("masked validation loss is non-finite")
    return value


def evaluate_masked_mse(
    model: RecurrentModel,
    data: RecurrentSequenceData,
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate a fitted model without changing it or participating in tuning."""

    if not isinstance(data, RecurrentSequenceData):
        raise TypeError("data must be RecurrentSequenceData")
    torch_device = torch.device(device)
    original_device = next(model.parameters()).device
    model.to(torch_device)
    try:
        return _dataset_loss(model, data, torch_device)
    finally:
        model.to(original_device)


@torch.no_grad()
def predict_recurrent(
    model: RecurrentModel,
    inputs: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict after tuning; this function has no model-selection side effect."""

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("inputs must be a finite [trial, time, input] array")
    torch_device = torch.device(device)
    original_device = next(model.parameters()).device
    model.to(torch_device)
    model.eval()
    try:
        prediction, states = model(torch.tensor(values, device=torch_device))
        return prediction.cpu().numpy(), states.cpu().numpy()
    finally:
        model.to(original_device)


def _checkpoint_fingerprint(model: nn.Module) -> str:
    parts: list[bytes] = []
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous().numpy()
        parts.extend(
            [
                name.encode("utf-8"),
                str(values.dtype).encode("utf-8"),
                repr(values.shape).encode("utf-8"),
                values.tobytes(),
            ]
        )
    return _fingerprint_parts(parts)


def _training_tensors(
    data: RecurrentSequenceData, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(data.inputs, dtype=torch.float32, device=device),
        torch.tensor(data.targets, dtype=torch.float32, device=device),
        torch.tensor(data.loss_mask, dtype=torch.bool, device=device),
    )


def _train_one_epoch(
    model: RecurrentModel,
    optimizer: torch.optim.Optimizer,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    trial_count: int,
    batch_size: int,
    grad_clip: float,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    tensor_inputs, tensor_targets, tensor_mask = tensors
    model.train()
    order = torch.randperm(trial_count, generator=generator)
    squared_error_sum = 0.0
    selected_scalar_count = 0
    for start in range(0, trial_count, batch_size):
        batch = order[start : start + batch_size].to(device)
        batch_mask = tensor_mask[batch]
        if not torch.any(batch_mask):
            continue
        prediction, _ = model(tensor_inputs[batch])
        difference = prediction[batch_mask] - tensor_targets[batch][batch_mask]
        loss = torch.mean(difference**2)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        squared_error_sum += float(torch.sum(difference.detach() ** 2).cpu())
        selected_scalar_count += int(difference.numel())
    if selected_scalar_count == 0:
        raise ValueError("no training loss points were encountered")
    train_loss = squared_error_sum / selected_scalar_count
    if not np.isfinite(train_loss):
        raise FloatingPointError("epoch training loss is non-finite")
    return float(train_loss)


def _fit_candidate(
    training: RecurrentSequenceData,
    validation: RecurrentSequenceData,
    config: RecurrentCandidate,
    *,
    candidate_id: str,
    candidate_seed: int,
    device: torch.device,
) -> tuple[RecurrentModel | None, CandidateAudit]:
    train_history: list[float] = []
    validation_history: list[float] = []
    model: RecurrentModel | None = None
    count: int | None = None
    best_epoch: int | None = None
    best_validation_loss = np.inf
    stopped_early = False
    try:
        torch.manual_seed(candidate_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(candidate_seed)
        model = _make_model(
            config,
            input_size=training.inputs.shape[-1],
            output_size=training.targets.shape[-1],
        ).to(device)
        count = parameter_count(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        tensors = _training_tensors(training, device)
        generator = torch.Generator(device="cpu").manual_seed(candidate_seed)
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        for epoch in range(1, config.max_epochs + 1):
            train_loss = _train_one_epoch(
                model,
                optimizer,
                tensors,
                trial_count=training.trial_count,
                batch_size=config.batch_size,
                grad_clip=config.grad_clip,
                generator=generator,
                device=device,
            )
            validation_loss = _dataset_loss(model, validation, device)
            train_history.append(train_loss)
            validation_history.append(float(validation_loss))
            if validation_loss < best_validation_loss - config.min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    stopped_early = True
                    break
        if best_state is None or best_epoch is None:
            raise RuntimeError("no finite validation checkpoint was produced")
        model.load_state_dict(best_state)
        model.eval()
        restored_loss = _dataset_loss(model, validation, device)
        if not np.isclose(restored_loss, best_validation_loss, rtol=1e-6, atol=1e-8):
            raise RuntimeError(
                "restored checkpoint does not match the best validation loss"
            )
        audit = CandidateAudit(
            candidate_id=candidate_id,
            config=config,
            candidate_seed=candidate_seed,
            status="complete",
            selected=False,
            parameter_count=count,
            epochs_ran=len(train_history),
            best_epoch=best_epoch,
            best_validation_loss=float(best_validation_loss),
            train_loss_history=tuple(train_history),
            validation_loss_history=tuple(validation_history),
            stopped_early=stopped_early,
            checkpoint_sha256=_checkpoint_fingerprint(model),
            error_type=None,
            error=None,
        )
        return model, audit
    except Exception as error:
        audit = CandidateAudit(
            candidate_id=candidate_id,
            config=config,
            candidate_seed=candidate_seed,
            status="failed",
            selected=False,
            parameter_count=count,
            epochs_ran=len(train_history),
            best_epoch=best_epoch,
            best_validation_loss=(
                float(best_validation_loss)
                if np.isfinite(best_validation_loss)
                else None
            ),
            train_loss_history=tuple(train_history),
            validation_loss_history=tuple(validation_history),
            stopped_early=stopped_early,
            checkpoint_sha256=None,
            error_type=type(error).__name__,
            error=str(error),
        )
        return None, audit


def tune_recurrent_baseline(
    training: RecurrentSequenceData,
    validation: RecurrentSequenceData,
    candidates: Sequence[RecurrentCandidate],
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> TunedRecurrentResult:
    """Tune on disjoint inner blocks and restore the best candidate checkpoint.

    The function intentionally has no test-set argument.  Candidate selection
    is by minimum masked validation MSE, with parameter count and stable
    candidate ID as deterministic tie breakers.
    """

    if not isinstance(training, RecurrentSequenceData) or not isinstance(
        validation, RecurrentSequenceData
    ):
        raise TypeError("training and validation must be RecurrentSequenceData")
    if training.inputs.shape[-1] != validation.inputs.shape[-1]:
        raise ValueError("training/validation input dimensions differ")
    if training.targets.shape[-1] != validation.targets.shape[-1]:
        raise ValueError("training/validation output dimensions differ")
    overlap = set(training.block_tokens) & set(validation.block_tokens)
    if overlap:
        raise ValueError("training and validation blocks overlap")
    root_seed = _nonnegative_integer(seed, "seed")
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("at least one candidate is required")
    if not all(isinstance(item, RecurrentCandidate) for item in candidate_list):
        raise TypeError("every candidate must be RecurrentCandidate")
    encoded = [json.dumps(item.to_dict(), sort_keys=True) for item in candidate_list]
    if len(encoded) != len(set(encoded)):
        raise ValueError("candidate configurations must be unique")
    torch_device = torch.device(device)
    _configure_determinism(torch_device)

    audits: list[CandidateAudit] = []
    fitted: dict[str, RecurrentModel] = {}
    for config in candidate_list:
        candidate_id, candidate_seed = _candidate_identity(config, root_seed)
        try:
            model, audit = _fit_candidate(
                training,
                validation,
                config,
                candidate_id=candidate_id,
                candidate_seed=candidate_seed,
                device=torch_device,
            )
        except Exception as error:
            # Preserve even an unexpected candidate-runner failure rather than
            # aborting and silently losing the remainder of the grid.
            model = None
            audit = CandidateAudit(
                candidate_id=candidate_id,
                config=config,
                candidate_seed=candidate_seed,
                status="failed",
                selected=False,
                parameter_count=None,
                epochs_ran=0,
                best_epoch=None,
                best_validation_loss=None,
                train_loss_history=(),
                validation_loss_history=(),
                stopped_early=False,
                checkpoint_sha256=None,
                error_type=type(error).__name__,
                error=str(error),
            )
        audits.append(audit)
        if model is not None and audit.status == "complete":
            fitted[candidate_id] = model
    complete = [item for item in audits if item.status == "complete"]
    if not complete:
        raise AllCandidatesFailedError(audits)
    selected = min(
        complete,
        key=lambda item: (
            float(item.best_validation_loss),
            int(item.parameter_count),
            item.candidate_id,
        ),
    )
    audits = [
        replace(item, selected=item.candidate_id == selected.candidate_id)
        for item in audits
    ]
    model = fitted[selected.candidate_id]
    model.eval()
    metadata: dict[str, Any] = {
        "status": "complete",
        "root_seed": root_seed,
        "device": str(torch_device),
        "selection_metric": "masked_validation_mse",
        "selection_rule": (
            "minimum_validation_mse_then_parameter_count_then_candidate_id"
        ),
        "candidate_initialization_policy": ("shared_within_cell_type_and_hidden_size"),
        "selection_data_scope": "inner_validation_blocks_only",
        "test_data_used_for_selection": False,
        "train_name": training.name,
        "validation_name": validation.name,
        "train_trial_count": training.trial_count,
        "validation_trial_count": validation.trial_count,
        "inner_development_trial_count": (
            training.trial_count + validation.trial_count
        ),
        "train_block_count": training.block_count,
        "validation_block_count": validation.block_count,
        "inner_development_block_count": (
            training.block_count + validation.block_count
        ),
        "train_block_fingerprint": _block_fingerprint(training.block_ids),
        "validation_block_fingerprint": _block_fingerprint(validation.block_ids),
        "train_data_fingerprint": training.audit_fingerprint(),
        "validation_data_fingerprint": validation.audit_fingerprint(),
        "inner_development_trial_multiset_fingerprint": (
            _trial_multiset_fingerprint(training, validation)
        ),
        "input_size": int(training.inputs.shape[-1]),
        "output_size": int(training.targets.shape[-1]),
        "candidate_count": len(audits),
        "candidate_failure_count": sum(item.status == "failed" for item in audits),
        "selected_best_epoch": selected.best_epoch,
        "selected_validation_loss": selected.best_validation_loss,
        "selected_parameter_count": selected.parameter_count,
        "training_algorithm": model.checkpoint_metadata()["training_algorithm"],
        "used_autograd": True,
        "eligible_for_local_initialization": False,
    }
    return TunedRecurrentResult(
        model=model,
        selected_candidate_id=selected.candidate_id,
        selected_config=selected.config,
        candidate_audits=tuple(audits),
        selection_metadata=metadata,
    )


def refit_selected_recurrent_baseline(
    development: RecurrentSequenceData,
    tuning: TunedRecurrentResult,
    *,
    device: str | torch.device = "cpu",
) -> RefitRecurrentResult:
    """Freshly refit the selected candidate on all inner-development blocks.

    The preregistered epoch rule is exact: train for the selected candidate's
    one-based inner ``best_epoch``.  The supplied development set must be the
    exact trial multiset previously partitioned into inner train/validation;
    this both prevents an outer test set from entering refit and ensures the
    final baseline does not discard the validation blocks used for selection.
    """

    if not isinstance(development, RecurrentSequenceData):
        raise TypeError("development must be RecurrentSequenceData")
    if not isinstance(tuning, TunedRecurrentResult):
        raise TypeError("tuning must be TunedRecurrentResult")
    metadata = tuning.selection_metadata
    expected_trials = int(metadata["inner_development_trial_count"])
    expected_blocks = int(metadata["inner_development_block_count"])
    expected_fingerprint = str(metadata["inner_development_trial_multiset_fingerprint"])
    observed_fingerprint = _trial_multiset_fingerprint(development)
    if (
        development.trial_count != expected_trials
        or development.block_count != expected_blocks
        or observed_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "development must exactly equal the inner train/validation trial union"
        )
    if development.inputs.shape[-1] != int(metadata["input_size"]):
        raise ValueError("development input dimension differs from inner tuning")
    if development.targets.shape[-1] != int(metadata["output_size"]):
        raise ValueError("development output dimension differs from inner tuning")
    selected_audits = [item for item in tuning.candidate_audits if item.selected]
    if len(selected_audits) != 1:
        raise ValueError("tuning must contain exactly one selected candidate audit")
    selected = selected_audits[0]
    if (
        selected.status != "complete"
        or selected.candidate_id != tuning.selected_candidate_id
        or selected.config != tuning.selected_config
        or selected.best_epoch is None
    ):
        raise ValueError("selected tuning metadata is inconsistent")
    planned_epochs = _positive_integer(selected.best_epoch, "selected best_epoch")
    root_seed = _nonnegative_integer(metadata["root_seed"], "root_seed")
    encoded = json.dumps(
        tuning.selected_config.to_dict(), sort_keys=True, separators=(",", ":")
    )
    seed_digest = hashlib.blake2s(
        f"{root_seed}:full_development_refit:{encoded}".encode("utf-8"),
        digest_size=4,
    ).digest()
    refit_seed = int.from_bytes(seed_digest, "little")
    torch_device = torch.device(device)
    _configure_determinism(torch_device)
    history: list[float] = []
    model: RecurrentModel | None = None
    count: int | None = None
    development_data_fingerprint = development.audit_fingerprint()
    try:
        torch.manual_seed(refit_seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(refit_seed)
        model = _make_model(
            tuning.selected_config,
            input_size=development.inputs.shape[-1],
            output_size=development.targets.shape[-1],
        ).to(torch_device)
        count = parameter_count(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tuning.selected_config.learning_rate,
            weight_decay=tuning.selected_config.weight_decay,
        )
        tensors = _training_tensors(development, torch_device)
        generator = torch.Generator(device="cpu").manual_seed(refit_seed)
        for _ in range(planned_epochs):
            history.append(
                _train_one_epoch(
                    model,
                    optimizer,
                    tensors,
                    trial_count=development.trial_count,
                    batch_size=tuning.selected_config.batch_size,
                    grad_clip=tuning.selected_config.grad_clip,
                    generator=generator,
                    device=torch_device,
                )
            )
        model.eval()
        audit = RefitAudit(
            candidate_id=tuning.selected_candidate_id,
            config=tuning.selected_config,
            refit_seed=refit_seed,
            initialization_rule="fresh_parameters_from_deterministic_refit_seed",
            epoch_rule="exact_selected_inner_best_epoch",
            data_scope="exact_inner_train_validation_trial_union",
            test_data_used_for_refit=False,
            planned_epochs=planned_epochs,
            epochs_ran=len(history),
            status="complete",
            parameter_count=count,
            train_loss_history=tuple(history),
            development_name=development.name,
            development_trial_count=development.trial_count,
            development_block_count=development.block_count,
            development_data_fingerprint=development_data_fingerprint,
            development_trial_multiset_fingerprint=observed_fingerprint,
            checkpoint_sha256=_checkpoint_fingerprint(model),
            error_type=None,
            error=None,
        )
        return RefitRecurrentResult(model=model, audit=audit)
    except Exception as error:
        audit = RefitAudit(
            candidate_id=tuning.selected_candidate_id,
            config=tuning.selected_config,
            refit_seed=refit_seed,
            initialization_rule="fresh_parameters_from_deterministic_refit_seed",
            epoch_rule="exact_selected_inner_best_epoch",
            data_scope="exact_inner_train_validation_trial_union",
            test_data_used_for_refit=False,
            planned_epochs=planned_epochs,
            epochs_ran=len(history),
            status="failed",
            parameter_count=count,
            train_loss_history=tuple(history),
            development_name=development.name,
            development_trial_count=development.trial_count,
            development_block_count=development.block_count,
            development_data_fingerprint=development_data_fingerprint,
            development_trial_multiset_fingerprint=observed_fingerprint,
            checkpoint_sha256=None,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise RefitFailedError(audit) from error
