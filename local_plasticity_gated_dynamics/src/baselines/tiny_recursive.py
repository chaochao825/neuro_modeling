"""Small recursive reasoning baselines with an explicit non-local boundary.

The implementation is a deliberately reduced, independently written model
inspired by the update schedule of the Tiny Recursion Model (TRM).  It is not
the official 5M/7M checkpoint and does not implement ACT, puzzle embeddings,
StableMax, or the official transductive ARC protocol.  The only intended use
is as a BPTT baseline for controlled mechanism tests in this repository.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ReasoningMode = Literal["trm_like", "single_state_core_call_matched"]
TrainingLossScope = Literal["blank_only", "all_tokens"]
CheckpointMetric = Literal["validation_loss", "blank_cell_accuracy"]


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


@dataclass(frozen=True)
class TinyRecursiveConfig:
    """Architecture shared by the two parameter/compute-matched conditions."""

    seq_len: int = 81
    vocab_size: int = 10
    hidden_size: int = 64
    num_heads: int = 4
    layers: int = 1
    expansion: float = 2.0
    high_cycles: int = 2
    low_cycles: int = 2
    supervision_steps: int = 2
    mode: ReasoningMode = "trm_like"

    def __post_init__(self) -> None:
        for name in (
            "seq_len",
            "vocab_size",
            "hidden_size",
            "num_heads",
            "layers",
            "high_cycles",
            "low_cycles",
            "supervision_steps",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        expansion = _finite_float(self.expansion, "expansion")
        if expansion <= 0.0:
            raise ValueError("expansion must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mode not in {"trm_like", "single_state_core_call_matched"}:
            raise ValueError("unknown tiny-recursive mode")
        object.__setattr__(self, "expansion", expansion)

    @property
    def core_calls_per_segment(self) -> int:
        """Shared-core calls in one detached-carry supervision segment."""

        return 2 * self.high_cycles * self.low_cycles

    @property
    def core_calls(self) -> int:
        """Shared-core calls in one fixed-step evaluation trajectory."""

        return self.supervision_steps * self.core_calls_per_segment


class _ReasoningBlock(nn.Module):
    """A compact non-causal attention/MLP block with post normalization."""

    def __init__(self, config: TinyRecursiveConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        expanded = max(1, int(round(config.hidden_size * config.expansion)))
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, expanded),
            nn.SiLU(),
            nn.Linear(expanded, config.hidden_size),
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            states,
            states,
            states,
            need_weights=False,
        )
        states = self.attention_norm(states + attended)
        return self.mlp_norm(states + self.mlp(states))


class _SharedReasoningCore(nn.Module):
    def __init__(self, config: TinyRecursiveConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_ReasoningBlock(config) for _ in range(config.layers)]
        )

    def forward(
        self, states: torch.Tensor, input_injection: torch.Tensor
    ) -> torch.Tensor:
        states = states + input_injection
        for block in self.blocks:
            states = block(states)
        return states


@dataclass(frozen=True)
class TinyRecursiveCarry:
    """Detached answer/reasoning state passed between supervision segments."""

    answer: torch.Tensor
    latent: torch.Tensor


@dataclass(frozen=True)
class TinyRecursiveOutput:
    """Final and deep-supervision outputs from one fixed-compute pass."""

    logits: torch.Tensor
    cycle_logits: tuple[torch.Tensor, ...]
    answer_states: tuple[torch.Tensor, ...]
    latent_state: torch.Tensor
    carry: TinyRecursiveCarry
    core_calls_per_segment: int


class TinyRecursiveBaseline(nn.Module):
    """Micro-TRM-like baseline and its matched single-state comparator.

    ``trm_like`` alternates updates of a latent reasoning state ``z`` and an
    answer state ``y`` through one shared core. The
    ``single_state_core_call_matched`` mode runs the identical core for the
    identical nominal number of calls while keeping only one evolving answer
    state. Both modes have the same trainable parameterization and nominal
    recurrent-block call budget; this is not a claim of matched backward FLOPs,
    memory, wall time, or physical energy.
    """

    training_algorithm = "bptt_tiny_recursive_baseline"
    uses_bptt = True
    eligible_for_local_initialization = False

    def __init__(self, config: TinyRecursiveConfig) -> None:
        super().__init__()
        if not isinstance(config, TinyRecursiveConfig):
            raise TypeError("config must be TinyRecursiveConfig")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.seq_len, config.hidden_size)
        )
        self.register_buffer(
            "answer_initial", torch.empty(1, config.seq_len, config.hidden_size)
        )
        self.register_buffer(
            "latent_initial", torch.empty(1, config.seq_len, config.hidden_size)
        )
        self.core = _SharedReasoningCore(config)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.answer_initial, std=0.02)
        nn.init.normal_(self.latent_initial, std=0.02)

    def _validate_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 2 or tokens.shape[1] != self.config.seq_len:
            raise ValueError(f"tokens must have shape [batch, {self.config.seq_len}]")
        if tokens.dtype == torch.bool or tokens.dtype.is_floating_point:
            raise TypeError("tokens must use an integer dtype")
        tokens = tokens.to(dtype=torch.long)
        if tokens.numel() and (
            int(tokens.min()) < 0 or int(tokens.max()) >= self.config.vocab_size
        ):
            raise ValueError("tokens contain an out-of-vocabulary value")
        return tokens

    def initial_carry(self, batch_size: int) -> TinyRecursiveCarry:
        batch_size = _positive_integer(batch_size, "batch_size")
        return TinyRecursiveCarry(
            answer=self.answer_initial.expand(batch_size, -1, -1),
            latent=self.latent_initial.expand(batch_size, -1, -1),
        )

    def _cycle(
        self,
        answer: torch.Tensor,
        latent: torch.Tensor,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.mode == "trm_like":
            for _low_step in range(self.config.low_cycles):
                latent = self.core(latent, answer + encoded)
                answer = self.core(answer, latent)
        else:
            for _call in range(2 * self.config.low_cycles):
                answer = self.core(answer, encoded + latent)
        return answer, latent

    def forward(
        self,
        tokens: torch.Tensor,
        carry: TinyRecursiveCarry | None = None,
    ) -> TinyRecursiveOutput:
        tokens = self._validate_tokens(tokens)
        batch_size = tokens.shape[0]
        encoded = self.token_embedding(tokens) + self.position_embedding
        if carry is None:
            carry = self.initial_carry(batch_size)
        elif (
            carry.answer.shape
            != (batch_size, self.config.seq_len, self.config.hidden_size)
            or carry.latent.shape != carry.answer.shape
        ):
            raise ValueError("carry shape does not match the token batch")
        answer, latent = carry.answer, carry.latent
        cycle_logits: list[torch.Tensor] = []
        answer_states: list[torch.Tensor] = []

        # Match the official TRM gradient boundary: the prefix outer cycles
        # update the carry without a graph, while every low-cycle update in the
        # final outer cycle participates in BPTT.
        with torch.no_grad():
            for _high_step in range(self.config.high_cycles - 1):
                answer, latent = self._cycle(answer, latent, encoded)
                answer_states.append(answer)
                cycle_logits.append(self.output_head(answer))
        answer, latent = self._cycle(answer, latent, encoded)
        answer_states.append(answer)
        cycle_logits.append(self.output_head(answer))
        new_carry = TinyRecursiveCarry(answer=answer.detach(), latent=latent.detach())

        return TinyRecursiveOutput(
            logits=cycle_logits[-1],
            cycle_logits=tuple(cycle_logits),
            answer_states=tuple(answer_states),
            latent_state=latent,
            carry=new_carry,
            core_calls_per_segment=self.config.core_calls_per_segment,
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "training_algorithm": self.training_algorithm,
            "uses_bptt": self.uses_bptt,
            "used_autograd": True,
            "eligible_for_local_initialization": False,
            "architecture_family": "micro_trm_like_independent_reimplementation",
            "reasoning_mode": self.config.mode,
            "core_calls_per_segment": self.config.core_calls_per_segment,
            "supervision_steps": self.config.supervision_steps,
            "core_calls_per_evaluation": self.config.core_calls,
            "gradient_schedule": "no_grad_prefix_last_outer_cycle_bptt",
            "carry_detached_between_supervision_steps": True,
            "official_checkpoint_compatible": False,
            "official_arc_protocol": False,
            "act_enabled": False,
            "puzzle_embeddings_enabled": False,
            "stablemax_enabled": False,
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class TinyRecursiveTrainingConfig:
    """Fixed-budget optimizer settings; test data are absent by construction."""

    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cpu"
    loss_scope: TrainingLossScope = "blank_only"
    checkpoint_metric: CheckpointMetric = "validation_loss"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in (
            "learning_rate",
            "weight_decay",
            "grad_clip",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.learning_rate <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if self.loss_scope not in {"blank_only", "all_tokens"}:
            raise ValueError("loss_scope must be 'blank_only' or 'all_tokens'")
        if self.checkpoint_metric not in {
            "validation_loss",
            "blank_cell_accuracy",
        }:
            raise ValueError(
                "checkpoint_metric must be 'validation_loss' or "
                "'blank_cell_accuracy'"
            )


@dataclass(frozen=True)
class TinyRecursiveFitReceipt:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    validation_exact_accuracy: tuple[float, ...]
    validation_blank_cell_accuracy: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    best_validation_blank_cell_accuracy: float
    selected_validation_loss: float
    optimizer_steps: int
    checkpoint_sha256: str
    training_seed: int
    train_examples: int
    validation_examples: int
    training_data_sha256: str
    validation_data_sha256: str
    epoch_permutation_sha256: str
    loss_scope: TrainingLossScope
    checkpoint_metric: CheckpointMetric
    selected_train_blank_cell_accuracy: float
    selected_train_full_cell_accuracy: float
    selected_train_raw_clue_accuracy: float
    selected_train_exact_accuracy: float
    selected_validation_blank_cell_accuracy: float
    selected_validation_full_cell_accuracy: float
    selected_validation_raw_clue_accuracy: float
    selected_validation_exact_accuracy: float
    blank_only_loss: bool
    test_data_used_for_fit: bool = False
    fixed_training_budget: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_sha256(model: nn.Module) -> str:
    """Hash a CPU state dict without relying on pickle file metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _array_pair_sha256(inputs: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (inputs, targets):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validated_arrays(
    inputs: object,
    targets: object,
    *,
    seq_len: int,
    vocab_size: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    input_array = np.asarray(inputs)
    target_array = np.asarray(targets)
    expected = (input_array.shape[0], seq_len) if input_array.ndim == 2 else None
    if input_array.ndim != 2 or target_array.shape != expected:
        raise ValueError(f"{name} inputs/targets must have shape [example, {seq_len}]")
    if input_array.shape[0] < 1 or input_array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} inputs must be a non-empty integer array")
    if target_array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} targets must be integers")
    input_array = np.asarray(input_array, dtype=np.int64)
    target_array = np.asarray(target_array, dtype=np.int64)
    if (
        np.any((input_array < 0) | (input_array >= vocab_size))
        or np.any((target_array < 1) | (target_array >= vocab_size))
        or np.any(np.all(input_array > 0, axis=1))
    ):
        raise ValueError(f"{name} contains invalid tokens or a board without blanks")
    given = input_array > 0
    if not np.array_equal(input_array[given], target_array[given]):
        raise ValueError(f"{name} targets must preserve all visible input tokens")
    return input_array.copy(), target_array.copy()


def _reasoning_loss(
    output: TinyRecursiveOutput,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    scope: TrainingLossScope,
) -> torch.Tensor:
    blank = inputs.eq(0)

    def one(logits: torch.Tensor) -> torch.Tensor:
        # Blank is an input symbol, not a valid Sudoku answer class.
        adjusted = torch.cat(
            (torch.full_like(logits[..., :1], -1e4), logits[..., 1:]), dim=-1
        )
        losses = F.cross_entropy(
            adjusted.reshape(-1, adjusted.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        return losses[blank].mean() if scope == "blank_only" else losses.mean()

    return one(output.logits)


@torch.no_grad()
def predict_tiny_recursive(
    model: TinyRecursiveBaseline,
    inputs: object,
    *,
    batch_size: int = 128,
    device: str | torch.device | None = None,
    clamp_visible_tokens: bool = True,
) -> np.ndarray:
    """Predict full boards and deterministically clamp the visible clues."""

    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(clamp_visible_tokens, (bool, np.bool_)):
        raise TypeError("clamp_visible_tokens must be boolean")
    values = np.asarray(inputs)
    if values.ndim != 2 or values.shape[1] != model.config.seq_len:
        raise ValueError("inputs have the wrong sequence shape")
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError("inputs must be integer tokens")
    values = np.array(values, dtype=np.int64, copy=True)
    target_device = (
        next(model.parameters()).device if device is None else torch.device(device)
    )
    was_training = model.training
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch_array = values[start : start + batch_size]
        batch = torch.as_tensor(batch_array, dtype=torch.long, device=target_device)
        carry = None
        output = None
        for _step in range(model.config.supervision_steps):
            output = model(batch, carry)
            carry = output.carry
        if output is None:
            raise AssertionError(
                "validated supervision_steps unexpectedly produced no output"
            )
        logits = output.logits
        adjusted = torch.cat(
            (torch.full_like(logits[..., :1], -1e4), logits[..., 1:]), dim=-1
        )
        predicted = adjusted.argmax(dim=-1).cpu().numpy()
        if clamp_visible_tokens:
            predicted[batch_array > 0] = batch_array[batch_array > 0]
        predictions.append(predicted)
    model.train(was_training)
    return np.concatenate(predictions, axis=0)


def _exact_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.all(predictions == targets, axis=1)))


def _sudoku_prediction_metrics(
    model: TinyRecursiveBaseline,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Return task-macro diagnostics without treating cells as replicates."""

    raw = predict_tiny_recursive(
        model,
        inputs,
        batch_size=batch_size,
        device=device,
        clamp_visible_tokens=False,
    )
    blank = inputs == 0
    visible = ~blank
    blank_accuracy = np.mean(
        [
            float(np.mean(prediction[mask] == target[mask]))
            for prediction, target, mask in zip(raw, targets, blank, strict=True)
        ]
    )
    clue_accuracy = np.mean(
        [
            float(np.mean(prediction[mask] == target[mask]))
            for prediction, target, mask in zip(raw, targets, visible, strict=True)
        ]
    )
    full_accuracy = float(np.mean(np.mean(raw == targets, axis=1)))
    clamped = raw.copy()
    clamped[visible] = inputs[visible]
    return {
        "blank_cell_accuracy": float(blank_accuracy),
        "full_cell_accuracy": full_accuracy,
        "raw_clue_accuracy": float(clue_accuracy),
        "exact_accuracy": _exact_accuracy(clamped, targets),
    }


def fit_tiny_recursive(
    model: TinyRecursiveBaseline,
    training_inputs: object,
    training_targets: object,
    validation_inputs: object,
    validation_targets: object,
    config: TinyRecursiveTrainingConfig,
    *,
    seed: int,
) -> TinyRecursiveFitReceipt:
    """Fit on train and select checkpoints on validation; no test API exists."""

    if not isinstance(model, TinyRecursiveBaseline):
        raise TypeError("model must be TinyRecursiveBaseline")
    if not isinstance(config, TinyRecursiveTrainingConfig):
        raise TypeError("config must be TinyRecursiveTrainingConfig")
    seed = _nonnegative_integer(seed, "seed")
    train_x, train_y = _validated_arrays(
        training_inputs,
        training_targets,
        seq_len=model.config.seq_len,
        vocab_size=model.config.vocab_size,
        name="training",
    )
    validation_x, validation_y = _validated_arrays(
        validation_inputs,
        validation_targets,
        seq_len=model.config.seq_len,
        vocab_size=model.config.vocab_size,
        name="validation",
    )
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    validation_accuracy_history: list[float] = []
    validation_blank_accuracy_history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_score = -float("inf")
    selected_validation_loss = float("inf")
    best_epoch = 0
    optimizer_steps = 0
    permutation_digest = hashlib.sha256()

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator).numpy()
        permutation_digest.update(np.ascontiguousarray(permutation).tobytes())
        epoch_losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            inputs = torch.as_tensor(train_x[indices], dtype=torch.long, device=device)
            targets = torch.as_tensor(train_y[indices], dtype=torch.long, device=device)
            carry = None
            for _step in range(model.config.supervision_steps):
                optimizer.zero_grad(set_to_none=True)
                output = model(inputs, carry)
                carry = output.carry
                loss = _reasoning_loss(
                    output,
                    inputs,
                    targets,
                    scope=config.loss_scope,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite tiny-recursive training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer_steps += 1
                epoch_losses.append(float(loss.detach().cpu()))
        train_loss_history.append(float(np.mean(epoch_losses)))

        model.eval()
        with torch.no_grad():
            validation_inputs_tensor = torch.as_tensor(
                validation_x, dtype=torch.long, device=device
            )
            validation_targets_tensor = torch.as_tensor(
                validation_y, dtype=torch.long, device=device
            )
            validation_carry = None
            validation_output = None
            for _step in range(model.config.supervision_steps):
                validation_output = model(validation_inputs_tensor, validation_carry)
                validation_carry = validation_output.carry
            if validation_output is None:
                raise AssertionError("validated supervision_steps produced no output")
            validation_loss = _reasoning_loss(
                validation_output,
                validation_inputs_tensor,
                validation_targets_tensor,
                scope=config.loss_scope,
            )
        validation_value = float(validation_loss.cpu())
        validation_loss_history.append(validation_value)
        validation_metrics = _sudoku_prediction_metrics(
            model,
            validation_x,
            validation_y,
            batch_size=config.batch_size,
            device=device,
        )
        validation_accuracy_history.append(validation_metrics["exact_accuracy"])
        validation_blank_accuracy_history.append(
            validation_metrics["blank_cell_accuracy"]
        )
        checkpoint_score = (
            -validation_value
            if config.checkpoint_metric == "validation_loss"
            else validation_metrics["blank_cell_accuracy"]
        )
        if checkpoint_score > best_checkpoint_score or (
            checkpoint_score == best_checkpoint_score
            and validation_value < selected_validation_loss
        ):
            best_checkpoint_score = checkpoint_score
            selected_validation_loss = validation_value
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    selected_train_metrics = _sudoku_prediction_metrics(
        model,
        train_x,
        train_y,
        batch_size=config.batch_size,
        device=device,
    )
    selected_validation_metrics = _sudoku_prediction_metrics(
        model,
        validation_x,
        validation_y,
        batch_size=config.batch_size,
        device=device,
    )
    return TinyRecursiveFitReceipt(
        train_loss=tuple(train_loss_history),
        validation_loss=tuple(validation_loss_history),
        validation_exact_accuracy=tuple(validation_accuracy_history),
        validation_blank_cell_accuracy=tuple(validation_blank_accuracy_history),
        best_epoch=best_epoch,
        best_validation_loss=float(min(validation_loss_history)),
        best_validation_blank_cell_accuracy=float(
            max(validation_blank_accuracy_history)
        ),
        selected_validation_loss=selected_validation_loss,
        optimizer_steps=optimizer_steps,
        checkpoint_sha256=state_dict_sha256(model),
        training_seed=seed,
        train_examples=len(train_x),
        validation_examples=len(validation_x),
        training_data_sha256=_array_pair_sha256(train_x, train_y),
        validation_data_sha256=_array_pair_sha256(validation_x, validation_y),
        epoch_permutation_sha256=permutation_digest.hexdigest(),
        loss_scope=config.loss_scope,
        checkpoint_metric=config.checkpoint_metric,
        selected_train_blank_cell_accuracy=selected_train_metrics[
            "blank_cell_accuracy"
        ],
        selected_train_full_cell_accuracy=selected_train_metrics[
            "full_cell_accuracy"
        ],
        selected_train_raw_clue_accuracy=selected_train_metrics[
            "raw_clue_accuracy"
        ],
        selected_train_exact_accuracy=selected_train_metrics["exact_accuracy"],
        selected_validation_blank_cell_accuracy=selected_validation_metrics[
            "blank_cell_accuracy"
        ],
        selected_validation_full_cell_accuracy=selected_validation_metrics[
            "full_cell_accuracy"
        ],
        selected_validation_raw_clue_accuracy=selected_validation_metrics[
            "raw_clue_accuracy"
        ],
        selected_validation_exact_accuracy=selected_validation_metrics[
            "exact_accuracy"
        ],
        blank_only_loss=config.loss_scope == "blank_only",
    )
