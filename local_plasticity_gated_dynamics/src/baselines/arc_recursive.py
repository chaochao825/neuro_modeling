"""A faithful-small recursive ARC baseline with demonstration-only TTA.

The update ordering and gradient boundary follow the public TRM mechanism,
but this is an independent, reduced implementation rather than the official
7M model or checkpoint.  It is a BPTT baseline only: no component is used to
initialize or train the repository's local-learning main models.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.arc_recursive_dataset import (
    ARC_COLORS,
    ARC_PAD_TOKEN,
    ARC_TARGET_IGNORE,
    ARCGridExamples,
    pack_arc_grid,
    public_arc_support_examples,
    seeded_arc_transforms,
    unpack_arc_grid,
)
from src.data.structured_protocol import PublicTask


ARCReasoningMode = Literal["trm_like", "single_state_core_call_matched"]
ARCAdaptScope = Literal["full", "puzzle_only"]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ARCRecursiveConfig:
    """Architecture shared by recursive and core-call-matched controls."""

    max_grid_size: int = 30
    hidden_size: int = 128
    num_heads: int = 4
    layers: int = 2
    expansion: float = 4.0
    high_cycles: int = 3
    low_cycles: int = 4
    supervision_steps: int = 4
    num_puzzle_embeddings: int = 1
    mode: ARCReasoningMode = "trm_like"

    def __post_init__(self) -> None:
        for name in (
            "max_grid_size",
            "hidden_size",
            "num_heads",
            "layers",
            "high_cycles",
            "low_cycles",
            "supervision_steps",
            "num_puzzle_embeddings",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        expansion = _finite_float(self.expansion, name="expansion")
        if expansion <= 0.0:
            raise ValueError("expansion must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mode not in {"trm_like", "single_state_core_call_matched"}:
            raise ValueError("unknown ARC recursive mode")
        object.__setattr__(self, "expansion", expansion)

    @property
    def seq_len(self) -> int:
        return self.max_grid_size * self.max_grid_size

    @property
    def core_calls_per_segment(self) -> int:
        return 2 * self.high_cycles * self.low_cycles

    @property
    def core_calls_per_prediction(self) -> int:
        return self.supervision_steps * self.core_calls_per_segment


class _ARCReasoningBlock(nn.Module):
    def __init__(self, config: ARCRecursiveConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        expanded = max(1, round(config.hidden_size * config.expansion))
        self.gate = nn.Linear(config.hidden_size, 2 * expanded, bias=False)
        self.project = nn.Linear(expanded, config.hidden_size, bias=False)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self, states: torch.Tensor, *, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attention(
            states,
            states,
            states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        states = self.attention_norm(states + attended)
        left, right = self.gate(states).chunk(2, dim=-1)
        update = self.project(F.silu(left) * right)
        return self.mlp_norm(states + update)


class _ARCSharedCore(nn.Module):
    def __init__(self, config: ARCRecursiveConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_ARCReasoningBlock(config) for _ in range(config.layers)]
        )

    def forward(
        self,
        states: torch.Tensor,
        injection: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        states = states + injection
        for block in self.blocks:
            states = block(states, key_padding_mask=key_padding_mask)
        return states


@dataclass(frozen=True)
class ARCRecursiveCarry:
    answer: torch.Tensor
    latent: torch.Tensor


@dataclass(frozen=True)
class ARCRecursiveOutput:
    cell_logits: torch.Tensor
    height_logits: torch.Tensor
    width_logits: torch.Tensor
    carry: ARCRecursiveCarry
    answer_state: torch.Tensor
    latent_state: torch.Tensor
    core_calls_per_segment: int


class ARCRecursiveBaseline(nn.Module):
    """Direct variable-shape ARC decoder with official-order recursion."""

    training_algorithm = "bptt_faithful_small_trm_arc_baseline"
    uses_bptt = True
    eligible_for_local_initialization = False

    def __init__(self, config: ARCRecursiveConfig) -> None:
        super().__init__()
        if not isinstance(config, ARCRecursiveConfig):
            raise TypeError("config must be ARCRecursiveConfig")
        self.config = config
        hidden = config.hidden_size
        self.token_embedding = nn.Embedding(ARC_PAD_TOKEN + 1, hidden)
        self.row_embedding = nn.Parameter(
            torch.empty(1, config.max_grid_size, hidden)
        )
        self.column_embedding = nn.Parameter(
            torch.empty(1, config.max_grid_size, hidden)
        )
        self.puzzle_embedding = nn.Embedding(config.num_puzzle_embeddings, hidden)
        self.novel_puzzle_embedding = nn.Parameter(torch.empty(hidden))
        self.register_buffer(
            "answer_initial", torch.empty(1, config.seq_len, hidden)
        )
        self.register_buffer(
            "latent_initial", torch.empty(1, config.seq_len, hidden)
        )
        self.core = _ARCSharedCore(config)
        self.cell_head = nn.Linear(hidden, ARC_COLORS, bias=False)
        self.height_head = nn.Linear(hidden, config.max_grid_size)
        self.width_head = nn.Linear(hidden, config.max_grid_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.column_embedding, std=0.02)
        nn.init.normal_(self.novel_puzzle_embedding, std=0.02)
        nn.init.normal_(self.answer_initial, std=0.02)
        nn.init.normal_(self.latent_initial, std=0.02)

    def _tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 2 or tokens.shape[1] != self.config.seq_len:
            raise ValueError(
                f"tokens must have shape [batch, {self.config.seq_len}]"
            )
        if tokens.dtype == torch.bool or tokens.dtype.is_floating_point:
            raise TypeError("tokens must use an integer dtype")
        tokens = tokens.to(dtype=torch.long)
        if tokens.numel() and (
            int(tokens.min()) < 0 or int(tokens.max()) > ARC_PAD_TOKEN
        ):
            raise ValueError("tokens contain an invalid ARC symbol")
        if torch.any(tokens.ne(ARC_PAD_TOKEN).sum(dim=1).eq(0)):
            raise ValueError("every ARC input must contain at least one grid cell")
        return tokens

    def initial_carry(self, batch_size: int) -> ARCRecursiveCarry:
        batch_size = _positive_integer(batch_size, name="batch_size")
        return ARCRecursiveCarry(
            answer=self.answer_initial.expand(batch_size, -1, -1),
            latent=self.latent_initial.expand(batch_size, -1, -1),
        )

    def _puzzle_vector(
        self,
        batch_size: int,
        *,
        puzzle_ids: torch.Tensor | None,
        puzzle_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        if puzzle_ids is not None and puzzle_vector is not None:
            raise ValueError("pass puzzle_ids or puzzle_vector, not both")
        if puzzle_ids is not None:
            if puzzle_ids.shape != (batch_size,) or puzzle_ids.dtype == torch.bool:
                raise ValueError("puzzle_ids must have shape [batch]")
            puzzle_ids = puzzle_ids.to(dtype=torch.long)
            if puzzle_ids.numel() and (
                int(puzzle_ids.min()) < 0
                or int(puzzle_ids.max()) >= self.config.num_puzzle_embeddings
            ):
                raise ValueError("puzzle_ids contain an unknown task")
            return self.puzzle_embedding(puzzle_ids)
        if puzzle_vector is None:
            puzzle_vector = self.novel_puzzle_embedding
        if puzzle_vector.ndim == 1:
            if puzzle_vector.shape != (self.config.hidden_size,):
                raise ValueError("puzzle_vector has the wrong hidden dimension")
            return puzzle_vector.unsqueeze(0).expand(batch_size, -1)
        if puzzle_vector.shape != (batch_size, self.config.hidden_size):
            raise ValueError("batched puzzle_vector has the wrong shape")
        return puzzle_vector

    def _cycle(
        self,
        answer: torch.Tensor,
        latent: torch.Tensor,
        encoded: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.mode == "trm_like":
            for _low_step in range(self.config.low_cycles):
                latent = self.core(
                    latent,
                    answer + encoded,
                    key_padding_mask=key_padding_mask,
                )
                answer = self.core(
                    answer,
                    latent,
                    key_padding_mask=key_padding_mask,
                )
        else:
            for _call in range(2 * self.config.low_cycles):
                answer = self.core(
                    answer,
                    encoded + latent,
                    key_padding_mask=key_padding_mask,
                )
        return answer, latent

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        puzzle_ids: torch.Tensor | None = None,
        puzzle_vector: torch.Tensor | None = None,
        carry: ARCRecursiveCarry | None = None,
    ) -> ARCRecursiveOutput:
        tokens = self._tokens(tokens)
        batch_size = tokens.shape[0]
        mask = tokens.eq(ARC_PAD_TOKEN)
        positions = (
            self.row_embedding[:, :, None, :] + self.column_embedding[:, None, :, :]
        ).reshape(1, self.config.seq_len, self.config.hidden_size)
        task_vector = self._puzzle_vector(
            batch_size, puzzle_ids=puzzle_ids, puzzle_vector=puzzle_vector
        )
        encoded = (
            self.token_embedding(tokens)
            + positions
            + task_vector[:, None, :]
        )
        if carry is None:
            carry = self.initial_carry(batch_size)
        expected = (batch_size, self.config.seq_len, self.config.hidden_size)
        if carry.answer.shape != expected or carry.latent.shape != expected:
            raise ValueError("carry shape does not match tokens")
        answer, latent = carry.answer, carry.latent
        with torch.no_grad():
            for _high_step in range(self.config.high_cycles - 1):
                answer, latent = self._cycle(
                    answer, latent, encoded, key_padding_mask=mask
                )
        answer, latent = self._cycle(
            answer, latent, encoded, key_padding_mask=mask
        )
        valid = (~mask).to(dtype=answer.dtype)
        pooled = (answer * valid[..., None]).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return ARCRecursiveOutput(
            cell_logits=self.cell_head(answer),
            height_logits=self.height_head(pooled),
            width_logits=self.width_head(pooled),
            carry=ARCRecursiveCarry(answer.detach(), latent.detach()),
            answer_state=answer,
            latent_state=latent,
            core_calls_per_segment=self.config.core_calls_per_segment,
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "training_algorithm": self.training_algorithm,
            "uses_bptt": True,
            "used_autograd": True,
            "eligible_for_local_initialization": False,
            "architecture_family": "faithful_small_trm_arc_independent",
            "protocol": "demo_tta",
            "official_checkpoint_compatible": False,
            "official_7m_reproduction": False,
            "act_enabled": False,
            "stablemax_enabled": False,
            "puzzle_embeddings_enabled": True,
            "two_dimensional_positions": True,
            "gradient_schedule": "no_grad_prefix_last_high_cycle_bptt",
            "update_order": "latent_then_answer_per_low_cycle",
            "carry_detached_between_supervision_steps": True,
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class ARCRecursiveTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    puzzle_learning_rate: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        for name in (
            "learning_rate",
            "puzzle_learning_rate",
            "weight_decay",
            "grad_clip",
        ):
            value = _finite_float(getattr(self, name), name=name)
            if name == "weight_decay" and value < 0.0:
                raise ValueError("weight_decay must be non-negative")
            if name != "weight_decay" and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be non-empty")


@dataclass(frozen=True)
class ARCRecursiveFitReceipt:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    optimizer_steps: int
    training_seed: int
    training_examples: int
    validation_examples: int
    training_data_sha256: str
    validation_data_sha256: str
    epoch_permutation_sha256: str
    checkpoint_sha256: str
    test_data_used_for_fit: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _examples_sha256(examples: ARCGridExamples) -> str:
    digest = hashlib.sha256()
    for value in (
        examples.inputs,
        examples.targets,
        examples.input_shapes,
        examples.target_shapes,
        examples.puzzle_indices,
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    for value in examples.example_ids:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _validate_examples(
    model: ARCRecursiveBaseline, examples: ARCGridExamples, *, training: bool
) -> None:
    if not isinstance(examples, ARCGridExamples):
        raise TypeError("examples must be ARCGridExamples")
    if examples.max_grid_size != model.config.max_grid_size:
        raise ValueError("example packing does not match the model")
    if training and int(np.max(examples.puzzle_indices)) >= (
        model.config.num_puzzle_embeddings
    ):
        raise ValueError("training examples require unavailable puzzle embeddings")


def _arc_loss(
    output: ARCRecursiveOutput,
    targets: torch.Tensor,
    target_shapes: torch.Tensor,
) -> torch.Tensor:
    cell = F.cross_entropy(
        output.cell_logits.reshape(-1, ARC_COLORS),
        targets.reshape(-1),
        ignore_index=ARC_TARGET_IGNORE,
    )
    height = F.cross_entropy(output.height_logits, target_shapes[:, 0] - 1)
    width = F.cross_entropy(output.width_logits, target_shapes[:, 1] - 1)
    return cell + 0.5 * (height + width)


def _supervised_loss(
    model: ARCRecursiveBaseline,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    target_shapes: torch.Tensor,
    *,
    puzzle_ids: torch.Tensor | None = None,
    puzzle_vector: torch.Tensor | None = None,
) -> torch.Tensor:
    carry = None
    losses: list[torch.Tensor] = []
    for _step in range(model.config.supervision_steps):
        output = model(
            inputs,
            puzzle_ids=puzzle_ids,
            puzzle_vector=puzzle_vector,
            carry=carry,
        )
        losses.append(_arc_loss(output, targets, target_shapes))
        carry = output.carry
    return torch.stack(losses).mean()


@torch.no_grad()
def _validation_loss(
    model: ARCRecursiveBaseline,
    examples: ARCGridExamples,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    weighted = 0.0
    count = 0
    for start in range(0, len(examples.inputs), batch_size):
        end = min(start + batch_size, len(examples.inputs))
        inputs = torch.tensor(examples.inputs[start:end], device=device)
        targets = torch.tensor(examples.targets[start:end], device=device)
        shapes = torch.tensor(examples.target_shapes[start:end], device=device)
        loss = _supervised_loss(model, inputs, targets, shapes)
        weighted += float(loss.cpu()) * (end - start)
        count += end - start
    model.train(was_training)
    return weighted / count


def fit_arc_recursive(
    model: ARCRecursiveBaseline,
    training: ARCGridExamples,
    validation: ARCGridExamples,
    config: ARCRecursiveTrainingConfig,
    *,
    seed: int,
) -> ARCRecursiveFitReceipt:
    """Fit on inner-train and select only on public validation demos."""

    if not isinstance(model, ARCRecursiveBaseline):
        raise TypeError("model must be ARCRecursiveBaseline")
    if not isinstance(config, ARCRecursiveTrainingConfig):
        raise TypeError("config must be ARCRecursiveTrainingConfig")
    seed = _nonnegative_integer(seed, name="seed")
    _validate_examples(model, training, training=True)
    _validate_examples(model, validation, training=False)
    device = torch.device(config.device)
    model.to(device)
    embedding_parameters = list(model.puzzle_embedding.parameters())
    embedding_ids = {id(parameter) for parameter in embedding_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in embedding_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.learning_rate},
            {"params": embedding_parameters, "lr": config.puzzle_learning_rate},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_history: list[float] = []
    validation_history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    optimizer_steps = 0
    order_digest = hashlib.sha256()
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(
            len(training.inputs), generator=generator
        ).numpy()
        order_digest.update(np.asarray(permutation, dtype=np.int64).tobytes())
        cumulative = 0.0
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            inputs = torch.tensor(training.inputs[indices], device=device)
            targets = torch.tensor(training.targets[indices], device=device)
            shapes = torch.tensor(training.target_shapes[indices], device=device)
            puzzle_ids = torch.tensor(
                training.puzzle_indices[indices], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _supervised_loss(
                model,
                inputs,
                targets,
                shapes,
                puzzle_ids=puzzle_ids,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer_steps += 1
            cumulative += float(loss.detach().cpu()) * len(indices)
        train_history.append(cumulative / len(training.inputs))
        current_validation = _validation_loss(
            model,
            validation,
            batch_size=config.batch_size,
            device=device,
        )
        validation_history.append(current_validation)
        if current_validation < best_loss:
            best_loss = current_validation
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise AssertionError("fixed positive epoch budget produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return ARCRecursiveFitReceipt(
        train_loss=tuple(train_history),
        validation_loss=tuple(validation_history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        optimizer_steps=optimizer_steps,
        training_seed=seed,
        training_examples=len(training.inputs),
        validation_examples=len(validation.inputs),
        training_data_sha256=_examples_sha256(training),
        validation_data_sha256=_examples_sha256(validation),
        epoch_permutation_sha256=order_digest.hexdigest(),
        checkpoint_sha256=state_dict_sha256(model),
    )


@dataclass(frozen=True)
class ARCTestTimeConfig:
    adaptation_epochs: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 8
    support_augmentations: int = 3
    inference_augmentations: int = 8
    scope: ARCAdaptScope = "full"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adaptation_epochs",
            _nonnegative_integer(self.adaptation_epochs, name="adaptation_epochs"),
        )
        for name in ("batch_size", "inference_augmentations"):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "support_augmentations",
            _nonnegative_integer(
                self.support_augmentations, name="support_augmentations"
            ),
        )
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = _finite_float(getattr(self, name), name=name)
            if name == "weight_decay" and value < 0.0:
                raise ValueError("weight_decay must be non-negative")
            if name != "weight_decay" and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.scope not in {"full", "puzzle_only"}:
            raise ValueError("scope must be 'full' or 'puzzle_only'")


@dataclass(frozen=True)
class ARCAdaptationReceipt:
    optimizer_steps: int
    support_examples: int
    adaptation_loss: tuple[float, ...]
    model_sha256: str
    public_task_fingerprint: str
    public_support_only: bool = True
    query_inputs_used_for_loss: bool = False
    query_targets_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _adapt_to_task(
    model: ARCRecursiveBaseline,
    task: PublicTask,
    config: ARCTestTimeConfig,
    *,
    seed: int,
) -> tuple[ARCRecursiveBaseline, torch.Tensor, ARCAdaptationReceipt]:
    seed = _nonnegative_integer(seed, name="seed")
    adapted = copy.deepcopy(model)
    device = next(model.parameters()).device
    adapted.to(device)
    puzzle = nn.Parameter(adapted.novel_puzzle_embedding.detach().clone())
    support = public_arc_support_examples(
        task,
        max_grid_size=model.config.max_grid_size,
        augmentations_per_pair=config.support_augmentations,
        seed=seed,
    )
    if config.scope == "puzzle_only":
        for parameter in adapted.parameters():
            parameter.requires_grad_(False)
        parameters: list[nn.Parameter] = [puzzle]
    else:
        parameters = list(adapted.parameters()) + [puzzle]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    losses: list[float] = []
    optimizer_steps = 0
    for _epoch in range(config.adaptation_epochs):
        adapted.train()
        permutation = torch.randperm(len(support.inputs), generator=generator).numpy()
        cumulative = 0.0
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            inputs = torch.tensor(support.inputs[indices], device=device)
            targets = torch.tensor(support.targets[indices], device=device)
            shapes = torch.tensor(support.target_shapes[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = _supervised_loss(
                adapted,
                inputs,
                targets,
                shapes,
                puzzle_vector=puzzle,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            optimizer.step()
            optimizer_steps += 1
            cumulative += float(loss.detach().cpu()) * len(indices)
        losses.append(cumulative / len(support.inputs))
    receipt = ARCAdaptationReceipt(
        optimizer_steps=optimizer_steps,
        support_examples=len(support.inputs),
        adaptation_loss=tuple(losses),
        model_sha256=state_dict_sha256(adapted),
        public_task_fingerprint=task.fingerprint,
    )
    return adapted, puzzle.detach(), receipt


def _candidate_key(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    contiguous = np.ascontiguousarray(grid, dtype=np.int64)
    return (tuple(int(value) for value in contiguous.shape), contiguous.tobytes())


def solve_arc_task(
    model: ARCRecursiveBaseline,
    task: PublicTask,
    config: ARCTestTimeConfig,
    *,
    seed: int,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Adapt on demonstrations and return at most two voted query attempts."""

    if not isinstance(model, ARCRecursiveBaseline):
        raise TypeError("model must be ARCRecursiveBaseline")
    if not isinstance(task, PublicTask) or task.family != "arc":
        raise TypeError("task must be a public ARC task")
    if not isinstance(config, ARCTestTimeConfig):
        raise TypeError("config must be ARCTestTimeConfig")
    adapted, puzzle, receipt = _adapt_to_task(model, task, config, seed=seed)
    adapted.eval()
    device = next(adapted.parameters()).device
    transforms = seeded_arc_transforms(
        count=config.inference_augmentations,
        seed=seed + 1,
        include_identity=True,
    )
    ranked_by_query: list[list[np.ndarray]] = []
    vote_counts_by_query: list[list[int]] = []
    for query_index, raw_grid in enumerate(tuple(task.query.get("inputs", ()))):
        candidates: dict[tuple[tuple[int, int], bytes], tuple[np.ndarray, int, int]] = {}
        insertion = 0
        for transform in transforms:
            transformed = transform.apply(raw_grid)
            packed, _shape = pack_arc_grid(
                transformed, max_grid_size=adapted.config.max_grid_size
            )
            tokens = torch.as_tensor(packed[None, :], device=device)
            carry = None
            for _step in range(adapted.config.supervision_steps):
                with torch.no_grad():
                    output = adapted(tokens, puzzle_vector=puzzle, carry=carry)
                carry = output.carry
                height = int(output.height_logits.argmax(dim=-1).item()) + 1
                width = int(output.width_logits.argmax(dim=-1).item()) + 1
                predicted = output.cell_logits.argmax(dim=-1).cpu().numpy()[0]
                transformed_candidate = unpack_arc_grid(
                    predicted,
                    (height, width),
                    max_grid_size=adapted.config.max_grid_size,
                )
                candidate = transform.invert(transformed_candidate)
                key = _candidate_key(candidate)
                previous = candidates.get(key)
                if previous is None:
                    candidates[key] = (candidate, 1, insertion)
                    insertion += 1
                else:
                    candidates[key] = (previous[0], previous[1] + 1, previous[2])
        ranked = sorted(candidates.values(), key=lambda item: (-item[1], item[2]))
        ranked_by_query.append([item[0] for item in ranked[:2]])
        vote_counts_by_query.append([int(item[1]) for item in ranked[:2]])
    if not ranked_by_query or any(not values for values in ranked_by_query):
        attempts: tuple[tuple[np.ndarray, ...], ...] = ()
    else:
        n_attempts = min(2, max(len(values) for values in ranked_by_query))
        attempts = tuple(
            tuple(
                values[min(attempt_index, len(values) - 1)]
                for values in ranked_by_query
            )
            for attempt_index in range(n_attempts)
        )
    prediction: Mapping[str, object] = {"attempts": attempts}
    diagnostics: Mapping[str, object] = {
        "adaptation": receipt.to_dict(),
        "n_queries": len(ranked_by_query),
        "n_attempts": len(attempts),
        "vote_counts_by_query": tuple(tuple(row) for row in vote_counts_by_query),
        "inference_augmentation_fingerprints": tuple(
            transform.fingerprint for transform in transforms
        ),
        "query_targets_used": False,
        "core_calls_per_candidate": adapted.config.core_calls_per_prediction,
        "query_indexed_only": tuple(range(len(ranked_by_query))),
    }
    return prediction, diagnostics
