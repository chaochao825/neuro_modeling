from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.baselines.tiny_recursive import (
    TinyRecursiveBaseline,
    TinyRecursiveConfig,
    TinyRecursiveTrainingConfig,
    fit_tiny_recursive,
    parameter_count,
    predict_tiny_recursive,
    state_dict_sha256,
)
from src.data.structured_protocol import (
    CapabilityError,
    PublicTask,
    build_structured_dataset,
)
from src.data.tiny_reasoning_data import (
    augment_sudoku_training,
    public_sudoku_test_inputs,
    split_sudoku_training_tasks,
)


def _solution() -> np.ndarray:
    return np.asarray(
        [
            [(3 * (row % 3) + row // 3 + col) % 9 + 1 for col in range(9)]
            for row in range(9)
        ],
        dtype=np.int64,
    )


def _dataset(n_train: int = 6, n_test: int = 2):
    tasks = []
    targets = []
    base = _solution()
    for index in range(n_train + n_test):
        mapping = np.arange(10)
        mapping[1:] = np.roll(np.arange(1, 10), index)
        solution = mapping[base]
        puzzle = solution.copy()
        puzzle.ravel()[index % 7 :: 7] = 0
        split = "train" if index < n_train else "test"
        tasks.append(
            PublicTask(
                task_id=f"sudoku-{index}",
                family="sudoku",
                split=split,
                source_group=f"source-{index}",
                augmentation_group=f"augmentation-{index}",
                context={"support_inputs": (), "support_outputs": ()},
                query={"grid": puzzle},
            )
        )
        targets.append(solution)

    def scorer(task, prediction, target):
        predicted = np.asarray(prediction)
        return {"exact": bool(np.array_equal(predicted, target))}

    return build_structured_dataset(tasks, targets, scorer=scorer)


def test_tiny_recursive_and_flat_are_parameter_and_core_call_matched() -> None:
    shared = dict(
        seq_len=9,
        vocab_size=5,
        hidden_size=8,
        num_heads=2,
        layers=1,
        high_cycles=2,
        low_cycles=2,
        supervision_steps=3,
    )
    torch.manual_seed(17)
    recursive = TinyRecursiveBaseline(TinyRecursiveConfig(**shared, mode="trm_like"))
    torch.manual_seed(17)
    flat = TinyRecursiveBaseline(
        TinyRecursiveConfig(**shared, mode="single_state_core_call_matched")
    )
    assert parameter_count(recursive) == parameter_count(flat)
    assert state_dict_sha256(recursive) == state_dict_sha256(flat)

    tokens = torch.randint(0, 5, (3, 9))
    recursive_output = recursive(tokens)
    flat_output = flat(tokens)
    assert recursive_output.logits.shape == (3, 9, 5)
    assert recursive_output.core_calls_per_segment == 8
    assert flat_output.core_calls_per_segment == 8
    assert recursive.config.core_calls == flat.config.core_calls == 24
    assert len(recursive_output.cycle_logits) == 2
    assert len(flat_output.cycle_logits) == 2
    assert recursive.checkpoint_metadata()["eligible_for_local_initialization"] is False
    assert recursive.checkpoint_metadata()["uses_bptt"] is True


def test_trm_gradient_prefix_is_stopped_and_segment_carry_is_detached() -> None:
    torch.manual_seed(3)
    model = TinyRecursiveBaseline(
        TinyRecursiveConfig(
            seq_len=9,
            vocab_size=5,
            hidden_size=8,
            num_heads=2,
            layers=1,
            high_cycles=2,
            low_cycles=2,
        )
    )
    inputs = torch.randint(0, 5, (2, 9))
    targets = torch.randint(1, 5, (2, 9))
    output = model(inputs)
    assert output.answer_states[0].requires_grad is False
    assert output.answer_states[-1].requires_grad is True
    assert output.carry.answer.requires_grad is False
    assert output.carry.latent.requires_grad is False
    loss = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, 5), targets.reshape(-1)
    )
    loss.backward()
    assert model.core.blocks[0].mlp[0].weight.grad is not None
    next_output = model(inputs, output.carry)
    assert next_output.answer_states[0].requires_grad is False


def test_sudoku_inner_split_and_augmentation_are_group_safe_and_deterministic() -> None:
    dataset = _dataset()
    training, validation = split_sudoku_training_tasks(
        dataset, validation_fraction=0.34, seed=11
    )
    assert set(training.source_groups).isdisjoint(validation.source_groups)
    assert set(training.augmentation_groups).isdisjoint(validation.augmentation_groups)
    assert set(training.content_groups).isdisjoint(validation.content_groups)
    first = augment_sudoku_training(training, augmentations_per_task=2, seed=13)
    second = augment_sudoku_training(training, augmentations_per_task=2, seed=13)
    np.testing.assert_array_equal(first.inputs, second.inputs)
    np.testing.assert_array_equal(first.targets, second.targets)
    assert len(first.inputs) == 3 * len(training.inputs)
    for puzzle, solution in zip(first.inputs, first.targets, strict=True):
        clue_mask = puzzle > 0
        np.testing.assert_array_equal(puzzle[clue_mask], solution[clue_mask])
        board = solution.reshape(9, 9)
        assert all(set(row) == set(range(1, 10)) for row in board)
        assert all(set(board[:, column]) == set(range(1, 10)) for column in range(9))


def test_public_test_adapter_never_exposes_a_training_view() -> None:
    dataset = _dataset()
    inputs, tasks = public_sudoku_test_inputs(dataset)
    assert inputs.shape == (2, 81)
    assert not inputs.flags.writeable
    with pytest.raises(CapabilityError, match="test targets are unavailable"):
        dataset.target_store.training_view(tasks[0])


def test_tiny_recursive_fit_is_deterministic_and_has_no_test_argument() -> None:
    dataset = _dataset(n_train=6, n_test=1)
    training, validation = split_sudoku_training_tasks(
        dataset, validation_fraction=0.34, seed=19
    )
    architecture = TinyRecursiveConfig(
        hidden_size=8,
        num_heads=2,
        layers=1,
        high_cycles=1,
        low_cycles=1,
        supervision_steps=2,
    )
    training_config = TinyRecursiveTrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )

    def fitted():
        torch.manual_seed(23)
        model = TinyRecursiveBaseline(architecture)
        receipt = fit_tiny_recursive(
            model,
            training.inputs,
            training.targets,
            validation.inputs,
            validation.targets,
            training_config,
            seed=29,
        )
        return model, receipt

    first_model, first_receipt = fitted()
    second_model, second_receipt = fitted()
    assert first_receipt == second_receipt
    assert first_receipt.optimizer_steps == 8
    np.testing.assert_array_equal(
        predict_tiny_recursive(first_model, validation.inputs, batch_size=2),
        predict_tiny_recursive(second_model, validation.inputs, batch_size=2),
    )
    assert first_receipt.test_data_used_for_fit is False
    assert "test" not in inspect.signature(fit_tiny_recursive).parameters
