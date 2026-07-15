from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.baselines.arc_recursive import (
    ARCRecursiveBaseline,
    ARCRecursiveConfig,
    ARCRecursiveTrainingConfig,
    ARCTestTimeConfig,
    fit_arc_recursive,
    parameter_count,
    solve_arc_task,
)
from src.data.arc_recursive_dataset import (
    ARC_PAD_TOKEN,
    ARC_TARGET_IGNORE,
    ARCTransform,
    build_arc_examples,
    pack_arc_grid,
    pack_arc_target,
    public_arc_support_examples,
    seeded_arc_transforms,
    split_arc_training_tasks,
    unpack_arc_grid,
)
from src.data.arc_tasks import load_arc_directory, score_arc_attempts
from src.data.structured_protocol import CapabilityError


def test_exp18_smoke_run_persists_all_conditions(tmp_path: Path) -> None:
    from experiments.exp18_arc_recursive_baseline import run_seed
    from figures.exp18_arc_recursive_plot import plot_exp18
    from scripts.summarize_exp18_arc_recursive import publish_snapshot

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "smoke"
        / "exp18_arc_recursive_arc.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["save_checkpoints"] = False
    run_path = run_seed(config, 0, str(tmp_path / "results"))
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    rows = [
        json.loads(line)
        for line in (run_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    task_rows = [row for row in rows if row.get("level") == "task"]
    assert len(task_rows) == 4 * 3
    assert not any(row["query_targets_used"] for row in task_rows)
    summaries = [row for row in rows if row.get("level") == "condition_summary"]
    assert len(summaries) == 4
    assert all(row["n_tasks"] == 3 for row in summaries)
    published = publish_snapshot(
        (run_path,), tmp_path / "published", prefix="exp18_unit"
    )
    assert all(path.is_file() for path in published.values())
    figures = plot_exp18(tmp_path / "published", prefix="exp18_unit")
    assert all(path.is_file() for path in figures.values())


def _write_task(path: Path, *, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": [
            {
                "input": [[offset, 0], [0, offset]],
                "output": [[offset, 0], [0, offset]],
            },
            {
                "input": [[0, offset], [offset, 0]],
                "output": [[0, offset], [offset, 0]],
            },
        ],
        "test": [
            {
                "input": [[offset, offset], [0, 0]],
                "output": [[offset, offset], [0, 0]],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(tmp_path: Path):
    _write_task(tmp_path / "training" / "a.json", offset=1)
    _write_task(tmp_path / "training" / "b.json", offset=2)
    _write_task(tmp_path / "training" / "c.json", offset=3)
    _write_task(tmp_path / "evaluation" / "d.json", offset=4)
    return load_arc_directory(
        tmp_path,
        dataset_name="fixture",
        dataset_revision="unit-test",
        namespace_task_ids=True,
        attempt_aware_scoring=True,
    )


def test_arc_transform_pack_and_inverse_are_exact() -> None:
    grid = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    transform = ARCTransform(
        rotation_quarters=1,
        reflect=True,
        color_permutation=(0, 2, 3, 4, 5, 6, 7, 8, 9, 1),
    )
    assert np.array_equal(transform.invert(transform.apply(grid)), grid)
    packed, shape = pack_arc_grid(grid, max_grid_size=4)
    target, target_shape = pack_arc_target(grid, max_grid_size=4)
    assert shape == target_shape == (2, 3)
    assert np.count_nonzero(packed == ARC_PAD_TOKEN) == 10
    assert np.count_nonzero(target == ARC_TARGET_IGNORE) == 10
    assert np.array_equal(unpack_arc_grid(packed, shape, max_grid_size=4), grid)
    first = seeded_arc_transforms(count=8, seed=17)
    second = seeded_arc_transforms(count=8, seed=17)
    assert [item.fingerprint for item in first] == [
        item.fingerprint for item in second
    ]
    assert len({item.fingerprint for item in first}) == 8


def test_attempt_scorer_allows_querywise_top_two_but_fails_closed() -> None:
    targets = (np.asarray([[1]]), np.asarray([[2]]))
    prediction = {
        "attempts": (
            (np.asarray([[1]]), np.asarray([[9]])),
            (np.asarray([[8]]), np.asarray([[2]])),
        )
    }
    result = score_arc_attempts(None, prediction, targets)  # type: ignore[arg-type]
    assert result["exact"]
    assert result["query_winning_attempt"] == (1, 2)
    too_many = {"attempts": (*prediction["attempts"], prediction["attempts"][0])}
    result = score_arc_attempts(None, too_many, targets)  # type: ignore[arg-type]
    assert not result["exact"]
    assert result["too_many_attempts"]


def test_arc_examples_split_tasks_and_never_expose_test_query(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    training, validation = split_arc_training_tasks(
        dataset, validation_fraction=0.34, seed=3
    )
    assert training and validation
    assert {task.source_group for task in training}.isdisjoint(
        {task.source_group for task in validation}
    )
    train_examples = build_arc_examples(
        dataset,
        training,
        max_grid_size=3,
        augmentations_per_pair=1,
        seed=4,
        include_query_targets=True,
        name="inner_train",
    )
    assert train_examples.n_puzzles == len(training)
    test_task = dataset.for_split("test")[0]
    support = public_arc_support_examples(
        test_task, max_grid_size=3, augmentations_per_pair=1, seed=5
    )
    assert len(support.inputs) == 4
    assert all("support" in value for value in support.example_ids)
    with pytest.raises(ValueError, match="test query targets"):
        build_arc_examples(
            dataset,
            (test_task,),
            max_grid_size=3,
            augmentations_per_pair=0,
            seed=0,
            include_query_targets=True,
            name="forbidden",
        )
    with pytest.raises(CapabilityError):
        dataset.target_store.training_view(test_task)


def test_recursive_and_single_state_have_matched_parameters_and_core_calls() -> None:
    common = ARCRecursiveConfig(
        max_grid_size=3,
        hidden_size=8,
        num_heads=2,
        layers=1,
        high_cycles=2,
        low_cycles=1,
        supervision_steps=2,
        num_puzzle_embeddings=2,
    )
    torch.manual_seed(11)
    recursive = ARCRecursiveBaseline(common)
    single = ARCRecursiveBaseline(
        replace(common, mode="single_state_core_call_matched")
    )
    single.load_state_dict(recursive.state_dict())
    assert parameter_count(recursive) == parameter_count(single)
    assert recursive.config.core_calls_per_prediction == (
        single.config.core_calls_per_prediction
    )
    tokens = torch.tensor(
        [[1, 2, ARC_PAD_TOKEN, 3, 4, ARC_PAD_TOKEN, ARC_PAD_TOKEN,
          ARC_PAD_TOKEN, ARC_PAD_TOKEN]],
        dtype=torch.long,
    )
    output = recursive(tokens, puzzle_ids=torch.tensor([0]))
    assert output.cell_logits.shape == (1, 9, 10)
    assert output.height_logits.shape == (1, 3)
    assert output.core_calls_per_segment == 4
    assert output.carry.answer.grad_fn is None
    assert output.carry.latent.grad_fn is None


def test_tiny_fit_and_demo_tta_solver_run_without_query_targets(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    training_tasks, validation_tasks = split_arc_training_tasks(
        dataset, validation_fraction=0.34, seed=0
    )
    training = build_arc_examples(
        dataset,
        training_tasks,
        max_grid_size=3,
        augmentations_per_pair=0,
        seed=0,
        include_query_targets=True,
        name="inner_train",
    )
    validation = build_arc_examples(
        None,
        validation_tasks,
        max_grid_size=3,
        augmentations_per_pair=0,
        seed=1,
        include_query_targets=False,
        name="inner_validation_public_demos",
    )
    torch.manual_seed(0)
    model = ARCRecursiveBaseline(
        ARCRecursiveConfig(
            max_grid_size=3,
            hidden_size=8,
            num_heads=2,
            layers=1,
            high_cycles=1,
            low_cycles=1,
            supervision_steps=1,
            num_puzzle_embeddings=training.n_puzzles,
        )
    )
    receipt = fit_arc_recursive(
        model,
        training,
        validation,
        ARCRecursiveTrainingConfig(
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            puzzle_learning_rate=1e-3,
            weight_decay=0.0,
            device="cpu",
        ),
        seed=2,
    )
    assert not receipt.test_data_used_for_fit
    assert receipt.optimizer_steps > 0
    test_task = dataset.for_split("test")[0]
    prediction, diagnostics = solve_arc_task(
        model,
        test_task,
        ARCTestTimeConfig(
            adaptation_epochs=1,
            learning_rate=1e-3,
            batch_size=8,
            support_augmentations=0,
            inference_augmentations=1,
        ),
        seed=3,
    )
    assert 1 <= len(prediction["attempts"]) <= 2
    assert diagnostics["query_targets_used"] is False
    assert diagnostics["adaptation"]["query_targets_used"] is False
    score = dataset.target_store.score(test_task, prediction)
    assert score["n_attempts_received"] <= 2
