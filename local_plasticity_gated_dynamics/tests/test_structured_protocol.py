"""Capability and group-split contracts for structured tasks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.data.structured_protocol import (
    CapabilityError,
    PublicTask,
    StructuredProtocolError,
    build_structured_dataset,
    grouped_split_indices,
)


def _task(task_id: str, split: str, *, group: str | None = None) -> PublicTask:
    return PublicTask(
        task_id=task_id,
        family="toy",
        split=split,
        source_group=group or task_id,
        augmentation_group=group or task_id,
        context={"support_inputs": [np.array([1])], "support_outputs": [[2]]},
        query={"inputs": [np.array([3])]},
        metadata={"version": "fixture"},
    )


def _score(task, prediction, target):
    del task
    return {"exact": prediction == target}


def test_public_task_is_target_free_and_recursively_immutable() -> None:
    task = _task("train_0", "train")
    assert task.group_id == "train_0"
    assert task.support_inputs[0].tolist() == [1]
    assert task.support_outputs == ((2,),)
    assert task.query_inputs[0].tolist() == [3]
    assert not task.query_inputs[0].flags.writeable
    with pytest.raises(ValueError):
        task.query_inputs[0][0] = 9
    with pytest.raises(TypeError):
        task.query["new"] = 1
    with pytest.raises(StructuredProtocolError, match="target-bearing"):
        PublicTask(
            task_id="bad",
            family="arc",
            split="test",
            source_group="bad",
            augmentation_group="bad",
            context={},
            query={"nested": {"solution": [[1]]}},
        )
    with pytest.raises(StructuredProtocolError, match="target-bearing"):
        replace(task, metadata={"query_target": [1]})
    with pytest.raises(StructuredProtocolError, match="target-bearing"):
        replace(task, context={"solution": [1]})
    with pytest.raises(StructuredProtocolError, match="target-bearing"):
        replace(task, metadata={"queryTarget": [1]})
    with pytest.raises(StructuredProtocolError, match="target-bearing"):
        replace(
            task,
            context={
                "demonstrations": [
                    {"input": [1], "output": [2], "nested": {"output": [9]}}
                ]
            },
        )


def test_target_store_grants_supervised_views_only_off_test_split() -> None:
    train = _task("train_0", "train")
    validation = _task("validation_0", "validation")
    test = _task("test_0", "test")
    dataset = build_structured_dataset(
        (train, validation, test), (1, 2, 3), scorer=_score
    )
    assert dataset.target_store.training_view(train).target == 1
    assert dataset.target_store.training_view(validation).target == 2
    with pytest.raises(CapabilityError, match="test targets"):
        dataset.target_store.training_view(test)
    with pytest.raises(CapabilityError, match="identity-bound"):
        dataset.target_store.training_view(replace(test, split="train"))
    assert not hasattr(dataset.target_store, "get_target")
    assert dataset.target_store.score(test, 3)["exact"]
    assert dataset.train_task_ids == ("train_0",)
    assert dataset.validation_task_ids == ("validation_0",)
    assert dataset.test_task_ids == ("test_0",)


def test_source_and_augmentation_groups_cannot_cross_splits() -> None:
    train = _task("train_0", "train", group="shared")
    test = _task("test_0", "test", group="shared")
    with pytest.raises(StructuredProtocolError, match="crosses splits"):
        build_structured_dataset((train, test), (1, 2), scorer=_score)


def test_grouped_split_is_seeded_and_keeps_both_group_families_together() -> None:
    source = ("s0", "s0", "s1", "s2", "s3", "s4")
    augmentation = ("a0", "a1", "a1", "a2", "a3", "a4")
    first = grouped_split_indices(
        source,
        augmentation,
        seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    second = grouped_split_indices(
        source,
        augmentation,
        seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    assert first == second
    for groups in (source, augmentation):
        for group in set(groups):
            assert (
                len(
                    {
                        first[index]
                        for index, value in enumerate(groups)
                        if value == group
                    }
                )
                == 1
            )
