from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from experiments.common import load_json_config
from experiments.exp09_hidden_context_gate import (
    FORMAL_H,
    FORMAL_Q,
    _validate_registered_config,
    run_seed,
)
from src.tasks.hidden_context import (
    HiddenContextConfig,
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_gate import (
    BASE_GATES,
    MD_INTERVENTIONS,
    HiddenGateCondition,
    build_hidden_gate_conditions,
    evaluate_gate_prediction,
    fit_hidden_gate,
    split_hidden_context_dataset,
)


def test_registered_hidden_context_grid_is_exact_and_unique() -> None:
    config = load_json_config("configs/formal/exp09_hidden_context_gate.json")
    conditions = build_hidden_gate_conditions(config)

    assert config["seeds"] == list(range(30))
    assert tuple(config["cue_reliabilities"]) == FORMAL_Q
    assert tuple(config["context_hazards"]) == FORMAL_H
    assert len(conditions) == 128
    assert len({condition.name for condition in conditions}) == 128
    assert sum(condition.intervention == "none" for condition in conditions) == 80
    assert sum(condition.intervention != "none" for condition in conditions) == 48
    assert {condition.gate_model for condition in conditions} == set(BASE_GATES)
    assert {
        condition.intervention
        for condition in conditions
        if condition.intervention != "none"
    } == set(MD_INTERVENTIONS)
    assert all(
        condition.gate_model == "md_recurrent_belief"
        for condition in conditions
        if condition.intervention != "none"
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("task", "sensory_noise_std", 0.9),
        ("switch_metrics", "posterior_threshold", 0.75),
        ("interventions", "delay_trials", 2),
        (None, "outer_test_fraction", 0.25),
        ("md_gate", "learning_rate", 0.04),
    ],
)
def test_formal_protocol_validator_fails_closed(
    section: str | None, key: str, value: object
) -> None:
    config = load_json_config("configs/formal/exp09_hidden_context_gate.json")
    if section is None:
        config[key] = value
    else:
        config[section][key] = value

    with pytest.raises(ValueError, match="preregistered protocol"):
        _validate_registered_config(config)


def test_hidden_context_split_is_whole_episode_and_shared_across_q() -> None:
    base = HiddenContextConfig(
        n_episodes=12,
        trials_per_episode=20,
        context_hazard=0.2,
        cue_reliability=0.55,
        dt_ms=100,
        cue_ms=100,
        sensory_ms=200,
        delay_ms=100,
        response_ms=100,
    )
    tape = make_hidden_context_random_tape(base, seed=4)
    low = generate_hidden_context(base, seed=4, random_tape=tape)
    high = generate_hidden_context(
        replace(base, cue_reliability=0.85), seed=4, random_tape=tape
    )
    low_split = split_hidden_context_dataset(
        low, outer_test_fraction=0.25, validation_fraction=0.25, seed=9
    )
    high_split = split_hidden_context_dataset(
        high, outer_test_fraction=0.25, validation_fraction=0.25, seed=9
    )

    assert low_split.fingerprint == high_split.fingerprint
    for low_part, high_part in zip(
        (low_split.train, low_split.dev, low_split.test),
        (high_split.train, high_split.dev, high_split.test),
        strict=True,
    ):
        np.testing.assert_array_equal(
            low_part.gate.episode_ids, high_part.gate.episode_ids
        )
        for episode in np.unique(low_part.gate.episode_ids):
            assert np.sum(low_part.gate.episode_ids == episode) == 20


def test_no_gate_complete_row_exposes_fail_closed_provenance() -> None:
    task = HiddenContextConfig(
        n_episodes=16,
        trials_per_episode=40,
        context_hazard=0.4,
        cue_reliability=0.7,
        dt_ms=100,
        cue_ms=100,
        sensory_ms=200,
        delay_ms=100,
        response_ms=100,
    )
    dataset = generate_hidden_context(task, seed=6)
    splits = split_hidden_context_dataset(
        dataset, outer_test_fraction=0.25, validation_fraction=0.25, seed=6
    )
    config = load_json_config("configs/smoke/exp09_hidden_context_gate.json")
    config.update(
        task=asdict(task),
        outer_test_fraction=0.25,
        validation_fraction=0.25,
    )
    config["switch_metrics"]["minimum_eligible_switches"] = 1
    fitted = fit_hidden_gate(
        "no_gate",
        splits,
        context_hazard=0.4,
        cue_reliability=0.7,
        config=config,
        seed=6,
    )
    condition = HiddenGateCondition(0.7, 0.4, "no_gate")
    metrics = evaluate_gate_prediction(
        fitted,
        fitted.prediction,
        splits,
        condition,
        config=config,
        profile="smoke",
        seed=6,
    )

    assert metrics["status"] == "complete"
    assert metrics["context_nll"] == pytest.approx(np.log(2.0))
    assert metrics["context_brier"] == pytest.approx(0.25)
    assert metrics["gate_test_accessed_true_context"] is False
    assert metrics["gate_fit_accessed_true_context"] is False
    assert metrics["third_factor_accessed_true_context"] is False
    assert metrics["predictions_frozen_before_truth_scoring"] is True
    assert metrics["train_dev_test_episode_disjoint"] is True
    assert metrics["statistics_unit"] == "seed"


def test_setup_failure_materializes_every_planned_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_json_config("configs/smoke/exp09_hidden_context_gate.json")

    def fail_tape(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced shared tape failure")

    monkeypatch.setattr(
        "experiments.exp09_hidden_context_gate.make_hidden_context_random_tape",
        fail_tape,
    )
    path = run_seed(config, 0, str(tmp_path))
    planned = json.loads((path / "planned_conditions.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))

    assert len(planned) == len(records) == 128
    assert all(record["status"] == "failed" for record in records)
    assert {record["condition"] for record in records} == {
        item["condition"] for item in planned
    }
    assert status["status"] == "complete_with_failures"
    assert status["condition_failures"] == 128
