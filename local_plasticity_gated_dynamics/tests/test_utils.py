import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed, make_rng
from src.utils.splits import (
    TrainOnlyTransformer,
    grouped_kfold,
    grouped_train_test_split,
)


def test_label_derived_random_streams_are_stable_and_independent() -> None:
    assert derive_seed(1, "data") == derive_seed(1, "data")
    assert derive_seed(1, "data") != derive_seed(1, "feedback")
    assert np.array_equal(
        make_rng(3, "x").normal(size=5), make_rng(3, "x").normal(size=5)
    )
    assert derive_seed(1, "a::b") != derive_seed(1, "a", "b")
    assert derive_seed(1, 1) != derive_seed(1, "1")
    with pytest.raises(TypeError, match="integer"):
        derive_seed(True, "x")


def test_grouped_splits_keep_groups_disjoint() -> None:
    groups = np.repeat(np.arange(8), 3)
    train, test = grouped_train_test_split(groups, test_fraction=0.25, seed=2)
    assert set(groups[train]).isdisjoint(set(groups[test]))
    for fold_train, fold_test in grouped_kfold(groups, n_splits=4):
        assert set(groups[fold_train]).isdisjoint(set(groups[fold_test]))
    mixed = [1, "1", 1, "1"]
    mixed_train, mixed_test = grouped_train_test_split(mixed, seed=0)
    train_labels = {(type(mixed[index]), mixed[index]) for index in mixed_train}
    test_labels = {(type(mixed[index]), mixed[index]) for index in mixed_test}
    assert train_labels.isdisjoint(test_labels)
    with pytest.raises(ValueError, match="finite"):
        grouped_train_test_split([0.0, np.nan, 1.0], seed=0)


def test_train_only_transformer_records_fit_provenance() -> None:
    x = np.arange(40, dtype=float).reshape(10, 4)
    transformer = TrainOnlyTransformer(n_components=2).fit(x[:8], sample_ids=range(8))
    assert transformer.transform(x[8:]).shape == (2, 2)
    assert np.array_equal(transformer.fit_sample_ids, np.arange(8))
    transformer.fit(x[:6])
    assert transformer.fit_sample_ids is None
    tuple_ids = [("session-a", index) for index in range(6)]
    transformer.fit(x[:6], sample_ids=tuple_ids)
    assert transformer.fit_sample_ids.shape == (6,)
    assert transformer.fit_sample_ids[0] == ("session-a", 0)
    before = transformer.transform(x[6:])
    with pytest.raises(ValueError):
        transformer.fit(np.ones((3, 1)), sample_ids=range(3))
    np.testing.assert_allclose(transformer.transform(x[6:]), before)
    assert transformer.fit_sample_ids[0] == ("session-a", 0)
    with pytest.raises(ValueError, match="sample_ids"):
        TrainOnlyTransformer().fit(x, sample_ids=[1])


def test_artifacts_register_plan_and_retain_failed_and_invalid_conditions(
    tmp_path: Path,
) -> None:
    with ExperimentRun("exp", 2, {"profile": "smoke"}, results_root=tmp_path) as run:
        run.register_conditions([{"condition": "ok"}, {"condition": "bad"}])
        run.record({"status": "complete", "score": 1.0}, condition="ok")
        run.mark_condition_failure(RuntimeError("boom"), condition="bad")
        run.mark_condition_invalid("not mathematically defined", condition="impossible")
        path = run.path
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert status["status"] == "complete_with_failures"
    assert {record["status"] for record in records} == {"complete", "failed", "invalid"}
    assert (path / "config.json").is_file()
    assert (path / "run.log").is_file()
    environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    assert {"python", "platform", "executable", "packages", "git"} <= environment.keys()
    assert {"numpy", "scipy", "pandas", "scikit-learn", "torch"} <= environment[
        "packages"
    ].keys()
    assert {"commit", "tree", "dirty"} == environment["git"].keys()
    assert "RuntimeError: boom" in (path / "run.log").read_text(encoding="utf-8")
    assert (
        len(json.loads((path / "planned_conditions.json").read_text(encoding="utf-8")))
        == 2
    )
    with pytest.raises(RuntimeError, match="immutable"):
        run.record({"late": 1.0})
    with pytest.raises(RuntimeError, match="immutable"):
        run.mark_condition_invalid("late")


def test_artifact_top_level_exception_is_persisted(tmp_path: Path) -> None:
    path = None
    with pytest.raises(ValueError, match="fatal"):
        with ExperimentRun("exp", 0, {}, results_root=tmp_path) as run:
            path = run.path
            raise ValueError("fatal")
    assert path is not None
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"


def test_invalid_only_artifact_is_not_reported_as_clean(tmp_path: Path) -> None:
    with ExperimentRun("exp", 0, {}, results_root=tmp_path) as run:
        run.register_conditions([{"condition": "undefined"}])
        run.mark_condition_invalid("scientifically undefined", condition="undefined")
        path = run.path

    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete_with_failures"
    assert status["condition_failures"] == 0
    assert status["condition_invalid"] == 1


def test_running_artifact_persists_start_time(tmp_path: Path) -> None:
    run = ExperimentRun("exp", 0, {}, results_root=tmp_path)
    try:
        status = json.loads((run.path / "status.json").read_text(encoding="utf-8"))
        manifest = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
        assert status["started_at"] == manifest["started_at"]
        assert status["status"] == "running"
        assert "run_label" not in status
        assert "run_label" not in manifest
        config = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
        assert "run_label" not in config
    finally:
        run.__exit__(None, None, None)


def test_artifacts_reject_provenance_overrides_and_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        ExperimentRun("exp", 0, {"seed": 99}, results_root=tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        ExperimentRun("exp", 0, {"run_label": "forged"}, results_root=tmp_path)
    with pytest.raises(ValueError, match="path separators"):
        ExperimentRun("../escape", 0, {}, results_root=tmp_path)
    with ExperimentRun("exp", 0, {}, results_root=tmp_path) as run:
        with pytest.raises(ValueError, match="reserved"):
            run.record({"seed": 99})
        run.record({"vector": np.array([1.0, np.nan])})
        encoded = json.loads(
            run.metrics_path.read_text(encoding="utf-8").splitlines()[0]
        )
        assert encoded["vector"] == [1.0, "nan"]
        run.register_conditions([{"condition": "a"}])
        with pytest.raises(RuntimeError, match="already"):
            run.register_conditions([{"condition": "b"}])


def test_artifact_run_label_is_path_safe_and_bound_to_metadata(
    tmp_path: Path,
) -> None:
    with ExperimentRun(
        "exp",
        7,
        {},
        results_root=tmp_path,
        run_label="formal-panel-a",
    ) as run:
        path = run.path
        assert path.name.endswith("_formal-panel-a")
    for name in ("config.json", "manifest.json", "status.json"):
        payload = json.loads((path / name).read_text(encoding="utf-8"))
        assert payload["run_label"] == "formal-panel-a"
    with pytest.raises(ValueError, match="path separators"):
        ExperimentRun(
            "exp",
            0,
            {},
            results_root=tmp_path,
            run_label="../escape",
        )
    with pytest.raises(ValueError, match="portable path-safe"):
        ExperimentRun("exp", 0, {}, results_root=tmp_path, run_label="bad:label")
    with pytest.raises(ValueError, match="portable path-safe"):
        ExperimentRun("exp", 0, {}, results_root=tmp_path, run_label="CON")
