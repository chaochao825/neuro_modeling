from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd

import scripts.summarize_exp15_arc_matched as summary
import src.utils.artifacts as artifacts
from experiments.common import load_json_config
from experiments.exp15_task_specialized_reasoning import run_seed
from figures.exp15_arc_matched_plot import plot_exp15_arc_matched
from scripts.summarize_exp15_arc_matched import (
    load_published_snapshot,
    publish_snapshot,
    validate_formal_run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "arc"
    for split, colors in (("training", (1, 2)), ("evaluation", (3, 4))):
        directory = root / split
        directory.mkdir(parents=True)
        for index, color in enumerate(colors):
            support = [[color, 0, 0], [0, 0, 0]]
            query = [[0, color], [0, 0], [0, 0]]
            payload = {
                "train": [
                    {
                        "input": support,
                        "output": [list(row) for row in zip(*support[::-1])],
                    }
                ],
                "test": [
                    {"input": query, "output": [list(row) for row in zip(*query[::-1])]}
                ],
            }
            (directory / f"task_{index}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    license_path = root / "LICENSE"
    license_path.write_text("Apache fixture\n", encoding="utf-8")
    manifest_path = tmp_path / "arc.sha256"
    paths = sorted(root.rglob("*.json"))
    manifest_path.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="utf-8",
        newline="\n",
    )
    source_receipt = {
        "commit": "fixture-revision",
        "license": "Apache-2.0",
        "name": "ARC-fixture",
        "url": "https://example.test/ARC",
        "splits": {"training": {"tasks": 2}, "evaluation": {"tasks": 2}},
    }
    validation_path = tmp_path / "arc_validation.json"
    validation_path.write_text(
        json.dumps({"datasets": [source_receipt]}, sort_keys=True), encoding="utf-8"
    )
    acquisition_path = tmp_path / "arc_acquisition.json"
    acquisition_path.write_text(
        json.dumps(
            {
                "arc": [source_receipt],
                "arc_manifest_sha256": {
                    "ARC-fixture": _sha(manifest_path),
                    "validation": _sha(validation_path),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = load_json_config("configs/smoke/exp15_task_specialized_arc.json")
    config.update(
        profile="formal",
        n_bootstrap=100,
        minimum_test_tasks=2,
        data={
            "path": str(root),
            "dataset_name": "ARC-fixture",
            "revision": "fixture-revision",
            "license": "Apache-2.0",
            "license_status": "verified",
            "source_url": "https://example.test/ARC",
            "acquisition_manifest_path": str(acquisition_path),
            "acquisition_manifest_sha256": _sha(acquisition_path),
            "validation_receipt_path": str(validation_path),
            "validation_receipt_sha256": _sha(validation_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha(manifest_path),
            "license_sha256": _sha(license_path),
            "expected_split_counts": {"training": 2, "evaluation": 2},
            "test_split_role": "ood",
            "exclude_relative_paths": [],
        },
    )
    config.pop("synthetic_fixture")
    return config


def test_exp15_matched_snapshot_recomputes_raw_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _formal_fixture(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "_software_provenance",
        lambda: {
            "packages": {},
            "git": {"commit": "a" * 40, "dirty": False},
        },
    )
    run_dir = run_seed(config, 0, tmp_path / "runs")
    with pytest.raises(ValueError, match="run config differs from the registered"):
        validate_formal_run(run_dir)

    registered_path = tmp_path / "registered_exp15_arc.json"
    registered_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(summary, "REGISTERED_FORMAL_CONFIG_PATH", registered_path)
    monkeypatch.setattr(
        summary, "REGISTERED_FORMAL_CONFIG_SHA256", _sha(registered_path)
    )
    monkeypatch.setattr(summary, "_current_git_state", lambda: ("a" * 40, False))

    conditions, comparison, raw, manifest = validate_formal_run(run_dir)
    assert len(conditions) == 2
    assert len(comparison) == 1
    assert len(raw) == 7
    assert manifest["source_tree_verified"] is True

    results_root = tmp_path / "published"
    publish_snapshot(run_dir, results_root)
    loaded_conditions, loaded_comparison, loaded_manifest = load_published_snapshot(
        results_root
    )
    assert len(loaded_conditions) == 2
    assert len(loaded_comparison) == 1
    assert len(loaded_manifest) == 1
    figure = plot_exp15_arc_matched(results_root)
    assert len(figure.axes) == 4
    assert (results_root / "exp15_arc_matched_formal.png").is_file()
    assert (results_root / "exp15_arc_matched_formal.pdf").is_file()

    condition_path = results_root / "exp15_arc_matched_formal_conditions.csv"
    tampered = pd.read_csv(condition_path)
    tampered.loc[0, "exact_accuracy"] = 0.123456
    tampered.to_csv(condition_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="differs from raw"):
        load_published_snapshot(results_root)
