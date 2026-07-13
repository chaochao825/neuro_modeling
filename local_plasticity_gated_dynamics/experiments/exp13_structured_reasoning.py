"""Capability-safe ARC/Maze/Sudoku hybrid reasoning benchmark.

The experiment recomputes proposals from each public task, freezes one matched
panel, and compares non-spiking flat/fast-slow/trace controllers with a
target-free heuristic, an isolated GRU/BPTT baseline, and a labeled oracle
ceiling.  It tests functional proposal selection, not a biological mechanism
or a proposal-free HRM/CTM reproduction.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    PROJECT_ROOT,
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.structured_benchmark import (
    STRUCTURED_CONDITIONS,
    assert_matched_candidate_panel,
    build_candidate_panel,
    fit_receipt_dict,
    run_structured_condition,
)
from src.data.arc_manifest import (
    validate_arc_acquisition_receipts,
    validate_arc_source_manifest,
)
from src.data.arc_tasks import load_arc_directory
from src.data.maze_tasks import load_maze_tasks
from src.data.structured_protocol import StructuredDataset
from src.data.sudoku_tasks import load_sudoku_tasks
from src.utils.artifacts import ExperimentRun


def _json_grid(value: np.ndarray) -> list[list[int]]:
    return np.asarray(value, dtype=int).tolist()


def _arc_rule(index: int, value: np.ndarray) -> np.ndarray:
    rule = index % 5
    if rule == 0:
        return value.copy()
    if rule == 1:
        return np.rot90(value)
    if rule == 2:
        return np.fliplr(value)
    if rule == 3:
        output = value.copy()
        output[value == 1] = 3
        output[value == 2] = 4
        return output
    return np.repeat(np.repeat(value, 2, axis=0), 2, axis=1)


def _write_arc_fixture(root: Path, fixture: Mapping[str, Any]) -> Path:
    rng = np.random.default_rng(int(fixture.get("seed", 1729)))
    counts = {
        "training": int(fixture.get("n_train_tasks", 15)),
        "evaluation": int(fixture.get("n_test_tasks", 8)),
    }
    for directory, count in counts.items():
        target = root / directory
        target.mkdir(parents=True, exist_ok=True)
        for local_index in range(count):
            global_index = local_index + (0 if directory == "training" else 10_000)
            demonstrations = []
            for example in range(2):
                grid = rng.integers(0, 3, size=(3 + example, 4), dtype=np.int16)
                grid[0, 0] = (global_index + example) % 3
                demonstrations.append(
                    {
                        "input": _json_grid(grid),
                        "output": _json_grid(_arc_rule(global_index, grid)),
                    }
                )
            query = rng.integers(0, 3, size=(4, 3), dtype=np.int16)
            query[0, 0] = global_index % 3
            payload = {
                "train": demonstrations,
                "test": [
                    {
                        "input": _json_grid(query),
                        "output": _json_grid(_arc_rule(global_index, query)),
                    }
                ],
            }
            (target / f"fixture_{global_index:05d}.json").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
    return root


def _write_maze_fixture(root: Path, fixture: Mapping[str, Any]) -> Path:
    rng = np.random.default_rng(int(fixture.get("seed", 1729)))
    path = root / "mazes.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    counts = (
        ("train", int(fixture.get("n_train_tasks", 18))),
        ("test", int(fixture.get("n_test_tasks", 10))),
    )
    index = 0
    for split, count in counts:
        for _ in range(count):
            size = int(fixture.get("maze_size", 9))
            maze = np.zeros((size, size), dtype=int)
            interior = rng.random((size - 2, size - 2)) < 0.22
            maze[1:-1, 1:-1] = interior.astype(int)
            # The top and right borders guarantee reachability while the
            # interior still changes search cost and alternate paths.
            maze[0, :] = 0
            maze[:, -1] = 0
            records.append(
                {
                    "task_id": f"maze_fixture_{index:05d}",
                    "source_group": f"maze_fixture_{index:05d}",
                    "split": split,
                    "maze": maze.tolist(),
                    "start": [0, 0],
                    "goal": [size - 1, size - 1],
                    "wall_value": 1,
                    "source_version": "synthetic_smoke_not_scientific",
                }
            )
            index += 1
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _sudoku_solution() -> np.ndarray:
    base = 3
    side = base * base

    def pattern(row: int, col: int) -> int:
        return (base * (row % base) + row // base + col) % side

    return np.asarray(
        [[pattern(row, col) + 1 for col in range(side)] for row in range(side)],
        dtype=int,
    )


def _write_sudoku_fixture(root: Path, fixture: Mapping[str, Any]) -> Path:
    rng = np.random.default_rng(int(fixture.get("seed", 1729)))
    path = root / "sudoku.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    base_solution = _sudoku_solution()
    counts = (
        ("train", int(fixture.get("n_train_tasks", 18))),
        ("test", int(fixture.get("n_test_tasks", 10))),
    )
    records = []
    index = 0
    for split, count in counts:
        for _ in range(count):
            digits = rng.permutation(np.arange(1, 10))
            solution = digits[base_solution - 1]
            puzzle = solution.copy()
            keep = rng.random((9, 9)) < float(fixture.get("clue_fraction", 0.48))
            puzzle[~keep] = 0
            records.append(
                {
                    "task_id": f"sudoku_fixture_{index:05d}",
                    "source_group": f"sudoku_fixture_{index:05d}",
                    "split": split,
                    "puzzle": "".join(str(value) for value in puzzle.ravel()),
                    "solution": "".join(str(value) for value in solution.ravel()),
                    "source_version": "synthetic_smoke_not_scientific",
                }
            )
            index += 1
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _resolve_data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_preparation_manifest(
    data: Mapping[str, Any], *, family: str, dataset_path: Path
) -> tuple[Path, Mapping[str, Any], str]:
    """Bind a formal derived Maze/Sudoku file to its audited manifest."""

    expected_digest = data.get("manifest_sha256")
    if (
        not isinstance(expected_digest, str)
        or _SHA256.fullmatch(expected_digest) is None
    ):
        raise ValueError(
            "formal Maze/Sudoku manifest_sha256 must be a pinned lowercase digest; "
            "run the public-data preparation script, review it, and replace the "
            "explicit placeholder"
        )
    raw_manifest_path = data.get("manifest_path")
    if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
        raise ValueError("formal Maze/Sudoku data require manifest_path")
    manifest_path = _resolve_data_path(raw_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_digest = _file_sha256(manifest_path)
    if actual_digest != expected_digest:
        raise ValueError("formal Maze/Sudoku preparation manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "formal Maze/Sudoku preparation manifest is invalid JSON"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError("formal Maze/Sudoku preparation manifest must be an object")
    if manifest.get("schema_version") != "exp13_public_benchmark_manifest_v1":
        raise ValueError("unsupported Maze/Sudoku preparation manifest schema")
    if manifest.get("family") != family:
        raise ValueError("preparation manifest task family disagrees with config")
    if manifest.get("status") not in {"complete", "complete_with_exclusions"}:
        raise ValueError("preparation manifest does not describe a completed build")
    builder = manifest.get("builder")
    builder_script = PROJECT_ROOT / "scripts" / "prepare_exp13_public_benchmarks.py"
    if (
        not isinstance(builder, Mapping)
        or builder.get("script") != builder_script.name
        or builder.get("script_sha256") != _file_sha256(builder_script)
    ):
        raise ValueError(
            "preparation manifest is not bound to the current public-data builder"
        )
    dataset = manifest.get("dataset")
    output = manifest.get("output")
    split_receipt = manifest.get("split")
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(output, Mapping)
        or not isinstance(split_receipt, Mapping)
    ):
        raise ValueError("preparation manifest lacks dataset/output/split receipts")
    if dataset.get("license_status") != "verified":
        raise ValueError("preparation manifest does not verify the data license")
    for manifest_key, config_key in (
        ("name", "dataset_name"),
        ("revision", "revision"),
        ("license", "license"),
    ):
        if str(dataset.get(manifest_key)) != str(data.get(config_key)):
            raise ValueError(
                f"preparation manifest {manifest_key} disagrees with formal config"
            )
    if split_receipt.get("test_split_role") != data.get("test_split_role"):
        raise ValueError(
            "preparation manifest test_split_role disagrees with formal config"
        )
    output_name = output.get("path")
    output_digest = output.get("sha256")
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("preparation manifest output path is invalid")
    manifested_dataset = manifest_path.parent / output_name
    if manifested_dataset.resolve() != dataset_path.resolve():
        raise ValueError("preparation manifest is bound to a different dataset path")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not isinstance(output_digest, str) or _SHA256.fullmatch(output_digest) is None:
        raise ValueError("preparation manifest output SHA-256 is invalid")
    if _file_sha256(dataset_path) != output_digest:
        raise ValueError("prepared Maze/Sudoku dataset SHA-256 mismatch")
    return manifest_path, manifest, actual_digest


def _load_dataset(
    config: Mapping[str, Any], run_path: Path
) -> tuple[StructuredDataset, bool, Mapping[str, Any]]:
    family = str(config["family"]).strip().lower()
    fixture = config.get("synthetic_fixture")
    data = dict(config.get("data", {}))
    if fixture is not None:
        if str(config.get("profile")) != "smoke":
            raise ValueError(
                "synthetic structured fixtures are allowed only in smoke runs"
            )
        fixture_root = run_path / "synthetic_fixture_not_scientific"
        if family == "arc":
            path = _write_arc_fixture(fixture_root, fixture)
            dataset = load_arc_directory(
                path,
                dataset_name="synthetic_arc_fixture",
                dataset_revision="not_scientific",
                namespace_task_ids=True,
            )
        elif family == "maze":
            path = _write_maze_fixture(fixture_root, fixture)
            dataset = load_maze_tasks(path, split=None)
        elif family == "sudoku":
            path = _write_sudoku_fixture(fixture_root, fixture)
            dataset = load_sudoku_tasks(path, split=None)
        else:
            raise ValueError(f"unsupported structured family {family!r}")
        return (
            dataset,
            True,
            {
                "source": "synthetic_smoke_fixture",
                "license_status": "not_applicable",
                "source_revision": "not_scientific",
                "source_manifest_verified": False,
                "source_acquisition_verified": False,
                "source_manifest_receipt": None,
                "source_acquisition_receipt": None,
            },
        )

    formal_profile = str(config.get("profile")) == "formal"
    if formal_profile and data.get("license_status") != "verified":
        raise ValueError(
            "formal public data require license_status='verified'; acquisition stays fail-closed"
        )
    test_split_role = data.get("test_split_role")
    if formal_profile and test_split_role not in {"ood", "non_ood"}:
        raise ValueError(
            "formal structured data must register test_split_role as ood or non_ood"
        )
    raw_path = data.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        reason = data.get("unavailable_reason", "public data path is not configured")
        raise FileNotFoundError(str(reason))
    path = _resolve_data_path(raw_path)
    preparation_manifest: Mapping[str, Any] | None = None
    arc_manifest_receipt: Mapping[str, Any] | None = None
    arc_acquisition_receipt: Mapping[str, Any] | None = None
    actual_manifest_sha256 = data.get("manifest_sha256")
    if formal_profile:
        if family == "arc":
            raw_manifest_path = data.get("manifest_path")
            if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
                raise ValueError("formal ARC data require manifest_path")
            expected_split_counts = data.get("expected_split_counts")
            if not isinstance(expected_split_counts, Mapping):
                raise ValueError("formal ARC data require expected_split_counts")
            receipt = validate_arc_source_manifest(
                path,
                _resolve_data_path(raw_manifest_path),
                expected_manifest_sha256=str(data.get("manifest_sha256", "")),
                expected_license_sha256=str(data.get("license_sha256", "")),
                expected_split_counts=expected_split_counts,
            )
            arc_manifest_receipt = receipt.to_dict()
            actual_manifest_sha256 = receipt.manifest_sha256
            raw_acquisition_path = data.get("acquisition_manifest_path")
            raw_validation_path = data.get("validation_receipt_path")
            if not isinstance(raw_acquisition_path, str) or not isinstance(
                raw_validation_path, str
            ):
                raise ValueError(
                    "formal ARC data require acquisition and validation receipt paths"
                )
            arc_acquisition_receipt = validate_arc_acquisition_receipts(
                _resolve_data_path(raw_acquisition_path),
                _resolve_data_path(raw_validation_path),
                expected_acquisition_manifest_sha256=str(
                    data.get("acquisition_manifest_sha256", "")
                ),
                expected_validation_sha256=str(
                    data.get("validation_receipt_sha256", "")
                ),
                dataset_name=str(data.get("dataset_name", "")),
                revision=str(data.get("revision", "")),
                source_url=str(data.get("source_url", "")),
                license_name=str(data.get("license", "")),
                source_manifest_sha256=receipt.manifest_sha256,
                expected_split_counts=expected_split_counts,
            )
        elif family in {"maze", "sudoku"}:
            _, preparation_manifest, actual_manifest_sha256 = (
                _validated_preparation_manifest(data, family=family, dataset_path=path)
            )
    if family == "arc":
        dataset = load_arc_directory(
            path,
            dataset_name=str(data.get("dataset_name", "ARC")),
            dataset_revision=str(data.get("revision", "unspecified")),
            exclude_relative_paths=tuple(data.get("exclude_relative_paths", ())),
            namespace_task_ids=True,
        )
    elif family == "maze":
        dataset = load_maze_tasks(path, split=data.get("default_split"))
    elif family == "sudoku":
        dataset = load_sudoku_tasks(path, split=data.get("default_split"))
    else:
        raise ValueError(f"unsupported structured family {family!r}")
    provenance = {
        "source": str(path),
        "dataset_name": data.get("dataset_name"),
        "source_revision": data.get("revision"),
        "license": data.get("license"),
        "license_status": data.get("license_status"),
        "manifest_sha256": actual_manifest_sha256,
        "source_manifest_verified": arc_manifest_receipt is not None,
        "source_acquisition_verified": arc_acquisition_receipt is not None,
        "source_manifest_receipt": arc_manifest_receipt,
        "source_acquisition_receipt": arc_acquisition_receipt,
        "test_split_role": test_split_role,
        "preparation_manifest_status": (
            preparation_manifest.get("status")
            if preparation_manifest is not None
            else None
        ),
        "preparation_manifest_summary": (
            preparation_manifest.get("summary")
            if preparation_manifest is not None
            else None
        ),
        # Embed the complete reviewed receipt in each run artifact because the
        # generated data/structured directory is intentionally git-ignored.
        "preparation_manifest": preparation_manifest,
        "exclude_relative_paths": data.get("exclude_relative_paths", []),
    }
    return dataset, False, provenance


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    family = str(config["family"]).strip().lower()
    conditions = tuple(config.get("conditions", STRUCTURED_CONDITIONS))
    unknown = set(conditions) - set(STRUCTURED_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown structured conditions: {sorted(unknown)}")
    run_config = {
        **config,
        "training_algorithm": "matched_hybrid_structured_candidate_selection",
        "used_autograd": "baseline_only",
        "parent_checkpoint": None,
        "spiking_model": False,
        "neural_evidence_claim": False,
    }
    with ExperimentRun(
        "exp13_structured_reasoning", seed, run_config, results_root=results_root
    ) as run:
        planned = [
            {"condition": condition, "task_family": family} for condition in conditions
        ]
        run.register_conditions(planned)
        recorded: set[str] = set()
        results = []
        try:
            dataset, fixture_only, source_provenance = _load_dataset(config, run.path)
            split_counts = {
                split: len(dataset.for_split(split))
                for split in ("train", "validation", "test")
            }
            if split_counts["train"] < 1 or split_counts["test"] < 1:
                raise ValueError("exp13 requires at least one train and one test task")
            panel = build_candidate_panel(
                dataset,
                max_arc_candidates=int(config.get("max_arc_candidates", 96)),
            )
            provenance = {
                **source_provenance,
                "family": family,
                "fixture_only": fixture_only,
                "split_counts": split_counts,
                "n_proposal_failures": len(panel.failures),
                "proposal_failures": dict(panel.failures),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "split": task.split,
                        "source_group": task.source_group,
                        "augmentation_group": task.augmentation_group,
                        "public_fingerprint": task.fingerprint,
                        "public_metadata": dict(task.metadata),
                        "candidate_fingerprint": (
                            panel.proposal_batches[task.task_id].candidate_fingerprint
                            if task.task_id in panel.proposal_batches
                            else None
                        ),
                    }
                    for task in dataset.tasks
                ],
            }
            (run.path / "public_task_provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
            )
            fit_receipts: dict[str, Any] = {}
            for condition in conditions:
                try:
                    result = run_structured_condition(
                        dataset,
                        panel,
                        condition=condition,
                        seed=seed,
                        model_config=dict(config.get("model", {})),
                        n_bootstrap=int(config.get("n_bootstrap", 10_000)),
                        fit_splits=tuple(config.get("fit_splits", ["train"])),
                    )
                    results.append(result)
                    fit_receipts[condition] = fit_receipt_dict(result.fit_receipt)
                    metrics = dict(result.summary)
                    metrics.pop("condition", None)
                    hyperparameters_frozen = bool(
                        config.get("hyperparameters_frozen_before_test", False)
                    )
                    formal_evidence_eligible = (
                        str(config.get("profile")) == "formal"
                        and not fixture_only
                        and source_provenance.get("license_status") == "verified"
                        and hyperparameters_frozen
                        and split_counts["test"]
                        >= int(config.get("minimum_test_tasks", 100))
                    )
                    metrics.update(
                        profile=str(config.get("profile", "unspecified")),
                        fixture_only=fixture_only,
                        dataset_name=source_provenance.get("dataset_name"),
                        source_revision=source_provenance.get("source_revision"),
                        source_manifest_sha256=source_provenance.get("manifest_sha256"),
                        data_license=source_provenance.get("license"),
                        data_license_status=source_provenance.get("license_status"),
                        split_counts=split_counts,
                        hyperparameters_frozen_before_test=hyperparameters_frozen,
                        test_split_role=source_provenance.get("test_split_role"),
                        formal_evidence_eligible=formal_evidence_eligible,
                        core_support_eligible=(
                            formal_evidence_eligible
                            and source_provenance.get("test_split_role") == "ood"
                        ),
                        core_claim_classification="inconclusive",
                        core_claim_reason=(
                            "requires_seed_level_paired_aggregate_against_flat_local"
                        ),
                    )
                    run.record(metrics, condition=condition, task_family=family)
                    recorded.add(condition)
                except Exception as error:
                    run.mark_condition_failure(
                        error, condition=condition, task_family=family
                    )
                    recorded.add(condition)
            if results:
                assert_matched_candidate_panel(results)
                task_metrics = pd.concat(
                    [result.task_metrics for result in results], ignore_index=True
                )
                task_metrics.to_csv(run.path / "task_metrics.csv.gz", index=False)
            (run.path / "fit_receipts.json").write_text(
                json.dumps(fit_receipts, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception as error:
            for dimensions in planned:
                condition = str(dimensions["condition"])
                if condition not in recorded:
                    run.mark_condition_failure(error, **dimensions)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "Structured reasoning",
        "configs/smoke/exp13_structured_reasoning.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
