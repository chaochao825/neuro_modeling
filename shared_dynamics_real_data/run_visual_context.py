"""Run leakage-safe visual responding/spontaneous LDS comparisons.

Every attempted seed/fold/dimension/model cell is written immediately to an
immutable JSONL run directory.  A failed cell remains in the raw metrics.
Computational seeds sample matched neuron subsets; they are robustness checks,
not biological replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from shared_dynamics_real_data.data import BUNDLED_DATASETS, load_visual_context_pair
from shared_dynamics_real_data.pipeline import SharedDynamicsPipeline
from shared_dynamics_real_data.splits import BlockFold, purged_contiguous_folds


EXPERIMENT = "visual_context_shared_lds"


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {"tests", "build", "__pycache__"} for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256(path: Path, *, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_seed_override(text: str | None, configured: Iterable[int]) -> list[int]:
    source = configured if not text else (part.strip() for part in text.split(","))
    seeds = [int(value) for value in source if str(value).strip()]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct non-negative integers")
    return seeds


def _validate_model_specs(raw_specs: Any) -> list[dict[str, str]]:
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("model_specs must be a non-empty list")
    specs: list[dict[str, str]] = []
    names: set[str] = set()
    for raw in raw_specs:
        if not isinstance(raw, dict):
            raise TypeError("each model spec must be an object")
        spec = {
            "model": str(raw.get("model", "")),
            "family": str(raw.get("family", "")),
            "basis_control": str(raw.get("basis_control", "")),
        }
        if not all(spec.values()) or spec["model"] in names:
            raise ValueError("model specs need unique non-empty names and fields")
        if spec["family"] not in {"common", "shared", "separate"}:
            raise ValueError(f"invalid family in {spec}")
        if spec["basis_control"] not in {
            "aligned",
            "random",
            "orthogonal",
            "shuffled",
        }:
            raise ValueError(f"invalid basis control in {spec}")
        specs.append(spec)
        names.add(spec["model"])
    return specs


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "profile",
        "seeds",
        "n_units",
        "n_splits",
        "purge",
        "latent_dims",
        "ridge",
        "variance_floor",
        "top_k",
        "model_specs",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    normalized = dict(config)
    normalized["profile"] = str(config["profile"])
    normalized["seeds"] = _parse_seed_override(None, config["seeds"])
    normalized["model_specs"] = _validate_model_specs(config["model_specs"])
    for key in ("n_units", "n_splits", "purge", "top_k"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        normalized[key] = int(value)
    dims = config["latent_dims"]
    if not isinstance(dims, list) or not dims:
        raise ValueError("latent_dims must be a non-empty list")
    normalized["latent_dims"] = sorted({int(value) for value in dims})
    if (
        normalized["n_units"] < 2
        or normalized["n_splits"] < 2
        or normalized["purge"] < 0
        or normalized["top_k"] < 1
        or any(value < 1 for value in normalized["latent_dims"])
    ):
        raise ValueError("unit/fold/dimension options are outside valid ranges")
    normalized["ridge"] = float(config["ridge"])
    normalized["variance_floor"] = float(config["variance_floor"])
    if normalized["ridge"] < 0 or normalized["variance_floor"] <= 0:
        raise ValueError("ridge must be non-negative and variance_floor positive")
    return normalized


def _environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "statsmodels", "torch"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not-installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "packages": versions,
    }


def _data_manifest(data_root: Path, recordings: dict[str, np.ndarray]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for context, values in recordings.items():
        path = data_root / BUNDLED_DATASETS[context]
        entries[context] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "shape_time_units": list(values.shape),
            "dtype": str(values.dtype),
            "mean_activity": float(values.mean()),
        }
    manifest = {
        "dataset_pair": "visual_responding__visual_spontaneous",
        "aligned_unit_count": int(next(iter(recordings.values())).shape[1]),
        "unit_alignment_assumption": (
            "responding/spontaneous row order is aligned as documented upstream; "
            "the MAT files contain no independent unit identifiers"
        ),
        "recordings": entries,
        "metadata_available": ["binary_activity", "context_recording_label"],
        "metadata_missing": [
            "trial",
            "stimulus_identity",
            "behavior",
            "animal_id",
            "ei_identity",
            "natural_switch_time",
        ],
    }
    manifest["data_fingerprint"] = _fingerprint(
        {
            context: {"sha256_or_status": entry["sha256"]}
            for context, entry in entries.items()
        }
    )
    return manifest


def _attempted_data_manifest(data_root: Path) -> dict[str, Any]:
    """Fingerprint the requested files before parsing so failures stay aggregable."""

    entries: dict[str, Any] = {}
    for context in ("visual_responding", "visual_spontaneous"):
        filename = BUNDLED_DATASETS[context]
        path = data_root / filename
        try:
            sha256 = _sha256(path) if path.is_file() else "missing"
        except OSError as error:
            sha256 = f"unreadable:{type(error).__name__}"
        entries[context] = {
            "filename": filename,
            "exists": path.is_file(),
            "sha256_or_status": sha256,
        }
    return {
        "dataset_pair": "visual_responding__visual_spontaneous",
        "load_status": "attempted",
        "recordings": entries,
        "data_fingerprint": _fingerprint(
            {
                context: {
                    "sha256_or_status": entry["sha256_or_status"],
                }
                for context, entry in entries.items()
            }
        ),
    }


def _planned_conditions(
    seeds: list[int], config: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "computational_seed": seed,
            "fold": fold,
            "latent_dim": latent_dim,
            **spec,
        }
        for seed in seeds
        for fold in range(config["n_splits"])
        for latent_dim in config["latent_dims"]
        for spec in config["model_specs"]
    ]


def _unit_subset(n_total: int, n_units: int, seed: int) -> np.ndarray:
    if n_units > n_total:
        raise ValueError(f"requested {n_units} units from a population of {n_total}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_units, replace=False)).astype(np.int64)


def _fold_provenance(fold: BlockFold) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for context in sorted({segment.context for segment in fold.test}):
        test = next(segment for segment in fold.test if segment.context == context)
        train = [segment for segment in fold.train if segment.context == context]
        result[context] = {
            "test_start": int(test.indices[0]),
            "test_stop_exclusive": int(test.indices[-1] + 1),
            "test_timepoints": int(test.values.shape[0]),
            "train_segments": [
                [int(segment.indices[0]), int(segment.indices[-1] + 1)]
                for segment in train
            ],
            "train_timepoints": int(sum(segment.values.shape[0] for segment in train)),
        }
    return result


def run_experiment(
    config: dict[str, Any],
    *,
    data_root: Path,
    results_root: Path,
    seeds: list[int],
) -> Path:
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError("run seeds must be a subset of config['seeds']")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = results_root / "runs" / f"{stamp}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    analysis_payload = {
        key: config[key]
        for key in (
            "profile",
            "n_units",
            "n_splits",
            "purge",
            "latent_dims",
            "ridge",
            "variance_floor",
            "top_k",
            "model_specs",
        )
    }
    analysis_payload.update(
        {
            "experiment": EXPERIMENT,
            "time_split": "purged_contiguous_block",
            "preprocessing_fit_scope": "train_only",
            "algorithm_version": "shared-basis-woodbury-v1",
            "configured_seeds": config["seeds"],
            "source_fingerprint": _source_fingerprint(),
        }
    )
    analysis_fingerprint = _fingerprint(analysis_payload)
    resolved_config = {
        **config,
        "seeds": seeds,
        "configured_seeds": config["seeds"],
        "data_root": str(data_root.resolve()),
        "results_root": str(results_root.resolve()),
        "experiment": EXPERIMENT,
        "stats_unit": "recording_pair",
        "biological_inference_allowed": False,
        "time_split": "purged_contiguous_block",
        "preprocessing_fit_scope": "train_only",
        "algorithm_version": "shared-basis-woodbury-v1",
        "source_fingerprint": analysis_payload["source_fingerprint"],
        "analysis_fingerprint": analysis_fingerprint,
    }
    _write_json(run_dir / "config.json", resolved_config)
    _write_json(run_dir / "environment.json", _environment())
    planned = _planned_conditions(seeds, config)
    _write_json(run_dir / "planned_conditions.json", planned)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    unit_subsets: dict[str, list[int]] = {}
    log_path = run_dir / "run.log"
    metrics_path = run_dir / "metrics.jsonl"
    run_error: BaseException | None = None
    with log_path.open("w", encoding="utf-8", buffering=1) as log, metrics_path.open(
        "w", encoding="utf-8", buffering=1
    ) as raw:
        try:
            # This manifest exists even if MAT parsing fails; a successful load
            # replaces it with shape and alignment metadata below.
            _write_json(run_dir / "data_manifest.json", _attempted_data_manifest(data_root))
            recordings = load_visual_context_pair(data_root)
            manifest = _data_manifest(data_root, recordings)
            _write_json(run_dir / "data_manifest.json", manifest)
            n_total = int(next(iter(recordings.values())).shape[1])
            for seed in seeds:
                indices = _unit_subset(n_total, config["n_units"], seed)
                unit_subsets[str(seed)] = indices.tolist()
                subset_hash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
                selected = {
                    context: values[:, indices]
                    for context, values in recordings.items()
                }
                folds = purged_contiguous_folds(
                    selected,
                    n_splits=config["n_splits"],
                    purge=config["purge"],
                )
                for fold in folds:
                    provenance = _fold_provenance(fold)
                    for latent_dim in config["latent_dims"]:
                        for spec in config["model_specs"]:
                            cell_started = time.perf_counter()
                            row: dict[str, Any] = {
                                "experiment": EXPERIMENT,
                                "profile": config["profile"],
                                "dataset_pair": manifest["dataset_pair"],
                                "stats_unit": "recording_pair",
                                "biological_unit_count": 1,
                                "biological_inference_allowed": False,
                                "analysis_fingerprint": analysis_fingerprint,
                                "data_fingerprint": manifest["data_fingerprint"],
                                "computational_seed": seed,
                                "configured_seed_universe": json.dumps(
                                    config["seeds"], separators=(",", ":")
                                ),
                                "unit_subset_sha256": subset_hash,
                                "n_units": config["n_units"],
                                "fold": fold.fold,
                                "purge": fold.purge,
                                "fold_provenance": json.dumps(
                                    provenance, sort_keys=True, separators=(",", ":")
                                ),
                                "latent_dim": latent_dim,
                                **spec,
                                "status": "running",
                                "error_type": "",
                                "error_message": "",
                            }
                            try:
                                pipeline = SharedDynamicsPipeline(
                                    family=spec["family"],
                                    latent_dim=latent_dim,
                                    max_units=None,
                                    basis_control=spec["basis_control"],
                                    random_state=seed * 1009 + fold.fold,
                                    ridge=config["ridge"],
                                    variance_floor=config["variance_floor"],
                                ).fit(fold.train)
                                score = pipeline.score(fold.test)
                                row.update(asdict(score))
                                row["top_k"] = min(config["top_k"], latent_dim)
                                row["top_k_singular_energy"] = (
                                    pipeline.model_.mean_topk_singular_energy(row["top_k"])
                                )
                                row["selected_unit_count"] = int(
                                    pipeline.preprocessor_.unit_indices_.size
                                )
                                row["train_preprocessing_timepoints"] = int(
                                    pipeline.preprocessor_.fit_timepoints_
                                )
                                row["status"] = "complete"
                            except Exception as error:  # retain every failed cell
                                row["status"] = "failed"
                                row["error_type"] = type(error).__name__
                                row["error_message"] = str(error)
                                log.write(
                                    f"cell failure seed={seed} fold={fold.fold} "
                                    f"d={latent_dim} model={spec['model']}\n"
                                )
                                log.write(traceback.format_exc() + "\n")
                            row["elapsed_seconds"] = time.perf_counter() - cell_started
                            rows.append(row)
                            raw.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                            log.write(
                                f"{row['status']} seed={seed} fold={fold.fold} "
                                f"d={latent_dim} model={spec['model']} "
                                f"elapsed={row['elapsed_seconds']:.3f}s\n"
                            )
            _write_json(run_dir / "unit_subsets.json", unit_subsets)
        except BaseException as error:  # save top-level data/config failures too
            run_error = error
            log.write(traceback.format_exc() + "\n")

    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "metrics.csv", index=False)
    complete = int((frame.get("status", pd.Series(dtype=str)) == "complete").sum())
    failed = int((frame.get("status", pd.Series(dtype=str)) == "failed").sum())
    status = {
        "experiment": EXPERIMENT,
        "profile": config["profile"],
        "started_utc": stamp,
        "elapsed_seconds": time.perf_counter() - started,
        "planned_cells": len(planned),
        "recorded_cells": int(len(frame)),
        "complete_cells": complete,
        "failed_cells": failed,
        "status": (
            "failed"
            if run_error is not None
            else "complete"
            if len(frame) == len(planned) and failed == 0
            else "complete_with_failures"
        ),
        "top_level_error_type": type(run_error).__name__ if run_error else "",
        "top_level_error_message": str(run_error) if run_error else "",
    }
    _write_json(run_dir / "status.json", status)
    if run_error is not None:
        raise RuntimeError(f"run failed; artifacts retained at {run_dir}") from run_error
    return run_dir


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "formal.json",
    )
    parser.add_argument("--data-root", type=Path, default=root / "minimal_computation_original")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated override.")
    args = parser.parse_args()
    config = validate_config(_read_json(args.config))
    seeds = _parse_seed_override(args.seeds, config["seeds"])
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError(
            "--seeds may select only a subset of the configured seed universe"
        )
    run_dir = run_experiment(
        config,
        data_root=args.data_root,
        results_root=args.results_root,
        seeds=seeds,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
