"""Immutable experiment artifacts that retain successful and failed seeds."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import traceback
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _software_provenance() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "statsmodels",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    repository = Path(__file__).resolve().parents[2]
    git: dict[str, Any] = {"commit": None, "tree": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        git = {
            "commit": commit or None,
            "tree": tree or None,
            "dirty": bool(status.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    return {"packages": packages, "git": git}


def _safe_path_component(name: str, value: object, *, optional: bool = False) -> str:
    if optional and value is None:
        return ""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty path-safe string")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{name} must not contain path separators")
    if (
        any(character in '<>:"|?*' or ord(character) < 32 for character in value)
        or value.endswith((" ", "."))
        or value.split(".", maxsplit=1)[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
    ):
        raise ValueError(f"{name} must be a portable path-safe string")
    return value


class ExperimentRun(AbstractContextManager["ExperimentRun"]):
    """Context manager for one experiment/seed with failure-preserving output."""

    def __init__(
        self,
        experiment: str,
        seed: int,
        config: Mapping[str, Any],
        *,
        results_root: str | Path = "results",
        run_label: str | None = None,
    ) -> None:
        experiment = _safe_path_component("experiment", experiment)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        if {"experiment", "seed", "run_id", "run_label"} & set(config):
            raise ValueError("config must not redefine reserved provenance fields")
        if run_label is not None:
            _safe_path_component("run_label", run_label)
        config_payload = _jsonable(dict(config))
        json.dumps(config_payload, sort_keys=True, ensure_ascii=False)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        label = f"_{run_label}" if run_label else ""
        self.path = (
            Path(results_root)
            / "runs"
            / experiment
            / f"seed_{seed:04d}"
            / f"{stamp}{label}"
        )
        self.path.mkdir(parents=True, exist_ok=False)
        self.run_id = str(uuid.uuid4())
        self.experiment = experiment
        self.seed = seed
        self.run_label = run_label
        self.config = config_payload
        self.metrics_path = self.path / "metrics.jsonl"
        self.metrics_path.touch()
        self.logger = logging.getLogger(f"{experiment}.{seed}.{stamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(self.path / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self._handler = handler
        self.started_at = datetime.now(timezone.utc)
        self._condition_failures = 0
        self._condition_invalid = 0
        self._entered = False
        self._closed = False

        _write_json(
            self.path / "config.json",
            {
                "experiment": experiment,
                "seed": seed,
                **({"run_label": run_label} if run_label is not None else {}),
                **self.config,
            },
        )
        _write_json(
            self.path / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
                **_software_provenance(),
            },
        )
        _write_json(
            self.path / "status.json",
            {
                "status": "running",
                "seed": seed,
                **({"run_label": run_label} if run_label is not None else {}),
                "started_at": self.started_at.isoformat(),
            },
        )
        _write_json(
            self.path / "manifest.json",
            {
                "schema_version": "1.0",
                "run_id": self.run_id,
                "experiment": experiment,
                "seed": seed,
                **({"run_label": run_label} if run_label is not None else {}),
                "profile": self.config.get("profile", "unspecified"),
                "training_algorithm": self.config.get("training_algorithm"),
                "used_autograd": self.config.get("used_autograd"),
                "parent_checkpoint": self.config.get("parent_checkpoint"),
                "evidence_provenance": self.config.get("evidence_provenance"),
                "status": "running",
                "started_at": self.started_at.isoformat(),
            },
        )

    def __enter__(self) -> "ExperimentRun":
        if self._closed:
            raise RuntimeError("experiment run is already finalized")
        if self._entered:
            raise RuntimeError("experiment run context has already been entered")
        self._entered = True
        self.logger.info("run started")
        return self

    def _ensure_mutable(self) -> None:
        if self._closed:
            raise RuntimeError("experiment run is finalized and immutable")

    def record(self, metrics: Mapping[str, Any], **dimensions: Any) -> None:
        """Append one raw metric record; callers may record failed conditions too."""

        self._ensure_mutable()
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        reserved = {"run_id", "experiment", "seed", "recorded_at"}
        conflicts = reserved & (set(metrics) | set(dimensions))
        if conflicts:
            raise ValueError(
                f"records cannot override reserved fields: {sorted(conflicts)}"
            )
        overlap = set(metrics) & set(dimensions)
        if overlap:
            raise ValueError(f"metric and dimension keys overlap: {sorted(overlap)}")
        payload = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "seed": self.seed,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **dimensions,
            **metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False)
                + "\n"
            )

    def register_conditions(self, conditions: list[Mapping[str, Any]]) -> None:
        """Persist the complete planned grid before any condition is run."""

        self._ensure_mutable()
        if not conditions:
            raise ValueError("conditions must be a non-empty list")
        planned_path = self.path / "planned_conditions.json"
        if planned_path.exists():
            raise RuntimeError("planned conditions have already been registered")
        normalized = []
        seen: set[str] = set()
        for index, condition in enumerate(conditions):
            if not isinstance(condition, Mapping):
                raise TypeError("every planned condition must be a mapping")
            payload = _jsonable(dict(condition))
            if "condition_index" in payload:
                raise ValueError("condition payload cannot override condition_index")
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if encoded in seen:
                raise ValueError(f"duplicate planned condition at index {index}")
            seen.add(encoded)
            normalized.append({"condition_index": index, **payload})
        _write_json(planned_path, normalized)

    def mark_condition_failure(self, error: BaseException, **dimensions: Any) -> None:
        self._ensure_mutable()
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        self._condition_failures += 1
        self.logger.error(
            "condition failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self.record(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            **dimensions,
        )

    def record_failed_condition(
        self, metrics: Mapping[str, Any], **dimensions: Any
    ) -> None:
        """Retain computed diagnostics for a scientifically failed condition."""

        payload = dict(metrics)
        payload["status"] = "failed"
        self._condition_failures += 1
        self.logger.warning(
            "condition failed scientific gate: %s",
            payload.get("failure_reason", "unspecified"),
        )
        self.record(payload, **dimensions)

    def mark_condition_invalid(self, reason: str, **dimensions: Any) -> None:
        """Retain a mathematically impossible or inapplicable planned cell."""

        self._ensure_mutable()
        if not isinstance(reason, str) or not reason:
            raise ValueError("invalid-condition reason must be non-empty")
        self._condition_invalid += 1
        self.logger.warning("invalid condition: %s", reason)
        self.record({"status": "invalid", "reason": reason}, **dimensions)

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        ended_at = datetime.now(timezone.utc)
        final_status = (
            "failed"
            if exc is not None
            else (
                "complete_with_failures"
                if self._condition_failures or self._condition_invalid
                else "complete"
            )
        )
        status: dict[str, Any] = {
            "status": final_status,
            "seed": self.seed,
            **(
                {"run_label": self.run_label}
                if self.run_label is not None
                else {}
            ),
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - self.started_at).total_seconds(),
            "condition_failures": self._condition_failures,
            "condition_invalid": self._condition_invalid,
        }
        if exc is not None:
            status.update(
                error_type=type(exc).__name__,
                error=str(exc),
                traceback="".join(traceback.format_exception(exc_type, exc, tb)),
            )
            self.logger.error("run failed: %s", exc)
        else:
            self.logger.info("run completed")
        try:
            _write_json(self.path / "status.json", status)
            manifest_path = self.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                status=final_status,
                ended_at=ended_at.isoformat(),
                condition_failures=self._condition_failures,
                condition_invalid=self._condition_invalid,
            )
            _write_json(manifest_path, manifest)
        except Exception:
            self.logger.exception("failed to finalize run artifacts")
            if exc is None:
                raise
        finally:
            self._handler.close()
            self.logger.removeHandler(self._handler)
            self._closed = True
        return False
