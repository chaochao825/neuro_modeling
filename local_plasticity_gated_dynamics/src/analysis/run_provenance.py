"""Portable, fail-closed provenance binding for formal experiment run sets."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd


EXP10_EXPERIMENT = "exp10_hidden_context_ei_bridge"
EXP10_SEEDS = tuple(range(30))
EXP10_RUN_FILES = (
    "config.json",
    "planned_conditions.json",
    "status.json",
    "manifest.json",
    "environment.json",
    "metrics.jsonl",
    "run.log",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DERIVED_SOURCE_COLUMNS = {
    "source_metrics_path",
    "source_run_attempt",
    "source_run_status",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_exp10_formal_attempts(results_root: Path) -> dict[int, list[Path]]:
    """Return every locally materialized formal attempt, including failures."""

    run_root = results_root / "runs" / EXP10_EXPERIMENT
    attempts: dict[int, list[Path]] = {}
    for seed in EXP10_SEEDS:
        seed_root = run_root / f"seed_{seed:04d}"
        current: list[Path] = []
        if seed_root.is_dir():
            for run_dir in seed_root.iterdir():
                config_path = run_dir / "config.json"
                if not config_path.is_file():
                    continue
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if str(config.get("profile")) == "formal":
                    current.append(run_dir)
        if current:
            attempts[seed] = current
    return attempts


def latest_exp10_formal_attempts(results_root: Path) -> dict[int, Path]:
    """Select the latest attempt per seed without filtering failed retries."""

    attempts = discover_exp10_formal_attempts(results_root)
    if not attempts:
        return {}
    missing = sorted(set(EXP10_SEEDS) - set(attempts))
    if missing:
        raise ValueError(
            "local exp10 formal attempt set is partial; refusing snapshot fallback "
            f"(missing seeds: {missing})"
        )
    return {
        seed: max(attempts[seed], key=lambda path: path.name) for seed in EXP10_SEEDS
    }


def _round_trip_jsonl(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    stream = io.StringIO()
    frame.to_csv(stream, index=False, lineterminator="\n")
    stream.seek(0)
    return pd.read_csv(stream, low_memory=False)


def canonical_seed_rows_sha256(raw: pd.DataFrame, seed: int) -> str:
    """Hash one seed's published scientific rows independent of source paths."""

    frame = raw.loc[raw["seed"].astype(int).eq(seed)].copy()
    frame = frame.drop(
        columns=sorted(_DERIVED_SOURCE_COLUMNS & set(frame.columns)), errors="ignore"
    )
    columns = sorted(frame.columns)
    frame = frame[columns]
    sort_columns = [
        name
        for name in (
            "cue_reliability",
            "context_hazard",
            "gate_model",
            "intervention",
        )
        if name in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort")
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    if not set(values).issubset({"true", "false"}):
        raise ValueError(f"exp10 formal {name} must contain only true/false")
    return values.eq("true")


def validate_exp10_checkpoint_contract(raw: pd.DataFrame) -> None:
    """Bind pipeline/intervention labels to recorded readout and gate reuse."""

    required = {
        "seed",
        "cue_reliability",
        "context_hazard",
        "gate_model",
        "intervention",
        "intervention_postfit",
        "intervention_reuses_intact_gate_checkpoint",
        "intervention_reuses_intact_readout",
        "intervention_reuses_intact_receiver",
        "readout_checkpoint_id",
        "gate_checkpoint_id",
        "network_initialization_id",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"exp10 formal checkpoint contract lacks columns: {missing}")
    is_intervention = raw["intervention"].astype(str).ne("none")
    for column in (
        "intervention_postfit",
        "intervention_reuses_intact_gate_checkpoint",
        "intervention_reuses_intact_readout",
        "intervention_reuses_intact_receiver",
    ):
        if not _strict_bool(raw[column], name=column).equals(is_intervention):
            raise ValueError(
                f"exp10 formal {column} does not match the intervention rows"
            )

    cell_keys = ["seed", "cue_reliability", "context_hazard"]
    for cell_key, cell in raw.groupby(cell_keys, sort=False):
        base = cell.loc[cell["intervention"].astype(str).eq("none")]
        md = cell.loc[cell["gate_model"].astype(str).eq("md_recurrent_belief")]
        if (
            len(base) != 4
            or base["gate_model"].astype(str).nunique() != 4
            or base["readout_checkpoint_id"].isna().any()
            or base["readout_checkpoint_id"].astype(str).nunique() != 4
            or len(md) != 4
            or md["readout_checkpoint_id"].isna().any()
            or md["readout_checkpoint_id"].astype(str).nunique() != 1
            or md["gate_checkpoint_id"].isna().any()
            or md["gate_checkpoint_id"].astype(str).nunique() != 1
            or cell["network_initialization_id"].isna().any()
            or cell["network_initialization_id"].astype(str).nunique() != 1
        ):
            raise ValueError(
                "exp10 formal checkpoint/readout reuse contract failed for "
                f"seed/q/h cell {cell_key}"
            )


def build_exp10_run_manifest(results_root: Path, raw: pd.DataFrame) -> pd.DataFrame:
    """Build a portable hash inventory from the latest clean 30-seed run set."""

    latest = latest_exp10_formal_attempts(results_root)
    if not latest:
        raise FileNotFoundError("no local exp10 formal attempts for run-manifest build")
    records: list[dict[str, object]] = []
    for seed, run_dir in latest.items():
        missing = [name for name in EXP10_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(
                f"latest exp10 formal seed {seed} lacks run artifacts: {missing}"
            )
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        run_manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        environment = json.loads(
            (run_dir / "environment.json").read_text(encoding="utf-8")
        )
        planned = json.loads(
            (run_dir / "planned_conditions.json").read_text(encoding="utf-8")
        )
        metrics_rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        git = environment.get("git", {})
        commit = str(git.get("commit", ""))
        dirty = git.get("dirty")
        run_id = str(run_manifest.get("run_id", ""))
        raw_seed = raw.loc[raw["seed"].astype(int).eq(seed)]
        if (
            str(config.get("profile")) != "formal"
            or int(config.get("seed", -1)) != seed
            or str(status.get("status")) != "complete"
            or str(run_manifest.get("profile")) != "formal"
            or int(run_manifest.get("seed", -1)) != seed
            or not _GIT_COMMIT.fullmatch(commit)
            or dirty is not False
            or not isinstance(planned, list)
            or len(planned) != 28
            or len(metrics_rows) != 28
            or raw_seed.shape[0] != 28
            or raw_seed["run_id"].astype(str).nunique() != 1
            or str(raw_seed["run_id"].iloc[0]) != run_id
            or {str(row.get("run_id", "")) for row in metrics_rows} != {run_id}
            or {str(row.get("status", "")) for row in metrics_rows} != {"complete"}
        ):
            raise ValueError(
                f"latest exp10 formal seed {seed} fails clean-run provenance contract"
            )
        metrics_round_trip = _round_trip_jsonl(metrics_rows)
        if canonical_seed_rows_sha256(metrics_round_trip, seed) != (
            canonical_seed_rows_sha256(raw, seed)
        ):
            raise ValueError(
                f"exp10 formal seed {seed} metrics do not reproduce scoped raw rows"
            )
        record: dict[str, object] = {
            "seed": seed,
            "run_id": run_id,
            "source_run_attempt": run_dir.name,
            "git_commit": commit,
            "git_dirty": False,
            "metrics_row_count": len(metrics_rows),
            "scoped_rows_sha256": canonical_seed_rows_sha256(raw, seed),
        }
        for name in EXP10_RUN_FILES:
            column = name.replace(".", "_") + "_sha256"
            record[column] = file_sha256(run_dir / name)
        records.append(record)
    manifest = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
    validate_exp10_run_manifest(manifest, raw)
    return manifest


def validate_exp10_run_manifest(manifest: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Verify the published clean-run inventory against the scoped raw table."""

    hash_columns = [name.replace(".", "_") + "_sha256" for name in EXP10_RUN_FILES]
    required = {
        "seed",
        "run_id",
        "source_run_attempt",
        "git_commit",
        "git_dirty",
        "metrics_row_count",
        "scoped_rows_sha256",
        *hash_columns,
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"exp10 run manifest lacks columns: {missing}")
    dirty = manifest["git_dirty"].astype(str).str.lower()
    if (
        len(manifest) != 30
        or set(manifest["seed"].astype(int)) != set(EXP10_SEEDS)
        or manifest["seed"].astype(int).duplicated().any()
        or manifest["run_id"].astype(str).nunique() != 30
        or manifest["git_commit"].astype(str).nunique() != 1
        or not _GIT_COMMIT.fullmatch(str(manifest["git_commit"].iloc[0]))
        or set(dirty) != {"false"}
        or set(manifest["metrics_row_count"].astype(int)) != {28}
    ):
        raise ValueError("exp10 run manifest violates the clean 30-seed contract")
    for column in ["scoped_rows_sha256", *hash_columns]:
        if (
            not manifest[column]
            .astype(str)
            .map(lambda value: bool(_SHA256.fullmatch(value)))
            .all()
        ):
            raise ValueError(f"exp10 run manifest contains invalid {column}")
    raw_run_ids = raw.groupby(raw["seed"].astype(int))["run_id"].agg(
        lambda values: set(values.astype(str))
    )
    for row in manifest.to_dict("records"):
        seed = int(row["seed"])
        if raw_run_ids.get(seed, set()) != {str(row["run_id"])}:
            raise ValueError(f"exp10 run manifest/raw run ID mismatch for seed {seed}")
        if str(row["scoped_rows_sha256"]) != canonical_seed_rows_sha256(raw, seed):
            raise ValueError(
                f"exp10 run manifest/raw row hash mismatch for seed {seed}"
            )
