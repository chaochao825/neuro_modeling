#!/usr/bin/env python3
"""Merge disjoint ORBIT feature shards without rewriting source artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.orbit_streaming import (
    FEATURE_MANIFEST_COLUMNS,
    OrbitFeatureStore,
    load_official_orbit_splits,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_failure_rows(path: Path, *, shard: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    frame["source_shard"] = str(shard)
    return frame


def merge_feature_shards(
    shard_roots: Iterable[str | Path],
    *,
    output_root: str | Path,
    split: str,
    official_splits_path: str | Path,
    require_complete_split: bool = True,
) -> Path:
    shards = tuple(Path(value).expanduser().resolve() for value in shard_roots)
    if len(shards) < 2 or len(shards) != len(set(shards)):
        raise ValueError("at least two unique feature shard roots are required")
    output = Path(output_root).expanduser().resolve()
    if output in shards:
        raise ValueError("merged output cannot overwrite a source shard")
    manifests = []
    failures = []
    provenance = []
    for shard in shards:
        manifest_path = shard / "feature_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"shard manifest is missing: {manifest_path}")
        frame = pd.read_csv(manifest_path, keep_default_na=False)
        missing = set(FEATURE_MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"shard manifest misses columns: {sorted(missing)}")
        frame = frame.loc[frame["split"].astype(str) == split].copy()
        if frame.empty:
            raise ValueError(f"shard has no {split} features: {shard}")
        frame["source_shard"] = str(shard)
        manifests.append(frame)
        failures.append(
            _read_failure_rows(shard / f"failures_{split}.csv", shard=shard)
        )
        provenance_path = shard / f"provenance_{split}.json"
        provenance.append(
            {
                "root": str(shard),
                "manifest_sha256": _sha256(manifest_path),
                "provenance_sha256": (
                    _sha256(provenance_path) if provenance_path.is_file() else None
                ),
            }
        )
    panel = pd.concat(manifests, ignore_index=True)
    key = ["split", "user_id", "object_name", "video_type", "video_id"]
    if panel.duplicated(key).any():
        raise ValueError("feature shards contain duplicate videos")
    if panel["feature_dim"].astype(int).nunique() != 1:
        raise ValueError("feature shards disagree on feature dimension")
    official = load_official_orbit_splits(official_splits_path)
    observed = set(panel["user_id"].astype(str))
    expected = set(official[split])
    if not observed <= expected:
        raise ValueError("feature shards contain users outside the official split")
    if require_complete_split and observed != expected:
        raise ValueError(f"merged split misses users: {sorted(expected - observed)}")

    output.mkdir(parents=True, exist_ok=True)
    for row in panel.to_dict("records"):
        relative = Path(str(row["feature_path"]))
        source = (Path(str(row["source_shard"])) / relative).resolve()
        destination = (output / relative).resolve()
        if not source.is_file() or not source.is_relative_to(
            Path(str(row["source_shard"]))
        ):
            raise FileNotFoundError(f"invalid shard feature path: {source}")
        if not destination.is_relative_to(output):
            raise ValueError(f"feature path escapes merged output: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256(destination) != _sha256(source):
                raise RuntimeError(f"existing merged feature differs: {destination}")
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    clean = panel.drop(columns="source_shard")[list(FEATURE_MANIFEST_COLUMNS)]
    clean = clean.sort_values(key).reset_index(drop=True)
    _atomic_csv(clean, output / "feature_manifest.csv")
    nonempty_failures = [frame for frame in failures if not frame.empty]
    failure_panel = (
        pd.concat(nonempty_failures, ignore_index=True)
        if nonempty_failures
        else pd.DataFrame()
    )
    _atomic_csv(failure_panel, output / f"failures_{split}.csv")
    _atomic_json(
        {
            "schema_version": "orbit-feature-shard-merge-v1",
            "split": split,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_shards": len(shards),
            "n_videos": len(clean),
            "n_users": len(observed),
            "n_failures": len(failure_panel),
            "feature_dim": int(clean["feature_dim"].iloc[0]),
            "sources": provenance,
        },
        output / f"provenance_{split}.json",
    )
    OrbitFeatureStore(
        output,
        split=split,
        official_splits_path=official_splits_path,
        require_complete_split=require_complete_split,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--official-splits", default="data/orbit_official_splits.json")
    parser.add_argument("--allow-incomplete-split", action="store_true")
    args = parser.parse_args()
    print(
        merge_feature_shards(
            args.shard_roots,
            output_root=args.output_root,
            split=args.split,
            official_splits_path=args.official_splits,
            require_complete_split=not args.allow_incomplete_split,
        )
    )


if __name__ == "__main__":
    main()
