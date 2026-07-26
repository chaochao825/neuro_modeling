#!/usr/bin/env python3
"""Validate and embed the preregistered CORe50 archive without modifying it."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "exp37_core50_change_aware_prefix_v1"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
FRAME_NUMBER = re.compile(r"(\d+)(?=\.[^.]+$)")
MANIFEST_COLUMNS = (
    "session_id",
    "object_id",
    "feature_path",
    "n_frames",
    "feature_dim",
    "source_fingerprint",
)


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if isinstance(chunk_bytes, bool) or int(chunk_bytes) < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(int(chunk_bytes)):
            digest.update(block)
    return digest.hexdigest()


def validate_preregistration(config_path: Path) -> dict[str, Any]:
    receipt_path = PROJECT_ROOT / "provenance/exp37_preregistration_receipt_20260726.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp37 preregistration version mismatch")
    for path, key in (
        (PROJECT_ROOT / str(receipt["protocol_path"]), "protocol_sha256"),
        (config_path.resolve(), "config_sha256"),
        (PROJECT_ROOT / str(receipt["cohort_path"]), "cohort_sha256"),
    ):
        if _sha256(path) != str(receipt[key]):
            raise ValueError(f"Exp37 preregistration hash mismatch: {path}")
    return receipt


def _natural_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for piece in re.split(r"(\d+)", path.name):
        parts.append(int(piece) if piece.isdigit() else piece.lower())
    return tuple(parts)


def locate_dataset_root(raw_root: str | Path, *, expected_sessions: Iterable[str]) -> Path:
    base = Path(raw_root).expanduser().resolve()
    sessions = tuple(map(str, expected_sessions))
    candidates: list[Path] = []
    if all((base / session).is_dir() for session in sessions):
        candidates.append(base)
    if base.is_dir():
        for child in base.rglob(sessions[0]):
            if child.is_dir() and all((child.parent / session).is_dir() for session in sessions):
                candidates.append(child.parent)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(f"expected one CORe50 dataset root, found {len(unique)}")
    return unique[0]


def discover_core50_cells(
    raw_root: str | Path,
    *,
    expected_sessions: Iterable[str],
    expected_objects: Iterable[str],
) -> tuple[Path, list[tuple[str, str, tuple[Path, ...]]]]:
    sessions = tuple(map(str, expected_sessions))
    objects = tuple(map(str, expected_objects))
    dataset_root = locate_dataset_root(raw_root, expected_sessions=sessions)
    cells: list[tuple[str, str, tuple[Path, ...]]] = []
    seen_paths: set[Path] = set()
    for session in sessions:
        observed_objects = {path.name for path in (dataset_root / session).iterdir() if path.is_dir()}
        if observed_objects != set(objects):
            raise ValueError(f"{session} object schema mismatch")
        for object_id in objects:
            images = tuple(
                sorted(
                    (
                        path.resolve()
                        for path in (dataset_root / session / object_id).iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    ),
                    key=_natural_key,
                )
            )
            if not images:
                raise ValueError(f"empty CORe50 cell: {session}/{object_id}")
            duplicates = seen_paths.intersection(images)
            if duplicates:
                raise ValueError("CORe50 image path appears in multiple cells")
            if any(path.stat().st_size < 1 for path in images):
                raise ValueError(f"empty CORe50 image file: {session}/{object_id}")
            seen_paths.update(images)
            cells.append((session, object_id, images))
    if len(cells) != len(sessions) * len(objects):
        raise RuntimeError("CORe50 schema coverage mismatch")
    return dataset_root, cells


def _source_fingerprint(paths: Sequence[Path], encoder_identity: str) -> str:
    digest = hashlib.sha256(encoder_identity.encode("utf-8"))
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_encoder(device: str) -> tuple[Any, Any, int, str]:
    try:
        import torch
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("feature extraction requires torch, torchvision, Pillow") from error
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().requires_grad_(False).to(device)
    identity = (
        "torchvision::efficientnet_b0::"
        f"{weights.__class__.__name__}.{weights.name}"
    )
    return model, weights.transforms(), 1280, identity


def _embed_images(
    paths: Sequence[Path],
    *,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
    decode_workers: int,
) -> np.ndarray:
    import torch
    from PIL import Image

    def load(path: Path) -> Any:
        with Image.open(path) as image:
            return transform(image.convert("RGB"))

    outputs: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=max(1, decode_workers)) as executor:
        for start in range(0, len(paths), batch_size):
            tensors = list(executor.map(load, paths[start : start + batch_size]))
            batch = torch.stack(tensors).to(device, non_blocking=True)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if str(device).startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                encoded = model(batch)
            outputs.append(encoded.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    receipt = validate_preregistration(args.config)
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp37 config version mismatch")
    archive = Path(str(config["archive_path"])).expanduser().resolve()
    if archive.stat().st_size != int(config["archive_content_length"]):
        raise ValueError("CORe50 archive content length mismatch")
    archive_sha256 = _sha256(archive)
    transport_path = archive.parent / "acquisition_transport_attestation.json"
    if not transport_path.is_file():
        raise FileNotFoundError(f"CORe50 transport attestation missing: {transport_path}")
    transport = json.loads(transport_path.read_text(encoding="utf-8"))
    if (
        transport.get("protocol_version") != PROTOCOL_VERSION
        or int(transport.get("archive_content_length", -1)) != archive.stat().st_size
        or transport.get("archive_observed_sha256") != archive_sha256
        or transport.get("outcomes_inspected") is not False
    ):
        raise ValueError("CORe50 transport attestation does not match the archive")
    dataset_root, cells = discover_core50_cells(
        config["raw_root"],
        expected_sessions=config["support_sessions"]
        + config["development_sessions"]
        + config["external_sessions"],
        expected_objects=config["objects"],
    )
    schema = pd.DataFrame(
        [
            {
                "session_id": session,
                "object_id": object_id,
                "n_images": len(paths),
                "first_image": paths[0].name,
                "last_image": paths[-1].name,
            }
            for session, object_id, paths in cells
        ]
    )
    output = Path(str(config["feature_root"])).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(schema, output / "schema_audit.csv")
    acquisition = {
        "protocol_version": PROTOCOL_VERSION,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "archive_url": config["archive_url"],
        "archive_content_length": int(archive.stat().st_size),
        "archive_sha256": archive_sha256,
        "transport_attestation_path": str(transport_path),
        "actual_transport": str(transport.get("actual_transport")),
        "torrent_metadata_sha256": str(transport.get("torrent_metadata_sha256")),
        "preregistered_http_metadata": {
            "etag": receipt["archive_etag"],
            "last_modified": receipt["archive_last_modified"],
        },
        "dataset_root": str(dataset_root),
        "n_sessions": int(schema["session_id"].nunique()),
        "n_objects": int(schema["object_id"].nunique()),
        "n_cells": int(len(schema)),
        "n_images": int(schema["n_images"].sum()),
        "schema_complete": True,
        "outcomes_inspected": False,
    }
    (output / "acquisition_attestation.json").write_text(
        json.dumps(acquisition, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.validate_only:
        print(json.dumps(acquisition, indent=2, sort_keys=True))
        return

    if args.batch_size < 1 or args.decode_workers < 0:
        raise ValueError("batch-size must be positive and decode-workers non-negative")
    model, transform, feature_dim, identity = _build_encoder(args.device)
    manifest_path = output / "feature_manifest.csv"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if manifest_path.is_file():
        prior = pd.read_csv(manifest_path, keep_default_na=False)
        existing = {
            (str(row.session_id), str(row.object_id)): row._asdict()
            for row in prior.itertuples(index=False)
        }
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (session, object_id, paths) in enumerate(cells, start=1):
        relative = Path(session) / f"{object_id}.npz"
        destination = output / relative
        fingerprint = _source_fingerprint(paths, identity)
        cached = existing.get((session, object_id))
        if (
            cached is not None
            and str(cached.get("source_fingerprint")) == fingerprint
            and destination.is_file()
        ):
            try:
                with np.load(destination, allow_pickle=False) as payload:
                    shape = np.asarray(payload["features"]).shape
                if shape == (len(paths), feature_dim):
                    completed.append({key: cached[key] for key in MANIFEST_COLUMNS})
                    print(f"[{index}/{len(cells)}] cached {session}/{object_id}", flush=True)
                    continue
            except (OSError, KeyError, ValueError):
                pass
        try:
            features = _embed_images(
                paths,
                model=model,
                transform=transform,
                device=args.device,
                batch_size=args.batch_size,
                decode_workers=args.decode_workers,
            )
            if features.shape != (len(paths), feature_dim) or not np.all(np.isfinite(features)):
                raise ValueError("encoder returned invalid features")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                features=features,
                frame_names=np.asarray([path.name for path in paths]),
            )
            temporary.replace(destination)
            completed.append(
                {
                    "session_id": session,
                    "object_id": object_id,
                    "feature_path": relative.as_posix(),
                    "n_frames": len(paths),
                    "feature_dim": feature_dim,
                    "source_fingerprint": fingerprint,
                }
            )
            print(f"[{index}/{len(cells)}] embedded {session}/{object_id}", flush=True)
        except Exception as error:  # noqa: BLE001 - every failed cell is retained
            failures.append(
                {
                    "session_id": session,
                    "object_id": object_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"[{index}/{len(cells)}] FAILED {session}/{object_id}: {error}", flush=True)
        _atomic_csv(pd.DataFrame(completed, columns=MANIFEST_COLUMNS), manifest_path)
        _atomic_csv(
            pd.DataFrame(
                failures, columns=("session_id", "object_id", "error_type", "error")
            ),
            output / "failures.csv",
        )

    provenance = {
        **acquisition,
        "encoder_identity": identity,
        "feature_dim": feature_dim,
        "n_completed_cells": len(completed),
        "n_failed_cells": len(failures),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": args.device,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if failures or len(completed) != len(cells):
        raise RuntimeError(
            f"CORe50 extraction incomplete: {len(completed)}/{len(cells)} cells"
        )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
