#!/usr/bin/env python3
"""Build a resumable, fail-closed ORBIT per-video embedding cache.

The raw benchmark is never modified.  Each video is embedded independently
and written as a compressed NumPy file.  A manifest is atomically refreshed
after every successful video, while failures are retained in a separate CSV.
The extractor is frozen and preprocessing is fixed by its published weights.
"""

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

from src.data.orbit_streaming import (
    FEATURE_MANIFEST_COLUMNS,
    load_orbit_external_cohort,
    load_official_orbit_splits,
)


FRAME_PATTERN = re.compile(r"-(\d+)\.jpe?g$", re.IGNORECASE)
VIDEO_TYPES = ("clean", "clutter")
ENCODERS = ("efficientnet_b0", "efficientnet_v2_s", "vit_b_32")


def parse_frame_index(path: str | Path) -> int:
    match = FRAME_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"cannot parse ORBIT frame index: {path}")
    value = int(match.group(1))
    if value < 0:
        raise ValueError("frame index must be non-negative")
    return value


def parse_user_ids(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    users = tuple(item.strip() for item in value.split(",") if item.strip())
    if not users or len(users) != len(set(users)):
        raise ValueError("user-ids must be a non-empty unique comma-separated list")
    return users


def file_md5(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming MD5 for comparison with a dataset publisher checksum."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source archive not found: {source}")
    if isinstance(chunk_bytes, bool) or int(chunk_bytes) < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.md5(usedforsecurity=False)
    with source.open("rb") as stream:
        while chunk := stream.read(int(chunk_bytes)):
            digest.update(chunk)
    return digest.hexdigest()


def discover_orbit_videos(
    raw_root: str | Path,
    *,
    split: str,
    allowed_users: Iterable[str],
) -> list[tuple[str, str, str, Path]]:
    """Discover canonical ``user/object/type/video`` directories."""

    base = Path(raw_root).expanduser().resolve()
    root = base / ("Dataset" if split == "external" else split)
    if not root.is_dir():
        raise FileNotFoundError(f"ORBIT split directory not found: {root}")
    expected = set(map(str, allowed_users))
    observed = {path.name for path in root.iterdir() if path.is_dir()}
    unexpected = observed - expected
    if unexpected:
        raise ValueError(
            f"raw {split} directory contains users outside official split: "
            f"{sorted(unexpected)}"
        )
    videos: list[tuple[str, str, str, Path]] = []
    for user_id in sorted(observed):
        for object_dir in sorted(
            (root / user_id).iterdir(), key=lambda item: item.name
        ):
            if not object_dir.is_dir():
                continue
            for video_type in VIDEO_TYPES:
                type_dir = object_dir / video_type
                if not type_dir.is_dir():
                    continue
                for video_dir in sorted(type_dir.iterdir(), key=lambda item: item.name):
                    if video_dir.is_dir():
                        videos.append((user_id, object_dir.name, video_type, video_dir))
    if not videos:
        raise ValueError(f"no ORBIT videos found under {root}")
    return videos


def _frame_paths(video_dir: Path, *, max_frames: int) -> list[Path]:
    paths = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and FRAME_PATTERN.search(path.name)
    ]
    paths.sort(key=parse_frame_index)
    if not paths:
        raise ValueError(f"video contains no numbered JPEG frames: {video_dir}")
    return paths[:max_frames]


def _annotation_mask(
    frame_paths: Sequence[Path],
    *,
    annotations_root: Path,
    split: str,
    video_id: str,
) -> np.ndarray:
    path = (
        annotations_root / f"{video_id}.json"
        if split == "external"
        else annotations_root / split / f"{video_id}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"ORBIT frame annotations not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"annotation file is not a JSON object: {path}")
    present: list[bool] = []
    for frame in frame_paths:
        annotation = payload.get(frame.name)
        if not isinstance(annotation, dict):
            raise ValueError(f"annotation missing for {frame.name}")
        flag = annotation.get("object_not_present_issue")
        if not isinstance(flag, bool):
            raise ValueError(
                f"object_not_present_issue is not Boolean for {frame.name}"
            )
        present.append(not flag)
    return np.asarray(present, dtype=np.bool_)


def _protocol_object_present_mask(
    frame_paths: Sequence[Path],
    *,
    video_type: str,
    annotations_root: Path,
    split: str,
    video_id: str,
) -> np.ndarray:
    """Use extra annotations only where the ORBIT query protocol permits."""

    if video_type == "clean":
        # ORBIT forbids extra clean-frame annotations during personalization.
        return np.ones(len(frame_paths), dtype=np.bool_)
    if video_type != "clutter":
        raise ValueError("video_type must be clean or clutter")
    return _annotation_mask(
        frame_paths,
        annotations_root=annotations_root,
        split=split,
        video_id=video_id,
    )


def _source_fingerprint(
    frame_paths: Sequence[Path],
    annotation_path: Path | None,
    encoder_identity: str,
) -> str:
    digest = hashlib.sha256(encoder_identity.encode("utf-8"))
    paths = (
        (*frame_paths, annotation_path) if annotation_path is not None else frame_paths
    )
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_encoder(name: str, device: str) -> tuple[Any, Any, int, str]:
    try:
        import torch
        from torchvision.models import (
            EfficientNet_B0_Weights,
            EfficientNet_V2_S_Weights,
            ViT_B_32_Weights,
            efficientnet_b0,
            efficientnet_v2_s,
            vit_b_32,
        )
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError(
            "feature extraction requires torch, torchvision, and Pillow"
        ) from error

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Identity()
        feature_dim = 1280
    elif name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        model.classifier = torch.nn.Identity()
        feature_dim = 1280
    elif name == "vit_b_32":
        weights = ViT_B_32_Weights.IMAGENET1K_V1
        model = vit_b_32(weights=weights)
        model.heads = torch.nn.Identity()
        feature_dim = 768
    else:
        raise ValueError(f"unknown encoder: {name}")
    model.eval().requires_grad_(False).to(device)
    identity = f"torchvision::{name}::{weights.__class__.__name__}.{weights.name}"
    return model, weights.transforms(), feature_dim, identity


def _embed_frames(
    frame_paths: Sequence[Path],
    *,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
    decode_workers: int,
) -> np.ndarray:
    import torch
    from PIL import Image

    def load_frame(path: Path) -> Any:
        with Image.open(path) as image:
            return transform(image.convert("RGB"))

    batches: list[np.ndarray] = []
    use_amp = str(device).startswith("cuda")
    with ThreadPoolExecutor(max_workers=decode_workers or 1) as executor:
        for start in range(0, len(frame_paths), batch_size):
            paths = frame_paths[start : start + batch_size]
            tensors = list(executor.map(load_frame, paths))
            batch = torch.stack(tensors).to(device, non_blocking=True)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                features = model(batch)
            batches.append(features.detach().float().cpu().numpy())
    result = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[0] != len(frame_paths):
        raise RuntimeError("encoder returned an invalid feature matrix")
    if not np.isfinite(result).all():
        raise RuntimeError("encoder returned non-finite features")
    return result


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--official-splits",
        default=str(
            Path(__file__).resolve().parents[1] / "data/orbit_official_splits.json"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "external"),
        required=True,
    )
    parser.add_argument(
        "--external-cohort",
        default=None,
        help="frozen collector manifest required when --split=external",
    )
    parser.add_argument(
        "--source-archive",
        default=None,
        help="immutable source archive used to verify external provenance",
    )
    parser.add_argument(
        "--expected-archive-md5",
        default=None,
        help="publisher MD5 required with --source-archive",
    )
    parser.add_argument("--encoder", choices=ENCODERS, default="efficientnet_b0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--max-frames-per-video", type=int, default=1000)
    users = parser.add_mutually_exclusive_group()
    users.add_argument("--max-users", type=int, default=None)
    users.add_argument("--user-ids", default=None)
    parser.add_argument("--require-complete-split", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.max_frames_per_video < 1:
        raise ValueError("batch size and max frames must be positive")
    if args.decode_workers < 0:
        raise ValueError("decode-workers must be non-negative")
    if args.max_users is not None and args.max_users < 1:
        raise ValueError("max-users must be positive")
    requested_users = parse_user_ids(args.user_ids)
    archive_md5: str | None = None
    if (args.source_archive is None) != (args.expected_archive_md5 is None):
        raise ValueError(
            "--source-archive and --expected-archive-md5 must be provided together"
        )
    if args.split == "external" and args.source_archive is None:
        raise ValueError("external split requires immutable archive provenance")
    if args.source_archive is not None:
        expected_md5 = str(args.expected_archive_md5).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
            raise ValueError("expected archive MD5 must contain 32 hexadecimal digits")
        archive_md5 = file_md5(args.source_archive)
        if archive_md5 != expected_md5:
            raise ValueError(
                f"source archive MD5 mismatch: {archive_md5} != {expected_md5}"
            )
    if args.split == "external":
        if args.external_cohort is None:
            raise ValueError("external split requires --external-cohort")
        expected_users = load_orbit_external_cohort(args.external_cohort)
    else:
        if args.external_cohort is not None:
            raise ValueError("--external-cohort is only valid for external split")
        splits = load_official_orbit_splits(args.official_splits)
        expected_users = splits[args.split]
    videos = discover_orbit_videos(
        args.raw_root, split=args.split, allowed_users=expected_users
    )
    observed_users = sorted({item[0] for item in videos})
    if args.require_complete_split and set(observed_users) != set(expected_users):
        raise ValueError(
            f"raw split is incomplete; observed {len(observed_users)} of "
            f"{len(expected_users)} expected users"
        )
    if args.max_users is not None:
        selected = set(observed_users[: args.max_users])
        videos = [item for item in videos if item[0] in selected]
    elif requested_users is not None:
        missing = set(requested_users) - set(observed_users)
        if missing:
            raise ValueError(f"requested users are absent: {sorted(missing)}")
        selected = set(requested_users)
        videos = [item for item in videos if item[0] in selected]

    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, transform, feature_dim, encoder_identity = _build_encoder(
        args.encoder, args.device
    )
    output_root = Path(args.output_root).expanduser().resolve()
    annotation_root = Path(args.annotations_root).expanduser().resolve()
    manifest_path = output_root / "feature_manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        missing = set(FEATURE_MANIFEST_COLUMNS) - set(manifest.columns)
        if missing:
            raise ValueError(f"existing feature manifest misses {sorted(missing)}")
        rows = manifest.to_dict("records")
    else:
        rows = []
    row_keys = {
        (str(row["split"]), str(row["video_id"])): index
        for index, row in enumerate(rows)
    }
    failures: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)
    for video_number, (user_id, object_name, video_type, video_dir) in enumerate(
        videos, start=1
    ):
        video_id = video_dir.name
        try:
            frame_paths = _frame_paths(video_dir, max_frames=args.max_frames_per_video)
            annotation_path = None
            if video_type == "clutter":
                annotation_path = (
                    annotation_root / f"{video_id}.json"
                    if args.split == "external"
                    else annotation_root / args.split / f"{video_id}.json"
                )
            fingerprint = _source_fingerprint(
                frame_paths, annotation_path, encoder_identity
            )
            relative = (
                Path(args.split)
                / user_id
                / object_name
                / video_type
                / f"{video_id}.npz"
            )
            output_path = output_root / relative
            key = (args.split, video_id)
            previous = rows[row_keys[key]] if key in row_keys else None
            if previous is not None and output_path.is_file():
                if str(previous["source_fingerprint"]) != fingerprint:
                    raise RuntimeError(
                        "existing cache fingerprint changed; choose a new output root"
                    )
                print(f"[{video_number}/{len(videos)}] cached {video_id}", flush=True)
                continue
            if output_path.exists():
                raise RuntimeError(
                    "unmanifested feature file exists; choose a new output root"
                )
            mask = _protocol_object_present_mask(
                frame_paths,
                video_type=video_type,
                annotations_root=annotation_root,
                split=args.split,
                video_id=video_id,
            )
            embeddings = _embed_frames(
                frame_paths,
                model=model,
                transform=transform,
                device=args.device,
                batch_size=args.batch_size,
                decode_workers=args.decode_workers,
            )
            if embeddings.shape[1] != feature_dim:
                raise RuntimeError(
                    f"encoder feature dimension {embeddings.shape[1]} != {feature_dim}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(output_path.name + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    embeddings=embeddings,
                    frame_indices=np.asarray(
                        [parse_frame_index(path) for path in frame_paths],
                        dtype=np.int64,
                    ),
                    object_present=mask,
                )
            temporary.replace(output_path)
            row = {
                "split": args.split,
                "user_id": user_id,
                "object_name": object_name,
                "video_type": video_type,
                "video_id": video_id,
                "feature_path": relative.as_posix(),
                "n_frames": len(frame_paths),
                "feature_dim": feature_dim,
                "source_fingerprint": fingerprint,
            }
            if key in row_keys:
                rows[row_keys[key]] = row
            else:
                row_keys[key] = len(rows)
                rows.append(row)
            _atomic_csv(
                pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS).sort_values(
                    ["split", "user_id", "object_name", "video_type", "video_id"]
                ),
                manifest_path,
            )
            print(f"[{video_number}/{len(videos)}] wrote {video_id}", flush=True)
        except Exception as error:  # preserve every failed video
            failures.append(
                {
                    "split": args.split,
                    "user_id": user_id,
                    "object_name": object_name,
                    "video_type": video_type,
                    "video_id": video_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"[{video_number}/{len(videos)}] FAILED {video_id}: {error}",
                file=sys.stderr,
                flush=True,
            )
    failure_path = output_root / f"failures_{args.split}.csv"
    _atomic_csv(pd.DataFrame(failures), failure_path)
    _atomic_json(
        {
            "schema_version": "orbit-feature-cache-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": started.isoformat(),
            "raw_root": str(Path(args.raw_root).expanduser().resolve()),
            "annotations_root": str(annotation_root),
            "official_splits": str(Path(args.official_splits).resolve()),
            "external_cohort": (
                str(Path(args.external_cohort).resolve())
                if args.external_cohort is not None
                else None
            ),
            "source_archive": (
                str(Path(args.source_archive).resolve())
                if args.source_archive is not None
                else None
            ),
            "source_archive_md5": archive_md5,
            "split": args.split,
            "encoder_identity": encoder_identity,
            "feature_dim": feature_dim,
            "decode_workers": args.decode_workers,
            "max_frames_per_video": args.max_frames_per_video,
            "n_planned_videos": len(videos),
            "requested_users": list(requested_users or ()),
            "n_failures": len(failures),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": args.device,
        },
        output_root / f"provenance_{args.split}.json",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} ORBIT videos failed; see {failure_path}")


if __name__ == "__main__":
    main()
