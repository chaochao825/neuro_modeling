#!/usr/bin/env python3
"""Build a resumable, fail-closed ORBIT per-video embedding cache.

The raw benchmark is never modified.  Each video is embedded independently
and written as a compressed NumPy file.  A manifest is atomically refreshed
after every successful video, while failures are retained in a separate CSV.
The extractor is frozen and preprocessing is fixed by its published weights.
"""

from __future__ import annotations

import argparse
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


def discover_orbit_videos(
    raw_root: str | Path,
    *,
    split: str,
    allowed_users: Iterable[str],
) -> list[tuple[str, str, str, Path]]:
    """Discover canonical ``user/object/type/video`` directories."""

    root = Path(raw_root).expanduser().resolve() / split
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
    path = annotations_root / split / f"{video_id}.json"
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
) -> np.ndarray:
    import torch
    from PIL import Image

    batches: list[np.ndarray] = []
    use_amp = str(device).startswith("cuda")
    for start in range(0, len(frame_paths), batch_size):
        tensors = []
        for path in frame_paths[start : start + batch_size]:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
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
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--encoder", choices=ENCODERS, default="efficientnet_b0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-frames-per-video", type=int, default=1000)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--require-complete-split", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.max_frames_per_video < 1:
        raise ValueError("batch size and max frames must be positive")
    if args.max_users is not None and args.max_users < 1:
        raise ValueError("max-users must be positive")
    splits = load_official_orbit_splits(args.official_splits)
    videos = discover_orbit_videos(
        args.raw_root, split=args.split, allowed_users=splits[args.split]
    )
    observed_users = sorted({item[0] for item in videos})
    if args.require_complete_split and set(observed_users) != set(splits[args.split]):
        raise ValueError(
            f"raw split is incomplete; observed {len(observed_users)} of "
            f"{len(splits[args.split])} official users"
        )
    if args.max_users is not None:
        selected = set(observed_users[: args.max_users])
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
            annotation_path = (
                annotation_root / args.split / f"{video_id}.json"
                if video_type == "clutter"
                else None
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
            "split": args.split,
            "encoder_identity": encoder_identity,
            "feature_dim": feature_dim,
            "max_frames_per_video": args.max_frames_per_video,
            "n_planned_videos": len(videos),
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
