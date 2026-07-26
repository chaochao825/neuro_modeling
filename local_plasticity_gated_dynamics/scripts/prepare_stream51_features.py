#!/usr/bin/env python3
"""Validate and embed frozen Stream-51 video splits directly from the archive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config
from experiments.exp38_stream51_soft_memory import (
    PROTOCOL_VERSION,
    validate_config,
    validate_implementation_receipt,
    validate_preregistration,
    validate_qualification_receipt,
)
from src.data.stream51_streaming import parse_stream51_ordering_line


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = "Stream-51/"
MANIFEST_COLUMNS = (
    "video_key",
    "class_id",
    "split",
    "feature_path",
    "n_frames",
    "feature_dim",
    "source_fingerprint",
)
FAILURE_COLUMNS = (
    "attempt_id",
    "attempted_at",
    "video_key",
    "class_id",
    "split",
    "error_type",
    "error",
)


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(int(chunk_bytes)):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def merge_failure_history(
    prior: pd.DataFrame, current_rows: Iterable[Mapping[str, Any]]
) -> pd.DataFrame:
    """Append extraction failures without erasing failures from earlier attempts."""

    current = pd.DataFrame(list(current_rows), columns=FAILURE_COLUMNS)
    if prior.empty:
        historical = pd.DataFrame(columns=FAILURE_COLUMNS)
    else:
        missing = set(FAILURE_COLUMNS) - set(prior.columns)
        if missing:
            raise ValueError("prior Stream-51 failure history has an invalid schema")
        historical = prior.loc[:, FAILURE_COLUMNS].copy()
    return pd.concat([historical, current], ignore_index=True)


def load_frozen_cohort(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stream-51 cohort is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stream51_video_split_v1":
        raise ValueError("Stream-51 cohort schema mismatch")
    videos = pd.DataFrame(payload.get("videos", []))
    required = {
        "video_key",
        "class_id",
        "class_name",
        "clip_id",
        "video_id",
        "split",
        "n_available_frames",
    }
    if videos.empty or not required <= set(videos.columns):
        raise ValueError("Stream-51 cohort video table is incomplete")
    if videos["video_key"].duplicated().any():
        raise ValueError("Stream-51 cohort contains duplicate videos")
    if set(videos["split"]) != {"support", "development", "external"}:
        raise ValueError("Stream-51 cohort split set is incomplete")
    return videos


def validate_bbox_schema(
    image_shape_value: Any, bbox_value: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Validate official width/height and tracker bounds after safe clipping.

    A small number of official tracker boxes extend one to four pixels beyond
    the JPEG canvas.  The reference loader clips them, so raw in-canvas bounds
    are not a valid schema requirement; a non-empty clipped rectangle is.
    """

    shape = np.asarray(image_shape_value, dtype=np.int64)
    bounds = np.asarray(bbox_value, dtype=np.int64)
    if shape.shape != (2,) or bounds.shape != (4,) or np.any(shape <= 0):
        raise ValueError("Stream-51 image shape or bbox is invalid")
    xmax, xmin, ymax, ymin = map(int, bounds)
    width, height = map(int, shape)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Stream-51 bbox has non-positive extent")
    if min(xmax, width) <= max(xmin, 0) or min(ymax, height) <= max(ymin, 0):
        raise ValueError("Stream-51 bbox is empty after reference clipping")
    return shape, bounds


def load_archive_metadata(archive: zipfile.ZipFile) -> pd.DataFrame:
    member = f"{ARCHIVE_PREFIX}Stream-51_meta_train.json"
    try:
        rows = json.loads(archive.read(member))
    except KeyError as error:
        raise ValueError("Stream-51 training metadata is missing") from error
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError(f"Stream-51 metadata row {index} has invalid schema")
        class_id, clip_id, video_id, frame_id, image_shape, bbox, path = row
        record = parse_stream51_ordering_line(f"{path} {class_id}")
        if (record.clip_id, record.video_id, record.frame_id) != (
            int(clip_id),
            int(video_id),
            int(frame_id),
        ):
            raise ValueError("Stream-51 path and metadata identifiers disagree")
        shape, bounds = validate_bbox_schema(image_shape, bbox)
        parsed.append(
            {
                "video_key": record.video_key,
                "class_id": int(class_id),
                "clip_id": int(clip_id),
                "video_id": int(video_id),
                "frame_id": int(frame_id),
                "image_width": int(shape[0]),
                "image_height": int(shape[1]),
                "bbox_xmax": int(bounds[0]),
                "bbox_xmin": int(bounds[1]),
                "bbox_ymax": int(bounds[2]),
                "bbox_ymin": int(bounds[3]),
                "source_path": str(path),
                "archive_member": f"{ARCHIVE_PREFIX}{path}",
            }
        )
    metadata = pd.DataFrame(parsed)
    if len(metadata) != 150736 or metadata["source_path"].duplicated().any():
        raise ValueError("Stream-51 metadata coverage mismatch")
    return metadata


def select_registered_frames(
    metadata: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    splits: Iterable[str],
    frames_per_split: Mapping[str, int],
) -> dict[str, pd.DataFrame]:
    requested = tuple(map(str, splits))
    if not requested or not set(requested) <= {"support", "development", "external"}:
        raise ValueError("requested splits are invalid")
    metadata_groups = {key: group for key, group in metadata.groupby("video_key")}
    selected: dict[str, pd.DataFrame] = {}
    for row in cohort[cohort["split"].isin(requested)].itertuples(index=False):
        key = str(row.video_key)
        if key not in metadata_groups:
            raise ValueError(f"cohort video is absent from metadata: {key}")
        group = metadata_groups[key].sort_values("frame_id").reset_index(drop=True)
        if len(group) != int(row.n_available_frames):
            raise ValueError(f"cohort frame count disagrees for {key}")
        if int(group["class_id"].iloc[0]) != int(row.class_id):
            raise ValueError(f"cohort class disagrees for {key}")
        maximum = int(frames_per_split[str(row.split)])
        if maximum < 2:
            raise ValueError("registered frame cap must be at least two")
        if len(group) > maximum:
            indices = np.linspace(0, len(group) - 1, maximum).round().astype(int)
            group = group.iloc[indices].reset_index(drop=True)
        selected[key] = group
    expected = set(cohort.loc[cohort["split"].isin(requested), "video_key"].astype(str))
    if set(selected) != expected:
        raise RuntimeError("selected Stream-51 videos do not match cohort")
    return selected


def crop_registered_bbox(image: Any, row: Mapping[str, Any], *, ratio: float) -> Any:
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("bbox ratio must be finite and positive")
    xmax = int(row["bbox_xmax"])
    xmin = int(row["bbox_xmin"])
    ymax = int(row["bbox_ymax"])
    ymin = int(row["bbox_ymin"])
    center_x = xmin + (xmax - xmin) / 2.0
    center_y = ymin + (ymax - ymin) / 2.0
    half_width = (xmax - xmin) * ratio / 2.0
    half_height = (ymax - ymin) * ratio / 2.0
    left = max(int(center_x - half_width), 0)
    right = min(int(center_x + half_width), int(image.size[0]))
    top = max(int(center_y - half_height), 0)
    bottom = min(int(center_y + half_height), int(image.size[1]))
    if left >= right or top >= bottom:
        raise ValueError("registered bbox produces an empty crop")
    return image.crop((left, top, right, bottom))


def _source_fingerprint(
    rows: pd.DataFrame, archive: zipfile.ZipFile, *, encoder_identity: str
) -> str:
    digest = hashlib.sha256(encoder_identity.encode("utf-8"))
    for row in rows.itertuples(index=False):
        info = archive.getinfo(str(row.archive_member))
        digest.update(str(row.source_path).encode("utf-8"))
        digest.update(str(info.CRC).encode("ascii"))
        digest.update(str(info.file_size).encode("ascii"))
        digest.update(
            f"{row.bbox_xmax},{row.bbox_xmin},{row.bbox_ymax},{row.bbox_ymin}".encode(
                "ascii"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _build_encoder(device: str) -> tuple[Any, Any, int, str]:
    try:
        import torch
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as error:  # pragma: no cover - optional runtime gate
        raise RuntimeError("feature extraction requires torch torchvision Pillow") from error
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().requires_grad_(False).to(device)
    identity = f"torchvision::efficientnet_b0::{weights.__class__.__name__}.{weights.name}"
    return model, weights.transforms(), 1280, identity


def _embed_video(
    rows: pd.DataFrame,
    *,
    archive: zipfile.ZipFile,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
    decode_workers: int,
    bbox_ratio: float,
) -> np.ndarray:
    import torch
    from PIL import Image

    outputs: list[np.ndarray] = []
    row_records = rows.to_dict("records")

    def decode(row: Mapping[str, Any]) -> Any:
        with Image.open(BytesIO(archive.read(str(row["archive_member"])))) as image:
            rgb = image.convert("RGB")
            cropped = crop_registered_bbox(rgb, row, ratio=bbox_ratio)
            return transform(cropped)

    for start in range(0, len(row_records), batch_size):
        batch_rows = row_records[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=int(decode_workers)) as pool:
            tensors = list(pool.map(decode, batch_rows))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if str(device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            encoded = model(batch)
        outputs.append(encoded.detach().float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def validate_archive_schema(
    config: Mapping[str, Any], *, stage: str
) -> tuple[Path, str, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    archive_path = Path(str(config["archive_path"])).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Stream-51 archive is missing: {archive_path}")
    if archive_path.stat().st_size != int(config["archive_content_length"]):
        raise ValueError("Stream-51 archive content length mismatch")
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != str(config["archive_sha256"]):
        raise ValueError("Stream-51 archive SHA-256 mismatch")
    cohort = load_frozen_cohort(PROJECT_ROOT / str(config["cohort_path"]))
    splits = ("support", "development") if stage == "support_development" else ("external",)
    caps = {
        "support": int(config["encoder"]["support_frames_per_video"]),
        "development": int(config["encoder"]["development_frames_per_video"]),
        "external": int(config["encoder"]["external_frames_per_video"]),
    }
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Stream-51 archive CRC failure: {bad_member}")
        metadata = load_archive_metadata(archive)
        selected = select_registered_frames(
            metadata, cohort, splits=splits, frames_per_split=caps
        )
    return archive_path, archive_sha256, cohort, metadata, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("support_development", "external"),
        default="support_development",
    )
    parser.add_argument("--qualification-receipt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_json_config(config_path)
    validate_config(config)
    preregistration = validate_preregistration(config_path)
    validate_implementation_receipt()
    if args.stage == "external":
        if args.qualification_receipt is None:
            raise ValueError("external feature generation requires qualification receipt")
        validate_qualification_receipt(args.qualification_receipt, config=config)
    archive_path, archive_sha256, cohort, metadata, selected = validate_archive_schema(
        config, stage=args.stage
    )
    output = Path(str(config["feature_root"])).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    attestation = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": args.stage,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "archive_url": config["archive_url"],
        "archive_path": str(archive_path),
        "archive_content_length": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "preregistered_http_metadata": {
            "etag": preregistration["archive_etag"],
            "last_modified": preregistration["archive_last_modified"],
        },
        "n_cohort_videos": int(len(cohort)),
        "n_selected_videos": int(len(selected)),
        "n_selected_frames": int(sum(len(rows) for rows in selected.values())),
        "external_features_accessed": args.stage == "external",
        "outcomes_inspected": False,
        "schema_complete": True,
        "bbox_schema": {
            "xmax_over_width": int(
                np.sum(metadata["bbox_xmax"] > metadata["image_width"])
            ),
            "ymax_over_height": int(
                np.sum(metadata["bbox_ymax"] > metadata["image_height"])
            ),
            "xmin_below_zero": int(np.sum(metadata["bbox_xmin"] < 0)),
            "ymin_below_zero": int(np.sum(metadata["bbox_ymin"] < 0)),
            "reference_clipping_applied": True,
        },
    }
    (output / f"acquisition_attestation_{args.stage}.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.validate_only:
        print(json.dumps(attestation, indent=2, sort_keys=True))
        return
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["encoder"]["batch_size"])
    )
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    decode_workers = int(config["encoder"]["decode_workers"])
    if decode_workers < 1:
        raise ValueError("decode_workers must be positive")
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempted_at = datetime.now(timezone.utc).isoformat()
    model, transform, feature_dim, encoder_identity = _build_encoder(args.device)
    manifest_path = output / "feature_manifest.csv"
    completed: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        prior = pd.read_csv(manifest_path, keep_default_na=False)
        completed = {
            str(row.video_key): row._asdict()
            for row in prior.itertuples(index=False)
        }
    failure_path = output / f"failures_{args.stage}.csv"
    prior_failures = (
        pd.read_csv(failure_path, keep_default_na=False)
        if failure_path.is_file()
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    failures: list[dict[str, Any]] = []
    cohort_index = cohort.set_index("video_key")
    with zipfile.ZipFile(archive_path) as archive:
        for index, (video_key, rows) in enumerate(sorted(selected.items()), start=1):
            split = str(cohort_index.loc[video_key, "split"])
            class_id = int(cohort_index.loc[video_key, "class_id"])
            relative = Path(split) / f"{video_key}.npz"
            destination = output / relative
            fingerprint = _source_fingerprint(
                rows, archive, encoder_identity=encoder_identity
            )
            cached = completed.get(video_key)
            if (
                cached is not None
                and str(cached.get("source_fingerprint")) == fingerprint
                and destination.is_file()
            ):
                try:
                    with np.load(destination, allow_pickle=False) as payload:
                        shape = np.asarray(payload["features"]).shape
                    if shape == (len(rows), feature_dim):
                        print(f"[{index}/{len(selected)}] cached {video_key}", flush=True)
                        continue
                except (OSError, KeyError, ValueError):
                    pass
            try:
                features = _embed_video(
                    rows,
                    archive=archive,
                    model=model,
                    transform=transform,
                    device=args.device,
                    batch_size=batch_size,
                    decode_workers=decode_workers,
                    bbox_ratio=float(config["encoder"]["bbox_padding_ratio"]),
                )
                if features.shape != (len(rows), feature_dim) or not np.all(
                    np.isfinite(features)
                ):
                    raise ValueError("encoder returned invalid Stream-51 features")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    features=features,
                    frame_ids=rows["frame_id"].to_numpy(dtype=np.int64),
                    source_paths=rows["source_path"].astype(str).to_numpy(),
                )
                temporary.replace(destination)
                completed[video_key] = {
                    "video_key": video_key,
                    "class_id": class_id,
                    "split": split,
                    "feature_path": relative.as_posix(),
                    "n_frames": len(rows),
                    "feature_dim": feature_dim,
                    "source_fingerprint": fingerprint,
                }
                print(f"[{index}/{len(selected)}] embedded {video_key}", flush=True)
            except Exception as error:  # noqa: BLE001 - retain every video failure
                failures.append(
                    {
                        "attempt_id": attempt_id,
                        "attempted_at": attempted_at,
                        "video_key": video_key,
                        "class_id": class_id,
                        "split": split,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(
                    f"[{index}/{len(selected)}] FAILED {video_key}: {error}",
                    flush=True,
                )
            complete_frame = pd.DataFrame(
                sorted(completed.values(), key=lambda row: str(row["video_key"])),
                columns=MANIFEST_COLUMNS,
            )
            _atomic_csv(complete_frame, manifest_path)
            _atomic_csv(
                merge_failure_history(prior_failures, failures), failure_path
            )
    expected = set(selected)
    observed = set(completed) & expected
    provenance = {
        **attestation,
        "encoder_identity": encoder_identity,
        "feature_dim": feature_dim,
        "n_completed_stage_videos": len(observed),
        "n_failed_stage_videos": len(failures),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": args.device,
        "decode_workers": decode_workers,
    }
    (output / f"provenance_{args.stage}.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures or observed != expected:
        raise RuntimeError(
            f"Stream-51 feature stage incomplete: {len(observed)}/{len(expected)}"
        )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
