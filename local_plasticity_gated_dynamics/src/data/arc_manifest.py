"""Fail-closed validation for pinned ARC directory snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


class ARCManifestError(ValueError):
    """Raised when an ARC source tree differs from its reviewed manifest."""


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_LINE = re.compile(
    r"(?P<sha256>[0-9a-f]{64})  "
    r"(?P<path>(?:training|evaluation)/[A-Za-z0-9_.-]+\.json)"
)
_SPLITS = ("training", "evaluation")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ARCManifestReceipt:
    """Portable receipt for an exactly verified ARC source tree."""

    manifest_name: str
    manifest_sha256: str
    manifest_format: str
    license_name: str
    license_sha256: str
    n_files: int
    n_bytes: int
    split_counts: Mapping[str, int]
    source_tree_verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_counts",
            MappingProxyType(
                {str(key): int(value) for key, value in self.split_counts.items()}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_name": self.manifest_name,
            "manifest_sha256": self.manifest_sha256,
            "manifest_format": self.manifest_format,
            "license_name": self.license_name,
            "license_sha256": self.license_sha256,
            "n_files": self.n_files,
            "n_bytes": self.n_bytes,
            "split_counts": dict(self.split_counts),
            "source_tree_verified": self.source_tree_verified,
        }


def validate_arc_source_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_license_sha256: str,
    expected_split_counts: Mapping[str, int],
) -> ARCManifestReceipt:
    """Verify every ARC JSON byte against a reviewed ``sha256sum`` manifest.

    The manifest intentionally covers both official splits, including any task
    excluded later for cross-split leakage.  Exclusions therefore cannot hide a
    changed or missing source file.
    """

    root = Path(data_root)
    manifest = Path(manifest_path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if (
        _SHA256.fullmatch(str(expected_manifest_sha256)) is None
        or _SHA256.fullmatch(str(expected_license_sha256)) is None
    ):
        raise ARCManifestError(
            "ARC manifest and license digests must be lowercase SHA-256"
        )
    actual_manifest_sha256 = _file_sha256(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ARCManifestError("ARC source manifest SHA-256 mismatch")

    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ARCManifestError("ARC source manifest must be UTF-8 text") from error
    if not lines:
        raise ARCManifestError("ARC source manifest is empty")

    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ARCManifestError(
                f"invalid ARC source manifest line {line_number}: {line!r}"
            )
        relative = match.group("path")
        if relative in entries:
            raise ARCManifestError(f"duplicate ARC manifest path: {relative}")
        entries[relative] = match.group("sha256")
    if list(entries) != sorted(entries):
        raise ARCManifestError("ARC source manifest paths must be sorted")

    normalized_counts = {
        str(split): int(count) for split, count in expected_split_counts.items()
    }
    if set(normalized_counts) != set(_SPLITS) or any(
        count < 1 for count in normalized_counts.values()
    ):
        raise ARCManifestError(
            "expected ARC split counts must contain positive training/evaluation counts"
        )
    manifest_counts = {
        split: sum(relative.startswith(f"{split}/") for relative in entries)
        for split in _SPLITS
    }
    if manifest_counts != normalized_counts:
        raise ARCManifestError("ARC source manifest split counts differ from config")

    root_resolved = root.resolve()
    discovered: set[str] = set()
    for path in root.rglob("*.json"):
        if not path.is_file() or path.is_symlink():
            raise ARCManifestError("ARC JSON sources must be regular non-symlink files")
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ARCManifestError("ARC JSON source escapes the configured root")
        discovered.add(path.relative_to(root).as_posix())
    if discovered != set(entries):
        missing = sorted(set(entries) - discovered)
        extra = sorted(discovered - set(entries))
        raise ARCManifestError(
            f"ARC source tree coverage mismatch; missing={missing!r}, extra={extra!r}"
        )

    n_bytes = 0
    for relative, expected_digest in entries.items():
        path = root / Path(relative)
        if _file_sha256(path) != expected_digest:
            raise ARCManifestError(f"ARC source file SHA-256 mismatch: {relative}")
        n_bytes += path.stat().st_size

    license_path = root / "LICENSE"
    if (
        not license_path.is_file()
        or license_path.is_symlink()
        or _file_sha256(license_path) != expected_license_sha256
    ):
        raise ARCManifestError("ARC source LICENSE SHA-256 mismatch")

    return ARCManifestReceipt(
        manifest_name=manifest.name,
        manifest_sha256=actual_manifest_sha256,
        manifest_format="sha256sum_sorted_posix_v1",
        license_name=license_path.name,
        license_sha256=expected_license_sha256,
        n_files=len(entries),
        n_bytes=n_bytes,
        split_counts=manifest_counts,
    )


def validate_arc_acquisition_receipts(
    acquisition_manifest_path: str | Path,
    validation_path: str | Path,
    *,
    expected_acquisition_manifest_sha256: str,
    expected_validation_sha256: str,
    dataset_name: str,
    revision: str,
    source_url: str,
    license_name: str,
    source_manifest_sha256: str,
    expected_split_counts: Mapping[str, int],
) -> dict[str, object]:
    """Bind a byte manifest to the audited pinned-source acquisition receipt."""

    acquisition_path = Path(acquisition_manifest_path)
    validation_receipt_path = Path(validation_path)
    for label, path, expected_digest in (
        (
            "acquisition manifest",
            acquisition_path,
            expected_acquisition_manifest_sha256,
        ),
        ("ARC validation receipt", validation_receipt_path, expected_validation_sha256),
    ):
        if _SHA256.fullmatch(str(expected_digest)) is None:
            raise ARCManifestError(f"{label} digest must be lowercase SHA-256")
        if not path.is_file():
            raise FileNotFoundError(path)
        if _file_sha256(path) != expected_digest:
            raise ARCManifestError(f"{label} SHA-256 mismatch")
    try:
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ARCManifestError(
            "ARC acquisition receipts must be valid UTF-8 JSON"
        ) from error
    if not isinstance(acquisition, Mapping) or not isinstance(validation, Mapping):
        raise ARCManifestError("ARC acquisition receipts must be JSON objects")

    sources = acquisition.get("arc")
    validated_sources = validation.get("datasets")
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not isinstance(validated_sources, Sequence)
        or isinstance(validated_sources, (str, bytes))
    ):
        raise ARCManifestError("ARC acquisition receipts lack dataset records")
    matches = [
        item
        for item in sources
        if isinstance(item, Mapping) and item.get("name") == dataset_name
    ]
    validation_matches = [
        item
        for item in validated_sources
        if isinstance(item, Mapping) and item.get("name") == dataset_name
    ]
    if len(matches) != 1 or len(validation_matches) != 1:
        raise ARCManifestError(
            "ARC acquisition receipts do not uniquely identify dataset"
        )
    source = matches[0]
    validated = validation_matches[0]
    expected_identity = {
        "commit": revision,
        "license": license_name,
        "name": dataset_name,
        "url": source_url,
    }
    if any(source.get(key) != value for key, value in expected_identity.items()) or any(
        validated.get(key) != value for key, value in expected_identity.items()
    ):
        raise ARCManifestError("ARC acquisition identity differs from formal config")

    manifest_hashes = acquisition.get("arc_manifest_sha256")
    if (
        not isinstance(manifest_hashes, Mapping)
        or manifest_hashes.get(dataset_name) != source_manifest_sha256
        or manifest_hashes.get("validation") != expected_validation_sha256
    ):
        raise ARCManifestError(
            "ARC acquisition receipt does not bind reviewed manifests"
        )
    expected_counts = {
        str(split): int(count) for split, count in expected_split_counts.items()
    }
    for receipt in (source, validated):
        splits = receipt.get("splits")
        if not isinstance(splits, Mapping):
            raise ARCManifestError("ARC acquisition receipt lacks split statistics")
        observed_counts = {
            split: int(splits.get(split, {}).get("tasks", -1))
            if isinstance(splits.get(split), Mapping)
            else -1
            for split in _SPLITS
        }
        if observed_counts != expected_counts:
            raise ARCManifestError("ARC acquisition split counts differ from config")

    return {
        "acquisition_manifest_name": acquisition_path.name,
        "acquisition_manifest_sha256": expected_acquisition_manifest_sha256,
        "validation_name": validation_receipt_path.name,
        "validation_sha256": expected_validation_sha256,
        "dataset_name": dataset_name,
        "revision": revision,
        "source_url": source_url,
        "license": license_name,
        "source_manifest_sha256": source_manifest_sha256,
        "split_counts": expected_counts,
        "created_utc": acquisition.get("created_utc"),
        "source_acquisition_verified": True,
    }


__all__ = [
    "ARCManifestError",
    "ARCManifestReceipt",
    "validate_arc_acquisition_receipts",
    "validate_arc_source_manifest",
]
