#!/usr/bin/env python3
"""Create a checksum-gated transport attestation for the CORe50 archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_json_config
from scripts.prepare_core50_features import PROTOCOL_VERSION, validate_preregistration


PUBLISHED_MD5 = "745f3373fed08d69343f1058ee559e13"
TORRENT_URL = "https://orion.hyper.ai/tracker/download?torrent=6411"
TORRENT_SELECTED_FILE_INDEX = 5


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_attestation(
    *,
    config_path: Path,
    archive_path: Path,
    torrent_path: Path,
    official_failure_log: Path,
) -> dict[str, Any]:
    receipt = validate_preregistration(config_path)
    config = load_json_config(config_path)
    archive = archive_path.expanduser().resolve()
    torrent = torrent_path.expanduser().resolve()
    failure_log = official_failure_log.expanduser().resolve()
    for path in (archive, torrent, failure_log):
        if not path.is_file():
            raise FileNotFoundError(path)
    size = archive.stat().st_size
    if size != int(config["archive_content_length"]):
        raise RuntimeError("acquired CORe50 archive length does not match preregistration")
    md5 = _digest(archive, "md5")
    if md5 != PUBLISHED_MD5:
        raise RuntimeError("acquired CORe50 archive does not match published MD5")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "attested_at": datetime.now(timezone.utc).isoformat(),
        "canonical_source_url": str(config["archive_url"]),
        "canonical_http_metadata": {
            "content_length": int(receipt["archive_content_length"]),
            "etag": str(receipt["archive_etag"]),
            "last_modified": str(receipt["archive_last_modified"]),
        },
        "actual_transport": "BitTorrent mirror after canonical HTTP stall",
        "torrent_metadata_url": TORRENT_URL,
        "torrent_metadata_sha256": _digest(torrent, "sha256"),
        "torrent_selected_file_index": TORRENT_SELECTED_FILE_INDEX,
        "archive_path": str(archive),
        "archive_content_length": int(size),
        "archive_published_md5": PUBLISHED_MD5,
        "archive_observed_md5": md5,
        "archive_observed_sha256": _digest(archive, "sha256"),
        "official_http_failure_log": str(failure_log),
        "official_http_partial_preserved": True,
        "images_inspected": False,
        "model_outputs_inspected": False,
        "outcomes_inspected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prospective/exp37_core50_change_aware_prefix.json"),
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--torrent", type=Path, required=True)
    parser.add_argument("--official-failure-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_attestation(
        config_path=args.config.resolve(),
        archive_path=args.archive,
        torrent_path=args.torrent,
        official_failure_log=args.official_failure_log,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
