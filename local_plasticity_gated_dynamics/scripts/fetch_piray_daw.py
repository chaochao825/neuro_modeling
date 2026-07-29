"""Fetch and verify the Piray--Daw Zenodo v1.0 source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import uuid
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "provenance" / "piray_daw_zenodo_v1.json"
BASE_URL = "https://zenodo.org/api/records/13840905/files/{name}/content"


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download(name: str, destination: Path, *, md5: str, size: int) -> None:
    if destination.is_file():
        if destination.stat().st_size == size and _digest(destination, "md5") == md5:
            return
        raise ValueError(f"existing {destination} does not match the frozen manifest")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with urllib.request.urlopen(BASE_URL.format(name=name), timeout=60) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if temporary.stat().st_size != size or _digest(temporary, "md5") != md5:
            raise ValueError(f"downloaded {name} does not match the frozen manifest")
        temporary.replace(destination)
    except Exception:
        # Preserve a failed transfer as an auditable partial rather than deleting it.
        if temporary.exists():
            failed = destination.with_name(f"{destination.name}.failed-{uuid.uuid4().hex}")
            temporary.replace(failed)
        raise


def _extract_verified_data(archive: Path, output: Path, hashes: dict[str, str]) -> None:
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        for relative, expected_sha256 in hashes.items():
            if relative not in names:
                raise ValueError(f"data.zip is missing {relative}")
            target = (output / relative).resolve()
            if output.resolve() not in target.parents:
                raise ValueError(f"unsafe archive member {relative}")
            if target.is_file():
                if _digest(target, "sha256") != expected_sha256:
                    raise ValueError(f"existing extracted file does not match: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
            with handle.open(relative) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if _digest(temporary, "sha256") != expected_sha256:
                raise ValueError(f"extracted file does not match: {relative}")
            temporary.replace(target)


def fetch(output: Path, *, include_code: bool = False) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    names = ["README.md", "data.zip"]
    if include_code:
        names.append("code.zip")
    for name in names:
        entry = manifest["files"][name]
        destination_name = "README.upstream.md" if name == "README.md" else name
        _download(
            name,
            output / destination_name,
            md5=str(entry["md5"]),
            size=int(entry["size_bytes"]),
        )
    _extract_verified_data(
        output / "data.zip",
        output,
        dict(manifest["extracted_data_sha256"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "piray_daw_v1",
    )
    parser.add_argument("--include-code", action="store_true")
    args = parser.parse_args()
    fetch(args.output.resolve(), include_code=args.include_code)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
