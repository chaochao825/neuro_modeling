from __future__ import annotations

from pathlib import Path

from scripts.attest_core50_acquisition import _digest


def test_acquisition_digest_is_streamed(tmp_path: Path) -> None:
    path = tmp_path / "bytes"
    path.write_bytes(b"core50 transport")
    assert _digest(path, "md5") == "36872f7e463d227003213e8fd75a576c"
    assert _digest(path, "sha256") == (
        "ce699cdc96d13aa9b6f2b8393ff4aab5eb5ba413313751237257904a1955be59"
    )
