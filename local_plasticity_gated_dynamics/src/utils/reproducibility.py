"""Deterministic random-number helpers shared by every experiment."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeedState:
    """Recorded state returned after configuring deterministic execution."""

    seed: int
    torch_available: bool
    deterministic_algorithms: bool


def _validated_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be non-negative")
    return result


def _typed_bytes(value: object) -> bytes:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        payload = b"".join(
            len(item).to_bytes(8, "little") + item
            for item in (_typed_bytes(element) for element in value)
        )
    elif isinstance(value, Path):
        payload = str(value).encode("utf-8")
    elif isinstance(value, (str, bytes, int, bool, type(None))):
        payload = value if isinstance(value, bytes) else repr(value).encode("utf-8")
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("seed labels must be finite")
        payload = repr(value).encode("utf-8")
    else:
        raise TypeError(
            "seed labels must be scalar primitives, paths, or tuples of supported labels"
        )
    value_type = type(value)
    header = f"{value_type.__module__}.{value_type.__qualname__}".encode("utf-8")
    return len(header).to_bytes(4, "little") + header + payload


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch without hiding missing PyTorch.

    `PYTHONHASHSEED` only affects new Python processes, but recording it here
    makes child processes deterministic as well.
    """

    seed = _validated_seed(seed)
    if not isinstance(deterministic_torch, (bool, np.bool_)):
        raise TypeError("deterministic_torch must be boolean")
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Must be present before the first CUDA context is initialized for
    # deterministic cuBLAS execution in BPTT/GRU baseline processes.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    torch_available = False
    deterministic = False
    try:
        import torch

        torch_available = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic = True
    except ImportError:
        pass
    return SeedState(seed, torch_available, deterministic)


def derive_seed(seed: int, *labels: object) -> int:
    """Derive a stable independent 32-bit seed from a root seed and labels."""

    seed = _validated_seed(seed)
    hasher = hashlib.blake2s(digest_size=4)
    for value in (seed, *labels):
        payload = _typed_bytes(value)
        hasher.update(len(payload).to_bytes(8, "little"))
        hasher.update(payload)
    digest = hasher.digest()
    return int.from_bytes(digest, "little", signed=False)


def make_rng(seed: int, *labels: object) -> np.random.Generator:
    """Return an independent NumPy generator with stable label-based seeding."""

    return np.random.default_rng(derive_seed(seed, *labels))
