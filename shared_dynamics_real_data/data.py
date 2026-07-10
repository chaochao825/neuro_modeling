"""Readers for the four MATLAB datasets bundled outside version control.

The upstream files store ``X(unit, time)``.  Public functions return the more
conventional ``activity(time, unit)`` representation used by this package.
No data-dependent filtering or normalization is performed while loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.io import loadmat


BUNDLED_DATASETS: Mapping[str, str] = {
    "c_elegans": "data_C_elegans.mat",
    "hippocampus": "data_mouse_hippocampus.mat",
    "visual_responding": "data_mouse_visual_responding.mat",
    "visual_spontaneous": "data_mouse_visual_spontaneous.mat",
}


def load_activity_mat(path: str | Path, *, variable: str = "X") -> np.ndarray:
    """Load one MATLAB-v5 activity matrix as ``[time, unit]`` float32 data."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    variables = loadmat(source, variable_names=[variable])
    if variable not in variables:
        raise KeyError(f"variable {variable!r} is absent from {source}")
    raw = variables[variable]
    if sparse.issparse(raw):
        raw = raw.toarray()
    values = np.asarray(raw)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(f"{source} variable {variable!r} must be a 2-D matrix")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError(f"{source} variable {variable!r} must be real numeric")
    values = np.asarray(values.T, dtype=np.float32, order="C")
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite activity")
    values.setflags(write=False)
    return values


def load_bundled_datasets(data_root: str | Path) -> dict[str, np.ndarray]:
    """Load all four datasets from ``minimal_computation_original``-style root."""

    root = Path(data_root)
    missing = [name for name in BUNDLED_DATASETS.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing bundled data under {root}: {', '.join(sorted(missing))}"
        )
    return {
        label: load_activity_mat(root / filename)
        for label, filename in BUNDLED_DATASETS.items()
    }


def load_visual_context_pair(data_root: str | Path) -> dict[str, np.ndarray]:
    """Load responding/spontaneous recordings and verify neuron alignment size."""

    root = Path(data_root)
    pair = {
        label: load_activity_mat(root / BUNDLED_DATASETS[label])
        for label in ("visual_responding", "visual_spontaneous")
    }
    n_units = {values.shape[1] for values in pair.values()}
    if len(n_units) != 1:
        raise ValueError(
            "visual responding and spontaneous matrices do not have the same unit count"
        )
    return pair
