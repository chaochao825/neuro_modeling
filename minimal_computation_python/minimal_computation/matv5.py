"""Minimal MATLAB v5 reader for the bundled Minimal_computation datasets.

This parser intentionally supports the subset used here: numeric/sparse
`miMATRIX` variables, optionally compressed with zlib. It returns dense NumPy
arrays because the bundled datasets are small enough for the reproduction.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

MX_SPARSE_CLASS = 5
MX_DOUBLE_CLASS = 6
MX_SINGLE_CLASS = 7
MX_INT8_CLASS = 8
MX_UINT8_CLASS = 9
MX_INT16_CLASS = 10
MX_UINT16_CLASS = 11
MX_INT32_CLASS = 12
MX_UINT32_CLASS = 13
MX_INT64_CLASS = 14
MX_UINT64_CLASS = 15

DTYPES = {
    MI_INT8: np.int8,
    MI_UINT8: np.uint8,
    MI_INT16: np.int16,
    MI_UINT16: np.uint16,
    MI_INT32: np.int32,
    MI_UINT32: np.uint32,
    MI_SINGLE: np.float32,
    MI_DOUBLE: np.float64,
    MI_INT64: np.int64,
    MI_UINT64: np.uint64,
}


@dataclass
class Tag:
    dtype: int
    nbytes: int
    data_offset: int
    next_offset: int
    small: bool = False


def _align8(n: int) -> int:
    return n + ((8 - n % 8) % 8)


def _read_tag(buf: bytes, offset: int) -> Tag:
    raw0, raw1 = struct.unpack_from("<II", buf, offset)
    dtype_small = raw0 & 0xFFFF
    nbytes_small = raw0 >> 16
    if nbytes_small and dtype_small:
        return Tag(dtype_small, nbytes_small, offset + 4, offset + 8, True)
    dtype = raw0
    nbytes = raw1
    data_offset = offset + 8
    return Tag(dtype, nbytes, data_offset, data_offset + _align8(nbytes), False)


def _read_numeric(buf: bytes, offset: int) -> Tuple[np.ndarray, int]:
    tag = _read_tag(buf, offset)
    raw = buf[tag.data_offset : tag.data_offset + tag.nbytes]
    dtype = DTYPES.get(tag.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported MATLAB numeric dtype {tag.dtype}")
    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).copy()
    return arr, tag.next_offset


def _read_name(buf: bytes, offset: int) -> Tuple[str, int]:
    tag = _read_tag(buf, offset)
    raw = buf[tag.data_offset : tag.data_offset + tag.nbytes]
    return raw.decode("utf-8"), tag.next_offset


def _parse_matrix(payload: bytes) -> Tuple[str, np.ndarray]:
    offset = 0
    flags, offset = _read_numeric(payload, offset)
    class_id = int(flags[0] & 0xFF)
    nzmax = int(flags[1]) if flags.size > 1 else 0
    dims, offset = _read_numeric(payload, offset)
    dims = tuple(int(x) for x in dims)
    name, offset = _read_name(payload, offset)

    if class_id == MX_SPARSE_CLASS:
        ir, offset = _read_numeric(payload, offset)
        jc, offset = _read_numeric(payload, offset)
        data, offset = _read_numeric(payload, offset)
        if data.size == 0 and nzmax:
            data = np.ones(nzmax, dtype=float)
        is_binary = data.size == 0 or np.all((data == 0) | (data == 1))
        out = np.zeros(dims, dtype=np.uint8 if is_binary else float)
        for col in range(dims[1]):
            start = int(jc[col])
            end = int(jc[col + 1])
            rows = ir[start:end].astype(int)
            vals = data[start:end]
            out[rows, col] = vals.astype(out.dtype, copy=False)
        return name, out

    data, _ = _read_numeric(payload, offset)
    if not dims:
        arr = data
    else:
        arr = data.reshape(dims, order="F")
    return name, arr


def loadmat_vars(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    buf = path.read_bytes()
    if not buf.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"{path} is not a MATLAB v5 MAT-file")
    offset = 128
    out: Dict[str, np.ndarray] = {}
    while offset + 8 <= len(buf):
        tag = _read_tag(buf, offset)
        payload = buf[tag.data_offset : tag.data_offset + tag.nbytes]
        if tag.dtype == MI_COMPRESSED:
            payload = zlib.decompress(payload)
            inner = _read_tag(payload, 0)
            if inner.dtype != MI_MATRIX:
                raise ValueError("Compressed payload did not contain miMATRIX")
            name, arr = _parse_matrix(payload[inner.data_offset : inner.data_offset + inner.nbytes])
            out[name] = arr
        elif tag.dtype == MI_MATRIX:
            name, arr = _parse_matrix(payload)
            out[name] = arr
        elif tag.dtype == 0 and tag.nbytes == 0:
            break
        offset = tag.next_offset
    return out


def load_activity(path: str | Path) -> np.ndarray:
    vars_ = loadmat_vars(path)
    if "X" not in vars_:
        raise KeyError(f"No variable X found in {path}")
    x = np.asarray(vars_["X"])
    return (x > 0).astype(np.uint8, copy=False)
