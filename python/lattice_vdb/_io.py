"""Readers for common vector file formats: .npy, .fvecs, .ivecs, .bvecs, .hdf5 (ann-benchmarks)."""

from __future__ import annotations

import os

import numpy as np


def _read_xvecs(path: str, dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    dim = int(raw[:4].view(np.int32)[0])
    item = np.dtype(dtype).itemsize
    row = 4 + dim * item
    if raw.size % row != 0:
        raise ValueError(f"{path}: size is not a multiple of the record size (dim={dim})")
    rows = raw.reshape(-1, row)[:, 4:]
    return np.ascontiguousarray(rows).view(dtype).reshape(-1, dim)


def load_vectors(path: str, key: str | None = None) -> np.ndarray:
    """Load an (n, dim) array. For .hdf5, `key` selects the dataset (default 'train')."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".fvecs":
        return _read_xvecs(path, np.float32)
    if ext == ".ivecs":
        return _read_xvecs(path, np.int32)
    if ext == ".bvecs":
        return _read_xvecs(path, np.uint8).astype(np.float32)
    if ext in (".hdf5", ".h5"):
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError("reading .hdf5 needs h5py: pip install 'lattice-vdb[hdf5]'") from e
        with h5py.File(path, "r") as f:
            return np.asarray(f[key or "train"])
    raise ValueError(f"unsupported file type {ext!r} (use .npy, .fvecs, .ivecs, .bvecs or .hdf5)")
