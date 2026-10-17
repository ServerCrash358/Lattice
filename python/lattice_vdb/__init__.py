"""Lattice: a disk-resident vector index (Vamana graph, SQ8 codes, mmap node store).

    import numpy as np, lattice_vdb
    lattice_vdb.build_index(vectors, "my.lattice", metric="cosine")
    index = lattice_vdb.Index("my.lattice", preload=True)
    ids = index.search(query, k=10, ef_search=64)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from . import _core

try:
    __version__ = version("lattice-vdb")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"

__all__ = ["Index", "build_index", "__version__"]

_METRICS = ("l2", "ip", "cosine")


def _meta_path(path: str) -> str:
    return path + ".json"


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32, copy=False)


def build_index(
    vectors,
    path: str,
    metric: str = "l2",
    r: int = 64,
    build_list: int = 100,
    threads: int = 0,
) -> None:
    """Build an index from an (n, dim) array and save it to `path`. Row i gets id i.

    metric: "l2", "ip" (inner product) or "cosine" (vectors are normalized, then inner product).
    r: max neighbours per node (higher: better recall, bigger index, slower build).
    build_list: build beam width (higher: better graph, slower build).
    threads: build threads, 0 = all cores.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    data = np.ascontiguousarray(vectors, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("vectors must be a non-empty (n, dim) array")
    if metric == "cosine":
        data = _normalize(data)
    core_metric = "l2" if metric == "l2" else "ip"
    _core.build_index(data, path, metric=core_metric, r=r, build_list=build_list, build_threads=threads)
    with open(_meta_path(path), "w") as f:
        json.dump({"metric": metric, "dim": int(data.shape[1]), "count": int(data.shape[0]),
                   "r": r, "build_list": build_list, "format": "lattice-vdb/1"}, f)


class Index:
    """A built index, opened disk-resident (zero-copy mmap). Safe to search from many threads."""

    def __init__(self, path: str, preload: bool = False):
        self.path = path
        self._index = _core.Index(path, preload)
        meta = {}
        if os.path.exists(_meta_path(path)):
            with open(_meta_path(path)) as f:
                meta = json.load(f)
        self.metric = meta.get("metric", "unknown")

    def __len__(self) -> int:
        return len(self._index)

    @property
    def dim(self) -> int:
        return self._index.dim

    def _prep(self, q) -> np.ndarray:
        q = np.ascontiguousarray(q, dtype=np.float32)
        return _normalize(q) if self.metric == "cosine" else q

    def search(self, query, k: int = 10, ef_search: int = 64) -> np.ndarray:
        """Ids of the k nearest neighbours of one query (best first)."""
        return self._index.search(self._prep(query), k, ef_search)

    def search_batch(self, queries, k: int = 10, ef_search: int = 64, threads: int = 0) -> np.ndarray:
        """Search many queries in parallel; returns an (n, k) int64 array (-1 pads short results)."""
        qs = self._prep(queries)
        if qs.ndim != 2:
            raise ValueError("queries must be an (n, dim) array")
        out = np.full((len(qs), k), -1, dtype=np.int64)
        workers = threads or os.cpu_count() or 1

        def one(i: int) -> None:
            res = self._index.search(qs[i], k, ef_search)
            out[i, : len(res)] = res

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, range(len(qs))))
        return out
