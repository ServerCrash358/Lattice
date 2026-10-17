"""The `lattice` command-line tool."""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.request

import click
import numpy as np

from . import Index, __version__, build_index
from ._io import load_vectors

DATASETS = {
    "sift1m": "sift-128-euclidean",
    "gist1m": "gist-960-euclidean",
    "glove100": "glove-100-angular",
}
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".lattice", "data")
_HEADER = struct.Struct("<7I2QI")  # magic, version, codes, dim, elem, r, sector, count, entry, metric


def _fail(msg: str) -> None:
    raise click.ClickException(msg)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="lattice")
def main() -> None:
    """Lattice: a disk-resident vector index (Vamana graph, SQ8 codes, mmap storage)."""


@main.command()
@click.argument("vectors", type=click.Path(exists=True, dir_okay=False))
@click.argument("index", type=click.Path(dir_okay=False))
@click.option("--metric", type=click.Choice(["l2", "ip", "cosine"]), default="l2", show_default=True)
@click.option("--r", "r", type=int, default=64, show_default=True, help="Max neighbours per node.")
@click.option("--build-list", type=int, default=100, show_default=True, help="Build beam width.")
@click.option("--threads", type=int, default=0, show_default=True, help="Build threads (0 = all cores).")
@click.option("--key", default=None, help="Dataset key inside an .hdf5 file (default: train).")
def build(vectors, index, metric, r, build_list, threads, key):
    """Build an index from a vector file (.npy, .fvecs, .bvecs, .hdf5)."""
    data = load_vectors(vectors, key)
    click.echo(f"building {data.shape[0]:,} x {data.shape[1]} vectors ({metric}, R={r}) ...")
    t = time.perf_counter()
    try:
        build_index(data, index, metric=metric, r=r, build_list=build_list, threads=threads)
    except Exception as e:  # noqa: BLE001
        _fail(str(e))
    click.echo(f"done in {time.perf_counter() - t:.1f}s -> {index} ({os.path.getsize(index) / 1e6:.1f} MB)")


@main.command()
@click.argument("index", type=click.Path(exists=True, dir_okay=False))
@click.argument("queries", type=click.Path(exists=True, dir_okay=False))
@click.option("--k", type=int, default=10, show_default=True)
@click.option("--ef", type=int, default=64, show_default=True, help="Search beam width.")
@click.option("--format", "fmt", type=click.Choice(["table", "json", "npy"]), default="table", show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="Output file (required for npy).")
@click.option("--key", default=None, help="Dataset key inside an .hdf5 file (default: test).")
def search(index, queries, k, ef, fmt, out, key):
    """Query an index with vectors from a file and print the top-k ids."""
    idx = Index(index)
    q = load_vectors(queries, key or "test")
    if q.ndim == 1:
        q = q[None, :]
    res = idx.search_batch(q, k=k, ef_search=ef)
    if fmt == "npy":
        if not out:
            _fail("--format npy needs --out FILE")
        np.save(out, res)
        click.echo(f"wrote {res.shape} ids to {out}")
        return
    text = (json.dumps(res.tolist()) if fmt == "json"
            else "\n".join(f"{i}\t" + " ".join(map(str, row)) for i, row in enumerate(res)))
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
    else:
        click.echo(text)


def _read_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(_HEADER.size)
    if len(raw) < _HEADER.size:
        _fail("file too short to be a Lattice index")
    magic, ver, codes, dim, elem, r, sector, count, entry, metric = _HEADER.unpack(raw)
    if magic != 0x56414D32:
        _fail("not a Lattice index (bad magic)")
    return {"format_version": ver, "codes": "sq8" if codes == 1 else codes, "dim": dim,
            "elem": "float32" if elem == 0 else elem, "r": r, "record_bytes": sector,
            "count": count, "entry_point": entry, "metric": {0: "ip", 1: "l2"}.get(metric, metric)}


@main.command()
@click.argument("index", type=click.Path(exists=True, dir_okay=False))
def info(index):
    """Show an index's dimension, size, metric, degree and file layout."""
    h = _read_header(index)
    meta_path = index + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            h["metric"] = json.load(f).get("metric", h["metric"])
    h["file_mb"] = round(os.path.getsize(index) / 1e6, 1)
    width = max(map(len, h))
    for name, value in h.items():
        click.echo(f"{name:<{width}}  {value}")


@main.command()
@click.argument("index", type=click.Path(exists=True, dir_okay=False))
def verify(index):
    """Check an index file's header, size and integrity."""
    _read_header(index)
    try:
        idx = Index(index)
        q = np.zeros(idx.dim, dtype=np.float32)
        q[0] = 1.0
        idx.search(q, k=1)
    except Exception as e:  # noqa: BLE001
        _fail(f"invalid: {e}")
    click.echo(f"OK: {len(idx):,} vectors, dim {idx.dim}")


@main.command()
@click.argument("index", type=click.Path(exists=True, dir_okay=False))
@click.argument("queries", type=click.Path(exists=True, dir_okay=False))
@click.argument("groundtruth", type=click.Path(exists=True, dir_okay=False))
@click.option("--k", type=int, default=10, show_default=True)
@click.option("--ef", default="32,64,128,256", show_default=True, help="Comma-separated beam widths.")
@click.option("--threads", default="1", show_default=True, help="Comma-separated thread counts for QPS.")
@click.option("--limit", type=int, default=None, help="Use only the first N queries.")
def bench(index, queries, groundtruth, k, ef, threads, limit):
    """Measure recall@k, latency (p50/p99) and QPS against ground truth."""
    idx = Index(index, preload=True)
    q = load_vectors(queries, "test")
    gt = load_vectors(groundtruth, "neighbors").astype(np.int64)
    if limit:
        q, gt = q[:limit], gt[:limit]
    if gt.shape[1] < k:
        _fail(f"ground truth has only {gt.shape[1]} neighbours per query, k={k}")
    efs = [int(x) for x in ef.split(",")]
    thread_counts = [int(x) for x in threads.split(",")]
    click.echo(f"{len(idx):,} vectors, {len(q):,} queries, k={k}")
    header = f"{'ef':>6} {'recall@k':>9} {'p50_ms':>8} {'p99_ms':>8}" + "".join(f" {'qps_' + str(t) + 't':>10}" for t in thread_counts)
    click.echo(header)
    for e in efs:
        lat, hits = [], 0
        for i in range(len(q)):
            t = time.perf_counter()
            res = idx.search(q[i], k=k, ef_search=e)
            lat.append(time.perf_counter() - t)
            hits += len(set(res.tolist()) & set(gt[i, :k].tolist()))
        lat_ms = np.sort(np.array(lat)) * 1000
        row = f"{e:>6} {hits / (len(q) * k):>9.4f} {lat_ms[len(lat_ms) // 2]:>8.3f} {lat_ms[int(0.99 * (len(lat_ms) - 1))]:>8.3f}"
        for t_count in thread_counts:
            t = time.perf_counter()
            idx.search_batch(q, k=k, ef_search=e, threads=t_count)
            row += f" {len(q) / (time.perf_counter() - t):>10,.0f}"
        click.echo(row)


@main.command()
@click.argument("dataset", type=click.Choice(sorted(DATASETS)))
@click.option("--dir", "data_dir", default=DEFAULT_DATA_DIR, show_default=True)
def fetch(dataset, data_dir):
    """Download a standard dataset (sift1m, gist1m, glove100) for testing."""
    os.makedirs(data_dir, exist_ok=True)
    name = DATASETS[dataset] + ".hdf5"
    dest = os.path.join(data_dir, name)
    if os.path.exists(dest):
        click.echo(f"already present: {dest}")
        return
    url = f"https://ann-benchmarks.com/{name}"
    click.echo(f"downloading {url}")
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    click.echo(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", nl=False)
        click.echo()
        os.replace(tmp, dest)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        _fail(f"download failed: {e}")
    click.echo(f"saved {dest}\nuse it with: lattice build {dest} index.lattice --metric "
               + ("cosine" if "angular" in name else "l2"))


if __name__ == "__main__":
    sys.exit(main())
