#!/usr/bin/env python3
"""Phase 0 exit gate (spec/phase-0.md, test 1): harness self-consistency.

Per the consistency review (finding 34), this is deliberately NOT "match
hnswlib's published SIFT1M numbers within 10%" -- published QPS is
hardware-bound and not portable to this machine. Instead:

  (a) Recall check: sweep hnswlib's ef_search over a range on SIFT1M and
      compare our harness's *measured* recall@10 at each point against
      hnswlib's own behavior (recall is a property of the algorithm/data,
      not the hardware, so this is a fair comparison to published curves).
  (b) Self-consistency check: run hnswlib's own bundled benchmark
      methodology twice back-to-back on this machine and confirm our
      harness's QPS measurement is stable (within 10% of itself), which
      validates the harness's timing methodology rather than the hardware.

Usage:
    bench/.venv/bin/python bench/harness/reproduce_hnswlib.py \\
        --dataset bench/data/ann-benchmarks/sift-128-euclidean.hdf5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import hnswlib
import numpy as np


def load_sift(path: Path):
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int64)  # ground-truth top-k ids
    return train, test, neighbors


def build_index(train: np.ndarray, M: int = 16, ef_construction: int = 200) -> hnswlib.Index:
    dim = train.shape[1]
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=train.shape[0], ef_construction=ef_construction, M=M)
    index.add_items(train, np.arange(train.shape[0]))
    return index


def recall_at_k(index: hnswlib.Index, test: np.ndarray, neighbors: np.ndarray,
                 ef_search: int, k: int = 10) -> tuple[float, float]:
    index.set_ef(ef_search)
    start = time.perf_counter()
    labels, _ = index.knn_query(test, k=k)
    elapsed = time.perf_counter() - start
    qps = test.shape[0] / elapsed

    hits = 0
    total = 0
    for i in range(test.shape[0]):
        gt = set(neighbors[i, :k].tolist())
        got = set(labels[i].tolist())
        hits += len(gt & got)
        total += k
    return hits / total, qps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                     default=Path("bench/data/ann-benchmarks/sift-128-euclidean.hdf5"))
    ap.add_argument("--ef-sweep", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    print(f"[load] {args.dataset}")
    train, test, neighbors = load_sift(args.dataset)
    print(f"[load] train={train.shape} test={test.shape} k_gt={neighbors.shape[1]}")

    print("[build] hnswlib index (M=16, ef_construction=200)")
    t0 = time.perf_counter()
    index = build_index(train)
    print(f"[build] done in {time.perf_counter() - t0:.1f}s")

    print(f"\n{'ef_search':>10} {'recall@'+str(args.k):>10} {'qps':>10}")
    results = []
    for ef in args.ef_sweep:
        recall, qps = recall_at_k(index, test, neighbors, ef, args.k)
        results.append((ef, recall, qps))
        print(f"{ef:>10} {recall:>10.4f} {qps:>10.1f}")

    # Self-consistency check (b): re-run the highest-QPS point twice, confirm
    # our own measurement is stable within 10% of itself.
    best_ef = results[0][0]
    _, qps_a = recall_at_k(index, test, neighbors, best_ef, args.k)
    _, qps_b = recall_at_k(index, test, neighbors, best_ef, args.k)
    drift = abs(qps_a - qps_b) / max(qps_a, qps_b)
    print(f"\n[self-consistency] ef={best_ef}: qps_run1={qps_a:.1f} qps_run2={qps_b:.1f} "
          f"drift={drift:.1%} ({'PASS' if drift < 0.10 else 'FAIL'} -- threshold 10%)")

    print("\nCompare the recall@ef_search curve above against hnswlib's published "
          "SIFT1M curve (e.g. ann-benchmarks.com results page) -- recall should "
          "match within +/-0.01 at each ef_search. This is the portable check; "
          "the QPS column is this-machine-only and not compared cross-hardware.")


if __name__ == "__main__":
    main()
