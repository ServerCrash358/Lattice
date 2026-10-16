#!/usr/bin/env python3
"""Exports the ann-benchmarks SIFT1M hdf5 file to flat float32/int32 binaries
so the C++ benchmark (bench/cpp/bench_vamana_sift.cpp) doesn't need to link
libhdf5. One-time conversion, not part of the Lattice build itself.

Usage:
    bench/.venv/bin/python bench/harness/export_sift_binary.py \\
        --input bench/data/ann-benchmarks/sift-128-euclidean.hdf5 \\
        --out-dir bench/data/sift1m_bin \\
        --train-limit 100000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("bench/data/ann-benchmarks/sift-128-euclidean.hdf5"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench/data/sift1m_bin"))
    ap.add_argument("--train-limit", type=int, default=0, help="0 = full 1M")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)

    if args.train_limit > 0:
        train = train[: args.train_limit]
        # Ground truth in the file is computed against the FULL 1M train set,
        # so at a smaller train_limit it's invalid as-is. We recompute exact
        # ground truth for the subset via brute force (small enough at
        # 100K x 10K x 128 to run in numpy in a few minutes).
        print(f"Recomputing exact ground truth for a {args.train_limit}-row train subset "
              f"(the file's neighbors[] assumes the full 1M set)...")
        k = neighbors.shape[1]
        new_neighbors = np.zeros((test.shape[0], k), dtype=np.int32)
        batch = 200
        for start in range(0, test.shape[0], batch):
            end = min(start + batch, test.shape[0])
            q = test[start:end]  # (b, 128)
            # squared L2 via ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
            d2 = (np.sum(q * q, axis=1, keepdims=True)
                  - 2 * q @ train.T
                  + np.sum(train * train, axis=1, keepdims=True).T)
            idx = np.argpartition(d2, k, axis=1)[:, :k]
            for i in range(idx.shape[0]):
                row = idx[i]
                new_neighbors[start + i] = row[np.argsort(d2[i, row])]
            if start % 2000 == 0:
                print(f"  ground truth: {start}/{test.shape[0]}")
        neighbors = new_neighbors

    train.tofile(args.out_dir / "train.f32")
    test.tofile(args.out_dir / "test.f32")
    neighbors.tofile(args.out_dir / "neighbors.i32")
    with open(args.out_dir / "meta.txt", "w") as f:
        f.write(f"{train.shape[0]} {train.shape[1]}\n")
        f.write(f"{test.shape[0]} {test.shape[1]}\n")
        f.write(f"{neighbors.shape[0]} {neighbors.shape[1]}\n")

    print(f"Wrote {train.shape[0]} train, {test.shape[0]} test, "
          f"{neighbors.shape[0]}x{neighbors.shape[1]} ground truth to {args.out_dir}")


if __name__ == "__main__":
    main()
