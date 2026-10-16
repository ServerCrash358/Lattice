"""FAISS baselines (HNSW-Flat, IVF-Flat) on the same SIFT1M files and metrics as the
Lattice benchmark: raw L2, k=10, queries issued ONE AT A TIME on a single thread
(ann-benchmarks protocol). Python call overhead (~tens of microseconds per query) is
included in the reported latency, which slightly favours the C++ harnesses.

Usage: python bench_faiss_sift1m.py <bin_dir> [num_queries]
"""
import sys
import time

import faiss
import numpy as np

d = sys.argv[1]
nq_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
n, dim = map(int, open(f"{d}/meta.txt").read().split()[:2])
nt = int(open(f"{d}/meta.txt").read().split()[2])
gtk = int(open(f"{d}/meta.txt").read().split()[5])
nq = min(nq_arg, nt)
xb = np.fromfile(f"{d}/train.f32", dtype=np.float32).reshape(n, dim)
xq = np.fromfile(f"{d}/test.f32", dtype=np.float32).reshape(nt, dim)[:nq]
gt = np.fromfile(f"{d}/neighbors.i32", dtype=np.int32).reshape(nt, gtk)[:nq]
print(f"FAISS {faiss.__version__}, real SIFT1M: n={n} dim={dim} queries={nq}")


def sweep(index, setter, values, label):
    faiss.omp_set_num_threads(1)
    print(f"\n{label}\n{'param':>10} {'recall@1':>10} {'recall@10':>10} {'p50_ms':>10} {'p99_ms':>10} {'QPS_1thread':>12}")
    for v in values:
        setter(v)
        lat, r1, r10 = [], 0.0, 0.0
        t_all = time.perf_counter()
        for q in range(nq):
            t = time.perf_counter()
            _, ids = index.search(xq[q : q + 1], 10)
            lat.append((time.perf_counter() - t) * 1000)
            row = ids[0]
            r10 += len(set(row.tolist()) & set(gt[q, :10].tolist())) / 10.0
            r1 += float(row[0] == gt[q, 0])
        total = time.perf_counter() - t_all
        lat.sort()
        print(f"{v:>10} {r1/nq:>10.4f} {r10/nq:>10.4f} {lat[len(lat)//2]:>10.2f} {lat[int(0.99*(len(lat)-1))]:>10.2f} {nq/total:>12.0f}", flush=True)


faiss.omp_set_num_threads(16)
t0 = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(dim, 32)
hnsw.hnsw.efConstruction = 200
hnsw.add(xb)
print(f"\nHNSW-Flat M=32 efConstruction=200 built in {time.perf_counter()-t0:.1f}s")
sweep(hnsw, lambda v: setattr(hnsw.hnsw, "efSearch", v), [16, 32, 64, 128, 256, 512], "FAISS HNSW-Flat (M=32)")

faiss.omp_set_num_threads(16)
t0 = time.perf_counter()
quant = faiss.IndexFlatL2(dim)
ivf = faiss.IndexIVFFlat(quant, dim, 4096)
ivf.train(xb[:200000])
ivf.add(xb)
print(f"\nIVF-Flat nlist=4096 built in {time.perf_counter()-t0:.1f}s")
sweep(ivf, lambda v: setattr(ivf, "nprobe", v), [1, 2, 4, 8, 16, 32, 64, 128], "FAISS IVF-Flat (nlist=4096)")
