# Benchmarks

All numbers below were measured by the harnesses in `bench/` on **real SIFT1M** (1,000,000 x 128-d, the 10,000-query ANN-Benchmarks test set, exact ground truth from the dataset, raw L2, k=10). Hardware: AMD Ryzen 7 260 laptop (8 cores / 16 threads, 23 GB RAM) under WSL2 (16 threads, 18 GB visible), NVMe-backed virtual disk, GCC 15 `-O3`. Every system is run on this same machine with the same protocol, so the comparison is like-for-like. Treat absolute numbers as laptop numbers.

Protocol (the ANN-Benchmarks convention): queries issued **one at a time on a single thread**, QPS = queries / total time, recall@10 and recall@1 against the shipped ground truth, p50/p99 per-query latency. A separate multi-thread section reports aggregate QPS with every thread issuing queries concurrently.

## 1. Lattice results (production path)

`VamanaIndex::build` (parallel, SQ8 codes-gated, R=64, build_list=100) -> `serialize_to_disk` (compact 832 B records, 0.83 GB file) -> `open_disk_resident_vamana` (zero-copy mmap node store) -> search. Warm page cache.

| ef_search | recall@1 | recall@10 | p50 | p99 | QPS 1 thread | QPS 16 threads |
|---|---|---|---|---|---|---|
| 32 | 0.981 | 0.975 | 0.32 ms | 1.29 ms | 2,637 | -- |
| 64 | 0.989 | 0.993 | 0.52 ms | 1.95 ms | 1,618 | 20,615 |
| 128 | 0.992 | 0.998 | 0.91 ms | 3.02 ms | 961 | 12,461 |
| 192 | 0.992 | 0.9990 | 1.30 ms | 3.96 ms | 682 | -- |
| 256 | 0.993 | 0.9991 | 1.66 ms | 4.81 ms | 537 | 6,670 |
| 1024 | 0.992 | 0.9992 | 3.06 ms | 6.15 ms | 317 | -- |

Build: ~200 s on 16 threads (about 25-30 min single-threaded; `build_threads=1` is the deterministic sequential build).
Memory when warm: 136 MB pinned heap (SQ8 codes) + 804 MB file-backed index pages in the OS page cache. Only the 136 MB must stay resident; the rest is evictable.

## 2. Head-to-head on the same machine

Single thread, recall@10 vs QPS, 10,000 queries:

| ~recall@10 | Lattice | hnswlib M=16 efC=500 | FAISS HNSW-Flat M=32 | FAISS IVF-Flat nlist=4096 |
|---|---|---|---|---|
| ~0.974-0.981 | 2,637 (0.975) | 3,160 (0.977) | 2,092 (0.981) | 877 (0.969) |
| ~0.990-0.995 | 1,618 (0.993) | 2,216 (0.989) | 1,271 (0.995) | 480 (0.991) |
| ~0.997 | 1,212 (0.997) | 1,413 (0.996) | -- | -- |
| ~0.999 | 682 (0.9990) | 802 (0.9989) | 698 (0.9988) | -- |

16 threads, aggregate QPS at matched recall:

| ~recall@10 | Lattice | hnswlib M=16 efC=500 |
|---|---|---|
| ~0.99 | 20,615 (0.993) | 8,238 (0.989) |
| ~0.999 | 6,670 (0.9991) | 2,919 (0.9989) |

Reading it honestly:
- **Single thread:** in-RAM hnswlib is about 15-17% faster than Lattice; Lattice is on par with or ahead of FAISS HNSW-Flat at the balanced point and far ahead of IVF-Flat.
- **16 threads:** Lattice delivers about 2.3-2.5x the QPS of hnswlib at matched recall. hnswlib scales poorly on this machine (about 3.7x from 1 to 16 threads vs Lattice's about 7-10x); we did not investigate hnswlib's internals.
- **Memory:** hnswlib's level-0 data is 652 MB; Lattice is 136 MB pinned + 804 MB evictable page cache (about 0.94 GB when warm). Lattice is not smaller when everything is cached; its advantage is that the index is persistent, memory-mappable and degrades gracefully when RAM is short rather than requiring the whole structure resident.
- **Build:** hnswlib 184-326 s (M/efC dependent), FAISS HNSW 186 s, Lattice ~200 s, all on 16 threads.
- FAISS numbers include Python per-call overhead (tens of microseconds), which slightly favours the C++ harnesses (Lattice, hnswlib).

## 3. Published results (context, not like-for-like)

- DiskANN (Subramanya et al., NeurIPS 2019): on **SIFT1B** from SSD, over 5,000 QPS at under 3 ms mean latency with 95%+ 1-recall@1 on a 16-core machine. Different scale and hardware; shown only as the headline DiskANN claim.
- ANN-Benchmarks (ann-benchmarks.com) publishes SIFT1M recall-vs-QPS **plots** for hnswlib, FAISS, ScaNN and others on its own hardware. The plot data is not machine-readable from the page, so no per-point numbers are quoted here; the head-to-head above re-runs hnswlib and FAISS under that protocol on this machine instead.
- Informal summaries of DiskANN/Vamana on SIFT1M quote recall@10 ~95% at 1-3 ms (balanced) and ~99.85% at 4-6 ms (high accuracy). These come from secondary summaries, not a primary source, so treat them as indicative. Lattice's measured numbers are recall@10 0.975 at 1.29 ms p99 and 0.999 at 3.96 ms p99.

## 4. Cold start (index evicted from the page cache)

Single thread, `ef_search=64`, first 500 queries after eviction (measured before the beam-dedup fix, which shifted what each ef_search value buys; the cold-vs-warm gap is the point):

| | p50 | p99 |
|---|---|---|
| warm | 0.34 ms | 0.81 ms |
| cold, no preload | 11.5 ms | 19 ms (settles to ~4.7 ms p50 after 4,000 queries) |
| cold, `preload=true` (open takes 1.1 s) | 0.41 ms | 0.99 ms |

Caveat: WSL's virtual disk and the Windows host add their own caching, so bare-metal NVMe cold numbers will differ.

## 5. Filtered search (1M vectors, 300 queries, exact filtered ground truth)

`VamanaIndex::search_filtered` scans only the matching vectors from the RAM-resident SQ8 codes (then reranks exactly) when the match set is small, otherwise widens the graph beam by 1/selectivity. "Baseline" is the earlier behaviour: unfiltered traversal with the filter applied to the result list.

| selectivity | filter | baseline recall@10 (ef=512) | planner recall@10 | planner p99 |
|---|---|---|---|---|
| 0.1% | uniform | 0.019 | 1.000 | 0.3 ms |
| 1% | uniform | 0.173 | 1.000 | 2.2 ms |
| 1% | clustered | 0.064 | 0.9997 | 2.6 ms |
| 10% | uniform | 0.988 | 1.000 (scan path) | 16 ms |
| 11.5% | clustered | 0.584 | 0.9993 (scan path) | 19.5 ms |
| 50% | uniform | 0.997 | 0.985-0.994 (graph path) | 1.7 ms |
| 50% | clustered | 0.934 | 0.88-0.92 (graph path) | 1.7 ms |

"Clustered" means the matching vectors are spatially localized (the hard case). The scan path costs about 95 ns per matching vector; the default `scan_max` is 150,000 matches. The 10% and 11.5% rows were measured with the scan forced on (the same code path the default now selects below 150K). **Known limitation:** above ~15% selectivity the planner uses graph traversal, which is weak for strongly clustered filters (recall 0.88-0.92 at 50%); scanning would reach 0.999 but costs about 45 ms p99.

## 6. What changed along the way (why the numbers moved)

Real-data testing found problems that synthetic data had hidden; in short:
- `robust_prune` used `-dot` as an inner-product distance, silently over-pruning the graph (recall 0.009 -> 0.74 once fixed).
- RaBitQ (1-bit) codes were far less discriminative than SQ8 (8-bit) at 1M real vectors; SQ8 is the default.
- Unit-normalizing SIFT for inner product caps recall@10 near 0.992 against SIFT's L2 ground truth; raw L2 removes the cap.
- SQ8 scoring went from 105 ns to 17 ns per candidate (AVX2), a per-query 4 MB allocation was removed, and the shared `io_uring` ring (which hung under concurrency) became per-thread.
- The build was parallelized (about 5-6x faster) and records were made compact (832 B vs 4096 B), with a zero-copy mmap node store.

## Reproducing

```bash
cmake --preset release && cmake --build build/release -j$(nproc)
# Lattice (builds and caches the index on first run):
./build/release/bench/cpp/bench_production_sift1m <sift1m_bin_dir> <index_path> 10000 64 100
#   modes: ram | iouring | cold | coldpre (as the 6th argument)
./build/release/bench/cpp/bench_filtered_sift1m <sift1m_bin_dir> <index_path> 300
# Baselines (see bench/baselines/):
g++ -O3 -march=native -std=c++20 -pthread -I<hnswlib> bench/baselines/bench_hnswlib_sift1m.cpp -o bench_hnswlib
./bench_hnswlib <sift1m_bin_dir> 10000 16 500
python bench/baselines/bench_faiss_sift1m.py <sift1m_bin_dir> 10000
```
