# bench/

Benchmarks and diagnostics. Datasets live in `bench/data/` (git-ignored); fetch/export scripts are in `bench/harness/`.

## Headline benchmarks (use these)
| File | What it measures |
|---|---|
| `cpp/bench_production_sift1m.cpp` | End to end on real SIFT1M through the production path: parallel build -> serialize -> disk-resident reopen -> recall@1/@10, p50/p99, 1-thread QPS and multi-thread QPS. Caches the built index. |
| `cpp/bench_filtered_sift1m.cpp` | Filtered-search recall/latency on real SIFT1M: baseline vs filter-aware planner, uniform and spatially clustered filters, selectivity 50% .. 0.1%. |
| `baselines/` | hnswlib (C++) and FAISS (Python) run on the same files with the same protocol, for direct comparison. See `docs/BENCHMARKS.md`. |
| `cpp/bench_sq8_kernel.cpp` | Per-candidate SQ8 scoring cost (scalar vs AVX2). |

## Historical diagnostics (kept as the evidence trail for the investigation summarized in `docs/BENCHMARKS.md` section 6)
`bench_rabitq_estimator_diagnostic.cpp` (brute-force estimator quality: SQ8 vs RaBitQ), `bench_sq8_sift1m_recall.cpp` (in-RAM port of the traversal used to isolate the `robust_prune` IP bug), `bench_rabitq_real_data.cpp` (RaBitQ error bound on real data), `bench_codes_microbench.cpp`, `bench_vamana_sift.cpp`, `bench_disk_parity_sift1m.cpp` (RAM vs disk parity), `bench_filtered_recall_10m.cpp` (synthetic multi-segment harness; the 10M target is out of scope).
