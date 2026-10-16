// Baseline: hnswlib (in-RAM HNSW) on the SAME SIFT1M files, query loop and metrics
// as bench/cpp/bench_production_sift1m.cpp, so the numbers are directly comparable
// (ann-benchmarks protocol: raw L2, k=10, queries issued one at a time, single thread).
//
// Build (hnswlib is header-only; fetch it with bench/baselines/README.md steps):
//   g++ -O3 -march=native -std=c++20 -pthread -I<hnswlib_dir> bench_hnswlib_sift1m.cpp -o bench_hnswlib_sift1m
// Usage: bench_hnswlib_sift1m <bin_dir> <num_queries> <M> <efConstruction> [build_threads]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "hnswlib/hnswlib.h"

static std::vector<float> read_f32(const std::string& p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}
static std::vector<int32_t> read_i32(const std::string& p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(int32_t));
    return v;
}

static double rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
    return 0;
}
int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <bin_dir> <num_queries> <M> <efConstruction> [build_threads]\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    size_t nq_arg = std::stoul(argv[2]);
    size_t M = std::stoul(argv[3]);
    size_t efc = std::stoul(argv[4]);
    unsigned bt = argc > 5 ? std::stoul(argv[5]) : std::thread::hardware_concurrency();

    std::ifstream meta(dir + "/meta.txt");
    size_t n, dim, nt, dim2, ngt, gtk;
    meta >> n >> dim >> nt >> dim2 >> ngt >> gtk;
    size_t nq = std::min(nq_arg, nt);
    auto train = read_f32(dir + "/train.f32", n * dim);
    auto test = read_f32(dir + "/test.f32", nq * dim);
    auto gt = read_i32(dir + "/neighbors.i32", ngt * gtk);
    printf("hnswlib baseline, real SIFT1M: n=%zu dim=%zu queries=%zu M=%zu efConstruction=%zu build_threads=%u\n",
           n, dim, nq, M, efc, bt);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n, M, efc);
    auto t0 = std::chrono::steady_clock::now();
    {
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        for (unsigned t = 0; t < bt; ++t)
            ts.emplace_back([&] {
                for (;;) {
                    size_t i = next.fetch_add(1);
                    if (i >= n) break;
                    index.addPoint(train.data() + i * dim, i);
                }
            });
        for (auto& th : ts) th.join();
    }
    printf("Build done in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    printf("Index memory: level-0 data (vectors+links+labels) = %.0f MB; process RSS (also holds the %.0f MB train array used by the harness) = %.0f MB\n",
           static_cast<double>(index.size_data_per_element_) * n / 1e6, static_cast<double>(n * dim * 4) / 1e6, rss_mb());

    printf("\n%10s %10s %10s %10s %10s %12s\n", "ef_search", "recall@1", "recall@10", "p50_ms", "p99_ms", "QPS_1thread");
    std::vector<size_t> efs = {10, 20, 40, 80, 120, 200, 400, 800};
    for (size_t ef : efs) {
        index.setEf(ef);
        std::vector<double> lat;
        double r1 = 0, r10 = 0;
        auto ts0 = std::chrono::steady_clock::now();
        for (size_t q = 0; q < nq; ++q) {
            auto t = std::chrono::steady_clock::now();
            auto res = index.searchKnn(test.data() + q * dim, 10);
            lat.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
            std::vector<size_t> got;
            while (!res.empty()) { got.push_back(res.top().second); res.pop(); }
            std::reverse(got.begin(), got.end());  // best first
            std::unordered_set<int32_t> exp(gt.data() + q * gtk, gt.data() + q * gtk + 10);
            size_t hits = 0;
            for (size_t id : got) hits += exp.count(static_cast<int32_t>(id));
            r10 += hits / 10.0;
            if (!got.empty() && static_cast<int32_t>(got[0]) == gt[q * gtk]) r1 += 1.0;
        }
        double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts0).count();
        std::sort(lat.begin(), lat.end());
        printf("%10zu %10.4f %10.4f %10.2f %10.2f %12.0f\n", ef, r1 / nq, r10 / nq, lat[lat.size() / 2],
               lat[static_cast<size_t>(0.99 * (lat.size() - 1))], nq / total);
        fflush(stdout);
    }

    unsigned hw = std::thread::hardware_concurrency();
    printf("\nMulti-thread throughput (hardware threads=%u, ~5s each):\n%10s %8s %12s\n", hw, "ef_search", "threads", "QPS");
    for (size_t ef : {40ul, 120ul, 400ul}) {
        index.setEf(ef);
        for (unsigned nthr : {1u, hw}) {
            std::atomic<uint64_t> done{0};
            std::atomic<bool> stop{false};
            std::vector<std::thread> ts;
            auto s = std::chrono::steady_clock::now();
            for (unsigned t = 0; t < nthr; ++t)
                ts.emplace_back([&, t] {
                    size_t q = t * 7;
                    while (!stop.load(std::memory_order_relaxed)) {
                        auto r = index.searchKnn(test.data() + (q++ % nq) * dim, 10);
                        (void)r;
                        done.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            std::this_thread::sleep_for(std::chrono::seconds(5));
            stop.store(true);
            for (auto& th : ts) th.join();
            printf("%10zu %8u %12.0f\n", ef, nthr,
                   done.load() / std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count());
        }
    }
    return 0;
}
