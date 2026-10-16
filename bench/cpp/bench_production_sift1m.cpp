// End-to-end SIFT1M benchmark through the PRODUCTION code path: VamanaIndex::build
// (SQ8 codes-gated, L2 on raw vectors) -> serialize_to_disk -> reopen via
// segment::open_disk_resident_vamana (io_uring node reads) -> search.
//
// The built index is cached at <index_path>; if it already exists the (slow,
// single-threaded) build is skipped and only the search sweep runs.
//
// Usage: bench_production_sift1m <bin_dir> <index_path> [num_queries] [r] [build_list]
//
// Note: node reads go through the OS page cache (a warm cache is typical for a
// benchmark of this size); this is "disk-resident" in layout and I/O path, not
// a cold-NVMe measurement.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <optional>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "lattice/quant/sq8.h"
#include "lattice/segment/segment.h"

using namespace lattice;
using namespace lattice::index_graph;

namespace {
std::vector<float> read_f32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return data;
}
std::vector<std::int32_t> read_i32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::int32_t> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(std::int32_t)));
    return data;
}
double recall_at_k(const std::vector<VectorId>& got, const std::int32_t* gt, std::uint32_t k) {
    std::unordered_set<std::int32_t> exp_set(gt, gt + k);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, got.size()); ++i)
        if (exp_set.count(static_cast<std::int32_t>(got[i]))) ++hits;
    return static_cast<double>(hits) / k;
}
}  // namespace

static void print_memory(const char* when) {
    std::ifstream f("/proc/self/status");
    std::string line;
    double anon = 0, file = 0;
    while (std::getline(f, line)) {
        if (line.rfind("RssAnon:", 0) == 0) anon = std::stod(line.substr(8)) / 1024.0;
        if (line.rfind("RssFile:", 0) == 0) file = std::stod(line.substr(8)) / 1024.0;
    }
    printf("Memory %s: heap (SQ8 codes etc.) = %.0f MB, file-backed index pages in page cache = %.0f MB\n", when, anon, file);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <bin_dir> <index_path> [num_queries] [r] [build_list]\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    std::string index_path = argv[2];
    std::size_t nq_arg = argc > 3 ? std::stoul(argv[3]) : 1000;
    std::uint32_t r = argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 64;
    std::uint32_t build_list = argc > 5 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 100;
    const bool ram_mode = argc > 6 && std::string(argv[6]) == "ram";  // compact in-RAM node store, no disk path
    const bool iouring = argc > 6 && std::string(argv[6]) == "iouring";  // io_uring reads instead of mmap
    const bool preload = argc > 6 && std::string(argv[6]) == "coldpre";  // evict, then open with preload=true
    const bool cold = argc > 6 && (std::string(argv[6]) == "cold" || preload);  // evict page cache, report warm-up curve
    std::optional<VamanaIndex> ram_idx;

    std::ifstream meta(dir + "/meta.txt");
    std::size_t n_train, dim, n_test, dim2, n_gt, gt_k;
    meta >> n_train >> dim >> n_test >> dim2 >> n_gt >> gt_k;
    std::size_t nq = std::min(nq_arg, n_test);
    auto test = read_f32(dir + "/test.f32", nq * dim);
    auto gt = read_i32(dir + "/neighbors.i32", n_gt * gt_k);
    printf("Production path, real SIFT1M: n=%zu dim=%zu queries=%zu R=%u build_list=%u metric=L2 (raw vectors)\n",
           n_train, dim, nq, r, build_list);

    if (ram_mode || !std::filesystem::exists(index_path)) {
        auto train = read_f32(dir + "/train.f32", n_train * dim);
        std::vector<const float*> ptrs(n_train);
        for (std::size_t i = 0; i < n_train; ++i) ptrs[i] = train.data() + i * dim;
        auto book = quant::Sq8CodeBook::train(ptrs, static_cast<std::uint32_t>(dim));
        if (!book) { printf("sq8 train failed\n"); return 1; }
        quant::Sq8Codes codes(*book);
        auto layout = NodeRecordLayout::make_compact(static_cast<std::uint32_t>(dim), ElemType::kFloat32, r);
        if (!layout) { printf("layout rejected: %s\n", layout.error().message.c_str()); return 1; }
        VamanaBuildParams bp;
        bp.r = r;
        bp.alpha = 1.2f;
        bp.build_list_size = build_list;
        bp.metric = Metric::kL2;
        printf("Building (single-threaded)...\n");
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        auto index = VamanaIndex::build(ptrs, static_cast<std::uint32_t>(dim), codes, bp, *layout);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!index) { printf("build failed: %s\n", index.error().message.c_str()); return 1; }
        printf("Build done in %.1f s\n", secs);
        if (ram_mode) {
            ram_idx.emplace(std::move(*index));
            printf("RAM mode: compact node store, stride=%zu B/node\n", ram_idx->layout().record_bytes());
        }
        else {
        auto ser = index->serialize_to_disk(index_path);
        if (!ser) { printf("serialize failed: %s\n", ser.error().message.c_str()); return 1; }
        }
    } else {
        printf("Reusing cached index at %s\n", index_path.c_str());
    }

    if (cold) {
        int fd = ::open(index_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
            printf("Evicted %s from the page cache (host-side cache, if any, may still serve reads)\n", index_path.c_str());
        }
    }
    auto open_t0 = std::chrono::steady_clock::now();
    auto disk = ram_mode ? Result<VamanaIndex>(std::unexpected(Error::make(ErrorCode::kInvalidArgument, "ram mode"))) : segment::open_disk_resident_vamana(index_path, iouring ? segment::NodeAccess::kIoUring : segment::NodeAccess::kMmap, preload);
    if (!ram_mode && !disk) { printf("open failed: %s\n", disk.error().message.c_str()); return 1; }
    VamanaIndex* idx = ram_mode ? &*ram_idx : &*disk;
    if (preload) {
        printf("open+preload took %.2f s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - open_t0).count());
    }
    printf("%s index: %zu vectors\n", ram_mode ? "In-RAM" : (iouring ? "Disk-resident (io_uring)" : "Disk-resident (mmap)"), idx->size());

    if (cold) {
        for (std::uint32_t ef : {64u, 512u}) {
            SearchParams sp;
            sp.ef_search = ef;
            sp.rerank_k = std::max<std::uint32_t>(40, ef / 2);
            printf("\nCold start, ef_search=%u: latency of consecutive 500-query buckets (first query ever = bucket 1)\n", ef);
            printf("%8s %10s %10s %10s\n", "bucket", "mean_ms", "p50_ms", "p99_ms");
            for (std::size_t b = 0; b < 8 && (b + 1) * 500 <= nq; ++b) {
                std::vector<double> lat;
                for (std::size_t q = b * 500; q < (b + 1) * 500; ++q) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto res = idx->search(test.data() + q * dim, 10, sp);
                    (void)res;
                    lat.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                double mean = 0;
                for (double l : lat) mean += l;
                std::sort(lat.begin(), lat.end());
                printf("%8zu %10.2f %10.2f %10.2f\n", b + 1, mean / lat.size(), lat[lat.size() / 2],
                       lat[static_cast<std::size_t>(0.99 * (lat.size() - 1))]);
                fflush(stdout);
            }
            // evict again before the next ef so each curve starts cold
            int fd = ::open(index_path.c_str(), O_RDONLY);
            if (fd >= 0) { ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
        }
        return 0;
    }

    printf("\n%10s %10s %10s %10s %10s %12s\n", "ef_search", "recall@1", "recall@10", "p50_ms", "p99_ms", "QPS_1thread");
    for (std::uint32_t ef : {16u, 32u, 64u, 96u, 128u, 192u, 256u, 512u, 1024u, 2048u}) {
        SearchParams sp;
        sp.ef_search = ef;
        sp.rerank_k = std::max<std::uint32_t>(40, ef / 2);
        std::vector<double> lat;
        double r1 = 0, r10 = 0;
        for (std::size_t q = 0; q < nq; ++q) {
            auto t0 = std::chrono::steady_clock::now();
            auto res = idx->search(test.data() + q * dim, 10, sp);
            lat.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if (!res) continue;
            r10 += recall_at_k(*res, gt.data() + q * gt_k, 10);
            if (!res->empty() && static_cast<std::int32_t>((*res)[0]) == gt[q * gt_k]) r1 += 1.0;
        }
        double total_ms = 0;
        for (double l : lat) total_ms += l;
        std::sort(lat.begin(), lat.end());
        printf("%10u %10.4f %10.4f %10.2f %10.2f %12.0f\n", ef, r1 / nq, r10 / nq, lat[lat.size() / 2],
               lat[static_cast<std::size_t>(0.99 * (lat.size() - 1))], nq / (total_ms / 1000.0));
        fflush(stdout);
    }

    print_memory("after the recall sweep (warm)");
    unsigned hw = std::thread::hardware_concurrency();
    printf("\nThroughput on the disk-resident index (hardware threads=%u, ~10s each):\n", hw);
    printf("%10s %8s %12s\n", "ef_search", "threads", "QPS");
    for (std::uint32_t ef : {64u, 128u, 256u, 512u}) {
        SearchParams sp;
        sp.ef_search = ef;
        sp.rerank_k = std::max<std::uint32_t>(40, ef / 2);
        for (unsigned nt : {1u, 4u, 8u, hw}) {
            if (nt > hw) continue;
            std::atomic<std::uint64_t> done{0};
            std::atomic<bool> stop{false};
            std::vector<std::thread> ts;
            auto start = std::chrono::steady_clock::now();
            for (unsigned t = 0; t < nt; ++t) {
                ts.emplace_back([&, t] {
                    std::size_t q = t * 7;
                    while (!stop.load(std::memory_order_relaxed)) {
                        auto res = idx->search(test.data() + (q++ % nq) * dim, 10, sp);
                        (void)res;
                        done.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));
            stop.store(true);
            for (auto& th : ts) th.join();
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("%10u %8u %12.0f\n", ef, nt, done.load() / secs);
            fflush(stdout);

        }
    }
    return 0;
}
