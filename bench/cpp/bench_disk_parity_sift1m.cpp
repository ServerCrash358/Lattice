// Phase 3 exit gate item 3: disk-resident Vamana reproduces the in-RAM
// backend's recall@10 (+-0.01) on the SAME 1M dataset, confirming the
// shared node-record format round-trips correctly across backends.
// tests/segment/test_disk_vamana.cpp already covers this property at
// smoke scale (n=1500, dim=64, synthetic random vectors); this extends it
// to the real, 1M-row SIFT1M dataset the exit gate literally asks for.
//
// One graph is built ONCE (in RAM, using the disk-resident record layout --
// sector_size=4096, not the compact in-RAM stride bench_vamana_sift.cpp
// uses for its own Phase 2 numbers), then serialized to disk and reopened
// as a disk-resident VamanaIndex. Both backends traverse byte-identical
// node records with the identical deterministic algorithm, so the real
// property under test is exact agreement between them, not just "both
// happen to hit similar recall" -- recall-vs-ground-truth is reported too,
// but agreement is the stronger, more direct check of the round-trip.
//
// Usage: bench_disk_parity_sift1m [bin_dir] [num_train] [num_queries] [r] [build_list]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
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
    std::unordered_set<std::int32_t> exp_set;
    for (std::uint32_t i = 0; i < k; ++i) exp_set.insert(gt[i]);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, got.size()); ++i) {
        if (exp_set.count(static_cast<std::int32_t>(got[i]))) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}
}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "bench/data/sift1m_bin";
    std::size_t num_train_arg = argc > 2 ? std::stoul(argv[2]) : 0;  // 0 = full file
    std::size_t num_queries_arg = argc > 3 ? std::stoul(argv[3]) : 100;
    std::uint32_t r = argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 64;
    std::uint32_t build_list = argc > 5 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 100;

    std::ifstream meta(data_dir + "/meta.txt");
    std::size_t n_train, dim, n_test, dim2, n_gt, gt_k;
    meta >> n_train >> dim >> n_test >> dim2 >> n_gt >> gt_k;

    std::size_t num_train = num_train_arg == 0 ? n_train : std::min(num_train_arg, n_train);
    std::size_t num_queries = std::min(num_queries_arg, n_test);
    printf("SIFT parity: dim=%zu num_train=%zu (file has %zu) num_queries=%zu R=%u build_list=%u\n",
           dim, num_train, n_train, num_queries, r, build_list);

    auto train = read_f32(data_dir + "/train.f32", num_train * dim);
    auto test = read_f32(data_dir + "/test.f32", num_queries * dim);
    auto gt = read_i32(data_dir + "/neighbors.i32", n_gt * gt_k);

    std::vector<const float*> ptrs(num_train);
    for (std::size_t i = 0; i < num_train; ++i) ptrs[i] = train.data() + i * dim;

    printf("Training SQ8 codebook...\n");
    auto book = quant::Sq8CodeBook::train(ptrs, static_cast<std::uint32_t>(dim));
    if (!book) {
        printf("SQ8 train failed: %s\n", book.error().message.c_str());
        return 1;
    }
    quant::Sq8Codes codes(*book);

    // Disk-resident record layout (sector_size=4096, D-STO-04) -- NOT the
    // Phase-2-only compact stride bench_vamana_sift.cpp uses. This is the
    // exact layout that gets serialized and read back, so both backends
    // agree on record structure by construction, not by coincidence.
    auto layout = NodeRecordLayout::make(static_cast<std::uint32_t>(dim), ElemType::kFloat32, r);
    if (!layout) {
        printf("Layout rejected: %s\n", layout.error().message.c_str());
        return 1;
    }

    VamanaBuildParams build_params;
    build_params.r = r;
    build_params.alpha = 1.2f;
    build_params.build_list_size = build_list;
    build_params.metric = Metric::kL2;  // SIFT1M ground truth is exact L2 (D-VEC-01 parity mode)

    printf("Building Vamana index (in-RAM, disk-resident layout): n=%zu dim=%zu R=%u build_list=%u ...\n",
           num_train, dim, r, build_list);
    auto t0 = std::chrono::steady_clock::now();
    auto ram_index = VamanaIndex::build(ptrs, static_cast<std::uint32_t>(dim), codes, build_params, *layout);
    auto t1 = std::chrono::steady_clock::now();
    if (!ram_index) {
        printf("Build failed: %s\n", ram_index.error().message.c_str());
        return 1;
    }
    printf("Build done in %.1f s (%.0f vectors/s)\n",
           std::chrono::duration<double>(t1 - t0).count(),
           num_train / std::chrono::duration<double>(t1 - t0).count());

    // /tmp (WSL-local ext4), not bench/data/ (the slow 9p-mounted F: drive
    // this repo lives on) -- avoids paying 9p round-trip latency on every
    // serialize/reopen I/O for what's otherwise a compute-bound benchmark.
    std::string disk_path = "/tmp/sift1m_parity.vamana";
    auto ser = ram_index->serialize_to_disk(disk_path);
    if (!ser) {
        printf("serialize_to_disk failed: %s\n", ser.error().message.c_str());
        return 1;
    }
    auto disk_index = segment::open_disk_resident_vamana(disk_path);
    if (!disk_index) {
        printf("open_disk_resident_vamana failed: %s\n", disk_index.error().message.c_str());
        return 1;
    }
    printf("Disk-resident index reopened: size=%zu (expected %zu)\n", disk_index->size(), num_train);

    SearchParams sp;
    sp.ef_search = 100;
    sp.rerank_k = 40;
    constexpr std::uint32_t k = 10;

    double total_recall_ram = 0.0, total_recall_disk = 0.0;
    std::size_t exact_agreement = 0;

    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* query = test.data() + q * dim;
        auto ram_result = ram_index->search(query, k, sp);
        auto disk_result = disk_index->search(query, k, sp);
        if (!ram_result || !disk_result) {
            printf("query %zu: search failed (ram=%d disk=%d)\n", q, ram_result.has_value(),
                   disk_result.has_value());
            continue;
        }
        if (*ram_result == *disk_result) ++exact_agreement;

        const std::int32_t* q_gt = gt.data() + q * gt_k;
        total_recall_ram += recall_at_k(*ram_result, q_gt, k);
        total_recall_disk += recall_at_k(*disk_result, q_gt, k);
    }

    double mean_recall_ram = total_recall_ram / num_queries;
    double mean_recall_disk = total_recall_disk / num_queries;
    double agreement_rate = static_cast<double>(exact_agreement) / num_queries;

    printf("\n%20s %10.4f\n", "recall@10 (RAM)", mean_recall_ram);
    printf("%20s %10.4f\n", "recall@10 (disk)", mean_recall_disk);
    printf("%20s %10.4f\n", "abs(diff)", std::abs(mean_recall_ram - mean_recall_disk));
    printf("%20s %10.4f\n", "exact agreement", agreement_rate);

    bool pass = std::abs(mean_recall_ram - mean_recall_disk) <= 0.01;
    printf("\n%s: |recall_ram - recall_disk| = %.4f (gate: <= 0.01)\n", pass ? "PASS" : "FAIL",
           std::abs(mean_recall_ram - mean_recall_disk));
    return pass ? 0 : 1;
}
