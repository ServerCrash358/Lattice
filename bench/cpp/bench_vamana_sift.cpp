// Phase 2 exit gate (spec/phase-2.md): recall@10 >= 0.95 and QPS within
// 1.5x hnswlib at equal recall, on SIFT1M at 100K/1M scale. This program
// builds a Vamana index over an exported SIFT1M subset and reports the
// recall/QPS curve across an ef_search sweep -- the raw numbers to compare
// against the hnswlib run from Phase 0's reproduce_hnswlib.py.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <unordered_set>
#include <vector>

#include "lattice/index_graph/index_graph.h"
#include "lattice/quant/sq8.h"

using namespace lattice;
using namespace lattice::index_graph;

namespace {

std::vector<float> read_f32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    return data;
}

std::vector<std::int32_t> read_i32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::int32_t> data(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(std::int32_t));
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "bench/data/sift1m_bin";
    std::uint32_t r = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 64;
    std::uint32_t build_list = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 100;

    // meta.txt: "N_train dim\nN_test dim\nN_gt gt_k\n"
    std::ifstream meta(data_dir + "/meta.txt");
    std::size_t n_train, dim, n_test, dim2, n_gt, gt_k;
    meta >> n_train >> dim >> n_test >> dim2 >> n_gt >> gt_k;

    printf("Loading: %zu train, %zu test, dim=%zu, gt_k=%zu\n", n_train, n_test, dim, gt_k);
    auto train = read_f32(data_dir + "/train.f32", n_train * dim);
    auto test = read_f32(data_dir + "/test.f32", n_test * dim);
    auto gt = read_i32(data_dir + "/neighbors.i32", n_gt * gt_k);

    std::vector<const float*> train_ptrs(n_train);
    for (std::size_t i = 0; i < n_train; ++i) train_ptrs[i] = train.data() + i * dim;

    printf("Training SQ8 codebook...\n");
    auto book = quant::Sq8CodeBook::train(train_ptrs, static_cast<std::uint32_t>(dim));
    if (!book) {
        printf("SQ8 train failed: %s\n", book.error().message.c_str());
        return 1;
    }
    quant::Sq8Codes codes(*book);

    // Phase 2 in-RAM backend: use a compact cache-line-aligned stride, not
    // the 4096-byte disk-sector padding Phase 3's disk backend needs --
    // paying that padding cost here means touching ~5x more memory per
    // node visit than necessary at 128-d (see index_graph.h's comment on
    // minimal_aligned_stride).
    auto stride = NodeRecordLayout::minimal_aligned_stride(static_cast<std::uint32_t>(dim),
                                                             ElemType::kFloat32, r);
    auto layout = NodeRecordLayout::make(static_cast<std::uint32_t>(dim), ElemType::kFloat32, r, stride);
    if (!layout) {
        printf("Layout rejected: %s\n", layout.error().message.c_str());
        return 1;
    }

    VamanaBuildParams build_params;
    build_params.r = r;
    build_params.alpha = 1.2f;
    build_params.build_list_size = build_list;
    build_params.metric = Metric::kL2;  // SIFT1M ground truth is exact L2 (D-VEC-01 parity mode)

    printf("Building Vamana index: n=%zu dim=%zu R=%u build_list=%u ...\n", n_train, dim, r,
           build_list);
    auto t0 = std::chrono::steady_clock::now();
    auto index = VamanaIndex::build(train_ptrs, static_cast<std::uint32_t>(dim), codes,
                                     build_params, *layout);
    auto t1 = std::chrono::steady_clock::now();
    if (!index) {
        printf("Build failed: %s\n", index.error().message.c_str());
        return 1;
    }
    double build_secs = std::chrono::duration<double>(t1 - t0).count();
    printf("Build done in %.1f s (%.0f vectors/s)\n", build_secs, n_train / build_secs);

    printf("\n%10s %10s %10s\n", "ef_search", "recall@10", "qps");
    for (std::uint32_t ef : {10u, 20u, 40u, 80u, 160u, 320u}) {
        SearchParams sp;
        sp.ef_search = ef;
        sp.rerank_k = std::max(40u, ef / 2);

        std::size_t hits = 0, total = 0;
        auto qt0 = std::chrono::steady_clock::now();
        for (std::size_t qi = 0; qi < n_test; ++qi) {
            const float* q = test.data() + qi * dim;
            auto result = index->search(q, 10, sp);
            if (!result) continue;
            std::unordered_set<std::int32_t> in_gt;
            for (std::size_t g = 0; g < 10 && g < gt_k; ++g) {
                in_gt.insert(gt[qi * gt_k + g]);
            }
            for (auto id : *result) {
                if (in_gt.count(static_cast<std::int32_t>(id))) ++hits;
            }
            total += 10;
        }
        auto qt1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(qt1 - qt0).count();
        double recall = static_cast<double>(hits) / static_cast<double>(total);
        double qps = n_test / secs;
        printf("%10u %10.4f %10.1f\n", ef, recall, qps);
    }
    return 0;
}
