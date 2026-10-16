// Phase 3 exit gate item 8 (first half): RaBitQCodes::distance_to_query_
// bounded's error bound empirically validated on real data -- SIFT1M and a
// real text-embedding set -- not just the synthetic uniform-random vectors
// test_rabitq.cpp already covers. Real embeddings cluster and have
// non-uniform per-dimension statistics; the bound's derivation (Cauchy-
// Schwarz on the dropped residual, see rabitq.h) doesn't assume any
// particular data distribution, but this is what actually checks that
// instead of just arguing it algebraically.
//
// Usage:
//   bench_rabitq_real_data <label> <bin_dir>
// <bin_dir> must contain meta.txt/train.f32/test.f32 in the same flat
// format export_sift_binary.py produces (works for any dataset exported
// that way, not just SIFT -- hence taking dim from meta.txt rather than
// hardcoding 128).
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "lattice/quant/rabitq.h"

using lattice::quant::RaBitQCodeBook;
using lattice::quant::RaBitQCodes;

namespace {
std::vector<float> read_f32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return data;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <label> <bin_dir> [num_database=5000] [num_queries=200]\n", argv[0]);
        return 2;
    }
    std::string label = argv[1];
    std::string bin_dir = argv[2];
    std::size_t num_database = argc > 3 ? std::stoul(argv[3]) : 5000;
    std::size_t num_queries = argc > 4 ? std::stoul(argv[4]) : 200;

    std::ifstream meta(bin_dir + "/meta.txt");
    if (!meta) {
        printf("[%s] could not open %s/meta.txt\n", label.c_str(), bin_dir.c_str());
        return 1;
    }
    std::size_t n_train, dim, n_test, dim2, n_gt, gt_k;
    meta >> n_train >> dim >> n_test >> dim2 >> n_gt >> gt_k;

    num_database = std::min(num_database, n_train);
    num_queries = std::min(num_queries, n_test);
    printf("[%s] dim=%zu train_used=%zu test_used=%zu (file has %zu train, %zu test)\n", label.c_str(),
           dim, num_database, num_queries, n_train, n_test);

    auto train = read_f32(bin_dir + "/train.f32", num_database * dim);
    auto test = read_f32(bin_dir + "/test.f32", num_queries * dim);

    auto book = RaBitQCodeBook::train(static_cast<std::uint32_t>(dim), /*seed=*/42);
    if (!book) {
        printf("[%s] RaBitQCodeBook::train failed: %s\n", label.c_str(), book.error().message.c_str());
        return 1;
    }
    RaBitQCodes codes(*book);

    std::vector<std::vector<std::byte>> encoded(num_database);
    for (std::size_t i = 0; i < num_database; ++i) {
        encoded[i] = codes.encode(train.data() + i * dim, static_cast<std::uint32_t>(dim));
    }

    std::size_t violations = 0, total_checks = 0;
    double max_slack_ratio = 0.0, sum_abs_err = 0.0, sum_bound = 0.0;

    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* query = test.data() + q * dim;
        for (std::size_t i = 0; i < num_database; ++i) {
            float exact = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) exact += query[d] * train[i * dim + d];

            auto bd = codes.distance_to_query_bounded(query, static_cast<std::uint32_t>(dim), encoded[i]);
            float actual_error = std::abs(bd.estimate - exact);

            ++total_checks;
            sum_abs_err += actual_error;
            sum_bound += bd.error_bound;
            // Same fp32-accumulation slack as test_rabitq.cpp's synthetic test.
            if (actual_error > bd.error_bound * 1.001f + 1e-4f) ++violations;
            if (bd.error_bound > 1e-6f) {
                max_slack_ratio = std::max(max_slack_ratio,
                                            static_cast<double>(actual_error / bd.error_bound));
            }
        }
    }

    printf("[%s] checks=%zu violations=%zu mean_abs_err=%.4f mean_bound=%.4f max_observed/bound=%.4f\n",
           label.c_str(), total_checks, violations, sum_abs_err / total_checks, sum_bound / total_checks,
           max_slack_ratio);
    if (violations > 0) {
        printf("[%s] FAIL: error bound violated %zu/%zu times\n", label.c_str(), violations, total_checks);
        return 1;
    }
    printf("[%s] PASS: error bound held across all %zu checks\n", label.c_str(), total_checks);
    return 0;
}
