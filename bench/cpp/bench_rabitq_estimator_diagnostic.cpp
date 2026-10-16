// Phase 3 exit gate item 2, diagnostic follow-up to section (i)'s finding:
// real SIFT1M at 1M rows collapses to recall@10=0.0094 under RaBitQ
// codes-gated traversal, ~2 orders of magnitude worse than the synthetic
// generator's 0.3133 at identical params -- and item 3 already proves the
// identical graph gets 0.9846 recall under non-codes-gated exact search.
// That isolates the fault to RaBitQ's estimator/traversal interaction, not
// graph quality, but doesn't yet say WHERE: is the 1-bit estimator itself
// unable to discriminate true neighbors from the crowd at 1M-vector scale
// (a codes/quantization problem), or is discrimination fine but the greedy
// beam search's early pruning throws true neighbors away before they're
// ever scored (a traversal/algorithm problem)?
//
// This test isolates the FIRST question directly, with no graph and no
// traversal at all: brute-force score every one of the 1M vectors against
// each query using ONLY RaBitQCodes::distance_to_query_fast (the same
// estimator greedy_search uses), take the estimator's own top-10, and
// measure recall@10 against SIFT1M's ground truth. This is the best case
// for RaBitQ -- with brute force, no beam can ever prune away a true
// neighbor, so ANY recall shortfall measured here is caused by the
// estimator's ranking quality alone, not by traversal/beam-search pruning.
// If recall here is still near-zero, the estimator itself is the problem.
// If it's high (close to item 3's 0.9846), the estimator is fine and the
// beam search's pruning during graph traversal is the actual culprit.
//
// Extended (section j follow-up): also scores the same brute-force setup
// with SQ8 (8-bit/dim, item 2 section f's earlier codes format, abandoned
// in favor of RaBitQ only after a SYNTHETIC-data 30K-row comparison) to
// test whether SQ8's higher per-dimension resolution holds up better than
// RaBitQ's 1-bit code at 1M-row REAL-data scale -- a different regime from
// where the RaBitQ-wins comparison was originally made.
//
// Usage: bench_rabitq_estimator_diagnostic [bin_dir] [num_train] [num_queries]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "lattice/quant/rabitq.h"
#include "lattice/quant/sq8.h"

using namespace lattice;

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
void normalize_in_place(std::vector<float>& v, std::size_t n, std::uint32_t dim) {
    for (std::size_t i = 0; i < n; ++i) {
        float* row = v.data() + i * dim;
        double norm2 = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d) norm2 += static_cast<double>(row[d]) * row[d];
        float inv = norm2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(norm2)) : 0.0f;
        for (std::uint32_t d = 0; d < dim; ++d) row[d] *= inv;
    }
}
double recall_at_k(const std::vector<std::int32_t>& got, const std::int32_t* gt, std::uint32_t k) {
    std::unordered_set<std::int32_t> exp_set;
    for (std::uint32_t i = 0; i < k; ++i) exp_set.insert(gt[i]);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, got.size()); ++i) {
        if (exp_set.count(got[i])) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}
}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "bench/data/sift1m_bin";
    std::size_t num_train_arg = argc > 2 ? std::stoul(argv[2]) : 0;
    std::size_t num_queries_arg = argc > 3 ? std::stoul(argv[3]) : 100;

    std::ifstream meta(data_dir + "/meta.txt");
    std::size_t n_train, dim, n_test, dim2, n_gt, gt_k;
    meta >> n_train >> dim >> n_test >> dim2 >> n_gt >> gt_k;

    std::size_t num_train = num_train_arg == 0 ? n_train : std::min(num_train_arg, n_train);
    std::size_t num_queries = std::min(num_queries_arg, n_test);
    printf("RaBitQ estimator-only diagnostic (brute force, no graph): dim=%zu num_train=%zu "
           "num_queries=%zu\n",
           dim, num_train, num_queries);

    auto train = read_f32(data_dir + "/train.f32", num_train * dim);
    auto test = read_f32(data_dir + "/test.f32", num_queries * dim);
    auto gt = read_i32(data_dir + "/neighbors.i32", n_gt * gt_k);

    normalize_in_place(train, num_train, static_cast<std::uint32_t>(dim));
    normalize_in_place(test, num_queries, static_cast<std::uint32_t>(dim));

    auto book = quant::RaBitQCodeBook::train(static_cast<std::uint32_t>(dim), 42);
    if (!book) {
        printf("RaBitQ train failed: %s\n", book.error().message.c_str());
        return 1;
    }
    quant::RaBitQCodes codes(*book);

    std::vector<const float*> ptrs(num_train);
    for (std::size_t i = 0; i < num_train; ++i) ptrs[i] = train.data() + i * dim;
    auto sq8_book = quant::Sq8CodeBook::train(ptrs, static_cast<std::uint32_t>(dim));
    if (!sq8_book) {
        printf("SQ8 train failed: %s\n", sq8_book.error().message.c_str());
        return 1;
    }
    quant::Sq8Codes sq8_codes(*sq8_book);

    printf("Encoding %zu vectors into RaBitQ + SQ8 codes...\n", num_train);
    auto t_enc0 = std::chrono::steady_clock::now();
    const std::size_t code_bytes = quant::RaBitQCodes::code_bytes(static_cast<std::uint32_t>(dim));
    std::vector<std::byte> all_codes(num_train * code_bytes);
    std::vector<std::byte> all_sq8_codes(num_train * dim);
    for (std::size_t i = 0; i < num_train; ++i) {
        auto c = codes.encode(train.data() + i * dim, static_cast<std::uint32_t>(dim));
        std::copy(c.begin(), c.end(), all_codes.begin() + static_cast<std::ptrdiff_t>(i * code_bytes));
        auto sc = sq8_codes.encode(train.data() + i * dim, static_cast<std::uint32_t>(dim));
        std::copy(sc.begin(), sc.end(), all_sq8_codes.begin() + static_cast<std::ptrdiff_t>(i * dim));
    }
    auto t_enc1 = std::chrono::steady_clock::now();
    printf("Encoding done in %.1f s\n",
           std::chrono::duration<double>(t_enc1 - t_enc0).count());

    // Also keep exact-IP ranking, as a sanity check that our own ranking
    // machinery reproduces SIFT1M's shipped ground truth after
    // unit-normalization (should match ~1.0 recall -- confirms the
    // normalization/metric-equivalence claim from bench_rabitq_sift1m_recall
    // .cpp's header comment, on this exact corpus).
    printf("\n%12s %14s %14s %14s\n", "num_queries", "exact_recall@10", "rabitq_recall@10",
           "sq8_recall@10");

    constexpr std::uint32_t k = 10;
    double total_exact_recall = 0.0;
    double total_rabitq_recall = 0.0;
    double total_sq8_recall = 0.0;
    std::vector<std::size_t> true_neighbor_rabitq_ranks;  // diagnostic: where does each
    std::vector<std::size_t> true_neighbor_sq8_ranks;     // estimator rank the TRUE top-1
                                                            // neighbor, out of num_train?

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* query = test.data() + q * dim;
        auto rotated = codes.rotate_query(query, static_cast<std::uint32_t>(dim));

        std::vector<float> exact_scores(num_train);
        std::vector<float> rabitq_scores(num_train);
        std::vector<float> sq8_scores(num_train);
        for (std::size_t i = 0; i < num_train; ++i) {
            const float* v = train.data() + i * dim;
            double dot = 0.0;
            for (std::uint32_t d = 0; d < dim; ++d) dot += static_cast<double>(query[d]) * v[d];
            exact_scores[i] = static_cast<float>(dot);

            std::span<const std::byte> code(all_codes.data() + i * code_bytes, code_bytes);
            rabitq_scores[i] =
                codes.distance_to_query_fast(rotated.data(), static_cast<std::uint32_t>(dim), code);

            std::span<const std::byte> sq8_code(all_sq8_codes.data() + i * dim, dim);
            sq8_scores[i] = sq8_codes.distance_to_query(query, static_cast<std::uint32_t>(dim), sq8_code);
        }

        std::vector<std::int32_t> idx(num_train);
        std::iota(idx.begin(), idx.end(), 0);

        std::vector<std::int32_t> exact_idx = idx;
        std::partial_sort(exact_idx.begin(), exact_idx.begin() + k, exact_idx.end(),
                           [&](std::int32_t a, std::int32_t b) { return exact_scores[a] > exact_scores[b]; });
        exact_idx.resize(k);

        std::vector<std::int32_t> rabitq_idx = idx;
        std::partial_sort(rabitq_idx.begin(), rabitq_idx.begin() + k, rabitq_idx.end(),
                           [&](std::int32_t a, std::int32_t b) { return rabitq_scores[a] > rabitq_scores[b]; });
        rabitq_idx.resize(k);

        std::vector<std::int32_t> sq8_idx = idx;
        std::partial_sort(sq8_idx.begin(), sq8_idx.begin() + k, sq8_idx.end(),
                           [&](std::int32_t a, std::int32_t b) { return sq8_scores[a] > sq8_scores[b]; });
        sq8_idx.resize(k);

        total_exact_recall += recall_at_k(exact_idx, gt.data() + q * gt_k, k);
        total_rabitq_recall += recall_at_k(rabitq_idx, gt.data() + q * gt_k, k);
        total_sq8_recall += recall_at_k(sq8_idx, gt.data() + q * gt_k, k);

        // Rank of the ground-truth #1 neighbor under each estimator's OWN
        // full ordering over all num_train vectors (0 = estimator agrees
        // it's the best; num_train-1 = estimator thinks it's the worst).
        std::int32_t true_top1 = gt[q * gt_k];
        std::vector<std::int32_t> full_rabitq_idx = idx;
        std::sort(full_rabitq_idx.begin(), full_rabitq_idx.end(),
                  [&](std::int32_t a, std::int32_t b) { return rabitq_scores[a] > rabitq_scores[b]; });
        auto pos = std::find(full_rabitq_idx.begin(), full_rabitq_idx.end(), true_top1);
        true_neighbor_rabitq_ranks.push_back(
            static_cast<std::size_t>(std::distance(full_rabitq_idx.begin(), pos)));

        std::vector<std::int32_t> full_sq8_idx = idx;
        std::sort(full_sq8_idx.begin(), full_sq8_idx.end(),
                  [&](std::int32_t a, std::int32_t b) { return sq8_scores[a] > sq8_scores[b]; });
        auto sq8_pos = std::find(full_sq8_idx.begin(), full_sq8_idx.end(), true_top1);
        true_neighbor_sq8_ranks.push_back(
            static_cast<std::size_t>(std::distance(full_sq8_idx.begin(), sq8_pos)));

        if ((q + 1) % 20 == 0 || q + 1 == num_queries) {
            printf("%12zu %14.4f %14.4f %14.4f\n", q + 1, total_exact_recall / (q + 1),
                   total_rabitq_recall / (q + 1), total_sq8_recall / (q + 1));
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("\nScoring done in %.1f s (%zu queries x %zu vectors)\n",
           std::chrono::duration<double>(t1 - t0).count(), num_queries, num_train);

    auto report_ranks = [&](const char* label, std::vector<std::size_t>& ranks) {
        std::sort(ranks.begin(), ranks.end());
        std::size_t median_rank = ranks[ranks.size() / 2];
        std::size_t p90_rank = ranks[static_cast<std::size_t>(0.90 * (ranks.size() - 1))];
        std::size_t worst_rank = ranks.back();
        printf("\nTrue top-1 neighbor's rank under %s's own ordering (out of %zu):\n", label,
               num_train);
        printf("  median=%zu  p90=%zu  worst=%zu\n", median_rank, p90_rank, worst_rank);
    };
    report_ranks("RaBitQ", true_neighbor_rabitq_ranks);
    report_ranks("SQ8", true_neighbor_sq8_ranks);

    printf("\nexact_recall@10=%.4f  rabitq_recall@10=%.4f  sq8_recall@10=%.4f (brute force, k=%u)\n",
           total_exact_recall / num_queries, total_rabitq_recall / num_queries,
           total_sq8_recall / num_queries, k);
    return 0;
}
