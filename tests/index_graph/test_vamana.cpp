// Phase 2 exit gate (spec/phase-2.md): recall@10 vs exact brute force at
// small scale here (fast dev-loop unit test); the full 100K/1M SIFT1M recall
// benchmark runs separately via bench/ scripts, not as a unit test, since it
// takes minutes rather than milliseconds.
#include "lattice/index_graph/index_graph.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <limits>
#include <random>
#include <unordered_set>

#include "lattice/index_flat/index_flat.h"
#include "lattice/quant/sq8.h"

using lattice::VectorId;
using lattice::index_graph::ElemType;
using lattice::index_graph::Metric;
using lattice::index_graph::NodeRecordLayout;
using lattice::index_graph::SearchParams;
using lattice::index_graph::VamanaBuildParams;
using lattice::index_graph::VamanaIndex;
using lattice::quant::Sq8CodeBook;
using lattice::quant::Sq8Codes;

namespace {

struct Dataset {
    std::vector<std::vector<float>> storage;
    std::vector<const float*> ptrs;
};

Dataset make_dataset(std::mt19937& rng, std::size_t n, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Dataset d;
    d.storage.resize(n, std::vector<float>(dim));
    for (auto& v : d.storage) {
        for (auto& x : v) x = dist(rng);
    }
    for (auto& v : d.storage) d.ptrs.push_back(v.data());
    return d;
}

double recall_at_k(const std::vector<VectorId>& got, const std::vector<VectorId>& expected,
                    std::uint32_t k) {
    std::unordered_set<std::uint64_t> exp_set;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, expected.size()); ++i) {
        exp_set.insert(static_cast<std::uint64_t>(expected[i]));
    }
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, got.size()); ++i) {
        if (exp_set.count(static_cast<std::uint64_t>(got[i]))) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace

TEST(NodeRecordLayout, RejectsRExceedingSectorCapacity) {
    // 768-d fp32: max R = (4096 - 4 - 3072) / 4 = 255 (finding 4's formula).
    auto ok = NodeRecordLayout::make(768, ElemType::kFloat32, 255);
    EXPECT_TRUE(ok.has_value());
    auto bad = NodeRecordLayout::make(768, ElemType::kFloat32, 256);
    EXPECT_FALSE(bad.has_value());
}

TEST(NodeRecordLayout, DimDependentCapAt960) {
    // 960-d fp32 (GIST parity): max R = (4096-4-3840)/4 = 63.
    auto ok = NodeRecordLayout::make(960, ElemType::kFloat32, 63);
    EXPECT_TRUE(ok.has_value());
    auto bad = NodeRecordLayout::make(960, ElemType::kFloat32, 64);
    EXPECT_FALSE(bad.has_value());
}

TEST(Vamana, BuildAndSearchMatchesBruteForceRecallOnSmallSet) {
    std::mt19937 rng(100);
    constexpr std::uint32_t dim = 64;
    constexpr std::size_t n = 2000;
    auto data = make_dataset(rng, n, dim);

    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);

    auto layout = NodeRecordLayout::make(dim, ElemType::kFloat32, 32);
    ASSERT_TRUE(layout.has_value());

    VamanaBuildParams build_params;
    build_params.r = 32;
    build_params.alpha = 1.2f;
    build_params.build_list_size = 75;
    build_params.metric = Metric::kInnerProduct;

    auto index = VamanaIndex::build(data.ptrs, dim, codes, build_params, *layout);
    ASSERT_TRUE(index.has_value()) << index.error().message;
    EXPECT_EQ(index->size(), n);

    // Flat view for brute-force ground truth.
    std::vector<float> flat(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        std::copy(data.ptrs[i], data.ptrs[i] + dim, flat.begin() + i * dim);
    }
    lattice::index_flat::FlatView view{flat.data(), dim, dim, n};

    SearchParams search_params;
    search_params.ef_search = 100;
    search_params.rerank_k = 20;

    constexpr std::uint32_t k = 10;
    constexpr int num_queries = 30;
    double total_recall = 0.0;
    for (int q = 0; q < num_queries; ++q) {
        auto query = make_dataset(rng, 1, dim);
        auto got = index->search(query.ptrs[0], k, search_params);
        ASSERT_TRUE(got.has_value());

        auto expected = lattice::index_flat::search(view, query.ptrs[0], k);
        ASSERT_TRUE(expected.has_value());

        total_recall += recall_at_k(*got, *expected, k);
    }
    double mean_recall = total_recall / num_queries;
    // Small dataset (n=2000), generous graph params -- expect high recall.
    // The full recall>=0.95 gate at SIFT1M/1M scale is a separate benchmark.
    EXPECT_GT(mean_recall, 0.85) << "mean recall@10 = " << mean_recall;
}

TEST(Vamana, SearchRespectsAdmitFilter) {
    std::mt19937 rng(101);
    constexpr std::uint32_t dim = 32;
    constexpr std::size_t n = 500;
    auto data = make_dataset(rng, n, dim);
    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make(dim, ElemType::kFloat32, 16);
    ASSERT_TRUE(layout.has_value());

    VamanaBuildParams build_params;
    build_params.r = 16;
    build_params.build_list_size = 50;
    auto index = VamanaIndex::build(data.ptrs, dim, codes, build_params, *layout);
    ASSERT_TRUE(index.has_value());

    auto query = make_dataset(rng, 1, dim);
    SearchParams sp;
    sp.ef_search = 50;
    sp.rerank_k = 20;
    auto admit = [](VectorId id) { return static_cast<std::uint64_t>(id) % 2 == 0; };
    auto got = index->search(query.ptrs[0], 10, sp, admit);
    ASSERT_TRUE(got.has_value());
    for (auto id : *got) EXPECT_EQ(static_cast<std::uint64_t>(id) % 2, 0u);
}

TEST(Vamana, BuildRejectsDimensionMismatch) {
    std::mt19937 rng(102);
    auto data = make_dataset(rng, 10, 32);
    auto book = Sq8CodeBook::train(data.ptrs, 32);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make(64, ElemType::kFloat32, 8);  // wrong dim on purpose
    ASSERT_TRUE(layout.has_value());
    VamanaBuildParams params;
    params.r = 8;
    auto index = VamanaIndex::build(data.ptrs, 32, codes, params, *layout);
    EXPECT_FALSE(index.has_value());
}

// Self-query hit rate on hard, unstructured data (random 64-d Gaussian): an index built with several
// threads must not be meaningfully worse than the sequential build. (Absolute rates are modest on
// this adversarial data: ~0.83 at ef=64; hnswlib reaches ~0.97 here. On real SIFT1M recall@1 is on
// par with hnswlib. saturate_graph=true raises connectivity a little.)
TEST(Vamana, ParallelBuildIsNotWorseThanSequentialOnHardData) {
    constexpr std::uint32_t dim = 64;
    constexpr std::size_t n = 20000;
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.f, 1.f);
    Dataset data;
    data.storage.resize(n, std::vector<float>(dim));
    for (auto& v : data.storage) {
        for (auto& x : v) x = gauss(rng);
        data.ptrs.push_back(v.data());
    }
    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, 32);
    ASSERT_TRUE(layout.has_value());

    double rate[2] = {0, 0};
    int slot = 0;
    for (std::uint32_t threads : {1u, 0u}) {
        VamanaBuildParams bp;
        bp.r = 32;
        bp.build_list_size = 100;
        bp.metric = Metric::kL2;
        bp.build_threads = threads;
        auto index = VamanaIndex::build(data.ptrs, dim, codes, bp, *layout);
        ASSERT_TRUE(index.has_value());
        SearchParams sp;
        sp.ef_search = 64;
        std::size_t hits = 0, total = 0;
        for (std::size_t i = 0; i < n; i += 10) {
            auto r = index->search(data.ptrs[i], 1, sp);
            ASSERT_TRUE(r.has_value());
            ++total;
            if (!r->empty() && static_cast<std::size_t>((*r)[0]) == i) ++hits;
        }
        rate[slot++] = static_cast<double>(hits) / static_cast<double>(total);
    }
    std::printf("self-query hit rate: sequential %.4f, parallel %.4f\n", rate[0], rate[1]);
    EXPECT_GT(rate[0], 0.70);
    EXPECT_GE(rate[1], rate[0] - 0.03);
}

// r larger than the record's neighbour capacity used to overrun every node record (heap
// corruption). It must be rejected up front.
TEST(Vamana, BuildRejectsDegreeLargerThanLayoutCapacity) {
    std::mt19937 rng(7);
    auto data = make_dataset(rng, 200, 16);
    auto book = Sq8CodeBook::train(data.ptrs, 16);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(16, ElemType::kFloat32, 8);
    ASSERT_TRUE(layout.has_value());
    VamanaBuildParams params;  // default r = 64 > 8
    auto index = VamanaIndex::build(data.ptrs, 16, codes, params, *layout);
    ASSERT_FALSE(index.has_value());
    EXPECT_EQ(index.error().code, lattice::ErrorCode::kInvalidArgument);
}

// search() used to cap results at rerank_k (default 40) even when k was larger.
TEST(Vamana, SearchReturnsKResultsWhenKExceedsRerankK) {
    std::mt19937 rng(11);
    constexpr std::uint32_t dim = 16;
    auto data = make_dataset(rng, 2000, dim);
    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, 16);
    ASSERT_TRUE(layout.has_value());
    VamanaBuildParams params;
    params.r = 16;
    params.build_threads = 1;
    auto index = VamanaIndex::build(data.ptrs, dim, codes, params, *layout);
    ASSERT_TRUE(index.has_value());
    SearchParams sp;  // rerank_k = 40, ef_search = 64
    auto r = index->search(data.ptrs[0], 100, sp);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->size(), 100u);
}

// NaN/Inf broke the strict weak ordering the sorts rely on (undefined behaviour).
TEST(Vamana, RejectsNonFiniteVectorsAndQueriesAndBadMatchingIds) {
    std::mt19937 rng(5);
    constexpr std::uint32_t dim = 8;
    auto data = make_dataset(rng, 300, dim);
    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, 8);
    ASSERT_TRUE(layout.has_value());
    VamanaBuildParams bp;
    bp.r = 8;
    bp.build_threads = 1;

    data.storage[17][3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(VamanaIndex::build(data.ptrs, dim, codes, bp, *layout).has_value());
    EXPECT_FALSE(Sq8CodeBook::train(data.ptrs, dim).has_value());
    data.storage[17][3] = 0.5f;

    auto index = VamanaIndex::build(data.ptrs, dim, codes, bp, *layout);
    ASSERT_TRUE(index.has_value());
    std::vector<float> q(dim, 0.1f);
    q[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(index->search(q.data(), 5, SearchParams{}).has_value());

    q[0] = 0.1f;
    std::vector<VectorId> ids = {static_cast<VectorId>(1), static_cast<VectorId>(5000)};  // 5000 >= size
    VamanaIndex::FilteredParams fp;
    fp.matching_ids = ids;
    EXPECT_FALSE(index->search_filtered(q.data(), 5, fp, {}).has_value());
}
