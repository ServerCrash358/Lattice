// module-index_flat.md's acceptance test: search results match a naive
// O(n*d) reference implementation exactly (this *is* the reference every
// other index's recall test compares against).
#include "lattice/index_flat/index_flat.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

using lattice::VectorId;
using lattice::index_flat::FlatView;
using lattice::index_flat::search;
using lattice::index_flat::search_batched;

namespace {
constexpr std::uint32_t kDim = 128;

std::vector<float> flat_random(std::mt19937& rng, std::size_t n, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<VectorId> naive_top_k(const float* base, std::size_t n, std::uint32_t dim,
                                   const float* query, std::uint32_t k) {
    std::vector<std::pair<float, VectorId>> scored;
    scored.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        float s = 0.0f;
        for (std::uint32_t d = 0; d < dim; ++d) s += query[d] * base[i * dim + d];
        scored.emplace_back(s, static_cast<VectorId>(i));
    }
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    if (scored.size() > k) scored.resize(k);
    std::vector<VectorId> out;
    for (auto& [s, id] : scored) out.push_back(id);
    return out;
}
}  // namespace

TEST(IndexFlat, MatchesNaiveTopKExactly) {
    std::mt19937 rng(7);
    constexpr std::size_t n = 500;
    auto data = flat_random(rng, n, kDim);
    auto query = flat_random(rng, 1, kDim);

    FlatView view{data.data(), kDim, kDim, n};
    auto result = search(view, query.data(), 10);
    ASSERT_TRUE(result.has_value());

    auto expected = naive_top_k(data.data(), n, kDim, query.data(), 10);
    ASSERT_EQ(result->size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ((*result)[i], expected[i]) << "rank=" << i;
    }
}

TEST(IndexFlat, RespectsAdmitFilter) {
    std::mt19937 rng(8);
    constexpr std::size_t n = 200;
    auto data = flat_random(rng, n, kDim);
    auto query = flat_random(rng, 1, kDim);

    FlatView view{data.data(), kDim, kDim, n};
    auto admit = [](VectorId id) { return static_cast<std::uint64_t>(id) % 2 == 0; };
    auto result = search(view, query.data(), 5, admit);
    ASSERT_TRUE(result.has_value());
    for (auto id : *result) {
        EXPECT_EQ(static_cast<std::uint64_t>(id) % 2, 0u);
    }
}

TEST(IndexFlat, KZeroReturnsEmpty) {
    std::mt19937 rng(9);
    auto data = flat_random(rng, 10, kDim);
    auto query = flat_random(rng, 1, kDim);
    FlatView view{data.data(), kDim, kDim, 10};
    auto result = search(view, query.data(), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(IndexFlat, ZeroDimIsAnError) {
    FlatView view{nullptr, 0, 0, 0};
    float dummy = 0.0f;
    auto result = search(view, &dummy, 1);
    EXPECT_FALSE(result.has_value());
}

TEST(IndexFlat, BatchedMatchesPerQuerySearch) {
    std::mt19937 rng(10);
    constexpr std::size_t n = 300;
    auto data = flat_random(rng, n, kDim);
    auto q1 = flat_random(rng, 1, kDim);
    auto q2 = flat_random(rng, 1, kDim);
    auto q3 = flat_random(rng, 1, kDim);

    FlatView view{data.data(), kDim, kDim, n};
    std::vector<const float*> queries = {q1.data(), q2.data(), q3.data()};
    std::vector<std::vector<VectorId>> out(3);
    auto batched_result = search_batched(view, queries, 7, out);
    ASSERT_TRUE(batched_result.has_value());

    for (std::size_t i = 0; i < queries.size(); ++i) {
        auto single = search(view, queries[i], 7);
        ASSERT_TRUE(single.has_value());
        EXPECT_EQ(out[i], *single) << "query=" << i;
    }
}

TEST(IndexFlat, BatchedRejectsMismatchedSizes) {
    std::mt19937 rng(11);
    auto data = flat_random(rng, 10, kDim);
    auto q1 = flat_random(rng, 1, kDim);
    FlatView view{data.data(), kDim, kDim, 10};
    std::vector<const float*> queries = {q1.data()};
    std::vector<std::vector<VectorId>> out(2);  // mismatched on purpose
    auto result = search_batched(view, queries, 3, out);
    EXPECT_FALSE(result.has_value());
}
