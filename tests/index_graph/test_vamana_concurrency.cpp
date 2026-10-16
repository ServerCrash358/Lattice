// Phase 2 exit gate item 4: TSan-clean concurrent query stress test against
// a static built index (D-CON-08). This is what caught the mutable
// last_stats_ data race that an earlier draft of VamanaIndex::search() had
// -- run this under the tsan preset, not just debug/release.
#include "lattice/index_graph/index_graph.h"

#include <gtest/gtest.h>

#include <atomic>
#include <random>
#include <thread>
#include <vector>

using lattice::VectorId;
using lattice::index_graph::ElemType;
using lattice::index_graph::NodeRecordLayout;
using lattice::index_graph::SearchParams;
using lattice::index_graph::VamanaBuildParams;
using lattice::index_graph::VamanaIndex;
using lattice::quant::Sq8CodeBook;
using lattice::quant::Sq8Codes;

TEST(VamanaConcurrency, ConcurrentSearchesOnStaticIndexAreRaceFree) {
    std::mt19937 rng(999);
    constexpr std::uint32_t dim = 32;
    constexpr std::size_t n = 1000;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> storage(n, std::vector<float>(dim));
    for (auto& v : storage) {
        for (auto& x : v) x = dist(rng);
    }
    std::vector<const float*> ptrs;
    for (auto& v : storage) ptrs.push_back(v.data());

    auto book = Sq8CodeBook::train(ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make(dim, ElemType::kFloat32, 16);
    ASSERT_TRUE(layout.has_value());

    VamanaBuildParams build_params;
    build_params.r = 16;
    build_params.build_list_size = 40;
    auto index = VamanaIndex::build(ptrs, dim, codes, build_params, *layout);
    ASSERT_TRUE(index.has_value());

    // Fixed query set, searched concurrently from many threads against the
    // one already-built (read-only from here on) index.
    constexpr int kThreads = 8;
    constexpr int kSearchesPerThread = 200;
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> q(dim);
        for (auto& x : q) x = dist(rng);
        queries.push_back(std::move(q));
    }

    std::atomic<int> total_results{0};
    std::atomic<bool> any_failure{false};

    auto worker = [&](int thread_id) {
        std::mt19937 local_rng(1000 + thread_id);
        std::uniform_int_distribution<std::size_t> qdist(0, queries.size() - 1);
        SearchParams sp;
        sp.ef_search = 30;
        sp.rerank_k = 10;
        for (int i = 0; i < kSearchesPerThread; ++i) {
            const auto& q = queries[qdist(local_rng)];
            VamanaIndex::Stats stats;
            auto result = index->search(q.data(), 5, sp, {}, &stats);
            if (!result.has_value() || result->empty()) {
                any_failure = true;
                continue;
            }
            // Every returned id must be a valid, in-range VectorId -- a real
            // data race (e.g. the old shared mutable-stats bug, or any
            // corruption from concurrent reads of the node store) would
            // tend to surface as an out-of-range or garbage id under TSan
            // interleaving, even if TSan itself doesn't catch it directly.
            for (auto id : *result) {
                if (static_cast<std::uint64_t>(id) >= n) any_failure = true;
            }
            total_results += static_cast<int>(result->size());
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
    for (auto& t : threads) t.join();

    EXPECT_FALSE(any_failure.load());
    EXPECT_GT(total_results.load(), 0);
}
