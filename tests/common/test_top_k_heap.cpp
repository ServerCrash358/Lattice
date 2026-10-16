#include "lattice/common/top_k_heap.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

using lattice::TopKHeap;

TEST(TopKHeap, KeepsTopKByScore) {
    TopKHeap<float, int> heap(3);
    std::vector<std::pair<float, int>> input = {{5.0f, 0}, {1.0f, 1}, {9.0f, 2},
                                                 {3.0f, 3}, {7.0f, 4}};
    for (auto [s, id] : input) heap.try_insert(s, id);

    auto out = heap.extract_sorted();
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0].first, 9.0f);
    EXPECT_FLOAT_EQ(out[1].first, 7.0f);
    EXPECT_FLOAT_EQ(out[2].first, 5.0f);
}

TEST(TopKHeap, RejectsWorseThanThresholdOnceFull) {
    TopKHeap<float, int> heap(2);
    EXPECT_TRUE(heap.try_insert(5.0f, 0));
    EXPECT_TRUE(heap.try_insert(3.0f, 1));
    EXPECT_FALSE(heap.try_insert(1.0f, 2));  // worse than both, rejected
    EXPECT_TRUE(heap.try_insert(10.0f, 3));  // better than min, accepted
}

TEST(TopKHeap, MatchesNaiveTopKAgainstRandomInput) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (int trial = 0; trial < 20; ++trial) {
        std::size_t n = 50 + (trial * 7) % 100;
        std::size_t k = 1 + (trial % 10);
        std::vector<std::pair<float, int>> all;
        for (std::size_t i = 0; i < n; ++i) all.emplace_back(dist(rng), static_cast<int>(i));

        TopKHeap<float, int> heap(k);
        for (auto [s, id] : all) heap.try_insert(s, id);
        auto got = heap.extract_sorted();

        auto expected = all;
        std::sort(expected.begin(), expected.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        expected.resize(std::min(k, expected.size()));

        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            EXPECT_FLOAT_EQ(got[i].first, expected[i].first) << "trial=" << trial << " i=" << i;
        }
    }
}
