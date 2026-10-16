// Phase 3 exit gate item 7: grouped top-k (D-RET-07) never under-fills
// below min(k, available_groups) in a synthetic test with a skewed
// group_id distribution.
#include "lattice/common/grouped_top_k.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

using namespace lattice;
using lattice::common::grouped_top_k;

namespace {

// A synthetic ranked corpus: row i has score (corpus_size - i) (so row 0 is
// the best, descending) and a group assigned by the caller-supplied
// `group_of_index` function. fetch_top_n just slices the first n rows,
// which are already the top-n by construction.
struct SyntheticCorpus {
    std::size_t size;
    std::function<std::optional<GroupId>(std::size_t)> group_of_index;

    common::FetchTopN fetch_top_n() const {
        return [this](std::size_t n) -> Result<std::vector<RowId>> {
            std::size_t take = std::min(n, size);
            std::vector<RowId> out;
            out.reserve(take);
            for (std::size_t i = 0; i < take; ++i) out.push_back(static_cast<RowId>(i));
            return out;
        };
    }
    common::GroupOf group_of() const {
        return [this](RowId row) { return group_of_index(static_cast<std::size_t>(row)); };
    }
};

}  // namespace

TEST(GroupedTopK, SkewedDistributionStillFillsToKDistinctGroups) {
    // 10,000 rows. The first 500 (highest-scoring) rows ALL belong to just
    // 3 giant groups (a heavily duplicated/near-duplicate document, the
    // exact D-RET-07 scenario) -- everything after row 500 belongs to its
    // own unique group. A naive "top-k then dedupe" with k=10 would see
    // only 3 distinct groups in its initial window and wrongly return 3
    // results; the corpus actually has thousands of available groups.
    constexpr std::size_t corpus_size = 10'000;
    constexpr std::size_t skew_prefix = 500;
    SyntheticCorpus corpus{corpus_size, [](std::size_t i) -> std::optional<GroupId> {
                                if (i < skew_prefix) return static_cast<GroupId>(i % 3);
                                return static_cast<GroupId>(1000 + i);  // unique per row
                            }};

    constexpr std::uint32_t k = 10;
    auto result = grouped_top_k(k, corpus_size, corpus.group_of(), corpus.fetch_top_n());
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->size(), k) << "10,000 rows have far more than 10 distinct groups available "
                                     "-- must not under-fill just because the top of the ranking "
                                     "is dominated by 3 groups";

    // No duplicate groups in the output (the whole point of "collapsed").
    std::unordered_set<std::uint64_t> seen;
    for (auto row : *result) {
        auto g = corpus.group_of()(row);
        ASSERT_TRUE(g.has_value());
        EXPECT_TRUE(seen.insert(static_cast<std::uint64_t>(*g)).second)
            << "duplicate group in grouped top-k result";
    }

    // Best-scoring member shown: since rows are ranked best-first and each
    // group's first (best) occurrence should be the one kept, results must
    // be in ascending RowId order in this synthetic corpus (lower index =
    // better score), reflecting rank order, not an arbitrary member.
    for (std::size_t i = 1; i < result->size(); ++i) {
        EXPECT_LT(static_cast<std::uint64_t>((*result)[i - 1]), static_cast<std::uint64_t>((*result)[i]));
    }
}

TEST(GroupedTopK, FewerGroupsThanKReturnsAllAvailableGroupsNotK) {
    // Only 4 distinct groups exist in the ENTIRE 200-row corpus. Asking
    // for k=10 must return exactly 4 (min(k, available_groups)), not 10
    // and not fewer than 4.
    constexpr std::size_t corpus_size = 200;
    SyntheticCorpus corpus{corpus_size,
                            [](std::size_t i) -> std::optional<GroupId> { return static_cast<GroupId>(i % 4); }};

    auto result = grouped_top_k(/*k=*/10, corpus_size, corpus.group_of(), corpus.fetch_top_n());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 4u);
}

TEST(GroupedTopK, AllUngroupedRowsAreTheirOwnSingletonGroups) {
    constexpr std::size_t corpus_size = 50;
    SyntheticCorpus corpus{corpus_size, [](std::size_t) -> std::optional<GroupId> { return std::nullopt; }};

    auto result = grouped_top_k(/*k=*/10, corpus_size, corpus.group_of(), corpus.fetch_top_n());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 10u);
    // Ungrouped rows are never collapsed with each other.
    std::unordered_set<std::uint64_t> ids;
    for (auto row : *result) EXPECT_TRUE(ids.insert(static_cast<std::uint64_t>(row)).second);
}

TEST(GroupedTopK, ZeroKReturnsEmpty) {
    constexpr std::size_t corpus_size = 50;
    SyntheticCorpus corpus{corpus_size,
                            [](std::size_t i) -> std::optional<GroupId> { return static_cast<GroupId>(i); }};
    auto result = grouped_top_k(/*k=*/0, corpus_size, corpus.group_of(), corpus.fetch_top_n());
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}
