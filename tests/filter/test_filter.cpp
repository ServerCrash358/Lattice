// D-CPP-07 property test named directly in the spec: filtered_result ⊆
// filter -- every row a RowFilter derived from a predicate admits must
// actually satisfy that predicate. Also covers D-QRY-01's headline claim:
// selectivity_count is always exact, never an estimate.
#include "lattice/filter/filter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <random>
#include <thread>
#include <unordered_set>

using namespace lattice::filter;
using lattice::RowId;

TEST(Bitmap, BasicOpsMatchReference) {
    Bitmap a, b;
    a.add(static_cast<RowId>(1));
    a.add(static_cast<RowId>(2));
    a.add(static_cast<RowId>(3));
    b.add(static_cast<RowId>(2));
    b.add(static_cast<RowId>(3));
    b.add(static_cast<RowId>(4));

    EXPECT_EQ(a.cardinality(), 3u);
    auto and_ab = a.and_with(b);
    EXPECT_EQ(and_ab.cardinality(), 2u);
    EXPECT_TRUE(and_ab.contains(static_cast<RowId>(2)));
    EXPECT_TRUE(and_ab.contains(static_cast<RowId>(3)));
    EXPECT_FALSE(and_ab.contains(static_cast<RowId>(1)));

    auto or_ab = a.or_with(b);
    EXPECT_EQ(or_ab.cardinality(), 4u);
}

TEST(TypedColumn, EqualityBitmapExactCardinality) {
    TypedColumn col(ColumnKind::kInt);
    // tenant=42 for rows 0,5,10; tenant=7 for row 1.
    col.set(static_cast<RowId>(0), 42);
    col.set(static_cast<RowId>(5), 42);
    col.set(static_cast<RowId>(10), 42);
    col.set(static_cast<RowId>(1), 7);

    EXPECT_EQ(col.bitmap_for(42).cardinality(), 3u);
    EXPECT_EQ(col.bitmap_for(7).cardinality(), 1u);
    EXPECT_EQ(col.bitmap_for(999).cardinality(), 0u);  // never seen -> empty
}

TEST(RangeIndex, ReturnsExactRowsInRange) {
    RangeIndex<std::int64_t> idx;
    idx.add(static_cast<RowId>(0), 10);
    idx.add(static_cast<RowId>(1), 20);
    idx.add(static_cast<RowId>(2), 30);
    idx.add(static_cast<RowId>(3), 40);

    auto bm = idx.rows_in_range(15, 35);
    EXPECT_EQ(bm.cardinality(), 2u);
    EXPECT_TRUE(bm.contains(static_cast<RowId>(1)));
    EXPECT_TRUE(bm.contains(static_cast<RowId>(2)));

    auto mm = idx.segment_min_max();
    ASSERT_TRUE(mm.has_value());
    EXPECT_EQ(mm->first, 10);
    EXPECT_EQ(mm->second, 40);
}

TEST(PredicateEvaluator, EqAndInAndCompositePredicatesAreExact) {
    constexpr std::size_t n = 1000;
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> tenant_dist(0, 9);
    std::uniform_int_distribution<int> category_dist(0, 4);

    auto tenant_col = std::make_shared<TypedColumn>(ColumnKind::kInt);
    auto category_col = std::make_shared<TypedColumn>(ColumnKind::kInt);
    std::vector<int> tenants(n), categories(n);
    for (std::size_t i = 0; i < n; ++i) {
        tenants[i] = tenant_dist(rng);
        categories[i] = category_dist(rng);
        tenant_col->set(static_cast<RowId>(i), tenants[i]);
        category_col->set(static_cast<RowId>(i), categories[i]);
    }

    PredicateEvaluator ev;
    ev.add_column("tenant", tenant_col);
    ev.add_column("category", category_col);
    ev.set_universe_size(n);

    Predicate tenant_eq{PredicateOp::kEq, "tenant", {3}, {}, {}, {}, {}, {}};
    Predicate category_eq{PredicateOp::kEq, "category", {1}, {}, {}, {}, {}, {}};
    Predicate both{PredicateOp::kAnd, "", {}, {}, {}, {}, {}, {tenant_eq, category_eq}};

    auto result = ev.evaluate(both);
    ASSERT_TRUE(result.has_value());

    // Reference: brute-force scan.
    std::size_t expected_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        bool matches = tenants[i] == 3 && categories[i] == 1;
        EXPECT_EQ(result->contains(static_cast<RowId>(i)), matches) << "row " << i;
        if (matches) ++expected_count;
    }
    EXPECT_EQ(result->cardinality(), expected_count);
    auto count = ev.selectivity_count(both);
    ASSERT_TRUE(count.has_value());
    EXPECT_EQ(*count, expected_count);  // exact, matches D-QRY-01's claim
}

TEST(PredicateEvaluator, RowFilterResultIsSubsetOfPredicate) {
    // The named property test: filtered_result subseteq filter.
    constexpr std::size_t n = 500;
    std::mt19937 rng(2);
    std::uniform_int_distribution<int> dist(0, 3);

    auto col = std::make_shared<TypedColumn>(ColumnKind::kInt);
    std::vector<int> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = dist(rng);
        col->set(static_cast<RowId>(i), values[i]);
    }

    PredicateEvaluator ev;
    ev.add_column("category", col);
    ev.set_universe_size(n);

    Predicate pred{PredicateOp::kIn, "category", {0, 2}, {}, {}, {}, {}, {}};
    auto bm = ev.evaluate(pred);
    ASSERT_TRUE(bm.has_value());
    auto row_filter = ev.to_row_filter_result(*bm);

    for (std::size_t i = 0; i < n; ++i) {
        bool filter_says_yes = row_filter.test(static_cast<RowId>(i));
        bool predicate_actually_true = (values[i] == 0 || values[i] == 2);
        // Every row the RowFilter admits must satisfy the predicate (subset
        // property) -- and since this predicate is exactly representable as
        // a bitmap, it should also be a perfect match here, not just a subset.
        EXPECT_EQ(filter_says_yes, predicate_actually_true) << "row " << i;
    }
}

// The first rows_in_range() call sorts lazily; concurrent first calls used to race on that sort.
TEST(RangeIndex, ConcurrentFirstQueriesAreSafeAndExact) {
    RangeIndex<std::int64_t> idx;
    std::mt19937 rng(9);
    std::uniform_int_distribution<std::int64_t> val(0, 999);
    std::vector<std::int64_t> values(20000);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = val(rng);
        idx.add(static_cast<RowId>(i), values[i]);
    }
    std::size_t expected = 0;
    for (auto v : values) expected += (v >= 100 && v <= 299);

    std::vector<std::thread> ts;
    std::atomic<int> wrong{0};
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&] {
            if (idx.rows_in_range(100, 299).cardinality() != expected) ++wrong;
        });
    for (auto& th : ts) th.join();
    EXPECT_EQ(wrong.load(), 0);
}

TEST(TypedColumn, SetReplacesSingleValueButTagSetAccumulates) {
    TypedColumn col(ColumnKind::kInt);
    col.set(static_cast<RowId>(1), 10);
    col.set(static_cast<RowId>(1), 20);  // row 1 changed value
    EXPECT_FALSE(col.bitmap_for(10).contains(static_cast<RowId>(1)));
    EXPECT_TRUE(col.bitmap_for(20).contains(static_cast<RowId>(1)));

    TypedColumn tags(ColumnKind::kTagSet);
    tags.set(static_cast<RowId>(1), 10);
    tags.set(static_cast<RowId>(1), 20);  // second tag on the same row
    EXPECT_TRUE(tags.bitmap_for(10).contains(static_cast<RowId>(1)));
    EXPECT_TRUE(tags.bitmap_for(20).contains(static_cast<RowId>(1)));
}

TEST(PredicateEvaluator, SelectivityCountReportsErrorsInsteadOfZero) {
    PredicateEvaluator ev;
    ev.set_universe_size(10);
    Predicate p;
    p.op = PredicateOp::kEq;
    p.column = "no_such_column";
    p.int_literals = {1};
    EXPECT_FALSE(ev.selectivity_count(p).has_value());
}

TEST(Bitmap, NotWithFlipsWithinUniverse) {
    Bitmap b;
    b.add(static_cast<RowId>(1));
    b.add(static_cast<RowId>(3));
    auto n = b.not_with(5);
    EXPECT_EQ(n.cardinality(), 3u);
    EXPECT_TRUE(n.contains(static_cast<RowId>(0)));
    EXPECT_FALSE(n.contains(static_cast<RowId>(1)));
    EXPECT_TRUE(n.contains(static_cast<RowId>(4)));
    EXPECT_FALSE(n.contains(static_cast<RowId>(5)));
}
