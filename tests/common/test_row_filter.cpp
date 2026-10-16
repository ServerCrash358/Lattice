#include "lattice/common/row_filter.h"

#include <gtest/gtest.h>

#include <bitset>
#include <vector>

using lattice::RowFilter;
using lattice::RowId;

TEST(RowFilter, AlwaysTrueAdmitsEverything) {
    auto f = RowFilter::always_true();
    for (std::uint64_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(f.test(static_cast<RowId>(i)));
    }
    EXPECT_FALSE(f.exact_cardinality().has_value());
}

TEST(RowFilter, CallableDelegates) {
    auto f = RowFilter::from_callable([](RowId id) { return static_cast<std::uint64_t>(id) % 2 == 0; });
    EXPECT_TRUE(f.test(static_cast<RowId>(0)));
    EXPECT_FALSE(f.test(static_cast<RowId>(1)));
    EXPECT_TRUE(f.test(static_cast<RowId>(42)));
}

TEST(RowFilter, BitmapMatchesReferenceBitset) {
    constexpr std::size_t kUniverse = 1000;
    std::bitset<kUniverse> reference;
    for (std::size_t i = 0; i < kUniverse; i += 7) reference.set(i);  // arbitrary pattern

    std::vector<std::uint64_t> words((kUniverse + 63) / 64, 0);
    for (std::size_t i = 0; i < kUniverse; ++i) {
        if (reference[i]) words[i / 64] |= (std::uint64_t{1} << (i % 64));
    }

    auto f = RowFilter::from_bitmap_view(words, kUniverse);
    for (std::size_t i = 0; i < kUniverse; ++i) {
        EXPECT_EQ(f.test(static_cast<RowId>(i)), reference[i]) << "i=" << i;
    }
    ASSERT_TRUE(f.exact_cardinality().has_value());
    EXPECT_EQ(*f.exact_cardinality(), reference.count());
}

TEST(RowFilter, BitmapOutOfRangeIsFalse) {
    std::vector<std::uint64_t> words = {0b101};
    auto f = RowFilter::from_bitmap_view(words, 3);
    EXPECT_FALSE(f.test(static_cast<RowId>(1000)));
}
