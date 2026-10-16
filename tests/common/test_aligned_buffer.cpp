#include "lattice/common/aligned_buffer.h"

#include <gtest/gtest.h>

#include <cstdint>

using lattice::AlignedBuffer;

TEST(AlignedBuffer, DefaultConstructedIsEmpty) {
    AlignedBuffer<float> buf;
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(AlignedBuffer, AllocatesAndZeroInitializes) {
    AlignedBuffer<int> buf(16);
    EXPECT_EQ(buf.size(), 16u);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_EQ(buf[i], 0);
    }
}

TEST(AlignedBuffer, IsAlignedTo64Bytes) {
    for (std::size_t n : {1, 3, 7, 100, 1024, 1 << 16}) {
        AlignedBuffer<float, 64> buf(n);
        auto addr = reinterpret_cast<std::uintptr_t>(buf.data());
        EXPECT_EQ(addr % 64, 0u) << "size=" << n;
    }
}

TEST(AlignedBuffer, MoveTransfersOwnership) {
    AlignedBuffer<int> a(8);
    a[0] = 42;
    AlignedBuffer<int> b(std::move(a));
    EXPECT_EQ(b[0], 42);
    EXPECT_EQ(a.data(), nullptr);  // NOLINT(bugprone-use-after-move) -- intentional check
    EXPECT_EQ(a.size(), 0u);
}

TEST(AlignedBuffer, SpanViewMatchesContents) {
    AlignedBuffer<int> buf(4);
    buf[0] = 1;
    buf[1] = 2;
    buf[2] = 3;
    buf[3] = 4;
    auto sp = buf.span();
    ASSERT_EQ(sp.size(), 4u);
    EXPECT_EQ(sp[2], 3);
}
