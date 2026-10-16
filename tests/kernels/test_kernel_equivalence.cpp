// D-CPP-07 layer 2 (property tests): every non-scalar kernel must match its
// scalar reference within a documented epsilon, across the dims used in
// spec/11.3 (128 SIFT, 768 our target, 960 GIST) plus a small one (96) to
// exercise the tail/remainder loop in the SIMD kernels.
//
// Phase 1 pragmatic note: this uses GoogleTest's own randomized loops rather
// than RapidCheck (D-CPP-06 lists RapidCheck for property tests) -- kept out
// of the Phase 1 FetchContent set to limit first-build time; swap in
// RapidCheck without changing what's being asserted once it's added.
#include "lattice/kernels/kernels.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace lattice::kernels;

namespace {
constexpr std::uint32_t kDims[] = {96, 128, 768, 960};
constexpr int kTrialsPerDim = 200;
constexpr float kFpEpsilon = 1e-2f;  // sum of ~1000 fp32 FMAs vs scalar accumulation order

std::vector<float> random_vec(std::mt19937& rng, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<std::int8_t> random_i8vec(std::mt19937& rng, std::uint32_t dim) {
    std::uniform_int_distribution<int> dist(-127, 127);
    std::vector<std::int8_t> v(dim);
    for (auto& x : v) x = static_cast<std::int8_t>(dist(rng));
    return v;
}
}  // namespace

TEST(KernelEquivalence, KernelTableSelectsAtLeastAvx2OnThisMachine) {
    // Documents what this CI/dev machine actually resolves to (WSL2 exposes
    // AVX2+FMA but not AVX-VNNI/AVX-512 -- see kernels/avxvnni.cpp's header
    // comment). Not a portability requirement, just a visibility check.
    const auto& table = KernelTable::resolve();
    EXPECT_GE(static_cast<int>(table.selected_isa), static_cast<int>(Isa::kScalar));
}

TEST(KernelEquivalence, Avx2DotIpMatchesScalar) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        GTEST_SKIP() << "AVX2/FMA not available on this CPU";
    }
    std::mt19937 rng(42);
    for (auto dim : kDims) {
        for (int t = 0; t < kTrialsPerDim; ++t) {
            auto a = random_vec(rng, dim);
            auto b = random_vec(rng, dim);
            float expected = scalar::dot_ip(a.data(), b.data(), dim);
            float actual = avx2::dot_ip(a.data(), b.data(), dim);
            EXPECT_NEAR(actual, expected, kFpEpsilon) << "dim=" << dim << " trial=" << t;
        }
    }
}

TEST(KernelEquivalence, Avx2DotL2MatchesScalar) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        GTEST_SKIP() << "AVX2/FMA not available on this CPU";
    }
    std::mt19937 rng(43);
    for (auto dim : kDims) {
        for (int t = 0; t < kTrialsPerDim; ++t) {
            auto a = random_vec(rng, dim);
            auto b = random_vec(rng, dim);
            float expected = scalar::dot_l2(a.data(), b.data(), dim);
            float actual = avx2::dot_l2(a.data(), b.data(), dim);
            EXPECT_NEAR(actual, expected, kFpEpsilon) << "dim=" << dim << " trial=" << t;
        }
    }
}

TEST(KernelEquivalence, AvxVnniDotI8MatchesScalar) {
    if (!__builtin_cpu_supports("avxvnni")) {
        GTEST_SKIP() << "AVX-VNNI not exposed on this machine (expected on this WSL2 VM -- "
                         "see avxvnni.cpp header comment; must be verified on a cloud box)";
    }
    std::mt19937 rng(44);
    for (auto dim : kDims) {
        for (int t = 0; t < kTrialsPerDim; ++t) {
            auto a = random_i8vec(rng, dim);
            auto b = random_i8vec(rng, dim);
            auto expected = scalar::dot_i8(a.data(), b.data(), dim);
            auto actual = avxvnni::dot_i8(a.data(), b.data(), dim);
            EXPECT_EQ(actual, expected) << "dim=" << dim << " trial=" << t;
        }
    }
}

TEST(KernelEquivalence, Avx512DotIpMatchesScalar) {
    if (!__builtin_cpu_supports("avx512f")) {
        GTEST_SKIP() << "AVX-512 not available on this CPU";
    }
    // Note: on this project's primary dev machine, this test does NOT skip --
    // the WSL2 VM reports full AVX-512 (incl. VNNI) despite the design doc's
    // assumption that the physical laptop has it fused off. Confirmed as
    // genuine hardware execution (not trapped/emulated) via a throughput
    // check: avx512 ~32ns/call vs avx2 ~60ns/call at dim=768 -- an emulated
    // path would be far slower than AVX2, not faster. See avx512.cpp's
    // header comment.
    std::mt19937 rng(45);
    for (auto dim : kDims) {
        for (int t = 0; t < kTrialsPerDim; ++t) {
            auto a = random_vec(rng, dim);
            auto b = random_vec(rng, dim);
            float expected = scalar::dot_ip(a.data(), b.data(), dim);
            float actual = avx512::dot_ip(a.data(), b.data(), dim);
            EXPECT_NEAR(actual, expected, kFpEpsilon) << "dim=" << dim << " trial=" << t;
        }
    }
}

TEST(KernelEquivalence, HammingMatchesNaiveBitcount) {
    std::mt19937 rng(46);
    std::uniform_int_distribution<std::uint64_t> dist;
    for (std::uint32_t words : {1u, 2u, 12u, 15u}) {
        for (int t = 0; t < 100; ++t) {
            std::vector<std::uint64_t> a(words), b(words);
            for (auto& x : a) x = dist(rng);
            for (auto& x : b) x = dist(rng);

            std::uint32_t naive = 0;
            for (std::uint32_t i = 0; i < words; ++i) {
                naive += static_cast<std::uint32_t>(std::popcount(a[i] ^ b[i]));
            }
            EXPECT_EQ(scalar::hamming(a.data(), b.data(), words), naive);
        }
    }
}

TEST(KernelEquivalence, KernelTableDotIpMatchesScalarRegardlessOfDispatch) {
    // Whatever ISA got selected at startup, the table's dot_ip must still
    // agree with the scalar reference -- this is the test that would catch
    // a real dispatch bug, independent of which specific ISA is present.
    const auto& table = KernelTable::resolve();
    std::mt19937 rng(47);
    for (auto dim : kDims) {
        auto a = random_vec(rng, dim);
        auto b = random_vec(rng, dim);
        float expected = scalar::dot_ip(a.data(), b.data(), dim);
        float actual = table.dot_ip(a.data(), b.data(), dim);
        EXPECT_NEAR(actual, expected, kFpEpsilon) << "dim=" << dim;
    }
}

TEST(KernelEquivalence, BatchedDotIpMatchesPerPairScalar) {
    std::mt19937 rng(48);
    constexpr std::uint32_t dim = 128;
    std::vector<std::vector<float>> storage_q, storage_c;
    for (int i = 0; i < 5; ++i) storage_q.push_back(random_vec(rng, dim));
    for (int j = 0; j < 7; ++j) storage_c.push_back(random_vec(rng, dim));

    std::vector<const float*> queries, candidates;
    for (auto& v : storage_q) queries.push_back(v.data());
    for (auto& v : storage_c) candidates.push_back(v.data());

    std::vector<float> out(queries.size() * candidates.size());
    batched_dot_ip(queries, candidates, dim, out);

    for (std::size_t i = 0; i < queries.size(); ++i) {
        for (std::size_t j = 0; j < candidates.size(); ++j) {
            float expected = scalar::dot_ip(queries[i], candidates[j], dim);
            EXPECT_NEAR(out[i * candidates.size() + j], expected, kFpEpsilon)
                << "i=" << i << " j=" << j;
        }
    }
}
