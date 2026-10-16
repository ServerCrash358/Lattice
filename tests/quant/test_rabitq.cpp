// Phase 3 exit gate item 8: RaBitQCodes::distance_to_query_bounded's error
// bound is empirically validated to actually upper-bound |estimate - exact|
// -- this must pass in Phase 3 (functional correctness), not deferred to
// Phase 5 (which only matures the adaptive-rerank policy built on top of it).
#include "lattice/quant/rabitq.h"

#include <gtest/gtest.h>

#include <random>
#include <vector>

using lattice::quant::RaBitQCodeBook;
using lattice::quant::RaBitQCodes;

namespace {
std::vector<float> random_vec(std::mt19937& rng, std::uint32_t dim, float scale = 1.0f) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}
}  // namespace

TEST(RaBitQ, TrainRejectsZeroDim) {
    auto r = RaBitQCodeBook::train(0, 1);
    EXPECT_FALSE(r.has_value());
}

TEST(RaBitQ, RotationIsApproximatelyOrthogonal) {
    // P^T P should be close to identity: rows are unit-norm and mutually
    // near-orthogonal (Gram-Schmidt's whole point).
    auto book = RaBitQCodeBook::train(64, 7);
    ASSERT_TRUE(book.has_value());
    std::uint32_t dim = 64;
    for (std::uint32_t i = 0; i < dim; ++i) {
        float self_dot = 0.0f;
        for (std::uint32_t j = 0; j < dim; ++j) {
            self_dot += book->rotation[i * dim + j] * book->rotation[i * dim + j];
        }
        EXPECT_NEAR(self_dot, 1.0f, 1e-3f) << "row " << i << " not unit norm";
    }
    for (std::uint32_t i = 0; i < dim; i += 7) {
        for (std::uint32_t k = i + 1; k < dim; k += 11) {
            float cross_dot = 0.0f;
            for (std::uint32_t j = 0; j < dim; ++j) {
                cross_dot += book->rotation[i * dim + j] * book->rotation[k * dim + j];
            }
            EXPECT_NEAR(cross_dot, 0.0f, 1e-2f) << "rows " << i << "," << k << " not orthogonal";
        }
    }
}

TEST(RaBitQ, CodeSizeMatchesExpectedFormula) {
    // At dim=768: ceil(768/8) + 8 = 96 + 8 = 104 bytes -- matches the
    // design doc's "~100 bytes/vector" figure closely.
    EXPECT_EQ(RaBitQCodes::code_bytes(768), 104u);
    EXPECT_EQ(RaBitQCodes::code_bytes(64), 8u + 8u);
}

// THE critical property test: the error bound must never be violated.
TEST(RaBitQ, ErrorBoundAlwaysHoldsAcrossLargeRandomSample) {
    std::mt19937 rng(2024);
    constexpr std::uint32_t dim = 768;  // D-VEC-07's frozen embedding dim
    constexpr int kNumVectors = 300;
    constexpr int kNumQueries = 50;

    auto book = RaBitQCodeBook::train(dim, 42);
    ASSERT_TRUE(book.has_value());
    RaBitQCodes codes(*book);

    std::vector<std::vector<float>> database;
    std::vector<std::vector<std::byte>> encoded;
    for (int i = 0; i < kNumVectors; ++i) {
        auto v = random_vec(rng, dim);
        encoded.push_back(codes.encode(v.data(), dim));
        database.push_back(std::move(v));
    }

    int violations = 0;
    int total_checks = 0;
    double max_slack_ratio = 0.0;  // how much headroom the bound has, for reporting

    for (int q = 0; q < kNumQueries; ++q) {
        auto query = random_vec(rng, dim);
        for (int i = 0; i < kNumVectors; ++i) {
            float exact = 0.0f;
            for (std::uint32_t d = 0; d < dim; ++d) exact += query[d] * database[i][d];

            auto bd = codes.distance_to_query_bounded(query.data(), dim, encoded[i]);
            float actual_error = std::abs(bd.estimate - exact);

            ++total_checks;
            // Tiny floating-point slack for the accumulated rotation/rounding
            // error inherent to fp32 arithmetic, not a weakening of the bound.
            if (actual_error > bd.error_bound * 1.001f + 1e-4f) {
                ++violations;
            }
            if (bd.error_bound > 1e-6f) {
                max_slack_ratio = std::max(max_slack_ratio,
                                            static_cast<double>(actual_error / bd.error_bound));
            }
        }
    }

    EXPECT_EQ(violations, 0) << violations << "/" << total_checks
                              << " error-bound violations found";
    // Report how tight the bound is (informational -- Phase 5 tightens this).
    RecordProperty("max_observed_error_to_bound_ratio", std::to_string(max_slack_ratio));
}

TEST(RaBitQ, FastPathMatchesBoundedEstimateExactly) {
    // rotate_query()/distance_to_query_fast() split distance_to_query_
    // bounded's O(dim^2)-per-call rotation into a one-time-per-query
    // O(dim^2) step reused across candidates -- must produce the exact
    // same estimate, not an approximation of it.
    std::mt19937 rng(777);
    constexpr std::uint32_t dim = 256;
    auto book = RaBitQCodeBook::train(dim, 11);
    ASSERT_TRUE(book.has_value());
    RaBitQCodes codes(*book);

    auto query = random_vec(rng, dim);
    auto rotated = codes.rotate_query(query.data(), dim);
    ASSERT_EQ(rotated.size(), dim);

    for (int i = 0; i < 50; ++i) {
        auto v = random_vec(rng, dim);
        auto code = codes.encode(v.data(), dim);
        auto bounded = codes.distance_to_query_bounded(query.data(), dim, code);
        float fast = codes.distance_to_query_fast(rotated.data(), dim, code);
        EXPECT_NEAR(fast, bounded.estimate, 1e-4f) << "fast path diverged from bounded estimate at i=" << i;
    }
}

TEST(RaBitQ, EstimateApproximatesExactDotProductReasonably) {
    std::mt19937 rng(55);
    constexpr std::uint32_t dim = 768;
    auto book = RaBitQCodeBook::train(dim, 99);
    ASSERT_TRUE(book.has_value());
    RaBitQCodes codes(*book);

    auto query = random_vec(rng, dim);
    double sum_abs_err = 0.0;
    int n = 0;
    for (int i = 0; i < 200; ++i) {
        auto v = random_vec(rng, dim);
        auto code = codes.encode(v.data(), dim);
        auto bd = codes.distance_to_query_bounded(query.data(), dim, code);

        float exact = 0.0f;
        for (std::uint32_t d = 0; d < dim; ++d) exact += query[d] * v[d];

        sum_abs_err += std::abs(bd.estimate - exact);
        ++n;
    }
    double mean_abs_err = sum_abs_err / n;
    // Loose sanity check: mean absolute error should be well within the
    // typical magnitude of dot products at this scale, not wildly off.
    EXPECT_LT(mean_abs_err, dim * 0.1) << "mean_abs_err=" << mean_abs_err;
}
