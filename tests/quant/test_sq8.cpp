// D-CPP-07 layer 2 (property): quantize->dequantize error is within a
// measured bound. D-VEC-02's Phase 1 recall gate ("recall@10 >= 0.99 with
// SQ8 + fp32 rerank") is exercised at the index level in Phase 2+ once an
// index exists to rerank through; this file covers the quantizer itself.
#include "lattice/quant/sq8.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using lattice::quant::Sq8CodeBook;
using lattice::quant::Sq8Codes;

namespace {
constexpr std::uint32_t kDim = 768;  // D-VEC-07's frozen embedding dimension

std::vector<std::vector<float>> make_sample(std::mt19937& rng, std::size_t n, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> sample(n, std::vector<float>(dim));
    for (auto& v : sample) {
        for (auto& x : v) x = dist(rng);
    }
    return sample;
}
}  // namespace

TEST(Sq8, TrainRejectsEmptySample) {
    std::vector<const float*> empty;
    auto r = Sq8CodeBook::train(empty, kDim);
    ASSERT_FALSE(r.has_value());
}

TEST(Sq8, TrainRejectsZeroDim) {
    std::mt19937 rng(1);
    auto sample = make_sample(rng, 10, kDim);
    std::vector<const float*> ptrs;
    for (auto& v : sample) ptrs.push_back(v.data());
    auto r = Sq8CodeBook::train(ptrs, 0);
    ASSERT_FALSE(r.has_value());
}

TEST(Sq8, EncodeDecodeRoundTripErrorIsBounded) {
    std::mt19937 rng(2);
    auto sample = make_sample(rng, 500, kDim);
    std::vector<const float*> ptrs;
    for (auto& v : sample) ptrs.push_back(v.data());

    auto book_r = Sq8CodeBook::train(ptrs, kDim);
    ASSERT_TRUE(book_r.has_value());
    Sq8Codes codes(*book_r);

    // 8-bit quantization over roughly [-1, 1] gives a per-dimension step of
    // about 2/255 ~= 0.0078; allow generous slack for rounding.
    constexpr float kMaxAbsError = 0.02f;

    for (auto& v : sample) {
        auto code = codes.encode(v.data(), kDim);
        std::vector<float> decoded(kDim);
        codes.decode(code, kDim, decoded.data());
        for (std::uint32_t d = 0; d < kDim; ++d) {
            EXPECT_LE(std::abs(decoded[d] - v[d]), kMaxAbsError)
                << "dim=" << d << " orig=" << v[d] << " decoded=" << decoded[d];
        }
    }
}

TEST(Sq8, DistanceToQueryApproximatesExactDotProduct) {
    std::mt19937 rng(3);
    auto sample = make_sample(rng, 500, kDim);
    std::vector<const float*> ptrs;
    for (auto& v : sample) ptrs.push_back(v.data());
    auto book_r = Sq8CodeBook::train(ptrs, kDim);
    ASSERT_TRUE(book_r.has_value());
    Sq8Codes codes(*book_r);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> query(kDim);
    for (auto& x : query) x = dist(rng);

    double sum_abs_rel_err = 0.0;
    int n = 0;
    for (auto& v : sample) {
        auto code = codes.encode(v.data(), kDim);
        float approx = codes.distance_to_query(query.data(), kDim, code);

        float exact = 0.0f;
        for (std::uint32_t d = 0; d < kDim; ++d) exact += query[d] * v[d];

        // Relative error should be small when the exact value isn't near zero.
        if (std::abs(exact) > 1.0f) {
            sum_abs_rel_err += std::abs(approx - exact) / std::abs(exact);
            ++n;
        }
    }
    ASSERT_GT(n, 0);
    double mean_rel_err = sum_abs_rel_err / n;
    EXPECT_LT(mean_rel_err, 0.05) << "mean relative error too high: " << mean_rel_err;
}

TEST(Sq8, ConstantDimensionDoesNotDivideByZero) {
    // A dimension with zero variance across the sample (range == 0) must not
    // produce NaN/Inf in scale.
    std::mt19937 rng(4);
    constexpr std::uint32_t dim = 4;
    std::vector<std::vector<float>> sample = {
        {1.0f, 5.0f, -1.0f, 0.0f},
        {1.0f, 5.0f, 1.0f, 0.5f},
        {1.0f, 5.0f, 0.0f, -0.5f},
    };
    std::vector<const float*> ptrs;
    for (auto& v : sample) ptrs.push_back(v.data());
    auto book_r = Sq8CodeBook::train(ptrs, dim);
    ASSERT_TRUE(book_r.has_value());
    for (float s : book_r->per_dim_scale) {
        EXPECT_TRUE(std::isfinite(s));
        EXPECT_GT(s, 0.0f);
    }

    Sq8Codes codes(*book_r);
    auto code = codes.encode(sample[0].data(), dim);
    std::vector<float> decoded(dim);
    codes.decode(code, dim, decoded.data());
    for (float x : decoded) EXPECT_TRUE(std::isfinite(x));
}
