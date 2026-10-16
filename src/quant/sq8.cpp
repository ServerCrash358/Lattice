#include "lattice/quant/sq8.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>

#include <limits>

namespace lattice::quant {

Result<Sq8CodeBook> Sq8CodeBook::train(std::span<const float* const> sample, std::uint32_t dim) {
    if (sample.empty()) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument,
                                            "Sq8CodeBook::train requires a non-empty sample"));
    }
    if (dim == 0) {
        return std::unexpected(
            Error::make(ErrorCode::kInvalidArgument, "Sq8CodeBook::train requires dim > 0"));
    }

    std::vector<float> min_v(dim, std::numeric_limits<float>::max());
    std::vector<float> max_v(dim, std::numeric_limits<float>::lowest());
    for (const float* v : sample) {
        for (std::uint32_t d = 0; d < dim; ++d) {
            if (!std::isfinite(v[d])) {
                return std::unexpected(Error::make(ErrorCode::kInvalidArgument,
                                                   "Sq8CodeBook::train: sample contains NaN/Inf"));
            }
            min_v[d] = std::min(min_v[d], v[d]);
            max_v[d] = std::max(max_v[d], v[d]);
        }
    }

    Sq8CodeBook book;
    book.per_dim_scale.resize(dim);
    book.per_dim_offset.resize(dim);
    for (std::uint32_t d = 0; d < dim; ++d) {
        float range = max_v[d] - min_v[d];
        // Degenerate (constant) dimension: scale doesn't matter, avoid div-by-zero.
        book.per_dim_scale[d] = (range > 1e-12f) ? (range / 255.0f) : 1.0f;
        book.per_dim_offset[d] = min_v[d];
    }
    return book;
}

std::vector<std::byte> Sq8Codes::encode(const float* v, std::uint32_t dim) const noexcept {
    std::vector<std::byte> code(dim);
    for (std::uint32_t d = 0; d < dim; ++d) {
        float scaled = (v[d] - book_->per_dim_offset[d]) / book_->per_dim_scale[d];
        // Clamp in the float domain BEFORE converting: casting NaN or an out-of-range float to
        // int is undefined behaviour. (NaN compares false, so it maps to 0.)
        float clamped = scaled >= 0.0f ? (scaled <= 255.0f ? scaled : 255.0f) : 0.0f;
        int q = static_cast<int>(clamped + 0.5f);  // round to nearest
        if (q > 255) q = 255;
        code[d] = static_cast<std::byte>(q);
    }
    return code;
}

void Sq8Codes::decode(std::span<const std::byte> code, std::uint32_t dim, float* out) const noexcept {
    for (std::uint32_t d = 0; d < dim; ++d) {
        auto q = static_cast<std::uint8_t>(code[d]);
        out[d] = book_->per_dim_offset[d] + static_cast<float>(q) * book_->per_dim_scale[d];
    }
}

// Scorers are hot (called once per neighbour considered during traversal).
// Explicit AVX2+FMA kernels (8 dims per step: widen u8 -> f32, dequantize with
// one FMA, accumulate) chosen at runtime; a plain scalar path remains for
// CPUs without AVX2 and for the dim%8 tail.
namespace {

__attribute__((target("avx2,fma"))) inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma"))) float ip_avx2(const float* q, const float* off, const float* scl,
                                                   const std::uint8_t* c, std::uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    std::uint32_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + d))));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + d + 8))));
        __m256 v0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(scl + d), _mm256_loadu_ps(off + d));
        __m256 v1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(scl + d + 8), _mm256_loadu_ps(off + d + 8));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), v0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8), v1, acc1);
    }
    for (; d + 8 <= dim; d += 8) {
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + d))));
        __m256 v0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(scl + d), _mm256_loadu_ps(off + d));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), v0, acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; d < dim; ++d) sum += q[d] * (off[d] + static_cast<float>(c[d]) * scl[d]);
    return sum;
}

__attribute__((target("avx2,fma"))) float l2_avx2(const float* q, const float* off, const float* scl,
                                                   const std::uint8_t* c, std::uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    std::uint32_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + d))));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + d + 8))));
        __m256 v0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(scl + d), _mm256_loadu_ps(off + d));
        __m256 v1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(scl + d + 8), _mm256_loadu_ps(off + d + 8));
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + d), v0);
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(q + d + 8), v1);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; d + 8 <= dim; d += 8) {
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + d))));
        __m256 v0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(scl + d), _mm256_loadu_ps(off + d));
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + d), v0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; d < dim; ++d) {
        float diff = q[d] - (off[d] + static_cast<float>(c[d]) * scl[d]);
        sum += diff * diff;
    }
    return sum;
}

bool has_avx2_fma() {
    static const bool v = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return v;
}

}  // namespace

float Sq8Codes::distance_to_query(const float* query, std::uint32_t dim,
                                   std::span<const std::byte> code) const noexcept {
    const float* off = book_->per_dim_offset.data();
    const float* scl = book_->per_dim_scale.data();
    const auto* c = reinterpret_cast<const std::uint8_t*>(code.data());
    if (has_avx2_fma()) return ip_avx2(query, off, scl, c, dim);
    float sum = 0.0f;
    for (std::uint32_t d = 0; d < dim; ++d) sum += query[d] * (off[d] + static_cast<float>(c[d]) * scl[d]);
    return sum;
}

float Sq8Codes::neg_l2_to_query(const float* query, std::uint32_t dim,
                                 std::span<const std::byte> code) const noexcept {
    const float* off = book_->per_dim_offset.data();
    const float* scl = book_->per_dim_scale.data();
    const auto* c = reinterpret_cast<const std::uint8_t*>(code.data());
    if (has_avx2_fma()) return -l2_avx2(query, off, scl, c, dim);
    float sum = 0.0f;
    for (std::uint32_t d = 0; d < dim; ++d) {
        float diff = query[d] - (off[d] + static_cast<float>(c[d]) * scl[d]);
        sum += diff * diff;
    }
    return -sum;
}

}  // namespace lattice::quant
