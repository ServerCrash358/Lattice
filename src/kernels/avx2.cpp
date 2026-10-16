// Compiled with -mavx2 -mfma (see CMakeLists.txt source-file properties).
// Only ever called when KernelTable::resolve() selected kAvx2Fma or higher.
#include <immintrin.h>

#include "lattice/kernels/kernels.h"

namespace lattice::kernels::avx2 {

namespace {
inline float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
}  // namespace

float dot_ip(const float* a, const float* b, std::uint32_t dim) noexcept {
    __m256 acc = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

float dot_l2(const float* a, const float* b, std::uint32_t dim) noexcept {
    __m256 acc = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

}  // namespace lattice::kernels::avx2
