// Compiled with -mavxvnni (see CMakeLists.txt). NOTE: this kernel cannot be
// runtime-tested on the primary dev machine -- this WSL2 VM does not expose
// AVX-VNNI even though the underlying Alder Lake CPU supports it (confirmed
// via /proc/cpuinfo during Phase 1 setup). It compiles and is covered by the
// property test's *scalar-equivalence* check only where __builtin_cpu_supports
// reports it; MUST be verified on real AVX-VNNI hardware (a cloud box, per
// D-ARCH-13) before any benchmark chart trusts it.
//
// vpdpbusd requires one unsigned-u8 and one signed-i8 operand. Our public
// dot_i8 contract (kernels.h) takes two SIGNED int8 arrays to match the
// scalar reference, so `a` is offset to unsigned (a+128) and the correction
// term 128 * sum(b) is subtracted back out: (a-128)*b = a*b - 128*b, so
// sum((a[i]+128)*b[i]) - 128*sum(b[i]) == sum(a[i]*b[i]) for the original
// signed a.
#include <immintrin.h>

#include "lattice/kernels/kernels.h"

namespace lattice::kernels::avxvnni {

std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::uint32_t dim) noexcept {
    __m256i acc = _mm256_setzero_si256();
    std::int32_t b_sum = 0;
    std::uint32_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m128i a_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i a_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
        __m256i a_signed = _mm256_set_m128i(a_hi, a_lo);
        // Offset signed a to unsigned: a_u8 = a_i8 + 128 (as u8 wraparound == bit pattern XOR 0x80).
        __m256i a_u8 = _mm256_xor_si256(a_signed, _mm256_set1_epi8(static_cast<char>(0x80)));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        acc = _mm256_dpbusd_epi32(acc, a_u8, vb);

        alignas(32) std::int8_t b_chunk[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_chunk), vb);
        for (int k = 0; k < 32; ++k) b_sum += b_chunk[k];
    }
    alignas(32) std::int32_t acc_arr[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc_arr), acc);
    std::int32_t sum = 0;
    for (int k = 0; k < 8; ++k) sum += acc_arr[k];
    sum -= 128 * b_sum;

    for (; i < dim; ++i) sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return sum;
}

}  // namespace lattice::kernels::avxvnni
