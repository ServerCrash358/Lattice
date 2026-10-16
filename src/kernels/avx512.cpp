// Compiled with -mavx512f (see CMakeLists.txt). The design doc assumed
// AVX-512 is fused off on the dev laptop, but the WSL2 VM's virtual CPU
// actually reports (and correctly executes) full AVX-512 including VNNI --
// confirmed both by passing the scalar-equivalence property test and by a
// throughput check (avx512 ~32ns/call vs avx2 ~60ns/call vs scalar
// ~466ns/call at dim=768; a trapped/emulated path would be far slower than
// AVX2, not faster, so this is genuine hardware execution). Still worth an
// independent cloud-box check before trusting a *peak-throughput* claim in
// a benchmark chart -- WSL2's virtualized CPUID reporting may not reflect
// every real deployment target -- but correctness testing is not blocked
// here the way D-ARCH-13 originally assumed.
#include <immintrin.h>

#include "lattice/kernels/kernels.h"

namespace lattice::kernels::avx512 {

float dot_ip(const float* a, const float* b, std::uint32_t dim) noexcept {
    __m512 acc = _mm512_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

}  // namespace lattice::kernels::avx512
