#pragma once
// kernels: SIMD distance functions, ISA dispatch, scalar references
// (D-VEC-01, D-VEC-04). Pure numeric kernels -- no knowledge of vectors'
// meaning (rows, IDs, filters). Depends only on `common`.
#include <cstdint>
#include <span>

namespace lattice::kernels {

enum class Isa { kScalar, kAvx2Fma, kAvxVnni, kAvx512 };

using DotFn = float (*)(const float* a, const float* b, std::uint32_t dim) noexcept;
using L2Fn = float (*)(const float* a, const float* b, std::uint32_t dim) noexcept;
using DotI8Fn = std::int32_t (*)(const std::int8_t* a, const std::int8_t* b,
                                  std::uint32_t dim) noexcept;
using HammingFn = std::uint32_t (*)(const std::uint64_t* a, const std::uint64_t* b,
                                     std::uint32_t words) noexcept;

// Scalar references -- always compiled, used by property tests and as the
// Isa::kScalar table entry. Never the fastest path, always correct.
namespace scalar {
float dot_ip(const float* a, const float* b, std::uint32_t dim) noexcept;
float dot_l2(const float* a, const float* b, std::uint32_t dim) noexcept;
std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::uint32_t dim) noexcept;
std::uint32_t hamming(const std::uint64_t* a, const std::uint64_t* b, std::uint32_t words) noexcept;
}  // namespace scalar

// AVX2+FMA fp32 kernels. Only call when KernelTable::resolve().selected_isa
// >= kAvx2Fma -- calling on unsupported hardware is UB (illegal instruction).
namespace avx2 {
float dot_ip(const float* a, const float* b, std::uint32_t dim) noexcept;
float dot_l2(const float* a, const float* b, std::uint32_t dim) noexcept;
}  // namespace avx2

// AVX-VNNI int8 kernel (vpdpbusd-based). Requires AVX-VNNI at runtime.
namespace avxvnni {
std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::uint32_t dim) noexcept;
}  // namespace avxvnni

// AVX-512 fp32/VNNI kernels. Requires AVX-512F(+VNNI) at runtime; compiled
// unconditionally so the binary is portable, but only ever dispatched to on
// hardware that reports support (D-ARCH-13: validated on a cloud box, not
// assumed present on the dev laptop -- and per Phase 1 testing, not even
// AVX-VNNI is exposed inside this machine's WSL2 VM).
namespace avx512 {
float dot_ip(const float* a, const float* b, std::uint32_t dim) noexcept;
}  // namespace avx512

struct KernelTable {
    Isa selected_isa;
    DotFn dot_ip;
    L2Fn dot_l2;
    DotI8Fn dot_i8;
    HammingFn hamming;

    // CPUID-based dispatch, resolved once (function-local static, D-CPP-02).
    static const KernelTable& resolve() noexcept;
};

// Batched form (D-CON-05): queries x candidates, cache-efficient inner loop.
// out[i * candidates.size() + j] = dot_ip(queries[i], candidates[j], dim).
void batched_dot_ip(std::span<const float* const> queries,
                     std::span<const float* const> candidates, std::uint32_t dim,
                     std::span<float> out) noexcept;

}  // namespace lattice::kernels
