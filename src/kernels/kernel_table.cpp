#include "lattice/kernels/kernels.h"

namespace lattice::kernels {

const KernelTable& KernelTable::resolve() noexcept {
    static const KernelTable table = [] {
        KernelTable t{};
        // Conservative default: always correct, always available.
        t.selected_isa = Isa::kScalar;
        t.dot_ip = &scalar::dot_ip;
        t.dot_l2 = &scalar::dot_l2;
        t.dot_i8 = &scalar::dot_i8;
        t.hamming = &scalar::hamming;

        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            t.selected_isa = Isa::kAvx2Fma;
            t.dot_ip = &avx2::dot_ip;
            t.dot_l2 = &avx2::dot_l2;
        }
        if (__builtin_cpu_supports("avxvnni")) {
            t.selected_isa = Isa::kAvxVnni;
            t.dot_i8 = &avxvnni::dot_i8;
        }
        if (__builtin_cpu_supports("avx512f")) {
            t.selected_isa = Isa::kAvx512;
            t.dot_ip = &avx512::dot_ip;
        }
        return t;
    }();
    return table;
}

void batched_dot_ip(std::span<const float* const> queries,
                     std::span<const float* const> candidates, std::uint32_t dim,
                     std::span<float> out) noexcept {
    const auto& table = KernelTable::resolve();
    for (std::size_t i = 0; i < queries.size(); ++i) {
        for (std::size_t j = 0; j < candidates.size(); ++j) {
            out[i * candidates.size() + j] = table.dot_ip(queries[i], candidates[j], dim);
        }
    }
}

}  // namespace lattice::kernels
