#pragma once
// index_flat: exact SIMD brute-force scan (D-IDX-01 third index, D-IDX-06
// small-segment path, D-CON-02 growing-segment search path). Depends only on
// `common` and `kernels` -- never on `quant` (this is the exactness
// baseline, always operates on raw fp32, never on quantized codes).
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "lattice/common/error.h"
#include "lattice/common/types.h"

namespace lattice::index_flat {

using AdmitFn = std::function<bool(VectorId)>;

// Non-owning view over a contiguous or strided vector array. Works equally
// over a sealed segment's resident array or the growing segment's
// append-only mirror -- `count` is a snapshot value the caller already
// resolved (e.g. a visible_count load), so this module performs no atomics
// of its own (D-CON-02's concurrency story lives entirely in `segment`).
struct FlatView {
    const float* base = nullptr;
    std::uint32_t dim = 0;
    std::size_t stride_elems = 0;  // usually == dim
    std::size_t count = 0;
};

Result<std::vector<VectorId>> search(const FlatView& view, const float* query, std::uint32_t k,
                                      const AdmitFn& admit = {});

// Batched form (D-CON-05): queries x candidates as a small matmul.
Result<void> search_batched(const FlatView& view, std::span<const float* const> queries,
                             std::uint32_t k, std::span<std::vector<VectorId>> out_results,
                             const AdmitFn& admit = {});

}  // namespace lattice::index_flat
