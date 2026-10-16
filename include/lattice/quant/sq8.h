#pragma once
// quant: SQ8 scalar quantization (D-VEC-02b, D-VEC-08: per-dimension,
// schema-scoped). Depends on `common` and `kernels`. Never depends on any
// index module. PQ/RaBitQ are Phase 3/5 scope (R5) -- not declared here.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lattice/common/error.h"

namespace lattice::quant {

struct Sq8CodeBook {
    std::vector<float> per_dim_scale;   // size == dim
    std::vector<float> per_dim_offset;  // size == dim

    // Trains scale/offset per dimension from a representative sample
    // (D-VEC-08). Maps [min, max] per dimension to the full uint8 range.
    static Result<Sq8CodeBook> train(std::span<const float* const> sample, std::uint32_t dim);
};

class Sq8Codes {
public:
    explicit Sq8Codes(const Sq8CodeBook& book) noexcept : book_(&book) {}

    // code.size() == dim, one byte per dimension.
    std::vector<std::byte> encode(const float* v, std::uint32_t dim) const noexcept;

    // Exact (within quantization error) inner-product-style distance between
    // a raw float32 query and an encoded SQ8 code, without fully
    // dequantizing the code into a temporary float buffer.
    float distance_to_query(const float* query, std::uint32_t dim,
                             std::span<const std::byte> code) const noexcept;

    // Negated squared L2 distance to a raw float32 query (higher is better,
    // matching index_graph's "higher is better" score convention for L2), so
    // L2-metric collections can use codes-gated traversal too.
    float neg_l2_to_query(const float* query, std::uint32_t dim,
                           std::span<const std::byte> code) const noexcept;

    // Convenience: reconstruct the approximate float32 vector from a code
    // (used by tests and by rerank paths that want a dequantized vector).
    void decode(std::span<const std::byte> code, std::uint32_t dim, float* out) const noexcept;

    // Exposes the underlying codebook so a caller can copy/persist it (e.g.
    // VamanaIndex::build owning its own copy for later codes-gated search,
    // including after a disk round-trip -- see index_graph.h).
    const Sq8CodeBook& book() const noexcept { return *book_; }

private:
    const Sq8CodeBook* book_;
};

}  // namespace lattice::quant
