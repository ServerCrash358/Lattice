#pragma once
// Internal (not part of the public index_graph interface): byte-level
// packing for the co-located node record (D-VEC-05): [vector][degree u32]
// [R * neighbour VectorId (u32)], zero-padded to sector_size. Shared by the
// Phase 2 in-RAM store and (unchanged) Phase 3's disk-resident store.
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include "lattice/common/types.h"
#include "lattice/index_graph/index_graph.h"

namespace lattice::index_graph::detail {

constexpr std::uint32_t kNoNeighbour = 0xFFFFFFFFu;

inline std::size_t degree_offset(const NodeRecordLayout& layout) noexcept {
    return layout.vector_bytes();
}
inline std::size_t neighbours_offset(const NodeRecordLayout& layout) noexcept {
    return layout.vector_bytes() + sizeof(std::uint32_t);
}

// Writes one full record (vector + degree + neighbour list) into `out`,
// which must be exactly layout.record_bytes() long and already zeroed.
inline void encode_record(std::span<std::byte> out, const NodeRecordLayout& layout,
                           const float* vector, std::span<const VectorId> neighbours) {
    std::memcpy(out.data(), vector, layout.vector_bytes());
    // Never write past the record's neighbour slots (build() also rejects r > degree_cap_r).
    std::uint32_t degree =
        static_cast<std::uint32_t>(std::min<std::size_t>(neighbours.size(), layout.degree_cap_r));
    std::memcpy(out.data() + degree_offset(layout), &degree, sizeof(degree));
    auto* nbr_out = reinterpret_cast<std::uint32_t*>(out.data() + neighbours_offset(layout));
    for (std::size_t i = 0; i < degree; ++i) {
        nbr_out[i] = static_cast<std::uint32_t>(neighbours[i]);
    }
    for (std::size_t i = degree; i < layout.degree_cap_r; ++i) {
        nbr_out[i] = kNoNeighbour;
    }
}

inline const float* record_vector(std::span<const std::byte> record) noexcept {
    return reinterpret_cast<const float*>(record.data());
}

// Degree and neighbour slots are read with relaxed atomic loads: during a parallel build another
// thread may be rewriting a node's list (InMemoryNodeStore::set_neighbours stores atomically),
// and plain concurrent reads would be a data race. Relaxed loads compile to plain loads on x86.
// Record fields are 4-byte aligned (vector bytes and record strides are multiples of 4).
inline std::uint32_t record_degree(std::span<const std::byte> record,
                                    const NodeRecordLayout& layout) noexcept {
    return __atomic_load_n(reinterpret_cast<const std::uint32_t*>(record.data() + degree_offset(layout)),
                           __ATOMIC_RELAXED);
}

inline void copy_neighbours(std::span<const std::byte> record, const NodeRecordLayout& layout,
                            std::uint32_t degree, std::vector<std::uint32_t>& out) {
    const auto* nbr = reinterpret_cast<const std::uint32_t*>(record.data() + neighbours_offset(layout));
    out.resize(degree);
    for (std::uint32_t i = 0; i < degree; ++i) out[i] = __atomic_load_n(nbr + i, __ATOMIC_RELAXED);
}

inline std::span<const std::uint32_t> record_neighbours(std::span<const std::byte> record,
                                                          const NodeRecordLayout& layout) noexcept {
    auto* nbr = reinterpret_cast<const std::uint32_t*>(record.data() + neighbours_offset(layout));
    return {nbr, layout.degree_cap_r};
}

}  // namespace lattice::index_graph::detail
