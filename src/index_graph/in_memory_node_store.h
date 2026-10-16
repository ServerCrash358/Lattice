#pragma once
// Phase 2's concrete NodeStore: fully resident array of sector-aligned node
// records (D-VEC-05's "in-RAM dev backend uses the identical node-record
// format, just fully resident"). Owns mutable access for the build phase;
// exposed read-only via the NodeStore interface for search.
#include <algorithm>
#include <vector>

#include "lattice/index_graph/index_graph.h"
#include "src/index_graph/node_record.h"

namespace lattice::index_graph::detail {

class InMemoryNodeStore final : public NodeStore {
public:
    InMemoryNodeStore(NodeRecordLayout layout, std::size_t count)
        : layout_(layout), count_(count), storage_(count * layout.record_bytes()) {}

    std::span<const std::byte> read_node(VectorId id) const override {
        auto idx = static_cast<std::size_t>(id);
        return {storage_.data() + idx * layout_.record_bytes(), layout_.record_bytes()};
    }

    std::size_t node_count() const noexcept override { return count_; }

    void prefetch(VectorId id) const noexcept override {
        auto idx = static_cast<std::size_t>(id);
        __builtin_prefetch(storage_.data() + idx * layout_.record_bytes(), 0, 1);
    }

    // Build-time only: write the initial vector + neighbour list for a node.
    void write_record(VectorId id, const float* vector, std::span<const VectorId> neighbours) {
        auto idx = static_cast<std::size_t>(id);
        std::span<std::byte> slot(storage_.data() + idx * layout_.record_bytes(),
                                   layout_.record_bytes());
        std::fill(slot.begin(), slot.end(), std::byte{0});
        encode_record(slot, layout_, vector, neighbours);
    }

    // Build-time only: rewrite just the neighbour list (RobustPrune updates).
    void set_neighbours(VectorId id, std::span<const VectorId> neighbours) {
        auto idx = static_cast<std::size_t>(id);
        std::byte* rec = storage_.data() + idx * layout_.record_bytes();
        // Never write past the record's neighbour slots (build() also rejects r > degree_cap_r).
        std::uint32_t degree =
            static_cast<std::uint32_t>(std::min<std::size_t>(neighbours.size(), layout_.degree_cap_r));
        // Relaxed atomic stores: concurrent build threads read these lists without a lock
        // (see record_degree / copy_neighbours). A reader may see a mix of old and new ids, but
        // every slot always holds a valid id or kNoNeighbour.
        __atomic_store_n(reinterpret_cast<std::uint32_t*>(rec + degree_offset(layout_)), degree, __ATOMIC_RELAXED);
        auto* nbr_out = reinterpret_cast<std::uint32_t*>(rec + neighbours_offset(layout_));
        for (std::size_t i = 0; i < degree; ++i) {
            __atomic_store_n(nbr_out + i, static_cast<std::uint32_t>(neighbours[i]), __ATOMIC_RELAXED);
        }
        for (std::size_t i = degree; i < layout_.degree_cap_r; ++i) {
            __atomic_store_n(nbr_out + i, kNoNeighbour, __ATOMIC_RELAXED);
        }
    }

private:
    NodeRecordLayout layout_;
    std::size_t count_;
    std::vector<std::byte> storage_;
};

}  // namespace lattice::index_graph::detail
