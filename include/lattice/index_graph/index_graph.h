#pragma once
// index_graph: Vamana graph index (D-IDX-01, D-IDX-02). One implementation,
// two storage backends sharing this exact node-record format (D-VEC-05):
// InMemoryNodeStore (Phase 2, build-time and small/dev-loop scale) and a
// disk-resident NodeStore built on io_uring (Phase 3, module-segment.md's
// DiskNodeStore -- lives in `segment`, not here, per finding 26: this
// module never depends on `storage`). Neither build() nor search()/
// greedy_search() changed to add the disk backend -- only open() and
// serialize_to_disk() are new, and NodeStore's abstract interface already
// supported this from Phase 2.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lattice/common/error.h"
#include "lattice/common/types.h"
#include "lattice/quant/sq8.h"

namespace lattice::index_graph {

// On-disk index file header, shared by serialize_to_disk (writer) and
// segment::open_disk_resident_vamana (reader). Explicit little-endian byte layout:
// magic u32, format_version u32, codes_kind u32, dim u32, elem_type u32, degree_cap_r u32,
// sector_size u32, node_count u64, entry_point u64, metric u32.
struct DiskFormat {
    static constexpr std::uint32_t kMagic = 0x56414D32;  // 'VAM2' ('VAM1' files predate versioning)
    static constexpr std::uint32_t kLegacyMagic = 0x56414D31;
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::uint32_t kCodesSq8 = 1;
    static constexpr std::size_t kHeaderBytes = 4 * 7 + 8 + 8 + 4;
};

using AdmitFn = std::function<bool(VectorId)>;

// Abstraction that lets the same index code run in-RAM (Phase 2) or over
// io_uring (Phase 3), without index_graph depending on `storage` (finding 26
// / module-index_graph.md).
class NodeStore {
public:
    virtual ~NodeStore() = default;
    virtual std::span<const std::byte> read_node(VectorId id) const = 0;
    virtual std::size_t node_count() const noexcept = 0;
    virtual void prefetch([[maybe_unused]] VectorId id) const noexcept {}
    // Reads that failed (I/O error or out-of-range id) and were served as a zeroed record.
    // read_node() has no error channel, so this counter is how such failures become visible.
    virtual std::uint64_t read_errors() const noexcept { return 0; }
};

enum class ElemType { kFloat32, kFloat16 };

struct NodeRecordLayout {
    std::uint32_t dim = 0;
    ElemType elem_type = ElemType::kFloat32;
    std::uint32_t degree_cap_r = 64;
    std::uint32_t sector_size = 4096;

    // D-VEC-05 / consistency-review finding 4: R_max is dim-dependent, never
    // a flat 255. Checked here at construction, not assumed by callers.
    static Result<NodeRecordLayout> make(std::uint32_t dim, ElemType t, std::uint32_t r,
                                          std::uint32_t sector_size = 4096) noexcept;

    // Record stride = exactly vector + degree + R neighbour ids, rounded up to a 64-byte cache
    // line (832B at 128-d/R=64, vs 4096B for a sector-padded record): 5x smaller file, far better
    // cache behaviour. Use with the mmap node store (segment::MmapNodeStore).
    static Result<NodeRecordLayout> make_compact(std::uint32_t dim, ElemType t, std::uint32_t r) noexcept {
        return make(dim, t, r, static_cast<std::uint32_t>(minimal_aligned_stride(dim, t, r)));
    }

    std::size_t vector_bytes() const noexcept {
        return static_cast<std::size_t>(dim) * (elem_type == ElemType::kFloat32 ? 4 : 2);
    }
    // vector bytes + degree(u32) + R * neighbour_id(u32), padded to sector_size.
    std::size_t record_bytes() const noexcept { return sector_size; }
    std::size_t max_r_for(std::uint32_t dim_, ElemType t) const noexcept;

    // `sector_size` is intentionally generic (a record stride, not literally
    // a disk sector): Phase 3's disk backend passes 4096 for real sector
    // alignment; Phase 2's in-RAM backend should NOT pay that padding cost
    // just to visit ~5x more cache lines per hop than necessary. Use this
    // helper to compute the minimal cache-line-aligned stride instead.
    static std::size_t minimal_aligned_stride(std::uint32_t dim, ElemType t, std::uint32_t r,
                                               std::size_t align = 64) noexcept {
        std::size_t vec_bytes = static_cast<std::size_t>(dim) * (t == ElemType::kFloat32 ? 4 : 2);
        std::size_t needed = vec_bytes + sizeof(std::uint32_t) + r * sizeof(std::uint32_t);
        return ((needed + align - 1) / align) * align;
    }
};

enum class Metric { kInnerProduct, kL2 };  // D-VEC-01: IP is primary, L2 for SIFT/GIST parity

struct VamanaBuildParams {
    std::uint32_t r = 64;               // default degree (D-VEC-05: tune 64-128)
    float alpha = 1.2f;                 // Vamana pruning parameter
    std::uint32_t build_list_size = 100;
    Metric metric = Metric::kInnerProduct;
    // Graph-build worker threads: 0 = hardware_concurrency, 1 = sequential and
    // fully deterministic (identical graph every run).
    std::uint32_t build_threads = 0;
    // Fill unused degree slots with the closest alpha-pruned candidates (DiskANN saturate_graph):
    // better connectivity on hard, high-dimensional data at slightly higher build cost.
    bool saturate_graph = false;  // opt-in: not yet validated on SIFT1M recall/QPS
    // D-CON-04 toggles, individually benchmarkable (Phase 2 exit gate item 3).
    bool enable_prefetch = true;
    bool enable_visited_generation_counter = true;
};

struct SearchParams {
    std::uint32_t ef_search = 64;
    std::uint32_t rerank_k = 40;  // D-VEC-03: default 4-10x k
    // Metric is NOT here: it's a property of the built graph (fixed by
    // VamanaBuildParams::metric), not a per-query knob -- you can't
    // reinterpret "closest" after the graph structure was built for one.
    bool enable_prefetch = true;
    bool enable_visited_generation_counter = true;
    // false: score every neighbour exactly (full node-record read) instead of from the in-RAM
    // SQ8 codes -- slower, but removes quantization noise from the traversal (for diagnosis).
    bool use_codes = true;
};

class VamanaIndex {
public:
    static Result<VamanaIndex> build(std::span<const float* const> vectors, std::uint32_t dim,
                                      const quant::Sq8Codes& codes,
                                      const VamanaBuildParams& params, NodeRecordLayout layout);

    // Attach to an already-built graph via any NodeStore implementation
    // (in-RAM or disk-resident) -- this is how the disk backend loads a
    // graph that was built in RAM and serialized (D-VEC-05: build happens
    // in RAM where random access is cheap; disk-residency is a
    // search-time-only property of the store it's attached to).
    // `codes_book`/`codes_blob` are the RAM-resident SQ8 codes this exact
    // graph was built with (see `build()`) -- required (not optional) so
    // that a reopened disk-resident index has the same codes-gated
    // traversal a freshly-built in-RAM index does (finding: search() used
    // to always read a full node record from disk just to score each
    // neighbour it considered, not just the ones it visited -- see
    // greedy_search's use of `codes_` below).
    static Result<VamanaIndex> open(std::unique_ptr<NodeStore> store, NodeRecordLayout layout,
                                     std::uint32_t dim, std::size_t node_count,
                                     VectorId entry_point, Metric metric,
                                     quant::Sq8CodeBook codes_book, std::vector<std::byte> codes_blob);

    // Writes the index file: a header (see DiskFormat), every node record, then the SQ8 codebook
    // and codes blob (sizes follow from dim/node_count). Crash-safe: written to `path` + ".tmp",
    // fsynced, then renamed over `path` (and the directory fsynced), so `path` is either the old
    // file or the complete new one, never a torn mix.
    Result<void> serialize_to_disk(const std::string& path) const;

    struct Stats {
        std::uint64_t candidates_visited = 0;
        std::uint64_t distance_computations = 0;
    };

    // `out_stats`, if non-null, is written with this call's stats. NOT a
    // mutable member on the index -- concurrent search() calls on the same
    // built (read-only) index must be race-free (D-CON-02's lock-free-reads
    // requirement), which an earlier draft violated by writing to a shared
    // `mutable Stats` member on every call.
    Result<std::vector<VectorId>> search(const float* query, std::uint32_t k,
                                          const SearchParams& params, AdmitFn admit = {},
                                          Stats* out_stats = nullptr) const;

    // Filter-aware search (D-QRY-01 planner, in-index half). The caller supplies
    // the exact matching set as sorted VectorIds plus an admit callback:
    //  - matching set <= `scan_max`: skip the graph entirely, score ONLY the
    //    matching vectors from the RAM-resident SQ8 codes, then rerank the best
    //    candidates exactly from their node records (recall ~1.0 regardless of
    //    how the filter correlates with the vectors).
    //  - otherwise: filtered graph traversal with the beam widened by
    //    1/selectivity (capped by `max_ef`) so enough matching nodes are reached.
    struct FilteredParams {
        SearchParams base;
        std::span<const VectorId> matching_ids;  // sorted ascending, exact
        std::uint32_t scan_max = 150000;  // ~95ns/vector scan: <=150K matches => ~14ms, recall ~1.0 even for clustered filters
        std::uint32_t max_ef = 16384;
        std::uint32_t scan_rerank = 100;  // exact-rerank pool size for the scan path
    };
    Result<std::vector<VectorId>> search_filtered(const float* query, std::uint32_t k,
                                                   const FilteredParams& params, AdmitFn admit,
                                                   Stats* out_stats = nullptr) const;

    std::size_t size() const noexcept { return node_count_; }
    // The stored full-precision vector of node `id` (dim floats). For disk-backed stores the span
    // is only valid until the next read on the same thread, so copy it out before searching again.
    std::span<const float> vector_of(VectorId id) const;
    // Node reads that failed and were served as zeroed records (see NodeStore::read_errors).
    // Non-zero means search results may be degraded; check it in health monitoring.
    std::uint64_t node_read_errors() const noexcept { return store_ ? store_->read_errors() : 0; }
    NodeRecordLayout layout() const noexcept { return layout_; }

    VamanaIndex(VamanaIndex&&) noexcept = default;
    VamanaIndex& operator=(VamanaIndex&&) noexcept = default;
    VamanaIndex(const VamanaIndex&) = delete;
    VamanaIndex& operator=(const VamanaIndex&) = delete;

private:
    VamanaIndex() = default;

    std::unique_ptr<NodeStore> store_;
    // Owned (not a caller-held reference, unlike the pre-fix design): must
    // survive a disk round-trip (build -> serialize -> reopen in a fresh
    // process/call), so VamanaIndex holds its own copy rather than pointing
    // at a codebook some caller might not keep alive.
    quant::Sq8CodeBook codes_book_;
    std::vector<std::byte> codes_blob_;  // contiguous SQ8 codes, one per VectorId (dim bytes each)
    std::uint32_t dim_ = 0;
    std::size_t node_count_ = 0;
    NodeRecordLayout layout_;
    Metric metric_ = Metric::kInnerProduct;
    VectorId entry_point_{};

    friend class VamanaBuilder;
};

}  // namespace lattice::index_graph
