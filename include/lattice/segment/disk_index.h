#pragma once
// Disk-resident Vamana index access (D-STO-04): open an index file written by
// VamanaIndex::serialize_to_disk and serve its node records zero-copy through mmap (default) or
// via io_uring reads. Split out of `segment` so it depends only on index_graph, quant and
// storage -- no RocksDB/CRoaring -- which is what the Python package links.
#include <atomic>
#include <cstdint>
#include <span>
#include <string>

#include "lattice/common/error.h"
#include "lattice/common/types.h"
#include "lattice/index_graph/index_graph.h"
#include "lattice/storage/storage.h"

namespace lattice::segment {

// --- Disk-resident Vamana backend (D-STO-04, R2/finding 15/26) -----------
// The io_uring-backed NodeStore implementation lives here, not in
// index_graph, because index_graph must never depend on `storage`
// (finding 26) -- `segment` is what depends on both and bridges them.
class DiskNodeStore final : public index_graph::NodeStore {
public:
    // `data_offset` is where node 0's record starts in the file (i.e. the
    // header size) -- passed explicitly by open_disk_resident_vamana(),
    // which is the single place that knows the on-disk header layout
    // (index_graph.cpp's serialize_to_disk), rather than duplicating that
    // constant here.
    DiskNodeStore(storage::AsyncNodeFile file, index_graph::NodeRecordLayout layout,
                  std::size_t count, std::uint64_t data_offset)
        : file_(std::move(file)), layout_(layout), count_(count), data_offset_(data_offset) {}

    std::span<const std::byte> read_node(VectorId id) const override;
    std::size_t node_count() const noexcept override { return count_; }
    std::uint64_t read_errors() const noexcept override { return read_errors_.load(std::memory_order_relaxed); }
    // Real async prefetch (submit without waiting) is a Phase 5 perf
    // follow-up alongside AsyncNodeFile's batching (see storage.h) -- no-op
    // here for now, which is always a valid (if unhelpful) NodeStore::prefetch.

private:
    mutable storage::AsyncNodeFile file_;  // read_at() is non-const on AsyncNodeFile
    index_graph::NodeRecordLayout layout_;
    std::size_t count_;
    std::uint64_t data_offset_;
    mutable std::atomic<std::uint64_t> read_errors_{0};
};


// Zero-copy NodeStore over a read-only mmap of the index file: read_node() returns a span
// straight into the mapping (no syscall, no copy), so the OS page cache provides the
// "RAM tier" automatically -- more free RAM means more of the file stays resident, and a
// cold start just page-faults nodes in. Pair with a compact record layout (see
// NodeRecordLayout::make_compact) so the whole 1M x 128-d graph is ~0.83GB instead of 4.2GB.
class MmapNodeStore final : public index_graph::NodeStore {
public:
    MmapNodeStore(storage::MappedFile file, index_graph::NodeRecordLayout layout, std::size_t count,
                  std::uint64_t data_offset)
        : file_(std::move(file)), layout_(layout), count_(count), data_offset_(data_offset) {}

    std::span<const std::byte> read_node(VectorId id) const override;
    std::size_t node_count() const noexcept override { return count_; }
    void prefetch(VectorId id) const noexcept override;
    std::uint64_t read_errors() const noexcept override { return read_errors_.load(std::memory_order_relaxed); }

private:
    storage::MappedFile file_;
    index_graph::NodeRecordLayout layout_;
    std::size_t count_;
    std::uint64_t data_offset_;
    mutable std::atomic<std::uint64_t> read_errors_{0};
};

enum class NodeAccess {
    kMmap,    // default: zero-copy through the page cache
    kIoUring  // explicit per-node reads via io_uring (copy + syscall per node)
};
// Opens a graph previously written by VamanaIndex::serialize_to_disk,
// attaching a DiskNodeStore over it (D-STO-04's disk-resident backend).
Result<index_graph::VamanaIndex> open_disk_resident_vamana(const std::string& path,
                                                           NodeAccess access = NodeAccess::kMmap,
                                                           bool preload = false);


}  // namespace lattice::segment
