#include "lattice/segment/disk_index.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "lattice/quant/sq8.h"

namespace lattice::segment {

// --- MmapNodeStore ------------------------------------------------------------

std::span<const std::byte> MmapNodeStore::read_node(VectorId id) const {
    auto idx = static_cast<std::size_t>(id);
    if (idx >= count_) {
        // Corrupt neighbour id: return a zeroed record (same contract as DiskNodeStore's I/O error).
        read_errors_.fetch_add(1, std::memory_order_relaxed);
        thread_local std::vector<std::byte> zeros;
        zeros.assign(layout_.record_bytes(), std::byte{0});
        return zeros;
    }
    return file_.data().subspan(data_offset_ + idx * layout_.record_bytes(), layout_.record_bytes());
}

void MmapNodeStore::prefetch(VectorId id) const noexcept {
    auto idx = static_cast<std::size_t>(id);
    if (idx < count_) __builtin_prefetch(file_.data().data() + data_offset_ + idx * layout_.record_bytes(), 0, 1);
}
// --- DiskNodeStore (D-STO-04, R2/finding 15/26) ----------------------------

std::span<const std::byte> DiskNodeStore::read_node(VectorId id) const {
    // thread_local, not a shared mutable member: safe for concurrent reads
    // from multiple threads (D-CON-02's lock-free-reads requirement) without
    // a lock, and safe for the single-buffer-reuse pattern within one
    // thread's traversal now that greedy_search copies out a node's
    // neighbour list before making any further read_node() calls (see
    // index_graph.cpp's greedy_search comment on this exact hazard).
    thread_local std::vector<std::byte> scratch;
    scratch.resize(layout_.record_bytes());

    auto idx = static_cast<std::size_t>(id);
    if (idx >= count_) {  // corrupt neighbour id: zeroed record, counted (same contract as below)
        std::fill(scratch.begin(), scratch.end(), std::byte{0});
        read_errors_.fetch_add(1, std::memory_order_relaxed);
        return scratch;
    }
    std::uint64_t offset = data_offset_ + idx * layout_.record_bytes();
    auto r = file_.read_at(offset, scratch);
    if (!r) {
        // NodeStore::read_node has no Result<> in its signature (D-MOD-01's
        // frozen interface) -- an I/O error here surfaces as a zeroed
        // record rather than a crash. Real production handling (retry,
        // propagate via a side-channel Stats field) is a Phase 5 hardening
        // item; Phase 3's job is proving the io_uring path works, which the
        // crash-injection harness (still outstanding) will stress further.
        std::fill(scratch.begin(), scratch.end(), std::byte{0});
        read_errors_.fetch_add(1, std::memory_order_relaxed);
    }
    return scratch;
}

Result<index_graph::VamanaIndex> open_disk_resident_vamana(const std::string& path, NodeAccess access,
                                                           bool preload) {
    using index_graph::DiskFormat;
    auto corrupt = [&](const std::string& why) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, why + ": " + path));
    };
    auto file_r = storage::PosixFile::open(path, false);
    if (!file_r) return std::unexpected(file_r.error());
    auto file_size = (*file_r)->size();
    if (!file_size) return std::unexpected(file_size.error());

    std::vector<std::byte> header(DiskFormat::kHeaderBytes);
    auto n = (*file_r)->pread(0, header);
    if (!n || *n != header.size()) return corrupt("truncated Vamana disk header");

    std::uint32_t magic, version, codes_kind, dim, elem_type_raw, degree_cap_r, sector_size, metric_raw;
    std::uint64_t node_count_u64, entry_point_u64;
    std::size_t off = 0;
    auto get32 = [&](std::uint32_t& v) { std::memcpy(&v, header.data() + off, 4); off += 4; };
    auto get64 = [&](std::uint64_t& v) { std::memcpy(&v, header.data() + off, 8); off += 8; };
    get32(magic);
    if (magic == DiskFormat::kLegacyMagic) {
        return corrupt("index file uses the pre-versioning 'VAM1' format; rebuild it with this version");
    }
    if (magic != DiskFormat::kMagic) return corrupt("bad Vamana disk file magic");
    get32(version);
    get32(codes_kind);
    get32(dim);
    get32(elem_type_raw);
    get32(degree_cap_r);
    get32(sector_size);
    get64(node_count_u64);
    get64(entry_point_u64);
    get32(metric_raw);

    if (version != DiskFormat::kVersion) return corrupt("unsupported index format version " + std::to_string(version));
    if (codes_kind != DiskFormat::kCodesSq8) return corrupt("unsupported codes kind");
    if (elem_type_raw != static_cast<std::uint32_t>(index_graph::ElemType::kFloat32)) {
        return corrupt("unsupported element type");
    }
    if (metric_raw > static_cast<std::uint32_t>(index_graph::Metric::kL2)) return corrupt("unknown metric");
    constexpr std::uint32_t kMaxDim = 1u << 16;
    if (dim == 0 || dim > kMaxDim) return corrupt("implausible dim " + std::to_string(dim));
    if (node_count_u64 == 0 || entry_point_u64 >= node_count_u64) return corrupt("bad node count / entry point");

    auto layout = index_graph::NodeRecordLayout::make(
        dim, static_cast<index_graph::ElemType>(elem_type_raw), degree_cap_r, sector_size);
    if (!layout) return std::unexpected(layout.error());
    std::size_t node_count = static_cast<std::size_t>(node_count_u64);

    // Exact expected size: header + records + codebook (2*dim floats) + codes blob (dim bytes
    // per node). Checked up front (with overflow guards) so a truncated or padded file is
    // rejected before any record is mapped or read.
    const std::uint64_t record_bytes = layout->record_bytes();
    if (node_count_u64 > (std::numeric_limits<std::uint64_t>::max() / 2) / (record_bytes + dim)) {
        return corrupt("node count overflows file size");
    }
    const std::uint64_t codebook_offset = header.size() + node_count_u64 * record_bytes;
    const std::uint64_t codebook_size = 2ull * dim * sizeof(float);
    const std::uint64_t codes_blob_offset = codebook_offset + codebook_size;
    const std::uint64_t expected_size = codes_blob_offset + node_count_u64 * dim;
    if (*file_size != expected_size) {
        return corrupt("file size " + std::to_string(*file_size) + " != expected " + std::to_string(expected_size));
    }

    std::vector<std::byte> codebook_bytes(codebook_size);
    auto cb_n = (*file_r)->pread(codebook_offset, codebook_bytes);
    if (!cb_n || *cb_n != codebook_bytes.size()) return corrupt("truncated Vamana codebook");
    quant::Sq8CodeBook codes_book;
    codes_book.per_dim_scale.resize(dim);
    codes_book.per_dim_offset.resize(dim);
    std::memcpy(codes_book.per_dim_scale.data(), codebook_bytes.data(), dim * sizeof(float));
    std::memcpy(codes_book.per_dim_offset.data(), codebook_bytes.data() + dim * sizeof(float),
                dim * sizeof(float));

    std::vector<std::byte> codes_blob(static_cast<std::size_t>(dim) * node_count);
    auto blob_n = (*file_r)->pread(codes_blob_offset, codes_blob);
    if (!blob_n || *blob_n != codes_blob.size()) return corrupt("truncated Vamana codes blob");

    std::unique_ptr<index_graph::NodeStore> store;
    if (access == NodeAccess::kMmap) {
        auto mapped = storage::MappedFile::open_readonly(path);
        if (!mapped) return std::unexpected(mapped.error());
        if (preload) {
            // Fault every page in once with normal readahead still on (sequential, so storage reads
            // at full bandwidth), removing
            // the cold-start penalty. Must precede advise_random(), which disables readahead.
            volatile std::uint8_t sink = 0;
            const auto bytes = mapped->data();
            for (std::size_t o = 0; o < bytes.size(); o += 4096) sink = static_cast<std::uint8_t>(sink + static_cast<std::uint8_t>(bytes[o]));
            (void)sink;
        }
        mapped->advise_random();  // graph traversal is random-access; disable readahead from here on
        store = std::make_unique<MmapNodeStore>(std::move(*mapped), *layout, node_count, header.size());
    } else {
        auto async_file = storage::AsyncNodeFile::open(path);
        if (!async_file) return std::unexpected(async_file.error());
        store = std::make_unique<DiskNodeStore>(std::move(*async_file), *layout, node_count, header.size());
    }

    return index_graph::VamanaIndex::open(std::move(store), *layout, dim, node_count,
                                           static_cast<VectorId>(entry_point_u64),
                                           static_cast<index_graph::Metric>(metric_raw),
                                           std::move(codes_book), std::move(codes_blob));
}

}  // namespace lattice::segment
