#pragma once
// segment: GrowingSegment (RocksDB row buffer + in-RAM vector mirror, no
// graph -- D-CON-02/finding 13), SealedSegment (seal pipeline), PK index
// (D-STO-08, corrected per finding 14), Snapshot (D-STO-05), merge
// (D-IDX-04). This is the module that wires together storage, filter,
// quant, index_graph, index_flat.
//
// Phase 3 scoping simplification (disclosed, not silent): a row here is
// {external_id, one dense vector, tombstone flag, source_offset}. Full
// typed-column/multi-field support (D-ARCH-03's named vectors + sparse +
// multi-vector + scalar columns) is deferred -- this slice proves the
// seal/merge/crash-consistency machinery end-to-end over the vector search
// path Phase 3's exit gate actually measures (filtered/unfiltered recall
// and latency), without also building the full schema system in the same
// pass. filter's PredicateEvaluator is still exercised in the exit-gate
// tests as a separate concern (segments carry only what they need to
// support a caller-supplied RowFilter, per D-MOD-01).
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "lattice/common/error.h"
#include "lattice/common/row_filter.h"
#include "lattice/common/types.h"
#include "lattice/index_flat/index_flat.h"
#include "lattice/index_graph/index_graph.h"
#include "lattice/storage/storage.h"

namespace rocksdb {
class DB;
class ColumnFamilyHandle;
}  // namespace rocksdb

namespace lattice::segment {

// --- GrowingSegment ---------------------------------------------------
// Metric contract (D-VEC-01): segments rank by inner product (flat scan and the sealed Vamana
// index alike), so callers must unit-normalise vectors before upsert for cosine semantics.
// Unnormalised vectors are accepted but rank by raw inner product.
class GrowingSegment {
public:
    static Result<std::unique_ptr<GrowingSegment>> create(const std::string& rocksdb_path,
                                                            std::uint32_t dim,
                                                            std::size_t seal_threshold_rows);
    ~GrowingSegment();

    // Upsert/delete are LWW by source_offset (D-STO-08); a delete for a
    // not-yet-existing id is stored as a durable offset-bearing tombstone
    // (finding 14), not dropped, so a later out-of-order create with a
    // smaller offset is correctly rejected after a restart.
    Result<Lsn> upsert(const std::string& external_id, const float* vector,
                        std::uint64_t source_offset);
    Result<Lsn> delete_row(const std::string& external_id, std::uint64_t source_offset);

    // Brute-force search over the in-RAM mirror only (D-IDX-06: growing
    // segment is never graph-indexed).
    Result<std::vector<std::string>> search(const float* query, std::uint32_t k,
                                             const RowFilter& filter = RowFilter::always_true());

    bool should_seal() const noexcept;
    std::size_t row_count() const noexcept;
    std::uint32_t dim() const noexcept { return dim_; }

    // Exposed for SealedSegment::seal_from and the PK index rebuild path.
    rocksdb::DB* db() noexcept { return db_; }
    rocksdb::ColumnFamilyHandle* cf() noexcept { return cf_; }
    const std::string& cf_name() const noexcept { return cf_name_; }

    GrowingSegment(const GrowingSegment&) = delete;
    GrowingSegment& operator=(const GrowingSegment&) = delete;

private:
    GrowingSegment() = default;

    // Loads every live (non-tombstone) row from RocksDB into the mirror; called by create() so a
    // reopened segment can search, count and seal rows written before the restart.
    Result<void> load_mirror_from_db();
    // Adds or overwrites `external_id`'s mirror slot and marks it live. Caller holds write_mutex_.
    void mirror_put(const std::string& external_id, const float* vector);

    rocksdb::DB* db_ = nullptr;  // owned by this segment (one RocksDB instance per growing segment)
    rocksdb::ColumnFamilyHandle* cf_ = nullptr;
    std::string cf_name_;
    std::uint32_t dim_ = 0;
    std::size_t seal_threshold_rows_ = 0;

    // Serializes upsert/delete so the last-write-wins check (read current offset, then write)
    // is atomic with respect to other writers of the same segment.
    std::mutex write_mutex_;

    // In-RAM vector mirror (finding 13): flat search scans this, never RocksDB, so a query never
    // pays a KV read per row. Slots are never reused; a deleted row's slot is marked dead.
    mutable std::shared_mutex mirror_mutex_;  // guards append vs. reallocation
    std::vector<float> mirror_;               // slot i's vector at [i*dim_, (i+1)*dim_)
    std::vector<std::string> mirror_ids_;      // slot i's external id
    std::vector<char> mirror_live_;            // slot i holds a live (not deleted) row
    std::unordered_map<std::string, std::size_t> slot_of_;  // external id -> slot
    std::atomic<std::size_t> visible_count_{0};  // slots published to readers (release/acquire)
    std::atomic<std::size_t> live_count_{0};     // live rows (row_count / should_seal)
};

struct MergePolicy;  // defined below; forward-declared for SealedSegment's friend decl

// --- SealedSegment ------------------------------------------------------
class SealedSegment {
public:
    // Declared (not defaulted) here and defined in segment.cpp: the
    // implicit destructor would need VamanaHolder's complete definition to
    // destroy the unique_ptr<VamanaHolder> member, but that type is only
    // forward-declared in this header (classic pimpl requirement).
    ~SealedSegment();

    // Iterates the growing segment's CF in sorted key order, skips
    // tombstoned/superseded rows, builds either an index_flat scan (below
    // D-IDX-06's threshold) or an in-RAM Vamana index, and returns the
    // manifest entry to publish (finding 11: caller writes this durably
    // BEFORE dropping the source CF).
    static Result<std::unique_ptr<SealedSegment>> seal_from(GrowingSegment& growing,
                                                              SegmentId id,
                                                              std::size_t small_segment_threshold);

    Result<std::vector<std::string>> search(const float* query, std::uint32_t k,
                                             const RowFilter& filter = RowFilter::always_true()) const;

    std::size_t row_count() const noexcept { return ids_.size(); }
    SegmentId id() const noexcept { return id_; }
    std::uint64_t max_rocksdb_seq() const noexcept { return max_seq_; }
    const std::string& source_cf() const noexcept { return source_cf_; }

    // VectorId -> external id, used to translate search results and by
    // merge to re-key rows into a new sealed segment (D-MOD-02: VectorId is
    // this module's internal index handle, never exposed past this point).
    const std::string& external_id_for(VectorId vid) const { return ids_.at(static_cast<std::size_t>(vid)); }

    // The row's original CDC source_offset (D-STO-08), preserved through
    // sealing/merge so that PrimaryKeyIndex::rebuild's LWW resolution stays
    // correct for rows that live in a sealed segment, not just a growing one.
    std::uint64_t source_offset_for(VectorId vid) const {
        return offsets_.at(static_cast<std::size_t>(vid));
    }

    // Used by merge() to read raw rows out of an existing sealed segment.
    std::uint32_t dim() const noexcept { return dim_; }
    // For graph-indexed segments the vector is read from the index's node record (the separate
    // copy is freed after the build), so the span is only valid until the next read on this
    // thread -- copy it out.
    std::span<const float> vector_at(VectorId vid) const;

    // Durable (external_id, source_offset) list for this segment, written
    // once at <dir>/<segment_id>.ids (exit gate item 6): this is what makes
    // PrimaryKeyIndex::rebuild() possible after the source RocksDB CF has
    // been dropped post-seal -- without it, a sealed segment's id->location
    // mapping would exist only in RAM and be unrecoverable after a restart.
    // Callers (commit_seal) write this BEFORE the manifest entry goes
    // durable, so a crash before the manifest write leaves, at worst, a
    // harmless orphan .ids file for a segment that isn't live yet.
    Result<void> write_id_file(const std::string& dir) const;

private:
    // seal_from() and merge() both funnel through this (defined in
    // segment.cpp): the index-building pipeline is identical whether rows
    // come from a GrowingSegment's RocksDB CF or from existing
    // SealedSegments. A private static member (not a free function) since
    // it needs to construct/populate a SealedSegment's private state.
    static Result<std::unique_ptr<SealedSegment>> build(SegmentId id, std::string source_cf,
                                                         std::uint64_t max_seq, std::uint32_t dim,
                                                         std::vector<std::string> ids,
                                                         std::vector<std::uint64_t> offsets,
                                                         std::vector<float> vectors,
                                                         std::size_t small_segment_threshold);

    friend Result<std::unique_ptr<SealedSegment>> merge(std::span<const SealedSegment* const>,
                                                         SegmentId, const MergePolicy&);

    // Declared here, defined in segment.cpp -- same pimpl reason as the
    // destructor above: an inline body here would force this TU to reason
    // about destroying the (incomplete-here) VamanaHolder for exception
    // unwinding, even though vamana_ is default-constructed to null.
    SealedSegment(SegmentId id, std::string source_cf, std::uint64_t max_seq);

    SegmentId id_;
    std::string source_cf_;
    std::uint64_t max_seq_;
    std::uint32_t dim_ = 0;

    std::vector<float> vectors_;       // contiguous, row-major
    std::vector<std::string> ids_;     // VectorId i -> external id
    std::vector<std::uint64_t> offsets_;  // VectorId i -> source_offset (D-STO-08)

    bool use_flat_ = true;
    std::unique_ptr<class VamanaHolder> vamana_;  // pimpl to avoid a hard index_graph dep in the header
};

// --- Primary key index (D-STO-08, corrected per finding 14) -------------
class PrimaryKeyIndex {
public:
    struct Location {
        std::optional<SegmentId> segment;  // nullopt = growing segment (or tombstoned if deleted=true)
        bool deleted = false;
    };

    PrimaryKeyIndex() = default;
    // Movable (not copyable) despite the mutex member -- needed so
    // rebuild() below can return a freshly-built instance by value. Moving
    // is safe: mutex_ is per-instance lock state, not data, so a moved-from
    // index just becomes an empty one under its own (untouched) mutex.
    PrimaryKeyIndex(PrimaryKeyIndex&& other) noexcept {
        std::lock_guard lock(other.mutex_);
        entries_ = std::move(other.entries_);
    }
    PrimaryKeyIndex& operator=(PrimaryKeyIndex&& other) noexcept {
        if (this == &other) return *this;
        std::scoped_lock lock(mutex_, other.mutex_);
        entries_ = std::move(other.entries_);
        return *this;
    }
    PrimaryKeyIndex(const PrimaryKeyIndex&) = delete;
    PrimaryKeyIndex& operator=(const PrimaryKeyIndex&) = delete;

    // LWW by source_offset; a delete for an id with no prior entry is
    // recorded as a tombstone with its offset, not dropped, so a later
    // create with a smaller offset is correctly rejected (finding 14).
    void apply(const std::string& external_id, std::uint64_t source_offset,
               std::optional<SegmentId> segment_if_present);
    void apply_delete(const std::string& external_id, std::uint64_t source_offset);

    std::optional<Location> lookup(const std::string& external_id) const;

    // Exit gate item 6: reconstructs a PrimaryKeyIndex from durable state
    // alone, for use after a restart (simulated by the crash-injection
    // harness as a real process crash mid-seal). Sources, per
    // module-segment.md's "rebuildable from" note, corrected to what's
    // actually persisted (finding 14's scope correction): each live
    // segment's <dir>/<id>.ids file (written by SealedSegment::write_id_file
    // before its manifest entry goes durable -- see commit_seal). This does
    // NOT see rows still sitting only in a growing segment's RocksDB CF;
    // callers must additionally call replay_growing_segment_cf() for any
    // growing segment(s) that are still live (not yet sealed away).
    static Result<PrimaryKeyIndex> rebuild(const storage::Manifest& manifest, const std::string& dir);

private:
    struct Entry {
        std::uint64_t source_offset;
        Location location;
    };
    mutable std::mutex mutex_;
    std::map<std::string, Entry> entries_;
};

// Replays a still-live growing segment's RocksDB CF into a PrimaryKeyIndex
// (rows that haven't been sealed away yet, so PrimaryKeyIndex::rebuild's
// .ids files don't cover them). Opens `rocksdb_dir` read-only -- this is a
// recovery-time read, not a write path, and must not require exclusive
// access to a directory the real GrowingSegment may still hold open.
Result<void> replay_growing_segment_cf(PrimaryKeyIndex& pk, const std::string& rocksdb_dir);

// --- Snapshot (D-STO-05) -------------------------------------------------
class Snapshot {
public:
    explicit Snapshot(std::vector<SegmentId> segments) : segments_(std::move(segments)) {}
    const std::vector<SegmentId>& visible_segments() const noexcept { return segments_; }

private:
    std::vector<SegmentId> segments_;
};

// --- The two-phase seal handoff (finding 11) ------------------------------
// seal_from() only builds the in-memory SealedSegment; commit_seal() is the
// operation with the actual crash-safety property: the manifest entry MUST
// be durable before the growing segment's RocksDB directory is removed. A
// crash between those two steps must be safe to resume from (the manifest
// entry alone is enough to know sealing already happened; the leftover
// RocksDB directory is a harmless, idempotently-cleanable orphan, not a
// correctness hazard -- matches D-STO-05's "deleted when nothing pins it").
//
// `crash_hook`, if set, is called at each checkpoint (1 = right before the
// manifest write, 2 = right after it succeeds, 3 = right after the RocksDB
// directory is removed) -- this is the crash-injection harness's hook
// (tools/crash_harness_main.cpp), which calls _exit() at a chosen
// checkpoint to simulate a real process crash at that exact point.
using CrashHook = std::function<void(int checkpoint)>;

Result<void> commit_seal(std::unique_ptr<GrowingSegment> growing, const SealedSegment& sealed,
                          const std::string& manifest_dir, const CrashHook& crash_hook = {});

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

// --- Merge (D-IDX-04: rebuild strategy, finding 7: bounded transient space) ---
struct MergePolicy {
    std::size_t max_segment_size_vectors = 2'000'000;  // Phase 3 measured default (finding 7)
    std::size_t small_segment_threshold = 10'000;      // D-IDX-06, matches seal_from's default
};

// Rebuild strategy (D-IDX-04, Phase 3): reads every row out of `inputs`,
// resolves duplicates by keeping the LAST occurrence of a given external_id
// across the input list (later segments in `inputs` are assumed newer --
// Phase 3 scoping simplification, disclosed: real per-segment delete
// tombstones for already-sealed rows are a follow-on piece; today a row is
// only ever deleted while still in the growing segment), and builds ONE new
// SealedSegment from the result via the same pipeline seal_from() uses.
//
// If the deduplicated row count exceeds policy.max_segment_size_vectors,
// this returns an error rather than silently building an oversized segment
// -- splitting a merge across multiple output segments is a real feature,
// not implemented here; the bound is enforced, not just measured.
Result<std::unique_ptr<SealedSegment>> merge(std::span<const SealedSegment* const> inputs,
                                              SegmentId new_id, const MergePolicy& policy);

}  // namespace lattice::segment
