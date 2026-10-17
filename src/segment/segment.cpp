#include "lattice/segment/segment.h"

#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_map>

#include "lattice/index_graph/index_graph.h"
#include "lattice/quant/sq8.h"

namespace lattice::segment {

// Row value encoding in RocksDB: [source_offset:8][tombstone:1][dim:4][vector: dim*4 if !tombstone]
namespace {
constexpr std::size_t kRowHeaderBytes = 8 + 1 + 4;
constexpr const char* kGrowingSegmentMarker = "LATTICE_GROWING_SEGMENT";

std::vector<char> encode_row(std::uint64_t offset, bool tombstone, const float* vector,
                              std::uint32_t dim) {
    std::vector<char> buf(kRowHeaderBytes + (tombstone ? 0 : dim * 4u));
    std::size_t off = 0;
    std::memcpy(buf.data() + off, &offset, 8); off += 8;
    char t = tombstone ? 1 : 0;
    std::memcpy(buf.data() + off, &t, 1); off += 1;
    std::memcpy(buf.data() + off, &dim, 4); off += 4;
    if (!tombstone) std::memcpy(buf.data() + off, vector, dim * 4u);
    return buf;
}

struct DecodedRow {
    std::uint64_t offset;
    bool tombstone;
    std::uint32_t dim;
    const char* vector_ptr;  // points into the source string; valid as long as it's alive
};

// nullopt if `value` is not a well-formed row: truncated, or its length disagrees with its dim
// field, or (when expected_dim != 0) its dim differs from the segment's. Reading such a row
// unchecked would over-read the RocksDB value.
std::optional<DecodedRow> decode_row(const std::string& value, std::uint32_t expected_dim) {
    if (value.size() < kRowHeaderBytes) return std::nullopt;
    DecodedRow r{};
    std::size_t off = 0;
    std::memcpy(&r.offset, value.data() + off, 8); off += 8;
    char t; std::memcpy(&t, value.data() + off, 1); off += 1;
    r.tombstone = (t != 0);
    std::memcpy(&r.dim, value.data() + off, 4); off += 4;
    if (expected_dim != 0 && r.dim != expected_dim) return std::nullopt;
    const std::size_t want = kRowHeaderBytes + (r.tombstone ? 0 : static_cast<std::size_t>(r.dim) * 4);
    if (value.size() != want) return std::nullopt;
    r.vector_ptr = r.tombstone ? nullptr : value.data() + off;
    return r;
}

Error corrupt_row(const std::string& key) {
    return Error::make(ErrorCode::kCorruption, "malformed row in growing segment for id: " + key);
}

bool all_finite(const float* v, std::uint32_t dim) {
    for (std::uint32_t d = 0; d < dim; ++d)
        if (!std::isfinite(v[d])) return false;
    return true;
}
}  // namespace

// --- GrowingSegment -------------------------------------------------------

Result<std::unique_ptr<GrowingSegment>> GrowingSegment::create(const std::string& rocksdb_path,
                                                                std::uint32_t dim,
                                                                std::size_t seal_threshold_rows) {
    if (dim == 0) return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "dim must be > 0"));
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::DB* db = nullptr;
    auto status = rocksdb::DB::Open(options, rocksdb_path, &db);
    if (!status.ok()) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "RocksDB open failed: " + status.ToString()));
    }

    auto seg = std::unique_ptr<GrowingSegment>(new GrowingSegment());
    seg->db_ = db;
    seg->cf_ = db->DefaultColumnFamily();
    seg->cf_name_ = rocksdb_path;  // one RocksDB instance per growing segment (own directory)
    // Marker proving this directory is a Lattice-managed growing segment: commit_seal deletes the
    // directory recursively after sealing and must never do that to a directory it did not create.
    {
        auto marker = storage::PosixFile::open(rocksdb_path + "/" + kGrowingSegmentMarker, /*create=*/true);
        if (!marker) return std::unexpected(marker.error());
    }
    seg->dim_ = dim;
    seg->seal_threshold_rows_ = seal_threshold_rows;
    // Pre-reserve so normal growth does not reallocate (writers take the mirror lock anyway).
    seg->mirror_.reserve((seal_threshold_rows + seal_threshold_rows / 5) * dim);

    // Rows written before a restart live only in RocksDB; without this they were invisible to
    // search() and row_count()/should_seal() reported 0 until the segment was sealed.
    auto loaded = seg->load_mirror_from_db();
    if (!loaded) return std::unexpected(loaded.error());
    return seg;
}

GrowingSegment::~GrowingSegment() { delete db_; }

Result<void> GrowingSegment::load_mirror_from_db() {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(rocksdb::ReadOptions(), cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();
        std::string value = it->value().ToString();
        auto decoded = decode_row(value, dim_);
        if (!decoded) return std::unexpected(corrupt_row(key));
        if (decoded->tombstone) continue;
        std::vector<float> v(dim_);
        std::memcpy(v.data(), decoded->vector_ptr, dim_ * sizeof(float));
        mirror_put(key, v.data());
    }
    if (!it->status().ok()) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "growing segment reload failed: " + it->status().ToString()));
    }
    return {};
}

void GrowingSegment::mirror_put(const std::string& external_id, const float* vector) {
    std::unique_lock lock(mirror_mutex_);
    auto found = slot_of_.find(external_id);
    if (found != slot_of_.end()) {
        std::size_t slot = found->second;
        std::memcpy(mirror_.data() + slot * dim_, vector, dim_ * sizeof(float));
        if (!mirror_live_[slot]) {
            mirror_live_[slot] = 1;
            live_count_.fetch_add(1, std::memory_order_release);
        }
        return;
    }
    std::size_t slot = mirror_ids_.size();
    mirror_.resize((slot + 1) * dim_);
    std::memcpy(mirror_.data() + slot * dim_, vector, dim_ * sizeof(float));
    mirror_ids_.push_back(external_id);
    mirror_live_.push_back(1);
    slot_of_.emplace(external_id, slot);
    live_count_.fetch_add(1, std::memory_order_release);
    visible_count_.store(slot + 1, std::memory_order_release);
}

Result<Lsn> GrowingSegment::upsert(const std::string& external_id, const float* vector,
                                    std::uint64_t source_offset) {
    if (vector == nullptr || !all_finite(vector, dim_)) {
        return std::unexpected(
            Error::make(ErrorCode::kInvalidArgument, "vector is null or contains NaN/Inf: " + external_id));
    }
    std::lock_guard writer(write_mutex_);  // LWW check + write must be atomic
    std::string existing;
    auto get_status = db_->Get(rocksdb::ReadOptions(), cf_, external_id, &existing);
    if (get_status.ok()) {
        auto decoded = decode_row(existing, dim_);
        if (!decoded) return std::unexpected(corrupt_row(external_id));
        if (decoded->offset >= source_offset) {
            // Stale write (LWW, D-STO-08): no-op, not an error.
            return static_cast<Lsn>(db_->GetLatestSequenceNumber());
        }
    } else if (!get_status.IsNotFound()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, get_status.ToString()));
    }

    auto encoded = encode_row(source_offset, false, vector, dim_);
    auto put_status = db_->Put(rocksdb::WriteOptions(), cf_, external_id,
                               rocksdb::Slice(encoded.data(), encoded.size()));
    if (!put_status.ok()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, put_status.ToString()));
    }
    mirror_put(external_id, vector);
    return static_cast<Lsn>(db_->GetLatestSequenceNumber());
}

Result<Lsn> GrowingSegment::delete_row(const std::string& external_id, std::uint64_t source_offset) {
    std::lock_guard writer(write_mutex_);
    std::string existing;
    auto get_status = db_->Get(rocksdb::ReadOptions(), cf_, external_id, &existing);
    if (get_status.ok()) {
        auto decoded = decode_row(existing, dim_);
        if (!decoded) return std::unexpected(corrupt_row(external_id));
        if (decoded->offset >= source_offset) {
            return static_cast<Lsn>(db_->GetLatestSequenceNumber());  // stale delete, no-op
        }
    } else if (!get_status.IsNotFound()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, get_status.ToString()));
    }
    // Durable offset-bearing tombstone even if the row never existed
    // (finding 14): a later out-of-order create with a smaller offset must
    // see this and be rejected.
    auto encoded = encode_row(source_offset, true, nullptr, dim_);
    auto put_status =
        db_->Put(rocksdb::WriteOptions(), cf_, external_id, rocksdb::Slice(encoded.data(), encoded.size()));
    if (!put_status.ok()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, put_status.ToString()));
    }
    // Hide the row from search immediately (it used to stay searchable until the next seal).
    {
        std::unique_lock lock(mirror_mutex_);
        auto found = slot_of_.find(external_id);
        if (found != slot_of_.end() && mirror_live_[found->second]) {
            mirror_live_[found->second] = 0;
            live_count_.fetch_sub(1, std::memory_order_release);
        }
    }
    return static_cast<Lsn>(db_->GetLatestSequenceNumber());
}

Result<std::vector<std::string>> GrowingSegment::search(const float* query, std::uint32_t k,
                                                          const RowFilter& filter) {
    std::shared_lock lock(mirror_mutex_);
    std::size_t count = visible_count_.load(std::memory_order_acquire);
    index_flat::FlatView view{mirror_.data(), dim_, dim_, count};
    auto admit = [&](VectorId vid) {
        auto slot = static_cast<std::size_t>(vid);
        return mirror_live_[slot] != 0 && filter.test(static_cast<RowId>(vid));
    };
    auto result = index_flat::search(view, query, k, admit);
    if (!result) return std::unexpected(result.error());

    std::vector<std::string> out;
    out.reserve(result->size());
    for (auto vid : *result) out.push_back(mirror_ids_[static_cast<std::size_t>(vid)]);
    return out;
}

bool GrowingSegment::should_seal() const noexcept {
    return live_count_.load(std::memory_order_acquire) >= seal_threshold_rows_;
}

std::size_t GrowingSegment::row_count() const noexcept {
    return live_count_.load(std::memory_order_acquire);
}

// --- SealedSegment ---------------------------------------------------------

// Pimpl so the header doesn't force every includer of segment.h to also see
// index_graph's template machinery.
class VamanaHolder {
public:
    std::unique_ptr<quant::Sq8CodeBook> book;
    std::unique_ptr<quant::Sq8Codes> codes;
    std::unique_ptr<index_graph::VamanaIndex> index;
};

// Defined here, not inline in the header (see segment.h's comment) --
// VamanaHolder is complete in this translation unit.
SealedSegment::SealedSegment(SegmentId id, std::string source_cf, std::uint64_t max_seq)
    : id_(id), source_cf_(std::move(source_cf)), max_seq_(max_seq) {}
SealedSegment::~SealedSegment() = default;

// Shared by seal_from() (rows come from a GrowingSegment's RocksDB CF) and
// merge() (rows come from existing SealedSegments) -- the index-building
// pipeline is identical either way, only the row SOURCE differs. A private
// static member (declared in segment.h) so it can construct/populate a
// SealedSegment's private state; merge() is a friend for the same reason.
Result<std::unique_ptr<SealedSegment>> SealedSegment::build(
    SegmentId id, std::string source_cf, std::uint64_t max_seq, std::uint32_t dim,
    std::vector<std::string> ids, std::vector<std::uint64_t> offsets, std::vector<float> vectors,
    std::size_t small_segment_threshold) {
    auto sealed = std::unique_ptr<SealedSegment>(new SealedSegment(id, std::move(source_cf), max_seq));
    sealed->dim_ = dim;
    sealed->vectors_ = std::move(vectors);
    sealed->ids_ = std::move(ids);
    sealed->offsets_ = std::move(offsets);

    std::size_t n = sealed->ids_.size();
    if (n < small_segment_threshold || n == 0) {
        sealed->use_flat_ = true;  // D-IDX-06: small segments stay brute-force
        return sealed;
    }

    // Build an in-RAM Vamana index (Phase 2's backend; disk-resident
    // serialization is a separate opt-in step via serialize_to_disk()).
    sealed->use_flat_ = false;
    sealed->vamana_ = std::make_unique<VamanaHolder>();

    std::vector<const float*> ptrs(n);
    for (std::size_t i = 0; i < n; ++i) ptrs[i] = sealed->vectors_.data() + i * dim;

    auto book = quant::Sq8CodeBook::train(ptrs, dim);
    if (!book) return std::unexpected(book.error());
    sealed->vamana_->book = std::make_unique<quant::Sq8CodeBook>(std::move(*book));
    sealed->vamana_->codes = std::make_unique<quant::Sq8Codes>(*sealed->vamana_->book);

    auto stride = index_graph::NodeRecordLayout::minimal_aligned_stride(dim, index_graph::ElemType::kFloat32, 64);
    auto layout = index_graph::NodeRecordLayout::make(dim, index_graph::ElemType::kFloat32, 64, stride);
    if (!layout) return std::unexpected(layout.error());

    index_graph::VamanaBuildParams params;
    params.r = 64;
    params.build_list_size = 100;
    params.metric = index_graph::Metric::kInnerProduct;

    auto index = index_graph::VamanaIndex::build(ptrs, dim, *sealed->vamana_->codes, params, *layout);
    if (!index) return std::unexpected(index.error());
    sealed->vamana_->index = std::make_unique<index_graph::VamanaIndex>(std::move(*index));
    // The index keeps every vector in its node records; holding a second full copy doubled the
    // segment's memory. vector_at() reads from the index from here on.
    std::vector<float>().swap(sealed->vectors_);

    return sealed;
}

std::span<const float> SealedSegment::vector_at(VectorId vid) const {
    if (!use_flat_ && vamana_) return vamana_->index->vector_of(vid);
    auto idx = static_cast<std::size_t>(vid);
    return {vectors_.data() + idx * dim_, dim_};
}

Result<std::unique_ptr<SealedSegment>> SealedSegment::seal_from(GrowingSegment& growing,
                                                                  SegmentId id,
                                                                  std::size_t small_segment_threshold) {
    std::uint32_t dim = growing.dim();
    std::uint64_t max_seq = growing.db()->GetLatestSequenceNumber();

    // Iterate the CF in sorted key order (D-STO-02's seal pipeline),
    // skipping tombstones, keeping only the latest (already-LWW-resolved by
    // upsert/delete) value per key.
    std::vector<float> vectors;
    std::vector<std::string> ids;
    std::vector<std::uint64_t> offsets;
    {
        std::unique_ptr<rocksdb::Iterator> it(
            growing.db()->NewIterator(rocksdb::ReadOptions(), growing.cf()));
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            // value_str must outlive `decoded` (decoded.vector_ptr points
            // into it) -- an earlier version passed a temporary directly to
            // decode_row(), which ASan caught as a use-after-free once the
            // temporary was destroyed at the end of that sub-expression.
            std::string value_str = it->value().ToString();
            std::string id_str = it->key().ToString();
            auto decoded = decode_row(value_str, dim);
            if (!decoded) return std::unexpected(corrupt_row(id_str));
            if (decoded->tombstone) continue;
            std::size_t base = vectors.size();
            vectors.resize(base + dim);
            std::memcpy(vectors.data() + base, decoded->vector_ptr, dim * sizeof(float));
            ids.push_back(std::move(id_str));
            offsets.push_back(decoded->offset);
        }
        if (!it->status().ok()) {
            return std::unexpected(
                Error::make(ErrorCode::kIoError, "seal iteration failed: " + it->status().ToString()));
        }
    }

    return SealedSegment::build(id, growing.cf_name(), max_seq, dim, std::move(ids), std::move(offsets),
                                 std::move(vectors), small_segment_threshold);
}

Result<std::vector<std::string>> SealedSegment::search(const float* query, std::uint32_t k,
                                                         const RowFilter& filter) const {
    std::vector<VectorId> result_ids;
    if (use_flat_ || !vamana_) {
        index_flat::FlatView view{vectors_.data(), dim_, dim_, ids_.size()};
        auto admit = [&](VectorId vid) { return filter.test(static_cast<RowId>(vid)); };
        auto r = index_flat::search(view, query, k, admit);
        if (!r) return std::unexpected(r.error());
        result_ids = std::move(*r);
    } else {
        index_graph::SearchParams sp;
        sp.ef_search = std::max<std::uint32_t>(64, k * 4);
        sp.rerank_k = std::max<std::uint32_t>(40, k * 2);
        if (filter.is_always_true()) {
            auto r = vamana_->index->search(query, k, sp);
            if (!r) return std::unexpected(r.error());
            result_ids = std::move(*r);
        } else {
            // Filter-aware planner (scan the matching rows when few, widen the beam when many):
            // a plain post-filter on the graph walk collapses recall at low selectivity.
            std::vector<VectorId> matching;
            for (std::size_t i = 0; i < ids_.size(); ++i) {
                if (filter.test(static_cast<RowId>(i))) matching.push_back(static_cast<VectorId>(i));
            }
            if (matching.empty()) return std::vector<std::string>{};
            index_graph::VamanaIndex::FilteredParams fp;
            fp.base = sp;
            fp.matching_ids = matching;
            index_graph::AdmitFn admit = [&](VectorId vid) { return filter.test(static_cast<RowId>(vid)); };
            auto r = vamana_->index->search_filtered(query, k, fp, admit);
            if (!r) return std::unexpected(r.error());
            result_ids = std::move(*r);
        }
    }
    std::vector<std::string> out;
    out.reserve(result_ids.size());
    for (auto vid : result_ids) out.push_back(ids_[static_cast<std::size_t>(vid)]);
    return out;
}

// (external_id, source_offset) file format, magic 'LATI': [count:4] then
// count * [offset:8][id_len:4][id bytes], followed by the shared FileFooter.
// Deliberately NOT the manifest's own footer-magic namespace ('LAT0') --
// this is a different file kind and must not round-trip as a manifest by
// accident.
namespace {
constexpr std::uint32_t kIdFileMagic = 0x4C415449;  // 'LATI'
}  // namespace

Result<void> SealedSegment::write_id_file(const std::string& dir) const {
    std::vector<std::byte> buf(4);
    std::uint32_t count = static_cast<std::uint32_t>(ids_.size());
    std::memcpy(buf.data(), &count, 4);
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        std::size_t base = buf.size();
        buf.resize(base + 8 + 4 + ids_[i].size());
        std::memcpy(buf.data() + base, &offsets_[i], 8);
        std::uint32_t id_len = static_cast<std::uint32_t>(ids_[i].size());
        std::memcpy(buf.data() + base + 8, &id_len, 4);
        std::memcpy(buf.data() + base + 12, ids_[i].data(), id_len);
    }

    storage::FileFooter footer{kIdFileMagic, 1, 0, count};
    footer.checksum = storage::crc32c(buf);
    std::vector<std::byte> footer_bytes(storage::kFileFooterBytes);
    storage::write_footer(footer_bytes, footer);
    buf.insert(buf.end(), footer_bytes.begin(), footer_bytes.end());

    std::string path = dir + "/" + std::to_string(static_cast<std::uint32_t>(id_)) + ".ids";
    // Truncate: a retried seal rewrites this file, and a stale longer tail would leave the old
    // footer at the end (checksum mismatch in PrimaryKeyIndex::rebuild).
    auto f = storage::PosixFile::open(path, /*create=*/true, /*truncate=*/true);
    if (!f) return std::unexpected(f.error());
    auto w = (*f)->pwrite(0, buf);
    if (!w || *w != buf.size()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "short write to .ids file"));
    }
    return (*f)->fsync();
}

// --- merge (D-IDX-04, finding 7) --------------------------------------------

Result<std::unique_ptr<SealedSegment>> merge(std::span<const SealedSegment* const> inputs,
                                              SegmentId new_id, const MergePolicy& policy) {
    if (inputs.empty()) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "merge requires >=1 input"));
    }
    std::uint32_t dim = 0;
    for (auto* seg : inputs) {
        if (!seg) {
            return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "null input segment"));
        }
    }

    // Last occurrence of a given external_id wins (later segment = newer,
    // per this function's disclosed scoping note in segment.h). Using a map
    // keyed by id to a (source_index, row_index) pair keeps this O(N log N)
    // over the total row count across all inputs rather than O(N^2).
    std::map<std::string, std::pair<std::size_t, std::size_t>> latest;  // id -> (input_idx, row_idx)
    for (std::size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        const auto* seg = inputs[input_idx];
        for (std::size_t row_idx = 0; row_idx < seg->row_count(); ++row_idx) {
            latest[seg->external_id_for(static_cast<VectorId>(row_idx))] = {input_idx, row_idx};
        }
    }

    if (latest.size() > policy.max_segment_size_vectors) {
        return std::unexpected(Error::make(
            ErrorCode::kUnsupported,
            "merged size " + std::to_string(latest.size()) + " exceeds max_segment_size_vectors " +
                std::to_string(policy.max_segment_size_vectors) +
                " (finding 7: splitting a merge output is not yet implemented)"));
    }

    // dim is the same across all sealed segments in a collection (D-ARCH-08
    // fixes dim per vector field); take it from whichever input has rows.
    for (auto* seg : inputs) {
        if (seg->row_count() == 0) continue;
        if (dim == 0) {
            dim = seg->dim();
        } else if (seg->dim() != dim) {
            // Mixing dims would silently misalign every row after the first mismatched segment.
            return std::unexpected(Error::make(ErrorCode::kDimensionMismatch, "merge inputs have different dims"));
        }
    }

    std::vector<std::string> ids;
    std::vector<std::uint64_t> offsets;
    std::vector<float> vectors;
    ids.reserve(latest.size());
    offsets.reserve(latest.size());
    vectors.reserve(latest.size() * dim);
    std::uint64_t max_seq = 0;
    for (auto& [id, loc] : latest) {
        const auto* seg = inputs[loc.first];
        max_seq = std::max(max_seq, seg->max_rocksdb_seq());
        ids.push_back(id);
        offsets.push_back(seg->source_offset_for(static_cast<VectorId>(loc.second)));
        auto v = seg->vector_at(static_cast<VectorId>(loc.second));
        vectors.insert(vectors.end(), v.begin(), v.end());
    }

    return SealedSegment::build(new_id, "merged", max_seq, dim, std::move(ids), std::move(offsets),
                                 std::move(vectors), policy.small_segment_threshold);
}

// --- commit_seal (finding 11's two-phase handoff) --------------------------

Result<void> commit_seal(std::unique_ptr<GrowingSegment> growing, const SealedSegment& sealed,
                          const std::string& manifest_dir, const CrashHook& crash_hook) {
    std::string source_cf = growing->cf_name();

    // Write the durable id->offset list BEFORE the manifest entry (exit gate
    // item 6): PrimaryKeyIndex::rebuild only trusts a segment's .ids file
    // once that segment is live in the manifest, so a crash here (before
    // checkpoint 1) just leaves a harmless orphan .ids file for a
    // not-yet-live segment -- same shape as the orphaned RocksDB directory
    // this function already tolerates below.
    auto id_file = sealed.write_id_file(manifest_dir);
    if (!id_file) return std::unexpected(id_file.error());

    if (crash_hook) crash_hook(1);  // checkpoint 1: right before the manifest write

    // A load failure must abort the seal: continuing with an empty manifest would write a new
    // manifest containing only this segment and silently drop every other live segment.
    auto loaded = storage::Manifest::load_current(manifest_dir);
    if (!loaded) return std::unexpected(loaded.error());
    storage::Manifest manifest = std::move(*loaded);
    auto entries = manifest.live_segments();

    // Idempotency (needed for retry-after-crash, not just a nicety): if
    // this segment id is ALREADY durably sealed (a prior attempt crashed
    // after checkpoint 2 but before/during cleanup), don't append a
    // duplicate entry -- just fall through to the (idempotent) cleanup step.
    bool already_sealed = false;
    for (auto& e : entries) {
        if (e.id == sealed.id()) {
            already_sealed = true;
            break;
        }
    }

    if (!already_sealed) {
        entries.push_back(storage::SealedSegmentEntry{sealed.id(), source_cf, sealed.max_rocksdb_seq()});
        auto swap_result = manifest.atomic_swap(manifest_dir, entries);
        if (!swap_result) {
            return std::unexpected(swap_result.error());  // never reached checkpoint 2: retry from scratch is safe
        }
    }

    if (crash_hook) crash_hook(2);  // checkpoint 2: manifest entry is now durable

    // Close the RocksDB handle before removing its files (an open handle
    // pointing at deleted files is a different, avoidable hazard).
    growing.reset();
    std::error_code ec;
    // Only delete what GrowingSegment::create made: a path without the marker (a mis-pointed or
    // shared directory) is left alone; leaving it is a harmless leak, deleting it is not.
    if (!source_cf.empty() && std::filesystem::exists(source_cf + "/" + kGrowingSegmentMarker, ec)) {
        std::filesystem::remove_all(source_cf, ec);
    }
    // A failure to remove here is NOT a correctness problem (D-STO-05: an
    // orphaned, unreferenced directory is a harmless leak, cleanable by a
    // background sweep later) -- the manifest already durably reflects the
    // seal, which is the only property that matters for correctness.

    if (crash_hook) crash_hook(3);  // checkpoint 3: cleanup done (or harmlessly skipped)
    return {};
}


// --- PrimaryKeyIndex (D-STO-08, finding 14) --------------------------------

void PrimaryKeyIndex::apply(const std::string& external_id, std::uint64_t source_offset,
                             std::optional<SegmentId> segment_if_present) {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(external_id);
    if (it != entries_.end() && it->second.source_offset >= source_offset) return;  // LWW
    entries_[external_id] = Entry{source_offset, Location{segment_if_present, false}};
}

void PrimaryKeyIndex::apply_delete(const std::string& external_id, std::uint64_t source_offset) {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(external_id);
    if (it != entries_.end() && it->second.source_offset >= source_offset) return;  // LWW
    // Tombstone retained durably (in this in-memory index's view) even if
    // there was no prior entry -- finding 14's delete-before-create fix.
    entries_[external_id] = Entry{source_offset, Location{std::nullopt, true}};
}

std::optional<PrimaryKeyIndex::Location> PrimaryKeyIndex::lookup(const std::string& external_id) const {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(external_id);
    if (it == entries_.end()) return std::nullopt;
    return it->second.location;
}

Result<PrimaryKeyIndex> PrimaryKeyIndex::rebuild(const storage::Manifest& manifest,
                                                  const std::string& dir) {
    PrimaryKeyIndex pk;
    for (const auto& entry : manifest.live_segments()) {
        std::string path =
            dir + "/" + std::to_string(static_cast<std::uint32_t>(entry.id)) + ".ids";
        auto f = storage::PosixFile::open(path, /*create=*/false);
        if (!f) {
            return std::unexpected(Error::make(
                ErrorCode::kCorruption,
                "PrimaryKeyIndex::rebuild: missing .ids file for live segment: " + path));
        }
        auto sz = (*f)->size();
        if (!sz) return std::unexpected(sz.error());
        if (*sz < storage::kFileFooterBytes + 4) {
            return std::unexpected(
                Error::make(ErrorCode::kCorruption, "id file too short: " + path));
        }
        std::vector<std::byte> buf(*sz);
        auto n = (*f)->pread(0, buf);
        if (!n || *n != *sz) {
            return std::unexpected(Error::make(ErrorCode::kIoError, "short read: " + path));
        }

        auto footer_tail = std::span<const std::byte>(buf).subspan(buf.size() - storage::kFileFooterBytes);
        auto footer = storage::read_and_verify_footer(footer_tail, kIdFileMagic);
        if (!footer) return std::unexpected(footer.error());
        auto body = std::span<const std::byte>(buf).first(buf.size() - storage::kFileFooterBytes);
        if (storage::crc32c(body) != footer->checksum) {
            return std::unexpected(
                Error::make(ErrorCode::kCorruption, "id file checksum mismatch: " + path));
        }

        std::uint32_t count;
        std::memcpy(&count, body.data(), 4);
        std::size_t off = 4;
        for (std::uint32_t i = 0; i < count; ++i) {
            std::uint64_t offset;
            std::memcpy(&offset, body.data() + off, 8); off += 8;
            std::uint32_t id_len;
            std::memcpy(&id_len, body.data() + off, 4); off += 4;
            std::string external_id(reinterpret_cast<const char*>(body.data() + off), id_len);
            off += id_len;
            pk.apply(external_id, offset, entry.id);
        }
    }
    return pk;
}

Result<void> replay_growing_segment_cf(PrimaryKeyIndex& pk, const std::string& rocksdb_dir) {
    rocksdb::DB* db = nullptr;
    rocksdb::Options options;
    auto status = rocksdb::DB::OpenForReadOnly(options, rocksdb_dir, &db);
    if (!status.ok()) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "replay_growing_segment_cf open failed: " + status.ToString()));
    }
    std::unique_ptr<rocksdb::DB> owned(db);

    std::unique_ptr<rocksdb::Iterator> it(owned->NewIterator(rocksdb::ReadOptions()));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string value_str = it->value().ToString();
        std::string id_str = it->key().ToString();
        auto decoded = decode_row(value_str, /*expected_dim=*/0);
        if (!decoded) return std::unexpected(corrupt_row(id_str));
        if (decoded->tombstone) {
            pk.apply_delete(id_str, decoded->offset);
        } else {
            pk.apply(id_str, decoded->offset, std::nullopt);
        }
    }
    if (!it->status().ok()) {
        return std::unexpected(Error::make(
            ErrorCode::kIoError, "replay_growing_segment_cf iteration failed: " + it->status().ToString()));
    }
    return {};
}

}  // namespace lattice::segment
