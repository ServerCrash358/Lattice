#pragma once
// storage: file abstraction (incl. fault injection), mmap, the Lattice WAL
// (segment-lifecycle events ONLY -- D-STO-01/R3), manifest, checksums,
// format versioning (D-STO-07). Depends only on `common`. Row-level
// durability is RocksDB's job (owned by `segment`); this module never
// stores a row.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lattice/common/error.h"
#include "lattice/common/types.h"

namespace lattice::storage {

// --- CRC32C (D-CPP-06: hand-rolled, ~30 lines) ---
std::uint32_t crc32c(std::span<const std::byte> data, std::uint32_t seed = 0) noexcept;

// --- Format versioning (D-STO-07) ---
struct FileFooter {
    std::uint32_t magic;
    std::uint32_t format_version;
    std::uint32_t checksum;
    std::uint64_t row_count;
};
constexpr std::size_t kFileFooterBytes = 20;  // 4+4+4+8, packed explicitly (no struct padding relied on)

void write_footer(std::span<std::byte> out, const FileFooter& footer) noexcept;
Result<FileFooter> read_and_verify_footer(std::span<const std::byte> tail,
                                           std::uint32_t expected_magic) noexcept;

// --- Fault-injectable file abstraction (D-STO-06) ---
class File {
public:
    virtual ~File() = default;
    virtual Result<std::size_t> pread(std::uint64_t offset, std::span<std::byte> out) = 0;
    virtual Result<std::size_t> pwrite(std::uint64_t offset, std::span<const std::byte> data) = 0;
    virtual Result<void> fsync() = 0;
    virtual Result<void> truncate(std::uint64_t size) = 0;
    virtual Result<std::uint64_t> size() const = 0;
};

// Real, POSIX-backed File implementation.
class PosixFile final : public File {
public:
    // `truncate`: discard existing contents (O_TRUNC). Use it whenever a file is being
    // rewritten from offset 0, otherwise a shorter rewrite leaves a stale tail behind.
    static Result<std::unique_ptr<PosixFile>> open(const std::string& path, bool create,
                                                   bool truncate = false);
    ~PosixFile() override;

    Result<std::size_t> pread(std::uint64_t offset, std::span<std::byte> out) override;
    Result<std::size_t> pwrite(std::uint64_t offset, std::span<const std::byte> data) override;
    Result<void> fsync() override;
    Result<void> truncate(std::uint64_t size) override;
    Result<std::uint64_t> size() const override;

private:
    explicit PosixFile(int fd) noexcept : fd_(fd) {}
    int fd_;
};

// Deterministic fault injection: wraps a real (or in-memory) File and can be
// told to fail the Nth operation of a given kind. Used by both the
// crash-injection harness for Lattice's own files AND (via a RocksDB
// FileSystem adapter, built in `segment`) for RocksDB's writes (finding 12).
enum class FaultKind { kNone, kShortWrite, kIoError, kCrashBeforeFsync };

class FaultInjectingFile final : public File {
public:
    FaultInjectingFile(std::unique_ptr<File> inner, std::shared_ptr<std::atomic<std::int64_t>> op_budget,
                        FaultKind kind_on_exhaustion);

    Result<std::size_t> pread(std::uint64_t offset, std::span<std::byte> out) override;
    Result<std::size_t> pwrite(std::uint64_t offset, std::span<const std::byte> data) override;
    Result<void> fsync() override;
    Result<void> truncate(std::uint64_t size) override;
    Result<std::uint64_t> size() const override;

private:
    bool should_fail() noexcept;
    std::unique_ptr<File> inner_;
    std::shared_ptr<std::atomic<std::int64_t>> op_budget_;  // decremented per op; fault once <= 0
    FaultKind kind_;
};

// Directory-wide fault injector: `remaining_ops` counts down across every
// operation on every file opened through it, so a crash-injection test can
// say "kill the process after exactly N filesystem operations" and get a
// reproducible, seed-driven crash point (D-STO-06's harness).
class FaultInjectingFileSystem {
public:
    explicit FaultInjectingFileSystem(std::int64_t crash_after_n_ops, FaultKind kind = FaultKind::kIoError);
    Result<std::unique_ptr<File>> open(const std::string& path, bool create);
    std::int64_t remaining_ops() const noexcept { return budget_->load(); }

    // Exposes the shared countdown + fault kind so a caller (segment, which
    // is the only module allowed to depend on RocksDB -- see module-
    // storage.md's "segment wires this in") can build a second adapter over
    // rocksdb::FileSystem sharing the SAME budget: "one fault-injection
    // engine, two adapters, so a single seeded crash scenario exercises
    // both Lattice's files and RocksDB's simultaneously" (finding 12).
    std::shared_ptr<std::atomic<std::int64_t>> shared_budget() const noexcept { return budget_; }
    FaultKind kind() const noexcept { return kind_; }

private:
    std::shared_ptr<std::atomic<std::int64_t>> budget_;
    FaultKind kind_;
};

// --- mmap for sealed, immutable files (D-STO-04) ---
class MappedFile {
public:
    static Result<MappedFile> open_readonly(const std::string& path);
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();

    std::span<const std::byte> data() const noexcept;
    void advise_random() const noexcept;
    void advise_willneed() const noexcept;

private:
    MappedFile(void* addr, std::size_t len) noexcept : addr_(addr), len_(len) {}
    void* addr_ = nullptr;
    std::size_t len_ = 0;
};

// --- io_uring async reads for the disk-resident Vamana backend (D-STO-04) ---
// Phase 3 scoping (disclosed): buffered reads via IORING_OP_READ, one
// blocking submit+wait per call. True O_DIRECT (needs aligned buffers/
// offsets matching the filesystem's logical block size) and real
// multi-request batching (submitting several reads before waiting on any,
// matching beam search's fan-out) are documented Phase 5 performance
// follow-ups -- this gets real io_uring usage working correctly first.
class AsyncNodeFile {
public:
    static Result<AsyncNodeFile> open(const std::string& path);
    AsyncNodeFile(AsyncNodeFile&&) noexcept;
    AsyncNodeFile& operator=(AsyncNodeFile&&) noexcept;
    AsyncNodeFile(const AsyncNodeFile&) = delete;
    AsyncNodeFile& operator=(const AsyncNodeFile&) = delete;
    ~AsyncNodeFile();

    // Reads exactly out.size() bytes starting at `offset` into `out`.
    Result<void> read_at(std::uint64_t offset, std::span<std::byte> out);

private:
    explicit AsyncNodeFile(int fd) noexcept : fd_(fd) {}
    int fd_ = -1;
};

// --- Lattice WAL: segment-lifecycle events ONLY (D-STO-01, R3) ---
enum class WalEventType : std::uint8_t {
    kSealBegin = 1,
    kSealComplete = 2,
    kMergeBegin = 3,
    kMergeComplete = 4,
    kManifestSwap = 5,
};

struct WalRecord {
    WalEventType type;
    SegmentId segment;
    Lsn lsn;
};

class Wal {
public:
    enum class SyncMode { kNone, kGroup, kEveryWrite };  // independent of RocksDB's own sync_mode

    static Result<Wal> open(std::unique_ptr<File> file);
    Result<Lsn> append(const WalRecord& rec, SyncMode mode);
    Result<std::vector<WalRecord>> replay_since(Lsn checkpoint);

private:
    explicit Wal(std::unique_ptr<File> file) : file_(std::move(file)) {}
    std::unique_ptr<File> file_;
    std::uint64_t write_offset_ = 0;
    Lsn next_lsn_ = 1;
};

// --- Manifest: atomic segment list + seal provenance (finding 11) ---
struct SealedSegmentEntry {
    SegmentId id;
    std::string source_rocksdb_cf;   // names the CF this segment was sealed from
    std::uint64_t max_rocksdb_seq;   // CF-drop is a no-op once this is already reflected
};

class Manifest {
public:
    // Empty manifest only when `dir` has no CURRENT file (a fresh directory). If CURRENT exists
    // but the manifest it names is missing, short, or fails its magic/CRC check, this returns
    // an error: treating that as "empty" would let the next atomic_swap drop every live segment.
    static Result<Manifest> load_current(const std::string& dir);

    // Writes the list as a brand-new MANIFEST-<n> (n = highest generation on disk + 1, so a
    // restart never reuses or overwrites a live manifest), then atomically repoints CURRENT.
    Result<void> atomic_swap(const std::string& dir, std::vector<SealedSegmentEntry> new_list);
    const std::vector<SealedSegmentEntry>& live_segments() const noexcept { return segments_; }

private:
    std::vector<SealedSegmentEntry> segments_;
};

}  // namespace lattice::storage
