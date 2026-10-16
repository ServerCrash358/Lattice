#include "lattice/storage/storage.h"

#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string_view>

namespace lattice::storage {

// --- CRC32C (Castagnoli polynomial 0x1EDC6F41), table-based, byte-at-a-time.
// D-CPP-06 estimates ~30 lines for a hardware-intrinsic version; this is the
// portable software fallback (correct on every machine, not just SSE4.2
// ones) -- swap in _mm_crc32_u64 later as a measured perf optimization,
// same interface. ---
namespace {
// Built once at compile time: a lazily-filled global table guarded by a plain bool was a data
// race when two threads computed their first CRC concurrently.
constexpr std::array<std::uint32_t, 256> kCrc32cTable = [] {
    constexpr std::uint32_t poly = 0x82F63B78u;  // reflected 0x1EDC6F41
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (poly ^ (c >> 1)) : (c >> 1);
        t[i] = c;
    }
    return t;
}();
}  // namespace

std::uint32_t crc32c(std::span<const std::byte> data, std::uint32_t seed) noexcept {
    std::uint32_t c = seed ^ 0xFFFFFFFFu;
    for (std::byte b : data) {
        c = kCrc32cTable[(c ^ static_cast<std::uint8_t>(b)) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

// --- FileFooter: explicit byte packing, no reliance on struct layout ---

void write_footer(std::span<std::byte> out, const FileFooter& footer) noexcept {
    std::size_t off = 0;
    std::memcpy(out.data() + off, &footer.magic, 4); off += 4;
    std::memcpy(out.data() + off, &footer.format_version, 4); off += 4;
    std::memcpy(out.data() + off, &footer.checksum, 4); off += 4;
    std::memcpy(out.data() + off, &footer.row_count, 8);
}

Result<FileFooter> read_and_verify_footer(std::span<const std::byte> tail,
                                           std::uint32_t expected_magic) noexcept {
    if (tail.size() < kFileFooterBytes) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "footer too short"));
    }
    FileFooter f{};
    std::size_t off = 0;
    std::memcpy(&f.magic, tail.data() + off, 4); off += 4;
    std::memcpy(&f.format_version, tail.data() + off, 4); off += 4;
    std::memcpy(&f.checksum, tail.data() + off, 4); off += 4;
    std::memcpy(&f.row_count, tail.data() + off, 8);

    if (f.magic != expected_magic) {
        return std::unexpected(Error::make(ErrorCode::kCorruption,
                                            "magic mismatch: file is not the expected format"));
    }
    return f;
}

// --- PosixFile ---

Result<std::unique_ptr<PosixFile>> PosixFile::open(const std::string& path, bool create,
                                                   bool truncate) {
    int flags = O_RDWR | (create ? O_CREAT : 0) | (truncate ? O_TRUNC : 0);
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "open failed: " + path + ": " + std::strerror(errno)));
    }
    return std::unique_ptr<PosixFile>(new PosixFile(fd));
}

PosixFile::~PosixFile() {
    if (fd_ >= 0) ::close(fd_);
}

Result<std::size_t> PosixFile::pread(std::uint64_t offset, std::span<std::byte> out) {
    ssize_t n = ::pread(fd_, out.data(), out.size(), static_cast<off_t>(offset));
    if (n < 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, std::strerror(errno)));
    }
    return static_cast<std::size_t>(n);
}

Result<std::size_t> PosixFile::pwrite(std::uint64_t offset, std::span<const std::byte> data) {
    ssize_t n = ::pwrite(fd_, data.data(), data.size(), static_cast<off_t>(offset));
    if (n < 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, std::strerror(errno)));
    }
    return static_cast<std::size_t>(n);
}

Result<void> PosixFile::fsync() {
    if (::fsync(fd_) != 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, std::strerror(errno)));
    }
    return {};
}

Result<void> PosixFile::truncate(std::uint64_t size) {
    if (::ftruncate(fd_, static_cast<off_t>(size)) != 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, std::strerror(errno)));
    }
    return {};
}

Result<std::uint64_t> PosixFile::size() const {
    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, std::strerror(errno)));
    }
    return static_cast<std::uint64_t>(st.st_size);
}

// --- FaultInjectingFile / FaultInjectingFileSystem ---
// In-process, deterministic fault simulation (D-STO-06): decrements a
// shared op budget on every operation; once exhausted, the configured fault
// fires. kCrashBeforeFsync is intentionally NOT simulated here -- a true
// process-crash simulation needs a subprocess harness (fork + kill), which
// belongs in the dedicated crash-injection test runner built alongside
// `segment` (that runner is still outstanding -- see Phase 3 status notes).

FaultInjectingFile::FaultInjectingFile(std::unique_ptr<File> inner,
                                       std::shared_ptr<std::atomic<std::int64_t>> op_budget,
                                       FaultKind kind_on_exhaustion)
    : inner_(std::move(inner)), op_budget_(std::move(op_budget)), kind_(kind_on_exhaustion) {}

bool FaultInjectingFile::should_fail() noexcept {
    return op_budget_->fetch_sub(1) <= 0;
}

Result<std::size_t> FaultInjectingFile::pread(std::uint64_t offset, std::span<std::byte> out) {
    if (should_fail() && kind_ == FaultKind::kIoError) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "injected fault: pread"));
    }
    return inner_->pread(offset, out);
}

Result<std::size_t> FaultInjectingFile::pwrite(std::uint64_t offset, std::span<const std::byte> data) {
    if (should_fail()) {
        if (kind_ == FaultKind::kIoError) {
            return std::unexpected(Error::make(ErrorCode::kIoError, "injected fault: pwrite"));
        }
        if (kind_ == FaultKind::kShortWrite && !data.empty()) {
            auto half = data.subspan(0, data.size() / 2);
            return inner_->pwrite(offset, half);
        }
    }
    return inner_->pwrite(offset, data);
}

Result<void> FaultInjectingFile::fsync() {
    if (should_fail() && kind_ == FaultKind::kIoError) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "injected fault: fsync"));
    }
    return inner_->fsync();
}

Result<void> FaultInjectingFile::truncate(std::uint64_t size) { return inner_->truncate(size); }
Result<std::uint64_t> FaultInjectingFile::size() const { return inner_->size(); }

FaultInjectingFileSystem::FaultInjectingFileSystem(std::int64_t crash_after_n_ops, FaultKind kind)
    : budget_(std::make_shared<std::atomic<std::int64_t>>(crash_after_n_ops)), kind_(kind) {}

Result<std::unique_ptr<File>> FaultInjectingFileSystem::open(const std::string& path, bool create) {
    auto inner = PosixFile::open(path, create);
    if (!inner) return std::unexpected(inner.error());
    return std::unique_ptr<File>(
        new FaultInjectingFile(std::move(*inner), budget_, kind_));
}

// --- MappedFile ---

Result<MappedFile> MappedFile::open_readonly(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "open failed: " + path));
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected(Error::make(ErrorCode::kIoError, "fstat failed: " + path));
    }
    std::size_t len = static_cast<std::size_t>(st.st_size);
    void* addr = len > 0 ? ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
    ::close(fd);
    if (len > 0 && addr == MAP_FAILED) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "mmap failed: " + path));
    }
    return MappedFile(addr, len);
}

MappedFile::MappedFile(MappedFile&& other) noexcept : addr_(other.addr_), len_(other.len_) {
    other.addr_ = nullptr;
    other.len_ = 0;
}
MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (addr_) ::munmap(addr_, len_);
        addr_ = other.addr_;
        len_ = other.len_;
        other.addr_ = nullptr;
        other.len_ = 0;
    }
    return *this;
}
MappedFile::~MappedFile() {
    if (addr_) ::munmap(addr_, len_);
}

std::span<const std::byte> MappedFile::data() const noexcept {
    return {static_cast<const std::byte*>(addr_), len_};
}
void MappedFile::advise_random() const noexcept {
    if (addr_) ::madvise(addr_, len_, MADV_RANDOM);
}
void MappedFile::advise_willneed() const noexcept {
    if (addr_) ::madvise(addr_, len_, MADV_WILLNEED);
}

// --- AsyncNodeFile (io_uring, D-STO-04) ------------------------------------

namespace {
// One io_uring ring per thread, shared across every AsyncNodeFile that thread
// reads (a ring is not bound to a file descriptor; the fd is given per SQE).
// A single ring shared across threads is NOT safe for the submit+wait
// pattern used below, which would break concurrent queries on a
// disk-resident index (D-CON-02: lock-free concurrent reads).
struct ThreadRing {
    io_uring ring{};
    bool ok = false;
    ThreadRing() { ok = io_uring_queue_init(/*entries=*/64, &ring, /*flags=*/0) == 0; }
    ~ThreadRing() {
        if (ok) io_uring_queue_exit(&ring);
    }
};
ThreadRing& thread_ring() {
    thread_local ThreadRing r;
    return r;
}
}  // namespace

Result<AsyncNodeFile> AsyncNodeFile::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "open failed: " + path));
    }
    if (!thread_ring().ok) {
        ::close(fd);
        return std::unexpected(Error::make(ErrorCode::kIoError, "io_uring_queue_init failed"));
    }
    return AsyncNodeFile(fd);
}

AsyncNodeFile::AsyncNodeFile(AsyncNodeFile&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
AsyncNodeFile& AsyncNodeFile::operator=(AsyncNodeFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}
AsyncNodeFile::~AsyncNodeFile() {
    if (fd_ >= 0) ::close(fd_);
}

Result<void> AsyncNodeFile::read_at(std::uint64_t offset, std::span<std::byte> out) {
    ThreadRing& tr = thread_ring();
    if (!tr.ok) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "io_uring_queue_init failed"));
    }
    io_uring* ring = &tr.ring;
    io_uring_sqe* sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "io_uring_get_sqe returned null"));
    }
    io_uring_prep_read(sqe, fd_, out.data(), static_cast<unsigned>(out.size()),
                        static_cast<__u64>(offset));

    int submitted = io_uring_submit(ring);
    if (submitted < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "io_uring_submit failed: " + std::to_string(-submitted)));
    }

    io_uring_cqe* cqe = nullptr;
    int rc = io_uring_wait_cqe(ring, &cqe);
    if (rc < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "io_uring_wait_cqe failed: " + std::to_string(-rc)));
    }
    int result = cqe->res;
    io_uring_cqe_seen(ring, cqe);

    if (result < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "async read failed: " + std::to_string(-result)));
    }
    if (static_cast<std::size_t>(result) != out.size()) {
        return std::unexpected(Error::make(ErrorCode::kIoError,
                                            "short async read: got " + std::to_string(result) +
                                                " of " + std::to_string(out.size()) + " bytes"));
    }
    return {};
}

// --- Wal: segment-lifecycle events only (D-STO-01/R3) ---
// Record layout on disk: [u8 type][u32 segment][u64 lsn][u32 crc32c] = 17 bytes.
namespace {
constexpr std::size_t kWalRecordBytes = 1 + 4 + 8 + 4;
}

Result<Wal> Wal::open(std::unique_ptr<File> file) {
    Wal wal(std::move(file));
    auto sz = wal.file_->size();
    if (!sz) return std::unexpected(sz.error());
    // Replay finds the valid prefix (it stops at the first torn/corrupt record). Appending after
    // a torn tail would put new records behind garbage where replay can never reach them, so
    // cut the file back to the valid prefix first.
    auto records = wal.replay_since(0);
    if (!records) return std::unexpected(records.error());
    const std::uint64_t valid_end = records->size() * kWalRecordBytes;
    if (valid_end < *sz) {
        auto t = wal.file_->truncate(valid_end);
        if (!t) return std::unexpected(t.error());
        auto s = wal.file_->fsync();
        if (!s) return std::unexpected(s.error());
    }
    wal.write_offset_ = valid_end;
    for (auto& r : *records) wal.next_lsn_ = std::max(wal.next_lsn_, r.lsn + 1);
    return wal;
}

Result<Lsn> Wal::append(const WalRecord& rec, SyncMode mode) {
    std::vector<std::byte> buf(kWalRecordBytes);
    std::size_t off = 0;
    auto type_byte = static_cast<std::uint8_t>(rec.type);
    std::memcpy(buf.data() + off, &type_byte, 1); off += 1;
    auto seg = static_cast<std::uint32_t>(rec.segment);
    std::memcpy(buf.data() + off, &seg, 4); off += 4;
    std::memcpy(buf.data() + off, &rec.lsn, 8); off += 8;
    std::uint32_t crc = crc32c(std::span<const std::byte>(buf.data(), off));
    std::memcpy(buf.data() + off, &crc, 4);

    auto n = file_->pwrite(write_offset_, buf);
    if (!n) return std::unexpected(n.error());
    if (*n != buf.size()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "short write to WAL"));
    }
    write_offset_ += buf.size();

    if (mode != SyncMode::kNone) {
        auto s = file_->fsync();
        if (!s) return std::unexpected(s.error());
    }
    return rec.lsn;
}

Result<std::vector<WalRecord>> Wal::replay_since(Lsn checkpoint) {
    auto total_size = file_->size();
    if (!total_size) return std::unexpected(total_size.error());

    std::vector<WalRecord> out;
    std::uint64_t offset = 0;
    while (offset + kWalRecordBytes <= *total_size) {
        std::vector<std::byte> buf(kWalRecordBytes);
        auto n = file_->pread(offset, buf);
        if (!n || *n != kWalRecordBytes) break;  // truncated tail -- stop, don't error (torn write)

        std::uint8_t type_byte;
        std::uint32_t seg, crc_stored;
        std::uint64_t lsn;
        std::size_t off = 0;
        std::memcpy(&type_byte, buf.data() + off, 1); off += 1;
        std::memcpy(&seg, buf.data() + off, 4); off += 4;
        std::memcpy(&lsn, buf.data() + off, 8); off += 8;
        std::memcpy(&crc_stored, buf.data() + off, 4);

        std::uint32_t crc_computed = crc32c(std::span<const std::byte>(buf.data(), off));
        if (crc_computed != crc_stored) break;  // torn/corrupt record -- stop replay here, don't error

        if (lsn >= checkpoint) {
            out.push_back(WalRecord{static_cast<WalEventType>(type_byte),
                                     static_cast<SegmentId>(seg), lsn});
        }
        offset += kWalRecordBytes;
    }
    return out;
}

// --- Manifest ---
// On-disk: one file per generation, "MANIFEST-<n>", plus a "CURRENT" file
// naming the live one. Format: [u32 count][entries...][footer].
// Each entry: [u32 id][u32 cf_name_len][cf_name bytes][u64 max_seq].

namespace {
constexpr std::uint32_t kManifestMagic = 0x4C415430;  // 'LAT0'
constexpr std::string_view kManifestPrefix = "MANIFEST-";
constexpr std::uint64_t kMaxCurrentFileBytes = 64;

// Highest MANIFEST-<n> generation present in `dir` (0 if none). The next manifest is always this
// plus one, so a new process never reuses a generation -- and never overwrites a live manifest.
Result<std::uint64_t> max_manifest_generation(const std::string& dir) {
    std::uint64_t best = 0;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "cannot list manifest dir " + dir + ": " + ec.message()));
    }
    for (const auto& entry : it) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(kManifestPrefix, 0) != 0) continue;
        std::string_view digits = std::string_view(name).substr(kManifestPrefix.size());
        if (digits.empty() || digits.size() > 19) continue;
        std::uint64_t g = 0;
        bool numeric = true;
        for (char ch : digits) {
            if (ch < '0' || ch > '9') { numeric = false; break; }
            g = g * 10 + static_cast<std::uint64_t>(ch - '0');
        }
        if (numeric) best = std::max(best, g);
    }
    return best;
}

Result<void> fsync_dir(const std::string& dir) {
    int dir_fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dir_fd < 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "open dir for fsync failed: " + dir + ": " + std::strerror(errno)));
    }
    int rc = ::fsync(dir_fd);
    int saved_errno = errno;
    ::close(dir_fd);
    if (rc != 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, "fsync dir failed: " + dir + ": " + std::strerror(saved_errno)));
    }
    return {};
}
}  // namespace

Result<Manifest> Manifest::load_current(const std::string& dir) {
    const std::string current_path = dir + "/CURRENT";
    Manifest m;
    std::error_code ec;
    if (!std::filesystem::exists(current_path, ec)) {
        if (ec) {
            return std::unexpected(
                Error::make(ErrorCode::kIoError, "cannot stat " + current_path + ": " + ec.message()));
        }
        return m;  // fresh directory: no manifest yet
    }

    auto current_file = PosixFile::open(current_path, false);
    if (!current_file) return std::unexpected(current_file.error());
    auto sz = (*current_file)->size();
    if (!sz) return std::unexpected(sz.error());
    if (*sz == 0 || *sz > kMaxCurrentFileBytes) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "CURRENT is empty or malformed"));
    }
    std::vector<std::byte> name_buf(*sz);
    auto n = (*current_file)->pread(0, name_buf);
    if (!n) return std::unexpected(n.error());
    if (*n != *sz) return std::unexpected(Error::make(ErrorCode::kIoError, "short read of CURRENT"));
    std::string manifest_name(reinterpret_cast<const char*>(name_buf.data()), *n);
    if (manifest_name.rfind(kManifestPrefix, 0) != 0 || manifest_name.find('/') != std::string::npos) {
        return std::unexpected(
            Error::make(ErrorCode::kCorruption, "CURRENT does not name a manifest: " + manifest_name));
    }

    auto mf = PosixFile::open(dir + "/" + manifest_name, false);
    if (!mf) {
        return std::unexpected(
            Error::make(ErrorCode::kCorruption, "CURRENT names a missing manifest: " + manifest_name));
    }
    auto msz = (*mf)->size();
    if (!msz) return std::unexpected(msz.error());
    if (*msz < kFileFooterBytes + 4) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "manifest too short: " + manifest_name));
    }
    std::vector<std::byte> buf(*msz);
    auto rn = (*mf)->pread(0, buf);
    if (!rn) return std::unexpected(rn.error());
    if (*rn != *msz) return std::unexpected(Error::make(ErrorCode::kIoError, "short read: " + manifest_name));

    auto footer = read_and_verify_footer(std::span<const std::byte>(buf).subspan(buf.size() - kFileFooterBytes),
                                         kManifestMagic);
    if (!footer) return std::unexpected(footer.error());
    auto body = std::span<const std::byte>(buf).first(buf.size() - kFileFooterBytes);
    if (crc32c(body) != footer->checksum) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "manifest checksum mismatch: " + manifest_name));
    }

    std::size_t off = 0;
    auto take = [&](void* dst, std::size_t bytes) -> bool {
        if (off + bytes > body.size()) return false;
        std::memcpy(dst, body.data() + off, bytes);
        off += bytes;
        return true;
    };
    auto corrupt = [&] {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "malformed manifest: " + manifest_name));
    };
    std::uint32_t count = 0;
    if (!take(&count, 4) || count != footer->row_count) return corrupt();
    for (std::uint32_t i = 0; i < count; ++i) {
        SealedSegmentEntry e{};
        std::uint32_t id = 0, name_len = 0;
        if (!take(&id, 4) || !take(&name_len, 4) || off + name_len > body.size()) return corrupt();
        e.id = static_cast<SegmentId>(id);
        e.source_rocksdb_cf.assign(reinterpret_cast<const char*>(body.data() + off), name_len);
        off += name_len;
        if (!take(&e.max_rocksdb_seq, 8)) return corrupt();
        m.segments_.push_back(std::move(e));
    }
    if (off != body.size()) return corrupt();
    return m;
}

Result<void> Manifest::atomic_swap(const std::string& dir, std::vector<SealedSegmentEntry> new_list) {
    std::vector<std::byte> buf;
    std::uint32_t count = static_cast<std::uint32_t>(new_list.size());
    buf.resize(4);
    std::memcpy(buf.data(), &count, 4);
    for (auto& e : new_list) {
        std::size_t base = buf.size();
        buf.resize(base + 4 + 4 + e.source_rocksdb_cf.size() + 8);
        std::uint32_t id = static_cast<std::uint32_t>(e.id);
        std::memcpy(buf.data() + base, &id, 4);
        std::uint32_t name_len = static_cast<std::uint32_t>(e.source_rocksdb_cf.size());
        std::memcpy(buf.data() + base + 4, &name_len, 4);
        std::memcpy(buf.data() + base + 8, e.source_rocksdb_cf.data(), name_len);
        std::memcpy(buf.data() + base + 8 + name_len, &e.max_rocksdb_seq, 8);
    }

    FileFooter footer{kManifestMagic, 1, 0, count};
    footer.checksum = crc32c(buf);
    std::vector<std::byte> footer_bytes(kFileFooterBytes);
    write_footer(footer_bytes, footer);
    buf.insert(buf.end(), footer_bytes.begin(), footer_bytes.end());

    auto max_gen = max_manifest_generation(dir);
    if (!max_gen) return std::unexpected(max_gen.error());
    const std::string manifest_name = std::string(kManifestPrefix) + std::to_string(*max_gen + 1);
    const std::string manifest_path = dir + "/" + manifest_name;

    auto mf = PosixFile::open(manifest_path, /*create=*/true, /*truncate=*/true);
    if (!mf) return std::unexpected(mf.error());
    auto w = (*mf)->pwrite(0, buf);
    if (!w || *w != buf.size()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "short write to manifest"));
    }
    auto s1 = (*mf)->fsync();
    if (!s1) return std::unexpected(s1.error());

    // write-temp + fsync + rename for CURRENT (atomic pointer swap). Truncate: a CURRENT.tmp left
    // by an earlier crash may hold a longer name, which would otherwise survive as a stale tail.
    const std::string tmp_path = dir + "/CURRENT.tmp";
    auto tmp = PosixFile::open(tmp_path, /*create=*/true, /*truncate=*/true);
    if (!tmp) return std::unexpected(tmp.error());
    std::vector<std::byte> name_bytes(manifest_name.size());
    std::memcpy(name_bytes.data(), manifest_name.data(), manifest_name.size());
    auto w2 = (*tmp)->pwrite(0, name_bytes);
    if (!w2 || *w2 != name_bytes.size()) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "short write to CURRENT.tmp"));
    }
    auto s2 = (*tmp)->fsync();
    if (!s2) return std::unexpected(s2.error());
    tmp->reset();  // close before rename

    if (::rename(tmp_path.c_str(), (dir + "/CURRENT").c_str()) != 0) {
        return std::unexpected(
            Error::make(ErrorCode::kIoError, std::string("rename CURRENT failed: ") + std::strerror(errno)));
    }
    // The rename is only durable once the directory itself is fsynced.
    auto ds = fsync_dir(dir);
    if (!ds) return std::unexpected(ds.error());

    segments_ = std::move(new_list);
    return {};
}

}  // namespace lattice::storage
