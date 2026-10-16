#include "lattice/storage/storage.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

using namespace lattice::storage;
using lattice::SegmentId;

namespace {
// Isolated temp directory per test, cleaned up on destruction.
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/lattice_storage_test_XXXXXX";
        char* p = mkdtemp(tmpl);
        path_ = p ? p : "/tmp";
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::vector<std::byte> to_bytes(const std::string& s) {
    std::vector<std::byte> out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}
}  // namespace

TEST(Crc32c, MatchesKnownTestVector) {
    // The standard CRC32C ("Castagnoli") check value for ASCII "123456789".
    auto bytes = to_bytes("123456789");
    EXPECT_EQ(crc32c(bytes), 0xE3069283u);
}

TEST(Crc32c, EmptyInputIsZero) {
    EXPECT_EQ(crc32c({}), 0u);
}

TEST(FileFooter, RoundTripsAndDetectsCorruption) {
    FileFooter footer{0x12345678, 1, 0xABCDEF01, 42};
    std::vector<std::byte> buf(kFileFooterBytes);
    write_footer(buf, footer);

    auto read = read_and_verify_footer(buf, 0x12345678);
    ASSERT_TRUE(read.has_value());
    EXPECT_EQ(read->format_version, 1u);
    EXPECT_EQ(read->checksum, 0xABCDEF01u);
    EXPECT_EQ(read->row_count, 42u);

    auto wrong_magic = read_and_verify_footer(buf, 0xDEADBEEF);
    EXPECT_FALSE(wrong_magic.has_value());

    auto too_short = read_and_verify_footer(std::span<const std::byte>(buf).subspan(0, 5), 0x12345678);
    EXPECT_FALSE(too_short.has_value());
}

TEST(PosixFile, WriteReadFsyncTruncateRoundTrip) {
    TempDir dir;
    std::string path = dir.path() + "/test.dat";
    auto f = PosixFile::open(path, true);
    ASSERT_TRUE(f.has_value());

    auto data = to_bytes("hello, lattice");
    auto w = (*f)->pwrite(0, data);
    ASSERT_TRUE(w.has_value());
    EXPECT_EQ(*w, data.size());
    EXPECT_TRUE((*f)->fsync().has_value());

    auto sz = (*f)->size();
    ASSERT_TRUE(sz.has_value());
    EXPECT_EQ(*sz, data.size());

    std::vector<std::byte> read_buf(data.size());
    auto r = (*f)->pread(0, read_buf);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(read_buf, data);

    ASSERT_TRUE((*f)->truncate(5).has_value());
    auto sz2 = (*f)->size();
    ASSERT_TRUE(sz2.has_value());
    EXPECT_EQ(*sz2, 5u);
}

TEST(FaultInjectingFileSystem, IoErrorFiresAfterBudgetExhausted) {
    TempDir dir;
    FaultInjectingFileSystem fs(/*crash_after_n_ops=*/2, FaultKind::kIoError);

    auto f = fs.open(dir.path() + "/fault.dat", true);
    ASSERT_TRUE(f.has_value());
    auto data = to_bytes("x");

    // First 2 ops succeed (budget was 2), 3rd fails.
    EXPECT_TRUE((*f)->pwrite(0, data).has_value());
    EXPECT_TRUE((*f)->fsync().has_value());
    auto third = (*f)->pwrite(0, data);
    EXPECT_FALSE(third.has_value());
}

TEST(FaultInjectingFileSystem, ShortWriteTornAtBudgetExhaustion) {
    TempDir dir;
    FaultInjectingFileSystem fs(1, FaultKind::kShortWrite);
    auto f = fs.open(dir.path() + "/short.dat", true);
    ASSERT_TRUE(f.has_value());

    auto data = to_bytes("0123456789");  // 10 bytes
    EXPECT_TRUE((*f)->pwrite(0, data).has_value());  // op 1: consumes budget, ok
    auto second = (*f)->pwrite(0, data);              // op 2: torn (half written)
    ASSERT_TRUE(second.has_value());
    EXPECT_LT(*second, data.size());
}

TEST(Wal, AppendAndReplayPreservesOrderAndContent) {
    TempDir dir;
    auto file = PosixFile::open(dir.path() + "/wal.log", true);
    ASSERT_TRUE(file.has_value());
    auto wal = Wal::open(std::move(*file));
    ASSERT_TRUE(wal.has_value());

    auto lsn1 = wal->append({WalEventType::kSealBegin, static_cast<SegmentId>(1), 1}, Wal::SyncMode::kEveryWrite);
    auto lsn2 = wal->append({WalEventType::kSealComplete, static_cast<SegmentId>(1), 2}, Wal::SyncMode::kEveryWrite);
    auto lsn3 = wal->append({WalEventType::kManifestSwap, static_cast<SegmentId>(1), 3}, Wal::SyncMode::kEveryWrite);
    ASSERT_TRUE(lsn1.has_value() && lsn2.has_value() && lsn3.has_value());

    auto records = wal->replay_since(0);
    ASSERT_TRUE(records.has_value());
    ASSERT_EQ(records->size(), 3u);
    EXPECT_EQ((*records)[0].type, WalEventType::kSealBegin);
    EXPECT_EQ((*records)[1].type, WalEventType::kSealComplete);
    EXPECT_EQ((*records)[2].type, WalEventType::kManifestSwap);
    EXPECT_EQ((*records)[2].lsn, 3u);

    auto since2 = wal->replay_since(2);
    ASSERT_TRUE(since2.has_value());
    EXPECT_EQ(since2->size(), 2u);  // lsn 2 and 3
}

TEST(Wal, TornTailRecordStopsReplayWithoutError) {
    TempDir dir;
    std::string path = dir.path() + "/wal_torn.log";
    {
        auto file = PosixFile::open(path, true);
        ASSERT_TRUE(file.has_value());
        auto wal = Wal::open(std::move(*file));
        ASSERT_TRUE(wal.has_value());
        wal->append({WalEventType::kSealBegin, static_cast<SegmentId>(1), 1}, Wal::SyncMode::kEveryWrite);
        wal->append({WalEventType::kSealComplete, static_cast<SegmentId>(1), 2}, Wal::SyncMode::kEveryWrite);
    }
    // Simulate a torn write: truncate mid-way through the second record.
    auto raw = PosixFile::open(path, false);
    ASSERT_TRUE(raw.has_value());
    auto sz = (*raw)->size();
    ASSERT_TRUE(sz.has_value());
    ASSERT_TRUE((*raw)->truncate(*sz - 3).has_value());

    auto file2 = PosixFile::open(path, false);
    ASSERT_TRUE(file2.has_value());
    auto wal2 = Wal::open(std::move(*file2));
    ASSERT_TRUE(wal2.has_value());
    auto records = wal2->replay_since(0);
    ASSERT_TRUE(records.has_value());
    // Only the first, intact record survives; the torn second is dropped,
    // not treated as a fatal error.
    EXPECT_EQ(records->size(), 1u);
}

TEST(Manifest, AtomicSwapAndLoadRoundTrips) {
    TempDir dir;
    Manifest m;
    std::vector<SealedSegmentEntry> entries = {
        {static_cast<SegmentId>(1), "cf-1", 100},
        {static_cast<SegmentId>(2), "cf-2", 200},
    };
    auto r = m.atomic_swap(dir.path(), entries);
    ASSERT_TRUE(r.has_value()) << r.error().message;

    auto loaded = Manifest::load_current(dir.path());
    ASSERT_TRUE(loaded.has_value());
    ASSERT_EQ(loaded->live_segments().size(), 2u);
    EXPECT_EQ(loaded->live_segments()[0].source_rocksdb_cf, "cf-1");
    EXPECT_EQ(loaded->live_segments()[0].max_rocksdb_seq, 100u);
    EXPECT_EQ(loaded->live_segments()[1].source_rocksdb_cf, "cf-2");
    EXPECT_EQ(loaded->live_segments()[1].max_rocksdb_seq, 200u);
}

TEST(Manifest, LoadCurrentOnFreshDirIsEmptyNotError) {
    TempDir dir;
    auto loaded = Manifest::load_current(dir.path());
    ASSERT_TRUE(loaded.has_value());
    EXPECT_TRUE(loaded->live_segments().empty());
}

namespace {
std::string read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string s;
    char buf[256];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}
void write_file(const std::string& path, const std::string& content) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
}
}  // namespace

// A new process used to restart its generation counter at 1 and overwrite MANIFEST-1 in place,
// even when MANIFEST-1 was the live manifest. The next generation must come from disk.
TEST(Manifest, NextGenerationComesFromDiskNotFromProcessState) {
    TempDir dir;
    Manifest first;
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(first.atomic_swap(dir.path(), {{static_cast<SegmentId>(i), "cf", 1}}).has_value());
    }
    std::string live = read_file(dir.path() + "/CURRENT");
    std::string live_contents = read_file(dir.path() + "/" + live);

    Manifest fresh;  // models a restarted process
    ASSERT_TRUE(fresh.atomic_swap(dir.path(), {{static_cast<SegmentId>(9), "cf-new", 2}}).has_value());
    std::string now = read_file(dir.path() + "/CURRENT");
    EXPECT_NE(now, live);
    EXPECT_EQ(read_file(dir.path() + "/" + live), live_contents) << "previous live manifest was modified";
    auto loaded = Manifest::load_current(dir.path());
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    ASSERT_EQ(loaded->live_segments().size(), 1u);
    EXPECT_EQ(loaded->live_segments()[0].source_rocksdb_cf, "cf-new");
}

// A CURRENT.tmp left over from a crash can hold a longer name; without truncation the new
// name was written over its prefix and CURRENT ended up naming an old manifest.
TEST(Manifest, StaleLongerCurrentTmpDoesNotCorruptCurrent) {
    TempDir dir;
    write_file(dir.path() + "/CURRENT.tmp", "MANIFEST-123456789");
    Manifest m;
    ASSERT_TRUE(m.atomic_swap(dir.path(), {{static_cast<SegmentId>(1), "cf-1", 1}}).has_value());
    EXPECT_EQ(read_file(dir.path() + "/CURRENT"), "MANIFEST-1");
    auto loaded = Manifest::load_current(dir.path());
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    EXPECT_EQ(loaded->live_segments().size(), 1u);
}

// A corrupt or missing live manifest must be an error, never "empty": an empty result let the
// next commit_seal write a manifest that dropped every live segment.
TEST(Manifest, CorruptOrMissingLiveManifestIsAnErrorNotEmpty) {
    TempDir dir;
    Manifest m;
    ASSERT_TRUE(m.atomic_swap(dir.path(), {{static_cast<SegmentId>(1), "cf-1", 1}, {static_cast<SegmentId>(2), "cf-2", 2}})
                    .has_value());
    std::string live_path = dir.path() + "/" + read_file(dir.path() + "/CURRENT");

    std::string bytes = read_file(live_path);
    bytes[6] = static_cast<char>(bytes[6] ^ 0x5A);  // flip bits inside an entry, footer intact
    write_file(live_path, bytes);
    auto corrupt = Manifest::load_current(dir.path());
    ASSERT_FALSE(corrupt.has_value());
    EXPECT_EQ(corrupt.error().code, lattice::ErrorCode::kCorruption);

    std::filesystem::remove(live_path);
    auto missing = Manifest::load_current(dir.path());
    ASSERT_FALSE(missing.has_value());
}

TEST(Crc32c, ConcurrentFirstUseIsConsistent) {
    auto bytes = to_bytes("123456789");
    std::vector<std::thread> ts;
    std::atomic<int> bad{0};
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&] {
            for (int i = 0; i < 1000; ++i)
                if (crc32c(bytes) != 0xE3069283u) ++bad;
        });
    for (auto& th : ts) th.join();
    EXPECT_EQ(bad.load(), 0);
}

// After a torn tail, new records used to be appended after the garbage, where replay (which
// stops at the first bad record) could never see them.
TEST(Wal, AppendAfterTornTailIsReplayed) {
    TempDir dir;
    std::string path = dir.path() + "/wal_torn_append.log";
    {
        auto file = PosixFile::open(path, true);
        auto wal = Wal::open(std::move(*file));
        ASSERT_TRUE(wal.has_value());
        ASSERT_TRUE(wal->append({WalEventType::kSealBegin, static_cast<SegmentId>(1), 1}, Wal::SyncMode::kEveryWrite));
        ASSERT_TRUE(wal->append({WalEventType::kSealComplete, static_cast<SegmentId>(1), 2}, Wal::SyncMode::kEveryWrite));
    }
    {
        auto raw = PosixFile::open(path, false);
        auto sz = (*raw)->size();
        ASSERT_TRUE((*raw)->truncate(*sz - 3).has_value());  // tear the second record
    }
    {
        auto file = PosixFile::open(path, false);
        auto wal = Wal::open(std::move(*file));
        ASSERT_TRUE(wal.has_value());
        ASSERT_TRUE(wal->append({WalEventType::kManifestSwap, static_cast<SegmentId>(1), 3}, Wal::SyncMode::kEveryWrite));
    }
    auto file = PosixFile::open(path, false);
    auto wal = Wal::open(std::move(*file));
    ASSERT_TRUE(wal.has_value());
    auto records = wal->replay_since(0);
    ASSERT_TRUE(records.has_value());
    ASSERT_EQ(records->size(), 2u);
    EXPECT_EQ((*records)[0].lsn, 1u);
    EXPECT_EQ((*records)[1].lsn, 3u);
    EXPECT_EQ((*records)[1].type, WalEventType::kManifestSwap);
}
