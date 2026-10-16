// Phase 3 exit gate item 1 (module-storage.md finding 12): "the RocksDB-
// adapter path" of the crash-injection gate had no implementation at all
// until rocksdb_fault_injection.h/.cpp -- this proves it actually works:
// RocksDB's own writes (not just Lattice's own File/WAL/manifest) can be
// fault-injected through the SAME engine, and RocksDB survives it (no
// corruption, no torn state) exactly the way D-STO-06's kill-loop harness
// requires of the Lattice-file path.
#include "lattice/segment/rocksdb_fault_injection.h"

#include <gtest/gtest.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include <cstdlib>
#include <filesystem>
#include <memory>

using namespace lattice;
using namespace lattice::segment;
using lattice::storage::FaultKind;

namespace {
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/lattice_rocksdb_fi_test_XXXXXX";
        char* p = mkdtemp(tmpl);
        path_ = p ? p : "/tmp";
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};
}  // namespace

TEST(RocksDbFaultInjection, IoErrorPropagatesThroughPutOnceBudgetExhausted) {
    TempDir dir;
    // DB::Open itself performs several writes/syncs (LOCK, IDENTITY,
    // MANIFEST, first WAL file) before any Put() -- start with a budget
    // large enough that Open always succeeds regardless of exactly how
    // many ops that costs (version-dependent, not worth pinning down),
    // then set the real test budget right after Open succeeds so the
    // fault fires during the Put() loop, which is what this test verifies.
    auto budget = std::make_shared<std::atomic<std::int64_t>>(1'000'000);
    auto fault_fs = std::make_shared<RocksDbFaultInjectingFileSystem>(budget, FaultKind::kIoError);
    // Options::env is a raw, non-owning Env* -- this local must outlive db.
    auto fault_env = rocksdb::NewCompositeEnv(fault_fs);

    rocksdb::Options options;
    options.create_if_missing = true;
    options.env = fault_env.get();

    rocksdb::DB* db_raw = nullptr;
    auto status = rocksdb::DB::Open(options, dir.path(), &db_raw);
    ASSERT_TRUE(status.ok()) << status.ToString();
    std::unique_ptr<rocksdb::DB> db(db_raw);

    budget->store(3);

    // Keep writing until an injected IOError surfaces through Put() itself
    // (RocksDB's WAL append/sync path routes through NewWritableFile, which
    // this fault-injecting filesystem intercepts) -- bounded loop so a
    // regression that silently swallows the fault doesn't hang the test.
    bool saw_failure = false;
    for (int i = 0; i < 200 && !saw_failure; ++i) {
        auto put_status = db->Put(rocksdb::WriteOptions(), "key" + std::to_string(i), "value");
        if (!put_status.ok()) saw_failure = true;
    }
    EXPECT_TRUE(saw_failure) << "expected the injected fault to eventually surface through Put()";
}

TEST(RocksDbFaultInjection, DbSurvivesInjectedFaultsWithoutCorruption) {
    // D-STO-06's actual property: after a fault-injected write session ends
    // (successfully or with an injected error), reopening the SAME
    // directory WITHOUT fault injection must succeed cleanly (no corrupt
    // MANIFEST/WAL), and every acknowledged write must be present.
    TempDir dir;
    // Same Open-vs-test-budget separation as the test above.
    auto budget = std::make_shared<std::atomic<std::int64_t>>(1'000'000);
    auto fault_fs = std::make_shared<RocksDbFaultInjectingFileSystem>(budget, FaultKind::kIoError);
    auto fault_env = rocksdb::NewCompositeEnv(fault_fs);

    rocksdb::Options options;
    options.create_if_missing = true;
    options.env = fault_env.get();

    rocksdb::DB* db_raw = nullptr;
    auto status = rocksdb::DB::Open(options, dir.path(), &db_raw);
    ASSERT_TRUE(status.ok()) << status.ToString();
    std::unique_ptr<rocksdb::DB> db(db_raw);

    budget->store(5);

    std::vector<std::string> acknowledged;
    for (int i = 0; i < 50; ++i) {
        std::string key = "row-" + std::to_string(i);
        auto put_status = db->Put(rocksdb::WriteOptions(), key, "v");
        if (put_status.ok()) acknowledged.push_back(key);
        // Keep going past the first failure too -- a real writer would
        // retry, and this checks the DB stays usable/consistent afterward,
        // not just that the first fault fires correctly.
    }
    db.reset();  // close cleanly, no fault injection on close itself

    // Reopen WITHOUT the fault-injecting filesystem -- default options.
    rocksdb::Options reopen_options;
    reopen_options.create_if_missing = false;
    rocksdb::DB* reopened_raw = nullptr;
    auto reopen_status = rocksdb::DB::Open(reopen_options, dir.path(), &reopened_raw);
    ASSERT_TRUE(reopen_status.ok()) << "DB failed to reopen cleanly after injected faults: "
                                     << reopen_status.ToString();
    std::unique_ptr<rocksdb::DB> reopened(reopened_raw);

    for (auto& key : acknowledged) {
        std::string value;
        auto get_status = reopened->Get(rocksdb::ReadOptions(), key, &value);
        EXPECT_TRUE(get_status.ok()) << "acknowledged write for " << key << " missing after reopen";
    }
}
