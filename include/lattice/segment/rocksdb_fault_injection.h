#pragma once
// segment: adapts storage::FaultInjectingFileSystem's deterministic,
// seeded fault-injection engine to rocksdb::FileSystem (finding 12,
// module-storage.md), closing a real gap: exit gate item 1 (Phase 3,
// >=10,000 injected crash points) is specified to run "against both the
// Lattice-file path and the RocksDB-adapter path" -- the RocksDB-adapter
// path had no implementation at all until this file. Lives in `segment`,
// not `storage`, because `storage` must never depend on RocksDB (D-CPP-06:
// RocksDB is linked only by `segment`) -- this is exactly what module-
// storage.md's own comment means by "segment wires this in".
//
// "One fault-injection engine, two adapters" (module-storage.md's
// invariant): this shares the SAME budget counter as a
// storage::FaultInjectingFileSystem instance (via shared_budget()), so a
// single seeded scenario can inject a fault into either a Lattice write or
// a RocksDB write, whichever happens to be the Nth filesystem operation --
// not two independently-seeded, uncorrelated fault streams.
#include <atomic>
#include <memory>

#include <rocksdb/file_system.h>

#include "lattice/storage/storage.h"

namespace lattice::segment {

// Wraps one real RocksDB writable file, injecting a fault (matching
// storage::FaultInjectingFile's exact semantics: kIoError fails the call,
// kShortWrite truncates an Append to half its data, kNone/
// kCrashBeforeFsync pass through unchanged -- the latter is handled by
// the process-level crash_hook mechanism elsewhere, not here) once the
// shared budget is exhausted.
class FaultInjectingFSWritableFile : public rocksdb::FSWritableFileWrapper {
public:
    FaultInjectingFSWritableFile(std::unique_ptr<rocksdb::FSWritableFile> inner,
                                  std::shared_ptr<std::atomic<std::int64_t>> budget,
                                  storage::FaultKind kind)
        : rocksdb::FSWritableFileWrapper(inner.get()),
          inner_(std::move(inner)),
          budget_(std::move(budget)),
          kind_(kind) {}

    rocksdb::IOStatus Append(const rocksdb::Slice& data, const rocksdb::IOOptions& options,
                              rocksdb::IODebugContext* dbg) override;
    rocksdb::IOStatus Sync(const rocksdb::IOOptions& options, rocksdb::IODebugContext* dbg) override;
    rocksdb::IOStatus Fsync(const rocksdb::IOOptions& options, rocksdb::IODebugContext* dbg) override;

private:
    bool should_fail() noexcept { return budget_->fetch_sub(1) <= 0; }

    std::unique_ptr<rocksdb::FSWritableFile> inner_;
    std::shared_ptr<std::atomic<std::int64_t>> budget_;
    storage::FaultKind kind_;
};

// Wraps rocksdb::FileSystem::Default(), intercepting only NewWritableFile
// (D-STO-06's kill-loop harness targets writes/fsyncs, matching
// storage::FaultInjectingFile's own scope -- reads are not fault-injected
// on either adapter).
class RocksDbFaultInjectingFileSystem : public rocksdb::FileSystemWrapper {
public:
    RocksDbFaultInjectingFileSystem(std::shared_ptr<std::atomic<std::int64_t>> budget,
                                     storage::FaultKind kind);

    const char* Name() const override { return "RocksDbFaultInjectingFileSystem"; }

    rocksdb::IOStatus NewWritableFile(const std::string& fname, const rocksdb::FileOptions& file_opts,
                                       std::unique_ptr<rocksdb::FSWritableFile>* result,
                                       rocksdb::IODebugContext* dbg) override;

private:
    std::shared_ptr<std::atomic<std::int64_t>> budget_;
    storage::FaultKind kind_;
};

}  // namespace lattice::segment
