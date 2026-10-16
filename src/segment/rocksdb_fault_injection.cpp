#include "lattice/segment/rocksdb_fault_injection.h"

namespace lattice::segment {

rocksdb::IOStatus FaultInjectingFSWritableFile::Append(const rocksdb::Slice& data,
                                                         const rocksdb::IOOptions& options,
                                                         rocksdb::IODebugContext* dbg) {
    if (should_fail()) {
        if (kind_ == storage::FaultKind::kIoError) {
            return rocksdb::IOStatus::IOError("injected fault: Append (finding 12)");
        }
        if (kind_ == storage::FaultKind::kShortWrite && data.size() > 0) {
            // Matches storage::FaultInjectingFile::pwrite's exact semantics:
            // write only half the data, not a hard failure.
            rocksdb::Slice half(data.data(), data.size() / 2);
            return inner_->Append(half, options, dbg);
        }
    }
    return inner_->Append(data, options, dbg);
}

rocksdb::IOStatus FaultInjectingFSWritableFile::Sync(const rocksdb::IOOptions& options,
                                                       rocksdb::IODebugContext* dbg) {
    if (should_fail() && kind_ == storage::FaultKind::kIoError) {
        return rocksdb::IOStatus::IOError("injected fault: Sync (finding 12)");
    }
    return inner_->Sync(options, dbg);
}

rocksdb::IOStatus FaultInjectingFSWritableFile::Fsync(const rocksdb::IOOptions& options,
                                                        rocksdb::IODebugContext* dbg) {
    if (should_fail() && kind_ == storage::FaultKind::kIoError) {
        return rocksdb::IOStatus::IOError("injected fault: Fsync (finding 12)");
    }
    return inner_->Fsync(options, dbg);
}

RocksDbFaultInjectingFileSystem::RocksDbFaultInjectingFileSystem(
    std::shared_ptr<std::atomic<std::int64_t>> budget, storage::FaultKind kind)
    : rocksdb::FileSystemWrapper(rocksdb::FileSystem::Default()),
      budget_(std::move(budget)),
      kind_(kind) {}

rocksdb::IOStatus RocksDbFaultInjectingFileSystem::NewWritableFile(
    const std::string& fname, const rocksdb::FileOptions& file_opts,
    std::unique_ptr<rocksdb::FSWritableFile>* result, rocksdb::IODebugContext* dbg) {
    std::unique_ptr<rocksdb::FSWritableFile> inner;
    auto status = target()->NewWritableFile(fname, file_opts, &inner, dbg);
    if (!status.ok()) return status;
    *result = std::make_unique<FaultInjectingFSWritableFile>(std::move(inner), budget_, kind_);
    return status;
}

}  // namespace lattice::segment
