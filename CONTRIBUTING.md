# Contributing to Lattice

Thanks for your interest! Lattice is a disk-resident vector index in C++; contributions that improve correctness, performance (measured on real data), tests, or documentation are welcome.

## Getting set up

Linux (or WSL2 on Windows), C++23 (GCC 15+ or Clang 17+), CMake 3.25+.

```bash
sudo apt install librocksdb-dev liburing-dev libroaring-dev
cmake --preset release && cmake --build build/release -j$(nproc)
ctest --test-dir build/release
```

Other presets: `debug`, `asan`, `tsan`, `bench`, `clang-release`. On WSL2, build on the Linux filesystem (for example `~/lattice`), not under `/mnt/c` — Windows-drive mounts have unreliable `fsync` and unrepresentative I/O latency, which invalidates the crash-consistency and disk benchmarks.

## Layout

```
include/lattice/<module>/   public headers
src/<module>/               implementation
tests/<module>/             unit and property tests
examples/                   runnable examples (built by CMake)
bench/                      benchmarks, baselines, dataset scripts
docs/                       architecture and benchmark write-ups
```

Modules: `common`, `kernels`, `quant`, `index_flat`, `index_graph`, `filter`, `storage`, `segment`. Dependencies are kept narrow on purpose: RocksDB is linked only by `segment`, `liburing` only by `storage`, CRoaring only by `filter`, and FAISS/hnswlib appear only in `bench/baselines`.

## Code conventions

- **C++23**, headers plus sources, no modules. Only features available in both GCC and Clang.
- **Errors:** `std::expected<T, Error>` at fallible boundaries; no exceptions across module boundaries. A corrupt-on-read condition is an `Error`, never undefined behaviour.
- **Hot paths** (per-distance or per-node work) are `noexcept` and allocation-free after setup.
- **Ownership:** RAII; `unique_ptr` for exclusive ownership, `span`/`string_view` for views, no raw `new`/`delete`.
- **Naming:** types `PascalCase`, functions `snake_case`, constants `kPascalCase`, member variables with a trailing underscore, files `snake_case`. ID types (`RowId`, `VectorId`, `SegmentId`) are always the wrapper types, never raw integers, at API boundaries.
- **Docs:** every public function gets a comment with what it does, preconditions, and which error codes it can return.
- Keep the build warning-free under `-Wall -Wextra` (enabled in the top-level `CMakeLists.txt`), and match the formatting of the surrounding code.

## Testing and benchmarking expectations

- Add or update tests for every behaviour change. Index changes need a recall test against brute-force ground truth; concurrency changes should be exercised under the `tsan` preset; storage changes should be run through the fault-injection tests.
- **Validate performance and recall claims on real data** (SIFT1M or similar), not only synthetic data. Synthetic benchmarks hid real bugs in this project's history.
- Performance-relevant changes should include a before/after benchmark (same machine, same config) in the pull request description.

## Pull requests

- One logical change per commit, with a message that explains the *why*.
- Keep PRs focused; mention any behaviour, format, or API change explicitly.

## License of contributions

By submitting a contribution you agree that it is licensed under the Apache License 2.0 (see `LICENSE`, section 5), the same terms as the project. No CLA is currently required.
