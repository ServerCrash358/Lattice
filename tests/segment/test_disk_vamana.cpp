// Phase 3 exit gate item 3: disk-resident Vamana reproduces the in-RAM
// backend's recall@10 (+-0.01) on the same dataset, confirming the shared
// node-record format round-trips correctly across backends.
#include "lattice/segment/segment.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>
#include <unordered_set>

using namespace lattice;
using namespace lattice::index_graph;
using namespace lattice::segment;
using lattice::quant::Sq8CodeBook;
using lattice::quant::Sq8Codes;

namespace {
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/lattice_disk_vamana_test_XXXXXX";
        char* p = mkdtemp(tmpl);
        path_ = p ? p : "/tmp";
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

struct Dataset {
    std::vector<std::vector<float>> storage;
    std::vector<const float*> ptrs;
};

Dataset make_dataset(std::mt19937& rng, std::size_t n, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Dataset d;
    d.storage.resize(n, std::vector<float>(dim));
    for (auto& v : d.storage) {
        for (auto& x : v) x = dist(rng);
    }
    for (auto& v : d.storage) d.ptrs.push_back(v.data());
    return d;
}

double recall_at_k(const std::vector<VectorId>& got, const std::vector<VectorId>& expected,
                    std::uint32_t k) {
    std::unordered_set<std::uint64_t> exp_set;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, expected.size()); ++i) {
        exp_set.insert(static_cast<std::uint64_t>(expected[i]));
    }
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, got.size()); ++i) {
        if (exp_set.count(static_cast<std::uint64_t>(got[i]))) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}
}  // namespace

TEST(DiskVamana, RoundTripsAndReproducesInRamRecall) {
    TempDir dir;
    std::mt19937 rng(2024);
    constexpr std::uint32_t dim = 64;
    constexpr std::size_t n = 1500;
    auto data = make_dataset(rng, n, dim);

    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);

    auto layout = NodeRecordLayout::make(dim, ElemType::kFloat32, 32);
    ASSERT_TRUE(layout.has_value());

    VamanaBuildParams build_params;
    build_params.r = 32;
    build_params.build_list_size = 75;
    build_params.metric = Metric::kInnerProduct;

    auto ram_index = VamanaIndex::build(data.ptrs, dim, codes, build_params, *layout);
    ASSERT_TRUE(ram_index.has_value()) << ram_index.error().message;

    std::string disk_path = dir.path() + "/vamana.disk";
    auto ser = ram_index->serialize_to_disk(disk_path);
    ASSERT_TRUE(ser.has_value()) << ser.error().message;

    auto disk_index = open_disk_resident_vamana(disk_path);
    ASSERT_TRUE(disk_index.has_value()) << disk_index.error().message;
    EXPECT_EQ(disk_index->size(), n);

    SearchParams sp;
    sp.ef_search = 100;
    sp.rerank_k = 20;

    constexpr std::uint32_t k = 10;
    constexpr int num_queries = 30;
    double total_recall_ram = 0.0, total_recall_disk = 0.0;
    int exact_agreement = 0;

    for (int q = 0; q < num_queries; ++q) {
        auto query = make_dataset(rng, 1, dim);

        auto ram_result = ram_index->search(query.ptrs[0], k, sp);
        auto disk_result = disk_index->search(query.ptrs[0], k, sp);
        ASSERT_TRUE(ram_result.has_value());
        ASSERT_TRUE(disk_result.has_value());

        // Both backends traverse the IDENTICAL graph (same node records,
        // same greedy_search algorithm) -- with the same deterministic
        // search parameters they should return the same result set.
        if (*ram_result == *disk_result) ++exact_agreement;

        total_recall_ram += recall_at_k(*ram_result, *ram_result, k);    // trivially 1.0, sanity only
        total_recall_disk += recall_at_k(*disk_result, *ram_result, k);  // disk vs RAM as "ground truth"
    }

    double mean_recall_disk_vs_ram = total_recall_disk / num_queries;
    // Phase 3 gate: reproduces in-RAM recall within +-0.01. Since both
    // traverse the same graph deterministically, this should be very close
    // to 1.0, not just within the gate's tolerance.
    EXPECT_GE(mean_recall_disk_vs_ram, 0.99)
        << "disk backend diverged from in-RAM backend's results more than expected";
    EXPECT_GE(exact_agreement, num_queries - 1)
        << "expected near-total exact agreement between backends on identical graph/params";
}

TEST(DiskVamana, RejectsCorruptMagic) {
    TempDir dir;
    std::string path = dir.path() + "/garbage.disk";
    {
        auto f = storage::PosixFile::open(path, true);
        ASSERT_TRUE(f.has_value());
        std::vector<std::byte> junk(64, std::byte{0xFF});
        ASSERT_TRUE((*f)->pwrite(0, junk).has_value());
    }
    auto opened = open_disk_resident_vamana(path);
    EXPECT_FALSE(opened.has_value());
}

// Regression: AsyncNodeFile used one io_uring ring shared by every caller, so
// concurrent searches on a disk-resident index corrupted the ring. Concurrent
// results must equal sequential results, for both metrics.
TEST(DiskVamana, ConcurrentSearchesMatchSequentialResults) {
    for (Metric metric : {Metric::kInnerProduct, Metric::kL2}) {
        TempDir dir;
        std::mt19937 rng(77);
        constexpr std::uint32_t dim = 32;
        constexpr std::size_t n = 1200;
        auto data = make_dataset(rng, n, dim);

        auto book = Sq8CodeBook::train(data.ptrs, dim);
        ASSERT_TRUE(book.has_value());
        Sq8Codes codes(*book);
        auto layout = NodeRecordLayout::make(dim, ElemType::kFloat32, 24);
        ASSERT_TRUE(layout.has_value());

        VamanaBuildParams bp;
        bp.r = 24;
        bp.build_list_size = 60;
        bp.metric = metric;
        auto ram_index = VamanaIndex::build(data.ptrs, dim, codes, bp, *layout);
        ASSERT_TRUE(ram_index.has_value());
        std::string path = dir.path() + "/v.disk";
        ASSERT_TRUE(ram_index->serialize_to_disk(path).has_value());
        auto disk_index = open_disk_resident_vamana(path);
        ASSERT_TRUE(disk_index.has_value());

        constexpr int num_queries = 64;
        auto queries = make_dataset(rng, num_queries, dim);
        SearchParams sp;
        sp.ef_search = 64;
        sp.rerank_k = 20;

        std::vector<std::vector<VectorId>> expected(num_queries);
        for (int q = 0; q < num_queries; ++q) {
            auto r = disk_index->search(queries.ptrs[q], 10, sp);
            ASSERT_TRUE(r.has_value());
            expected[q] = *r;
        }

        std::atomic<int> mismatches{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&] {
                for (int round = 0; round < 5; ++round) {
                    for (int q = 0; q < num_queries; ++q) {
                        auto r = disk_index->search(queries.ptrs[q], 10, sp);
                        if (!r.has_value() || *r != expected[q]) ++mismatches;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        EXPECT_EQ(mismatches.load(), 0) << "metric=" << static_cast<int>(metric);
    }
}

// Compact record layout + mmap node store must return exactly what the in-RAM index and the
// io_uring-backed store return, for both metrics, including under concurrent searches.
TEST(DiskVamana, CompactMmapMatchesIoUringAndRamAndIsThreadSafe) {
    for (Metric metric : {Metric::kInnerProduct, Metric::kL2}) {
        TempDir dir;
        std::mt19937 rng(321);
        constexpr std::uint32_t dim = 48;
        constexpr std::size_t n = 1100;
        auto data = make_dataset(rng, n, dim);
        auto book = Sq8CodeBook::train(data.ptrs, dim);
        ASSERT_TRUE(book.has_value());
        Sq8Codes codes(*book);
        auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, 24);
        ASSERT_TRUE(layout.has_value());
        EXPECT_LT(layout->record_bytes(), 1024u);  // genuinely compact, not sector-padded

        VamanaBuildParams bp;
        bp.r = 24;
        bp.build_list_size = 60;
        bp.metric = metric;
        bp.build_threads = 1;  // deterministic
        auto ram = VamanaIndex::build(data.ptrs, dim, codes, bp, *layout);
        ASSERT_TRUE(ram.has_value());
        std::string path = dir.path() + "/c.disk";
        ASSERT_TRUE(ram->serialize_to_disk(path).has_value());
        auto mm = open_disk_resident_vamana(path, NodeAccess::kMmap);
        auto uring = open_disk_resident_vamana(path, NodeAccess::kIoUring);
        ASSERT_TRUE(mm.has_value() && uring.has_value());

        constexpr int nq = 40;
        auto queries = make_dataset(rng, nq, dim);
        SearchParams sp;
        sp.ef_search = 64;
        sp.rerank_k = 20;
        std::vector<std::vector<VectorId>> expected(nq);
        for (int q = 0; q < nq; ++q) {
            auto a = ram->search(queries.ptrs[q], 10, sp);
            auto b = mm->search(queries.ptrs[q], 10, sp);
            auto c = uring->search(queries.ptrs[q], 10, sp);
            ASSERT_TRUE(a && b && c);
            EXPECT_EQ(*a, *b);
            EXPECT_EQ(*a, *c);
            expected[q] = *a;
        }
        std::atomic<int> bad{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < 4; ++t)
            ts.emplace_back([&] {
                for (int round = 0; round < 5; ++round)
                    for (int q = 0; q < nq; ++q) {
                        auto r = mm->search(queries.ptrs[q], 10, sp);
                        if (!r || *r != expected[q]) ++bad;
                    }
            });
        for (auto& th : ts) th.join();
        EXPECT_EQ(bad.load(), 0) << "metric=" << static_cast<int>(metric);
    }
}

// The index file is versioned and its size is validated on open: truncated files, legacy
// 'VAM1' files and garbage are rejected with an error instead of being misread.
TEST(DiskVamana, RejectsTruncatedAndLegacyFilesAndLeavesNoTempFile) {
    TempDir dir;
    std::mt19937 rng(5);
    constexpr std::uint32_t dim = 16;
    auto data = make_dataset(rng, 500, dim);
    auto book = Sq8CodeBook::train(data.ptrs, dim);
    ASSERT_TRUE(book.has_value());
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, 16);
    ASSERT_TRUE(layout.has_value());
    VamanaBuildParams bp;
    bp.r = 16;
    bp.build_threads = 1;
    auto index = VamanaIndex::build(data.ptrs, dim, codes, bp, *layout);
    ASSERT_TRUE(index.has_value());
    std::string path = dir.path() + "/v.lattice";
    ASSERT_TRUE(index->serialize_to_disk(path).has_value());
    EXPECT_FALSE(std::filesystem::exists(path + ".tmp"));
    ASSERT_TRUE(open_disk_resident_vamana(path).has_value());

    auto full_size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, full_size - 1);
    EXPECT_FALSE(open_disk_resident_vamana(path).has_value());

    ASSERT_TRUE(index->serialize_to_disk(path).has_value());  // rewrite over the damaged file
    {
        auto f = storage::PosixFile::open(path, false);
        ASSERT_TRUE(f.has_value());
        std::uint32_t legacy = 0x56414D31;  // 'VAM1'
        std::vector<std::byte> b(4);
        std::memcpy(b.data(), &legacy, 4);
        ASSERT_TRUE((*f)->pwrite(0, b).has_value());
    }
    auto legacy = open_disk_resident_vamana(path);
    ASSERT_FALSE(legacy.has_value());
    EXPECT_NE(legacy.error().message.find("VAM1"), std::string::npos);
}

// A corrupt neighbour id is served as a zeroed record; that must be visible via node_read_errors().
TEST(DiskVamana, CorruptNeighbourIdIsCountedAsReadError) {
    TempDir dir;
    std::mt19937 rng(6);
    constexpr std::uint32_t dim = 8;
    auto data = make_dataset(rng, 300, dim);
    auto book = Sq8CodeBook::train(data.ptrs, dim);
    Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, 8);
    VamanaBuildParams bp;
    bp.r = 8;
    bp.build_threads = 1;
    auto index = VamanaIndex::build(data.ptrs, dim, codes, bp, *layout);
    ASSERT_TRUE(index.has_value());
    std::string path = dir.path() + "/c.lattice";
    ASSERT_TRUE(index->serialize_to_disk(path).has_value());
    {
        // Every node's first neighbour slot -> an id far past the end.
        auto f = storage::PosixFile::open(path, false);
        ASSERT_TRUE(f.has_value());
        std::uint32_t bogus = 0x7FFFFFF0u;
        std::vector<std::byte> b(4);
        std::memcpy(b.data(), &bogus, 4);
        for (std::size_t i = 0; i < 300; ++i) {
            std::uint64_t off = index_graph::DiskFormat::kHeaderBytes + i * layout->record_bytes() + dim * 4 + 4;
            ASSERT_TRUE((*f)->pwrite(off, b).has_value());
        }
    }
    auto disk = open_disk_resident_vamana(path);
    ASSERT_TRUE(disk.has_value());
    // Corrupt neighbour ids are skipped by the traversal (they used to index the visited set and
    // codes blob out of bounds); results stay valid for both scoring paths.
    for (bool use_codes : {true, false}) {
        SearchParams sp;
        sp.use_codes = use_codes;
        auto r = disk->search(data.ptrs[0], 5, sp);
        ASSERT_TRUE(r.has_value());
        for (auto id : *r) EXPECT_LT(static_cast<std::size_t>(id), 300u);
    }
    // A direct read of an out-of-range node is served zeroed and counted.
    EXPECT_EQ(disk->node_read_errors(), 0u);
    (void)disk->vector_of(static_cast<VectorId>(1u << 30));
    EXPECT_EQ(disk->node_read_errors(), 1u);
}
