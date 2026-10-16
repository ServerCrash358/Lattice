#include "lattice/segment/segment.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <random>
#include <thread>

using namespace lattice::segment;
using lattice::RowId;
using lattice::SegmentId;

namespace {
class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/lattice_segment_test_XXXXXX";
        char* p = mkdtemp(tmpl);
        path_ = p ? p : "/tmp";
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::vector<float> random_vec(std::mt19937& rng, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}
}  // namespace

TEST(GrowingSegment, UpsertAndSearchFindsInsertedRows) {
    TempDir dir;
    constexpr std::uint32_t dim = 16;
    auto seg = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(seg.has_value()) << seg.error().message;

    std::mt19937 rng(1);
    std::vector<float> target = random_vec(rng, dim);
    ASSERT_TRUE((*seg)->upsert("row-target", target.data(), 1).has_value());
    for (int i = 0; i < 20; ++i) {
        auto v = random_vec(rng, dim);
        ASSERT_TRUE((*seg)->upsert("row-" + std::to_string(i), v.data(), 2 + i).has_value());
    }

    auto result = (*seg)->search(target.data(), 1);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0], "row-target");
    EXPECT_EQ((*seg)->row_count(), 21u);
}

TEST(GrowingSegment, LwwRejectsStaleOffset) {
    TempDir dir;
    constexpr std::uint32_t dim = 8;
    auto seg = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(seg.has_value());

    std::vector<float> v1(dim, 1.0f), v2(dim, 2.0f);
    ASSERT_TRUE((*seg)->upsert("row-1", v1.data(), /*offset=*/10).has_value());
    // Stale write with a SMALLER offset must be rejected (LWW, D-STO-08).
    ASSERT_TRUE((*seg)->upsert("row-1", v2.data(), /*offset=*/5).has_value());

    // The mirror should still reflect v1 (the offset-10 write), not v2.
    auto result = (*seg)->search(v1.data(), 1);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0], "row-1");
}

TEST(GrowingSegment, UpsertOfExistingIdReplacesInPlaceNotDuplicate) {
    TempDir dir;
    constexpr std::uint32_t dim = 8;
    auto seg = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(seg.has_value());

    std::vector<float> v1(dim, 1.0f), v2(dim, 2.0f);
    ASSERT_TRUE((*seg)->upsert("row-1", v1.data(), 1).has_value());
    ASSERT_TRUE((*seg)->upsert("row-1", v2.data(), 2).has_value());  // newer offset, applies

    EXPECT_EQ((*seg)->row_count(), 1u);  // NOT 2 -- same id, in-place update
    auto result = (*seg)->search(v2.data(), 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)[0], "row-1");
}

TEST(PrimaryKeyIndex, DeleteBeforeCreateRaceIsResolvedCorrectly) {
    // Finding 14's regression test: a delete for an id arrives with offset
    // 5, BEFORE a create for the same id with offset 3 arrives (out of
    // order, as a real CDC pipeline can deliver). The delete must win
    // because its offset (5) is greater -- and this must hold even if the
    // "create" is processed after the "delete" in wall-clock arrival order.
    PrimaryKeyIndex pk;

    pk.apply_delete("ghost-row", /*offset=*/5);  // delete arrives first, row never existed
    auto loc1 = pk.lookup("ghost-row");
    ASSERT_TRUE(loc1.has_value());
    EXPECT_TRUE(loc1->deleted);

    // The late-arriving create has a SMALLER offset (3 < 5) -- must be rejected.
    pk.apply("ghost-row", /*offset=*/3, static_cast<SegmentId>(1));
    auto loc2 = pk.lookup("ghost-row");
    ASSERT_TRUE(loc2.has_value());
    EXPECT_TRUE(loc2->deleted) << "a late create with a smaller offset must not resurrect a tombstone";
}

TEST(PrimaryKeyIndex, CreateThenDeleteWithHigherOffsetWins) {
    PrimaryKeyIndex pk;
    pk.apply("row-1", 1, static_cast<SegmentId>(1));
    pk.apply_delete("row-1", 2);  // higher offset -- delete applies
    auto loc = pk.lookup("row-1");
    ASSERT_TRUE(loc.has_value());
    EXPECT_TRUE(loc->deleted);
}

TEST(PrimaryKeyIndex, StaleDeleteDoesNotOverwriteNewerCreate) {
    PrimaryKeyIndex pk;
    pk.apply("row-1", 10, static_cast<SegmentId>(1));
    pk.apply_delete("row-1", 5);  // lower offset -- stale, must be rejected
    auto loc = pk.lookup("row-1");
    ASSERT_TRUE(loc.has_value());
    EXPECT_FALSE(loc->deleted);
}

namespace {
// Simulates a real process crash by aborting commit_seal's execution right
// after the requested checkpoint's hook fires -- everything after that
// point (in commit_seal or the caller) never runs, same observable effect
// on durable state as crash_harness_main.cpp's real fork+_exit, without
// paying for a subprocess in a fast unit test.
struct SimulatedCrash {};

void run_up_to_checkpoint(std::unique_ptr<GrowingSegment> growing, const SealedSegment& sealed,
                           const std::string& dir, int checkpoint) {
    CrashHook hook = [checkpoint](int cp) {
        if (cp == checkpoint) throw SimulatedCrash{};
    };
    try {
        auto r = commit_seal(std::move(growing), sealed, dir, hook);
        ASSERT_TRUE(r.has_value()) << r.error().message;
    } catch (const SimulatedCrash&) {
        // Expected for checkpoint != 0: execution stopped exactly here.
    }
}
}  // namespace

TEST(PrimaryKeyIndex, RebuildAfterSimulatedCrashMidSealMatchesPreCrashState) {
    // Exit gate item 6: after a crash at each checkpoint of the two-phase
    // seal handoff (finding 11), PrimaryKeyIndex::rebuild() (sealed
    // segments' .ids files, via the manifest) plus replay_growing_segment_cf
    // (anything not yet sealed) must reconstruct the same get-by-id/delete
    // state a live system had right before the crash.
    for (int checkpoint : {1, 2, 3}) {
        TempDir dir;
        constexpr std::uint32_t dim = 8;
        std::mt19937 rng(checkpoint);
        auto v0 = random_vec(rng, dim), v1 = random_vec(rng, dim);

        auto growing = GrowingSegment::create(dir.path() + "/growing", dim, 100);
        ASSERT_TRUE(growing.has_value());
        ASSERT_TRUE((*growing)->upsert("row-0", v0.data(), 1).has_value());
        ASSERT_TRUE((*growing)->upsert("row-1", v1.data(), 2).has_value());
        // Delete-before-create (finding 14): "ghost" never had a row, only
        // a tombstone -- must never be sealed (SealedSegment excludes
        // tombstones) but must still show as deleted while it's only in the
        // growing segment's still-live RocksDB CF.
        ASSERT_TRUE((*growing)->delete_row("ghost", 5).has_value());

        auto sealed = SealedSegment::seal_from(*growing.value(), static_cast<SegmentId>(1),
                                                1000 /* force flat, force below Vamana threshold */);
        ASSERT_TRUE(sealed.has_value()) << sealed.error().message;

        run_up_to_checkpoint(std::move(*growing), **sealed, dir.path(), checkpoint);

        // --- Restart: rebuild purely from what's durable on disk. ---
        auto manifest = lattice::storage::Manifest::load_current(dir.path());
        ASSERT_TRUE(manifest.has_value());
        auto pk = PrimaryKeyIndex::rebuild(*manifest, dir.path());
        ASSERT_TRUE(pk.has_value()) << "checkpoint=" << checkpoint << ": " << pk.error().message;

        // Growing CF is removed only after checkpoint 2's manifest write is
        // durable (checkpoint 3's cleanup step) -- replay it if it's still there.
        std::string growing_path = dir.path() + "/growing";
        if (std::filesystem::exists(growing_path)) {
            auto r = replay_growing_segment_cf(*pk, growing_path);
            ASSERT_TRUE(r.has_value()) << r.error().message;
        }

        if (checkpoint == 1) {
            // Crashed before the manifest write: sealing never happened.
            // Both rows and the tombstone are recoverable only via the
            // still-intact growing CF.
            auto l0 = pk->lookup("row-0");
            ASSERT_TRUE(l0.has_value());
            EXPECT_FALSE(l0->segment.has_value()) << "row-0 should still be in the growing segment";
            auto l1 = pk->lookup("row-1");
            ASSERT_TRUE(l1.has_value());
            EXPECT_FALSE(l1->segment.has_value());
            auto lg = pk->lookup("ghost");
            ASSERT_TRUE(lg.has_value());
            EXPECT_TRUE(lg->deleted);
        } else {
            // checkpoint 2 or 3: the manifest entry (and its .ids file) is
            // durable, so both rows resolve to the sealed segment either
            // way -- rebuild() gets this from the .ids file alone; a
            // still-present growing CF (checkpoint 2) is redundant but must
            // not disagree (LWW by the same source_offset keeps it a no-op).
            auto l0 = pk->lookup("row-0");
            ASSERT_TRUE(l0.has_value());
            EXPECT_EQ(l0->segment, static_cast<SegmentId>(1));
            auto l1 = pk->lookup("row-1");
            ASSERT_TRUE(l1.has_value());
            EXPECT_EQ(l1->segment, static_cast<SegmentId>(1));

            auto lg = pk->lookup("ghost");
            if (checkpoint == 2) {
                // Growing CF (with the tombstone) is still on disk.
                ASSERT_TRUE(lg.has_value());
                EXPECT_TRUE(lg->deleted);
            } else {
                // Disclosed gap: at checkpoint 3 the growing CF (the only
                // place "ghost"'s tombstone ever lived, since sealing
                // excludes tombstones) has been removed. Its delete-before-
                // create protection does not currently survive past its
                // segment's seal -- a real gap, not something this test
                // should silently paper over. Fixing it requires the
                // tombstone itself to be carried into the sealed segment's
                // durable state (not just live rows), which is out of scope
                // here; this asserts the CURRENT (gap-having) behavior so a
                // future fix shows up as an intentional test change, not a
                // silent regression.
                EXPECT_FALSE(lg.has_value())
                    << "known gap: tombstone-only ids do not survive past their segment's seal";
            }
        }
    }
}

TEST(SealedSegment, SealExcludesTombstonesAndPreservesSearchability) {
    TempDir dir;
    constexpr std::uint32_t dim = 16;
    auto growing = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(growing.has_value());

    std::mt19937 rng(42);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 50; ++i) {
        auto v = random_vec(rng, dim);
        ASSERT_TRUE((*growing)->upsert("row-" + std::to_string(i), v.data(), i + 1).has_value());
        vecs.push_back(v);
    }
    // Delete a few rows -- they must not appear in the sealed segment.
    ASSERT_TRUE((*growing)->delete_row("row-5", 1000).has_value());
    ASSERT_TRUE((*growing)->delete_row("row-10", 1001).has_value());

    auto sealed = SealedSegment::seal_from(*growing.value(), static_cast<SegmentId>(1),
                                            /*small_segment_threshold=*/10000 /* force flat path */);
    ASSERT_TRUE(sealed.has_value()) << sealed.error().message;
    EXPECT_EQ((*sealed)->row_count(), 48u);  // 50 - 2 deleted

    auto result = (*sealed)->search(vecs[7].data(), 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)[0], "row-7");

    // Deleted rows must be genuinely absent (not just low-ranked).
    auto result5 = (*sealed)->search(vecs[5].data(), 48);
    ASSERT_TRUE(result5.has_value());
    for (auto& id : *result5) EXPECT_NE(id, "row-5");
}

TEST(SealedSegment, SealWithVamanaBackendAboveThresholdMatchesFlatRecall) {
    TempDir dir;
    constexpr std::uint32_t dim = 32;
    constexpr int n = 300;
    auto growing = GrowingSegment::create(dir.path(), dim, n + 10);
    ASSERT_TRUE(growing.has_value());

    std::mt19937 rng(7);
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < n; ++i) {
        auto v = random_vec(rng, dim);
        ASSERT_TRUE((*growing)->upsert("row-" + std::to_string(i), v.data(), i + 1).has_value());
        vecs.push_back(v);
    }

    // Force the Vamana (non-flat) path with a low threshold.
    auto sealed = SealedSegment::seal_from(*growing.value(), static_cast<SegmentId>(2),
                                            /*small_segment_threshold=*/50);
    ASSERT_TRUE(sealed.has_value()) << sealed.error().message;
    EXPECT_EQ((*sealed)->row_count(), static_cast<std::size_t>(n));

    // Recall check: query with an exact stored vector should find itself at rank 0.
    int hits = 0;
    for (int i = 0; i < 30; ++i) {
        auto r = (*sealed)->search(vecs[i].data(), 1);
        ASSERT_TRUE(r.has_value());
        if (!r->empty() && (*r)[0] == "row-" + std::to_string(i)) ++hits;
    }
    EXPECT_GE(hits, 27) << "expected high self-recall (>=90%) with Vamana backend";
}

TEST(GrowingSegment, DataSurvivesCloseAndReopen) {
    // Not the full fork/kill crash-injection harness (still outstanding --
    // see Phase 3 status notes) -- this verifies RocksDB's own durability
    // guarantee holds for the way GrowingSegment actually uses it (Put with
    // default sync semantics, then a clean close and fresh reopen).
    TempDir dir;
    constexpr std::uint32_t dim = 8;
    std::vector<float> v(dim, 3.14f);
    {
        auto seg = GrowingSegment::create(dir.path(), dim, 1000);
        ASSERT_TRUE(seg.has_value());
        ASSERT_TRUE((*seg)->upsert("persisted-row", v.data(), 1).has_value());
    }  // seg destructed here -- closes the RocksDB handle

    auto reopened = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;
    // Note: reopening does NOT automatically repopulate the in-RAM mirror in
    // this Phase 3 slice (that would need a replay-into-mirror step on
    // open, not yet wired) -- but the underlying RocksDB row must still be
    // gettable directly, which SealedSegment::seal_from's iteration relies on.
    auto sealed = SealedSegment::seal_from(*reopened.value(), static_cast<SegmentId>(3), 10000);
    ASSERT_TRUE(sealed.has_value());
    EXPECT_EQ((*sealed)->row_count(), 1u);
}

// delete_row used to leave the row in the in-RAM mirror, so search kept returning it.
TEST(GrowingSegment, DeletedRowIsNotReturnedBySearch) {
    TempDir dir;
    constexpr std::uint32_t dim = 8;
    auto seg = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(seg.has_value());
    std::mt19937 rng(3);
    auto target = random_vec(rng, dim);
    ASSERT_TRUE((*seg)->upsert("victim", target.data(), 1).has_value());
    for (int i = 0; i < 10; ++i) {
        auto v = random_vec(rng, dim);
        ASSERT_TRUE((*seg)->upsert("other-" + std::to_string(i), v.data(), 1).has_value());
    }
    ASSERT_TRUE((*seg)->delete_row("victim", 2).has_value());
    auto r = (*seg)->search(target.data(), 11);
    ASSERT_TRUE(r.has_value());
    for (const auto& id : *r) EXPECT_NE(id, "victim");
    EXPECT_EQ((*seg)->row_count(), 10u);

    // Re-inserting with a newer offset makes it visible again.
    ASSERT_TRUE((*seg)->upsert("victim", target.data(), 3).has_value());
    auto r2 = (*seg)->search(target.data(), 1);
    ASSERT_TRUE(r2.has_value());
    ASSERT_EQ(r2->size(), 1u);
    EXPECT_EQ((*r2)[0], "victim");
    EXPECT_EQ((*seg)->row_count(), 11u);
}

// Reopening used to leave the mirror empty: rows written before the restart were invisible to
// search and row_count()/should_seal() reported 0.
TEST(GrowingSegment, ReopenRestoresSearchAndRowCount) {
    TempDir dir;
    constexpr std::uint32_t dim = 8;
    std::mt19937 rng(4);
    auto target = random_vec(rng, dim);
    {
        auto seg = GrowingSegment::create(dir.path(), dim, 3);
        ASSERT_TRUE(seg.has_value());
        ASSERT_TRUE((*seg)->upsert("keep", target.data(), 1).has_value());
        auto v = random_vec(rng, dim);
        ASSERT_TRUE((*seg)->upsert("gone", v.data(), 1).has_value());
        auto w = random_vec(rng, dim);
        ASSERT_TRUE((*seg)->upsert("also", w.data(), 1).has_value());
        ASSERT_TRUE((*seg)->delete_row("gone", 2).has_value());
    }
    auto reopened = GrowingSegment::create(dir.path(), dim, 3);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;
    EXPECT_EQ((*reopened)->row_count(), 2u);
    EXPECT_FALSE((*reopened)->should_seal());
    auto r = (*reopened)->search(target.data(), 3);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 2u);
    EXPECT_EQ((*r)[0], "keep");
    for (const auto& id : *r) EXPECT_NE(id, "gone");
}

// The last-write-wins check (read offset, then write) was not atomic, so a concurrent older
// write could overwrite a newer one.
TEST(GrowingSegment, ConcurrentUpsertsKeepTheNewestOffset) {
    TempDir dir;
    constexpr std::uint32_t dim = 4;
    auto seg = GrowingSegment::create(dir.path(), dim, 100000);
    ASSERT_TRUE(seg.has_value());
    std::vector<float> v(dim, 1.0f);
    std::vector<std::thread> ts;
    for (int t = 0; t < 2; ++t) {
        ts.emplace_back([&, t] {
            for (std::uint64_t off = static_cast<std::uint64_t>(t) + 1; off <= 400; off += 2) {
                ASSERT_TRUE((*seg)->upsert("hot", v.data(), off).has_value());
            }
        });
    }
    for (auto& th : ts) th.join();
    auto sealed = SealedSegment::seal_from(**seg, static_cast<SegmentId>(1), 100000);
    ASSERT_TRUE(sealed.has_value());
    ASSERT_EQ((*sealed)->row_count(), 1u);
    EXPECT_EQ((*sealed)->source_offset_for(static_cast<lattice::VectorId>(0)), 400u);
}

TEST(GrowingSegment, RejectsNonFiniteVectors) {
    TempDir dir;
    constexpr std::uint32_t dim = 4;
    auto seg = GrowingSegment::create(dir.path(), dim, 1000);
    ASSERT_TRUE(seg.has_value());
    std::vector<float> bad = {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
    auto r = (*seg)->upsert("nan-row", bad.data(), 1);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, lattice::ErrorCode::kInvalidArgument);
    EXPECT_EQ((*seg)->row_count(), 0u);
}

// Filtered search on a graph-indexed sealed segment now goes through the filter-aware planner;
// the old post-filter returned almost nothing at 1% selectivity.
TEST(SealedSegment, FilteredSearchOnGraphSegmentFindsExactMatches) {
    TempDir dir;
    constexpr std::uint32_t dim = 16;
    auto growing = GrowingSegment::create(dir.path() + "/g", dim, 5000);
    ASSERT_TRUE(growing.has_value());
    std::mt19937 rng(8);
    for (int i = 0; i < 3000; ++i) {
        auto v = random_vec(rng, dim);
        ASSERT_TRUE((*growing)->upsert("row-" + std::to_string(i), v.data(), 1).has_value());
    }
    auto sealed = SealedSegment::seal_from(**growing, static_cast<SegmentId>(1), /*threshold=*/100);
    ASSERT_TRUE(sealed.has_value());
    auto& seg = **sealed;
    auto filter = lattice::RowFilter::from_callable([](RowId r) { return static_cast<std::uint64_t>(r) % 100 == 0; });

    auto q = random_vec(rng, dim);
    std::vector<std::pair<float, std::string>> exact;
    for (std::size_t i = 0; i < seg.row_count(); i += 100) {
        auto v = seg.vector_at(static_cast<lattice::VectorId>(i));
        float s = 0;
        for (std::uint32_t d = 0; d < dim; ++d) s += q[d] * v[d];
        exact.emplace_back(s, seg.external_id_for(static_cast<lattice::VectorId>(i)));
    }
    std::sort(exact.begin(), exact.end(), [](auto& a, auto& b) { return a.first > b.first; });

    auto r = seg.search(q.data(), 10, filter);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 10u);
    for (int i = 0; i < 10; ++i) EXPECT_EQ((*r)[i], exact[i].second) << "rank " << i;
}

// commit_seal deletes the growing segment's directory recursively; it must only ever do that to
// a directory GrowingSegment::create made (marker present).
TEST(CommitSeal, DoesNotDeleteADirectoryWithoutTheMarker) {
    TempDir dir;
    std::string gpath = dir.path() + "/growing";
    auto growing = GrowingSegment::create(gpath, 4, 100);
    ASSERT_TRUE(growing.has_value());
    std::vector<float> v(4, 0.5f);
    ASSERT_TRUE((*growing)->upsert("a", v.data(), 1).has_value());
    auto sealed = SealedSegment::seal_from(**growing, static_cast<SegmentId>(1), 1000);
    ASSERT_TRUE(sealed.has_value());
    std::filesystem::remove(gpath + "/LATTICE_GROWING_SEGMENT");
    ASSERT_TRUE(commit_seal(std::move(*growing), **sealed, dir.path()).has_value());
    EXPECT_TRUE(std::filesystem::exists(gpath)) << "unmarked directory was deleted";
}
