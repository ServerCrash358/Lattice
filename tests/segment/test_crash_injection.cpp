// Phase 3 exit gate item 1: crash-consistency over >=10,000 injected crash
// points across write/seal/manifest-swap. Real process crashes (fork +
// _exit in the child, per crash_harness_main.cpp), not just simulated I/O
// errors -- the actual property under test is finding 11's two-phase seal
// handoff: a crash at any point must leave the manifest either not-yet-
// updated or durably-updated, NEVER torn, and a subsequent seal retry must
// always succeed.
//
// Trial count: LATTICE_CRASH_TRIALS env var overrides the default (200,
// sized for a normal test run in low tens of seconds). Set it to 10000+ for
// the full exit-gate run (each trial forks a process and opens RocksDB
// twice, so the full run takes materially longer -- minutes, not seconds).
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>

#include "lattice/segment/segment.h"

using namespace lattice::segment;
using namespace lattice::storage;
using lattice::SegmentId;

namespace {

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/lattice_crash_test_XXXXXX";
        char* p = mkdtemp(tmpl);
        path_ = p ? p : "/tmp";
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

struct RunResult {
    bool crashed;      // true if killed by signal (our deliberate _exit still shows as normal exit(137), not a signal -- see note below)
    int exit_code;
};

// Runs crash_harness_main as a child process. _exit(137) is a NORMAL exit
// (not a signal) from waitpid's point of view -- it's how the harness
// simulates "the process is gone", without actually sending itself a
// signal, which keeps the parent's wait logic simple (WIFEXITED, not
// WIFSIGNALED) while still being a real fork+exit, not an in-process stand-in.
RunResult run_harness(const std::string& dir, unsigned seed, int num_rows, int checkpoint) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: replace this process image entirely with the harness.
        execl(CRASH_HARNESS_PATH, CRASH_HARNESS_PATH, dir.c_str(), std::to_string(seed).c_str(),
              std::to_string(num_rows).c_str(), std::to_string(checkpoint).c_str(), nullptr);
        _exit(127);  // execl failed
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return {WEXITSTATUS(status) == 137, WEXITSTATUS(status)};
    return {true, -1};  // killed by an actual signal (shouldn't happen here, but treat as "crashed")
}

int trial_count() {
    const char* env = std::getenv("LATTICE_CRASH_TRIALS");
    return env ? std::atoi(env) : 200;
}

}  // namespace

TEST(CrashInjection, ManifestNeverTornAcrossManyRandomCrashPoints) {
    int trials = trial_count();
    std::mt19937 seed_rng(0xC0FFEE);
    std::uniform_int_distribution<int> checkpoint_dist(0, 3);
    std::uniform_int_distribution<int> rows_dist(5, 40);

    int torn_manifests = 0;
    int failed_retry_after_crash = 0;

    for (int t = 0; t < trials; ++t) {
        TempDir dir;
        unsigned seed = seed_rng();
        int checkpoint = checkpoint_dist(seed_rng);
        int num_rows = rows_dist(seed_rng);

        auto result = run_harness(dir.path(), seed, num_rows, checkpoint);
        if (checkpoint == 0) {
            EXPECT_EQ(result.exit_code, 0) << "trial " << t << " (seed=" << seed
                                            << "): checkpoint 0 (no crash) should complete normally";
        }

        // Invariant 1: whatever happened, the manifest (if it exists at
        // all) must load cleanly -- never a torn/corrupt footer.
        auto manifest = Manifest::load_current(dir.path());
        if (!manifest.has_value()) {
            ++torn_manifests;
            ADD_FAILURE() << "trial " << t << " (seed=" << seed << ", checkpoint=" << checkpoint
                          << "): manifest failed to load: " << manifest.error().message;
            continue;
        }

        // Invariant 2: re-running the SAME seal sequence INTO THE SAME
        // (possibly-crashed) directory -- simulating an actual restart and
        // retry -- must succeed regardless of where the previous attempt
        // crashed. This is what actually matters: recovery is always
        // possible, not stuck in a bad state. commit_seal's idempotency
        // check (skip the manifest write if this segment id is already
        // durably sealed) is exactly what makes this safe to repeat.
        auto retry = run_harness(dir.path(), seed, num_rows, /*checkpoint=*/0 /* run to completion */);
        if (retry.exit_code != 0) {
            ++failed_retry_after_crash;
            ADD_FAILURE() << "trial " << t << " (seed=" << seed << ", checkpoint=" << checkpoint
                          << "): retry-to-completion failed with exit code " << retry.exit_code;
        }
    }

    EXPECT_EQ(torn_manifests, 0) << "out of " << trials << " trials";
    EXPECT_EQ(failed_retry_after_crash, 0) << "out of " << trials << " trials";
}

TEST(CrashInjection, CrashAtEachCheckpointIndividuallyRecovers) {
    // A focused version of the above: one trial per checkpoint (0-3),
    // deterministic, easy to debug in isolation if the randomized test above
    // ever fails.
    for (int checkpoint = 0; checkpoint <= 3; ++checkpoint) {
        TempDir dir;
        auto result = run_harness(dir.path(), /*seed=*/42, /*num_rows=*/20, checkpoint);
        if (checkpoint == 0) {
            EXPECT_EQ(result.exit_code, 0) << "checkpoint 0 (no crash) should complete normally";
        }

        auto manifest = Manifest::load_current(dir.path());
        ASSERT_TRUE(manifest.has_value()) << "checkpoint=" << checkpoint << ": manifest corrupt";

        if (checkpoint == 0 || checkpoint >= 2) {
            // Completed normally, or crashed at/after the manifest write
            // succeeded: the entry must be visible.
            EXPECT_EQ(manifest->live_segments().size(), 1u) << "checkpoint=" << checkpoint;
        }
        // checkpoint 1 (crash BEFORE the manifest write): entry must be
        // absent -- the write never happened, which is correct, not a bug.
        if (checkpoint == 1) {
            EXPECT_EQ(manifest->live_segments().size(), 0u)
                << "checkpoint=1 (pre-manifest-write crash): entry should not exist yet";
        }
    }
}
