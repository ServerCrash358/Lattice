// Crash-injection harness child process (D-STO-06, Phase 3 exit gate item
// 1). Run standalone with:
//   crash_harness_main <dir> <seed> <num_rows> <checkpoint 0|1|2|3>
// Builds a growing segment, upserts num_rows random vectors, seals it, and
// calls commit_seal() with a crash_hook that calls _exit(137) at the given
// checkpoint (0 = no crash, run to completion normally) -- simulating a
// real process crash at that exact point in the two-phase seal handoff
// (finding 11), not just an in-process fault-injection stand-in.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "lattice/segment/segment.h"

using namespace lattice::segment;
using lattice::SegmentId;

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <dir> <seed> <num_rows> <checkpoint 0-3>\n", argv[0]);
        return 2;
    }
    std::string dir = argv[1];
    unsigned seed = static_cast<unsigned>(std::atoi(argv[2]));
    int num_rows = std::atoi(argv[3]);
    int checkpoint = std::atoi(argv[4]);

    constexpr std::uint32_t dim = 16;
    std::string growing_path = dir + "/growing";

    auto growing = GrowingSegment::create(growing_path, dim, static_cast<std::size_t>(num_rows) + 10);
    if (!growing) {
        std::fprintf(stderr, "create failed: %s\n", growing.error().message.c_str());
        return 1;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < num_rows; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        auto r = (*growing)->upsert("row-" + std::to_string(i), v.data(), static_cast<std::uint64_t>(i + 1));
        if (!r) {
            std::fprintf(stderr, "upsert failed: %s\n", r.error().message.c_str());
            return 1;
        }
    }

    auto sealed = SealedSegment::seal_from(*growing.value(), static_cast<SegmentId>(1),
                                            static_cast<std::size_t>(num_rows) + 100 /* force flat */);
    if (!sealed) {
        std::fprintf(stderr, "seal_from failed: %s\n", sealed.error().message.c_str());
        return 1;
    }

    CrashHook hook;
    if (checkpoint != 0) {
        hook = [checkpoint](int cp) {
            if (cp == checkpoint) {
                std::fflush(nullptr);  // flush stdio buffers, NOT an fsync -- the crash is deliberate
                _exit(137);            // hard kill: no destructors, no further cleanup
            }
        };
    }

    auto commit_result = commit_seal(std::move(*growing), **sealed, dir, hook);
    if (!commit_result) {
        std::fprintf(stderr, "commit_seal failed: %s\n", commit_result.error().message.c_str());
        return 1;
    }
    return 0;  // completed normally (checkpoint 0, or the requested checkpoint never fired)
}
