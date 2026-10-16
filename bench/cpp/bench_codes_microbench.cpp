// Quick microbenchmark (not a full Vamana build) comparing per-candidate
// scoring throughput of SQ8::distance_to_query vs RaBitQ::
// distance_to_query_fast at a fixed dim -- meant to answer "is RaBitQ's
// branchless per-candidate scorer now competitive with SQ8?" in minutes,
// before committing to another multi-hour full-index rebuild+benchmark
// cycle (the first RaBitQ attempt cost ~7min build + ~7min query per
// config; this answers the same underlying question in seconds).
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "lattice/quant/rabitq.h"
#include "lattice/quant/sq8.h"

using namespace lattice::quant;

namespace {
std::vector<float> random_vec(std::mt19937& rng, std::uint32_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::uint32_t dim = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 768;
    std::size_t num_candidates = argc > 2 ? std::stoul(argv[2]) : 200000;
    std::size_t num_queries = argc > 3 ? std::stoul(argv[3]) : 50;

    std::mt19937 rng(42);

    // SQ8 setup
    std::vector<std::vector<float>> db_storage(num_candidates);
    std::vector<const float*> db_ptrs(num_candidates);
    for (std::size_t i = 0; i < num_candidates; ++i) {
        db_storage[i] = random_vec(rng, dim);
        db_ptrs[i] = db_storage[i].data();
    }
    auto sq8_book = Sq8CodeBook::train(db_ptrs, dim);
    if (!sq8_book) { printf("sq8 train failed\n"); return 1; }
    Sq8Codes sq8_codes(*sq8_book);
    std::vector<std::vector<std::byte>> sq8_encoded(num_candidates);
    for (std::size_t i = 0; i < num_candidates; ++i) sq8_encoded[i] = sq8_codes.encode(db_ptrs[i], dim);

    // RaBitQ setup
    auto rq_book = RaBitQCodeBook::train(dim, 42);
    if (!rq_book) { printf("rabitq train failed\n"); return 1; }
    RaBitQCodes rq_codes(*rq_book);
    std::vector<std::vector<std::byte>> rq_encoded(num_candidates);
    for (std::size_t i = 0; i < num_candidates; ++i) rq_encoded[i] = rq_codes.encode(db_ptrs[i], dim);

    std::vector<std::vector<float>> queries(num_queries);
    for (auto& q : queries) q = random_vec(rng, dim);

    printf("dim=%u num_candidates=%zu num_queries=%zu (total scoring calls = %zu)\n", dim,
           num_candidates, num_queries, num_candidates * num_queries);

    // SQ8 timing: distance_to_query is already "fast path" (no per-query setup needed).
    volatile float sink = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* query = queries[q].data();
        for (std::size_t i = 0; i < num_candidates; ++i) {
            sink += sq8_codes.distance_to_query(query, dim, sq8_encoded[i]);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sq8_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // RaBitQ timing: rotate once per query (paid once), score each candidate fast.
    auto t2 = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
        auto rotated = rq_codes.rotate_query(queries[q].data(), dim);
        for (std::size_t i = 0; i < num_candidates; ++i) {
            sink += rq_codes.distance_to_query_fast(rotated.data(), dim, rq_encoded[i]);
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    double rabitq_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double sq8_ns_per_call = sq8_ms * 1e6 / (num_candidates * num_queries);
    double rq_ns_per_call = rabitq_ms * 1e6 / (num_candidates * num_queries);

    printf("SQ8:    %.1f ms total, %.2f ns/call\n", sq8_ms, sq8_ns_per_call);
    printf("RaBitQ: %.1f ms total, %.2f ns/call (%.2fx %s than SQ8)\n", rabitq_ms, rq_ns_per_call,
           rq_ns_per_call > sq8_ns_per_call ? rq_ns_per_call / sq8_ns_per_call
                                            : sq8_ns_per_call / rq_ns_per_call,
           rq_ns_per_call > sq8_ns_per_call ? "SLOWER" : "FASTER");
    printf("(sink=%f, ignore)\n", sink);
    return 0;
}
