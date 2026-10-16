#pragma once
// D-CON-04: visited set as a generation-counter array, reset in O(1) per
// query instead of a hash set or cleared bitset. Toggleable (via
// SearchParams::enable_visited_generation_counter) against a plain
// std::vector<bool>-backed fallback so the two can be benchmarked against
// each other (Phase 2 exit gate item 3).
#include <algorithm>
#include <cstdint>
#include <vector>

namespace lattice::index_graph::detail {

class GenerationVisitedSet {
public:
    explicit GenerationVisitedSet(std::size_t n) : gen_(n, 0), current_gen_(0) {}

    void reset() noexcept { ++current_gen_; }  // O(1): no memset needed
    // Reuse across queries on the same thread: grow-only, no clearing. Safe
    // because generation numbers only ever increase, so any stale entry is
    // older than the new generation and reads as "not visited". Allocating and
    // zero-filling an n-sized array per query (4MB at 1M rows) used to
    // dominate query and build time and serialized threads on the allocator.
    void prepare(std::size_t n) {
        if (gen_.size() < n) gen_.resize(n, 0);
        if (++current_gen_ == 0) {  // wrapped after 2^32 queries: clear once
            std::fill(gen_.begin(), gen_.end(), 0);
            current_gen_ = 1;
        }
    }


    bool test_and_set(std::size_t idx) noexcept {
        bool was_visited = gen_[idx] == current_gen_;
        gen_[idx] = current_gen_;
        return was_visited;
    }

    bool is_visited(std::size_t idx) const noexcept { return gen_[idx] == current_gen_; }

private:
    std::vector<std::uint32_t> gen_;
    std::uint32_t current_gen_;
};

// Fallback used when enable_visited_generation_counter is false, for the
// direct on/off comparison D-CON-04 asks for.
class BitsetVisitedSet {
public:
    explicit BitsetVisitedSet(std::size_t n) : bits_(n, false) {}
    void reset() noexcept { std::fill(bits_.begin(), bits_.end(), false); }
    bool test_and_set(std::size_t idx) noexcept {
        bool was_visited = bits_[idx];
        bits_[idx] = true;
        return was_visited;
    }

    bool is_visited(std::size_t idx) const noexcept { return bits_[idx]; }

private:
    std::vector<bool> bits_;
};

}  // namespace lattice::index_graph::detail
