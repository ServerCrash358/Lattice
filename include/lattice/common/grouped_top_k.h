#pragma once
// common: grouped top-k (D-RET-07, D-STO-08's "entity vs document identity"
// distinction). Collapses a ranked candidate stream to at most one row per
// group_id (the best-scoring member), so near-duplicate/multi-chunk rows
// belonging to the same document don't crowd out other documents' top-k
// slots. `query`'s planner/executor (Phase 4) is the eventual real caller
// over segment search results; this is deliberately independent of it
// (D-MOD-03: no query/segment dependency here) so the collapsing algorithm
// itself -- and its exit-gate correctness property -- can be built and
// tested in Phase 3 without the executor existing yet.
//
// The property under test (spec/phase-3.md exit gate item 7): the result
// must never under-fill below min(k, available_groups), even when the
// first k candidates by raw score are dominated by a few large groups.
// A naive "take top-k then dedupe by group" is WRONG for this: if the top
// k candidates only cover 2 distinct groups, deduping collapses to 2
// results even though the corpus has far more groups deeper in the
// ranking. Fixing that requires the ability to pull a bigger candidate
// window and keep pulling until either k distinct groups are found or the
// whole candidate source is exhausted -- which is what `fetch_top_n` (a
// caller-supplied "give me your top n" callback) is for.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lattice/common/error.h"
#include "lattice/common/types.h"

namespace lattice::common {

// Returns the top `n` candidates by score, best first. Must be a pure
// function of `n` (not incremental/stateful) -- grouped_top_k re-invokes it
// with larger `n` on each expansion, not "give me n more".
using FetchTopN = std::function<Result<std::vector<RowId>>(std::size_t n)>;

// Returns row's group, or nullopt if it belongs to no group (D-ARCH-03: an
// ungrouped row is its own singleton document, never collapsed with any
// other row regardless of score).
using GroupOf = std::function<std::optional<GroupId>(RowId)>;

inline Result<std::vector<RowId>> grouped_top_k(std::uint32_t k, std::size_t total_available,
                                                 const GroupOf& group_of,
                                                 const FetchTopN& fetch_top_n) {
    if (k == 0) return std::vector<RowId>{};
    if (total_available == 0) return std::vector<RowId>{};

    std::size_t n = std::min<std::size_t>(total_available, std::max<std::size_t>(k, 1) * 4);
    for (;;) {
        auto candidates = fetch_top_n(n);
        if (!candidates) return std::unexpected(candidates.error());

        std::vector<RowId> result;
        result.reserve(k);
        std::unordered_set<std::uint64_t> seen_groups;      // GroupId's underlying value
        std::unordered_set<std::uint64_t> seen_ungrouped;   // RowId's underlying value

        for (RowId row : *candidates) {
            if (result.size() >= k) break;
            auto grp = group_of(row);
            if (grp.has_value()) {
                auto [it, inserted] = seen_groups.insert(static_cast<std::uint64_t>(*grp));
                if (!inserted) continue;  // a better-scoring member of this group already kept
            } else {
                auto raw = static_cast<std::uint64_t>(row);
                if (!seen_ungrouped.insert(raw).second) continue;  // defensive: shouldn't repeat
            }
            result.push_back(row);
        }

        if (result.size() >= k) return result;
        if (candidates->size() < n) {
            // fetch_top_n returned fewer than asked: the candidate source
            // itself is exhausted, not just this window. What we collapsed
            // to IS the true available_groups count (<= k) -- returning
            // fewer than k here is correct, not an under-fill.
            return result;
        }
        if (n >= total_available) {
            // Already asked for everything there is; nothing more to gain
            // from expanding further.
            return result;
        }
        n = std::min(total_available, n * 2);
    }
}

}  // namespace lattice::common
