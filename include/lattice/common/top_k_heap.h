#pragma once
// common: fixed-capacity top-k heap (D-QRY-05, D-MOD-03). No allocation
// after construction -- the vector is reserved once, everything else is
// noexcept swap/compare. Keeps the SMALLEST of the top-k by Score at the
// root (min-heap), so try_insert can reject in O(1) once full and evict
// in O(log k) otherwise. Score must be operator< comparable (higher = better).
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace lattice {

template <typename Score, typename Id>
class TopKHeap {
public:
    explicit TopKHeap(std::size_t k) noexcept : k_(k) { items_.reserve(k); }

    // Returns true if `id` made it into the top-k (inserted or replaced the
    // current worst); false if it was rejected.
    bool try_insert(Score s, Id id) noexcept {
        if (items_.size() < k_) {
            items_.emplace_back(s, id);
            std::push_heap(items_.begin(), items_.end(), Greater{});
            return true;
        }
        if (items_.empty() || !(s > items_.front().first)) {
            return false;
        }
        std::pop_heap(items_.begin(), items_.end(), Greater{});
        items_.back() = {s, id};
        std::push_heap(items_.begin(), items_.end(), Greater{});
        return true;
    }

    // Current k-th best score (the worst score still in the top-k), used as
    // the cross-segment pruning threshold (D-QRY-05).
    Score threshold() const noexcept { return items_.empty() ? Score{} : items_.front().first; }

    bool full() const noexcept { return items_.size() >= k_; }
    std::size_t size() const noexcept { return items_.size(); }

    std::vector<std::pair<Score, Id>> extract_sorted() noexcept {
        std::vector<std::pair<Score, Id>> out = std::move(items_);
        std::sort(out.begin(), out.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        items_.clear();
        return out;
    }

private:
    // Min-heap ordering: smallest score at the front, so it's the first
    // candidate evicted when a better one arrives.
    struct Greater {
        bool operator()(const std::pair<Score, Id>& a, const std::pair<Score, Id>& b) const noexcept {
            return a.first > b.first;
        }
    };

    std::size_t k_;
    std::vector<std::pair<Score, Id>> items_;
};

}  // namespace lattice
