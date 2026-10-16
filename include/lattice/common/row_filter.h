#pragma once
// common: opaque predicate handed to index_graph / index_ivf / index_flat
// (D-MOD-01). Indexes depend on this type only, never on `filter`'s schema
// or predicate AST.
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

#include "lattice/common/types.h"

namespace lattice {

class RowFilter {
public:
    static RowFilter always_true() noexcept {
        RowFilter f;
        f.kind_ = Kind::kAlwaysTrue;
        return f;
    }

    static RowFilter from_callable(std::function<bool(RowId)> fn) noexcept {
        RowFilter f;
        f.kind_ = Kind::kCallable;
        f.callable_ = std::move(fn);
        return f;
    }

    // `universe_size` bounds valid RowIds; `roaring_words` is a caller-owned
    // bitset view (64-bit words, LSB-first) that must outlive this RowFilter.
    static RowFilter from_bitmap_view(std::span<const std::uint64_t> roaring_words,
                                       std::size_t universe_size) noexcept {
        RowFilter f;
        f.kind_ = Kind::kBitmap;
        f.bitmap_words_ = roaring_words;
        f.universe_size_ = universe_size;
        return f;
    }

    bool test(RowId id) const noexcept {
        switch (kind_) {
            case Kind::kAlwaysTrue:
                return true;
            case Kind::kCallable:
                return callable_(id);
            case Kind::kBitmap: {
                auto idx = static_cast<std::size_t>(id);
                if (idx >= universe_size_) return false;
                std::size_t word = idx / 64;
                std::size_t bit = idx % 64;
                if (word >= bitmap_words_.size()) return false;
                return (bitmap_words_[word] >> bit) & 1u;
            }
        }
        return false;
    }

    bool is_always_true() const noexcept { return kind_ == Kind::kAlwaysTrue; }

    // Exact cardinality, when known without a scan (bitmap-backed only).
    std::optional<std::size_t> exact_cardinality() const noexcept {
        if (kind_ != Kind::kBitmap) return std::nullopt;
        std::size_t count = 0;
        for (auto w : bitmap_words_) count += static_cast<std::size_t>(__builtin_popcountll(w));
        return count;
    }

private:
    enum class Kind { kAlwaysTrue, kCallable, kBitmap };

    Kind kind_ = Kind::kAlwaysTrue;
    std::function<bool(RowId)> callable_;
    std::span<const std::uint64_t> bitmap_words_;
    std::size_t universe_size_ = 0;
};

}  // namespace lattice
