#pragma once
// filter: typed columns, roaring bitmaps, predicate AST, exact selectivity,
// numeric range filtering (D-RET-03 tag half, D-RET-04, D-QRY-07 MVP).
// Depends only on `common`. Never depended on by any index_* module
// (D-MOD-01) -- indexes receive a common::RowFilter built by to_row_filter,
// never this module's Predicate/Bitmap types directly.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "lattice/common/error.h"
#include "lattice/common/row_filter.h"
#include "lattice/common/types.h"

// CRoaring 4.x namespaces its C API under roaring::api in C++ translation
// units, so a hand-rolled forward declaration of a plain global
// roaring_bitmap_t doesn't match -- include the real header instead of
// trying to forward-declare it.
#include <roaring/roaring.h>

namespace lattice::filter {

// --- Bitmap: thin RAII wrapper over CRoaring (D-CPP-06: vendored, not rebuilt) ---
class Bitmap {
public:
    Bitmap();
    ~Bitmap();
    Bitmap(const Bitmap& other);
    Bitmap& operator=(const Bitmap& other);
    Bitmap(Bitmap&& other) noexcept;
    Bitmap& operator=(Bitmap&& other) noexcept;

    void add(RowId id);
    void remove(RowId id);
    bool contains(RowId id) const noexcept;

    // EXACT, not estimated -- this is D-QRY-01's headline claim: the planner
    // always has a true cardinality to plan against, never a guess.
    std::size_t cardinality() const noexcept;

    Bitmap and_with(const Bitmap& other) const;
    Bitmap or_with(const Bitmap& other) const;
    Bitmap not_with(std::size_t universe_size) const;

    roaring_bitmap_t* raw() noexcept { return bm_.get(); }
    const roaring_bitmap_t* raw() const noexcept { return bm_.get(); }

private:
    struct Deleter {
        void operator()(roaring_bitmap_t* p) const noexcept;
    };
    std::unique_ptr<roaring_bitmap_t, Deleter> bm_;
};

// --- Typed columns (D-RET-04) ---
enum class ColumnKind { kInt, kFloat, kTimestamp, kEnum, kTagSet };

// A single-value-per-row categorical/int column, backed by one Bitmap per
// distinct value -- this is what makes equality/set-membership predicates
// resolve to an exact bitmap with no scan.
class TypedColumn {
public:
    explicit TypedColumn(ColumnKind kind) noexcept : kind_(kind) {}

    ColumnKind kind() const noexcept { return kind_; }

    // Records that `id` has value `v` (int64-encoded: raw int, or a
    // dictionary-encoded id for enum/tag-set columns -- encoding policy is
    // the caller's/segment's, this class only indexes whatever key it's given).
    // For single-valued kinds a later set() for the same row REPLACES its value (the row leaves
    // the old value's bitmap); for kTagSet each call adds one more tag.
    void set(RowId id, std::int64_t value);

    // Exact bitmap of rows equal to `value` (empty Bitmap if never seen).
    const Bitmap& bitmap_for(std::int64_t value) const;
    bool has_value(std::int64_t value) const noexcept { return values_.count(value) != 0; }

private:
    ColumnKind kind_;
    std::map<std::int64_t, Bitmap> values_;
    std::map<std::uint64_t, std::int64_t> value_of_;  // row -> current value (single-valued kinds)
    static const Bitmap kEmpty;
};

// --- Numeric range index (D-QRY-07 MVP) ---
template <typename T>
class RangeIndex {
public:
    void add(RowId id, T value) {
        sorted_.emplace_back(value, id);
        dirty_ = true;
        if (!min_ || value < *min_) min_ = value;
        if (!max_ || value > *max_) max_ = value;
    }

    std::optional<std::pair<T, T>> segment_min_max() const noexcept {
        if (!min_ || !max_) return std::nullopt;
        return std::make_pair(*min_, *max_);
    }

    // Binary search + materialize as a bitmap (MVP; bit-sliced index is
    // Advanced, gated empirically per D-QRY-07). Safe to call from concurrent queries; add()
    // must not run concurrently with queries.
    Bitmap rows_in_range(T lo, T hi) {
        std::lock_guard<std::mutex> lock(sort_mutex_);  // the lazy sort mutates sorted_
        ensure_sorted();
        Bitmap out;
        auto it = std::lower_bound(sorted_.begin(), sorted_.end(), lo,
                                    [](const auto& e, T v) { return e.first < v; });
        for (; it != sorted_.end() && it->first <= hi; ++it) out.add(it->second);
        return out;
    }

private:
    void ensure_sorted() {
        if (dirty_) {
            std::sort(sorted_.begin(), sorted_.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            dirty_ = false;
        }
    }
    std::vector<std::pair<T, RowId>> sorted_;
    bool dirty_ = false;
    std::optional<T> min_, max_;
    std::mutex sort_mutex_;
};

// --- Predicate AST ---
enum class PredicateOp { kEq, kIn, kRangeInt, kRangeFloat, kAnd, kOr, kNot };

struct Predicate {
    PredicateOp op;
    std::string column;
    std::vector<std::int64_t> int_literals;    // for kEq (single)/kIn (multi)
    std::int64_t range_lo_i = 0, range_hi_i = 0;
    double range_lo_f = 0, range_hi_f = 0;
    std::vector<Predicate> children;  // for kAnd/kOr (2+)/kNot (1)
};

// --- Evaluator: the only bridge to common::RowFilter (D-MOD-01) ---
class PredicateEvaluator {
public:
    void add_column(const std::string& name, std::shared_ptr<TypedColumn> col) {
        columns_[name] = std::move(col);
    }
    void add_range_index_int(const std::string& name, std::shared_ptr<RangeIndex<std::int64_t>> idx) {
        int_ranges_[name] = std::move(idx);
    }
    void add_range_index_float(const std::string& name, std::shared_ptr<RangeIndex<double>> idx) {
        float_ranges_[name] = std::move(idx);
    }
    void set_universe_size(std::size_t n) noexcept { universe_size_ = n; }

    Result<Bitmap> evaluate(const Predicate& pred) const;
    // Exact number of matching rows, or the evaluation error (e.g. an unknown column) -- an
    // error must not look like "zero matches" to a planner.
    Result<std::size_t> selectivity_count(const Predicate& pred) const;

    // The ONLY function that hands an index a predicate's result -- as an
    // opaque RowFilter, never the Predicate/Bitmap types themselves.
    RowFilter to_row_filter_result(const Bitmap& bm) const;

private:
    std::map<std::string, std::shared_ptr<TypedColumn>> columns_;
    std::map<std::string, std::shared_ptr<RangeIndex<std::int64_t>>> int_ranges_;
    std::map<std::string, std::shared_ptr<RangeIndex<double>>> float_ranges_;
    std::size_t universe_size_ = 0;
};

}  // namespace lattice::filter
