#include "lattice/filter/filter.h"

#include <roaring/roaring.h>

namespace lattice::filter {

void Bitmap::Deleter::operator()(roaring_bitmap_t* p) const noexcept {
    if (p) roaring_bitmap_free(p);
}

Bitmap::Bitmap() : bm_(roaring_bitmap_create()) {}
Bitmap::~Bitmap() = default;

Bitmap::Bitmap(const Bitmap& other) : bm_(roaring_bitmap_copy(other.bm_.get())) {}

Bitmap& Bitmap::operator=(const Bitmap& other) {
    if (this != &other) bm_.reset(roaring_bitmap_copy(other.bm_.get()));
    return *this;
}

Bitmap::Bitmap(Bitmap&& other) noexcept = default;
Bitmap& Bitmap::operator=(Bitmap&& other) noexcept = default;

void Bitmap::add(RowId id) { roaring_bitmap_add(bm_.get(), static_cast<std::uint32_t>(id)); }
void Bitmap::remove(RowId id) { roaring_bitmap_remove(bm_.get(), static_cast<std::uint32_t>(id)); }

bool Bitmap::contains(RowId id) const noexcept {
    return roaring_bitmap_contains(bm_.get(), static_cast<std::uint32_t>(id));
}

std::size_t Bitmap::cardinality() const noexcept {
    return static_cast<std::size_t>(roaring_bitmap_get_cardinality(bm_.get()));
}

Bitmap Bitmap::and_with(const Bitmap& other) const {
    Bitmap out;
    out.bm_.reset(roaring_bitmap_and(bm_.get(), other.bm_.get()));
    return out;
}

Bitmap Bitmap::or_with(const Bitmap& other) const {
    Bitmap out;
    out.bm_.reset(roaring_bitmap_or(bm_.get(), other.bm_.get()));
    return out;
}

Bitmap Bitmap::not_with(std::size_t universe_size) const {
    // One range flip instead of a contains/add per row of the universe.
    Bitmap out;
    out.bm_.reset(roaring_bitmap_flip(bm_.get(), 0, static_cast<std::uint64_t>(universe_size)));
    return out;
}

const Bitmap TypedColumn::kEmpty{};

void TypedColumn::set(RowId id, std::int64_t value) {
    if (kind_ != ColumnKind::kTagSet) {
        auto row = static_cast<std::uint64_t>(id);
        auto prev = value_of_.find(row);
        if (prev != value_of_.end() && prev->second != value) {
            auto old = values_.find(prev->second);
            if (old != values_.end()) old->second.remove(id);
        }
        value_of_[row] = value;
    }
    values_[value].add(id);
}

const Bitmap& TypedColumn::bitmap_for(std::int64_t value) const {
    auto it = values_.find(value);
    return it != values_.end() ? it->second : kEmpty;
}

namespace {
Result<Bitmap> evaluate_impl(const Predicate& pred, const PredicateEvaluator& ev,
                              const std::map<std::string, std::shared_ptr<TypedColumn>>& columns,
                              const std::map<std::string, std::shared_ptr<RangeIndex<std::int64_t>>>& int_ranges,
                              const std::map<std::string, std::shared_ptr<RangeIndex<double>>>& float_ranges,
                              std::size_t universe_size) {
    switch (pred.op) {
        case PredicateOp::kEq: {
            auto it = columns.find(pred.column);
            if (it == columns.end() || pred.int_literals.empty()) {
                return std::unexpected(
                    Error::make(ErrorCode::kNotFound, "unknown column: " + pred.column));
            }
            return it->second->bitmap_for(pred.int_literals[0]);
        }
        case PredicateOp::kIn: {
            auto it = columns.find(pred.column);
            if (it == columns.end()) {
                return std::unexpected(
                    Error::make(ErrorCode::kNotFound, "unknown column: " + pred.column));
            }
            Bitmap result;
            for (auto v : pred.int_literals) result = result.or_with(it->second->bitmap_for(v));
            return result;
        }
        case PredicateOp::kRangeInt: {
            auto it = int_ranges.find(pred.column);
            if (it == int_ranges.end()) {
                return std::unexpected(
                    Error::make(ErrorCode::kNotFound, "unknown int range column: " + pred.column));
            }
            return it->second->rows_in_range(pred.range_lo_i, pred.range_hi_i);
        }
        case PredicateOp::kRangeFloat: {
            auto it = float_ranges.find(pred.column);
            if (it == float_ranges.end()) {
                return std::unexpected(Error::make(ErrorCode::kNotFound,
                                                    "unknown float range column: " + pred.column));
            }
            return it->second->rows_in_range(pred.range_lo_f, pred.range_hi_f);
        }
        case PredicateOp::kAnd: {
            if (pred.children.empty()) {
                return std::unexpected(
                    Error::make(ErrorCode::kInvalidArgument, "AND requires >=1 child"));
            }
            auto acc = evaluate_impl(pred.children[0], ev, columns, int_ranges, float_ranges,
                                      universe_size);
            if (!acc) return acc;
            for (std::size_t i = 1; i < pred.children.size(); ++i) {
                auto rhs = evaluate_impl(pred.children[i], ev, columns, int_ranges, float_ranges,
                                          universe_size);
                if (!rhs) return rhs;
                acc = acc->and_with(*rhs);
            }
            return acc;
        }
        case PredicateOp::kOr: {
            if (pred.children.empty()) {
                return std::unexpected(
                    Error::make(ErrorCode::kInvalidArgument, "OR requires >=1 child"));
            }
            auto acc = evaluate_impl(pred.children[0], ev, columns, int_ranges, float_ranges,
                                      universe_size);
            if (!acc) return acc;
            for (std::size_t i = 1; i < pred.children.size(); ++i) {
                auto rhs = evaluate_impl(pred.children[i], ev, columns, int_ranges, float_ranges,
                                          universe_size);
                if (!rhs) return rhs;
                acc = acc->or_with(*rhs);
            }
            return acc;
        }
        case PredicateOp::kNot: {
            if (pred.children.size() != 1) {
                return std::unexpected(
                    Error::make(ErrorCode::kInvalidArgument, "NOT requires exactly 1 child"));
            }
            auto child = evaluate_impl(pred.children[0], ev, columns, int_ranges, float_ranges,
                                        universe_size);
            if (!child) return child;
            return child->not_with(universe_size);
        }
    }
    return std::unexpected(Error::make(ErrorCode::kInternal, "unhandled predicate op"));
}
}  // namespace

Result<Bitmap> PredicateEvaluator::evaluate(const Predicate& pred) const {
    return evaluate_impl(pred, *this, columns_, int_ranges_, float_ranges_, universe_size_);
}

Result<std::size_t> PredicateEvaluator::selectivity_count(const Predicate& pred) const {
    auto r = evaluate(pred);
    if (!r) return std::unexpected(r.error());
    return r->cardinality();
}

RowFilter PredicateEvaluator::to_row_filter_result(const Bitmap& bm) const {
    // Copies the bitmap into the closure (Bitmap is cheap to copy via
    // CRoaring's copy-on-write-friendly containers) so the RowFilter is
    // self-contained and safe to outlive the evaluator call that built it.
    return RowFilter::from_callable(
        [bm](RowId id) { return bm.contains(id); });
}

}  // namespace lattice::filter
