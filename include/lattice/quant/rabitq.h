#pragma once
// quant: RaBitQ (D-VEC-02d, R5: functional in Phase 3, headline-quality in
// Phase 5). Per-segment codebook (D-VEC-08, finding 30) -- unlike SQ8, which
// is trained once per collection.
//
// Encoding: a random orthogonal rotation P (generated once per codebook)
// preserves inner products exactly (<Pq,Px> = <q,x>) while making a
// normalized vector's per-coordinate distribution behave like i.i.d.
// Gaussians -- this is what makes a simple 1-bit-per-dimension sign code a
// good direction estimate regardless of the original data's distribution.
// For a unit vector x_n with rotated form x' = P*x_n and sign code x_bar
// (each coordinate +-1/sqrt(dim)), x' decomposes as x' = c*x_bar + residual,
// where c = <x', x_bar> and x_bar/residual are orthogonal. The estimator
// drops the residual: <q,x> ~= ||x|| * c * <q', x_bar>.
//
// Error bound: this codebase derives its bound from Cauchy-Schwarz applied
// to the dropped residual term, which gives a bound that ALWAYS holds
// (deterministic, not probabilistic):
//     |<q,x> - estimate| <= ||x|| * ||q|| * sqrt(1 - c^2)
// This is looser than the SIGMOD 2024 paper's tighter concentration-based
// probabilistic bound, but it is simple to derive correctly and always
// true -- exactly what Phase 3's exit gate needs validated (the estimator
// is genuinely bounded, not that the tightest possible bound is achieved).
// Phase 5 revisits tightening this alongside the adaptive-rerank policy.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lattice/common/error.h"

namespace lattice::quant {

struct RaBitQCodeBook {
    std::uint32_t dim = 0;
    std::vector<float> rotation;  // dim*dim, row-major orthogonal matrix P

    static Result<RaBitQCodeBook> train(std::uint32_t dim, std::uint64_t seed);
};

class RaBitQCodes {
public:
    explicit RaBitQCodes(const RaBitQCodeBook& book) noexcept : book_(&book) {}

    // Layout: ceil(dim/8) sign-bit bytes, then norm (f32), then c (f32).
    std::vector<std::byte> encode(const float* v, std::uint32_t dim) const noexcept;

    struct BoundedDistance {
        float estimate;
        float error_bound;  // |estimate - exact| <= error_bound, always (see header comment)
    };
    BoundedDistance distance_to_query_bounded(const float* query, std::uint32_t dim,
                                               std::span<const std::byte> code) const noexcept;

    // `distance_to_query_bounded` re-rotates `query` on every call -- an
    // O(dim^2) cost per candidate, fine for a one-shot correctness check
    // but wrong for a hot loop that scores many candidates against the
    // SAME query (e.g. graph traversal). These two split that: rotate once
    // per query (O(dim^2), paid once), then score each candidate in O(dim)
    // via the already-rotated form. Same estimator, same numbers -- just
    // not recomputing the same rotation on every single candidate.
    std::vector<float> rotate_query(const float* query, std::uint32_t dim) const noexcept;
    float distance_to_query_fast(const float* rotated_query, std::uint32_t dim,
                                  std::span<const std::byte> code) const noexcept;

    static std::size_t code_bytes(std::uint32_t dim) noexcept {
        return (dim + 7) / 8 + 2 * sizeof(float);
    }

    // Exposes the underlying codebook so a caller can copy/persist it (e.g.
    // VamanaIndex::build owning its own copy for later codes-gated search,
    // including after a disk round-trip).
    const RaBitQCodeBook& book() const noexcept { return *book_; }

private:
    const RaBitQCodeBook* book_;
};

}  // namespace lattice::quant
