#include "lattice/quant/rabitq.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace lattice::quant {

namespace {

// Rotate v (length dim) by row-major dim x dim matrix P: out = P @ v.
void rotate(const std::vector<float>& P, const float* v, std::uint32_t dim, float* out) {
    for (std::uint32_t i = 0; i < dim; ++i) {
        float sum = 0.0f;
        const float* row = &P[static_cast<std::size_t>(i) * dim];
        for (std::uint32_t j = 0; j < dim; ++j) sum += row[j] * v[j];
        out[i] = sum;
    }
}

}  // namespace

Result<RaBitQCodeBook> RaBitQCodeBook::train(std::uint32_t dim, std::uint64_t seed) {
    if (dim == 0) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "dim must be > 0"));
    }

    // Random orthogonal matrix via modified Gram-Schmidt on a Gaussian
    // random matrix (Householder QR would be more numerically stable at
    // very high dim, but Gram-Schmidt is simple and dim=768 is well within
    // its comfortable accuracy range for this purpose).
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<std::vector<float>> rows(dim, std::vector<float>(dim));
    for (auto& row : rows) {
        for (auto& x : row) x = gauss(rng);
    }

    for (std::uint32_t i = 0; i < dim; ++i) {
        float norm = 0.0f;
        // Degenerate draws are astronomically unlikely for a Gaussian matrix, but if one happens
        // the redrawn row must be orthogonalised too (the earlier retry skipped that step and
        // could leave a non-orthogonal row).
        for (int attempt = 0; attempt < 8; ++attempt) {
            // Subtract projections onto all previous (already-orthonormal) rows.
            for (std::uint32_t k = 0; k < i; ++k) {
                float dot = 0.0f;
                for (std::uint32_t j = 0; j < dim; ++j) dot += rows[i][j] * rows[k][j];
                for (std::uint32_t j = 0; j < dim; ++j) rows[i][j] -= dot * rows[k][j];
            }
            norm = 0.0f;
            for (std::uint32_t j = 0; j < dim; ++j) norm += rows[i][j] * rows[i][j];
            norm = std::sqrt(norm);
            if (norm >= 1e-8f) break;
            for (auto& x : rows[i]) x = gauss(rng);
        }
        if (norm < 1e-8f) {
            return std::unexpected(Error::make(ErrorCode::kInternal, "failed to build an orthogonal rotation"));
        }
        for (auto& x : rows[i]) x /= norm;
    }

    RaBitQCodeBook book;
    book.dim = dim;
    book.rotation.resize(static_cast<std::size_t>(dim) * dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        std::copy(rows[i].begin(), rows[i].end(),
                  book.rotation.begin() + static_cast<std::size_t>(i) * dim);
    }
    return book;
}

std::vector<std::byte> RaBitQCodes::encode(const float* v, std::uint32_t dim) const noexcept {
    std::vector<float> normalized(dim);
    float norm = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) norm += v[i] * v[i];
    norm = std::sqrt(norm);
    float inv_norm = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) normalized[i] = v[i] * inv_norm;

    std::vector<float> rotated(dim);
    rotate(book_->rotation, normalized.data(), dim, rotated.data());

    std::size_t n_bytes = (dim + 7) / 8;
    std::vector<std::byte> out(n_bytes + 2 * sizeof(float), std::byte{0});

    float sum_abs = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) {
        if (rotated[i] >= 0.0f) {
            out[i / 8] |= std::byte{static_cast<unsigned char>(1u << (i % 8))};
        }
        sum_abs += std::abs(rotated[i]);
    }
    float c = sum_abs / std::sqrt(static_cast<float>(dim));  // <x', x_bar>

    std::memcpy(out.data() + n_bytes, &norm, sizeof(float));
    std::memcpy(out.data() + n_bytes + sizeof(float), &c, sizeof(float));
    return out;
}

RaBitQCodes::BoundedDistance RaBitQCodes::distance_to_query_bounded(
    const float* query, std::uint32_t dim, std::span<const std::byte> code) const noexcept {
    std::size_t n_bytes = (dim + 7) / 8;
    float norm_x, c;
    std::memcpy(&norm_x, code.data() + n_bytes, sizeof(float));
    std::memcpy(&c, code.data() + n_bytes + sizeof(float), sizeof(float));

    // Rotate the query. This recomputes q' = P @ query on every call --
    // fine for a one-shot correctness check (this function), wrong for a
    // hot loop scoring many candidates against the same query; see
    // rotate_query()/distance_to_query_fast() below for that case.
    std::vector<float> rotated_q(dim);
    rotate(book_->rotation, query, dim, rotated_q.data());

    float dot_q_xbar = 0.0f;
    float norm_q_sq = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) {
        norm_q_sq += query[i] * query[i];
        bool bit = (static_cast<unsigned char>(code[i / 8]) >> (i % 8)) & 1u;
        float sign = bit ? 1.0f : -1.0f;
        dot_q_xbar += sign * rotated_q[i];
    }
    dot_q_xbar /= std::sqrt(static_cast<float>(dim));
    float norm_q = std::sqrt(norm_q_sq);

    float estimate = norm_x * c * dot_q_xbar;
    float residual = std::sqrt(std::max(0.0f, 1.0f - c * c));
    float error_bound = norm_x * norm_q * residual;

    return {estimate, error_bound};
}

std::vector<float> RaBitQCodes::rotate_query(const float* query, std::uint32_t dim) const noexcept {
    std::vector<float> rotated(dim);
    rotate(book_->rotation, query, dim, rotated.data());
    return rotated;
}

float RaBitQCodes::distance_to_query_fast(const float* rotated_query, std::uint32_t dim,
                                           std::span<const std::byte> code) const noexcept {
    std::size_t n_bytes = (dim + 7) / 8;
    float norm_x, c;
    std::memcpy(&norm_x, code.data() + n_bytes, sizeof(float));
    std::memcpy(&c, code.data() + n_bytes + sizeof(float), sizeof(float));

    // Branchless, byte-at-a-time: the original per-bit `bit ? 1.0f : -1.0f`
    // ternary (plus a fresh i/8, i%8 division every iteration) is a
    // data-dependent branch the compiler can't turn into straight-line
    // vector code -- measured to make this ~3x SLOWER than SQ8's simple
    // float multiply-add loop despite RaBitQ's 7x smaller code (see
    // spec/phase-3.md exit gate item 2 section f). Processing one code
    // byte at a time with a compile-time-unrolled inner 8-loop and
    // `2.0f*bit - 1.0f` instead of a ternary removes the branch and the
    // per-iteration division, which is what a hot per-candidate scoring
    // loop actually needs to be competitive.
    float dot_q_xbar = 0.0f;
    std::uint32_t full_bytes = dim / 8;
    std::uint32_t i = 0;
    for (std::uint32_t byte_idx = 0; byte_idx < full_bytes; ++byte_idx) {
        unsigned char byte = static_cast<unsigned char>(code[byte_idx]);
        for (int b = 0; b < 8; ++b) {
            float bit = static_cast<float>((byte >> b) & 1u);
            float sign = 2.0f * bit - 1.0f;
            dot_q_xbar += sign * rotated_query[i];
            ++i;
        }
    }
    for (; i < dim; ++i) {
        unsigned char byte = static_cast<unsigned char>(code[i / 8]);
        float bit = static_cast<float>((byte >> (i % 8)) & 1u);
        float sign = 2.0f * bit - 1.0f;
        dot_q_xbar += sign * rotated_query[i];
    }
    dot_q_xbar /= std::sqrt(static_cast<float>(dim));

    return norm_x * c * dot_q_xbar;
}

}  // namespace lattice::quant
