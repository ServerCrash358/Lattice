// Micro-benchmark: per-candidate SQ8 scoring cost at SIFT dimensionality.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "lattice/quant/sq8.h"

using namespace lattice::quant;

int main() {
    constexpr std::uint32_t dim = 128;
    constexpr std::size_t n = 100000;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(0.0f, 200.0f);
    std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
    std::vector<const float*> ptrs(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& x : vecs[i]) x = u(rng);
        ptrs[i] = vecs[i].data();
    }
    auto book = Sq8CodeBook::train(ptrs, dim);
    Sq8Codes codes(*book);
    std::vector<std::byte> blob(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        auto c = codes.encode(ptrs[i], dim);
        std::copy(c.begin(), c.end(), blob.begin() + i * dim);
    }
    std::vector<float> q(dim);
    for (auto& x : q) x = u(rng);

    for (int pass = 0; pass < 2; ++pass) {
        float sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        constexpr int reps = 50;
        for (int r = 0; r < reps; ++r)
            for (std::size_t i = 0; i < n; ++i) {
                std::span<const std::byte> code(blob.data() + i * dim, dim);
                sink += pass == 0 ? codes.neg_l2_to_query(q.data(), dim, code)
                                  : codes.distance_to_query(q.data(), dim, code);
            }
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
                    (static_cast<double>(reps) * n);
        printf("%s: %.1f ns/call (sink=%g)\n", pass == 0 ? "neg_l2_to_query " : "distance_to_query", ns, sink);
    }
    return 0;
}
