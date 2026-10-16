#include "lattice/kernels/kernels.h"

namespace lattice::kernels::scalar {

float dot_ip(const float* a, const float* b, std::uint32_t dim) noexcept {
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

float dot_l2(const float* a, const float* b, std::uint32_t dim) noexcept {
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, std::uint32_t dim) noexcept {
    std::int32_t sum = 0;
    for (std::uint32_t i = 0; i < dim; ++i) {
        sum += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    }
    return sum;
}

std::uint32_t hamming(const std::uint64_t* a, const std::uint64_t* b, std::uint32_t words) noexcept {
    std::uint32_t sum = 0;
    for (std::uint32_t i = 0; i < words; ++i) {
        sum += static_cast<std::uint32_t>(__builtin_popcountll(a[i] ^ b[i]));
    }
    return sum;
}

}  // namespace lattice::kernels::scalar
