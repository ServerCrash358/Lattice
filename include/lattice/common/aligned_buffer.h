#pragma once
// common: 64-byte aligned owning buffer (D-CPP-03, D-VEC-05, D-CON-06).
// Move-only; no copy. No allocation after construction -- accessors are
// noexcept and safe on any hot path.
#include <cstddef>
#include <new>
#include <span>
#include <utility>

namespace lattice {

template <typename T, std::size_t Align = 64>
class AlignedBuffer {
public:
    AlignedBuffer() noexcept : data_(nullptr), size_(0) {}

    explicit AlignedBuffer(std::size_t count)
        : data_(count == 0 ? nullptr
                            : static_cast<T*>(::operator new[](count * sizeof(T),
                                                                std::align_val_t{Align}))),
          size_(count) {
        for (std::size_t i = 0; i < size_; ++i) {
            ::new (static_cast<void*>(data_ + i)) T();
        }
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~AlignedBuffer() { reset(); }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    std::span<T> span() noexcept { return std::span<T>(data_, size_); }
    std::span<const T> span() const noexcept { return std::span<const T>(data_, size_); }

    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    void reset() noexcept {
        if (data_ != nullptr) {
            for (std::size_t i = 0; i < size_; ++i) {
                data_[i].~T();
            }
            ::operator delete[](static_cast<void*>(data_), std::align_val_t{Align});
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* data_;
    std::size_t size_;
};

}  // namespace lattice
