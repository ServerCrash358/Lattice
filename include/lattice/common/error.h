#pragma once
// common: error handling (D-CPP-04). std::expected<T, Error> at every
// fallible boundary; exceptions are reserved for programmer errors that
// should abort (see LATTICE_ASSERT below), never for I/O/format/validation
// failures, which are always a value.
#include <expected>
#include <string>
#include <utility>

namespace lattice {

enum class ErrorCode {
    kOk = 0,
    kIoError,
    kCorruption,
    kNotFound,
    kInvalidArgument,
    kSchemaMismatch,
    kDimensionMismatch,
    kAlreadyExists,
    kConcurrentModification,
    kUnsupported,
    kInternal,
};

struct Error {
    ErrorCode code;
    std::string message;

    static Error make(ErrorCode c, std::string msg) noexcept {
        return Error{c, std::move(msg)};
    }
};

template <typename T>
using Result = std::expected<T, Error>;

}  // namespace lattice
