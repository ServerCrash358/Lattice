#pragma once
// common: opaque ID types (D-MOD-02, D-ARCH-03). Never implicitly convertible
// to each other or to a raw integer -- that's the whole point.
#include <cstdint>

namespace lattice {

enum class RowId : std::uint64_t {};      // dense per-segment row identifier
enum class VectorId : std::uint64_t {};   // dense per-index vector identifier; != RowId
enum class SegmentId : std::uint32_t {};
enum class GroupId : std::uint64_t {};    // D-ARCH-03 group_id
using Lsn = std::uint64_t;                // = RocksDB sequence number (D-ARCH-04/D-STO-01)

constexpr RowId kInvalidRowId = static_cast<RowId>(~std::uint64_t{0});
constexpr VectorId kInvalidVectorId = static_cast<VectorId>(~std::uint64_t{0});

}  // namespace lattice
