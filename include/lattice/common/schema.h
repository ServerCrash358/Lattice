#pragma once
// common: collection schema types (D-ARCH-08, D-MOD-03). Population and
// persistence are owned by storage's manifest code (Phase 3); this header
// only defines the type.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lattice {

struct VectorFieldSchema {
    std::string name;
    std::uint32_t dim = 0;
    std::string model_id;  // D-ARCH-02
    enum class Metric { kInnerProduct, kL2 };
    Metric metric = Metric::kInnerProduct;  // D-VEC-01: IP by default
};

struct CollectionSchema {
    std::string name;
    std::uint32_t schema_version = 1;
    std::vector<VectorFieldSchema> dense_fields;
    std::optional<VectorFieldSchema> sparse_field;
    std::optional<VectorFieldSchema> multi_vector_field;
};

}  // namespace lattice
