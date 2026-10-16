#include "lattice/index_flat/index_flat.h"

#include "lattice/common/top_k_heap.h"
#include "lattice/kernels/kernels.h"

namespace lattice::index_flat {

Result<std::vector<VectorId>> search(const FlatView& view, const float* query, std::uint32_t k,
                                      const AdmitFn& admit) {
    if (view.dim == 0) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "FlatView.dim == 0"));
    }
    if (k == 0) return std::vector<VectorId>{};

    const auto& table = kernels::KernelTable::resolve();
    TopKHeap<float, VectorId> heap(k);

    for (std::size_t i = 0; i < view.count; ++i) {
        auto vid = static_cast<VectorId>(i);
        if (admit && !admit(vid)) continue;
        const float* candidate = view.base + i * view.stride_elems;
        float score = table.dot_ip(query, candidate, view.dim);
        heap.try_insert(score, vid);
    }

    std::vector<VectorId> result;
    result.reserve(heap.size());
    for (auto& [score, id] : heap.extract_sorted()) {
        (void)score;
        result.push_back(id);
    }
    return result;
}

Result<void> search_batched(const FlatView& view, std::span<const float* const> queries,
                             std::uint32_t k, std::span<std::vector<VectorId>> out_results,
                             const AdmitFn& admit) {
    if (out_results.size() != queries.size()) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument,
                                            "out_results.size() must equal queries.size()"));
    }
    for (std::size_t i = 0; i < queries.size(); ++i) {
        auto r = search(view, queries[i], k, admit);
        if (!r) return std::unexpected(r.error());
        out_results[i] = std::move(*r);
    }
    return {};
}

}  // namespace lattice::index_flat
