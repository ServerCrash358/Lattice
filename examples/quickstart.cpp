// Lattice quick start: build an index, save it, reopen it disk-resident (zero-copy mmap),
// run a plain search and a filtered search. Self-contained: uses random vectors.
//
//   cmake --preset release && cmake --build build/release --target example_quickstart
//   ./build/release/examples/example_quickstart
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "lattice/quant/sq8.h"
#include "lattice/segment/segment.h"

using namespace lattice;
using namespace lattice::index_graph;

int main() {
    constexpr std::uint32_t dim = 64;
    constexpr std::size_t n = 20000;

    // 1. Some vectors (replace with your embeddings).
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::vector<std::vector<float>> data(n, std::vector<float>(dim));
    std::vector<const float*> ptrs(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& x : data[i]) x = gauss(rng);
        ptrs[i] = data[i].data();
    }

    // 2. Train the 8-bit codes that stay in RAM, pick a compact node layout, and build the graph.
    auto book = quant::Sq8CodeBook::train(ptrs, dim);
    quant::Sq8Codes codes(*book);
    auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, /*R=*/32);
    VamanaBuildParams params;
    params.r = 32;
    params.build_list_size = 100;
    params.metric = Metric::kL2;  // build_threads defaults to all cores
    auto built = VamanaIndex::build(ptrs, dim, codes, params, *layout);
    if (!built) { std::printf("build failed: %s\n", built.error().message.c_str()); return 1; }

    // 3. Save, then reopen disk-resident: node records are served zero-copy from the mmap'd file.
    const char* path = "quickstart.lattice";
    if (!built->serialize_to_disk(path)) { std::printf("save failed\n"); return 1; }
    auto index = segment::open_disk_resident_vamana(path, segment::NodeAccess::kMmap, /*preload=*/true);
    if (!index) { std::printf("open failed: %s\n", index.error().message.c_str()); return 1; }

    // 4. Plain nearest-neighbour search, checked against an exact brute-force scan.
    SearchParams sp;
    sp.ef_search = 128;  // higher = better recall, slower
    auto exact_top5 = [&](const float* q) {
        std::vector<std::pair<float, std::size_t>> all;
        for (std::size_t i = 0; i < n; ++i) {
            float d = 0;
            for (std::uint32_t k = 0; k < dim; ++k) d += (q[k] - data[i][k]) * (q[k] - data[i][k]);
            all.emplace_back(d, i);
        }
        std::partial_sort(all.begin(), all.begin() + 5, all.end());
        std::vector<std::size_t> ids;
        for (int j = 0; j < 5; ++j) ids.push_back(all[j].second);
        return ids;
    };
    double recall = 0;
    constexpr int num_queries = 20;
    for (int qi = 0; qi < num_queries; ++qi) {
        std::vector<float> q = data[1000 + qi];  // a slightly perturbed copy of a data vector
        for (auto& x : q) x += 0.05f * gauss(rng);
        auto approx = index->search(q.data(), 5, sp);
        auto exact = exact_top5(q.data());
        for (auto id : *approx)
            for (auto e : exact) recall += (static_cast<std::size_t>(id) == e) ? 1.0 / 5 : 0.0;
    }
    std::printf("recall@5 vs brute force over %d queries: %.2f\n", num_queries, recall / num_queries);

    // 5. Filtered search: only even ids are allowed. You pass the exact matching set (sorted).
    std::vector<VectorId> matching;
    for (std::size_t i = 0; i < n; i += 2) matching.push_back(static_cast<VectorId>(i));
    VamanaIndex::FilteredParams fp;
    fp.base = sp;
    fp.matching_ids = matching;
    auto admit = [](VectorId id) { return static_cast<std::uint64_t>(id) % 2 == 0; };
    auto filtered = index->search_filtered(data[123].data(), 5, fp, admit);
    std::printf("top-5 even ids for vector 123:");
    for (auto id : *filtered) std::printf(" %llu", static_cast<unsigned long long>(id));
    std::printf("\n");
    std::remove(path);
    return 0;
}
