// Phase 3 exit gate item 2, section (k)'s concrete follow-up: (j)/(k)
// showed SQ8 (8-bit/dim) recovers 96.95% brute-force recall@10 on real
// SIFT1M at 1M-row scale, vs RaBitQ's 11.55% -- but that was estimator-only,
// with NO graph traversal in the loop. This benchmark closes the gap: it
// re-runs bench_rabitq_sift1m_recall.cpp's exact experiment (real SIFT1M,
// 1M rows, R=64/build_list=100, codes-gated traversal, unit-normalized IP
// metric) but with SQ8 codes gating the traversal instead of RaBitQ, to
// confirm the brute-force estimator result survives real beam-search
// pruning too.
//
// index_graph.h's production VamanaIndex is RaBitQ-only as of item 2
// section f's swap (it embeds a RaBitQCodeBook member directly) -- rather
// than modifying that shared, tested production class to special-case a
// second codes format for a one-off diagnostic, this file reimplements the
// same build()/search() algorithm (greedy_search, robust_prune -- verbatim
// port of the pre-RaBitQ-swap version at commit 0ebc40c, the last commit
// where this exact algorithm used SQ8) parameterized on quant::Sq8Codes
// instead. It reuses the SAME shared, format-agnostic infrastructure the
// production class uses (NodeStore, InMemoryNodeStore, NodeRecordLayout,
// node_record.h's record accessors, visited_set.h) -- only the codes type
// embedded in the index and used for traversal-time gating differs, so
// this is a faithful re-run of the identical algorithm, not a
// reimplementation that could introduce its own bugs/differences.
//
// Usage: bench_sq8_sift1m_recall [bin_dir] [num_train] [num_queries] [r] [build_list]
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "lattice/index_graph/index_graph.h"
#include "lattice/kernels/kernels.h"
#include "lattice/quant/sq8.h"
#include "src/index_graph/in_memory_node_store.h"
#include "src/index_graph/node_record.h"
#include "src/index_graph/visited_set.h"

using namespace lattice;
using namespace lattice::index_graph;
using detail::InMemoryNodeStore;

namespace {

std::vector<float> read_f32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return data;
}
std::vector<std::int32_t> read_i32(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::int32_t> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(std::int32_t)));
    return data;
}
void normalize_in_place(std::vector<float>& v, std::size_t n, std::uint32_t dim) {
    for (std::size_t i = 0; i < n; ++i) {
        float* row = v.data() + i * dim;
        double norm2 = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d) norm2 += static_cast<double>(row[d]) * row[d];
        float inv = norm2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(norm2)) : 0.0f;
        for (std::uint32_t d = 0; d < dim; ++d) row[d] *= inv;
    }
}
double recall_at_k(const std::vector<VectorId>& got, const std::int32_t* gt, std::uint32_t k) {
    std::unordered_set<std::int32_t> exp_set;
    for (std::uint32_t i = 0; i < k; ++i) exp_set.insert(gt[i]);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, got.size()); ++i) {
        if (exp_set.count(static_cast<std::int32_t>(got[i]))) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

inline float score(const kernels::KernelTable& table, const float* a, const float* b,
                    std::uint32_t dim, Metric metric) noexcept {
    if (metric == Metric::kInnerProduct) return table.dot_ip(a, b, dim);
    return -table.dot_l2(a, b, dim);
}

// See src/index_graph/index_graph.cpp's prune_distance for the derivation:
// -dot_ip is not a genuine metric distance the way dot_l2 is, so it breaks
// the alpha-relaxed pruning test for IP metric whenever alpha != 1.
inline float prune_distance(const kernels::KernelTable& table, const float* a, const float* b,
                             std::uint32_t dim, Metric metric) noexcept {
    if (metric == Metric::kInnerProduct) return 2.0f - 2.0f * table.dot_ip(a, b, dim);
    return table.dot_l2(a, b, dim);
}

struct SearchResult {
    std::vector<std::pair<float, VectorId>> visited;
    std::uint64_t distance_computations = 0;
};

// Verbatim port of index_graph.cpp's pre-RaBitQ-swap greedy_search
// (commit 0ebc40c), parameterized on quant::Sq8Codes.
SearchResult greedy_search(const NodeStore& store, const NodeRecordLayout& layout,
                            const float* query, VectorId entry, std::uint32_t list_size,
                            Metric metric, bool enable_prefetch, bool enable_gen_visited,
                            const AdmitFn& admit, const kernels::KernelTable& kernel_table,
                            const quant::Sq8Codes* codes = nullptr,
                            std::span<const std::byte> codes_blob = {}) {
    SearchResult result;
    std::size_t n = store.node_count();

    std::unique_ptr<detail::GenerationVisitedSet> gen_visited;
    std::unique_ptr<detail::BitsetVisitedSet> bit_visited;
    if (enable_gen_visited) {
        gen_visited = std::make_unique<detail::GenerationVisitedSet>(n);
        gen_visited->reset();
    } else {
        bit_visited = std::make_unique<detail::BitsetVisitedSet>(n);
    }
    auto is_visited = [&](std::size_t idx) -> bool {
        return enable_gen_visited ? gen_visited->is_visited(idx) : bit_visited->is_visited(idx);
    };
    auto mark_visited = [&](std::size_t idx) {
        if (enable_gen_visited) gen_visited->test_and_set(idx);
        else bit_visited->test_and_set(idx);
    };

    using ScoredId = std::pair<float, VectorId>;
    std::vector<ScoredId> l_set;
    l_set.reserve(list_size + 1);
    auto l_set_insert = [&](float s, VectorId id) {
        auto pos = std::upper_bound(l_set.begin(), l_set.end(), ScoredId{s, id},
                                     [](const ScoredId& a, const ScoredId& b) { return a.first > b.first; });
        l_set.insert(pos, ScoredId{s, id});
        if (l_set.size() > list_size) l_set.pop_back();
    };

    auto entry_rec = store.read_node(entry);
    float entry_score = score(kernel_table, query, detail::record_vector(entry_rec), layout.dim, metric);
    l_set_insert(entry_score, entry);
    result.distance_computations++;

    for (;;) {
        std::size_t scan = 0;
        while (scan < l_set.size() && is_visited(static_cast<std::size_t>(l_set[scan].second))) {
            ++scan;
        }
        if (scan >= l_set.size()) break;

        auto [best_score, best_id] = l_set[scan];
        mark_visited(static_cast<std::size_t>(best_id));

        std::uint32_t degree;
        std::vector<std::uint32_t> neighbours_copy;
        float exact_score;
        {
            auto rec = store.read_node(best_id);
            exact_score = score(kernel_table, query, detail::record_vector(rec), layout.dim, metric);
            auto neighbours = detail::record_neighbours(rec, layout);
            degree = detail::record_degree(rec, layout);
            neighbours_copy.assign(neighbours.begin(), neighbours.begin() + degree);
        }
        if (!admit || admit(best_id)) result.visited.emplace_back(exact_score, best_id);

        for (std::uint32_t i = 0; i < degree; ++i) {
            std::uint32_t nbr_raw = neighbours_copy[i];
            if (nbr_raw == detail::kNoNeighbour) continue;
            auto nbr_id = static_cast<VectorId>(nbr_raw);
            if (is_visited(static_cast<std::size_t>(nbr_id))) continue;

            float s;
            if (codes != nullptr) {
                std::size_t base = static_cast<std::size_t>(nbr_id) * layout.dim;
                auto code = codes_blob.subspan(base, layout.dim);
                s = (metric == Metric::kL2) ? codes->neg_l2_to_query(query, layout.dim, code)
                                            : codes->distance_to_query(query, layout.dim, code);
            } else {
                if (enable_prefetch) store.prefetch(nbr_id);
                auto nbr_rec = store.read_node(nbr_id);
                s = score(kernel_table, query, detail::record_vector(nbr_rec), layout.dim, metric);
            }
            result.distance_computations++;
            l_set_insert(s, nbr_id);
        }
    }

    std::sort(result.visited.begin(), result.visited.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    return result;
}

std::vector<VectorId> robust_prune(VectorId p, const float* p_vec,
                                    std::vector<std::pair<float, VectorId>> candidates,
                                    const NodeStore& store, const NodeRecordLayout& layout,
                                    float alpha, std::uint32_t r, Metric metric,
                                    const kernels::KernelTable& kernel_table) {
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.second < b.second; });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                  [](auto& a, auto& b) { return a.second == b.second; }),
                      candidates.end());
    std::erase_if(candidates, [&](auto& c) { return c.second == p; });

    std::vector<VectorId> result;
    result.reserve(r);

    while (!candidates.empty() && result.size() < r) {
        auto best_it = std::max_element(candidates.begin(), candidates.end(),
                                         [](auto& a, auto& b) { return a.first < b.first; });
        VectorId pstar = best_it->second;
        result.push_back(pstar);
        candidates.erase(best_it);

        auto pstar_rec = store.read_node(pstar);
        const float* pstar_vec = detail::record_vector(pstar_rec);

        std::erase_if(candidates, [&](auto& c) {
            auto c_rec = store.read_node(c.second);
            float d_p_c = prune_distance(kernel_table, p_vec, detail::record_vector(c_rec), layout.dim, metric);
            float d_pstar_c =
                prune_distance(kernel_table, pstar_vec, detail::record_vector(c_rec), layout.dim, metric);
            return alpha * d_pstar_c <= d_p_c;
        });
    }
    return result;
}

// Minimal SQ8-flavoured stand-in for VamanaIndex, build()+search() only
// (no serialize/open -- this benchmark builds and searches in-process, no
// disk round-trip needed for this diagnostic question).
class Sq8VamanaIndex {
public:
    static std::unique_ptr<Sq8VamanaIndex> build(std::span<const float* const> vectors,
                                                   std::uint32_t dim, const quant::Sq8Codes& codes,
                                                   const VamanaBuildParams& params,
                                                   NodeRecordLayout layout) {
        std::size_t n = vectors.size();
        std::uint32_t r = params.r;

        auto store = std::make_unique<InMemoryNodeStore>(layout, n);
        const auto& kernel_table = kernels::KernelTable::resolve();

        std::mt19937 rng(12345);
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<VectorId> nbrs;
            nbrs.reserve(r);
            std::uniform_int_distribution<std::size_t> dist(0, n - 1);
            std::unordered_set<std::size_t> chosen;
            while (nbrs.size() < r && chosen.size() < n - 1) {
                std::size_t j = dist(rng);
                if (j == i || chosen.count(j)) continue;
                chosen.insert(j);
                nbrs.push_back(static_cast<VectorId>(j));
            }
            store->write_record(static_cast<VectorId>(i), vectors[i], nbrs);
        }

        std::vector<double> centroid(dim, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::uint32_t d = 0; d < dim; ++d) centroid[d] += vectors[i][d];
        for (auto& c : centroid) c /= static_cast<double>(n);
        std::vector<float> centroid_f(centroid.begin(), centroid.end());

        VectorId entry = static_cast<VectorId>(0);
        float best_entry_score = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            float s = score(kernel_table, centroid_f.data(), vectors[i], dim, params.metric);
            if (s > best_entry_score) {
                best_entry_score = s;
                entry = static_cast<VectorId>(i);
            }
        }

        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);

        for (float pass_alpha : {1.0f, params.alpha}) {
            std::shuffle(order.begin(), order.end(), rng);
            for (std::size_t idx : order) {
                auto p = static_cast<VectorId>(idx);
                const float* p_vec = vectors[idx];

                auto sr = greedy_search(*store, layout, p_vec, entry, params.build_list_size,
                                         params.metric, params.enable_prefetch,
                                         params.enable_visited_generation_counter, {}, kernel_table);

                auto p_rec = store->read_node(p);
                auto cur_nbrs = detail::record_neighbours(p_rec, layout);
                std::uint32_t cur_degree = detail::record_degree(p_rec, layout);
                std::vector<std::pair<float, VectorId>> candidates = sr.visited;
                for (std::uint32_t i = 0; i < cur_degree; ++i) {
                    if (cur_nbrs[i] == detail::kNoNeighbour) continue;
                    auto nid = static_cast<VectorId>(cur_nbrs[i]);
                    auto nrec = store->read_node(nid);
                    float s = score(kernel_table, p_vec, detail::record_vector(nrec), dim, params.metric);
                    candidates.emplace_back(s, nid);
                }

                auto new_neighbours = robust_prune(p, p_vec, std::move(candidates), *store, layout,
                                                    pass_alpha, r, params.metric, kernel_table);
                store->set_neighbours(p, new_neighbours);

                for (auto nid : new_neighbours) {
                    auto nrec = store->read_node(nid);
                    std::uint32_t ndeg = detail::record_degree(nrec, layout);
                    auto nnbrs = detail::record_neighbours(nrec, layout);
                    std::vector<std::pair<float, VectorId>> ncands;
                    const float* n_vec = detail::record_vector(nrec);
                    for (std::uint32_t i = 0; i < ndeg; ++i) {
                        if (nnbrs[i] == detail::kNoNeighbour) continue;
                        auto other = static_cast<VectorId>(nnbrs[i]);
                        if (other == p) continue;
                        auto orec = store->read_node(other);
                        float s = score(kernel_table, n_vec, detail::record_vector(orec), dim, params.metric);
                        ncands.emplace_back(s, other);
                    }
                    float s_np = score(kernel_table, n_vec, p_vec, dim, params.metric);
                    ncands.emplace_back(s_np, p);

                    if (ncands.size() > r) {
                        auto pruned = robust_prune(nid, n_vec, std::move(ncands), *store, layout,
                                                    pass_alpha, r, params.metric, kernel_table);
                        store->set_neighbours(nid, pruned);
                    } else {
                        std::vector<VectorId> ids;
                        for (auto& [s, id] : ncands) ids.push_back(id);
                        store->set_neighbours(nid, ids);
                    }
                }
            }
        }

        std::vector<std::byte> codes_blob(n * dim);
        for (std::size_t i = 0; i < n; ++i) {
            auto code = codes.encode(vectors[i], dim);
            std::copy(code.begin(), code.end(), codes_blob.begin() + i * dim);
        }

        auto index = std::make_unique<Sq8VamanaIndex>();
        index->store_ = std::move(store);
        index->codes_book_ = codes.book();
        index->codes_blob_ = std::move(codes_blob);
        index->dim_ = dim;
        index->layout_ = layout;
        index->metric_ = params.metric;
        index->entry_point_ = entry;
        return index;
    }

    // `use_codes=false` bypasses SQ8 gating entirely -- every neighbour
    // considered gets an exact fp32 score from RAM, same as the pre-item-2
    // traversal. Added as a control: if disabling codes-gating on this
    // exact same unit-normalized/IP-metric graph ALSO collapses to
    // near-zero recall, the fault is in the normalized-IP traversal setup
    // itself, not in either codes format's quantization quality.
    std::vector<VectorId> search(const float* query, std::uint32_t k, const SearchParams& params,
                                  bool use_codes = true) const {
        const auto& kernel_table = kernels::KernelTable::resolve();
        quant::Sq8Codes codes_view(codes_book_);
        const quant::Sq8Codes* codes_ptr =
            use_codes ? &codes_view : nullptr;
        std::span<const std::byte> blob_span =
            codes_ptr ? std::span<const std::byte>(codes_blob_) : std::span<const std::byte>{};

        auto sr = greedy_search(*store_, layout_, query, entry_point_, std::max(params.ef_search, k),
                                 metric_, params.enable_prefetch, params.enable_visited_generation_counter,
                                 {}, kernel_table, codes_ptr, blob_span);

        std::uint32_t take = std::min<std::uint32_t>(params.rerank_k, static_cast<std::uint32_t>(sr.visited.size()));
        std::vector<std::pair<float, VectorId>> reranked(sr.visited.begin(), sr.visited.begin() + take);
        std::sort(reranked.begin(), reranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<VectorId> result;
        result.reserve(std::min<std::size_t>(k, reranked.size()));
        for (std::size_t i = 0; i < reranked.size() && result.size() < k; ++i) result.push_back(reranked[i].second);
        return result;
    }

private:
    std::unique_ptr<NodeStore> store_;
    quant::Sq8CodeBook codes_book_;
    std::vector<std::byte> codes_blob_;
    std::uint32_t dim_ = 0;
    NodeRecordLayout layout_;
    Metric metric_ = Metric::kInnerProduct;
    VectorId entry_point_{};
};

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "bench/data/sift1m_bin";
    std::size_t num_train_arg = argc > 2 ? std::stoul(argv[2]) : 0;
    std::size_t num_queries_arg = argc > 3 ? std::stoul(argv[3]) : 100;
    std::uint32_t r = argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 64;
    std::uint32_t build_list = argc > 5 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 100;
    const bool use_l2 = argc > 6 && std::string(argv[6]) == "l2";  // raw vectors + L2 (SIFT-native)

    std::ifstream meta(data_dir + "/meta.txt");
    std::size_t n_train, dim, n_test, dim2, n_gt, gt_k;
    meta >> n_train >> dim >> n_test >> dim2 >> n_gt >> gt_k;

    std::size_t num_train = num_train_arg == 0 ? n_train : std::min(num_train_arg, n_train);
    std::size_t num_queries = std::min(num_queries_arg, n_test);
    printf("Real SIFT1M, SQ8 codes-gated traversal: dim=%zu num_train=%zu num_queries=%zu R=%u "
           "build_list=%u\n",
           dim, num_train, num_queries, r, build_list);

    auto train = read_f32(data_dir + "/train.f32", num_train * dim);
    auto test = read_f32(data_dir + "/test.f32", num_queries * dim);
    auto gt = read_i32(data_dir + "/neighbors.i32", n_gt * gt_k);

    if (!use_l2) normalize_in_place(train, num_train, static_cast<std::uint32_t>(dim));
    if (!use_l2) normalize_in_place(test, num_queries, static_cast<std::uint32_t>(dim));

    std::vector<const float*> ptrs(num_train);
    for (std::size_t i = 0; i < num_train; ++i) ptrs[i] = train.data() + i * dim;

    auto sq8_book = quant::Sq8CodeBook::train(ptrs, static_cast<std::uint32_t>(dim));
    if (!sq8_book) {
        printf("SQ8 train failed: %s\n", sq8_book.error().message.c_str());
        return 1;
    }
    quant::Sq8Codes sq8_codes(*sq8_book);

    auto layout = NodeRecordLayout::make(static_cast<std::uint32_t>(dim), ElemType::kFloat32, r);
    if (!layout) {
        printf("Layout rejected: %s\n", layout.error().message.c_str());
        return 1;
    }

    VamanaBuildParams build_params;
    build_params.r = r;
    build_params.alpha = 1.2f;
    build_params.build_list_size = build_list;
    build_params.metric = use_l2 ? Metric::kL2 : Metric::kInnerProduct;

    printf("Building Vamana index (in-RAM, SQ8 codes-gated): n=%zu dim=%zu R=%u build_list=%u ...\n",
           num_train, dim, r, build_list);
    auto t0 = std::chrono::steady_clock::now();
    auto index = Sq8VamanaIndex::build(ptrs, static_cast<std::uint32_t>(dim), sq8_codes, build_params, *layout);
    auto t1 = std::chrono::steady_clock::now();
    double build_secs = std::chrono::duration<double>(t1 - t0).count();
    printf("Build done in %.1f s (%.0f vectors/s)\n", build_secs, num_train / build_secs);

    // Single-threaded latency + recall@1/@10 sweep (the picture's Recall@1,
    // Recall@10 and Query Latency rows).
    printf("\n%10s %10s %10s %10s %10s\n", "ef_search", "recall@1", "recall@10", "p50_ms", "p99_ms");
    for (std::uint32_t ef : {64u, 128u, 256u, 512u, 1024u, 2048u, 4096u}) {
        SearchParams sp;
        sp.ef_search = ef;
        sp.rerank_k = std::max<std::uint32_t>(40, ef / 2);
        constexpr std::uint32_t k = 10;

        std::vector<double> latencies_ms;
        latencies_ms.reserve(num_queries);
        double total_recall10 = 0.0, total_recall1 = 0.0;
        for (std::size_t q = 0; q < num_queries; ++q) {
            const float* query = test.data() + q * dim;
            auto qt0 = std::chrono::steady_clock::now();
            auto result = index->search(query, k, sp);
            auto qt1 = std::chrono::steady_clock::now();
            latencies_ms.push_back(std::chrono::duration<double, std::milli>(qt1 - qt0).count());
            total_recall10 += recall_at_k(result, gt.data() + q * gt_k, k);
            if (!result.empty() && static_cast<std::int32_t>(result[0]) == gt[q * gt_k]) total_recall1 += 1.0;
        }
        std::sort(latencies_ms.begin(), latencies_ms.end());
        double p50 = latencies_ms[latencies_ms.size() / 2];
        double p99 = latencies_ms[static_cast<std::size_t>(0.99 * (latencies_ms.size() - 1))];
        printf("%10u %10.4f %10.4f %10.2f %10.2f\n", ef, total_recall1 / num_queries,
               total_recall10 / num_queries, p50, p99);
    }

    // Multi-threaded throughput (the picture's QPS row). search() is const and
    // allocates all per-query state locally, so concurrent calls on the shared
    // read-only index are race-free (D-CON-02).
    if (argc > 7) return 0;  // 7th arg = skip QPS
    unsigned hw = std::thread::hardware_concurrency();
    printf("\nThroughput (hardware threads=%u), each thread cycles the %zu queries for ~10s:\n", hw,
           num_queries);
    printf("%10s %8s %12s\n", "ef_search", "threads", "QPS");
    for (std::uint32_t ef : {64u, 128u, 256u, 512u, 1024u}) {
        SearchParams sp;
        sp.ef_search = ef;
        sp.rerank_k = std::max<std::uint32_t>(40, ef / 2);
        for (unsigned nthreads : {1u, hw}) {
            std::atomic<std::uint64_t> done{0};
            std::atomic<bool> stop{false};
            std::vector<std::thread> threads;
            auto start = std::chrono::steady_clock::now();
            for (unsigned t = 0; t < nthreads; ++t) {
                threads.emplace_back([&, t] {
                    std::size_t q = t * 7;
                    while (!stop.load(std::memory_order_relaxed)) {
                        const float* query = test.data() + (q % num_queries) * dim;
                        auto r = index->search(query, 10, sp);
                        (void)r;
                        ++q;
                        done.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));
            stop.store(true);
            for (auto& th : threads) th.join();
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("%10u %8u %12.0f\n", ef, nthreads, static_cast<double>(done.load()) / secs);
            if (nthreads == hw && hw == 1) break;
        }
    }
    return 0;
}
