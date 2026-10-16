#include "lattice/index_graph/index_graph.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdio>
#include <filesystem>

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>

#include "lattice/kernels/kernels.h"
#include "src/index_graph/in_memory_node_store.h"
#include "src/index_graph/node_record.h"
#include "src/index_graph/visited_set.h"

namespace lattice::index_graph {

using detail::InMemoryNodeStore;

// --- NodeRecordLayout ---------------------------------------------------

std::size_t NodeRecordLayout::max_r_for(std::uint32_t dim_, ElemType t) const noexcept {
    std::size_t vec_bytes = static_cast<std::size_t>(dim_) * (t == ElemType::kFloat32 ? 4 : 2);
    std::size_t overhead = sizeof(std::uint32_t);  // degree field
    if (sector_size <= overhead + vec_bytes) return 0;
    return (sector_size - overhead - vec_bytes) / sizeof(std::uint32_t);
}

Result<NodeRecordLayout> NodeRecordLayout::make(std::uint32_t dim, ElemType t, std::uint32_t r,
                                                 std::uint32_t sector_size) noexcept {
    if (dim == 0) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "dim must be > 0"));
    }
    NodeRecordLayout layout;
    layout.dim = dim;
    layout.elem_type = t;
    layout.degree_cap_r = r;
    layout.sector_size = sector_size;

    std::size_t max_r = layout.max_r_for(dim, t);
    if (r > max_r) {
        return std::unexpected(Error::make(
            ErrorCode::kInvalidArgument,
            "R exceeds sector capacity for this dim/elem_type (finding 4's R_max formula): "
            "requested R=" +
                std::to_string(r) + " but max is " + std::to_string(max_r)));
    }
    return layout;
}

// --- Metric-aware scoring (higher is always better) ----------------------

namespace {

// Takes the resolved KernelTable by reference rather than calling
// KernelTable::resolve() internally -- that function is cheap once resolved
// (a static-local guard check) but this is called millions of times per
// query, and hoisting the resolve to one call per search measurably helped.
inline float score(const kernels::KernelTable& table, const float* a, const float* b,
                    std::uint32_t dim, Metric metric) noexcept {
    if (metric == Metric::kInnerProduct) {
        return table.dot_ip(a, b, dim);
    }
    // L2: smaller squared distance is better, so negate to keep "higher is
    // better" uniform across the rest of the algorithm (D-VEC-01's L2
    // kernel is used only for SIFT/GIST parity benchmarks).
    return -table.dot_l2(a, b, dim);
}

// A genuine (non-negative, translation-consistent) "distance" for
// robust_prune's alpha-relaxed pruning test, which assumes a real metric
// space -- unlike `score()` above (which only needs a "higher is better"
// ordering, fine for greedy_search's ranking). For L2, `dot_l2` already IS
// squared Euclidean distance, so `-score(...)` was correct there. For IP,
// `-score(...)` is just `-dot_ip`, which is NOT a metric distance in the
// sense the pruning test needs: for unit-normalized vectors (D-VEC-01:
// "Normalise at ingest; inner product everywhere internally"), the true
// squared Euclidean distance is `2 - 2*dot_ip` -- an AFFINE function of
// `dot_ip` with a nonzero additive constant, not merely a sign flip.
// Plugging `-dot_ip` directly into `alpha*d(p*,c) <= d(p,c)` silently
// drops that constant, which (for alpha != 1, i.e. every build's second
// pass) shifts the pruning threshold and makes IP-metric pruning
// systematically more aggressive than the requested alpha intends --
// found via spec/phase-3.md item 2 section (l): a control test showed a
// 1M-row real-SIFT1M recall collapse was byte-identical whether traversal
// used RaBitQ, SQ8, or NO quantization at all, ruling out codes quality
// entirely and pointing at the graph's own edges (built via this
// function) as the actual defect.
inline bool all_finite(const float* v, std::uint32_t dim) noexcept {
    for (std::uint32_t d = 0; d < dim; ++d)
        if (!std::isfinite(v[d])) return false;
    return true;
}

inline float prune_distance(const kernels::KernelTable& table, const float* a, const float* b,
                             std::uint32_t dim, Metric metric) noexcept {
    if (metric == Metric::kInnerProduct) {
        return 2.0f - 2.0f * table.dot_ip(a, b, dim);
    }
    return table.dot_l2(a, b, dim);
}

// One greedy best-first search pass over the graph, used both by build
// (finding candidate neighbours for a new point) and by query search.
// Returns the visited set in descending-score order (best first).
struct SearchResult {
    std::vector<std::pair<float, VectorId>> visited;  // score, id -- descending
    std::uint64_t distance_computations = 0;
};

// L_set is DiskANN's terminology: a single candidate list, capped at
// `list_size`, containing both expanded and not-yet-expanded points, always
// kept as (approximately) the list_size closest points seen so far. This is
// the standard efficient formulation (matches hnswlib's ef-search list) --
// an earlier draft of this function re-sorted the whole frontier on every
// expansion (O(L log L) per step), which would have made even a 100K build
// take hours. The current version keeps a flat sorted vector (O(L) insert
// via memmove, no per-node heap allocation) rather than a std::set (O(log L)
// insert but real per-op overhead from tree-node churn) -- see the l_set
// declaration below for why the flat vector measured faster despite the
// worse big-O at the L~64-200 sizes actually used here.
// `codes`/`codes_blob`, when both non-null, gate neighbour-scoring during
// traversal: a not-yet-visited neighbour is scored from its RAM-resident
// SQ8 code (no disk read) instead of reading its full node record just to
// compute an exact score. Only nodes actually chosen for EXPANSION still
// read their node record from disk -- unavoidable, since that's the only
// way to get a node's neighbour list, but that same read also yields its
// exact fp32 vector "for free", which is what gets stored in
// `result.visited` (see below) so the final ranking is still exact, not
// approximate, despite traversal being codes-gated. Works for both metrics:
// IP uses Sq8Codes::distance_to_query, L2 uses Sq8Codes::neg_l2_to_query
// (both "higher is better", matching score()).
SearchResult greedy_search(const NodeStore& store, const NodeRecordLayout& layout,
                            const float* query, VectorId entry, std::uint32_t list_size,
                            Metric metric, bool enable_prefetch, bool enable_gen_visited,
                            const AdmitFn& admit, const kernels::KernelTable& kernel_table,
                            const quant::Sq8Codes* codes = nullptr,
                            std::span<const std::byte> codes_blob = {}) {
    SearchResult result;
    std::size_t n = store.node_count();

    thread_local detail::GenerationVisitedSet tls_gen_visited(0);
    detail::GenerationVisitedSet* gen_visited = nullptr;
    std::unique_ptr<detail::BitsetVisitedSet> bit_visited;
    if (enable_gen_visited) {
        tls_gen_visited.prepare(n);
        gen_visited = &tls_gen_visited;
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

    // Flat, sorted-descending candidate list capped at list_size. Insert is
    // O(list_size) via memmove, but for the L~64-200 sizes actually used
    // here that beats a std::set: no per-node heap allocation, no pointer
    // chasing, and the whole structure stays in a handful of cache lines --
    // this replaced an earlier std::set-based version that was ~8x slower
    // than hnswlib at matched recall, dominated by red-black-tree node
    // churn (one alloc/dealloc per neighbour considered, of which there are
    // list_size * avg_degree per query).
    // The visited set above marks a node when it is first DISCOVERED, so each node is scored
    // and inserted into the list at most once; `expanded` records which list entries have had
    // their neighbours explored. (Marking only on expansion let the same node occupy several
    // list slots, shrinking the effective beam below list_size and re-scoring it each time.)
    struct Candidate {
        float score;
        VectorId id;
        bool expanded;
    };
    std::vector<Candidate> l_set;
    l_set.reserve(list_size + 1);
    auto l_set_insert = [&](float s, VectorId id) {
        // Full list and not better than the current worst: it would be inserted last and
        // popped immediately, so skip the memmove entirely.
        if (l_set.size() >= list_size && s <= l_set.back().score) return;
        auto pos = std::upper_bound(l_set.begin(), l_set.end(), s,
                                    [](float v, const Candidate& c) { return v > c.score; });
        l_set.insert(pos, Candidate{s, id, false});
        if (l_set.size() > list_size) l_set.pop_back();
    };

    auto entry_rec = store.read_node(entry);
    float entry_score = score(kernel_table, query, detail::record_vector(entry_rec), layout.dim, metric);
    mark_visited(static_cast<std::size_t>(entry));
    l_set_insert(entry_score, entry);
    result.distance_computations++;

    for (;;) {
        // Scan from the front each time (NOT a monotonic cursor): a newly discovered neighbour
        // can score better than an already-expanded node and land before it.
        std::size_t scan = 0;
        while (scan < l_set.size() && l_set[scan].expanded) ++scan;
        if (scan >= l_set.size()) break;  // nothing left to expand

        l_set[scan].expanded = true;
        const VectorId best_id = l_set[scan].id;

        // Copy the neighbour list out before any further read_node() call: a disk-backed
        // store's span is only valid until its next read. The same read yields the exact fp32
        // vector, so `result.visited` holds exact scores even though traversal is codes-gated.
        std::uint32_t degree;
        std::vector<std::uint32_t> neighbours_copy;
        float exact_score;
        {
            auto rec = store.read_node(best_id);
            exact_score = score(kernel_table, query, detail::record_vector(rec), layout.dim, metric);
            degree = std::min(detail::record_degree(rec, layout), layout.degree_cap_r);
            detail::copy_neighbours(rec, layout, degree, neighbours_copy);
        }
        if (!admit || admit(best_id)) result.visited.emplace_back(exact_score, best_id);

        for (std::uint32_t i = 0; i < degree; ++i) {
            std::uint32_t nbr_raw = neighbours_copy[i];
            // kNoNeighbour, or an id past the end (only possible in a corrupt index file): skip it
            // before it can index the visited set or the codes blob out of bounds.
            if (nbr_raw == detail::kNoNeighbour || nbr_raw >= n) continue;
            auto nbr_id = static_cast<VectorId>(nbr_raw);
            if (is_visited(static_cast<std::size_t>(nbr_id))) continue;
            mark_visited(static_cast<std::size_t>(nbr_id));

            float s;
            if (codes != nullptr) {
                // RAM-resident approximate score: no node-record read for a merely considered
                // neighbour (only expanded nodes pay a read).
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

// RobustPrune (Vamana build): greedily pick the best-scoring remaining
// candidate, keep it, then drop any candidate that candidate "covers"
// adequately (alpha-relaxed domination), until R neighbours are chosen.
std::vector<VectorId> robust_prune(VectorId p, const float* p_vec,
                                    std::vector<std::pair<float, VectorId>> candidates,
                                    const NodeStore& store, const NodeRecordLayout& layout,
                                    float alpha, std::uint32_t r, Metric metric,
                                    const kernels::KernelTable& kernel_table, bool saturate) {
    // Remove p itself and de-duplicate by id (keep the best score seen).
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.second < b.second; });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                  [](auto& a, auto& b) { return a.second == b.second; }),
                      candidates.end());
    std::erase_if(candidates, [&](auto& c) { return c.second == p; });

    std::vector<VectorId> result;
    result.reserve(r);
    std::vector<std::pair<float, VectorId>> rejected;  // alpha-pruned, kept for saturation

    while (!candidates.empty() && result.size() < r) {
        // Pick candidate closest (best score) to p.
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
            // Classic alpha*d(p*,c) <= d(p,c) pruning rule, on a genuine
            // metric distance (see prune_distance's doc comment for why
            // this must NOT be `-score(...)` for IP metric).
            bool drop = alpha * d_pstar_c <= d_p_c;
            if (drop && saturate) rejected.push_back(c);
            return drop;
        });
    }
    // Saturate (DiskANN 'saturate_graph'): if alpha-pruning left slots unused, fill them with the
    // closest rejected candidates. Keeps nodes better connected, which matters most on hard,
    // high-intrinsic-dimension data where plain alpha-pruning leaves many nodes barely reachable.
    if (saturate && result.size() < r) {
        std::sort(rejected.begin(), rejected.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (auto& [s_, id] : rejected) {
            if (result.size() >= r) break;
            result.push_back(id);
        }
    }
    return result;
}

}  // namespace

// --- Build ----------------------------------------------------------------

Result<VamanaIndex> VamanaIndex::build(std::span<const float* const> vectors, std::uint32_t dim,
                                        const quant::Sq8Codes& codes,
                                        const VamanaBuildParams& params, NodeRecordLayout layout) {
    if (vectors.empty()) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "empty vector set"));
    }
    if (layout.dim != dim) {
        return std::unexpected(
            Error::make(ErrorCode::kDimensionMismatch, "layout.dim must match dim"));
    }
    // Node records hold exactly layout.degree_cap_r neighbour slots; a larger r would make every
    // neighbour-list write overrun its record (heap corruption).
    if (params.r == 0 || params.r > layout.degree_cap_r) {
        return std::unexpected(Error::make(
            ErrorCode::kInvalidArgument,
            "VamanaBuildParams::r (" + std::to_string(params.r) + ") must be in [1, layout.degree_cap_r=" +
                std::to_string(layout.degree_cap_r) + "]"));
    }
    if (layout.elem_type != ElemType::kFloat32) {
        return std::unexpected(Error::make(ErrorCode::kUnsupported, "only ElemType::kFloat32 is implemented"));
    }
    std::size_t n = vectors.size();
    if (n >= static_cast<std::size_t>(detail::kNoNeighbour)) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument,
                                           "too many vectors: neighbour ids are stored as 32-bit"));
    }
    std::uint32_t r = params.r;
    // NaN/Inf scores break the strict weak ordering every sort/upper_bound below relies on.
    for (std::size_t i = 0; i < n; ++i) {
        if (!all_finite(vectors[i], dim)) {
            return std::unexpected(Error::make(ErrorCode::kInvalidArgument,
                                               "vector " + std::to_string(i) + " contains NaN/Inf"));
        }
    }

    auto store = std::make_unique<InMemoryNodeStore>(layout, n);
    const auto& kernel_table = kernels::KernelTable::resolve();

    // 1. Random initial graph: each node gets R random neighbours (D-IDX-02's
    // build seed). Simple, deterministic given a fixed seed for reproducibility.
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

    // 2. Entry point: the point closest to the centroid (DiskANN's
    // approximate-medoid heuristic), computed once with exact fp32.
    std::vector<double> centroid(dim, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::uint32_t d = 0; d < dim; ++d) centroid[d] += vectors[i][d];
    }
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

    // 3. Two construction passes (Vamana): alpha=1 first (NSG-style, tight
    // pruning), then the user's alpha (more relaxed, better long-range
    // edges for recall).
    //
    // Parallel build (DiskANN-style): workers claim nodes from an atomic
    // counter; a striped mutex guards each node's neighbour-list
    // read-modify-write. Greedy-search reads of other nodes' lists are
    // deliberately unlocked: a slot only ever holds a valid id or the empty
    // sentinel (aligned 4-byte stores), so a concurrent reader sees a slightly
    // stale but valid list. build_threads=1 reproduces the old sequential,
    // deterministic build exactly.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    unsigned nthreads = params.build_threads != 0
                            ? params.build_threads
                            : std::max(1u, std::thread::hardware_concurrency());
    constexpr std::size_t kLockStripes = 8192;
    std::vector<std::mutex> locks(kLockStripes);
    auto lock_of = [&](VectorId id) -> std::mutex& {
        return locks[static_cast<std::size_t>(id) % kLockStripes];
    };

    auto process = [&](std::size_t idx, float pass_alpha) {
        auto p = static_cast<VectorId>(idx);
        const float* p_vec = vectors[idx];

        auto sr = greedy_search(*store, layout, p_vec, entry, params.build_list_size,
                                 params.metric, params.enable_prefetch,
                                 params.enable_visited_generation_counter, {}, kernel_table);

        std::vector<std::uint32_t> cur_nbrs;
        {
            std::lock_guard<std::mutex> g(lock_of(p));
            auto p_rec = store->read_node(p);
            auto nb = detail::record_neighbours(p_rec, layout);
            std::uint32_t cur_degree = detail::record_degree(p_rec, layout);
            cur_nbrs.assign(nb.begin(), nb.begin() + cur_degree);
        }
        std::vector<std::pair<float, VectorId>> candidates = sr.visited;
        for (std::uint32_t nb : cur_nbrs) {
            if (nb == detail::kNoNeighbour) continue;
            auto nid = static_cast<VectorId>(nb);
            auto nrec = store->read_node(nid);
            float s = score(kernel_table, p_vec, detail::record_vector(nrec), dim, params.metric);
            candidates.emplace_back(s, nid);
        }

        auto new_neighbours = robust_prune(p, p_vec, std::move(candidates), *store, layout,
                                            pass_alpha, r, params.metric, kernel_table, params.saturate_graph);
        {
            std::lock_guard<std::mutex> g(lock_of(p));
            store->set_neighbours(p, new_neighbours);
        }

        // Make the edge bidirectional: add p to each new neighbour's list,
        // re-pruning that neighbour if it now exceeds R.
        for (auto nid : new_neighbours) {
            std::lock_guard<std::mutex> g(lock_of(nid));
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
                                            pass_alpha, r, params.metric, kernel_table, params.saturate_graph);
                store->set_neighbours(nid, pruned);
            } else {
                std::vector<VectorId> ids;
                for (auto& [s, id] : ncands) ids.push_back(id);
                store->set_neighbours(nid, ids);
            }
        }
    };

    for (float pass_alpha : {1.0f, params.alpha}) {
        std::shuffle(order.begin(), order.end(), rng);
        if (nthreads == 1) {
            for (std::size_t idx : order) process(idx, pass_alpha);
        } else {
            std::atomic<std::size_t> next{0};
            constexpr std::size_t kChunk = 32;
            std::vector<std::thread> workers;
            for (unsigned t = 0; t < nthreads; ++t) {
                workers.emplace_back([&] {
                    for (;;) {
                        std::size_t begin = next.fetch_add(kChunk, std::memory_order_relaxed);
                        if (begin >= n) break;
                        std::size_t end = std::min(begin + kChunk, n);
                        for (std::size_t i = begin; i < end; ++i) process(order[i], pass_alpha);
                    }
                });
            }
            for (auto& w : workers) w.join();
        }
    }

    // Encode SQ8 codes for every point (D-VEC-05: codes array always exists
    // in RAM). Build itself still scores over InMemoryNodeStore with exact
    // fp32 (those reads are cheap RAM accesses, not disk I/O, so there's no
    // latency win to gating them -- and changing build's scoring risks
    // silently regressing existing recall numbers for no benefit). These
    // codes are what search() uses for codes-gated traversal instead,
    // including after a disk round-trip (see serialize_to_disk/open below).
    std::vector<std::byte> codes_blob(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        auto code = codes.encode(vectors[i], dim);
        std::copy(code.begin(), code.end(), codes_blob.begin() + i * dim);
    }

    VamanaIndex index;
    index.store_ = std::move(store);
    index.codes_book_ = codes.book();
    index.codes_blob_ = std::move(codes_blob);
    index.dim_ = dim;
    index.node_count_ = n;
    index.layout_ = layout;
    index.metric_ = params.metric;
    index.entry_point_ = entry;
    return index;
}

// --- Search -----------------------------------------------------------------

Result<std::vector<VectorId>> VamanaIndex::search(const float* query, std::uint32_t k,
                                                    const SearchParams& params, AdmitFn admit,
                                                    Stats* out_stats) const {
    if (k == 0) return std::vector<VectorId>{};
    if (query == nullptr || !all_finite(query, dim_)) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "query is null or contains NaN/Inf"));
    }

    const auto& kernel_table = kernels::KernelTable::resolve();

    // Codes-gated traversal (see greedy_search's doc comment), both metrics.
    quant::Sq8Codes codes_view(codes_book_);
    const quant::Sq8Codes* codes_ptr = params.use_codes ? &codes_view : nullptr;
    std::span<const std::byte> blob_span = std::span<const std::byte>(codes_blob_);

    auto sr = greedy_search(*store_, layout_, query, entry_point_,
                             std::max(params.ef_search, k), metric_, params.enable_prefetch,
                             params.enable_visited_generation_counter, admit, kernel_table,
                             codes_ptr, blob_span);
    if (out_stats) {
        out_stats->candidates_visited = sr.visited.size();
        out_stats->distance_computations = sr.distance_computations;
    }

    // Rerank the top rerank_k candidates with exact fp32 (D-VEC-03). In
    // Phase 2's in-RAM backend the exact vector is already what traversal
    // used, so this is a no-op re-score for consistency with the interface
    // Phase 3's disk backend will actually need it for.
    // At least k: rerank_k only widens the pool; capping at rerank_k returned fewer than k
    // results whenever k > rerank_k (e.g. k=100 with the default rerank_k=40).
    std::uint32_t take = std::min<std::uint32_t>(std::max(params.rerank_k, k),
                                                 static_cast<std::uint32_t>(sr.visited.size()));
    std::vector<std::pair<float, VectorId>> reranked(sr.visited.begin(), sr.visited.begin() + take);

    std::sort(reranked.begin(), reranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<VectorId> result;
    result.reserve(std::min<std::size_t>(k, reranked.size()));
    for (std::size_t i = 0; i < reranked.size() && result.size() < k; ++i) {
        result.push_back(reranked[i].second);
    }
    return result;
}

// --- Filter-aware search ------------------------------------------------------

Result<std::vector<VectorId>> VamanaIndex::search_filtered(const float* query, std::uint32_t k,
                                                             const FilteredParams& params,
                                                             AdmitFn admit,
                                                             Stats* out_stats) const {
    if (k == 0 || params.matching_ids.empty()) return std::vector<VectorId>{};
    if (query == nullptr || !all_finite(query, dim_)) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "query is null or contains NaN/Inf"));
    }
    // matching_ids is trusted as the exact match set; an id past the end would index outside the
    // codes blob (undefined behaviour), so reject it.
    for (VectorId id : params.matching_ids) {
        if (static_cast<std::size_t>(id) >= node_count_) {
            return std::unexpected(Error::make(ErrorCode::kInvalidArgument,
                                               "matching_ids contains an id >= index size"));
        }
    }
    const std::size_t m = params.matching_ids.size();

    if (m > params.scan_max) {
        // Graph path: widen the beam so that ~ef matching nodes get reached.
        double selectivity = static_cast<double>(m) / static_cast<double>(node_count_);
        auto widened = static_cast<std::uint32_t>(std::min<double>(
            params.max_ef, std::ceil(static_cast<double>(params.base.ef_search) / selectivity)));
        SearchParams sp = params.base;
        sp.ef_search = std::max(widened, params.base.ef_search);
        sp.rerank_k = std::max(sp.rerank_k, std::min(sp.ef_search, params.max_ef));
        return search(query, k, sp, std::move(admit), out_stats);
    }

    // Scan path: SQ8 codes of the matching vectors only (RAM, no disk reads),
    // then exact rerank of the best `scan_rerank` from their node records.
    const auto& kernel_table = kernels::KernelTable::resolve();
    quant::Sq8Codes codes_view(codes_book_);
    std::vector<std::pair<float, VectorId>> scored;
    scored.reserve(m);
    for (VectorId id : params.matching_ids) {
        auto code = std::span<const std::byte>(codes_blob_).subspan(
            static_cast<std::size_t>(id) * dim_, dim_);
        float s = metric_ == Metric::kL2 ? codes_view.neg_l2_to_query(query, dim_, code)
                                           : codes_view.distance_to_query(query, dim_, code);
        scored.emplace_back(s, id);
    }
    std::size_t pool = std::min<std::size_t>(std::max<std::uint32_t>(params.scan_rerank, k), scored.size());
    std::partial_sort(scored.begin(), scored.begin() + pool, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<float, VectorId>> reranked;
    reranked.reserve(pool);
    for (std::size_t i = 0; i < pool; ++i) {
        auto rec = store_->read_node(scored[i].second);
        reranked.emplace_back(score(kernel_table, query, detail::record_vector(rec), dim_, metric_),
                              scored[i].second);
    }
    std::sort(reranked.begin(), reranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<VectorId> result;
    for (std::size_t i = 0; i < reranked.size() && result.size() < k; ++i) result.push_back(reranked[i].second);
    if (out_stats) {
        out_stats->candidates_visited = pool;
        out_stats->distance_computations = m + pool;
    }
    return result;
}

// --- open() / serialize_to_disk() (Phase 3 disk-resident backend) ---------

Result<void> VamanaIndex::serialize_to_disk(const std::string& path) const {
    const std::string tmp_path = path + ".tmp";
    std::FILE* file = std::fopen(tmp_path.c_str(), "wb");
    if (!file) {
        return std::unexpected(Error::make(ErrorCode::kIoError, "failed to open " + tmp_path + " for write"));
    }
    auto fail = [&](const std::string& what) -> Result<void> {
        std::fclose(file);
        std::remove(tmp_path.c_str());
        return std::unexpected(Error::make(ErrorCode::kIoError, what + " (" + tmp_path + ")"));
    };

    std::byte header[DiskFormat::kHeaderBytes];
    std::size_t off = 0;
    auto put32 = [&](std::uint32_t v) { std::memcpy(header + off, &v, 4); off += 4; };
    auto put64 = [&](std::uint64_t v) { std::memcpy(header + off, &v, 8); off += 8; };
    put32(DiskFormat::kMagic);
    put32(DiskFormat::kVersion);
    put32(DiskFormat::kCodesSq8);
    put32(dim_);
    put32(static_cast<std::uint32_t>(layout_.elem_type));
    put32(layout_.degree_cap_r);
    put32(layout_.sector_size);
    put64(static_cast<std::uint64_t>(node_count_));
    put64(static_cast<std::uint64_t>(entry_point_));
    put32(static_cast<std::uint32_t>(metric_));

    if (std::fwrite(header, 1, sizeof(header), file) != sizeof(header)) return fail("short header write");
    for (std::size_t i = 0; i < node_count_; ++i) {
        auto rec = store_->read_node(static_cast<VectorId>(i));
        if (std::fwrite(rec.data(), 1, rec.size(), file) != rec.size()) {
            return fail("short record write at node " + std::to_string(i));
        }
    }
    // Codebook (per_dim_scale, per_dim_offset: dim floats each), then the codes blob
    // (node_count * dim bytes).
    if (std::fwrite(codes_book_.per_dim_scale.data(), sizeof(float), dim_, file) != dim_ ||
        std::fwrite(codes_book_.per_dim_offset.data(), sizeof(float), dim_, file) != dim_) {
        return fail("short codebook write");
    }
    if (std::fwrite(codes_blob_.data(), 1, codes_blob_.size(), file) != codes_blob_.size()) {
        return fail("short codes blob write");
    }

    // Durability: flush stdio buffers, fsync the data, and check fclose (a deferred write error
    // such as ENOSPC can surface only there); then atomically replace `path`.
    if (std::fflush(file) != 0 || ::fsync(::fileno(file)) != 0) return fail("flush/fsync failed");
    if (std::fclose(file) != 0) {
        std::remove(tmp_path.c_str());
        return std::unexpected(Error::make(ErrorCode::kIoError, "close failed for " + tmp_path));
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return std::unexpected(Error::make(ErrorCode::kIoError, "rename to " + path + " failed"));
    }
    std::string dir = std::filesystem::path(path).parent_path().string();
    if (dir.empty()) dir = ".";
    int dir_fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dir_fd >= 0) {
        int rc = ::fsync(dir_fd);
        ::close(dir_fd);
        if (rc != 0) return std::unexpected(Error::make(ErrorCode::kIoError, "fsync of " + dir + " failed"));
    }
    return {};
}

std::span<const float> VamanaIndex::vector_of(VectorId id) const {
    auto rec = store_->read_node(id);
    return {detail::record_vector(rec), dim_};
}

Result<VamanaIndex> VamanaIndex::open(std::unique_ptr<NodeStore> store, NodeRecordLayout layout,
                                       std::uint32_t dim, std::size_t node_count,
                                       VectorId entry_point, Metric metric,
                                       quant::Sq8CodeBook codes_book,
                                       std::vector<std::byte> codes_blob) {
    if (!store) {
        return std::unexpected(Error::make(ErrorCode::kInvalidArgument, "null NodeStore"));
    }
    if (codes_book.per_dim_scale.size() != dim || codes_book.per_dim_offset.size() != dim) {
        return std::unexpected(
            Error::make(ErrorCode::kCorruption, "codes_book dimension mismatch on open"));
    }
    if (codes_blob.size() != static_cast<std::size_t>(dim) * node_count) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "codes_blob size mismatch on open"));
    }
    if (node_count == 0 || static_cast<std::size_t>(entry_point) >= node_count || layout.dim != dim ||
        store->node_count() != node_count) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "inconsistent index header on open"));
    }
    if (metric != Metric::kInnerProduct && metric != Metric::kL2) {
        return std::unexpected(Error::make(ErrorCode::kCorruption, "unknown metric on open"));
    }
    VamanaIndex index;
    index.store_ = std::move(store);
    index.layout_ = layout;
    index.dim_ = dim;
    index.node_count_ = node_count;
    index.entry_point_ = entry_point;
    index.metric_ = metric;
    index.codes_book_ = std::move(codes_book);
    index.codes_blob_ = std::move(codes_blob);
    return index;
}

}  // namespace lattice::index_graph
