// Phase 3 exit gate item 2 (spec/phase-3.md): on a 10M-vector disk-resident
// collection, filtered recall@10 >= 0.95 at 1% selectivity with filtered
// p99 <= 25ms, and unfiltered p99 <= 10ms at recall@10 >= 0.95.
//
// Hardware note (disclosed): SealedSegment::build has no out-of-core build
// path -- a single 10M-row segment build needs its full float32 vectors +
// SQ8 codes + graph edges resident in RAM at once (~40GB+), which does not
// fit this machine (see the WSL box this was developed on: ~15GB
// available). This program instead builds NUM_SEGMENTS smaller sealed,
// disk-resident segments (each comfortably under the RAM ceiling to BUILD),
// keeps only their compact codes + open disk-resident graphs live at query
// time, and queries across all of them per collection query, matching the
// project's actual disk-first architecture (RAM holds compact traversal
// state, disk holds the full vectors + graph, D-ARCH-01) rather than
// simulating a single flat 10M-row segment that the current code can't
// build in one pass.
//
// Ground truth (both unfiltered and filtered) is computed exactly via
// index_flat's SIMD brute-force scan, one segment's raw vectors at a time
// (so it never needs more than one segment's worth of vectors in RAM
// either), accumulated into a running global top-10 per query across all
// segments. The actual timed benchmark queries only the disk-resident
// VamanaIndex per segment (this is what's under measurement); the final
// exact-distance cross-segment merge step re-reads just the handful of
// returned candidates (<= num_segments * k per query) from each segment's
// flat vector file, not a rescan, and is included in the reported latency
// since a real multi-segment collection query would pay it too.
//
// Status (see spec/phase-3.md exit gate item 2 for full detail): this
// harness works and has been smoke-tested. Of the three gaps originally
// found while building it, two are now resolved:
//   1. [FIXED] greedy_search's traversal loop used to read disk for every
//      neighbour it scored (not just the nodes it visits), unconditionally,
//      using exact fp32 instead of the RAM-resident SQ8 approximate
//      distance -- measured at 2,559 disk reads for a single ef_search=64
//      query. Fixed in index_graph.cpp (pulled forward from Phase 5's
//      adaptive-rerank scope): neighbours are now scored from their
//      RAM-resident SQ8 code, no disk read; a node only hits disk when
//      actually chosen for expansion. Measured impact at this harness's own
//      30,000-row/dim=768 config: p99 drops ~20-24x across the whole
//      ef_search range (e.g. ef=2048: 3,417ms -> 140.8ms) at unchanged
//      recall. NOT a full pass yet at this scale -- ef=2048 (where recall
//      clears 0.95) is still ~14x over the unfiltered p99 budget -- but the
//      mechanism now works as designed; real hardware, build-param tuning,
//      or a RaBitQ swap (smaller codes than SQ8's 768 B/vector) are the
//      remaining levers, not further traversal-architecture changes.
//   2. [RESOLVED] VamanaIndex::build's scaling was originally measured as
//      ~N^1.5 (5,000 rows ~18s, 50,000 rows 621.9s -> ~17h/1M-row estimate),
//      but a larger, cleaner data point (100,000 rows: 1,109.7s, only 1.78x
//      the 50K time for 2x the rows) shows that was dominated by fixed
//      startup overhead at the tiny 5K sample, not a real trend -- true
//      scaling is roughly linear. Corrected: ~3h for a single 1M-row
//      segment (still single-threaded). No longer the blocker.
//   3. A wide ef_search sweep (see run_mode's diagnostic loop below) showed
//      recall and latency were COUPLED by (1) before the fix -- recall
//      reaches 0.985 unfiltered / 0.865 filtered by ef_search=2048 (the
//      graph is NOT under-built -- consistent with item 3's 0.9846 recall
//      on real SIFT1M at the same R=64/build_list=100), but every extra
//      candidate visited to reach that recall cost another full disk read.
//      With (1) fixed, build-param tuning is now an independently
//      productive next step (previously it wasn't, since (1) dominated
//      every measurement regardless of params).
// Re-run this at larger scale or with real embeddings to check whether the
// remaining gap closes; the harness itself does not need further changes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "lattice/common/top_k_heap.h"
#include "lattice/index_flat/index_flat.h"
#include "lattice/index_graph/index_graph.h"
#include "lattice/quant/sq8.h"
#include "lattice/segment/segment.h"

using namespace lattice;
using namespace lattice::index_graph;

namespace {

struct Args {
    std::size_t num_segments = 10;
    std::size_t rows_per_segment = 1'000'000;
    std::uint32_t dim = 768;
    std::uint32_t r = 64;
    std::uint32_t build_list = 100;
    std::size_t num_queries = 50;
    std::size_t num_clusters = 200;
    std::uint32_t num_categories = 100;  // 1/num_categories selectivity per category
    std::uint32_t target_category = 7;
    std::string work_dir = "bench/data/disk10m_work";
    unsigned seed = 42;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto eat = [&](const char* flag) -> std::optional<std::string> {
            std::string prefix = std::string(flag) + "=";
            if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
            return std::nullopt;
        };
        if (auto v = eat("--segments")) a.num_segments = std::stoul(*v);
        else if (auto v = eat("--rows-per-segment")) a.rows_per_segment = std::stoul(*v);
        else if (auto v = eat("--dim")) a.dim = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat("--queries")) a.num_queries = std::stoul(*v);
        else if (auto v = eat("--work-dir")) a.work_dir = *v;
        else if (auto v = eat("--seed")) a.seed = static_cast<unsigned>(std::stoul(*v));
        else if (auto v = eat("--r")) a.r = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat("--build-list")) a.build_list = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat("--num-clusters")) a.num_clusters = std::stoul(*v);
    }
    return a;
}

// Unit-normalized vectors drawn from a fixed set of cluster centers plus
// gaussian noise (D-VEC-07's real embeddings cluster similarly; a uniform
// random high-dim cloud makes ANN recall behavior unrealistically easy/flat).
struct ClusterModel {
    std::vector<std::vector<float>> centers;  // num_clusters x dim, unit-normalized

    static ClusterModel make(std::mt19937& rng, std::size_t num_clusters, std::uint32_t dim) {
        ClusterModel m;
        std::normal_distribution<float> g(0.0f, 1.0f);
        m.centers.resize(num_clusters, std::vector<float>(dim));
        for (auto& c : m.centers) {
            for (auto& x : c) x = g(rng);
            normalize(c);
        }
        return m;
    }

    static void normalize(std::vector<float>& v) {
        double n2 = 0.0;
        for (float x : v) n2 += static_cast<double>(x) * x;
        float inv = static_cast<float>(1.0 / std::sqrt(std::max(n2, 1e-12)));
        for (auto& x : v) x *= inv;
    }

    void sample(std::mt19937& rng, std::vector<float>& out) const {
        std::uniform_int_distribution<std::size_t> pick(0, centers.size() - 1);
        std::normal_distribution<float> noise(0.0f, 0.2f);
        const auto& c = centers[pick(rng)];
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = c[i] + noise(rng);
        normalize(out);
    }
};

float dot(const float* a, const float* b, std::uint32_t dim) {
    float s = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

struct SegmentHandle {
    VamanaIndex index;
    std::vector<std::uint8_t> categories;  // local_vid -> category
    std::string raw_path;
};

float p99(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    std::size_t idx = static_cast<std::size_t>(0.99 * (v.size() - 1));
    return static_cast<float>(v[idx]);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::size_t total_rows = args.num_segments * args.rows_per_segment;
    printf("Phase 3 exit gate item 2: %zu segments x %zu rows/segment = %zu total, dim=%u\n",
           args.num_segments, args.rows_per_segment, total_rows, args.dim);
    printf("Filter: %u categories, target=%u (~%.2f%% selectivity)\n", args.num_categories,
           args.target_category, 100.0 / args.num_categories);

    std::filesystem::create_directories(args.work_dir);

    std::mt19937 model_rng(args.seed);
    auto clusters = ClusterModel::make(model_rng, args.num_clusters, args.dim);

    // Fixed query set, drawn once, held in RAM for the whole run (tiny: a
    // few hundred KB even at dim=768).
    std::mt19937 query_rng(args.seed + 999999u);
    std::vector<std::vector<float>> queries(args.num_queries, std::vector<float>(args.dim));
    for (auto& q : queries) clusters.sample(query_rng, q);

    constexpr std::uint32_t kK = 10;
    std::vector<TopKHeap<float, std::uint64_t>> gt_unfiltered(
        args.num_queries, TopKHeap<float, std::uint64_t>(kK));
    std::vector<TopKHeap<float, std::uint64_t>> gt_filtered(
        args.num_queries, TopKHeap<float, std::uint64_t>(kK));

    std::vector<SegmentHandle> segments;
    segments.reserve(args.num_segments);

    auto layout = NodeRecordLayout::make(args.dim, ElemType::kFloat32, args.r);
    if (!layout) {
        printf("layout rejected: %s\n", layout.error().message.c_str());
        return 1;
    }

    for (std::size_t seg = 0; seg < args.num_segments; ++seg) {
        auto t_gen0 = std::chrono::steady_clock::now();
        std::mt19937 data_rng(args.seed + static_cast<unsigned>(seg) * 7919u + 1u);
        std::uniform_int_distribution<std::uint32_t> cat_dist(0, args.num_categories - 1);

        std::vector<float> vecs(args.rows_per_segment * args.dim);
        std::vector<std::uint8_t> cats(args.rows_per_segment);
        for (std::size_t i = 0; i < args.rows_per_segment; ++i) {
            std::vector<float> tmp(args.dim);
            clusters.sample(data_rng, tmp);
            std::copy(tmp.begin(), tmp.end(), vecs.begin() + i * args.dim);
            cats[i] = static_cast<std::uint8_t>(cat_dist(data_rng));
        }
        auto t_gen1 = std::chrono::steady_clock::now();

        std::string raw_path = args.work_dir + "/seg_" + std::to_string(seg) + ".f32";
        {
            std::ofstream f(raw_path, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(vecs.data()),
                    static_cast<std::streamsize>(vecs.size() * sizeof(float)));
        }

        // --- exact ground truth contribution from this segment (while its
        // vectors are still resident) ---
        index_flat::FlatView view{vecs.data(), args.dim, args.dim, args.rows_per_segment};
        std::uint64_t global_base = static_cast<std::uint64_t>(seg) * args.rows_per_segment;
        for (std::size_t qi = 0; qi < args.num_queries; ++qi) {
            const float* q = queries[qi].data();

            auto unf = index_flat::search(view, q, kK);
            if (unf) {
                for (auto vid : *unf) {
                    std::size_t li = static_cast<std::size_t>(vid);
                    float score = dot(q, vecs.data() + li * args.dim, args.dim);
                    gt_unfiltered[qi].try_insert(score, global_base + li);
                }
            }
            index_flat::AdmitFn admit_cat = [&](VectorId vid) {
                return cats[static_cast<std::size_t>(vid)] == args.target_category;
            };
            auto flt = index_flat::search(view, q, kK, admit_cat);
            if (flt) {
                for (auto vid : *flt) {
                    std::size_t li = static_cast<std::size_t>(vid);
                    float score = dot(q, vecs.data() + li * args.dim, args.dim);
                    gt_filtered[qi].try_insert(score, global_base + li);
                }
            }
        }
        auto t_gt1 = std::chrono::steady_clock::now();

        // --- build in-RAM, serialize, reopen disk-resident, then drop the
        // big float buffer before moving to the next segment ---
        std::vector<const float*> ptrs(args.rows_per_segment);
        for (std::size_t i = 0; i < args.rows_per_segment; ++i) ptrs[i] = vecs.data() + i * args.dim;

        std::string disk_path = args.work_dir + "/seg_" + std::to_string(seg) + ".vamana";
        std::chrono::steady_clock::time_point t_build1;
        {
            // Scoped so the in-RAM build (book/codes/index, which together
            // hold a second full copy of this segment's vectors) is
            // destroyed before reopening disk-resident below -- this is
            // the point that keeps peak RAM bounded to ~one segment's
            // build footprint, not num_segments x that.
            auto book = quant::Sq8CodeBook::train(ptrs, args.dim);
            if (!book) {
                printf("seg %zu: SQ8 train failed: %s\n", seg, book.error().message.c_str());
                return 1;
            }
            quant::Sq8Codes codes(*book);

            VamanaBuildParams bp;
            bp.r = args.r;
            bp.alpha = 1.2f;
            bp.build_list_size = args.build_list;
            bp.metric = Metric::kInnerProduct;

            auto index = VamanaIndex::build(ptrs, args.dim, codes, bp, *layout);
            if (!index) {
                printf("seg %zu: build failed: %s\n", seg, index.error().message.c_str());
                return 1;
            }
            t_build1 = std::chrono::steady_clock::now();

            auto ser = index->serialize_to_disk(disk_path);
            if (!ser) {
                printf("seg %zu: serialize failed: %s\n", seg, ser.error().message.c_str());
                return 1;
            }
        }
        ptrs.clear();
        ptrs.shrink_to_fit();
        vecs.clear();
        vecs.shrink_to_fit();

        auto disk_index = segment::open_disk_resident_vamana(disk_path);
        if (!disk_index) {
            printf("seg %zu: reopen failed: %s\n", seg, disk_index.error().message.c_str());
            return 1;
        }
        auto t_all1 = std::chrono::steady_clock::now();

        printf("seg %zu/%zu: gen=%.1fs gt=%.1fs build=%.1fs total=%.1fs\n", seg + 1, args.num_segments,
               std::chrono::duration<double>(t_gen1 - t_gen0).count(),
               std::chrono::duration<double>(t_gt1 - t_gen1).count(),
               std::chrono::duration<double>(t_build1 - t_gt1).count(),
               std::chrono::duration<double>(t_all1 - t_gen0).count());

        segments.push_back(SegmentHandle{std::move(*disk_index), std::move(cats), raw_path});
    }

    // Open one persistent handle per segment's raw vector file for the
    // final cross-segment exact-distance merge (small, targeted reads).
    std::vector<std::ifstream> raw_files;
    raw_files.reserve(segments.size());
    for (auto& s : segments) raw_files.emplace_back(s.raw_path, std::ios::binary);

    auto read_vec = [&](std::size_t seg_idx, std::size_t local_vid, std::vector<float>& out) {
        raw_files[seg_idx].seekg(static_cast<std::streamoff>(local_vid) *
                                  static_cast<std::streamoff>(args.dim) * sizeof(float));
        raw_files[seg_idx].read(reinterpret_cast<char*>(out.data()),
                                 static_cast<std::streamsize>(args.dim * sizeof(float)));
    };

    auto run_mode = [&](bool filtered, std::uint32_t ef_search) {
        SearchParams sp;
        sp.ef_search = ef_search;
        sp.rerank_k = std::max<std::uint32_t>(40, ef_search / 2);

        double total_recall = 0.0;
        std::vector<double> latencies_ms;
        latencies_ms.reserve(args.num_queries);
        std::vector<float> cand_vec(args.dim);
        double search_ms_total = 0.0, merge_ms_total = 0.0;

        for (std::size_t qi = 0; qi < args.num_queries; ++qi) {
            const float* q = queries[qi].data();
            auto t0 = std::chrono::steady_clock::now();

            TopKHeap<float, std::uint64_t> merged(kK);
            for (std::size_t seg = 0; seg < segments.size(); ++seg) {
                AdmitFn admit;
                if (filtered) {
                    std::uint32_t target = args.target_category;
                    const auto* cats = &segments[seg].categories;
                    admit = [cats, target](VectorId vid) {
                        return (*cats)[static_cast<std::size_t>(vid)] == target;
                    };
                }
                auto ts0 = std::chrono::steady_clock::now();
                auto r = segments[seg].index.search(q, kK, sp, admit);
                auto ts1 = std::chrono::steady_clock::now();
                search_ms_total += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
                if (!r) continue;
                for (auto vid : *r) {
                    std::size_t li = static_cast<std::size_t>(vid);
                    read_vec(seg, li, cand_vec);
                    float score = dot(q, cand_vec.data(), args.dim);
                    merged.try_insert(score, static_cast<std::uint64_t>(seg) * args.rows_per_segment + li);
                }
                auto ts2 = std::chrono::steady_clock::now();
                merge_ms_total += std::chrono::duration<double, std::milli>(ts2 - ts1).count();
            }
            auto t1 = std::chrono::steady_clock::now();
            latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

            auto got = merged.extract_sorted();
            auto& gt = filtered ? gt_filtered[qi] : gt_unfiltered[qi];
            auto gt_sorted = gt.extract_sorted();  // NOTE: consumes; fine, one pass per mode
            std::vector<std::uint64_t> gt_ids;
            for (auto& [s, id] : gt_sorted) { (void)s; gt_ids.push_back(id); }
            std::size_t hits = 0;
            for (auto& [s, id] : got) {
                (void)s;
                if (std::find(gt_ids.begin(), gt_ids.end(), id) != gt_ids.end()) ++hits;
            }
            total_recall += static_cast<double>(hits) / static_cast<double>(std::max<std::size_t>(1, gt_ids.size()));
            // Put ground truth back for the next ef_search sweep value.
            for (auto& id : gt_ids) { /* re-seed below */ (void)id; }
            for (auto& [s, id] : gt_sorted) gt.try_insert(s, id);
        }

        double mean_recall = total_recall / args.num_queries;
        printf("  ef_search=%4u  recall@10=%.4f  p99=%.2fms  (search=%.1fms merge=%.1fms total across %zu queries x %zu segs)\n",
               ef_search, mean_recall, p99(latencies_ms), search_ms_total, merge_ms_total,
               args.num_queries, segments.size());
        return std::pair<double, double>{mean_recall, p99(latencies_ms)};
    };

    printf("\n--- Unfiltered ---\n");
    // Wide ef_search sweep (diagnostic, not just the exit-gate pair): tells
    // us whether low recall at ef=64/128 is a BUILD-quality ceiling (recall
    // stays low even at huge ef -- more thorough search can't find edges
    // that were never built) or a SEARCH-time parameter problem (recall
    // converges toward the target once ef_search is large enough).
    for (std::uint32_t ef : {64u, 128u, 256u, 512u, 1024u, 2048u}) run_mode(false, ef);

    printf("\n--- Filtered (category=%u, ~%.2f%% selectivity) ---\n", args.target_category,
           100.0 / args.num_categories);
    for (std::uint32_t ef : {64u, 128u, 256u, 512u, 1024u, 2048u}) run_mode(true, ef);

    return 0;
}
