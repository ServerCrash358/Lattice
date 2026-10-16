// Filtered-search recall/latency on real SIFT1M (production disk-resident index).
//
// For each selectivity, compares:
//   baseline  : VamanaIndex::search with an admit callback (filter applied to the result
//               list only; traversal is unfiltered) -- what search() did before the planner.
//   planner   : VamanaIndex::search_filtered (scan matching vectors via RAM codes when few,
//               else beam widened by 1/selectivity).
// Ground truth is exact (brute-force L2 over the matching set). Two filter families:
//   uniform   : attribute independent of the vector (easy case)
//   clustered : attribute = nearest of 64 random centers, so matching vectors are spatially
//               localized and usually far from the query (hard case)
//
// Usage: bench_filtered_sift1m <bin_dir> <index_path> [num_queries]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "lattice/kernels/kernels.h"
#include "lattice/segment/segment.h"

using namespace lattice;
using namespace lattice::index_graph;

static std::vector<float> read_f32(const std::string& p, std::size_t n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

struct Stat {
    double recall = 0;
    std::vector<double> lat;
    void add(double r, double ms) { recall += r; lat.push_back(ms); }
    void print(const char* name, std::size_t nq) {
        std::sort(lat.begin(), lat.end());
        printf("    %-22s recall@10=%.4f  p50=%7.2f ms  p99=%7.2f ms\n", name, recall / nq, lat[lat.size() / 2],
               lat[static_cast<std::size_t>(0.99 * (lat.size() - 1))]);
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <bin_dir> <index_path> [num_queries]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    std::size_t nq = argc > 3 ? std::stoul(argv[3]) : 500;
    std::ifstream meta(dir + "/meta.txt");
    std::size_t n, dim, nt, d2, ngt, gtk;
    meta >> n >> dim >> nt >> d2 >> ngt >> gtk;
    nq = std::min(nq, nt);
    auto train = read_f32(dir + "/train.f32", n * dim);
    auto test = read_f32(dir + "/test.f32", nq * dim);
    auto disk = segment::open_disk_resident_vamana(argv[2]);
    if (!disk) { printf("open failed: %s\n", disk.error().message.c_str()); return 1; }
    const auto& kt = kernels::KernelTable::resolve();
    printf("Filtered search, real SIFT1M: n=%zu dim=%zu queries=%zu (disk-resident, SQ8 codes-gated, L2)\n", n, dim, nq);

    std::mt19937 rng(7);
    // clustered attribute: nearest of 64 random centers
    std::vector<std::size_t> centers(64);
    for (auto& c : centers) c = rng() % n;
    std::vector<std::uint8_t> cluster_of(n);
    for (std::size_t i = 0; i < n; ++i) {
        float best = 1e30f;
        for (std::size_t c = 0; c < centers.size(); ++c) {
            float d = kt.dot_l2(train.data() + i * dim, train.data() + centers[c] * dim, static_cast<std::uint32_t>(dim));
            if (d < best) { best = d; cluster_of[i] = static_cast<std::uint8_t>(c); }
        }
    }
    std::vector<std::uint32_t> uniform_key(n);
    for (auto& k : uniform_key) k = static_cast<std::uint32_t>(rng() % 1000000);

    for (const char* family : {"uniform", "clustered"}) {
        for (double sel : {0.5, 0.1, 0.01, 0.001}) {
            std::vector<std::uint8_t> mask(n, 0);
            if (std::string(family) == "uniform") {
                auto thr = static_cast<std::uint32_t>(sel * 1000000);
                for (std::size_t i = 0; i < n; ++i) mask[i] = uniform_key[i] < thr;
            } else {
                // pick whole clusters until reaching the target selectivity
                std::vector<std::size_t> cnt(64, 0);
                for (std::size_t i = 0; i < n; ++i) ++cnt[cluster_of[i]];
                std::vector<int> order(64);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                std::size_t have = 0, target = static_cast<std::size_t>(sel * n);
                std::vector<std::uint8_t> chosen(64, 0);
                for (int c : order) {
                    if (have >= target) break;
                    if (have + cnt[c] > target * 3 / 2 + 1 && have > 0) continue;
                    chosen[c] = 1;
                    have += cnt[c];
                }
                for (std::size_t i = 0; i < n; ++i) mask[i] = chosen[cluster_of[i]];
            }
            std::vector<VectorId> matching;
            for (std::size_t i = 0; i < n; ++i)
                if (mask[i]) matching.push_back(static_cast<VectorId>(i));
            if (matching.size() < 10) continue;
            printf("\n[%s] selectivity %.3f%% (%zu matching vectors)\n", family, 100.0 * matching.size() / n, matching.size());

            Stat base64, base512, base2048, plan64, plan128, scanall;
            for (std::size_t q = 0; q < nq; ++q) {
                const float* qv = test.data() + q * dim;
                // exact filtered ground truth
                std::vector<std::pair<float, VectorId>> all;
                all.reserve(matching.size());
                for (VectorId id : matching)
                    all.emplace_back(kt.dot_l2(qv, train.data() + static_cast<std::size_t>(id) * dim, static_cast<std::uint32_t>(dim)), id);
                std::partial_sort(all.begin(), all.begin() + 10, all.end());
                std::unordered_set<VectorId> gt;
                for (int i = 0; i < 10; ++i) gt.insert(all[i].second);

                auto admit = [&](VectorId v) { return mask[static_cast<std::size_t>(v)] != 0; };
                auto run = [&](Stat& st, auto&& fn) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto res = fn();
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    double hits = 0;
                    if (res) for (auto id : *res) hits += gt.count(id);
                    st.add(hits / 10.0, ms);
                };
                for (auto [ef, st] : {std::pair<std::uint32_t, Stat*>{64, &base64}, {512, &base512}, {2048, &base2048}}) {
                    SearchParams sp; sp.ef_search = ef; sp.rerank_k = std::max<std::uint32_t>(40, ef / 2);
                    run(*st, [&] { return disk->search(qv, 10, sp, admit); });
                }
                {
                    VamanaIndex::FilteredParams fp;
                    fp.base.ef_search = 64; fp.base.rerank_k = 40;
                    fp.matching_ids = matching;
                    fp.scan_max = 0xFFFFFFFFu;
                    run(scanall, [&] { return disk->search_filtered(qv, 10, fp, admit); });
                }
                for (auto [ef, st] : {std::pair<std::uint32_t, Stat*>{64, &plan64}, {128, &plan128}}) {
                    VamanaIndex::FilteredParams fp;
                    fp.base.ef_search = ef; fp.base.rerank_k = 40;
                    fp.matching_ids = matching;
                    run(*st, [&] { return disk->search_filtered(qv, 10, fp, admit); });
                }
            }
            scanall.print("scan-all (scan_max=inf)", nq);
            base64.print("baseline ef=64", nq);
            base512.print("baseline ef=512", nq);
            base2048.print("baseline ef=2048", nq);
            plan64.print("planner  ef=64", nq);
            plan128.print("planner  ef=128", nq);
        }
    }
    return 0;
}
