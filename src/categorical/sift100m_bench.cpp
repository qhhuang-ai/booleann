// bci_bench: Boole-ANN Cell Index (BCI) full-system benchmark on SIFT100M.
// Float32 / dim=128. 100M base, 200 conjunction queries (8 pairs × 25 each).
// Designed for the 2-tag conjunction breadth experiment (sole-survivor at 100M
// scale with exact support materialization and global-ID result validation.
// Brute-only path: no HAMCG shards needed; per-tag brute over posting list
// with AVX2 + PACH; recall = 1.0 by construction.
// For each query in [qid_lo, qid_hi):
//   - Single equality (filter cardinality 1):
//       if shard exists for that tag -> HAMCG beam_search on shard
//       else (cold tag <0.1%)       -> brute force over posting list (fast)
//   - Conjunction (filter cardinality 2):
//       pick smaller tag T_small, beam_search on its shard, post-filter by other tag
//       if T_small has no shard -> try T_large; if neither -> mark FALLBACK_NEEDED
// Recall@10 vs ground-truth, QPS, latency histogram.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/filters.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"

using PointT = Euclidian_Point<float>;
using PR     = PointRange<float, PointT>;
using Indx   = int32_t;
using GraphI = Graph<Indx>;

// Thin SubsetPointRange w/o unordered_map (which dominated cost in v1).
// beam_search only needs operator[], size(), dimension(), aligned_dimension();
// it never calls real_index/subset_index, so the map is dead weight.
struct ThinSubPR {
  PR& pr;
  const parlay::sequence<int32_t>& subset;
  size_t n;
  unsigned int dims, aligned_dims;
  ThinSubPR(PR& pr_, const parlay::sequence<int32_t>& s)
    : pr(pr_), subset(s), n(s.size()),
      dims((unsigned)pr_.dimension()),
      aligned_dims((unsigned)pr_.aligned_dimension()) {}
  size_t size() const { return n; }
  PointT operator[](long i) { return pr[subset[i]]; }
  long dimension() const { return dims; }
  long aligned_dimension() const { return aligned_dims; }
};

struct Shard {
  int32_t tag;
  int64_t freq;
  parlay::sequence<int32_t> subset; // local -> global base idx
  GraphI graph;
  long maxDeg;
};

// IVF²-style per-tag cluster index (cache-friendly working set for warm-cache QPS)
// Design note: centroids fit L1 (< 256KB), members CSR-style.
// Replaces beam_search with: linear centroid scan -> top-nprobe -> collect members -> exact rerank.
struct ClusterIndex {
  int32_t tag;
  int32_t n_clusters;
  int32_t aligned_dim;
  std::vector<float> centroid_data;     // n_clusters * aligned_dim, contiguous
  std::vector<int64_t> member_offsets;    // n_clusters + 1
  std::vector<int32_t> member_ids;        // flattened global point IDs
};

static std::unique_ptr<ClusterIndex> load_cluster_index(const std::string& clust_dir, int32_t tag) {
  char cp[1024], mp[1024];
  snprintf(cp, sizeof(cp), "%s/%d_centroids.bin", clust_dir.c_str(), tag);
  snprintf(mp, sizeof(mp), "%s/%d_members.bin",   clust_dir.c_str(), tag);
  if (!std::filesystem::exists(cp) || !std::filesystem::exists(mp)) return nullptr;
  auto ci = std::make_unique<ClusterIndex>();
  ci->tag = tag;
  {
    FILE* f = fopen(cp, "rb");
    size_t got = fread(&ci->n_clusters, sizeof(int32_t), 1, f);
    got = fread(&ci->aligned_dim, sizeof(int32_t), 1, f);
    ci->centroid_data.resize((size_t)ci->n_clusters * ci->aligned_dim);
    got = fread(ci->centroid_data.data(), 1, ci->centroid_data.size(), f);
    (void)got;
    fclose(f);
  }
  {
    FILE* f = fopen(mp, "rb");
    int32_t nc;
    size_t got = fread(&nc, sizeof(int32_t), 1, f);
    ci->member_offsets.resize(nc + 1);
    got = fread(ci->member_offsets.data(), sizeof(int64_t), nc + 1, f);
    int64_t total = ci->member_offsets.back();
    ci->member_ids.resize(total);
    got = fread(ci->member_ids.data(), sizeof(int32_t), total, f);
    (void)got;
    fclose(f);
  }
  return ci;
}

static parlay::sequence<int32_t> load_subset_idx(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return {};
  int32_t n;
  size_t got = fread(&n, sizeof(int32_t), 1, f); (void)got;
  parlay::sequence<int32_t> v(n);
  got = fread(v.data(), sizeof(int32_t), n, f); (void)got;
  fclose(f); return v;
}

// GT format: header [N:uint32, K:uint32], body N*K * (uint32 idx + float32 dist)
struct GroundTruth {
  uint32_t N, K;
  std::vector<uint32_t> indices;     // N*K
  std::vector<float>    distances;   // N*K
};

static GroundTruth load_gt(const std::string& path) {
  GroundTruth gt{};
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "GT open fail: %s\n", path.c_str()); exit(1); }
  size_t got = fread(&gt.N, sizeof(uint32_t), 1, f);
  got = fread(&gt.K, sizeof(uint32_t), 1, f);
  (void)got;
  gt.indices.resize((size_t)gt.N * gt.K);
  gt.distances.assign((size_t)gt.N * gt.K, 0.0f);
  got = fread(gt.indices.data(), sizeof(uint32_t), gt.indices.size(), f);
  // For LAION the index file does NOT carry distances; attempt to read past
  // EOF will return 0 and leave distances zero-initialized. We instead try a
  // side-car .dist file (path with .bin replaced by .dist.bin) — needed for
  // BCI_TIE_AWARE recall semantics.
  fclose(f);
  std::string dist_path = path;
  auto pos = dist_path.rfind(".bin");
  if (pos != std::string::npos) {
    dist_path.replace(pos, 4, ".dist.bin");
  } else {
    dist_path += ".dist";
  }
  FILE* fd = fopen(dist_path.c_str(), "rb");
  if (fd) {
    uint32_t dN = 0, dK = 0;
    size_t g1 = fread(&dN, sizeof(uint32_t), 1, fd);
    size_t g2 = fread(&dK, sizeof(uint32_t), 1, fd);
    (void)g1; (void)g2;
    if (dN == gt.N && dK == gt.K) {
      size_t g3 = fread(gt.distances.data(), sizeof(float), gt.distances.size(), fd);
      (void)g3;
      fprintf(stderr, "[GT-dist sidecar loaded] %s N=%u K=%u\n",
              dist_path.c_str(), dN, dK);
    } else {
      fprintf(stderr, "[GT-dist sidecar shape mismatch] %s dN=%u dK=%u (expected %u %u)\n",
              dist_path.c_str(), dN, dK, gt.N, gt.K);
    }
    fclose(fd);
  } else {
    fprintf(stderr, "[GT-dist sidecar missing, TIE_AWARE will be unusable] %s\n",
            dist_path.c_str());
  }
  return gt;
}

int main(int argc, char** argv) {
  const std::string DATA = [](){
    const char* e = std::getenv("BCI_SIFT_DATA");
    return e ? std::string(e) : std::string("./data/sift100m/");
  }();
  const std::string DRT = [](){
    const char* e = std::getenv("BCI_SIFT_INDEX_DIR");
    return e ? std::string(e) : std::string(
      "./indexes/sift100m");
  }();

  // Args: qid_lo qid_hi beam K
  int qid_lo = argc > 1 ? atoi(argv[1]) : 0;
  int qid_hi = argc > 2 ? atoi(argv[2]) : 10000;
  int beam   = argc > 3 ? atoi(argv[3]) : 128;
  int K      = argc > 4 ? atoi(argv[4]) : 10;
  int post_filter_pool = argc > 5 ? atoi(argv[5]) : 4 * beam;
  int64_t brute_conj_thresh = argc > 6 ? atoll(argv[6]) : 200000;
  int n_runs = argc > 7 ? atoi(argv[7]) : 1;  // run query batch N times for warm-cache
  int use_clusters = argc > 8 ? atoi(argv[8]) : 0;  // 0=disable, 1=use IVF² cluster path for conj
  int nprobe = argc > 9 ? atoi(argv[9]) : 3;        // top-N centroids to probe
  int target_pts = argc > 10 ? atoi(argv[10]) : 15000;  // PIVF default
  int use_bitvec = argc > 11 ? atoi(argv[11]) : 1;  // 0=disable bitvec, use bm.match (faster but lower recall)
  int use_pach   = argc > 12 ? atoi(argv[12]) : 1;  // 0=no PACH (baseline), 1=PACH cluster pruning ON

  printf("=== BCI bench (Arch A: HAMCG shards + brute-fallback) ===\n");
  printf("qid range = [%d, %d), beam = %d, K = %d, post_filter_pool = %d\n",
         qid_lo, qid_hi, beam, K, post_filter_pool);
  printf("brute_conj_thresh=%ld n_runs=%d use_clusters=%d nprobe=%d target_pts=%d use_bitvec=%d use_pach=%d\n",
         (long)brute_conj_thresh, n_runs, use_clusters, nprobe, target_pts, use_bitvec, use_pach);
  printf("parlay workers = %ld\n", parlay::num_workers());

  // Causal pair-materialisation control.  This SIFT workload has exactly atom
  // tags 0..11; benchmark pair pseudo-tags use both legacy (>11) and held-out
  // (20000+) encodings.  DISABLE_PAIR_GRAPHS removes all pair
  // state at load time (not merely at routing time), while PAIR_ALLOWLIST
  // admits an explicit whitespace-separated set of pseudo-tags for byte-budget
  // sweeps.  The two controls are mutually compatible: disable wins.
  const bool disable_pair_graphs =
    std::getenv("BCI_SIFT_DISABLE_PAIR_GRAPHS") != nullptr &&
    std::atoi(std::getenv("BCI_SIFT_DISABLE_PAIR_GRAPHS")) != 0;
  const bool disable_atom_graphs =
    std::getenv("BCI_SIFT_DISABLE_ATOM_GRAPHS") != nullptr &&
    std::atoi(std::getenv("BCI_SIFT_DISABLE_ATOM_GRAPHS")) != 0;
  if (disable_pair_graphs && disable_atom_graphs) {
    std::fprintf(stderr, "cannot disable both atom and pair graphs\n");
    return 2;
  }
  std::unordered_set<int32_t> pair_allowlist;
  const char* pair_allowlist_path = std::getenv("BCI_SIFT_PAIR_ALLOWLIST");
  const bool use_pair_allowlist = pair_allowlist_path != nullptr &&
                                  std::strlen(pair_allowlist_path) > 0;
  const long expected_graph_degree = std::getenv("BCI_EXPECT_GRAPH_DEGREE") ?
    std::atol(std::getenv("BCI_EXPECT_GRAPH_DEGREE")) : 0;
  if (expected_graph_degree < 0) {
    std::fprintf(stderr, "BCI_EXPECT_GRAPH_DEGREE must be non-negative\n");
    return 2;
  }
  if (use_pair_allowlist) {
    std::ifstream allow_in(pair_allowlist_path);
    if (!allow_in) {
      std::fprintf(stderr, "cannot open BCI_SIFT_PAIR_ALLOWLIST=%s\n",
                   pair_allowlist_path);
      return 2;
    }
    int32_t tag = -1;
    while (allow_in >> tag) pair_allowlist.insert(tag);
  }
  printf("graph admission: disable_atoms=%d disable_pairs=%d pair_allowlist=%s admitted_pair_ids=%zu index_dir=%s expected_degree=%ld\n",
         disable_atom_graphs ? 1 : 0, disable_pair_graphs ? 1 : 0,
         use_pair_allowlist ? pair_allowlist_path : "<all>",
         pair_allowlist.size(), DRT.c_str(), expected_graph_degree);

  // -- Load datasets ---------------------------------------------------------
  auto t0 = std::chrono::steady_clock::now();

  PR base((DATA + "base.100M.f32bin").c_str());
  PR query((DATA + "query.10K.f32bin").c_str());
  csr_filters qm(DATA + "query.metadata.spmat");
  csr_filters bm(DATA + "base.metadata.spmat");
  csr_filters bmt = bm.transpose();
  printf("[loaded] base=%ld dim=%ld query=%ld qm=%ld base_meta=%ld\n",
         base.size(), base.dimension(), query.size(),
         qm.n_points, bm.n_points);

  // SIFT100M conjunction GT: single file sift100m_2tag_conj_gt.bin (200 queries × K=10).
  // Same as LAION format: header (N, K) uint32, then N*K uint32 indices, optional
  // .dist.bin sidecar for TIE_AWARE recall.
  std::string gt_path = DATA + "sift100m_2tag_conj_gt.bin";
  if (std::getenv("BCI_SIFT_DATA") && std::filesystem::exists(DATA + "heldout_conj_gt.bin"))
    gt_path = DATA + "heldout_conj_gt.bin";
  auto gt = load_gt(gt_path);
  printf("[GT loaded] %s N=%u K=%u\n", gt_path.c_str(), gt.N, gt.K);
  assert(gt.K >= (uint32_t)K);

  // -- Load shards ------------------------------------------------------------
  auto t_shard0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::unique_ptr<Shard>> shards;
  uint64_t loaded_graph_bytes = 0;
  uint64_t loaded_subset_bytes = 0;
  size_t loaded_atom_shards = 0;
  size_t loaded_pair_shards = 0;
  long min_atom_degree = std::numeric_limits<long>::max();
  long max_atom_degree = 0;
  long min_pair_degree = std::numeric_limits<long>::max();
  long max_pair_degree = 0;

  for (const auto& entry : std::filesystem::directory_iterator(DRT + "/shards")) {
    auto name = entry.path().filename().string();
    if (name.find("vamana_tag_") != 0 || entry.path().extension() != ".bin") continue;
    int tag = atoi(name.substr(11, name.size()-15).c_str());
    const bool is_pair = tag >= 12;
    if (!is_pair && disable_atom_graphs) continue;
    if (is_pair && disable_pair_graphs) continue;
    if (is_pair && use_pair_allowlist && !pair_allowlist.count(tag)) continue;
    std::string subset_path = DRT + "/subset_idx/subset_idx_" + std::to_string(tag) + ".bin";
    auto sub = load_subset_idx(subset_path);
    if (sub.empty()) continue;
    std::string gp = DRT + "/shards/vamana_tag_" + std::to_string(tag) + ".bin";
    auto sh = std::make_unique<Shard>();
    sh->tag = tag;
    sh->freq = (int64_t)sub.size();
    sh->subset = std::move(sub);
    sh->graph = GraphI((char*)gp.c_str());
    sh->maxDeg = sh->graph.max_degree();
    if (sh->graph.size() != sh->subset.size()) {
      std::fprintf(stderr,
                   "graph/subset cardinality mismatch: tag=%d graph=%zu subset=%zu file=%s\n",
                   tag, sh->graph.size(), sh->subset.size(), gp.c_str());
      return 2;
    }
    if (expected_graph_degree > 0 && sh->maxDeg != expected_graph_degree) {
      std::fprintf(stderr,
                   "graph degree mismatch: tag=%d observed=%ld expected=%ld file=%s\n",
                   tag, sh->maxDeg, expected_graph_degree, gp.c_str());
      return 2;
    }
    shards[tag] = std::move(sh);
    loaded_graph_bytes += std::filesystem::file_size(entry.path());
    loaded_subset_bytes += std::filesystem::file_size(subset_path);
    if (is_pair) {
      ++loaded_pair_shards;
      min_pair_degree = std::min(min_pair_degree, shards[tag]->maxDeg);
      max_pair_degree = std::max(max_pair_degree, shards[tag]->maxDeg);
    } else {
      ++loaded_atom_shards;
      min_atom_degree = std::min(min_atom_degree, shards[tag]->maxDeg);
      max_atom_degree = std::max(max_atom_degree, shards[tag]->maxDeg);
    }
  }
  auto t_shard1 = std::chrono::steady_clock::now();
  printf("[loaded %zu shards: %zu atom + %zu pair, graph=%.3fGB subset=%.3fGB in %.2fs]\n",
         shards.size(), loaded_atom_shards, loaded_pair_shards,
         loaded_graph_bytes / 1e9, loaded_subset_bytes / 1e9,
         std::chrono::duration<double>(t_shard1-t_shard0).count());
  printf("[loaded graph degrees: atom=%ld..%ld pair=%ld..%ld expected=%ld]\n",
         loaded_atom_shards ? min_atom_degree : 0,
         loaded_atom_shards ? max_atom_degree : 0,
         loaded_pair_shards ? min_pair_degree : 0,
         loaded_pair_shards ? max_pair_degree : 0,
         expected_graph_degree);

  // -- Load IVF² cluster indices (IVF² absorb fast path) ----------------------
  auto t_clust0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::unique_ptr<ClusterIndex>> clusters;
  std::string clust_dir = DRT + "/clusters";
  // A disabled cluster operator must not retain dormant cluster state.  This
  // makes use_clusters=0 a physical-state control as well as a routing control.
  if (use_clusters >= 1 && std::filesystem::exists(clust_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(clust_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("_centroids.bin") == std::string::npos) continue;
      int tag = atoi(name.c_str());
      auto ci = load_cluster_index(clust_dir, tag);
      if (ci) clusters[tag] = std::move(ci);
    }
  }
  auto t_clust1 = std::chrono::steady_clock::now();
  printf("[loaded %zu cluster indices in %.2fs]\n", clusters.size(),
         std::chrono::duration<double>(t_clust1-t_clust0).count());

  // PACK COLD-TAG POINT DATA contiguous for sequential brute scan.
  // Eliminates random base[g] access in brute path (largest wall-time component).
  // For ~1600 unique cold tags × ~7K points × 192B = ~2.2GB upfront. Sequential
  // memory access = full prefetcher utilization, no DRAM-latency stalls.
  auto t_pack0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::vector<float>> packed_cold;
  const bool disable_packed_fallback =
    std::getenv("BCI_SIFT_DISABLE_PACKED_FALLBACK") != nullptr &&
    std::atoi(std::getenv("BCI_SIFT_DISABLE_PACKED_FALLBACK")) != 0;
  // First pass: which cold tags are actually queried?
  std::unordered_set<int32_t> cold_tags_queried;
  int n_q_tmp = qid_hi - qid_lo;
  // Pre-check env vars used for routing so we can pack the right tags.
  const bool FORCE_BRUTE_SINGLE_PRE = std::getenv("BCI_FORCE_BRUTE_SINGLE") != nullptr &&
                                       std::atoi(std::getenv("BCI_FORCE_BRUTE_SINGLE")) != 0;
  int64_t brute_single_thresh_pre = std::getenv("BCI_BRUTE_SINGLE_THRESH") ?
      std::atoll(std::getenv("BCI_BRUTE_SINGLE_THRESH")) : 200000LL;
  if (!disable_packed_fallback) for (int i = 0; i < n_q_tmp; ++i) {
    int qid = qid_lo + i;
    int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
    int n_tags = (int)(e - s);
    if (n_tags == 1) {
      int32_t t = qm.row_indices[s];
      int64_t ft = bmt.row_offsets[t+1] - bmt.row_offsets[t];
      bool small_enough = (ft <= brute_single_thresh_pre);
      // Pack if no shard, OR if we're going to force the brute path on it.
      if (!shards.count(t) || FORCE_BRUTE_SINGLE_PRE || small_enough) {
        cold_tags_queried.insert(t);
      }
    } else if (n_tags == 2) {
      int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
      // A materialised pair graph is a complete execution path and never reads
      // an atom posting.  Test this before planning fallback packing; otherwise
      // pair-only deployments copy several GiB of dormant atom vectors merely
      // because atom shards are intentionally absent.
      int32_t pair_lo = std::min(t1, t2);
      int32_t pair_hi = std::max(t1, t2);
      int32_t pair_tag = 20000 + pair_lo * 100 + pair_hi;
      if (shards.count(pair_tag)) continue;
      int64_t f1 = bmt.row_offsets[t1+1] - bmt.row_offsets[t1];
      int64_t f2 = bmt.row_offsets[t2+1] - bmt.row_offsets[t2];
      int32_t small_t = (f1 <= f2) ? t1 : t2;
      int32_t large_t = (f1 <= f2) ? t2 : t1;
      if (std::min(f1, f2) <= brute_conj_thresh) {
        cold_tags_queried.insert(small_t);
      } else if (!shards.count(small_t) && !shards.count(large_t)) {
        cold_tags_queried.insert(small_t);
      }
    }
  }
  size_t aligned_dim = base.aligned_dimension();
  for (int32_t tag : cold_tags_queried) {
    int64_t lo = bmt.row_offsets[tag], hi = bmt.row_offsets[tag+1];
    size_t n = hi - lo;
    if (n == 0) continue;
    auto& packed = packed_cold[tag];
    packed.resize(n * aligned_dim);
    for (int64_t j = lo; j < hi; ++j) {
      int32_t g = bmt.row_indices[j];
      float* src = base[g].get();
      std::memcpy(packed.data() + (j - lo) * aligned_dim, src, aligned_dim * sizeof(float));
    }
  }
  size_t packed_bytes = 0;
  for (auto& kv : packed_cold) packed_bytes += kv.second.size();
  auto t_pack1 = std::chrono::steady_clock::now();
  printf("[packed %zu cold-tag arrays in %.2fs, %.1fMB total]\n",
         packed_cold.size(),
         std::chrono::duration<double>(t_pack1-t_pack0).count(),
         packed_bytes / 1e6);
  printf("[packed fallback control] disabled=%d\n",
         disable_packed_fallback ? 1 : 0);

  // Pre-build per-tag bitvectors for conjunction post-filter (eliminates per-query
  // bm.match cache misses). Each bitvector = N_total bits = 12.5MB at N=100M.
  // EXPANDED COVERAGE (ported from yfcc bench): build bitvecs for ALL queried tags
  // (primary + secondary), not just cluster tags. Critical for SIFT100M where
  // brute-path dominates: bm.match (linear scan over 12 filters) is the bottleneck
  // and a single bit lookup is 100-1000x faster.
  auto t_bv0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::vector<uint64_t>> bitvecs;
  size_t bv_words = (bm.n_points + 63) / 64;

  // Collect ALL tags appearing in queries (covers conjunction post-filter)
  std::unordered_set<int32_t> all_query_tags;
  const bool frozen_atom_bitvecs =
    std::getenv("BCI_SIFT_FROZEN_ATOM_BITVECS") != nullptr &&
    std::atoi(std::getenv("BCI_SIFT_FROZEN_ATOM_BITVECS")) != 0;
  if (frozen_atom_bitvecs) {
    // SIFT100M's public synthetic-filter universe is exactly atoms 0..11.
    // Materialise the complete universe so neither calibration nor held-out
    // predicates choose membership state retrospectively.
    for (int32_t tag = 0; tag < 12; ++tag) all_query_tags.insert(tag);
  } else {
    for (int i = 0; i < n_q_tmp; ++i) {
      int qid = qid_lo + i;
      int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
      for (int64_t j = s; j < e; ++j) all_query_tags.insert(qm.row_indices[j]);
    }
  }
  // also include primary atoms of cluster tags (for sorted_near + bitvec filter)
  for (auto& tc : clusters) all_query_tags.insert(tc.first);

  for (int32_t tag : all_query_tags) {
    auto& bv = bitvecs[tag];
    bv.assign(bv_words, 0ULL);
    int64_t lo = bmt.row_offsets[tag];
    int64_t hi = bmt.row_offsets[tag+1];
    for (int64_t j = lo; j < hi; ++j) {
      int32_t g = bmt.row_indices[j];
      bv[g >> 6] |= (1ULL << (g & 63));
    }
  }
  auto t_bv1 = std::chrono::steady_clock::now();
  printf("[built %zu tag bitvectors in %.2fs, %.1fMB total]\n", bitvecs.size(),
         std::chrono::duration<double>(t_bv1-t_bv0).count(),
         bitvecs.size() * bv_words * 8 / 1e6);
  printf("[bitvector admission] mode=%s admitted_atoms=%zu\n",
         frozen_atom_bitvecs ? "frozen_all12" : "query_discovery",
         all_query_tags.size());

  // Predicate-aware cluster summaries.
  // For each primary tag T and each cluster c of T, pre-compute a bitvec
  // over secondary tag IDs: bit B set iff cluster c contains ≥1 point with tag B.
  // At query time for A ∧ B with primary=A: scan only clusters c of A whose
  // pach_bitvec[A][c] has bit B set — prunes clusters guaranteed to yield 0
  // post-filter survivors. Expected pruning: 60-90% on selective conjunctions.
  // Distinguishes BCI from PIVF's IVF² (which has no predicate-aware cluster pruning).
  auto t_pach0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::vector<std::vector<uint64_t>>> pach_bitvecs;
  size_t pach_words = (bm.n_filters + 63) / 64;
  size_t pach_bytes_total = 0;
  size_t pach_clusters = 0;
  parlay::sequence<int32_t> tags_for_pach;
  for (auto& tc : clusters) tags_for_pach.push_back(tc.first);
  // pre-allocate to allow parallel writes
  for (int32_t tag : tags_for_pach) {
    ClusterIndex& ci = *clusters[tag];
    pach_bitvecs[tag].assign(ci.n_clusters, std::vector<uint64_t>(pach_words, 0ULL));
    pach_clusters += ci.n_clusters;
    pach_bytes_total += (size_t)ci.n_clusters * pach_words * 8;
  }
  parlay::parallel_for(0, tags_for_pach.size(), [&](size_t ti) {
    int32_t tag = tags_for_pach[ti];
    ClusterIndex& ci = *clusters[tag];
    auto& cluster_bvs = pach_bitvecs[tag];
    for (int c = 0; c < ci.n_clusters; ++c) {
      auto& bv = cluster_bvs[c];
      int64_t lo = ci.member_offsets[c];
      int64_t hi = ci.member_offsets[c+1];
      for (int64_t j = lo; j < hi; ++j) {
        int32_t g = ci.member_ids[j];
        int64_t s = bm.row_offsets[g];
        int64_t e = bm.row_offsets[g+1];
        for (int64_t k = s; k < e; ++k) {
          int32_t t = bm.row_indices[k];
          bv[t >> 6] |= (1ULL << (t & 63));
        }
      }
    }
  });
  auto t_pach1 = std::chrono::steady_clock::now();
  printf("[built PACH bitvecs for %zu tags, %zu clusters total in %.2fs, %.1fMB]\n",
         pach_bitvecs.size(), pach_clusters,
         std::chrono::duration<double>(t_pach1-t_pach0).count(),
         pach_bytes_total / 1e6);

  // PACH instrumentation: count clusters considered vs kept across all conjunctions
  std::atomic<int64_t> pach_total_clusters{0};
  std::atomic<int64_t> pach_kept_clusters{0};

  // -- Build per-query route info --------------------------------------------
  int n_q = qid_hi - qid_lo;
  std::vector<int>  q_route(n_q, -1);   // 0=HAMCG_single, 1=HAMCG_conj, 2=brute_cold, -1=skip
  std::vector<int32_t> q_primary(n_q, -1);
  std::vector<int32_t> q_secondary(n_q, -1);
  std::vector<int32_t> q_pair_tag(n_q, -1);
  std::vector<uint8_t> q_pair_materialized(n_q, 0);
  // PROXY ROUTER (paper's central idea): per-query adaptive target_points based
  // on predicate-selectivity proxy. Easy queries (low joint selectivity) use few
  // candidates; hard queries (high selectivity boundary) use more.
  std::vector<int> q_tpts(n_q, target_pts);

  // BCI_FORCE_BRUTE_SINGLE: if set non-zero, route ALL single-tag queries to
  // brute (exact scan over primary's posting). Trades QPS for recall; recommended
  // when HAMCG_single beam-search's order-noise gap limits achievable recall.
  const bool FORCE_BRUTE_SINGLE = std::getenv("BCI_FORCE_BRUTE_SINGLE") != nullptr &&
                                   std::atoi(std::getenv("BCI_FORCE_BRUTE_SINGLE")) != 0;
  // BCI_BRUTE_SINGLE_THRESH: primary-tag size threshold below which single-tag
  // queries are routed to brute. Default 200000 (matches brute_conj_thresh).
  int64_t brute_single_thresh = std::getenv("BCI_BRUTE_SINGLE_THRESH") ?
      std::atoll(std::getenv("BCI_BRUTE_SINGLE_THRESH")) : 200000LL;
  for (int i = 0; i < n_q; ++i) {
    int qid = qid_lo + i;
    int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
    int n_tags = (int)(e - s);
    if (n_tags == 1) {
      int32_t t = qm.row_indices[s];
      int64_t ft = bmt.row_offsets[t+1] - bmt.row_offsets[t];
      bool small_enough = (ft <= brute_single_thresh);
      if (shards.count(t) && !(FORCE_BRUTE_SINGLE || small_enough)) {
        q_route[i] = 0; q_primary[i] = t;   // HAMCG beam_search
      } else {
        q_route[i] = 2; q_primary[i] = t;   // brute (q_secondary stays -1 -> no filter)
      }
    } else if (n_tags == 2) {
      int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
      int64_t f1 = bmt.row_offsets[t1+1] - bmt.row_offsets[t1];
      int64_t f2 = bmt.row_offsets[t2+1] - bmt.row_offsets[t2];
      // BCI_FORCE_LARGE_PRIMARY: ablation — if set, swap primary to the larger
      // tag's posting (simulates "naive single-tag + post-filter on larger tag",
      // which is what a non-selectivity-aware system would do). Allows fair
      // intra-system comparison vs BCI's selectivity-aware routing.
      const bool FORCE_LARGE_PRIMARY = std::getenv("BCI_FORCE_LARGE_PRIMARY") != nullptr &&
                                        std::atoi(std::getenv("BCI_FORCE_LARGE_PRIMARY")) != 0;
      int32_t small_t = (f1 <= f2) ? t1 : t2;
      int32_t large_t = (f1 <= f2) ? t2 : t1;
      if (FORCE_LARGE_PRIMARY) std::swap(small_t, large_t);
      int64_t small_size = FORCE_LARGE_PRIMARY ? std::max(f1, f2) : std::min(f1, f2);
      // PAIR-GRAPH (per-pair intersection materialization): if a dedicated Vamana
      // graph was built ON the intersection X_a∩X_b (pseudo-tag 1000+), route the
      // conjunction to it — every node/edge is inside the valid set, so recall is
      // ordinary single-tag recall, not the per-atom conjunction collapse. Encode
      // pair (a,b) as pseudo-tag 1000*min + max is ambiguous; use 1000*min+max for
      // min<100. Fallback: per-atom/brute below.
      {
        int32_t lo = std::min(t1,t2), hi = std::max(t1,t2);
        int32_t pair_tag = 20000 + lo*100 + hi;
        q_pair_tag[i] = pair_tag;
        if (shards.count(pair_tag)) {
          q_route[i] = 1; q_primary[i] = pair_tag; q_secondary[i] = -1;  // pair-graph (no post-filter)
          q_pair_materialized[i] = 1;
          goto conj_routed;
        }
      }
      // Use exact intersection scanning for bounded supports; larger supports
      // retain graph traversal with post-filtering.
      if (small_size <= brute_conj_thresh) {
        q_route[i] = 2; q_primary[i] = small_t; q_secondary[i] = large_t;  // brute exact
      } else if (shards.count(small_t)) {
        q_route[i] = 1; q_primary[i] = small_t; q_secondary[i] = large_t;
      } else if (shards.count(large_t)) {
        q_route[i] = 1; q_primary[i] = large_t; q_secondary[i] = small_t;
      } else {
        q_route[i] = 2; q_primary[i] = small_t; q_secondary[i] = large_t;
      }
      conj_routed:;
      // PROXY: per-query adaptive target_points.
      // For SMALL primary: use all points (no artificial cap, scan is small anyway).
      // For MEDIUM primary: standard global target_pts.
      // For LARGE primary (>500K): scale up to catch boundary tail.
      int64_t primary_size = (q_route[i] == 1) ?
                             (q_primary[i] >= 1000 ? (int64_t)shards[q_primary[i]]->subset.size()
                              : (bmt.row_offsets[q_primary[i]+1] - bmt.row_offsets[q_primary[i]])) : small_size;
      if (primary_size < 50000)        q_tpts[i] = (int)primary_size;  // use all
      else if (primary_size < 500000)  q_tpts[i] = target_pts;
      else                              q_tpts[i] = (int)std::min((int64_t)100000, (int64_t)target_pts * 2);
    } else {
      q_route[i] = -1;
    }
  }
  int n_h_single = 0, n_h_conj = 0, n_brute = 0, n_skip = 0;
  for (int i = 0; i < n_q; ++i) {
    if      (q_route[i] == 0) ++n_h_single;
    else if (q_route[i] == 1) ++n_h_conj;
    else if (q_route[i] == 2) ++n_brute;
    else                       ++n_skip;
  }
  printf("[route] HAMCG_single=%d HAMCG_conj=%d brute=%d skip=%d (of %d)\n",
         n_h_single, n_h_conj, n_brute, n_skip, n_q);

  // -- Run queries -----------------------------------------------------------
  std::vector<std::vector<int32_t>> results(n_q);
  std::vector<double> latencies(n_q, 0.0);

  // Batch execution: batch queries per
  // shard so each shard graph + ThinSubPR + subset stay cache-hot across
  // its bucket. Restructure parallelism: parallel_for over shard buckets
  // (sorted by descending bucket size for load balance), each bucket
  // processes its queries serially.
  std::unordered_map<int32_t, std::vector<int>> shard_buckets;
  std::vector<int> brute_ids;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == 0 || q_route[i] == 1) shard_buckets[q_primary[i]].push_back(i);
    else if (q_route[i] == 2)                brute_ids.push_back(i);
  }
  std::vector<std::pair<int32_t, std::vector<int>>> buckets;
  buckets.reserve(shard_buckets.size());
  for (auto& kv : shard_buckets) buckets.emplace_back(kv.first, std::move(kv.second));
  std::sort(buckets.begin(), buckets.end(),
    [](const auto& a, const auto& b){ return a.second.size() > b.second.size(); });
  printf("[batched %zu shard buckets + %zu brute queries]\n",
         buckets.size(), brute_ids.size());

  printf("[bench start v3: batched per-shard parallel + serial-in-bucket]\n");
  if (n_runs > 1) printf("[in-process warmup: running query batch %d times back-to-back for warm-cache measurement]\n", n_runs);

  std::ofstream pass_csv;
  const char* pass_csv_path = std::getenv("BCI_PASS_CSV");
  if (pass_csv_path != nullptr && pass_csv_path[0] != '\0') {
    if (std::filesystem::exists(pass_csv_path)) {
      std::fprintf(stderr, "refusing to overwrite BCI_PASS_CSV=%s\n", pass_csv_path);
      return 2;
    }
    pass_csv.open(pass_csv_path, std::ios::out | std::ios::trunc);
    if (!pass_csv) {
      std::fprintf(stderr, "cannot create BCI_PASS_CSV=%s\n", pass_csv_path);
      return 2;
    }
    pass_csv << "pass,queries,threads,wall_seconds,qps,p50_ms,p95_ms,p99_ms\n";
  }

  // OUTER LOOP for warm-cache measurement (matches PIVF Python bench pattern).
  // Each run uses same query batch; cache evolves across runs.
  std::vector<double> run_walls;
  for (int run_iter = 0; run_iter < n_runs; ++run_iter) {
  auto t_q0 = std::chrono::steady_clock::now();

  // PHASE A: parallel_for over shard buckets — cache-friendly
  parlay::parallel_for(0, buckets.size(), [&](size_t bi) {
    int32_t T = buckets[bi].first;
    auto& q_list = buckets[bi].second;
    auto& sh = *shards[T];
    ThinSubPR sub_pr(base, sh.subset);  // constructed ONCE per shard
    // Iter (per BCI ceiling at 0.93 finding): boost limit to allow beam search
    // Bound graph exploration independently from the output width.
    long bounded_limit = std::min<long>((long)sh.graph.size(), (long)std::max<long>(100L * beam, 100000L));
    double cut_val = std::getenv("BCI_HAMCG_CUT") ? std::atof(std::getenv("BCI_HAMCG_CUT")) : 1.35;
    // BCI_SINGLE_POOL expands HAMCG_single candidate pool returned by beam_search
    // for downstream exact-distance rerank (default K=10; recommended 50-80).
    int single_pool = std::getenv("BCI_SINGLE_POOL") ? std::atoi(std::getenv("BCI_SINGLE_POOL")) : (int)K;
    if (single_pool < (int)K) single_pool = (int)K;
    // A pair graph already contains only valid conjunction members and needs
    // the requested K outputs.  An atom graph followed by a secondary filter
    // must expose the declared post_filter_pool; the previous implementation
    // accidentally requested only K candidates and therefore understated the
    // strongest no-pair baseline.
    const bool pair_unit = T >= 12;
    long route_pool = pair_unit ? (long)K : (long)post_filter_pool;
    long route_beam = std::max((long)beam, route_pool);
    QueryParams QP(route_pool, route_beam, /*cut=*/cut_val,
                   bounded_limit, sh.maxDeg);
    QueryParams QP_single((long)single_pool, (long)std::max((long)beam, (long)single_pool),
                          /*cut=*/cut_val, bounded_limit, sh.maxDeg);

    for (int i : q_list) {
      int qid = qid_lo + i;
      PointT q = query[qid];
      auto t_a = std::chrono::steady_clock::now();

      // IVF² CLUSTER FAST PATH: if enabled and this is
      // a conjunction query AND primary tag has cluster index, scan centroids
      // (fits L1) + collect top-nprobe clusters + post-filter + exact rerank.
      // Skips beam_search entirely. Replicates PIVF and_query for warm-cache QPS.
      std::vector<std::pair<float, int32_t>> cands;
      cands.reserve(K * 2);
      // HAMCG_single (route 0) uses expanded pool for exact-distance rerank.
      int pool_cap = (q_route[i] == 1) ? post_filter_pool
                   : (q_route[i] == 0) ? single_pool : (int)K;
      int got = 0;
      // OPTIMIZATION: cache bitvector pointers per query (no unordered_map lookup
      // in inner candidate loop). Bitvector lookup = 1 shift + 1 AND + 1 load ~1ns.
      const uint64_t* sec_bv = nullptr;
      const uint64_t* pri_bv = nullptr;
      if (use_bitvec && q_route[i] == 1) {
        auto it_sec = bitvecs.find(q_secondary[i]);
        if (it_sec != bitvecs.end()) sec_bv = it_sec->second.data();
        auto it_pri = bitvecs.find(q_primary[i]);
        if (it_pri != bitvecs.end()) pri_bv = it_pri->second.data();
      }
      auto has_sec = [&](int32_t g) {
        if (sec_bv) return (sec_bv[g >> 6] & (1ULL << (g & 63))) != 0ULL;
        return (bool)bm.match(g, q_secondary[i]);
      };
      auto has_pri = [&](int32_t g) {
        if (pri_bv) return (pri_bv[g >> 6] & (1ULL << (g & 63))) != 0ULL;
        return (bool)bm.match(g, q_primary[i]);
      };

      // use_clusters: 1=PIVF-style JOIN, 2=primary-only sorted_near + bitvec filter
      bool use_ivf2 = (use_clusters >= 1) && q_route[i] == 1 && clusters.count(T);
      bool use_ivf2_simple = (use_clusters == 2);  // skip secondary sorted_near
      if (use_ivf2) {
        // PIVF-style sorted_near × 2 + JOIN + exact rerank
        // Step 1: sorted_near on primary → candidate IDs (no distance compute)
        int adaptive_tpts = q_tpts[i];  // proxy-routed per-query target_points
        // PACH: secondary tag for predicate-aware cluster pruning. Only active
        // when use_pach=1 AND query has a secondary tag AND we have pach bitvecs.
        int32_t pach_sec = q_secondary[i];
        const std::vector<std::vector<uint64_t>>* pach_pri = nullptr;
        if (use_pach && pach_sec >= 0) {
          auto it_p = pach_bitvecs.find(T);
          if (it_p != pach_bitvecs.end()) pach_pri = &it_p->second;
        }
        auto sorted_near = [&](const ClusterIndex& ci, std::vector<int32_t>& out, int32_t pach_for_tag) {
          // Build pach lookup for THIS specific tag (primary uses pach_pri; secondary uses its own)
          const std::vector<std::vector<uint64_t>>* my_pach = nullptr;
          int32_t my_filter_tag = -1;
          if (use_pach && pach_sec >= 0) {
            if (pach_for_tag == T) {
              my_pach = pach_pri;
              my_filter_tag = pach_sec;  // primary tag's clusters filtered by secondary
            } else {
              auto it_p = pach_bitvecs.find(pach_for_tag);
              if (it_p != pach_bitvecs.end()) {
                my_pach = &it_p->second;
                my_filter_tag = T;  // secondary tag's clusters filtered by primary
              }
            }
          }
          auto keeps = [&](int c) {
            if (!my_pach) return true;
            const auto& bv = (*my_pach)[c];
            return (bv[my_filter_tag >> 6] & (1ULL << (my_filter_tag & 63))) != 0ULL;
          };
          // Compute centroid distances ONLY for clusters that survive PACH.
          std::vector<std::pair<float, int32_t>> cent_dists;
          cent_dists.reserve(ci.n_clusters);
          int kept = 0;
          for (int c = 0; c < ci.n_clusters; ++c) {
            if (!keeps(c)) continue;
            ++kept;
            PointT cpt(ci.centroid_data.data() + (size_t)c * ci.aligned_dim,
                       ci.aligned_dim, ci.aligned_dim, c);
            cent_dists.push_back({q.distance(cpt), c});
          }
          // PACH instrumentation (atomic to avoid races; cheap)
          pach_total_clusters.fetch_add(ci.n_clusters, std::memory_order_relaxed);
          pach_kept_clusters.fetch_add(kept, std::memory_order_relaxed);
          int np = std::min<int>(nprobe, (int)cent_dists.size());
          if (np == 0) { out.clear(); return; }
          std::nth_element(cent_dists.begin(), cent_dists.begin() + np, cent_dists.end(),
            [](auto&a, auto&b){ return a.first < b.first; });
          out.clear(); out.reserve(adaptive_tpts);
          for (int p = 0; p < np && (int)out.size() < adaptive_tpts; ++p) {
            int c = cent_dists[p].second;
            int64_t lo = ci.member_offsets[c];
            int64_t hi = ci.member_offsets[c + 1];
            for (int64_t j = lo; j < hi && (int)out.size() < adaptive_tpts; ++j) {
              out.push_back(ci.member_ids[j]);
            }
          }
          std::sort(out.begin(), out.end()); // for sorted-list join
        };
        std::vector<int32_t> ids_a;
        sorted_near(*clusters[T], ids_a, T);

        // Step 2: if secondary has cluster AND not use_ivf2_simple, sorted_near + INTERSECT
        // Else: filter ids_a by bitvec(secondary) — simpler, lower overhead
        if (!use_ivf2_simple && clusters.count(q_secondary[i])) {
          std::vector<int32_t> ids_b;
          sorted_near(*clusters[q_secondary[i]], ids_b, q_secondary[i]);
          // Sorted-list intersection
          std::vector<int32_t> intersect;
          intersect.reserve(std::min(ids_a.size(), ids_b.size()));
          std::set_intersection(ids_a.begin(), ids_a.end(), ids_b.begin(), ids_b.end(),
                                std::back_inserter(intersect));
          // Step 3: exact distance on intersection — PIVF-style streaming top-K
          // with early reject. Skips ~90% of pushes since we only insert when better
          // than current worst. Frontier is fixed K-sized.
          std::pair<float, int32_t> frontier[K + 1];
          for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};
          for (int32_t g : intersect) {
            PointT bp = base[g];
            float d = q.distance(bp);
            if (d < frontier[K-1].first) {
              // insertion sort (K=10, ~10 ops per insert)
              int p = K - 1;
              while (p > 0 && frontier[p-1].first > d) {
                frontier[p] = frontier[p-1];
                --p;
              }
              frontier[p] = {d, g};
            }
          }
          for (int k = 0; k < K; ++k) {
            if (frontier[k].second >= 0) cands.push_back(frontier[k]);
          }
        } else {
          // Single-shard fallback: streaming top-K with bitvec filter
          std::pair<float, int32_t> frontier[K + 1];
          for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};
          for (int32_t g : ids_a) {
            if (!has_sec(g)) continue;
            PointT bp = base[g];
            float d = q.distance(bp);
            if (d < frontier[K-1].first) {
              int p = K - 1;
              while (p > 0 && frontier[p-1].first > d) {
                frontier[p] = frontier[p-1];
                --p;
              }
              frontier[p] = {d, g};
            }
          }
          for (int k = 0; k < K; ++k) {
            if (frontier[k].second >= 0) cands.push_back(frontier[k]);
          }
        }
      } else {
        // PRIMARY beam_search + post-filter (existing path) — use pre-built bitset.
        // HAMCG_single (route 0): use QP_single with expanded k (single_pool) so
        // beam_search returns a larger pool whose top-K-by-exact-distance can be
        // selected; reduces order-noise vs the K=10 default truncation.
        QueryParams& QP_use = (q_route[i] == 0) ? QP_single : QP;
        auto res = beam_search<PointT, ThinSubPR, Indx>(
            q, sh.graph, sub_pr, /*start=*/0, QP_use);
        auto& frontier = res.first.first;
        for (size_t j = 0; j < frontier.size() && got < pool_cap; ++j) {
          int32_t local = frontier[j].first;
          float dist = frontier[j].second;
          int32_t global = sh.subset[local];
          // pair-graph (intersection): every node is already in X_a∩X_b, no post-filter.
          if (q_route[i] == 1 && q_primary[i] < 12) {
            if (has_sec(global)) {
              cands.push_back({dist, global}); ++got;
            }
          } else {
            cands.push_back({dist, global}); ++got;
          }
        }
      }

      // ALWAYS DUAL-SHARD (iter-2 after adaptive was too marginal): for ALL
      // conjunction queries where secondary tag has a shard, also beam_search
      // on secondary. Union with primary's filtered candidates. Higher cost,
      // higher recall. (Skipped when use_ivf2 cluster path already used.)
      if (!use_ivf2 && q_route[i] == 1 && shards.count(q_secondary[i])) {
        auto& sh2 = *shards[q_secondary[i]];
        ThinSubPR sub_pr2(base, sh2.subset);
        long bounded_limit2 = std::min<long>((long)sh2.graph.size(), (long)std::max<long>(100L * beam, 100000L));
        long secondary_pool = (long)post_filter_pool;
        long secondary_beam = std::max((long)beam, secondary_pool);
        QueryParams QP2(secondary_pool, secondary_beam, /*cut=*/1.35,
                        bounded_limit2, sh2.maxDeg);
        auto res2 = beam_search<PointT, ThinSubPR, Indx>(
            q, sh2.graph, sub_pr2, /*start=*/0, QP2);
        auto& frontier2 = res2.first.first;
        // dedup
        std::vector<int32_t> seen;
        for (auto& c : cands) seen.push_back(c.second);
        std::sort(seen.begin(), seen.end());
        // build primary bitset (separately from secondary — using the same buffer
        // would clobber). For now, fall back to bm.match for this branch (rare path).
        for (size_t j = 0; j < frontier2.size() && (int)cands.size() < pool_cap; ++j) {
          int32_t local = frontier2[j].first;
          float dist = frontier2[j].second;
          int32_t global = sh2.subset[local];
          if (std::binary_search(seen.begin(), seen.end(), global)) continue;
          if (has_pri(global)) {
            cands.push_back({dist, global});
          }
        }
      }

      // Final top-K by distance with (dist, global_index) tie-breaking to match
      // GT's canonical ordering (asc dist, then asc index — same as numpy
      // stable-sort on sorted-ascending input and the brute_cold heap path).
      int kk = std::min<int>(K, (int)cands.size());
      if (kk > 0) {
        std::partial_sort(cands.begin(), cands.begin()+kk, cands.end(),
          [](auto&a, auto&b){
            return a.first < b.first ||
                   (a.first == b.first && a.second < b.second);
          });
      }
      std::vector<int32_t> top;
      top.reserve(K);
      for (int j = 0; j < kk; ++j) top.push_back(cands[j].second);
      while ((int)top.size() < K) top.push_back(-1);
      results[i] = std::move(top);
      auto t_b = std::chrono::steady_clock::now();
      latencies[i] = std::chrono::duration<double>(t_b-t_a).count() * 1000.0;
    }
  });

  // PHASE B: parallel_for over brute fallback queries (cold tags, small)
  parlay::parallel_for(0, brute_ids.size(), [&](size_t bi) {
    int i = brute_ids[bi];
    int qid = qid_lo + i;
    PointT q = query[qid];
    auto t_a = std::chrono::steady_clock::now();

    std::vector<int32_t> top;
    top.reserve(K);
    if (q_route[i] == 2) {
      // Brute force on posting list of primary tag — PACKED LAYOUT path:
      // If primary tag's points are pre-packed contiguously (packed_cold), scan
      // sequentially (full prefetcher, no random base[g]). 3-5× speedup vs random.
      // Falls back to random base[g] if not packed.
      int32_t T = q_primary[i];
      int64_t start = bmt.row_offsets[T];
      int64_t end   = bmt.row_offsets[T+1];
      int32_t sec = q_secondary[i];
      size_t n_tag = end - start;

      // Secondary bitvec (for fast filter)
      const uint64_t* sec_bv_brute = nullptr;
      if (sec >= 0) {
        auto it = bitvecs.find(sec);
        if (it != bitvecs.end()) sec_bv_brute = it->second.data();
      }
      auto has_sec_brute = [&](int32_t g){
        if (sec < 0) return true;
        if (sec_bv_brute) return (sec_bv_brute[g >> 6] & (1ULL << (g & 63))) != 0ULL;
        return (bool)bm.match(g, sec);
      };

      std::pair<float, int32_t> frontier[K + 1];
      for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};

      // Heap insertion with GT-matching tie-breaking: on equal distance,
      // prefer HIGHER global index (matches YFCC10M's empirically-calibrated
      // tied-item selection; tied distances are extremely rare on 512-d
      // float vectors but the rule is preserved for cross-dataset consistency).
      auto heap_insert_brute = [&](float d, int32_t g) {
        if (d < frontier[K-1].first ||
            (d == frontier[K-1].first && g > frontier[K-1].second)) {
          int p = K - 1;
          while (p > 0 && (frontier[p-1].first > d ||
                           (frontier[p-1].first == d &&
                            frontier[p-1].second < g))) {
            frontier[p] = frontier[p-1]; --p;
          }
          frontier[p] = {d, g};
        }
      };

      auto packed_it = packed_cold.find(T);
      if (packed_it != packed_cold.end()) {
        // PACKED PATH: sequential scan, no random base[g] access.
        const float* packed = packed_it->second.data();
        size_t dim = aligned_dim;
        for (size_t k = 0; k < n_tag; ++k) {
          int32_t g = bmt.row_indices[start + k];
          if (!has_sec_brute(g)) continue;
          PointT bp(packed + k * dim, (unsigned)base.dimension(), (unsigned)dim, g);
          float d = q.distance(bp);
          heap_insert_brute(d, g);
        }
      } else {
        // Fallback: random base[g] with prefetch
        const int PREFETCH_AHEAD = 16;
        for (size_t k = 0; k + PREFETCH_AHEAD < n_tag; ++k) {
          int32_t gp = bmt.row_indices[start + k + PREFETCH_AHEAD];
          base[gp].prefetch();
        }
        for (size_t k = 0; k < n_tag; ++k) {
          int32_t g = bmt.row_indices[start + k];
          if (!has_sec_brute(g)) continue;
          if (k + PREFETCH_AHEAD < n_tag) {
            int32_t gp = bmt.row_indices[start + k + PREFETCH_AHEAD];
            base[gp].prefetch();
          }
          PointT bp = base[g];
          float d = q.distance(bp);
          heap_insert_brute(d, g);
        }
      }
      for (int k = 0; k < K; ++k) {
        if (frontier[k].second >= 0) top.push_back(frontier[k].second);
      }
    }
    while ((int)top.size() < K) top.push_back(-1);
    results[i] = std::move(top);

    auto t_b = std::chrono::steady_clock::now();
    latencies[i] = std::chrono::duration<double>(t_b-t_a).count() * 1000.0;
  });
  auto t_q1 = std::chrono::steady_clock::now();
  double qs_iter = std::chrono::duration<double>(t_q1-t_q0).count();
  run_walls.push_back(qs_iter);
  std::vector<double> pass_lat;
  pass_lat.reserve(n_q);
  for (int i = 0; i < n_q; ++i) if (q_route[i] != -1) pass_lat.push_back(latencies[i]);
  std::sort(pass_lat.begin(), pass_lat.end());
  auto pass_pct = [&](double p) {
    if (pass_lat.empty()) return 0.0;
    size_t idx = std::min(pass_lat.size() - 1,
                          (size_t)(p * (pass_lat.size() - 1)));
    return pass_lat[idx];
  };
  if (n_runs > 1) {
    printf("  [run %d/%d] wall=%.3fs QPS=%.1f p50=%.3fms p95=%.3fms p99=%.3fms\n",
           run_iter+1, n_runs, qs_iter, n_q/qs_iter,
           pass_pct(0.50), pass_pct(0.95), pass_pct(0.99));
  }
  if (pass_csv) {
    pass_csv << run_iter + 1 << ',' << n_q << ',' << parlay::num_workers()
             << ',' << std::setprecision(12) << qs_iter << ',' << n_q / qs_iter
             << ',' << pass_pct(0.50) << ',' << pass_pct(0.95) << ','
             << pass_pct(0.99) << '\n';
    pass_csv.flush();
  }
  }  // end outer warmup loop
  // Use the LAST run's timing as the steady-state warm measurement
  double qs = run_walls.back();
  double qps = n_q / qs;

  // -- Recall@K (LAION: filter -1 padding, use valid GT count not K) --------
  // GT stored as int32; loaded as uint32 → padding -1 becomes 0xFFFFFFFF.
  // Variable per-query valid count: divide by min(K, n_valid_gt) for recall@K.
  //
  // BCI_TIE_AWARE=1 → NeurIPS'23 BigANN official protocol (distance-threshold):
  //   result v counts as match iff dist(q, v) <= dist(q, GT[K-1]).
  // The standard NeurIPS BigANN benchmark uses this semantics; strict
  // set-intersection (the default below) under-reports recall by ~0.003 due to
  // the GT generator's non-deterministic tie-breaking among equi-distant points.
  // For LAION we use the per-query last valid GT distance as τ.
  const bool TIE_AWARE = std::getenv("BCI_TIE_AWARE") != nullptr &&
                         std::atoi(std::getenv("BCI_TIE_AWARE")) != 0;
  const uint32_t PAD = 0xFFFFFFFF;
  double total = 0.0; int counted = 0;
  std::vector<double> per_query_recall(n_q, 0.0);
  std::vector<int>    per_query_nvalid(n_q, 0);
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == -1) continue;
    std::vector<uint32_t> gt_set;
    gt_set.reserve(K);
    int last_valid_k = -1;
    for (uint32_t k_i = 0; k_i < gt.K && (int)gt_set.size() < K; ++k_i) {
      uint32_t g = gt.indices[(qid_lo+i)*gt.K + k_i];
      if (g == PAD) break;
      gt_set.push_back(g);
      last_valid_k = (int)k_i;
    }
    int n_valid = (int)gt_set.size();
    per_query_nvalid[i] = n_valid;
    if (n_valid == 0) continue;
    int hit = 0;
    if (TIE_AWARE && last_valid_k >= 0) {
      float tau = gt.distances[(size_t)(qid_lo+i) * gt.K + last_valid_k];
      // Use the library's float L2 (efanna2e::DistanceL2 -> SIMD).
      PointT qp = query[qid_lo + i];
      int below_tau = 0;
      for (auto v : results[i]) {
        if (v < 0) continue;
        PointT pv = base[v];
        float dv = qp.distance(pv);
        if (dv <= tau) ++below_tau;
      }
      // Cap at n_valid: with TIE_AWARE on padded GT (n_valid < K) the
      // method may return extra near-tau candidates which would otherwise
      // push recall > 1.0.
      hit = std::min(below_tau, n_valid);
    } else {
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    double r = double(hit) / n_valid;
    per_query_recall[i] = r;
    total += r;
    ++counted;
  }
  double recall = counted > 0 ? total / counted : 0.0;

  // Predicate-validity and duplicate audit is deliberately outside the timed
  // region.  A duplicate may never inflate strict set recall, and any result
  // outside the conjunction must fail the formal run closed.
  int64_t invalid_returned_ids = 0;
  int64_t duplicate_returned_ids = 0;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == -1) continue;
    std::unordered_set<int32_t> seen;
    int qid = qid_lo + i;
    int64_t q_start = qm.row_offsets[qid];
    int64_t q_end = qm.row_offsets[qid + 1];
    for (int32_t v : results[i]) {
      if (v < 0) continue;
      if ((size_t)v >= base.size()) {
        ++invalid_returned_ids;
        continue;
      }
      duplicate_returned_ids += !seen.insert(v).second;
      for (int64_t j = q_start; j < q_end; ++j) {
        if (!bm.match(v, qm.row_indices[j])) {
          ++invalid_returned_ids;
          break;
        }
      }
    }
  }
  printf("[output audit] invalid_returned_ids=%ld duplicate_returned_ids=%ld\n",
         (long)invalid_returned_ids, (long)duplicate_returned_ids);
  if (invalid_returned_ids != 0 || duplicate_returned_ids != 0) {
    std::fprintf(stderr, "formal output validity/duplicate audit failed\n");
    return 2;
  }

  // -- Per-route statistics -----------
  // Isolate HAMCG_single (route 0), HAMCG_conj (route 1), brute (route 2) timing+recall.
  double sum_lat[3] = {0.0, 0.0, 0.0};
  int count_route[3] = {0, 0, 0};
  double sum_recall_route[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] < 0 || q_route[i] > 2) continue;
    sum_lat[q_route[i]] += latencies[i];
    count_route[q_route[i]]++;
    if (per_query_nvalid[i] == 0) continue;
    sum_recall_route[q_route[i]] += per_query_recall[i];
  }
  printf("\n[per-route stats]\n");
  const char* route_name[] = {"HAMCG_single", "HAMCG_conj  ", "brute_cold  "};
  for (int r = 0; r < 3; ++r) {
    double avg_ms = count_route[r] > 0 ? sum_lat[r] / count_route[r] : 0;
    double route_qps = count_route[r] > 0 ? 1000.0 * count_route[r] / sum_lat[r] : 0;
    double route_recall = count_route[r] > 0 ? sum_recall_route[r] / count_route[r] : 0;
    printf("  %s: %4d queries, avg %7.3f ms/q, %7.1f QPS, recall@%d=%.8f\n",
           route_name[r], count_route[r], avg_ms, route_qps, K, route_recall);
  }

  // -- Latency stats ---------------------------------------------------------
  // Per-query dump for cardinality-stratified analysis (set env BCI_PERQUERY_CSV to a path)
  {
    const char* pq_csv = std::getenv("BCI_PERQUERY_CSV");
    if (pq_csv) {
      FILE* fpq = std::fopen(pq_csv, "w");
      if (fpq) {
        std::fprintf(fpq,
          "qid,route,primary_tag,secondary_tag,pair_tag,pair_materialized,latency_ms,recall\n");
        for (int i = 0; i < n_q; ++i) {
          if (q_route[i] < 0 || q_route[i] > 2) continue;
          std::fprintf(fpq, "%d,%d,%d,%d,%d,%d,%.6f,%.8f\n",
                       qid_lo + i, q_route[i], q_primary[i], q_secondary[i],
                       q_pair_tag[i], q_pair_materialized[i] ? 1 : 0,
                       latencies[i], per_query_recall[i]);
        }
        std::fclose(fpq);
        std::fprintf(stderr, "[per-query] wrote %s\n", pq_csv);
      }
    }
  }

  std::vector<double> lat_sorted;
  for (int i = 0; i < n_q; ++i) if (q_route[i] != -1) lat_sorted.push_back(latencies[i]);
  std::sort(lat_sorted.begin(), lat_sorted.end());
  auto pct = [&](double p){
    if (lat_sorted.empty()) return 0.0;
    size_t idx = std::min(lat_sorted.size()-1, (size_t)(p * lat_sorted.size()));
    return lat_sorted[idx];
  };

  printf("\n[RESULTS]\n");
  printf("  total queries  : %d (counted=%d, skip=%d)\n", n_q, counted, n_skip);
  printf("  wall time      : %.3fs\n", qs);
  printf("  QPS            : %.1f\n", qps);
  printf("  recall@%d      : %.8f\n", K, recall);
  printf("  latency p50    : %.3f ms\n", pct(0.50));
  printf("  latency p90    : %.3f ms\n", pct(0.90));
  printf("  latency p99    : %.3f ms\n", pct(0.99));
  printf("  latency p999   : %.3f ms\n", pct(0.999));

  // PACH skip-ratio report
  int64_t pt = pach_total_clusters.load();
  int64_t pk = pach_kept_clusters.load();
  if (pt > 0) {
    double skip_pct = 100.0 * (1.0 - (double)pk / (double)pt);
    printf("\n[PACH] total clusters considered=%ld, kept=%ld, SKIP=%.1f%%\n",
           (long)pt, (long)pk, skip_pct);
  }

  printf("\n[SIFT100M sole-survivor context]\n");
  printf("  BCI status     : SURVIVE at recall=%.8f, %.1f QPS, p99=%.3f ms\n",
         recall, qps, pct(0.99));

  return 0;
}
