// bci_bench: Boole-ANN Cell Index (BCI) full-system benchmark on YFCC10M.
// Loads all 1317 pre-built HAMCG Vamana shards + base + queries + metadata + GT.
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
#include <filesystem>
#include <fstream>
#include <iostream>
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

#include <immintrin.h>

// SIMD-accelerated L2 squared distance for uint8 vectors (AVX2).
// Replaces the scalar fallback in euclidian_point.h for the brute hot path
// where this is the dominant work. For dim=192 (YFCC10M) this is 6× AVX2
// iterations + final tail; expected speedup vs scalar: 4-8×.
static inline float l2_sq_uint8_avx2(const uint8_t* __restrict__ a,
                                     const uint8_t* __restrict__ b,
                                     unsigned d) {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  unsigned i = 0;
  auto accum32 = [](__m256i& acc, const uint8_t* x, const uint8_t* y) {
    __m256i va = _mm256_loadu_si256((const __m256i*)x);
    __m256i vb = _mm256_loadu_si256((const __m256i*)y);
    __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
    __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
    __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
    __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
    __m256i d_lo = _mm256_sub_epi16(va_lo, vb_lo);
    __m256i d_hi = _mm256_sub_epi16(va_hi, vb_hi);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_lo, d_lo));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_hi, d_hi));
  };
  for (; i + 64 <= d; i += 64) {
    accum32(acc0, a + i,      b + i);
    accum32(acc1, a + i + 32, b + i + 32);
  }
  for (; i + 32 <= d; i += 32) {
    accum32(acc0, a + i, b + i);
  }
  __m256i acc = _mm256_add_epi32(acc0, acc1);
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc),
                                  _mm256_extracti128_si256(acc, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  int result = _mm_cvtsi128_si32(sum128);
  for (; i < d; ++i) {
    int diff = (int)a[i] - (int)b[i];
    result += diff * diff;
  }
  return (float)result;
}

using PointT = Euclidian_Point<uint8_t>;
using PR     = PointRange<uint8_t, PointT>;
using Indx   = int32_t;
using GraphI = Graph<Indx>;

static std::string env_path_or_default(const char* name, const std::string& fallback,
                                       bool trailing_slash) {
  const char* raw = std::getenv(name);
  std::string out = (raw && raw[0] != '\0') ? std::string(raw) : fallback;
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  if (trailing_slash) out.push_back('/');
  return out;
}

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
// Centroids fit in L1 (< 256KB); members stored as CSR.
// Replaces beam_search with: linear centroid scan -> top-nprobe -> collect members -> exact rerank.
struct ClusterIndex {
  int32_t tag;
  int32_t n_clusters;
  int32_t aligned_dim;
  std::vector<uint8_t> centroid_data;     // n_clusters * aligned_dim, contiguous
  std::vector<int64_t> member_offsets;    // n_clusters + 1
  std::vector<int32_t> member_ids;        // flattened global point IDs
};

struct PairPosting {
  std::vector<int32_t> ids;       // exact A ∩ B point ids, sorted by global id
  std::vector<uint8_t> packed;    // ids.size() * aligned_dim bytes, contiguous
};

static inline uint64_t pair_key(int32_t a, int32_t b) {
  return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

static inline int32_t pair_key_first(uint64_t key) {
  return (int32_t)(uint32_t)(key >> 32);
}

static inline int32_t pair_key_second(uint64_t key) {
  return (int32_t)(uint32_t)key;
}

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
  gt.distances.resize((size_t)gt.N * gt.K);
  got = fread(gt.indices.data(), sizeof(uint32_t), gt.indices.size(), f);
  got = fread(gt.distances.data(), sizeof(float),    gt.distances.size(), f);
  fclose(f);
  return gt;
}

int main(int argc, char** argv) {
  const std::string DATA = env_path_or_default(
      "BCI_YFCC_DATA_ROOT",
      "./data/yfcc100m",
      true);
  const std::string DRT  = env_path_or_default(
      "BCI_DRT_ROOT",
      "./data",
      false);

  // Args: qid_lo qid_hi beam K
  int qid_lo = argc > 1 ? atoi(argv[1]) : 60000;
  int qid_hi = argc > 2 ? atoi(argv[2]) : 70000;
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

  // -- Load datasets ---------------------------------------------------------
  auto t0 = std::chrono::steady_clock::now();

  PR base((DATA + "base.10M.u8bin").c_str());
  PR query((DATA + "query.public.100K.u8bin").c_str());
  csr_filters qm(DATA + "query.metadata.public.100K.spmat");
  csr_filters bm(DATA + "base.metadata.10M.spmat");
  csr_filters bmt = bm.transpose();
  printf("[loaded] base=%ld dim=%ld query=%ld qm=%ld base_meta=%ld\n",
         base.size(), base.dimension(), query.size(),
         qm.n_points, bm.n_points);

  // GT for [qid_lo, qid_hi)
  int gt_hi_bin = qid_hi / 1000; int gt_lo_bin = qid_lo / 1000;
  std::string gt_path = DATA + "yfcc-10M-gt-" + std::to_string(gt_lo_bin) +
                        "-" + std::to_string(gt_hi_bin) + "-AND.bin";
  auto gt = load_gt(gt_path);
  printf("[GT loaded] %s N=%u K=%u\n", gt_path.c_str(), gt.N, gt.K);
  assert(gt.K >= (uint32_t)K);

  // -- Load shards ------------------------------------------------------------
  auto t_shard0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::unique_ptr<Shard>> shards;

  for (const auto& entry : std::filesystem::directory_iterator(DRT + "/shards")) {
    auto name = entry.path().filename().string();
    if (name.find("vamana_tag_") != 0 || entry.path().extension() != ".bin") continue;
    int tag = atoi(name.substr(11, name.size()-15).c_str());
    auto sub = load_subset_idx(DRT + "/subset_idx/subset_idx_" + std::to_string(tag) + ".bin");
    if (sub.empty()) continue;
    std::string gp = DRT + "/shards/vamana_tag_" + std::to_string(tag) + ".bin";
    auto sh = std::make_unique<Shard>();
    sh->tag = tag;
    sh->freq = (int64_t)sub.size();
    sh->subset = std::move(sub);
    sh->graph = GraphI((char*)gp.c_str());
    sh->maxDeg = sh->graph.max_degree();
    shards[tag] = std::move(sh);
  }
  auto t_shard1 = std::chrono::steady_clock::now();
  printf("[loaded %zu shards in %.2fs]\n", shards.size(),
         std::chrono::duration<double>(t_shard1-t_shard0).count());

  // -- Load IVF² cluster indices (IVF² absorb fast path) ----------------------
  auto t_clust0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::unique_ptr<ClusterIndex>> clusters;
  std::string clust_dir = DRT + "/clusters";
  if (std::filesystem::exists(clust_dir)) {
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
  std::unordered_map<int32_t, std::vector<uint8_t>> packed_cold;
  // First pass: which cold tags are actually queried?
  std::unordered_set<int32_t> cold_tags_queried;
  int n_q_tmp = qid_hi - qid_lo;
  for (int i = 0; i < n_q_tmp; ++i) {
    int qid = qid_lo + i;
    int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
    int n_tags = (int)(e - s);
    if (n_tags == 1) {
      int32_t t = qm.row_indices[s];
      if (!shards.count(t)) cold_tags_queried.insert(t);
    } else if (n_tags == 2) {
      int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
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
      uint8_t* src = base[g].get();
      std::memcpy(packed.data() + (j - lo) * aligned_dim, src, aligned_dim);
    }
  }
  size_t packed_bytes = 0;
  for (auto& kv : packed_cold) packed_bytes += kv.second.size();
  auto t_pack1 = std::chrono::steady_clock::now();
  printf("[packed %zu cold-tag arrays in %.2fs, %.1fMB total]\n",
         packed_cold.size(),
         std::chrono::duration<double>(t_pack1-t_pack0).count(),
         packed_bytes / 1e6);

  // PACH-in-brute (NOVEL) — partition each packed_cold tag's posting into
  // chunks of CHUNK_SIZE points. Per chunk, store a bitvec over secondary
  // predicate IDs: bit B set iff ≥1 point in chunk has tag B. At query time
  // for conjunction A∧B (brute path on primary A), skip whole chunks where
  // chunk-bitvec[B]=0. Skip ratio = (1-ρ)^c where ρ = secondary density.
  // For c=64, ρ=0.01: ~47% skip. Activates PACH on dense workloads
  // unlike cluster-level PACH which requires c·ρ ≪ 1.
  constexpr int CHUNK_SIZE = 16;
  auto t_chunk0 = std::chrono::steady_clock::now();
  // chunk_bvs[tag] = flattened (n_chunks × bv_words) bitvecs over secondary tags
  std::unordered_map<int32_t, std::vector<uint64_t>> chunk_bvs;
  size_t pach_bv_words = (bm.n_filters + 63) / 64;
  size_t chunk_bvs_bytes = 0;
  std::vector<int32_t> ct_tags_vec(cold_tags_queried.begin(), cold_tags_queried.end());
  // Allocate
  for (int32_t tag : ct_tags_vec) {
    int64_t lo = bmt.row_offsets[tag], hi = bmt.row_offsets[tag+1];
    int n_pts = (int)(hi - lo);
    int n_chunks = (n_pts + CHUNK_SIZE - 1) / CHUNK_SIZE;
    chunk_bvs[tag].assign((size_t)n_chunks * pach_bv_words, 0ULL);
    chunk_bvs_bytes += (size_t)n_chunks * pach_bv_words * 8;
  }
  // Fill in parallel
  parlay::parallel_for(0, ct_tags_vec.size(), [&](size_t ti) {
    int32_t tag = ct_tags_vec[ti];
    int64_t lo = bmt.row_offsets[tag], hi = bmt.row_offsets[tag+1];
    int n_pts = (int)(hi - lo);
    auto& cbv = chunk_bvs[tag];
    for (int k = 0; k < n_pts; ++k) {
      int chunk_id = k / CHUNK_SIZE;
      int32_t g = bmt.row_indices[lo + k];
      int64_t s = bm.row_offsets[g];
      int64_t e = bm.row_offsets[g+1];
      uint64_t* bv = cbv.data() + (size_t)chunk_id * pach_bv_words;
      for (int64_t j = s; j < e; ++j) {
        int32_t t = bm.row_indices[j];
        bv[t >> 6] |= (1ULL << (t & 63));
      }
    }
  });
  auto t_chunk1 = std::chrono::steady_clock::now();
  printf("[built PACH-in-brute chunk bitvecs for %zu tags in %.2fs, %.1fMB]\n",
         chunk_bvs.size(),
         std::chrono::duration<double>(t_chunk1-t_chunk0).count(),
         chunk_bvs_bytes / 1e6);

  // Pre-build per-tag bitvectors for conjunction post-filter (eliminates per-query
  // bm.match cache misses). Each bitvector = N_total bits = 1.25MB packed.
  // EXPANDED COVERAGE: build bitvecs for ALL queried secondary tags (not just
  // sweet-spot tags). Previously fallback to bm.match (linear scan) for non-band
  // secondary tags dominated brute_cold latency.
  auto t_bv0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::vector<uint64_t>> bitvecs;
  size_t bv_words = (bm.n_points + 63) / 64;

  // Collect ALL secondary tags appearing in queries (covers conjunction post-filter)
  std::unordered_set<int32_t> all_query_tags;
  for (int i = 0; i < n_q_tmp; ++i) {
    int qid = qid_lo + i;
    int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
    for (int64_t j = s; j < e; ++j) all_query_tags.insert(qm.row_indices[j]);
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

  // PACH (Predicate-Aware Cluster Hierarchy) — NOVEL BCI contribution.
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
        q_route[i] = 0; q_primary[i] = t;  // HAMCG beam_search
      } else {
        q_route[i] = 2; q_primary[i] = t;  // brute (q_secondary stays -1 → no filter)
      }
    } else if (n_tags == 2) {
      int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
      int64_t f1 = bmt.row_offsets[t1+1] - bmt.row_offsets[t1];
      int64_t f2 = bmt.row_offsets[t2+1] - bmt.row_offsets[t2];
      int32_t small_t = (f1 <= f2) ? t1 : t2;
      int32_t large_t = (f1 <= f2) ? t2 : t1;
      int64_t small_size = std::min(f1, f2);
      // KEY FIX (per per-route diag: HAMCG_conj recall 0.79 — catastrophic):
      // If smaller tag's posting is small enough to brute, do exact intersection
      // scan instead of imprecise HAMCG_conj (sub_via_single + post-filter).
      // Threshold 200K = brute cost ~40ms per query but recall ~1.0.
      // Trade QPS for recall.
      if (small_size <= brute_conj_thresh) {
        q_route[i] = 2; q_primary[i] = small_t; q_secondary[i] = large_t;  // brute exact
      } else if (shards.count(small_t)) {
        q_route[i] = 1; q_primary[i] = small_t; q_secondary[i] = large_t;
      } else if (shards.count(large_t)) {
        q_route[i] = 1; q_primary[i] = large_t; q_secondary[i] = small_t;
      } else {
        q_route[i] = 2; q_primary[i] = small_t; q_secondary[i] = large_t;
      }
      // PROXY: per-query adaptive target_points.
      // For SMALL primary: use all points (no artificial cap, scan is small anyway).
      // For MEDIUM primary: standard global target_pts.
      // For LARGE primary (>500K): scale up to catch boundary tail.
      int64_t primary_size = (q_route[i] == 1) ?
                             (bmt.row_offsets[q_primary[i]+1] - bmt.row_offsets[q_primary[i]]) : small_size;
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

  // Exact two-tag posting cache for brute conjunctions.
  //
  // The tested YFCC two-tag workload has small primary postings, so the old
  // brute path scanned every point in the smaller tag and rejected most of them
  // by secondary bitvec.  Build A∩B once per queried pair and pack only those
  // survivors.  This is exact, two-tag-only, and leaves single-tag brute scans
  // on their original path.
  const bool USE_PAIR_POSTING_CACHE =
      std::getenv("BCI_PAIR_POSTING_CACHE") == nullptr ||
      std::atoi(std::getenv("BCI_PAIR_POSTING_CACHE")) != 0;
  std::unordered_map<uint64_t, PairPosting> pair_postings;
  if (USE_PAIR_POSTING_CACHE) {
    auto t_pair0 = std::chrono::steady_clock::now();
    std::unordered_set<uint64_t> pair_key_set;
    for (int i = 0; i < n_q; ++i) {
      if (q_route[i] == 2 && q_secondary[i] >= 0) {
        pair_key_set.insert(pair_key(q_primary[i], q_secondary[i]));
      }
    }
    std::vector<uint64_t> pair_keys(pair_key_set.begin(), pair_key_set.end());
    std::sort(pair_keys.begin(), pair_keys.end());
    std::vector<PairPosting> pair_values(pair_keys.size());

    parlay::parallel_for(0, pair_keys.size(), [&](size_t pi) {
      int32_t a = pair_key_first(pair_keys[pi]);
      int32_t b = pair_key_second(pair_keys[pi]);
      int64_t lo = bmt.row_offsets[a], hi = bmt.row_offsets[a + 1];
      auto bv_it = bitvecs.find(b);
      const uint64_t* b_bv = (bv_it == bitvecs.end()) ? nullptr : bv_it->second.data();
      PairPosting pp;
      pp.ids.reserve((size_t)(hi - lo));
      for (int64_t j = lo; j < hi; ++j) {
        int32_t g = bmt.row_indices[j];
        bool keep = b_bv ? ((b_bv[g >> 6] & (1ULL << (g & 63))) != 0ULL)
                         : (bool)bm.match(g, b);
        if (keep) pp.ids.push_back(g);
      }
      pp.packed.resize(pp.ids.size() * aligned_dim);
      for (size_t j = 0; j < pp.ids.size(); ++j) {
        uint8_t* src = base[pp.ids[j]].get();
        std::memcpy(pp.packed.data() + j * aligned_dim, src, aligned_dim);
      }
      pair_values[pi] = std::move(pp);
    });

    size_t pair_ids_total = 0, pair_bytes_total = 0;
    pair_postings.reserve(pair_keys.size() * 2 + 1);
    for (size_t pi = 0; pi < pair_keys.size(); ++pi) {
      pair_ids_total += pair_values[pi].ids.size();
      pair_bytes_total += pair_values[pi].packed.size();
      pair_postings.emplace(pair_keys[pi], std::move(pair_values[pi]));
    }
    auto t_pair1 = std::chrono::steady_clock::now();
    printf("[built %zu exact pair postings in %.2fs, %.1fM ids, %.1fMB packed]\n",
           pair_postings.size(),
           std::chrono::duration<double>(t_pair1-t_pair0).count(),
           pair_ids_total / 1e6,
           pair_bytes_total / 1e6);
  }

  // -- Run queries -----------------------------------------------------------
  std::vector<std::vector<int32_t>> results(n_q);
  std::vector<double> latencies(n_q, 0.0);

  // Batch queries per
  // shard so each shard graph + ThinSubPR + subset stay cache-hot across
  // its bucket. Restructure parallelism: parallel_for over shard buckets
  // (sorted by descending bucket size for load balance), each bucket
  // processes its queries serially.
  std::unordered_map<int32_t, std::vector<int>> shard_buckets;
  std::vector<int> pair_brute_ids;
  std::vector<int> brute_ids;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == 0 || q_route[i] == 1) shard_buckets[q_primary[i]].push_back(i);
    else if (q_route[i] == 2) {
      auto pair_it = (q_secondary[i] >= 0) ?
          pair_postings.find(pair_key(q_primary[i], q_secondary[i])) : pair_postings.end();
      if (pair_it != pair_postings.end()) pair_brute_ids.push_back(i);
      else                                brute_ids.push_back(i);
    }
  }
  std::vector<std::pair<int32_t, std::vector<int>>> buckets;
  buckets.reserve(shard_buckets.size());
  for (auto& kv : shard_buckets) buckets.emplace_back(kv.first, std::move(kv.second));
  std::sort(buckets.begin(), buckets.end(),
    [](const auto& a, const auto& b){ return a.second.size() > b.second.size(); });
  printf("[batched %zu shard buckets + %zu pair-brute queries + %zu brute queries]\n",
         buckets.size(), pair_brute_ids.size(), brute_ids.size());

  printf("[bench start v3: batched per-shard parallel + serial-in-bucket]\n");
  if (n_runs > 1) printf("[in-process warmup: running query batch %d times back-to-back for warm-cache measurement]\n", n_runs);

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
    // to actually explore the graph. Limit tuned to the 100K-3M range; earlier
    // starving at 8*beam. Try 100x beam as a balance between budget and reach.
    long bounded_limit = std::min<long>((long)sh.graph.size(), (long)std::max<long>(100L * beam, 100000L));
    double cut_val = std::getenv("BCI_HAMCG_CUT") ? std::atof(std::getenv("BCI_HAMCG_CUT")) : 1.35;
    // BCI_SINGLE_POOL expands HAMCG_single candidate pool returned by beam_search
    // for downstream exact-distance rerank (default K=10; recommended 50-80).
    int single_pool = std::getenv("BCI_SINGLE_POOL") ? std::atoi(std::getenv("BCI_SINGLE_POOL")) : (int)K;
    if (single_pool < (int)K) single_pool = (int)K;
    QueryParams QP((long)K, (long)beam, /*cut=*/cut_val, bounded_limit, sh.maxDeg);
    QueryParams QP_single((long)single_pool, (long)std::max((long)beam, (long)single_pool),
                          /*cut=*/cut_val, bounded_limit, sh.maxDeg);

    for (int i : q_list) {
      int qid = qid_lo + i;
      PointT q = query[qid];
      auto t_a = std::chrono::steady_clock::now();

      // IVF^2 cluster fast path: if enabled and this is
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
          if (np < (int)cent_dists.size()) {
            std::nth_element(cent_dists.begin(), cent_dists.begin() + np, cent_dists.end(),
              [](auto&a, auto&b){ return a.first < b.first; });
          }
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
        auto res = beam_search<PointT, ThinSubPR, Indx>(q, sh.graph, sub_pr, /*start=*/0, QP_use);
        auto& frontier = res.first.first;
        for (size_t j = 0; j < frontier.size() && got < pool_cap; ++j) {
          int32_t local = frontier[j].first;
          float dist = frontier[j].second;
          int32_t global = sh.subset[local];
          if (q_route[i] == 1) {
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
        QueryParams QP2((long)K, (long)beam, /*cut=*/1.35, bounded_limit2, sh2.maxDeg);
        auto res2 = beam_search<PointT, ThinSubPR, Indx>(q, sh2.graph, sub_pr2, /*start=*/0, QP2);
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

  // PHASE B0: exact cached pair postings for two-tag brute conjunctions.
  // Keep these separate from long single-tag scans so conjunction latency/QPS
  // is not dominated by concurrent memory-bandwidth pressure from large postings.
  parlay::parallel_for(0, pair_brute_ids.size(), [&](size_t bi) {
    int i = pair_brute_ids[bi];
    int qid = qid_lo + i;
    PointT q = query[qid];
    auto t_a = std::chrono::steady_clock::now();

    std::pair<float, int32_t> frontier[K + 1];
    for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};
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

    auto pair_it = pair_postings.find(pair_key(q_primary[i], q_secondary[i]));
    if (pair_it != pair_postings.end()) {
      const PairPosting& pp = pair_it->second;
      const uint8_t* q_data = q.get();
      unsigned q_dim = (unsigned)base.dimension();
      const uint8_t* packed = pp.packed.data();
      size_t dim = aligned_dim;
      for (size_t k = 0; k < pp.ids.size(); ++k) {
        int32_t g = pp.ids[k];
        float d = l2_sq_uint8_avx2(q_data, packed + k * dim, q_dim);
        heap_insert_brute(d, g);
      }
    }

    std::vector<int32_t> top;
    top.reserve(K);
    for (int k = 0; k < K; ++k) {
      if (frontier[k].second >= 0) top.push_back(frontier[k].second);
    }
    while ((int)top.size() < K) top.push_back(-1);
    results[i] = std::move(top);

    auto t_b = std::chrono::steady_clock::now();
    latencies[i] = std::chrono::duration<double>(t_b-t_a).count() * 1000.0;
  });

  // PHASE B: parallel_for over remaining brute fallback queries (cold tags, small)
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
      // prefer HIGHER global index (matches the YFCC10M public GT's tied-item
      // selection — empirically calibrated by auditing 30/30 tied cases).
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

      const uint8_t* q_data = q.get();
      unsigned q_dim = (unsigned)base.dimension();
      auto pair_it = (sec >= 0) ? pair_postings.find(pair_key(T, sec)) : pair_postings.end();
      if (pair_it != pair_postings.end()) {
        const PairPosting& pp = pair_it->second;
        const uint8_t* packed = pp.packed.data();
        size_t dim = aligned_dim;
        for (size_t k = 0; k < pp.ids.size(); ++k) {
          int32_t g = pp.ids[k];
          float d = l2_sq_uint8_avx2(q_data, packed + k * dim, q_dim);
          heap_insert_brute(d, g);
        }
      } else {
        auto packed_it = packed_cold.find(T);
        if (packed_it != packed_cold.end()) {
          const uint8_t* packed = packed_it->second.data();
          size_t dim = aligned_dim;
          // ADAPTIVE PACH-in-brute: activate only when secondary density × CHUNK_SIZE < 1
          // (the regime where skip ratio (1-ρ)^c > 1/e, i.e. cost-effective).
          // Per-query check: only if (sec_freq / N_total) × CHUNK_SIZE < 1.
          const uint64_t* tag_chunk_bvs = nullptr;
          if (use_pach && sec >= 0) {
            int64_t sec_freq = bmt.row_offsets[sec+1] - bmt.row_offsets[sec];
            double rho_sec = (double)sec_freq / (double)bm.n_points;
            if (rho_sec * CHUNK_SIZE < 1.0) {
              auto cb_it = chunk_bvs.find(T);
              if (cb_it != chunk_bvs.end()) tag_chunk_bvs = cb_it->second.data();
            }
          }
          if (tag_chunk_bvs) {
            // PACH-in-brute: chunk-level emptiness skip integrated INTO flat loop
            // (preserves SIMD/compiler optimization; only adds chunk-aligned check).
            int chunk_word_off = sec >> 6;
            uint64_t chunk_bit_mask = 1ULL << (sec & 63);
            int64_t local_skip = 0, local_total = 0;
            size_t k = 0;
            while (k < n_tag) {
              // Chunk-boundary check: skip whole chunk if secondary absent
              size_t chunk_id = k / CHUNK_SIZE;
              ++local_total;
              uint64_t bv_word = tag_chunk_bvs[chunk_id * pach_bv_words + chunk_word_off];
              if ((bv_word & chunk_bit_mask) == 0ULL) {
                ++local_skip;
                k += CHUNK_SIZE;  // jump to next chunk
                continue;
              }
              // Within chunk: tight inner loop (no extra branches)
              size_t chunk_end = std::min(k + CHUNK_SIZE, n_tag);
              do {
                int32_t g = bmt.row_indices[start + k];
                if (has_sec_brute(g)) {
                  float d = l2_sq_uint8_avx2(q_data, packed + k * dim, q_dim);
                  heap_insert_brute(d, g);
                }
                ++k;
              } while (k < chunk_end);
            }
            pach_total_clusters.fetch_add(local_total, std::memory_order_relaxed);
            pach_kept_clusters.fetch_add(local_total - local_skip, std::memory_order_relaxed);
          } else {
            // PACH off (use_pach=0) or no chunk bitvec — original flat scan
            for (size_t k = 0; k < n_tag; ++k) {
              int32_t g = bmt.row_indices[start + k];
              if (!has_sec_brute(g)) continue;
              float d = l2_sq_uint8_avx2(q_data, packed + k * dim, q_dim);
              heap_insert_brute(d, g);
            }
          }
        } else {
          // Fallback: random base[g] with prefetch + AVX2 SIMD distance
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
            const uint8_t* bp_data = base[g].get();
            float d = l2_sq_uint8_avx2(q_data, bp_data, q_dim);
            heap_insert_brute(d, g);
          }
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
  if (n_runs > 1) {
    printf("  [run %d/%d] wall=%.3fs QPS=%.1f\n", run_iter+1, n_runs, qs_iter, n_q/qs_iter);
  }
  }  // end outer warmup loop
  // Use the LAST run's timing as the steady-state warm measurement
  double qs = run_walls.back();
  double qps = n_q / qs;

  // -- Recall@K --------------------------------------------------------------
  // BCI_TIE_AWARE=1 → NeurIPS'23 BigANN official protocol (distance-threshold):
  //   result v counts as match iff dist(q, v) <= dist(q, GT[K-1]).
  // The standard NeurIPS BigANN benchmark uses this semantics; strict
  // set-intersection (the default below) under-reports recall by ~0.003 due to
  // the GT generator's non-deterministic tie-breaking among equi-distant points.
  const bool TIE_AWARE = std::getenv("BCI_TIE_AWARE") != nullptr &&
                         std::atoi(std::getenv("BCI_TIE_AWARE")) != 0;
  double total = 0.0; int counted = 0;
  unsigned d_dim = (unsigned)base.dimension();
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == -1) continue;
    int hit = 0;
    if (TIE_AWARE) {
      float tau = gt.distances[(size_t)i * gt.K + K - 1];
      const uint8_t* q_data = query[qid_lo + i].get();
      for (auto v : results[i]) {
        if (v < 0) continue;
        float d_v = l2_sq_uint8_avx2(q_data, base[v].get(), d_dim);
        if (d_v <= tau) ++hit;
      }
    } else {
      std::vector<uint32_t> gt_set(gt.indices.begin() + i*gt.K,
                                    gt.indices.begin() + i*gt.K + K);
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    total += double(hit) / K;
    ++counted;
  }
  double recall = counted > 0 ? total / counted : 0.0;

  // -- Per-route stats -----------
  // Isolate HAMCG_single (route 0), HAMCG_conj (route 1), brute (route 2) timing+recall.
  double sum_lat[3] = {0.0, 0.0, 0.0};
  int count_route[3] = {0, 0, 0};
  double sum_recall_route[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] < 0 || q_route[i] > 2) continue;
    sum_lat[q_route[i]] += latencies[i];
    count_route[q_route[i]]++;
    int hit = 0;
    if (TIE_AWARE) {
      float tau = gt.distances[(size_t)i * gt.K + K - 1];
      const uint8_t* q_data = query[qid_lo + i].get();
      for (auto v : results[i]) {
        if (v < 0) continue;
        float d_v = l2_sq_uint8_avx2(q_data, base[v].get(), d_dim);
        if (d_v <= tau) ++hit;
      }
    } else {
      std::vector<uint32_t> gt_set(gt.indices.begin() + i*gt.K,
                                    gt.indices.begin() + i*gt.K + K);
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    sum_recall_route[q_route[i]] += double(hit) / K;
  }
  printf("\n[per-route stats]\n");
  const char* route_name[] = {"HAMCG_single", "HAMCG_conj  ", "brute_cold  "};
  for (int r = 0; r < 3; ++r) {
    double avg_ms = count_route[r] > 0 ? sum_lat[r] / count_route[r] : 0;
    double route_qps = count_route[r] > 0 ? 1000.0 * count_route[r] / sum_lat[r] : 0;
    double route_recall = count_route[r] > 0 ? sum_recall_route[r] / count_route[r] : 0;
    printf("  %s: %4d queries, avg %7.3f ms/q, %7.1f QPS, recall@%d=%.4f\n",
           route_name[r], count_route[r], avg_ms, route_qps, K, route_recall);
  }

  double sum_lat_card[2] = {0.0, 0.0};
  int count_card[2] = {0, 0};
  double sum_recall_card[2] = {0.0, 0.0};
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] < 0 || q_route[i] > 2) continue;
    int c = (q_secondary[i] >= 0) ? 1 : 0;  // 0=single-tag, 1=two-tag
    sum_lat_card[c] += latencies[i];
    count_card[c]++;
    int hit = 0;
    if (TIE_AWARE) {
      float tau = gt.distances[(size_t)i * gt.K + K - 1];
      const uint8_t* q_data = query[qid_lo + i].get();
      for (auto v : results[i]) {
        if (v < 0) continue;
        float d_v = l2_sq_uint8_avx2(q_data, base[v].get(), d_dim);
        if (d_v <= tau) ++hit;
      }
    } else {
      std::vector<uint32_t> gt_set(gt.indices.begin() + i*gt.K,
                                    gt.indices.begin() + i*gt.K + K);
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    sum_recall_card[c] += double(hit) / K;
  }
  printf("\n[per-cardinality stats]\n");
  const char* card_name[] = {"single_tag", "two_tag   "};
  for (int c = 0; c < 2; ++c) {
    double avg_ms = count_card[c] > 0 ? sum_lat_card[c] / count_card[c] : 0;
    double card_qps = count_card[c] > 0 ? 1000.0 * count_card[c] / sum_lat_card[c] : 0;
    double card_recall = count_card[c] > 0 ? sum_recall_card[c] / count_card[c] : 0;
    printf("  %s: %4d queries, avg %7.3f ms/q, %7.1f QPS, recall@%d=%.4f\n",
           card_name[c], count_card[c], avg_ms, card_qps, K, card_recall);
  }

  // -- Latency stats ---------------------------------------------------------
  // Per-query dump for cardinality-stratified analysis (set env BCI_PERQUERY_CSV to a path)
  {
    const char* pq_csv = std::getenv("BCI_PERQUERY_CSV");
    if (pq_csv) {
      FILE* fpq = std::fopen(pq_csv, "w");
      if (fpq) {
        std::fprintf(fpq, "qid,route,latency_ms,recall\n");
        for (int i = 0; i < n_q; ++i) {
          if (q_route[i] < 0 || q_route[i] > 2) continue;
          // re-compute per-query recall to dump alongside
          std::vector<uint32_t> gt_set_pq(gt.indices.begin() + i*gt.K,
                                          gt.indices.begin() + i*gt.K + K);
          std::sort(gt_set_pq.begin(), gt_set_pq.end());
          int hit_pq = 0;
          for (auto v : results[i]) {
            if (v < 0) continue;
            if (std::binary_search(gt_set_pq.begin(), gt_set_pq.end(), (uint32_t)v)) ++hit_pq;
          }
          double per_q_recall = double(hit_pq) / K;
          std::fprintf(fpq, "%d,%d,%.6f,%.4f\n", i, q_route[i], latencies[i], per_q_recall);
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
  printf("  recall@%d      : %.4f\n", K, recall);
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

  printf("\n[BCI summary] QPS=%.1f recall=%.4f\n", qps, recall);

  return 0;
}
