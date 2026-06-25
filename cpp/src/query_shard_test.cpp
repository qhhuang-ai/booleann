// query_shard_test: validate one HAMCG Vamana shard end-to-end.
// 1. Loads a built shard (graph + base point indices for that tag)
// 2. Loads queries that have THAT tag in their single-equality filter
// 3. Computes brute-force GT over the tag subset
// 4. Runs beam_search on the shard graph
// 5. Reports recall@10 vs brute-force GT + per-query latency + QPS
//
// Usage: ./query_shard_test <tag_id> [beam] [k] [num_q_cap]
// Defaults: beam=64, k=10, num_q_cap=10000

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
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

using PointT = Euclidian_Point<uint8_t>;
using PR     = PointRange<uint8_t, PointT>;
using SubPR  = SubsetPointRange<uint8_t, PointT>;
using Indx   = int32_t;
using GraphI = Graph<Indx>;

static parlay::sequence<int32_t> load_subset_idx(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); int32_t n;
  if (fread(&n, sizeof(int32_t), 1, f) != 1) { fclose(f); return {}; }
  parlay::sequence<int32_t> v(n);
  size_t got = fread(v.data(), sizeof(int32_t), n, f); (void)got;
  fclose(f); return v;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <tag_id> [beam=64] [k=10] [cap=10000]\n", argv[0]);
    return 1;
  }
  int  tag       = atoi(argv[1]);
  int  beam_size = argc > 2 ? atoi(argv[2]) : 64;
  int  k         = argc > 3 ? atoi(argv[3]) : 10;
  int  cap_q     = argc > 4 ? atoi(argv[4]) : 10000;

  const std::string DATA = "./data/yfcc100m/";
  const std::string DRT  = "./data";

  printf("=== query_shard_test ===\n");
  printf("tag=%d beam=%d k=%d cap_q=%d\n", tag, beam_size, k, cap_q);

  // -- Load base + query metadata + query points + shard ----------------------
  PR base((DATA + "base.10M.u8bin").c_str());
  printf("[base] %ld pts dim=%ld\n", base.size(), base.dimension());

  PR query((DATA + "query.public.100K.u8bin").c_str());
  printf("[query] %ld pts dim=%ld\n", query.size(), query.dimension());

  csr_filters qm(DATA + "query.metadata.public.100K.spmat");
  qm.print_stats();

  // pick queries: filter cardinality == 1 AND that filter == tag
  std::vector<int64_t> q_ids;
  for (int64_t q = 0; q < qm.n_points; ++q) {
    int64_t s = qm.row_offsets[q], e = qm.row_offsets[q+1];
    if (e - s == 1 && qm.row_indices[s] == tag) q_ids.push_back(q);
  }
  printf("[matching queries] %zu single-equality queries with tag=%d\n",
         q_ids.size(), tag);
  if (q_ids.empty()) {
    printf("[skip] no matching queries for tag=%d. exit.\n", tag);
    return 0;
  }
  if ((int)q_ids.size() > cap_q) q_ids.resize(cap_q);

  auto subset_idx = load_subset_idx(DRT + "/subset_idx/subset_idx_" + std::to_string(tag) + ".bin");
  printf("[shard subset] %zu base indices\n", subset_idx.size());

  SubPR subset_pr(base, subset_idx);
  std::string gp = DRT + "/shards/vamana_tag_" + std::to_string(tag) + ".bin";
  GraphI G((char*)gp.c_str());
  printf("[graph loaded] n=%ld maxDeg=%ld from %s\n", G.size(), G.max_degree(), gp.c_str());

  // -- Brute-force GT over the tag subset (exact NN within subset) ------------
  printf("[bruteforce GT] computing top-%d for %zu queries x %zu candidates ...\n",
         k, q_ids.size(), subset_idx.size());
  auto t_gt0 = std::chrono::steady_clock::now();
  // GT[i] = vector<(dist, subset_idx_local)>
  std::vector<std::vector<std::pair<float,int32_t>>> GT(q_ids.size());
  parlay::parallel_for(0, q_ids.size(), [&](size_t i){
    PointT q = query[q_ids[i]];
    std::vector<std::pair<float,int32_t>> dists(subset_idx.size());
    for (size_t j = 0; j < subset_idx.size(); ++j) {
      PointT bp = subset_pr[(long)j];
      float d = q.distance(bp);
      dists[j] = {d, (int32_t)j};
    }
    std::partial_sort(dists.begin(), dists.begin()+k, dists.end(),
      [](const auto& a, const auto& b){ return a.first < b.first; });
    dists.resize(k);
    GT[i] = std::move(dists);
  });
  auto t_gt1 = std::chrono::steady_clock::now();
  double gt_s = std::chrono::duration<double>(t_gt1-t_gt0).count();
  printf("[bruteforce GT done] %.2fs\n", gt_s);

  // -- beam_search on shard ----------------------------------------------------
  QueryParams QP((long)k, (long)beam_size, /*cut=*/1.35, /*limit=*/G.size(), (long)G.max_degree());
  Indx start_point = 0; // shard start; we don't track build's start, but 0 is a valid node.

  printf("[beam_search] starting on shard, beam=%d ...\n", beam_size);
  std::vector<std::vector<Indx>> results(q_ids.size());
  std::atomic<int64_t> total_visited(0);
  auto t_q0 = std::chrono::steady_clock::now();
  parlay::parallel_for(0, q_ids.size(), [&](size_t i){
    PointT q = query[q_ids[i]];
    auto out = beam_search<PointT, SubPR, Indx>(q, G, subset_pr, start_point, QP).first.second;
    auto frontier = beam_search<PointT, SubPR, Indx>(q, G, subset_pr, start_point, QP).first.first;
    (void)out;
    // Use the visited list (frontier) for top-k; it's sorted by dist ascending.
    // Take first k entries.
    std::vector<Indx> top;
    for (size_t j = 0; j < frontier.size() && (int)top.size() < k; ++j) {
      top.push_back(frontier[j].first);
    }
    while ((int)top.size() < k) top.push_back(-1);
    results[i] = std::move(top);
  });
  auto t_q1 = std::chrono::steady_clock::now();
  double q_s = std::chrono::duration<double>(t_q1-t_q0).count();
  double qps = q_ids.size() / q_s;

  // -- Recall ------------------------------------------------------------------
  double total_recall = 0.0;
  for (size_t i = 0; i < q_ids.size(); ++i) {
    std::vector<int32_t> gt_set;
    for (auto& pr : GT[i]) gt_set.push_back(pr.second);
    std::sort(gt_set.begin(), gt_set.end());
    int hit = 0;
    for (auto idx : results[i]) {
      if (idx >= 0 && std::binary_search(gt_set.begin(), gt_set.end(), idx)) ++hit;
    }
    total_recall += double(hit) / k;
  }
  double avg_recall = total_recall / q_ids.size();

  printf("\n[RESULTS]\n");
  printf("  num queries     : %zu\n", q_ids.size());
  printf("  shard size      : %zu\n", subset_idx.size());
  printf("  beam            : %d\n", beam_size);
  printf("  beam_search time: %.3fs  (%.1f QPS, %.1f ms/q avg)\n",
         q_s, qps, 1000.0 * q_s / q_ids.size());
  printf("  recall@%d       : %.4f\n", k, avg_recall);
  printf("  GT brute time   : %.2fs (sanity check)\n", gt_s);
  return 0;
}
