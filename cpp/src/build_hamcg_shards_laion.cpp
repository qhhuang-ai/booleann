// build_hamcg_shards: build per-tag Vamana shards for YFCC10M HAMCG.
// Inputs (produced by extract_sweet_spot_tags):
//   data/sweet_spot_tags.bin  = int32 n + (int32 tag, int64 freq) * n
//   data/subset_idx/subset_idx_<tag>.bin = int32 n + int32 base_idx * n
// Output:
//   data/shards/vamana_tag_<tag>.bin (via Graph::save)
//   data/shards/build_log.csv = tag,n_points,build_seconds,R,L,alpha
//
// Strategy: sequential outer loop over tags (largest first, so high-value
// hot tags are built first and we can stop early if needed); each Vamana
// build internally parallelizes across all 96 parlay workers. Adaptive R
// adaptive based on shard size (R = 8, 10, or 12).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"

#include "vamana/index.h"

using PointT = Euclidian_Point<float>;
using PR     = PointRange<float, PointT>;
using SubPR  = SubsetPointRange<float, PointT>;
using Indx   = int32_t;
using GraphI = Graph<Indx>;
using KNNIdx = knn_index<PointT, SubPR, Indx>;

static parlay::sequence<int32_t> load_subset_idx(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "open fail: %s\n", path.c_str()); exit(1); }
  int32_t n;
  size_t got = fread(&n, sizeof(int32_t), 1, f);
  (void)got;
  parlay::sequence<int32_t> idx(n);
  got = fread(idx.data(), sizeof(int32_t), n, f);
  (void)got;
  fclose(f);
  return idx;
}

struct TagEntry { int32_t tag; int64_t freq; };

static std::vector<TagEntry> load_sweet_spot_tags(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "open fail: %s\n", path.c_str()); exit(1); }
  int32_t n;
  size_t got = fread(&n, sizeof(int32_t), 1, f);
  (void)got;
  std::vector<TagEntry> out(n);
  for (int32_t i = 0; i < n; ++i) {
    got = fread(&out[i].tag,  sizeof(int32_t), 1, f);
    got = fread(&out[i].freq, sizeof(int64_t), 1, f);
  }
  fclose(f);
  return out;
}

// Adaptive R schedule by shard size
// Conservative R schedule (R in {8,10,12}) caps recall at ~0.905;
// User mandate: comprehensive surpass requires recall >= 0.94. Boost R per-class.
// Higher R = denser graph = higher recall ceiling at cost of build time + RAM.
static int pick_R(int64_t shard_n) {
  if (shard_n >= 400000) return 32; // large class (was 12)
  if (shard_n >= 100000) return 24; // medium (was 10)
  return 16;                         // small (was 8)
}

int main(int argc, char** argv) {
  const std::string DATA       = "./data/laion1m/";
  const std::string base_path  = DATA + "base.1M.f32bin";
  const std::string data_root  = argc > 1 ? argv[1] :
    "./data_laion";
  const std::string shards_dir = data_root + "/shards";
  std::filesystem::create_directories(shards_dir);

  // Caller can pass max-shard cap (0 = all) and freq filter band as fraction of N.
  int    cap     = argc > 2 ? atoi(argv[2]) : 0;
  double min_pct = argc > 3 ? atof(argv[3]) : 0.10;   // default 0.10% of N
  double max_pct = argc > 4 ? atof(argv[4]) : 10.0;   // default 10.0% of N

  printf("=== HAMCG multi-shard Vamana builder ===\n");
  printf("data root  = %s\n", data_root.c_str());
  printf("shards out = %s\n", shards_dir.c_str());
  printf("cap        = %d (0 = build all in-range)\n", cap);
  printf("band       = [%.3f%%, %.3f%%] of N\n", min_pct, max_pct);

  auto tags = load_sweet_spot_tags(data_root + "/sweet_spot_tags.bin");
  printf("[loaded] %zu candidate tags\n", tags.size());

  // load base points once
  printf("[loading] base.10M.u8bin ...\n");
  auto t0 = std::chrono::steady_clock::now();
  PR base(base_path.c_str());
  auto load_secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  printf("[loaded] %ld points dim=%ld (%.1fs, RSS will be ~%.1fGB)\n",
         base.size(), base.dimension(), load_secs, base.size()*base.aligned_dimension()/1e9);

  // log
  std::string log_path = shards_dir + "/build_log.csv";
  std::ofstream log(log_path, std::ios::app);
  if (log.tellp() == 0) log << "tag,n_points,R,L,alpha,build_seconds,saved_bytes\n";

  // Pre-existing shards check (resumable)
  auto already_built = [&](int32_t tag){
    char p[1024];
    snprintf(p, sizeof(p), "%s/vamana_tag_%d.bin", shards_dir.c_str(), tag);
    return std::filesystem::exists(p);
  };

  // Filter + sort tags
  const int64_t N = base.size();
  std::vector<TagEntry> work;
  for (auto& te : tags) {
    double pct = 100.0 * te.freq / N;
    if (pct < min_pct || pct >= max_pct) continue;
    if (already_built(te.tag)) continue;
    work.push_back(te);
  }
  std::sort(work.begin(), work.end(), [](const TagEntry& a, const TagEntry& b){
    return a.freq > b.freq;
  });
  if (cap > 0 && (int)work.size() > cap) work.resize(cap);
  printf("[work] %zu shards to build (largest first)\n", work.size());

  double cum = 0.0;
  for (size_t i = 0; i < work.size(); ++i) {
    auto te = work[i];
    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/subset_idx/subset_idx_%d.bin",
             data_root.c_str(), te.tag);
    auto subset = load_subset_idx(idx_path);
    int R = pick_R(subset.size());
    int L = 200;
    double alpha = 1.175;

    BuildParams BP(R, L, alpha);

    auto ts = std::chrono::steady_clock::now();
    SubPR subset_pr(base, subset);
    GraphI G((long)R, subset.size());
    stats<Indx> S(G.size());
    KNNIdx idx(BP);
    idx.build_index(G, subset_pr, S);

    char out_path[1024];
    snprintf(out_path, sizeof(out_path), "%s/vamana_tag_%d.bin",
             shards_dir.c_str(), te.tag);
    G.save(out_path);

    auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-ts).count();
    cum += secs;
    auto sz = std::filesystem::file_size(out_path);
    log << te.tag << "," << subset.size() << "," << R << "," << L << ","
        << alpha << "," << secs << "," << sz << "\n";
    log.flush();
    printf("[%zu/%zu] tag=%d n=%d R=%d build=%.2fs (cum=%.1fs, ETA=%.1fs)\n",
           i+1, work.size(), te.tag, (int)subset.size(), R, secs, cum,
           cum * (work.size()-i-1) / std::max((size_t)1, i+1));
  }
  log.close();

  printf("[done] %zu shards built in %.1fs total\n", work.size(), cum);
  return 0;
}
