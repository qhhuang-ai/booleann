// build_shard_clusters: per-shard k-means clustering for IVF² absorb.
// Builds a k-means clustering of each tag's subset:
//   - centroid_data: contiguous T[] centroids (n_clusters × aligned_dim)
//   - cluster_members: per-cluster point IDs (CSR-like layout)
// Output: data/clusters/{tag}_centroids.bin + {tag}_members.bin
//
// Reads existing subset_idx/{tag}.bin files (1326 tags).
// Cluster_size = 5000 per PIVF defaults.
// Build time: ~1-2 hr for 1326 shards (k-means is iterative).
//
// Replace beam_search with centroid scan + sorted_near
// for warm-cache QPS.

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
#include "parlay/sequence.h"

#include "utils/euclidian_point.h"
#include "utils/point_range.h"

#include "IVF/clustering.h"

using PointT = Euclidian_Point<float>;
using PR     = PointRange<float, PointT>;
using Indx   = int32_t;
using KMeans = KMeansClusterer<float, PointT, Indx>;

static parlay::sequence<int32_t> load_subset_idx(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); int32_t n;
  if (fread(&n, sizeof(int32_t), 1, f) != 1) { fclose(f); return {}; }
  parlay::sequence<int32_t> v(n);
  size_t got = fread(v.data(), sizeof(int32_t), n, f); (void)got;
  fclose(f); return v;
}

struct TagEntry { int32_t tag; int64_t freq; };

static std::vector<TagEntry> load_sweet_spot_tags(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb"); int32_t n;
  size_t got = fread(&n, sizeof(int32_t), 1, f); (void)got;
  std::vector<TagEntry> out(n);
  for (int32_t i = 0; i < n; ++i) {
    got = fread(&out[i].tag,  sizeof(int32_t), 1, f);
    got = fread(&out[i].freq, sizeof(int64_t), 1, f);
  }
  fclose(f); return out;
}

int main(int argc, char** argv) {
  const std::string DATA    = "./data/laion1m/";
  const std::string data_root = argc > 1 ? argv[1] :
    "./data_laion";
  const std::string clust_dir = data_root + "/clusters";
  std::filesystem::create_directories(clust_dir);

  // Args: min_pct max_pct cluster_size cap
  double min_pct = argc > 2 ? atof(argv[2]) : 0.5;   // skip tiny tags; cluster only ≥0.5%
  double max_pct = argc > 3 ? atof(argv[3]) : 50.0;  // include all
  int cluster_size = argc > 4 ? atoi(argv[4]) : 5000; // PIVF default
  int cap = argc > 5 ? atoi(argv[5]) : 0;             // 0 = no cap

  printf("=== build_shard_clusters (IVF² absorb iter-1) ===\n");
  printf("data root  = %s\n", data_root.c_str());
  printf("clusters   = %s\n", clust_dir.c_str());
  printf("band       = [%.2f%%, %.2f%%] of N\n", min_pct, max_pct);
  printf("cluster_sz = %d (n_clusters = subset_size / cluster_size)\n", cluster_size);

  // Load base + sweet-spot tags
  printf("[loading] base.10M.u8bin ...\n");
  auto t0 = std::chrono::steady_clock::now();
  PR base((DATA + "base.1M.f32bin").c_str());
  printf("[loaded] %ld pts dim=%ld (%.1fs)\n", base.size(), base.dimension(),
         std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());

  auto tags = load_sweet_spot_tags(data_root + "/sweet_spot_tags.bin");
  const int64_t N = base.size();
  const int64_t LOW_F  = (int64_t)(min_pct * N / 100.0);
  const int64_t HIGH_F = (int64_t)(max_pct * N / 100.0);

  std::vector<TagEntry> work;
  for (auto& te : tags) {
    if (te.freq < LOW_F || te.freq >= HIGH_F) continue;
    char p[1024];
    snprintf(p, sizeof(p), "%s/%d_centroids.bin", clust_dir.c_str(), te.tag);
    if (std::filesystem::exists(p)) continue;  // resumable
    work.push_back(te);
  }
  std::sort(work.begin(), work.end(), [](const TagEntry& a, const TagEntry& b){
    return a.freq > b.freq;
  });
  if (cap > 0 && (int)work.size() > cap) work.resize(cap);
  printf("[work] %zu tags to cluster (largest first)\n", work.size());

  std::ofstream log(clust_dir + "/cluster_log.csv", std::ios::app);
  if (log.tellp() == 0) log << "tag,n_points,n_clusters,build_seconds,centroid_bytes,members_bytes\n";

  double cum = 0.0;
  for (size_t i = 0; i < work.size(); ++i) {
    auto te = work[i];
    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/subset_idx/subset_idx_%d.bin", data_root.c_str(), te.tag);
    auto subset = load_subset_idx(idx_path);

    int n_clust = std::max<int>(1, (int)(subset.size() / cluster_size));

    auto ts = std::chrono::steady_clock::now();
    KMeans clusterer((size_t)n_clust);
    auto clusters = clusterer.cluster(base, subset);

    // Compute centroids (uint8 means, like PIVF does)
    size_t dim = base.dimension();
    size_t aligned_dim = base.aligned_dimension();
    std::vector<float> centroid_data(n_clust * aligned_dim, 0);
    for (int c = 0; c < n_clust; ++c) {
      if (clusters[c].size() == 0) continue;
      std::vector<double> centroid(dim, 0.0);
      for (size_t j = 0; j < clusters[c].size(); ++j) {
        float* data = base[clusters[c][j]].get();
        for (size_t d = 0; d < dim; ++d) centroid[d] += data[d];
      }
      size_t offset = c * aligned_dim;
      for (size_t d = 0; d < dim; ++d) {
        centroid_data[offset + d] = (float)(centroid[d] / clusters[c].size());
      }
    }

    // Persist centroids
    char cp[1024], mp[1024];
    snprintf(cp, sizeof(cp), "%s/%d_centroids.bin", clust_dir.c_str(), te.tag);
    snprintf(mp, sizeof(mp), "%s/%d_members.bin",   clust_dir.c_str(), te.tag);
    {
      FILE* f = fopen(cp, "wb");
      int32_t nc = n_clust;
      fwrite(&nc, sizeof(int32_t), 1, f);
      int32_t adim = (int32_t)aligned_dim;
      fwrite(&adim, sizeof(int32_t), 1, f);
      fwrite(centroid_data.data(), 1, centroid_data.size(), f);
      fclose(f);
    }
    // Persist cluster members as CSR: nc, offsets (nc+1), then int32 ids
    {
      FILE* f = fopen(mp, "wb");
      int32_t nc = n_clust;
      fwrite(&nc, sizeof(int32_t), 1, f);
      std::vector<int64_t> offsets(nc + 1, 0);
      for (int c = 0; c < nc; ++c) offsets[c+1] = offsets[c] + (int64_t)clusters[c].size();
      fwrite(offsets.data(), sizeof(int64_t), nc + 1, f);
      // flatten members
      for (int c = 0; c < nc; ++c) {
        for (auto id : clusters[c]) {
          int32_t gid = (int32_t)id;
          fwrite(&gid, sizeof(int32_t), 1, f);
        }
      }
      fclose(f);
    }

    auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-ts).count();
    cum += secs;
    auto csz = std::filesystem::file_size(cp);
    auto msz = std::filesystem::file_size(mp);
    log << te.tag << "," << subset.size() << "," << n_clust << "," << secs << "," << csz << "," << msz << "\n";
    log.flush();
    printf("[%zu/%zu] tag=%d n=%d nc=%d build=%.2fs (cum=%.1fs, ETA=%.0fs)\n",
           i+1, work.size(), te.tag, (int)subset.size(), n_clust, secs, cum,
           cum * (work.size()-i-1) / std::max((size_t)1, i+1));
  }

  printf("[done] %zu shards clustered in %.1fs total\n", work.size(), cum);
  return 0;
}
