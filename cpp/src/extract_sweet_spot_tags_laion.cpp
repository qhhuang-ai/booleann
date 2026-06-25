// Extract YFCC10M HAMCG sweet-spot tag IDs (per-tag freq in 0.5%-10% band
// and persist:
//   sweet_spot_tags.bin = int32 n_tags + (int32 tag_id, int64 freq)*n_tags
//   subset_idx_<tag>.bin = int32 n_points + (int32 base_idx)*n_points
//
// Output dir: <out_root>/{sweet_spot_tags.bin, subset_idx_<tag>.bin}
// These files become the input to build_hamcg_shards which builds one
// Vamana per tag in parallel.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "utils/filters.h"

int main(int argc, char** argv) {
  const std::string DATA = "./data/laion1m/";
  const std::string base_meta_path = DATA + "base.metadata.laion1m.richer.spmat";
  const std::string out_root = argc > 1 ? argv[1] :
    "./data_laion";

  std::filesystem::create_directories(out_root);

  printf("=== Sweet-spot tag extractor ===\n");
  printf("base meta: %s\n", base_meta_path.c_str());
  printf("out root : %s\n", out_root.c_str());

  csr_filters bm(base_meta_path);
  bm.print_stats();
  const int64_t N = bm.n_points;
  const int64_t F = bm.n_filters;

  // Transpose to per-filter postings.
  csr_filters bmt = bm.transpose();

  // Per-tag frequency.
  std::vector<std::pair<int64_t,int32_t>> tag_freq(F); // (freq, tag_id)
  for (int64_t f = 0; f < F; ++f) {
    int64_t freq = bmt.row_offsets[f+1] - bmt.row_offsets[f];
    tag_freq[f] = {freq, (int32_t)f};
  }

  // Sweet-spot band 0.1%-10%: this gives the
  // build_hamcg_shards binary can sub-select on tighter sweet spot
  // (0.5%-5%) but still has the edges (0.1%-0.5%, 5%-10%) cached.
  // 2026-06-23 bilateral fix: include huge tags (>10%) too — they're queried
  // and brute on 1-2M points was 75% of bench wall time.
  const double LOW_PCT  = argc > 2 ? atof(argv[2]) : 0.1;
  const double HIGH_PCT = argc > 3 ? atof(argv[3]) : 50.0;  // was 10.0; now include 9 huge tags
  const int64_t LOW_F  = (int64_t)(LOW_PCT  * N / 100.0);
  const int64_t HIGH_F = (int64_t)(HIGH_PCT * N / 100.0);
  printf("[bands] in-range freq = [%ld, %ld] points (%.1f%% - %.1f%% of N=%ld)\n",
         LOW_F, HIGH_F, LOW_PCT, HIGH_PCT, N);

  std::vector<std::pair<int64_t,int32_t>> in_range;
  for (auto& tf : tag_freq) {
    if (tf.first >= LOW_F && tf.first < HIGH_F) in_range.push_back(tf);
  }
  std::sort(in_range.begin(), in_range.end(), std::greater<>());
  printf("[in-range tags] %zu (will build Vamana shard per tag)\n", in_range.size());
  if (in_range.size() > 0) {
    printf("  top: tag=%d freq=%ld (%.3f%%)\n", in_range.front().second, in_range.front().first,
           100.0 * in_range.front().first / N);
    printf("  bot: tag=%d freq=%ld (%.3f%%)\n", in_range.back().second, in_range.back().first,
           100.0 * in_range.back().first / N);
  }

  // Persist sweet_spot_tags.bin = [n_tags:int32] [(tag_id:int32, freq:int64)] * n
  {
    std::ofstream ofs(out_root + "/sweet_spot_tags.bin", std::ios::binary);
    int32_t n_tags = (int32_t)in_range.size();
    ofs.write((const char*)&n_tags, sizeof(int32_t));
    for (auto& tf : in_range) {
      int32_t tid = tf.second;
      int64_t freq = tf.first;
      ofs.write((const char*)&tid,  sizeof(int32_t));
      ofs.write((const char*)&freq, sizeof(int64_t));
    }
    ofs.close();
    printf("[wrote] %s/sweet_spot_tags.bin (%d tags)\n", out_root.c_str(), n_tags);
  }

  // For each in-range tag, persist its base-index subset.
  // Format: subset_idx_<tag>.bin = [n_points:int32][(base_idx:int32)] * n
  std::filesystem::create_directories(out_root + "/subset_idx");
  size_t total_indices = 0;
  parlay::parallel_for(0, in_range.size(), [&](size_t i) {
    int32_t tid = in_range[i].second;
    int64_t start = bmt.row_offsets[tid];
    int64_t end   = bmt.row_offsets[tid+1];
    int32_t n = (int32_t)(end - start);

    char path[1024];
    snprintf(path, sizeof(path), "%s/subset_idx/subset_idx_%d.bin",
             out_root.c_str(), tid);
    FILE* f = fopen(path, "wb");
    fwrite(&n, sizeof(int32_t), 1, f);
    fwrite(bmt.row_indices.get() + start, sizeof(int32_t), n, f);
    fclose(f);
  });

  // Report.
  for (auto& tf : in_range) total_indices += (size_t)tf.first;
  printf("[wrote] %s/subset_idx/subset_idx_<tag>.bin x %zu files\n",
         out_root.c_str(), in_range.size());
  printf("[stats] total indices across shards = %zu (%.1fx N due to multi-tag points)\n",
         total_indices, double(total_indices)/N);

  printf("\n[done]\n");
  return 0;
}
