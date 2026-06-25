// YFCC10M probe: load base + base_metadata + query_metadata via parlaylib
// infrastructure, compute per-tag frequency histogram for BCI win-regime
// calibration. HAMCG win band tag-freq
// 0.5-5% of N (= 50K-500K). Print bucket counts in [0.1%, 0.5%, 1%, 5%, 10%].

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "utils/euclidian_point.h"
#include "utils/filters.h"
#include "utils/point_range.h"

int main(int argc, char** argv) {
  const std::string DATA = "./data/yfcc100m/";
  const std::string base_meta_path  = DATA + "base.metadata.10M.spmat";
  const std::string query_meta_path = DATA + "query.metadata.public.100K.spmat";
  const std::string base_pts_path   = DATA + "base.10M.u8bin";

  printf("=== YFCC10M probe ===\n");
  printf("parlay workers = %ld\n", parlay::num_workers());

  // -- base metadata ---------------------------------------------------------
  csr_filters base_meta(base_meta_path);
  printf("\n[base metadata]\n");
  base_meta.print_stats();
  const int64_t N = base_meta.n_points;
  const int64_t F = base_meta.n_filters;

  // per-filter (transposed) frequency
  csr_filters base_meta_t = base_meta.transpose();
  std::vector<int64_t> freq(F);
  for (int64_t f = 0; f < F; ++f) {
    freq[f] = base_meta_t.row_offsets[f+1] - base_meta_t.row_offsets[f];
  }
  std::sort(freq.begin(), freq.end(), std::greater<int64_t>());

  printf("\n[per-tag frequency distribution, sorted desc]\n");
  printf("  top-1   %12ld  (%.3f%% of N)\n", freq[0],         100.0*freq[0]/N);
  printf("  top-10  %12ld  (%.3f%% of N)\n", freq[9],         100.0*freq[9]/N);
  printf("  top-100 %12ld  (%.3f%% of N)\n", freq[99],        100.0*freq[99]/N);
  if (F > 1000) printf("  top-1k  %12ld  (%.3f%% of N)\n", freq[999], 100.0*freq[999]/N);

  // BCI win-regime buckets (HAMCG sweet spot 0.5-5% of N)
  int64_t b_below_0p1 = 0, b_0p1_0p5 = 0, b_0p5_1 = 0, b_1_5 = 0, b_5_10 = 0, b_above_10 = 0;
  for (int64_t f = 0; f < F; ++f) {
    double pct = 100.0 * freq[f] / N;
    if (pct < 0.1)        ++b_below_0p1;
    else if (pct < 0.5)   ++b_0p1_0p5;
    else if (pct < 1.0)   ++b_0p5_1;
    else if (pct < 5.0)   ++b_1_5;
    else if (pct < 10.0)  ++b_5_10;
    else                  ++b_above_10;
  }
  printf("\n[BCI HAMCG win-regime buckets (per-tag frequency as %% of N)]\n");
  printf("  <0.1%%       : %8ld tags\n", b_below_0p1);
  printf("  0.1%%-0.5%%   : %8ld tags  <-- bottom of useful HAMCG band\n", b_0p1_0p5);
  printf("  0.5%%-1%%     : %8ld tags  <-- SWEET SPOT\n", b_0p5_1);
  printf("  1%%-5%%       : %8ld tags  <-- SWEET SPOT\n", b_1_5);
  printf("  5%%-10%%      : %8ld tags  <-- HAMCG edge\n", b_5_10);
  printf("  >=10%%       : %8ld tags  <-- too broad\n", b_above_10);
  printf("  total       : %8ld tags\n", F);

  // total budget if every sweet-spot tag got an HNSW shard
  int64_t shard_candidates = b_0p1_0p5 + b_0p5_1 + b_1_5 + b_5_10;
  int64_t shard_points = 0;
  for (int64_t f = 0; f < F; ++f) {
    double pct = 100.0 * freq[f] / N;
    if (pct >= 0.1 && pct < 10.0) shard_points += freq[f];
  }
  printf("\n[shard budget estimate]\n");
  printf("  candidate tags (0.1%%-10%%)        : %ld\n", shard_candidates);
  printf("  union of base points in those tags : %ld (%.1fx N, due to multi-tag points)\n",
         shard_points, double(shard_points)/N);

  // -- query metadata --------------------------------------------------------
  csr_filters query_meta(query_meta_path);
  printf("\n[query metadata]\n");
  query_meta.print_stats();
  const int64_t Q = query_meta.n_points;
  std::vector<int64_t> q_nfilt(Q);
  for (int64_t q = 0; q < Q; ++q) {
    q_nfilt[q] = query_meta.row_offsets[q+1] - query_meta.row_offsets[q];
  }
  int64_t q_zero = 0, q_one = 0, q_two = 0, q_many = 0;
  for (int64_t q = 0; q < Q; ++q) {
    if      (q_nfilt[q] == 0) ++q_zero;
    else if (q_nfilt[q] == 1) ++q_one;
    else if (q_nfilt[q] == 2) ++q_two;
    else                      ++q_many;
  }
  printf("[query filter cardinality]\n");
  printf("  0 filters (unfiltered) : %ld\n", q_zero);
  printf("  1 filter  (equality)   : %ld\n", q_one);
  printf("  2 filters (conjunction): %ld\n", q_two);
  printf("  3+ filters             : %ld\n", q_many);

  // -- base point dimension --------------------------------------------------
  printf("\n[base points]\n");
  PointRange<uint8_t, Euclidian_Point<uint8_t>> base_pts(base_pts_path.c_str());
  printf("  n     = %ld\n", base_pts.size());
  printf("  d     = %ld\n", base_pts.dimension());
  printf("  d_aln = %ld\n", base_pts.aligned_dimension());

  printf("\n[probe OK]\n");
  return 0;
}
