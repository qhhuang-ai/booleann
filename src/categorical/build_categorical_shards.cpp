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
// based on shard size to mirror ParlayIVF's R={8,10,12} weight classes.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"

#include "vamana/index.h"

using PointT = Euclidian_Point<uint8_t>;
using PR     = PointRange<uint8_t, PointT>;
using SubPR  = SubsetPointRange<uint8_t, PointT>;
using Indx   = int32_t;
using GraphI = Graph<Indx>;
using KNNIdx = knn_index<PointT, SubPR, Indx>;

// Builder-owned canonical two-pass adapter.
//
// The bundled single-pass Vamana fast path appends reverse edges whenever
//
//   current_degree + incoming_count <= R.
//
// Calling build_index twice on the same graph makes an incoming reverse edge
// likely to already be present.  append_neighbors() does not deduplicate, so
// the second pass can consume degree slots with repeated local IDs.  A
// post-build unique() is too late: the duplicates have already affected later
// second-pass beam searches.
//
// Keep the bundled robust-prune and beam-search semantics, but replace the
// reverse-edge update with a stable set union at every batch in both passes:
//
//   C(v) = unique_valid_nonself(old_out(v) ++ incoming(v)).
//
// Existing edges come first, then genuinely new incoming edges.  If |C(v)| is
// at most R, install it directly.  Otherwise robust-prune C(v) with add=false;
// this exposes exactly one copy of every old/new candidate to pruning.  The
// graph is therefore canonical during construction, not merely repaired after
// construction, while the alpha=1 first pass remains visible to every search
// and prune in the alpha=configured second pass.
struct CanonicalUnion {
  parlay::sequence<Indx> ids;
  uint64_t duplicate_ids = 0;
  uint64_t self_ids = 0;
  uint64_t out_of_range_ids = 0;
};

template <typename OldRange, typename IncomingRange>
static CanonicalUnion canonical_neighbor_union(
    Indx self, size_t graph_size, OldRange old_neighbors,
    const IncomingRange& incoming_neighbors) {
  CanonicalUnion out;
  const size_t expected =
      static_cast<size_t>(old_neighbors.size()) + incoming_neighbors.size();
  out.ids.reserve(expected);

  std::unordered_set<Indx> seen;
  seen.reserve(expected * 2 + 1);

  auto admit = [&](Indx candidate) {
    if (candidate < 0 || static_cast<size_t>(candidate) >= graph_size) {
      ++out.out_of_range_ids;
      return;
    }
    if (candidate == self) {
      ++out.self_ids;
      return;
    }
    if (!seen.insert(candidate).second) {
      ++out.duplicate_ids;
      return;
    }
    out.ids.push_back(candidate);
  };

  for (size_t j = 0; j < static_cast<size_t>(old_neighbors.size()); ++j) {
    admit(old_neighbors[j]);
  }
  for (size_t j = 0; j < incoming_neighbors.size(); ++j) {
    admit(incoming_neighbors[j]);
  }
  return out;
}

struct EmptyNeighborRange {
  size_t size() const { return 0; }
  Indx operator[](size_t) const { return -1; }
};

struct CanonicalPassCounters {
  std::atomic<uint64_t> duplicate_ids{0};
  std::atomic<uint64_t> self_ids{0};
  std::atomic<uint64_t> out_of_range_ids{0};

  void add(const CanonicalUnion& x) {
    if (x.duplicate_ids) {
      duplicate_ids.fetch_add(x.duplicate_ids, std::memory_order_relaxed);
    }
    if (x.self_ids) {
      self_ids.fetch_add(x.self_ids, std::memory_order_relaxed);
    }
    if (x.out_of_range_ids) {
      out_of_range_ids.fetch_add(
          x.out_of_range_ids, std::memory_order_relaxed);
    }
  }
};

struct GraphInvariantAudit {
  uint64_t vertices = 0;
  uint64_t edge_slots = 0;
  uint64_t duplicate_slots = 0;
  uint64_t self_edges = 0;
  uint64_t out_of_range_edges = 0;
  uint64_t degree_overflow_vertices = 0;
  size_t max_actual_degree = 0;
  long container_max_degree = 0;
};

static GraphInvariantAudit audit_graph_invariants(GraphI& graph,
                                                  long expected_R) {
  if (graph.max_degree() != expected_R) {
    throw std::runtime_error("graph container max degree is not configured R");
  }
  if (graph.max_degree() <= 0 || graph.max_degree() > 1024) {
    throw std::runtime_error("graph audit supports max degree in [1,1024]");
  }

  GraphInvariantAudit audit;
  audit.vertices = graph.size();
  audit.container_max_degree = graph.max_degree();
  std::array<Indx, 1024> sorted_ids{};
  for (size_t i = 0; i < graph.size(); ++i) {
    auto ngh = graph[static_cast<Indx>(i)];
    const size_t degree = ngh.size();
    audit.edge_slots += degree;
    audit.max_actual_degree = std::max(audit.max_actual_degree, degree);
    if (degree > static_cast<size_t>(expected_R)) {
      ++audit.degree_overflow_vertices;
    }
    for (size_t j = 0; j < degree; ++j) {
      const Indx id = ngh[static_cast<Indx>(j)];
      sorted_ids[j] = id;
      if (id == static_cast<Indx>(i)) ++audit.self_edges;
      if (id < 0 || static_cast<size_t>(id) >= graph.size()) {
        ++audit.out_of_range_edges;
      }
    }
    std::sort(sorted_ids.begin(), sorted_ids.begin() + degree);
    for (size_t j = 1; j < degree; ++j) {
      if (sorted_ids[j] == sorted_ids[j - 1]) ++audit.duplicate_slots;
    }
  }
  return audit;
}

static void require_canonical_graph(const GraphInvariantAudit& audit,
                                    const char* phase) {
  if (audit.duplicate_slots != 0 || audit.self_edges != 0 ||
      audit.out_of_range_edges != 0 ||
      audit.degree_overflow_vertices != 0) {
    throw std::runtime_error(
        std::string("non-canonical graph after ") + phase +
        ": duplicate/self/out-of-range/overflow invariant failed");
  }
}

struct CanonicalPassReport {
  uint64_t duplicate_candidates_removed = 0;
  uint64_t self_candidates_removed = 0;
  uint64_t out_of_range_candidates_removed = 0;
  GraphInvariantAudit graph;
};

// Builder-local canonical pass over the public graph, search, and prune APIs.
// Candidate IDs are canonicalized before each forward install and reverse
// update; the upstream builder implementation is not embedded here.
static CanonicalPassReport canonical_vamana_pass(
    GraphI& graph, SubPR& points, stats<Indx>& build_stats,
    int R, int L, double alpha) {
  if (graph.size() == 0 || points.size() != graph.size()) {
    throw std::runtime_error("canonical pass requires one point per vertex");
  }
  if (graph.max_degree() != R || R <= 0 || L < R ||
      !std::isfinite(alpha) || alpha < 1.0) {
    throw std::runtime_error("invalid canonical pass R/L/alpha contract");
  }

  BuildParams params(R, L, alpha);
  KNNIdx index(params);
  auto inserts = parlay::tabulate(
      graph.size(), [&](size_t i) { return static_cast<Indx>(i); });
  index.start_point = inserts[0];

  const size_t n = graph.size();
  const size_t m = inserts.size();
  size_t inc = 0;
  size_t count = 0;
  const double base = 2.0;
  const double max_fraction = 0.02;
  const size_t max_batch_size = std::min(
      static_cast<size_t>(max_fraction * static_cast<float>(n)), 1000000ul);
  const size_t effective_max_batch =
      max_batch_size == 0 ? n : max_batch_size;
  auto rperm = parlay::random_permutation<Indx>(static_cast<Indx>(m));
  auto shuffled_inserts =
      parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });

  CanonicalPassCounters counters;
  size_t next_progress_percent = 10;
  while (count < m) {
    size_t floor;
    size_t ceiling;
    if (std::pow(base, static_cast<double>(inc)) <= effective_max_batch) {
      floor = static_cast<size_t>(std::pow(base, static_cast<double>(inc))) - 1;
      ceiling = std::min(
          static_cast<size_t>(std::pow(base, static_cast<double>(inc + 1))), m);
      ceiling -= 1;
      count = ceiling;
    } else {
      floor = count;
      ceiling = std::min(count + effective_max_batch, m);
      count += effective_max_batch;
    }

    parlay::sequence<parlay::sequence<Indx>> new_out(ceiling - floor);
    parlay::parallel_for(floor, ceiling, [&](size_t i) {
      const Indx vertex = shuffled_inserts[i];
      QueryParams query_params(
          0L, static_cast<long>(L), 0.0, static_cast<long>(points.size()),
          graph.max_degree());
      auto visited =
          beam_search<PointT, SubPR, Indx>(
              points[vertex], graph, points, index.start_point, query_params)
              .first.second;
      build_stats.increment_visited(vertex, visited.size());
      auto pruned = index.robustPrune(vertex, visited, graph, points);
      auto canonical = canonical_neighbor_union(
          vertex, n, EmptyNeighborRange{}, pruned);
      counters.add(canonical);
      new_out[i - floor] = std::move(canonical.ids);
    });
    if (counters.out_of_range_ids.load(std::memory_order_relaxed) != 0 ||
        counters.self_ids.load(std::memory_order_relaxed) != 0) {
      throw std::runtime_error(
          "invalid/self candidate produced during canonical forward phase");
    }

    auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
      const Indx vertex = shuffled_inserts[i + floor];
      return parlay::tabulate(new_out[i].size(), [&](size_t j) {
        return std::make_pair(new_out[i][j], vertex);
      });
    });
    auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));

    parlay::parallel_for(floor, ceiling, [&](size_t i) {
      graph[shuffled_inserts[i]].update_neighbors(new_out[i - floor]);
    });

    parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
      auto& [vertex, incoming] = grouped_by[j];
      auto canonical =
          canonical_neighbor_union(vertex, n, graph[vertex], incoming);
      counters.add(canonical);
      if (canonical.out_of_range_ids || canonical.self_ids) return;

      if (canonical.ids.size() <= static_cast<size_t>(R)) {
        graph[vertex].update_neighbors(canonical.ids);
      } else {
        auto pruned = index.robustPrune(
            vertex, std::move(canonical.ids), graph, points, false);
        auto canonical_pruned = canonical_neighbor_union(
            vertex, n, EmptyNeighborRange{}, pruned);
        counters.add(canonical_pruned);
        if (canonical_pruned.out_of_range_ids || canonical_pruned.self_ids) {
          return;
        }
        graph[vertex].update_neighbors(canonical_pruned.ids);
      }
    });
    if (counters.out_of_range_ids.load(std::memory_order_relaxed) != 0 ||
        counters.self_ids.load(std::memory_order_relaxed) != 0) {
      throw std::runtime_error(
          "invalid/self candidate produced during canonical reverse phase");
    }
    const size_t completed = std::min(count, m);
    const size_t progress_percent =
        m == 0 ? 100 : (100 * completed) / m;
    if (progress_percent >= next_progress_percent || completed == m) {
      printf("[canonical alpha=%.6f] progress=%zu%% vertices=%zu/%zu\n",
             alpha, progress_percent, completed, m);
      fflush(stdout);
      while (next_progress_percent <= progress_percent) {
        next_progress_percent += 10;
      }
    }
    ++inc;
  }

  parlay::parallel_for(0, graph.size(), [&](size_t i) {
    auto less = [&](Indx j, Indx k) {
      return points[i].distance(points[j]) < points[i].distance(points[k]);
    };
    graph[static_cast<Indx>(i)].sort(less);
  });

  CanonicalPassReport report;
  report.duplicate_candidates_removed =
      counters.duplicate_ids.load(std::memory_order_relaxed);
  report.self_candidates_removed =
      counters.self_ids.load(std::memory_order_relaxed);
  report.out_of_range_candidates_removed =
      counters.out_of_range_ids.load(std::memory_order_relaxed);
  report.graph = audit_graph_invariants(graph, R);
  return report;
}

static int canonical_two_pass_selftest() {
  GraphI graph(64, 8);
  parlay::sequence<Indx> old = {1, 2};
  graph[0].update_neighbors(old);
  parlay::sequence<Indx> incoming = {2, 3};

  // Minimal legacy-fast-path counterexample:
  // raw size is 4 <= R, but append(old,{2,3}) creates {1,2,2,3}.
  parlay::sequence<Indx> legacy = {1, 2, 2, 3};
  graph[0].update_neighbors(legacy);
  auto bad = audit_graph_invariants(graph, 64);
  if (bad.duplicate_slots != 1 || bad.self_edges != 0 ||
      bad.out_of_range_edges != 0) {
    throw std::runtime_error("selftest failed to detect legacy duplicate");
  }

  graph[0].update_neighbors(old);
  auto canonical =
      canonical_neighbor_union(0, graph.size(), graph[0], incoming);
  if (canonical.ids.size() != 3 || canonical.ids[0] != 1 ||
      canonical.ids[1] != 2 || canonical.ids[2] != 3 ||
      canonical.duplicate_ids != 1) {
    throw std::runtime_error("stable old-plus-new union contract failed");
  }
  graph[0].update_neighbors(canonical.ids);
  auto good = audit_graph_invariants(graph, 64);
  require_canonical_graph(good, "synthetic canonical union");
  if (good.container_max_degree != 64 || good.max_actual_degree != 3) {
    throw std::runtime_error("R64 container/actual-degree contract failed");
  }

  parlay::sequence<Indx> malformed = {0, 3, 3, -1, 8, 4};
  auto screened =
      canonical_neighbor_union(0, graph.size(), graph[0], malformed);
  if (screened.self_ids != 1 || screened.out_of_range_ids != 2 ||
      screened.duplicate_ids != 2 || screened.ids.size() != 4 ||
      screened.ids[0] != 1 || screened.ids[1] != 2 ||
      screened.ids[2] != 3 || screened.ids[3] != 4) {
    throw std::runtime_error("self/range/duplicate screening contract failed");
  }

  printf("[PASS] canonical_two_pass_selftest "
         "legacy_duplicate_slots=%llu canonical_duplicate_slots=%llu "
         "self_edges=%llu out_of_range_edges=%llu "
         "container_R=%ld actual_max_degree=%zu "
         "stable_union=old_unique_then_new_unique\n",
         static_cast<unsigned long long>(bad.duplicate_slots),
         static_cast<unsigned long long>(good.duplicate_slots),
         static_cast<unsigned long long>(good.self_edges),
         static_cast<unsigned long long>(good.out_of_range_edges),
         good.container_max_degree, good.max_actual_degree);
  return 0;
}

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

// Degree schedule aligned with the matched Parlay-IVF comparison
// The default degree schedule is selected by support cardinality.
// Larger supports use denser graphs.
// Higher R = denser graph = higher recall ceiling at cost of build time + RAM.
static int pick_R(int64_t shard_n) {
  if (const char* e = std::getenv("BCI_SHARD_R")) { int r = atoi(e); if (r > 0) return r; }
  if (shard_n >= 400000) return 32; // large class
  if (shard_n >= 100000) return 24; // medium
  return 16;                         // small
}

int main(int argc, char** argv) {
  if (argc == 2 &&
      std::string(argv[1]) == "--canonical-two-pass-selftest") {
    return canonical_two_pass_selftest();
  }

  const std::string DATA       = "./data/yfcc10m/";
  const std::string base_path  = DATA + "base.10M.u8bin";
  const std::string data_root  = argc > 1 ? argv[1] :
    "./indexes/yfcc10m";
  const std::string shards_dir = data_root + "/shards";
  std::filesystem::create_directories(shards_dir);

  // Caller can pass max-shard cap (0 = all) and freq filter band as fraction of N.
  int    cap     = argc > 2 ? atoi(argv[2]) : 0;
  double min_pct = argc > 3 ? atof(argv[3]) : 0.10;   // default 0.10% of N
  double max_pct = argc > 4 ? atof(argv[4]) : 10.0;   // default 10.0% of N
  const int selected_tag = std::getenv("BCI_SHARD_TAG")
      ? std::atoi(std::getenv("BCI_SHARD_TAG")) : -1;
  const int build_L = std::getenv("BCI_SHARD_L")
      ? std::atoi(std::getenv("BCI_SHARD_L")) : 200;
  const double build_alpha = std::getenv("BCI_SHARD_ALPHA")
      ? std::atof(std::getenv("BCI_SHARD_ALPHA")) : 1.175;
  const bool build_two_pass = std::getenv("BCI_SHARD_TWO_PASS")
      ? std::atoi(std::getenv("BCI_SHARD_TWO_PASS")) != 0 : false;
  if (selected_tag < -1 || build_L <= 0 ||
      !(build_alpha > 0.0)) {
    throw std::runtime_error(
        "invalid BCI_SHARD_TAG/BCI_SHARD_L/BCI_SHARD_ALPHA");
  }

  printf("=== HAMCG multi-shard Vamana builder ===\n");
  printf("data root  = %s\n", data_root.c_str());
  printf("shards out = %s\n", shards_dir.c_str());
  printf("cap        = %d (0 = build all in-range)\n", cap);
  printf("band       = [%.3f%%, %.3f%%] of N\n", min_pct, max_pct);
  printf("selected   = %d (-1 = all); build L=%d alpha=%.6f "
         "two_pass=%d implementation=%s\n",
         selected_tag, build_L, build_alpha, (int)build_two_pass,
         build_two_pass ? "canonical_dedup_v1" : "single_pass_bundled");

  auto tags = load_sweet_spot_tags(data_root + "/sweet_spot_tags.bin");
  printf("[loaded] %zu candidate tags\n", tags.size());

  // load base points once
  printf("[loading] base.10M.u8bin ...\n");
  auto t0 = std::chrono::steady_clock::now();
  auto base = std::make_shared<PR>(base_path.c_str());
  auto load_secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  printf("[loaded] %ld points dim=%ld (%.1fs, RSS will be ~%.1fGB)\n",
         base->size(), base->dimension(), load_secs,
         base->size() * base->aligned_dimension() / 1e9);

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
  const int64_t N = base->size();
  std::vector<TagEntry> work;
  for (auto& te : tags) {
    if (selected_tag >= 0 && te.tag != selected_tag) continue;
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
    int L = build_L;
    double alpha = build_alpha;

    auto ts = std::chrono::steady_clock::now();
    SubPR subset_pr(*base, subset);
    GraphI G((long)R, subset.size());
    stats<Indx> S(G.size());
    if (build_two_pass) {
      auto first =
          canonical_vamana_pass(G, subset_pr, S, R, L, 1.0);
      require_canonical_graph(first.graph, "canonical alpha=1 first pass");
      printf("[canonical pass 1] removed_duplicate_candidates=%llu "
             "edges=%llu actual_max_degree=%zu container_R=%ld\n",
             static_cast<unsigned long long>(
                 first.duplicate_candidates_removed),
             static_cast<unsigned long long>(first.graph.edge_slots),
             first.graph.max_actual_degree, first.graph.container_max_degree);

      auto second =
          canonical_vamana_pass(G, subset_pr, S, R, L, alpha);
      require_canonical_graph(second.graph, "canonical alpha second pass");
      printf("[canonical pass 2] removed_duplicate_candidates=%llu "
             "edges=%llu duplicate_slots=%llu self_edges=%llu "
             "out_of_range_edges=%llu actual_max_degree=%zu container_R=%ld\n",
             static_cast<unsigned long long>(
                 second.duplicate_candidates_removed),
             static_cast<unsigned long long>(second.graph.edge_slots),
             static_cast<unsigned long long>(second.graph.duplicate_slots),
             static_cast<unsigned long long>(second.graph.self_edges),
             static_cast<unsigned long long>(
                 second.graph.out_of_range_edges),
             second.graph.max_actual_degree,
             second.graph.container_max_degree);
    } else {
      BuildParams params(R, L, alpha);
      KNNIdx index(params);
      index.build_index(G, subset_pr, S);
    }

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
