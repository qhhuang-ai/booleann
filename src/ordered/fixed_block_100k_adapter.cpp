// Minimal scale adapter for the retained Boole fixed-block SIFT range path.
//
// Algorithmic choices are intentionally identical to
// bci_cpp_port/src/{build_sift10m_range_blocks,sift10m_range_bci_bench}.cpp:
// disjoint fixed blocks, one Vamana graph per block, exact boundary
// post-filtering, and distance merge.  Only paths, N, attribute cardinality,
// and the two prospectively frozen batch sizes differ.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/stats.h"
#include "utils/types.h"
#include "vamana/index.h"

using PointT = Euclidian_Point<float>;
using Indx = int32_t;
using GraphI = Graph<Indx>;

constexpr int kN = 100'000;
constexpr int kDim = 128;
constexpr int kK = 10;
constexpr int kCardinality = 100'000;
constexpr int kNumBlocks = 200;
constexpr int kBlockWidth = kCardinality / kNumBlocks;
constexpr int kBuildR = 32;
constexpr int kBuildL = 200;
constexpr double kBuildAlpha = 1.175;
constexpr int kBeam = 64;
constexpr double kBoundaryFactor = 4.0;

class RawPointRange {
 public:
  RawPointRange(const std::string& path, size_t n, unsigned int dim)
      : n_(n), dim_(dim), bytes_(n * static_cast<size_t>(dim) * sizeof(float)) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open raw vectors " + path);
    const off_t size = ::lseek(fd_, 0, SEEK_END);
    if (size != static_cast<off_t>(bytes_)) {
      ::close(fd_);
      throw std::runtime_error("unexpected raw-vector size " + path);
    }
    void* mapped = ::mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("mmap failed for " + path);
    }
    data_ = static_cast<const float*>(mapped);
  }
  ~RawPointRange() {
    if (data_) ::munmap(const_cast<float*>(data_), bytes_);
    if (fd_ >= 0) ::close(fd_);
  }
  RawPointRange(const RawPointRange&) = delete;
  RawPointRange& operator=(const RawPointRange&) = delete;
  size_t size() const { return n_; }
  long dimension() const { return dim_; }
  long aligned_dimension() const { return dim_; }
  PointT operator[](long i) {
    return PointT(const_cast<float*>(
                      data_ + static_cast<size_t>(i) * dim_),
                  dim_, dim_, i);
  }

 private:
  int fd_ = -1;
  const float* data_ = nullptr;
  size_t n_ = 0;
  unsigned int dim_ = 0;
  size_t bytes_ = 0;
};

struct ThinSubPR {
  RawPointRange& points;
  const parlay::sequence<int32_t>& subset;
  ThinSubPR(RawPointRange& p, const parlay::sequence<int32_t>& s)
      : points(p), subset(s) {}
  size_t size() const { return subset.size(); }
  long dimension() const { return points.dimension(); }
  long aligned_dimension() const { return points.aligned_dimension(); }
  PointT operator[](long i) { return points[subset[i]]; }
};

using KNNIdx = knn_index<PointT, ThinSubPR, Indx>;

struct Range {
  int32_t lo;
  int32_t hi;
};

struct Shard {
  parlay::sequence<int32_t> subset;
  std::unique_ptr<GraphI> graph;
  long max_degree = 0;
};

struct StageAudit {
  size_t hits = 0;
  size_t total = 0;
  size_t duplicate_rows = 0;
  size_t invalid_ids = 0;
  size_t predicate_violations = 0;
  size_t missing_results = 0;
};

static std::vector<int32_t> load_i32(
    const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const size_t bytes = static_cast<size_t>(in.tellg());
  in.seekg(0);
  if (bytes != expected * sizeof(int32_t))
    throw std::runtime_error("unexpected int32 file size " + path);
  std::vector<int32_t> out(expected);
  in.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(bytes));
  if (!in) throw std::runtime_error("short read " + path);
  return out;
}

static parlay::sequence<int32_t> load_subset(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("cannot open subset " + path);
  int32_t n = 0;
  if (std::fread(&n, sizeof(n), 1, file) != 1 || n <= 0)
    throw std::runtime_error("bad subset header " + path);
  parlay::sequence<int32_t> ids(n);
  if (std::fread(ids.data(), sizeof(int32_t), n, file) !=
      static_cast<size_t>(n))
    throw std::runtime_error("short subset " + path);
  if (std::fgetc(file) != EOF)
    throw std::runtime_error("trailing subset bytes " + path);
  std::fclose(file);
  return ids;
}

template <class T>
static void write_raw_exclusive(
    const std::string& path, const std::vector<T>& values) {
  if (std::filesystem::exists(path))
    throw std::runtime_error("refusing to overwrite " + path);
  std::ofstream out(path, std::ios::binary | std::ios::out);
  if (!out) throw std::runtime_error("cannot create " + path);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  out.flush();
  if (!out) throw std::runtime_error("write failed " + path);
}

static long peak_rss_kib() {
  struct rusage usage {};
  if (::getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return usage.ru_maxrss;
}

static uintmax_t expected_graph_bytes(GraphI& graph) {
  uintmax_t bytes =
      2 * sizeof(Indx) + graph.size() * static_cast<uintmax_t>(sizeof(Indx));
  for (size_t vertex = 0; vertex < graph.size(); ++vertex)
    bytes += graph[static_cast<Indx>(vertex)].size() *
             static_cast<uintmax_t>(sizeof(Indx));
  return bytes;
}

static StageAudit audit_results(
    const std::vector<std::vector<int32_t>>& ids,
    const std::vector<std::vector<float>>& distances,
    const std::vector<int32_t>& attrs,
    const std::vector<Range>& ranges,
    const std::vector<int32_t>& gt,
    int nq) {
  StageAudit audit;
  audit.total = static_cast<size_t>(nq) * kK;
  for (int qi = 0; qi < nq; ++qi) {
    std::unordered_set<int32_t> truth;
    for (int j = 0; j < kK; ++j)
      truth.insert(gt[static_cast<size_t>(qi) * kK + j]);
    std::unordered_set<int32_t> seen;
    bool duplicate = false;
    for (int j = 0; j < kK; ++j) {
      if (j >= static_cast<int>(ids[qi].size()) ||
          j >= static_cast<int>(distances[qi].size()) ||
          !std::isfinite(distances[qi][j])) {
        ++audit.missing_results;
        continue;
      }
      const int32_t id = ids[qi][j];
      if (!seen.insert(id).second) duplicate = true;
      if (id < 0 || id >= kN) {
        ++audit.invalid_ids;
        continue;
      }
      if (attrs[id] < ranges[qi].lo || attrs[id] > ranges[qi].hi)
        ++audit.predicate_violations;
      audit.hits += truth.count(id);
    }
    audit.duplicate_rows += duplicate;
  }
  return audit;
}

static void persist_stage(
    const std::string& output_root,
    const std::string& name,
    const std::vector<std::vector<int32_t>>& ids,
    const std::vector<std::vector<float>>& distances,
    int nq) {
  std::vector<int32_t> flat_ids(static_cast<size_t>(nq) * kK, -1);
  std::vector<float> flat_distances(
      static_cast<size_t>(nq) * kK,
      std::numeric_limits<float>::infinity());
  for (int qi = 0; qi < nq; ++qi) {
    const int take = std::min<int>(kK, ids[qi].size());
    for (int j = 0; j < take; ++j) {
      flat_ids[static_cast<size_t>(qi) * kK + j] = ids[qi][j];
      flat_distances[static_cast<size_t>(qi) * kK + j] = distances[qi][j];
    }
  }
  write_raw_exclusive(output_root + "/" + name + "_ids.i32", flat_ids);
  write_raw_exclusive(
      output_root + "/" + name + "_distances.f32", flat_distances);
}

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      std::fprintf(
          stderr,
          "usage: %s BASE_RAW QUERY50_RAW ATTR100K_I32 RANGES50_I32 "
          "GT50X10_I32 SUBSET_ROOT GRAPH_ROOT OUTPUT_ROOT\n",
          argv[0]);
      return 2;
    }
    const std::string base_path = argv[1];
    const std::string query_path = argv[2];
    const std::string attr_path = argv[3];
    const std::string ranges_path = argv[4];
    const std::string gt_path = argv[5];
    const std::string subset_root = argv[6];
    const std::string graph_root = argv[7];
    const std::string output_root = argv[8];
    std::filesystem::create_directories(graph_root);
    std::filesystem::create_directories(output_root);

    std::printf(
        "[config] n=%d dim=%d k=%d cardinality=%d blocks=%d block_width=%d "
        "R=%d L=%d alpha=%.3f beam=%d boundary_factor=%.1f workers=%ld\n",
        kN, kDim, kK, kCardinality, kNumBlocks, kBlockWidth, kBuildR,
        kBuildL, kBuildAlpha, kBeam, kBoundaryFactor, parlay::num_workers());
    std::fflush(stdout);
    if (parlay::num_workers() != 8)
      throw std::runtime_error("expected exactly eight Parlay workers");

    RawPointRange base(base_path, kN, kDim);
    RawPointRange query(query_path, 50, kDim);
    const auto attrs = load_i32(attr_path, kN);
    const auto ranges_flat = load_i32(ranges_path, 50 * 2);
    const auto gt = load_i32(gt_path, 50 * kK);
    std::vector<Range> ranges(50);
    for (int qi = 0; qi < 50; ++qi) {
      ranges[qi] = {
          ranges_flat[2 * qi], ranges_flat[2 * qi + 1]};
      if (ranges[qi].lo < 0 || ranges[qi].hi >= kCardinality ||
          ranges[qi].lo > ranges[qi].hi)
        throw std::runtime_error("invalid range row");
    }

    std::vector<std::unique_ptr<Shard>> shards(kNumBlocks);
    std::vector<uint8_t> original_id_seen(kN, 0);
    size_t memberships = 0;
    for (int block = 0; block < kNumBlocks; ++block) {
      auto shard = std::make_unique<Shard>();
      shard->subset = load_subset(
          subset_root + "/subset_idx_" + std::to_string(block) + ".bin");
      memberships += shard->subset.size();
      for (int32_t id : shard->subset) {
        if (id < 0 || id >= kN || original_id_seen[id])
          throw std::runtime_error("invalid/duplicate subset original ID");
        if (attrs[id] / kBlockWidth != block)
          throw std::runtime_error("attribute/subset block mismatch");
        original_id_seen[id] = 1;
      }
      shards[block] = std::move(shard);
    }
    if (memberships != kN ||
        std::count(original_id_seen.begin(), original_id_seen.end(), 1) != kN)
      throw std::runtime_error("subset partition is not exactly one-copy N");
    std::printf("[subsets] memberships=%zu exact_partition=true\n", memberships);
    std::fflush(stdout);

    const auto build_begin = std::chrono::steady_clock::now();
    size_t graph_bytes = 0;
    for (int block = 0; block < kNumBlocks; ++block) {
      const std::string graph_path =
          graph_root + "/vamana_block_" + std::to_string(block) + ".bin";
      if (std::filesystem::exists(graph_path))
        throw std::runtime_error("refusing existing graph " + graph_path);
      ThinSubPR points(base, shards[block]->subset);
      GraphI graph(kBuildR, shards[block]->subset.size());
      stats<Indx> build_stats(graph.size());
      BuildParams params(kBuildR, kBuildL, kBuildAlpha);
      KNNIdx index(params);
      const auto one_begin = std::chrono::steady_clock::now();
      index.build_index(graph, points, build_stats);
      const uintmax_t expected_bytes = expected_graph_bytes(graph);
      graph.save(const_cast<char*>(graph_path.c_str()));
      const uintmax_t saved_bytes = std::filesystem::file_size(graph_path);
      if (saved_bytes != expected_bytes)
        throw std::runtime_error("graph serialization size mismatch");
      graph_bytes += saved_bytes;
      const double one_wall = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - one_begin).count();
      std::printf(
          "[build] block=%d/199 n=%zu wall_seconds=%.9f bytes=%ju\n",
          block, shards[block]->subset.size(), one_wall,
          static_cast<uintmax_t>(saved_bytes));
      std::fflush(stdout);
    }
    const double build_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - build_begin).count();
    const long build_peak_rss = peak_rss_kib();
    std::printf(
        "[build_done] wall_seconds=%.12f graph_files=%d graph_bytes=%zu "
        "peak_rss_kib=%ld\n",
        build_wall, kNumBlocks, graph_bytes, build_peak_rss);
    std::fflush(stdout);

    const auto load_begin = std::chrono::steady_clock::now();
    for (int block = 0; block < kNumBlocks; ++block) {
      const std::string graph_path =
          graph_root + "/vamana_block_" + std::to_string(block) + ".bin";
      shards[block]->graph =
          std::make_unique<GraphI>(const_cast<char*>(graph_path.c_str()));
      shards[block]->max_degree = shards[block]->graph->max_degree();
    }
    const double load_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_begin).count();
    std::printf(
        "[load_done] wall_seconds=%.12f peak_rss_kib=%ld\n",
        load_wall, peak_rss_kib());
    std::fflush(stdout);

    const std::string summary_path = output_root + "/engine_summary.tsv";
    if (std::filesystem::exists(summary_path))
      throw std::runtime_error("refusing to overwrite " + summary_path);
    std::ofstream summary(summary_path);
    if (!summary) throw std::runtime_error("cannot create engine summary");
    summary
        << "stage\tnq\twall_seconds\tqps\thits\ttotal\tduplicate_rows"
        << "\tinvalid_ids\tpredicate_violations\tmissing_results"
        << "\tpeak_rss_kib\n";

    auto execute_stage = [&](const std::string& name, int nq) {
      std::vector<std::vector<int32_t>> final_ids(nq);
      std::vector<std::vector<float>> final_distances(nq);
      const auto begin = std::chrono::steady_clock::now();
      parlay::parallel_for(0, nq, [&](size_t query_index) {
        const int qi = static_cast<int>(query_index);
        const Range range = ranges[qi];
        const int first = range.lo / kBlockWidth;
        const int last = range.hi / kBlockWidth;
        std::vector<std::pair<float, int32_t>> merged;
        merged.reserve(
            static_cast<size_t>(last - first + 1) * static_cast<size_t>(kK));
        for (int block = first; block <= last; ++block) {
          const int block_lo = block * kBlockWidth;
          const int block_hi = block_lo + kBlockWidth - 1;
          const int overlap =
              std::min(range.hi, block_hi) -
              std::max(range.lo, block_lo) + 1;
          const bool full_block = overlap == kBlockWidth;
          const double fraction =
              static_cast<double>(overlap) / kBlockWidth;
          auto& shard = *shards[block];
          int pool = full_block
                         ? kK
                         : static_cast<int>(
                               std::ceil(kBoundaryFactor * kK / fraction));
          pool = std::min<int>(pool, shard.subset.size());
          const int task_beam = std::max(kBeam, pool);
          const long limit = std::min<long>(
              shard.graph->size(),
              std::max<long>(100000L, 100L * task_beam));
          QueryParams params(
              pool, task_beam, 1.35, limit, shard.max_degree);
          ThinSubPR points(base, shard.subset);
          PointT q = query[qi];
          auto search = beam_search<PointT, ThinSubPR, Indx>(
              q, *shard.graph, points, 0, params);
          for (const auto& item : search.first.first) {
            const int32_t global_id = shard.subset[item.first];
            if (attrs[global_id] >= range.lo &&
                attrs[global_id] <= range.hi)
              merged.emplace_back(item.second, global_id);
          }
        }
        const int take = std::min<int>(kK, merged.size());
        if (take > 0) {
          std::partial_sort(
              merged.begin(), merged.begin() + take, merged.end(),
              [](const auto& a, const auto& b) {
                return a.first < b.first ||
                       (a.first == b.first && a.second < b.second);
              });
        }
        final_ids[qi].reserve(take);
        final_distances[qi].reserve(take);
        for (int j = 0; j < take; ++j) {
          final_ids[qi].push_back(merged[j].second);
          final_distances[qi].push_back(merged[j].first);
        }
      });
      const double wall = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - begin).count();
      const StageAudit audit = audit_results(
          final_ids, final_distances, attrs, ranges, gt, nq);
      persist_stage(output_root, name, final_ids, final_distances, nq);
      const double qps = static_cast<double>(nq) / wall;
      const long rss = peak_rss_kib();
      summary << name << '\t' << nq << '\t' << std::setprecision(17)
              << wall << '\t' << qps << '\t' << audit.hits << '\t'
              << audit.total << '\t' << audit.duplicate_rows << '\t'
              << audit.invalid_ids << '\t'
              << audit.predicate_violations << '\t'
              << audit.missing_results << '\t' << rss << '\n';
      summary.flush();
      std::printf(
          "[stage_done] stage=%s nq=%d wall_seconds=%.12f qps=%.9f "
          "hits=%zu total=%zu recall=%.9f duplicate_rows=%zu "
          "invalid_ids=%zu predicate_violations=%zu missing_results=%zu "
          "peak_rss_kib=%ld\n",
          name.c_str(), nq, wall, qps, audit.hits, audit.total,
          static_cast<double>(audit.hits) / audit.total,
          audit.duplicate_rows, audit.invalid_ids,
          audit.predicate_violations, audit.missing_results, rss);
      std::fflush(stdout);
      return audit;
    };

    const StageAudit q10 = execute_stage("stage1_q10", 10);
    if (q10.hits < 80 || q10.duplicate_rows || q10.invalid_ids ||
        q10.predicate_violations || q10.missing_results)
      throw std::runtime_error("q10 correctness gate failed; q50 forbidden");
    execute_stage("stage2_q50", 50);

    std::ofstream build_summary(output_root + "/build_summary.tsv");
    if (!build_summary) throw std::runtime_error("cannot create build summary");
    build_summary
        << "build_wall_seconds\tgraph_files\tgraph_bytes\tbuild_peak_rss_kib"
        << "\tload_wall_seconds\tfinal_peak_rss_kib\n"
        << std::setprecision(17) << build_wall << '\t' << kNumBlocks << '\t'
        << graph_bytes << '\t' << build_peak_rss << '\t' << load_wall << '\t'
        << peak_rss_kib() << '\n';
    build_summary.flush();
    std::printf("[done] status=success\n");
    std::fflush(stdout);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[fatal] %s\n", error.what());
    std::fflush(stderr);
    return 1;
  }
}
