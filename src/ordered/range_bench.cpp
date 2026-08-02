// Fixed-block HAMCG composition for WoW-compatible SIFT10M range queries.
//
// The numeric domain [0,5000) is partitioned into 200 disjoint 25-value
// blocks before any query is observed.  A range query searches every
// overlapping block graph and merges its candidates; candidates from the two
// boundary blocks are post-filtered by the exact range.  Each point belongs to
// exactly one graph, so graph membership remains linear in N.

#include <algorithm>
#include <cassert>
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
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parlay/parallel.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/types.h"

using PointT = Euclidian_Point<float>;
using HeaderPR = PointRange<float, PointT>;
using Indx = int32_t;
using GraphI = Graph<Indx>;

constexpr int kCardinality = 5000;
constexpr int kBlockWidth = 25;
constexpr int kNumBlocks = kCardinality / kBlockWidth;

class RawPointRange {
 public:
  RawPointRange(const std::string& path, size_t n, unsigned int dim)
      : n_(n), dim_(dim), bytes_(n * static_cast<size_t>(dim) * sizeof(float)) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open raw vectors " + path);
    off_t size = ::lseek(fd_, 0, SEEK_END);
    if (size != static_cast<off_t>(bytes_)) {
      ::close(fd_);
      throw std::runtime_error("unexpected raw vector size " + path);
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
    return PointT(const_cast<float*>(data_ + static_cast<size_t>(i) * dim_), dim_, dim_, i);
  }

 private:
  int fd_ = -1;
  const float* data_ = nullptr;
  size_t n_ = 0;
  unsigned int dim_ = 0;
  size_t bytes_ = 0;
};

struct ThinSubPR {
  RawPointRange& pr;
  const parlay::sequence<int32_t>& subset;
  ThinSubPR(RawPointRange& p, const parlay::sequence<int32_t>& s) : pr(p), subset(s) {}
  size_t size() const { return subset.size(); }
  PointT operator[](long i) { return pr[subset[i]]; }
  long dimension() const { return pr.dimension(); }
  long aligned_dimension() const { return pr.aligned_dimension(); }
};

struct Shard {
  parlay::sequence<int32_t> subset;
  GraphI graph;
  long max_degree = 0;
};

struct Range {
  int32_t lo;
  int32_t hi;
};

struct Task {
  int query_id;
  int block_id;
  int overlap_values;
  bool full_block;
  int clause_id;
};

static parlay::sequence<int32_t> load_subset(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open subset " + path);
  int32_t n = 0;
  if (std::fread(&n, sizeof(n), 1, f) != 1 || n < 0)
    throw std::runtime_error("bad subset header " + path);
  parlay::sequence<int32_t> ids(n);
  if (std::fread(ids.data(), sizeof(int32_t), n, f) != static_cast<size_t>(n))
    throw std::runtime_error("short subset " + path);
  std::fclose(f);
  return ids;
}

static std::vector<int32_t> load_i32(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  size_t bytes = static_cast<size_t>(in.tellg());
  in.seekg(0);
  if (bytes != expected * sizeof(int32_t))
    throw std::runtime_error("unexpected int32 file size " + path);
  std::vector<int32_t> out(expected);
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

static std::vector<Range> load_ranges(const std::string& path, int nq) {
  auto flat = load_i32(path, static_cast<size_t>(nq) * 2);
  std::vector<Range> ranges(nq);
  for (int i = 0; i < nq; ++i) {
    ranges[i] = {flat[2 * i], flat[2 * i + 1]};
    if (ranges[i].lo < 0 || ranges[i].hi >= kCardinality || ranges[i].lo > ranges[i].hi)
      throw std::runtime_error("invalid range at query " + std::to_string(i));
  }
  return ranges;
}

static std::vector<std::vector<Range>> load_predicates(
    const std::string& path, int nq, int clauses) {
  auto flat = load_i32(path, static_cast<size_t>(nq) * 2 * clauses);
  std::vector<std::vector<Range>> predicates(nq, std::vector<Range>(clauses));
  for (int qi = 0; qi < nq; ++qi) {
    for (int ci = 0; ci < clauses; ++ci) {
      size_t off = static_cast<size_t>(qi) * 2 * clauses + 2 * ci;
      Range range{flat[off], flat[off + 1]};
      if (range.lo < 0 || range.hi >= kCardinality || range.lo > range.hi)
        throw std::runtime_error("invalid predicate range at query " + std::to_string(qi));
      predicates[qi][ci] = range;
    }
  }
  return predicates;
}

static std::vector<std::vector<uint32_t>> load_wow_gt(const std::string& path, int nq) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open GT " + path);
  std::vector<std::vector<uint32_t>> gt(nq);
  for (int qi = 0; qi < nq; ++qi) {
    int32_t k = 0;
    if (std::fread(&k, sizeof(k), 1, f) != 1 || k < 0 || k > 100000)
      throw std::runtime_error("bad GT row " + std::to_string(qi));
    gt[qi].resize(k);
    if (std::fread(gt[qi].data(), sizeof(uint32_t), k, f) != static_cast<size_t>(k))
      throw std::runtime_error("short GT row " + std::to_string(qi));
  }
  std::fclose(f);
  return gt;
}

static std::vector<int> parse_beams(const std::string& spec) {
  std::vector<int> beams;
  size_t start = 0;
  while (start < spec.size()) {
    size_t end = spec.find(',', start);
    std::string token = spec.substr(start, end == std::string::npos ? end : end - start);
    int beam = std::stoi(token);
    if (beam <= 0) throw std::runtime_error("beam must be positive");
    beams.push_back(beam);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return beams;
}

int main(int argc, char** argv) {
  const std::string data = argc > 1 ? argv[1] : "./data/sift10m/range_blocks";
  const std::string ranges_path = argc > 2 ? argv[2] : data + "/ranges_unseen.bin";
  const std::string gt_path = argc > 3 ? argv[3] : data + "/gt_unseen.bin";
  const std::vector<int> beams = parse_beams(argc > 4 ? argv[4] : "64,128,256,512");
  const int runs = argc > 5 ? std::max(1, std::atoi(argv[5])) : 2;
  const double boundary_factor = argc > 6 ? std::atof(argv[6]) : 4.0;
  const int nq = argc > 7 ? std::atoi(argv[7]) : 1000;
  const std::string schedule = argc > 8 ? argv[8] : "block";
  if (schedule != "block" && schedule != "query")
    throw std::runtime_error("schedule must be block or query");
  const std::string predicate_mode = argc > 9 ? argv[9] : "range";
  if (predicate_mode != "range" && predicate_mode != "dnf2")
    throw std::runtime_error("predicate mode must be range or dnf2");
  const int clauses = predicate_mode == "dnf2" ? 2 : 1;
  const int K = 10;
  const char* pass_csv_path = std::getenv("BCI_RANGE_PASS_CSV");
  const char* perquery_csv_path = std::getenv("BCI_RANGE_PERQUERY_CSV");
  if ((pass_csv_path || perquery_csv_path) && schedule != "query")
    throw std::runtime_error("formal telemetry requires query scheduling");
  std::ofstream pass_csv;
  std::ofstream perquery_csv;
  if (pass_csv_path) {
    if (std::filesystem::exists(pass_csv_path))
      throw std::runtime_error("refusing to overwrite pass CSV");
    pass_csv.open(pass_csv_path, std::ios::out | std::ios::trunc);
    if (!pass_csv) throw std::runtime_error("cannot create pass CSV");
    pass_csv << "beam,pass,queries,threads,wall_seconds,qps\n";
  }
  if (perquery_csv_path) {
    if (std::filesystem::exists(perquery_csv_path))
      throw std::runtime_error("refusing to overwrite per-query CSV");
    perquery_csv.open(perquery_csv_path, std::ios::out | std::ios::trunc);
    if (!perquery_csv) throw std::runtime_error("cannot create per-query CSV");
    perquery_csv << "beam,qid,latency_us,correct,total,recall,returned_unique,result_ids\n";
  }

  const std::string base_path = argc > 10 ? argv[10] : "./data/sift10m/sift_base.fvecs";
  const std::string query_path = argc > 11 ? argv[11] : "./data/sift10m/sift_query.fvecs";
  const std::string meta_path = argc > 12 ? argv[12] : "./data/sift10m/meta_c5000_n10000000.bin";

  std::printf("=== fixed-block BCI range bench ===\n");
  std::printf("data=%s nq=%d block_width=%d blocks=%d workers=%ld runs=%d boundary_factor=%.2f schedule=%s predicate=%s\n",
              data.c_str(), nq, kBlockWidth, kNumBlocks, parlay::num_workers(), runs,
              boundary_factor, schedule.c_str(), predicate_mode.c_str());
  RawPointRange base(base_path, 10'000'000, 128);
  HeaderPR query(query_path.c_str());
  if (nq > static_cast<int>(query.size())) throw std::runtime_error("not enough queries");
  auto attrs = load_i32(meta_path, 10'000'000);
  auto predicates = load_predicates(ranges_path, nq, clauses);
  auto gt = load_wow_gt(gt_path, nq);

  std::vector<std::unique_ptr<Shard>> shards(kNumBlocks);
  auto load_start = std::chrono::steady_clock::now();
  size_t graph_bytes = 0;
  size_t memberships = 0;
  for (int block = 0; block < kNumBlocks; ++block) {
    auto sh = std::make_unique<Shard>();
    sh->subset = load_subset(data + "/subset_idx/subset_idx_" + std::to_string(block) + ".bin");
    std::string graph_path = data + "/shards/vamana_tag_" + std::to_string(block) + ".bin";
    if (!std::filesystem::exists(graph_path))
      throw std::runtime_error("missing graph " + graph_path);
    graph_bytes += std::filesystem::file_size(graph_path);
    memberships += sh->subset.size();
    sh->graph = GraphI(const_cast<char*>(graph_path.c_str()));
    sh->max_degree = sh->graph.max_degree();
    shards[block] = std::move(sh);
  }
  double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  std::printf("[loaded] memberships=%zu (%.3fx N) graph_bytes=%.3fGB load=%.9fs\n",
              memberships, memberships / 10'000'000.0, graph_bytes / 1e9, load_s);

  std::vector<Task> tasks;
  std::vector<std::vector<int>> block_task_ids(kNumBlocks);
  for (int qi = 0; qi < nq; ++qi) {
    for (int ci = 0; ci < clauses; ++ci) {
      const Range range = predicates[qi][ci];
      int first = range.lo / kBlockWidth;
      int last = range.hi / kBlockWidth;
      for (int block = first; block <= last; ++block) {
        int block_lo = block * kBlockWidth;
        int block_hi = block_lo + kBlockWidth - 1;
        int overlap = std::min(range.hi, block_hi) - std::max(range.lo, block_lo) + 1;
        Task task{qi, block, overlap, overlap == kBlockWidth, ci};
        int task_id = static_cast<int>(tasks.size());
        tasks.push_back(task);
        block_task_ids[block].push_back(task_id);
      }
    }
  }
  std::printf("[plan] tasks=%zu avg_blocks_per_query=%.2f\n", tasks.size(), tasks.size() / double(nq));

  for (int beam : beams) {
    std::vector<std::vector<std::pair<float, int32_t>>> task_results(
        schedule == "block" ? tasks.size() : 0);
    std::vector<std::vector<int32_t>> final_ids(nq);
    std::vector<double> final_query_latency_us(nq, 0.0);
    std::vector<double> run_walls;
    auto search_block = [&](int qi, int block, int overlap_values, bool full_block,
                            const Range active_range,
                            ThinSubPR& points,
                            std::vector<std::pair<float, int32_t>>& out) {
      auto& sh = *shards[block];
      double fraction = overlap_values / double(kBlockWidth);
      int pool = full_block ? K : static_cast<int>(std::ceil(boundary_factor * K / fraction));
      pool = std::min<int>(pool, sh.subset.size());
      int task_beam = std::max(beam, pool);
      long limit = std::min<long>(sh.graph.size(), std::max<long>(100000L, 100L * task_beam));
      QueryParams qp(pool, task_beam, 1.35, limit, sh.max_degree);
      PointT q = query[qi];
      auto search = beam_search<PointT, ThinSubPR, Indx>(
          q, sh.graph, points, 0, qp);
      auto& frontier = search.first.first;
      out.clear();
      out.reserve(frontier.size());
      for (const auto& item : frontier) {
        int32_t global = sh.subset[item.first];
        int32_t attr = attrs[global];
        if (attr >= active_range.lo && attr <= active_range.hi)
          out.emplace_back(item.second, global);
      }
    };
    auto finish_query = [&](int qi, std::vector<std::pair<float, int32_t>>& merged) {
      if (clauses > 1) {
        std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
          return a.second < b.second ||
                 (a.second == b.second && a.first < b.first);
        });
        size_t write = 0;
        for (size_t read = 0; read < merged.size();) {
          const int32_t id = merged[read].second;
          float best = merged[read].first;
          size_t next = read + 1;
          while (next < merged.size() && merged[next].second == id) {
            best = std::min(best, merged[next].first);
            ++next;
          }
          merged[write++] = {best, id};
          read = next;
        }
        merged.resize(write);
      }
      int take = std::min<int>(K, merged.size());
      if (take > 0) {
        std::partial_sort(merged.begin(), merged.begin() + take, merged.end(),
                          [](const auto& a, const auto& b) {
                            return a.first < b.first || (a.first == b.first && a.second < b.second);
                          });
      }
      final_ids[qi].clear();
      for (int i = 0; i < take; ++i) final_ids[qi].push_back(merged[i].second);
    };
    for (int run = 0; run < runs; ++run) {
      auto begin = std::chrono::steady_clock::now();
      if (schedule == "block") {
        parlay::parallel_for(0, kNumBlocks, [&](size_t block_index) {
          int block = static_cast<int>(block_index);
          auto& sh = *shards[block];
          ThinSubPR points(base, sh.subset);
          for (int task_id : block_task_ids[block]) {
            const Task& task = tasks[task_id];
            search_block(task.query_id, block, task.overlap_values, task.full_block,
                         predicates[task.query_id][task.clause_id],
                         points, task_results[task_id]);
          }
        });

        for (int qi = 0; qi < nq; ++qi) {
          std::vector<std::pair<float, int32_t>> merged;
          for (size_t task_id = 0; task_id < tasks.size(); ++task_id) {
            if (tasks[task_id].query_id == qi)
              merged.insert(merged.end(), task_results[task_id].begin(), task_results[task_id].end());
          }
          finish_query(qi, merged);
        }
      } else {
        parlay::parallel_for(0, nq, [&](size_t query_index) {
          int qi = static_cast<int>(query_index);
          auto query_begin = std::chrono::steady_clock::now();
          std::vector<std::pair<float, int32_t>> merged;
          size_t query_blocks = 0;
          for (int ci = 0; ci < clauses; ++ci) {
            query_blocks += static_cast<size_t>(
                predicates[qi][ci].hi / kBlockWidth -
                predicates[qi][ci].lo / kBlockWidth + 1);
          }
          merged.reserve(query_blocks * static_cast<size_t>(K));
          std::vector<std::pair<float, int32_t>> block_out;
          for (int ci = 0; ci < clauses; ++ci) {
            const Range range = predicates[qi][ci];
            int first = range.lo / kBlockWidth;
            int last = range.hi / kBlockWidth;
            for (int block = first; block <= last; ++block) {
              int block_lo = block * kBlockWidth;
              int block_hi = block_lo + kBlockWidth - 1;
              int overlap = std::min(range.hi, block_hi) - std::max(range.lo, block_lo) + 1;
              ThinSubPR points(base, shards[block]->subset);
              search_block(qi, block, overlap, overlap == kBlockWidth, range, points, block_out);
              merged.insert(merged.end(), block_out.begin(), block_out.end());
            }
          }
          finish_query(qi, merged);
          final_query_latency_us[qi] = std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - query_begin).count();
        });
      }
      double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      run_walls.push_back(wall);
      std::printf("  beam=%d run=%d/%d wall=%.6fs qps=%.2f\n", beam, run + 1, runs, wall, nq / wall);
      if (pass_csv) {
        pass_csv << beam << ',' << run + 1 << ',' << nq << ',' << parlay::num_workers()
                 << ',' << std::setprecision(12) << wall << ',' << nq / wall << '\n';
        pass_csv.flush();
      }
    }

    size_t correct = 0;
    size_t total = 0;
    size_t by_width_correct[3] = {0, 0, 0};
    size_t by_width_total[3] = {0, 0, 0};
    for (int qi = 0; qi < nq; ++qi) {
      int cls = qi % 3;
      std::unordered_set<uint32_t> truth(gt[qi].begin(), gt[qi].end());
      std::unordered_set<int32_t> returned(final_ids[qi].begin(), final_ids[qi].end());
      std::vector<int32_t> returned_sorted(returned.begin(), returned.end());
      std::sort(returned_sorted.begin(), returned_sorted.end());
      total += truth.size();
      by_width_total[cls] += truth.size();
      for (int32_t id : returned) {
        bool hit = truth.count(static_cast<uint32_t>(id));
        correct += hit;
        by_width_correct[cls] += hit;
        bool valid = false;
        for (const Range range : predicates[qi])
          valid = valid || (attrs[id] >= range.lo && attrs[id] <= range.hi);
        if (!valid)
          throw std::runtime_error("range-invalid result");
      }
      if (perquery_csv) {
        size_t query_correct = 0;
        for (int32_t id : returned) query_correct += truth.count(static_cast<uint32_t>(id));
        if (truth.empty()) throw std::runtime_error("empty ground-truth row");
        perquery_csv << beam << ',' << qi << ',' << std::setprecision(12)
                     << final_query_latency_us[qi] << ',' << query_correct << ','
                     << truth.size() << ',' << static_cast<double>(query_correct) / truth.size()
                     << ',' << returned.size() << ',';
        for (size_t result_index = 0; result_index < returned_sorted.size(); ++result_index) {
          if (result_index) perquery_csv << ';';
          perquery_csv << returned_sorted[result_index];
        }
        perquery_csv << '\n';
      }
    }
    if (perquery_csv) perquery_csv.flush();
    std::vector<double> measured_qps;
    size_t first_measured = runs > 1 ? 1 : 0;
    for (size_t i = first_measured; i < run_walls.size(); ++i) measured_qps.push_back(nq / run_walls[i]);
    std::sort(measured_qps.begin(), measured_qps.end());
    double qps_median = measured_qps[measured_qps.size() / 2];
    std::printf("[RESULT] beam=%d first_pass_qps=%.2f warm_qps_median=%.2f recall=%.6f correct=%zu total=%zu "
                "recall_class0=%.6f recall_class1=%.6f recall_class2=%.6f\n",
                beam, nq / run_walls.front(), qps_median,
                static_cast<double>(correct) / total, correct, total,
                static_cast<double>(by_width_correct[0]) / by_width_total[0],
                static_cast<double>(by_width_correct[1]) / by_width_total[1],
                static_cast<double>(by_width_correct[2]) / by_width_total[2]);
  }
  return 0;
}
