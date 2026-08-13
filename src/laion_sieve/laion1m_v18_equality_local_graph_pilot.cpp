// Development-only LAION1M v18 equality pilot for Boole-ANN's edge-only
// local-graph layout.  This translation unit deliberately reuses the exact
// workload parser, data contracts, leave-one-out oracle, and validity checks
// from the mixed SIEVE adapter.  It does not modify any baseline source.

#define BOOLEANN_LAION_MIXED_RUNNER_NO_MAIN
#include "laion1m_sieve_mixed_pilot_runner.cpp"

#include <filesystem>
#include <unordered_map>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/types.h"

#ifndef BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS
#define BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS "development_only=true"
#endif

namespace {

using LocalIndex = int32_t;
using LocalGraph = Graph<LocalIndex>;
using LocalPoint = Euclidian_Point<float>;

struct LocalSubsetRange {
  const float* base = nullptr;
  const parlay::sequence<int32_t>* local_to_global = nullptr;

  size_t size() const { return local_to_global->size(); }
  long dimension() const { return D; }
  long aligned_dimension() const { return D; }
  LocalPoint operator[](long local) {
    const int32_t global = (*local_to_global)[static_cast<size_t>(local)];
    return LocalPoint(base + uint64_t(global) * D, D, D, local);
  }
};

parlay::sequence<int32_t> read_local_to_global(const std::string& path) {
  Mapping mapping(path);
  require(mapping.size() >= sizeof(int32_t), "short subset file: " + path);
  const int32_t count = scalar<int32_t>(mapping.bytes());
  require(count >= 0 && mapping.size() == sizeof(int32_t) * (uint64_t(count) + 1),
          "subset extent differs: " + path);
  parlay::sequence<int32_t> result(static_cast<size_t>(count));
  std::memcpy(result.data(), mapping.bytes() + sizeof(int32_t),
              sizeof(int32_t) * static_cast<size_t>(count));
  return result;
}

struct AtomGraph {
  uint16_t token = 0;
  parlay::sequence<int32_t> local_to_global;
  std::unique_ptr<LocalGraph> graph;
};

struct GraphInventory {
  std::array<std::unique_ptr<AtomGraph>, TOKENS> atoms;
  uint64_t graph_file_bytes = 0;
  uint64_t subset_file_bytes = 0;
  uint64_t dense_graph_bytes = 0;
  uint64_t indexed_memberships = 0;
  uint32_t usable_atoms = 0;
  uint32_t rejected_atoms = 0;
};

enum class GraphInventoryMode : uint32_t {
  LegacyRejectToken0,
  CleanAll200,
};

bool known_legacy_token_zero_tail(const std::vector<uint32_t>& posting,
                                  const parlay::sequence<int32_t>& subset,
                                  uint16_t token) {
  if (token != 0 || subset.size() != posting.size() + 1 || subset.back() != 0)
    return false;
  return std::equal(posting.begin(), posting.end(), subset.begin());
}

GraphInventory load_graph_inventory(
    const Data& data, const std::string& root,
    GraphInventoryMode mode = GraphInventoryMode::LegacyRejectToken0) {
  GraphInventory inventory;
  for (uint16_t token = 0; token < TOKENS; ++token) {
    const std::string subset_path = root + "/subset_idx/subset_idx_" +
                                    std::to_string(token) + ".bin";
    const std::string graph_path = root + "/shards/vamana_tag_" +
                                   std::to_string(token) + ".bin";
    require(std::filesystem::is_regular_file(subset_path) &&
                std::filesystem::is_regular_file(graph_path),
            "missing local-graph asset for token " + std::to_string(token));
    inventory.subset_file_bytes += std::filesystem::file_size(subset_path);
    inventory.graph_file_bytes += std::filesystem::file_size(graph_path);
    auto subset = read_local_to_global(subset_path);
    const auto& posting = data.postings[token];
    const bool exact = subset.size() == posting.size() &&
        std::equal(posting.begin(), posting.end(), subset.begin());
    if (!exact) {
      require(mode == GraphInventoryMode::LegacyRejectToken0 &&
                  known_legacy_token_zero_tail(posting, subset, token),
              "local-to-global mapping differs from current CSR for token " +
                  std::to_string(token));
      ++inventory.rejected_atoms;
      std::cout << "ASSET_REJECT token=" << token
                << " reason=known_legacy_transpose_trailing_zero"
                << " " << BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS << std::endl;
      continue;
    }
    auto atom = std::make_unique<AtomGraph>();
    atom->token = token;
    atom->local_to_global = std::move(subset);
    atom->graph = std::make_unique<LocalGraph>(
        const_cast<char*>(graph_path.c_str()));
    require(atom->graph->size() == atom->local_to_global.size() &&
                atom->graph->max_degree() == 16,
            "R16 graph shape differs for token " + std::to_string(token));
    for (size_t local = 0; local < atom->local_to_global.size(); ++local) {
      const int32_t global = atom->local_to_global[local];
      require(global >= 0 && uint32_t(global) < N &&
                  (local == 0 || atom->local_to_global[local - 1] < global),
              "local-to-global map is not sorted unique");
    }
    inventory.indexed_memberships += atom->local_to_global.size();
    inventory.dense_graph_bytes += atom->local_to_global.size() *
        uint64_t(atom->graph->max_degree() + 1) * sizeof(LocalIndex);
    inventory.atoms[token] = std::move(atom);
    ++inventory.usable_atoms;
  }
  if (mode == GraphInventoryMode::LegacyRejectToken0)
    require(inventory.usable_atoms == 199 && inventory.rejected_atoms == 1,
            "legacy R16 inventory census differs");
  else
    require(inventory.usable_atoms == 200 && inventory.rejected_atoms == 0,
            "clean R16 inventory census differs");
  std::cout << "GRAPH_INVENTORY usable_atoms=" << inventory.usable_atoms
            << " rejected_atoms=" << inventory.rejected_atoms
            << " indexed_memberships=" << inventory.indexed_memberships
            << " graph_file_bytes=" << inventory.graph_file_bytes
            << " subset_file_bytes=" << inventory.subset_file_bytes
            << " dense_graph_bytes=" << inventory.dense_graph_bytes
            << " shared_vector_bytes=" << uint64_t(N) * D * sizeof(float)
            << " inventory_mode="
            << (mode == GraphInventoryMode::CleanAll200 ? "clean_all_200"
                                                        : "legacy_reject_token0")
            << " baseline_core_modified=false "
            << BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS
            << std::endl;
  return inventory;
}

#ifndef BOOLEANN_LAION_LOCAL_GRAPH_TYPES_ONLY
struct LocalOptions {
  std::string base;
  std::string spmat;
  std::string endpoints;
  std::string queries;
  std::string graph_root;
  std::array<std::string, ATTRIBUTES> numeric;
  std::vector<uint32_t> beams{128, 256, 512, 1024, 2048};
  uint32_t limit_queries = 16;
  uint32_t pool = 32;
  uint32_t lanes = 1;
  double cut = 1.35;
};

std::vector<uint32_t> parse_beams(const std::string& text) {
  std::vector<uint32_t> result;
  std::istringstream input(text);
  std::string field;
  while (std::getline(input, field, ',')) {
    const unsigned long value = std::stoul(field);
    require(value >= K + 1 && value <= 65536, "beam outside pilot envelope");
    result.push_back(static_cast<uint32_t>(value));
  }
  require(!result.empty() && std::is_sorted(result.begin(), result.end()) &&
              std::adjacent_find(result.begin(), result.end()) == result.end(),
          "beams must be nonempty, sorted, and unique");
  return result;
}

LocalOptions local_options(int argc, char** argv) {
  LocalOptions result;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(i + 1 < argc, "missing option value: " + key);
    const std::string value = argv[++i];
    if (key == "--base") result.base = value;
    else if (key == "--spmat") result.spmat = value;
    else if (key == "--endpoints") result.endpoints = value;
    else if (key == "--queries") result.queries = value;
    else if (key == "--graph-root") result.graph_root = value;
    else if (key == "--numeric-similarity") result.numeric[0] = value;
    else if (key == "--numeric-original-width") result.numeric[1] = value;
    else if (key == "--numeric-original-height") result.numeric[2] = value;
    else if (key == "--beams") result.beams = parse_beams(value);
    else if (key == "--limit-queries") result.limit_queries = std::stoul(value);
    else if (key == "--pool") result.pool = std::stoul(value);
    else if (key == "--lanes") result.lanes = std::stoul(value);
    else if (key == "--cut") result.cut = std::stod(value);
    else fail("unknown option: " + key);
  }
  require(!result.base.empty() && !result.spmat.empty() &&
              !result.endpoints.empty() && !result.queries.empty() &&
              !result.graph_root.empty() &&
              std::all_of(result.numeric.begin(), result.numeric.end(),
                          [](const std::string& value) { return !value.empty(); }) &&
              result.limit_queries > 0 && result.limit_queries <= 200 &&
              result.pool >= K + 1 && result.pool <= 4096 &&
              (result.lanes == 1 || result.lanes == 8) &&
              result.cut >= 1.0 && result.cut <= 4.0,
          "required local-graph option or bound differs");
  return result;
}
#endif

class LocalGraphEngine {
 public:
  LocalGraphEngine(const Data& data, GraphInventory& inventory, uint32_t beam,
                   uint32_t pool, double cut)
      : data_(data), inventory_(inventory), beam_(beam), pool_(pool), cut_(cut) {}

  Result query(const Spec& spec) {
    require(spec.family == Family::Equality,
            "local-graph pilot received a non-equality predicate");
    AtomGraph* atom = inventory_.atoms[spec.primary].get();
    require(atom != nullptr, "query routed to rejected/missing atom graph");
    LocalSubsetRange points{data_.base, &atom->local_to_global};
    // The query is a base row solely to implement leave-one-out evaluation.
    // Its ID must not be interpreted in the graph's local-ID domain.
    LocalPoint query_point(data_.base + uint64_t(spec.query_base_id) * D,
                           D, D, -1);
    const long visit_limit = std::min<long>(
        atom->graph->size(), std::max<long>(100L * beam_, 100000L));
    QueryParams parameters(static_cast<long>(pool_),
                           static_cast<long>(std::max(beam_, pool_)), cut_,
                           visit_limit, atom->graph->max_degree());
    // Match the official SIEVE leave-one-out protocol: self may remain a
    // traversal waypoint, but it is never eligible for the returned top-k.
    // The option parser requires pool >= K+1; self is removed after mapping
    // graph-local IDs back into the shared global-ID domain below.
    // The released ParlayANN API takes five arguments.  The reported v20b
    // comparison keeps this graph resident only and never dispatches an
    // equality request to it, but the standard call keeps the complete
    // project-owned adapter buildable against an unmodified checkout.
    auto searched = beam_search<LocalPoint, LocalSubsetRange, LocalIndex>(
        query_point, *atom->graph, points, LocalIndex(0), parameters);
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(searched.first.first.size());
    Result result;
    for (const auto& local_hit : searched.first.first) {
      const uint32_t global = static_cast<uint32_t>(
          atom->local_to_global[static_cast<size_t>(local_hit.first)]);
      if (global == spec.query_base_id) {
        ++result.removed_self;
        continue;
      }
      candidates.emplace_back(local_hit.second, global);
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) { return left.second == right.second; }),
        candidates.end());
    result.ids.fill(UINT32_MAX);
    result.size = std::min<uint32_t>(K, candidates.size());
    for (uint32_t i = 0; i < result.size; ++i) result.ids[i] = candidates[i].second;
    return result;
  }

 private:
  const Data& data_;
  GraphInventory& inventory_;
  uint32_t beam_;
  uint32_t pool_;
  double cut_;
};

#ifndef BOOLEANN_LAION_LOCAL_GRAPH_TYPES_ONLY
void audit_one(const Data& data, hnswlib::L2Space& space, const Spec& spec,
               const Result& result, Metrics& metrics) {
  const auto support = exact_support(data, spec);
  const Result truth = exact_top10(data, space, spec, support);
  const std::set<uint32_t> truth_ids(truth.ids.begin(), truth.ids.end());
  std::set<uint32_t> observed;
  metrics.denominator += K;
  metrics.removed_self += result.removed_self;
  metrics.underfull += result.size < K;
  for (uint32_t i = 0; i < result.size; ++i) {
    const uint32_t id = result.ids[i];
    if (!observed.insert(id).second) ++metrics.duplicate;
    if (id >= N) ++metrics.invalid;
    else {
      if (id == spec.query_base_id) ++metrics.forbidden_self;
      if (!matches(data, spec, id)) ++metrics.predicate_fail;
      if (truth_ids.count(id)) ++metrics.hits;
    }
  }
}

Metrics run_single(const Data& data, LocalGraphEngine& engine,
                   hnswlib::L2Space& space, const std::vector<Spec>& queries) {
  std::vector<Result> results(queries.size());
  for (const Spec& spec : queries) (void)engine.query(spec);
  const auto begin = Clock::now();
  for (size_t row = 0; row < queries.size(); ++row)
    results[row] = engine.query(queries[row]);
  const auto end = Clock::now();
  Metrics metrics;
  metrics.elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  metrics.queries = queries.size();
  for (size_t row = 0; row < queries.size(); ++row) {
    const Result repeat = engine.query(queries[row]);
    if (repeat.ids != results[row].ids || repeat.size != results[row].size)
      ++metrics.nondeterministic;
    audit_one(data, space, queries[row], results[row], metrics);
  }
  return metrics;
}

struct LocalForkShared {
  std::atomic<uint32_t> next{0};
  std::atomic<uint32_t> ready{0};
  std::atomic<uint32_t> go{0};
  std::atomic<uint32_t> done{0};
  std::array<Result, 200> results{};
  std::array<uint64_t, 200> service_ns{};
  std::array<uint64_t, 8> child_rss_kib{};
};

Metrics run_fork8(const Data& data, LocalGraphEngine& engine,
                  hnswlib::L2Space& space, const std::vector<Spec>& queries,
                  uint64_t& batch_ns, uint64_t& max_child_rss_kib) {
  require(queries.size() <= 200, "fork8 equality query cap differs");
  for (const Spec& spec : queries) (void)engine.query(spec);
  void* address = mmap(nullptr, sizeof(LocalForkShared), PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  require(address != MAP_FAILED, "fork8 shared mmap failed");
  auto* shared = new(address) LocalForkShared();
  static constexpr std::array<int, 8> cpus = {60, 61, 63, 64, 66, 67, 69, 70};
  std::vector<pid_t> children;
  for (uint32_t lane = 0; lane < 8; ++lane) {
    const pid_t pid = fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
      int status = 0;
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpus[lane], &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) status = 3;
      shared->ready.fetch_add(1, std::memory_order_release);
      while (shared->go.load(std::memory_order_acquire) == 0) sched_yield();
      if (status == 0) {
        for (;;) {
          const uint32_t row = shared->next.fetch_add(1, std::memory_order_relaxed);
          if (row >= queries.size()) break;
          const auto begin = Clock::now();
          shared->results[row] = engine.query(queries[row]);
          const auto end = Clock::now();
          shared->service_ns[row] =
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        }
      }
      shared->child_rss_kib[lane] = current_rss_kib(false);
      shared->done.fetch_add(1, std::memory_order_release);
      _exit(status);
    }
    children.push_back(pid);
  }
  while (shared->ready.load(std::memory_order_acquire) != 8) usleep(1000);
  const auto begin = Clock::now();
  shared->go.store(1, std::memory_order_release);
  while (shared->done.load(std::memory_order_acquire) != 8) usleep(100);
  const auto end = Clock::now();
  batch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  for (pid_t pid : children) {
    int state = 0;
    require(waitpid(pid, &state, 0) == pid && WIFEXITED(state) &&
                WEXITSTATUS(state) == 0,
            "fork8 local-graph worker failed");
  }
  Metrics metrics;
  metrics.queries = queries.size();
  for (size_t row = 0; row < queries.size(); ++row) {
    metrics.elapsed_ns += shared->service_ns[row];
    const Result repeat = engine.query(queries[row]);
    if (repeat.ids != shared->results[row].ids ||
        repeat.size != shared->results[row].size)
      ++metrics.nondeterministic;
    audit_one(data, space, queries[row], shared->results[row], metrics);
  }
  max_child_rss_kib = 0;
  for (uint64_t rss : shared->child_rss_kib)
    max_child_rss_kib = std::max(max_child_rss_kib, rss);
  shared->~LocalForkShared();
  munmap(address, sizeof(LocalForkShared));
  return metrics;
}

int local_main(int argc, char** argv) {
  const LocalOptions config = local_options(argc, argv);
  Data data = load_data(config.base, config.spmat, config.numeric, config.endpoints);
  std::vector<Spec> all = read_workload(config.queries, 0);
  std::vector<Spec> equality;
  for (const Spec& spec : all) {
    if (spec.family != Family::Equality) continue;
    if (equality.size() == config.limit_queries) break;
    equality.push_back(spec);
  }
  require(equality.size() == config.limit_queries,
          "equality limit exceeds workload rows");
  GraphInventory inventory = load_graph_inventory(data, config.graph_root);
  for (const Spec& spec : equality) {
    require(inventory.atoms[spec.primary] != nullptr,
            "selected v18 equality row uses rejected token graph");
    (void)exact_support(data, spec);
  }
  hnswlib::L2Space space(D);
  std::cout << "LOCAL_GRAPH_READY queries=" << equality.size()
            << " lanes=" << config.lanes << " pool=" << config.pool
            << " cut=" << config.cut
            << " current_rss_kib=" << current_rss_kib(false)
            << " peak_rss_kib=" << current_rss_kib(true)
            << " evidence_class=outcome_exposed_development_pilot" << std::endl;
  for (uint32_t beam : config.beams) {
    LocalGraphEngine engine(data, inventory, beam, config.pool, config.cut);
    uint64_t batch_ns = 0;
    uint64_t max_child_rss_kib = 0;
    Metrics metrics = config.lanes == 1
        ? run_single(data, engine, space, equality)
        : run_fork8(data, engine, space, equality, batch_ns, max_child_rss_kib);
    if (config.lanes == 1) batch_ns = metrics.elapsed_ns;
    const bool valid = metrics.invalid == 0 && metrics.predicate_fail == 0 &&
        metrics.duplicate == 0 && metrics.forbidden_self == 0 &&
        metrics.underfull == 0 && metrics.nondeterministic == 0;
    const double recall = double(metrics.hits) / metrics.denominator;
    std::cout << "BEAM_RESULT system=boole_local_graph family=equality"
              << " graph_degree=16 beam=" << beam << " pool=" << config.pool
              << " cut=" << config.cut << " queries=" << metrics.queries
              << " lanes=" << config.lanes
              << " service_sum_ns=" << metrics.elapsed_ns
              << " service_qps=" << metrics.qps()
              << " batch_wall_ns=" << batch_ns
              << " complete_batch_qps=" << double(metrics.queries) * 1e9 / batch_ns
              << " strict_recall=" << recall
              << " invalid=" << metrics.invalid
              << " predicate_fail=" << metrics.predicate_fail
              << " duplicate=" << metrics.duplicate
              << " forbidden_self=" << metrics.forbidden_self
              << " removed_self=" << metrics.removed_self
              << " underfull=" << metrics.underfull
              << " nondeterministic=" << metrics.nondeterministic
              << " validity=" << (valid ? "PASS" : "FAIL")
              << " recall_gate_0p9995=" << (recall >= 0.9995 ? "PASS" : "FAIL")
              << " current_rss_kib=" << current_rss_kib(false)
              << " peak_rss_kib=" << current_rss_kib(true)
              << " max_child_rss_kib=" << max_child_rss_kib << std::endl;
  }
  return 0;
}
#endif

}  // namespace

#if !defined(BOOLEANN_LAION_LOCAL_GRAPH_PILOT_NO_MAIN) && \
    !defined(BOOLEANN_LAION_LOCAL_GRAPH_TYPES_ONLY)
int main(int argc, char** argv) {
  try {
    return local_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
#endif
