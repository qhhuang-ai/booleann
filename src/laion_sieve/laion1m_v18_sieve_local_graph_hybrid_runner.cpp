// Development-only same-pool runner for the frozen LAION1M v18 workload.
//
// The adapter loads one official, unmodified SIEVE instance and one immutable
// inventory of Boole-ANN R16 local graphs before forking. Equality predicates
// use the frozen beam-256 local-graph route, conjunction uses SIEVE unchanged,
// and range/two-clause DNF use native support derivation plus exact top-k. All
// routes share the data mapping,
// request order, fork-8 shared-pull scheduler, leave-one-out oracle, and
// post-timing validity checks.
//
// This is development evidence.  The R16 graph inventory was not selected or
// charged in the earlier 16 MiB v18 manifest, so this runner must not be
// described as that manifest's budget-compliant joint design.

#define BOOLEANN_LAION_LOCAL_GRAPH_PILOT_NO_MAIN
#ifdef BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
#define BOOLEANN_LAION_LOCAL_GRAPH_TYPES_ONLY
#endif
#include "laion1m_v18_equality_local_graph_pilot.cpp"
#ifdef BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
#undef BOOLEANN_LAION_LOCAL_GRAPH_TYPES_ONLY
#endif

namespace {

#ifndef BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
struct HybridOptions {
  std::string base;
  std::string spmat;
  std::string endpoints;
  std::string history;
  std::string queries;
  std::string graph_root;
  std::array<std::string, ATTRIBUTES> numeric;
  uint32_t history_per_family = 0;
  uint32_t timed_per_family = 0;
  uint32_t cycles = 1;
  size_t m = 16;
  size_t ef_construction = 40;
  size_t ef_search = 200;
  size_t vector_budget = 100000;
  size_t cutoff = 512;
  size_t threads = 1;
  uint32_t lanes = 8;
  uint32_t graph_beam = 256;
  uint32_t graph_pool = 32;
  double graph_cut = 1.35;
};

HybridOptions hybrid_options(int argc, char** argv) {
  HybridOptions result;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(i + 1 < argc, "missing option value: " + key);
    const std::string value = argv[++i];
    if (key == "--base") result.base = value;
    else if (key == "--spmat") result.spmat = value;
    else if (key == "--endpoints") result.endpoints = value;
    else if (key == "--history") result.history = value;
    else if (key == "--queries") result.queries = value;
    else if (key == "--graph-root") result.graph_root = value;
    else if (key == "--numeric-similarity") result.numeric[0] = value;
    else if (key == "--numeric-original-width") result.numeric[1] = value;
    else if (key == "--numeric-original-height") result.numeric[2] = value;
    else if (key == "--history-per-family")
      result.history_per_family = std::stoul(value);
    else if (key == "--timed-per-family")
      result.timed_per_family = std::stoul(value);
    else if (key == "--cycles") result.cycles = std::stoul(value);
    else if (key == "--M") result.m = std::stoull(value);
    else if (key == "--ef-construction")
      result.ef_construction = std::stoull(value);
    else if (key == "--ef-search") result.ef_search = std::stoull(value);
    else if (key == "--index-vector-budget")
      result.vector_budget = std::stoull(value);
    else if (key == "--bitvector-cutoff") result.cutoff = std::stoull(value);
    else if (key == "--threads") result.threads = std::stoull(value);
    else if (key == "--lanes") result.lanes = std::stoul(value);
    else if (key == "--graph-beam") result.graph_beam = std::stoul(value);
    else if (key == "--graph-pool") result.graph_pool = std::stoul(value);
    else if (key == "--graph-cut") result.graph_cut = std::stod(value);
    else fail("unknown option: " + key);
  }
  require(!result.base.empty() && !result.spmat.empty() &&
              !result.endpoints.empty() && !result.history.empty() &&
              !result.queries.empty() && !result.graph_root.empty() &&
              std::all_of(result.numeric.begin(), result.numeric.end(),
                          [](const std::string& path) { return !path.empty(); }),
          "required same-pool input is missing");
  require(result.timed_per_family == 16 || result.timed_per_family == 200,
          "timed-per-family must select the frozen 16-row smoke or 200-row run");
  require(result.cycles == 1 && result.lanes == 8 && result.threads == 1,
          "same-pool contract requires cycles=1, fork lanes=8, build threads=1");
  require(result.m == 16 && result.ef_construction == 40 &&
              result.ef_search == 200 && result.vector_budget == 100000 &&
              result.cutoff == 512,
          "official SIEVE parameters differ from the frozen v18 control");
  require(result.graph_beam == 256 && result.graph_pool == 32 &&
              std::abs(result.graph_cut - 1.35) < 1e-12,
          "local-graph route differs from the frozen beam-256 gate");
  return result;
}
#endif

class SamePoolHybrid {
 public:
  SamePoolHybrid(SieveEngine& sieve, const Data& data,
                 GraphInventory& inventory, uint32_t beam, uint32_t pool,
                 double cut)
      : sieve_(sieve), data_(data), graph_(data, inventory, beam, pool, cut) {}

  void prewarm(const Spec& spec) {
    if (spec.family == Family::Conjunction)
      sieve_.prewarm(spec);
    else
      (void)query(spec);
  }

  Result query(const Spec& spec) {
    if (spec.family == Family::Equality) {
#ifdef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
      // Enumerating the complete posting and evaluating every eligible vector
      // makes equality top-k exact; no ANN-confidence heuristic is used as a
      // certificate.
      const std::vector<uint32_t> support = exact_support(data_, spec);
      Result result = exact_top10(data_, *sieve_.space, spec, support);
      result.removed_self = 1;  // exact_support verified and erased base ID
      return result;
#else
      return graph_.query(spec);
#endif
    }
    if (spec.family == Family::Range || spec.family == Family::Dnf2) {
      const std::vector<uint32_t> support = exact_support(data_, spec);
      Result result = exact_top10(data_, *sieve_.space, spec, support);
      result.removed_self = 1;
      return result;
    }
    return sieve_.query(spec);
  }

 private:
  SieveEngine& sieve_;
  const Data& data_;
  LocalGraphEngine graph_;
};

#ifndef BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
const char* hybrid_backend(Family family) {
  if (family == Family::Equality) {
#ifdef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
    return "complete_support_exact_topk";
#else
    return "local_graph_r16_beam256";
#endif
  }
  if (family == Family::Conjunction) return "official_sieve";
  return "native_exact_support_and_topk";
}

struct HybridForkShared {
  std::atomic<uint32_t> next{0};
  std::atomic<uint32_t> ready{0};
  std::atomic<uint32_t> go{0};
  std::atomic<uint32_t> done{0};
  std::array<Result, 800> results{};
  std::array<uint64_t, 800> service_ns{};
  std::array<uint64_t, 8> child_rss_kib{};
  std::array<uint64_t, 8> child_peak_rss_kib{};
  std::array<uint32_t, 8> child_queries{};
  std::array<int32_t, 8> child_status{};
};

void accumulate(Metrics& total, const Metrics& value) {
  total.elapsed_ns += value.elapsed_ns;
  total.queries += value.queries;
  total.hits += value.hits;
  total.denominator += value.denominator;
  total.invalid += value.invalid;
  total.predicate_fail += value.predicate_fail;
  total.duplicate += value.duplicate;
  total.forbidden_self += value.forbidden_self;
  total.removed_self += value.removed_self;
  total.underfull += value.underfull;
  total.nondeterministic += value.nondeterministic;
}

int run_same_pool_fork8(const Data& data, SieveEngine& sieve,
                        SamePoolHybrid& hybrid, const GraphInventory& inventory,
                        const std::vector<Spec>& queries) {
  require(queries.size() == 64 || queries.size() == 800,
          "same-pool batch must contain 16 or 200 rows per family");
  for (const Spec& spec : queries) hybrid.prewarm(spec);

  void* address = mmap(nullptr, sizeof(HybridForkShared),
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  require(address != MAP_FAILED, "same-pool shared result mmap failed");
  auto* shared = new(address) HybridForkShared();
  static constexpr std::array<int, 8> cpus =
      {60, 61, 63, 64, 66, 67, 69, 70};
  std::vector<pid_t> children;
  children.reserve(8);

  for (uint32_t lane = 0; lane < 8; ++lane) {
    const pid_t pid = fork();
    require(pid >= 0, "same-pool fork failed");
    if (pid == 0) {
      int32_t status = 0;
      try {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[lane], &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) status = 3;
        shared->ready.fetch_add(1, std::memory_order_release);
        while (shared->go.load(std::memory_order_acquire) == 0) sched_yield();
        if (status == 0) {
          for (;;) {
            const uint32_t row =
                shared->next.fetch_add(1, std::memory_order_relaxed);
            if (row >= queries.size()) break;
            const auto begin = Clock::now();
            shared->results[row] = hybrid.query(queries[row]);
            const auto end = Clock::now();
            shared->service_ns[row] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                    .count();
            ++shared->child_queries[lane];
          }
        }
      } catch (...) {
        status = 2;
      }
      shared->child_rss_kib[lane] = current_rss_kib(false);
      shared->child_peak_rss_kib[lane] = current_rss_kib(true);
      shared->child_status[lane] = status;
      shared->done.fetch_add(1, std::memory_order_release);
      _exit(status);
    }
    children.push_back(pid);
  }

  while (shared->ready.load(std::memory_order_acquire) != 8) usleep(1000);
  const auto batch_begin = Clock::now();
  shared->go.store(1, std::memory_order_release);
  while (shared->done.load(std::memory_order_acquire) != 8) usleep(100);
  const auto batch_end = Clock::now();
  const uint64_t batch_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end - batch_begin)
          .count();

  for (pid_t pid : children) {
    int state = 0;
    require(waitpid(pid, &state, 0) == pid, "same-pool waitpid failed");
    require(WIFEXITED(state) && WEXITSTATUS(state) == 0,
            "same-pool timed worker failed");
  }

  std::array<Metrics, 4> families{};
  for (size_t row = 0; row < queries.size(); ++row) {
    const Spec& spec = queries[row];
    const Result& result = shared->results[row];
    Metrics& metrics = families[static_cast<uint32_t>(spec.family)];
    ++metrics.queries;
    metrics.elapsed_ns += shared->service_ns[row];
    const Result repeat = hybrid.query(spec);
    if (repeat.ids != result.ids || repeat.size != result.size)
      ++metrics.nondeterministic;
    audit_one(data, *sieve.space, spec, result, metrics);
  }

  Metrics total;
  bool family_valid = true;
  bool equality_recall_gate = false;
  bool all_family_recall_gate = true;
  std::cout << std::setprecision(12);
  for (uint32_t family_index = 0; family_index < 4; ++family_index) {
    const Family family = static_cast<Family>(family_index);
    const Metrics& value = families[family_index];
    require(value.queries > 0 && value.denominator == value.queries * K,
            "family audit denominator differs");
    const double recall = double(value.hits) / value.denominator;
    const bool valid = value.invalid == 0 && value.predicate_fail == 0 &&
        value.duplicate == 0 && value.forbidden_self == 0 &&
        value.underfull == 0 && value.nondeterministic == 0;
    const bool recall_gate =
        (family == Family::Range || family == Family::Dnf2)
            ? recall == 1.0
            : recall >= 0.9995;
    if (family == Family::Equality) equality_recall_gate = recall_gate;
    family_valid = family_valid && valid;
    all_family_recall_gate = all_family_recall_gate && recall_gate;
    std::cout << "FORK_BLOCK system=boole_sieve_graph_hybrid"
              << " family=" << family_name(family)
              << " backend=" << hybrid_backend(family)
              << " queries=" << value.queries
              << " core_service_sum_ns=" << value.elapsed_ns
              << " core_service_qps=" << value.qps()
              << " strict_recall=" << recall
              << " invalid=" << value.invalid
              << " predicate_fail=" << value.predicate_fail
              << " duplicate=" << value.duplicate
              << " forbidden_self=" << value.forbidden_self
              << " removed_self=" << value.removed_self
              << " underfull=" << value.underfull
              << " nondeterministic=" << value.nondeterministic
              << " validity=" << (valid ? "PASS" : "FAIL")
              << " family_recall_gate=" << (recall_gate ? "PASS" : "FAIL");
    if (family == Family::Equality)
      std::cout << " recall_gate_0p9995="
                << (equality_recall_gate ? "PASS" : "FAIL");
    std::cout << std::endl;
    accumulate(total, value);
  }

  uint64_t max_child_rss_kib = 0;
  uint64_t max_child_peak_rss_kib = 0;
  uint64_t aggregate_child_rss_kib_unadjusted = 0;
  uint64_t aggregate_child_peak_rss_kib_unadjusted = 0;
  uint64_t child_query_sum = 0;
  for (uint32_t lane = 0; lane < 8; ++lane) {
    require(shared->child_status[lane] == 0,
            "same-pool child status differs after successful wait");
    max_child_rss_kib =
        std::max(max_child_rss_kib, shared->child_rss_kib[lane]);
    max_child_peak_rss_kib =
        std::max(max_child_peak_rss_kib, shared->child_peak_rss_kib[lane]);
    aggregate_child_rss_kib_unadjusted += shared->child_rss_kib[lane];
    aggregate_child_peak_rss_kib_unadjusted +=
        shared->child_peak_rss_kib[lane];
    child_query_sum += shared->child_queries[lane];
  }
  require(child_query_sum == queries.size(),
          "shared-pull child query census differs");

  const bool total_valid = family_valid && total.invalid == 0 &&
      total.predicate_fail == 0 && total.duplicate == 0 &&
      total.forbidden_self == 0 && total.underfull == 0 &&
      total.nondeterministic == 0;
  const bool gate = total_valid && all_family_recall_gate;
  std::cout << "FORK_SUMMARY system=boole_sieve_graph_hybrid"
            << " assignment=persistent_shared_pull"
            << " request_order=file_order lanes=8 queries=" << total.queries
            << " batch_wall_ns=" << batch_ns
            << " complete_batch_qps=" << double(total.queries) * 1e9 / batch_ns
            << " service_sum_ns=" << total.elapsed_ns
            << " strict_recall=" << double(total.hits) / total.denominator
            << " invalid=" << total.invalid
            << " predicate_fail=" << total.predicate_fail
            << " duplicate=" << total.duplicate
            << " forbidden_self=" << total.forbidden_self
            << " removed_self=" << total.removed_self
            << " underfull=" << total.underfull
            << " nondeterministic=" << total.nondeterministic
            << " validity=" << (total_valid ? "PASS" : "FAIL")
            << " equality_recall_gate_0p9995="
            << (equality_recall_gate ? "PASS" : "FAIL")
            << " all_family_recall_gate="
            << (all_family_recall_gate ? "PASS" : "FAIL")
            << " experiment_gate=" << (gate ? "PASS" : "STOP")
            << " parent_rss_kib=" << current_rss_kib(false)
            << " parent_peak_rss_kib=" << current_rss_kib(true)
            << " max_child_rss_kib=" << max_child_rss_kib
            << " max_child_peak_rss_kib=" << max_child_peak_rss_kib
            << " aggregate_child_rss_kib_unadjusted="
            << aggregate_child_rss_kib_unadjusted
            << " aggregate_child_peak_rss_kib_unadjusted="
            << aggregate_child_peak_rss_kib_unadjusted
            << " sieve_metadata_bytes=" << sieve.metadata_bytes
            << " sieve_memberships=" << sieve.memberships
            << " graph_file_bytes=" << inventory.graph_file_bytes
            << " graph_subset_bytes=" << inventory.subset_file_bytes
            << " graph_dense_bytes=" << inventory.dense_graph_bytes
            << " graph_indexed_memberships=" << inventory.indexed_memberships
            << " sieve_instances=1 graph_inventory_instances=1 data_instances=1"
            << " graph_not_in_v18_16mib_manifest=true"
            << " baseline_core_modified=false development_only=true"
            << std::endl;

  shared->~HybridForkShared();
  munmap(address, sizeof(HybridForkShared));
  return gate ? 0 : 2;
}

int hybrid_main(int argc, char** argv) {
  const HybridOptions config = hybrid_options(argc, argv);
  Data data =
      load_data(config.base, config.spmat, config.numeric, config.endpoints);
  std::vector<Spec> history =
      read_workload(config.history, config.history_per_family);
  std::vector<Spec> queries =
      read_workload(config.queries, config.timed_per_family);
  require(queries.size() == uint64_t(config.timed_per_family) * 4,
          "same-pool family census differs");
  std::array<uint32_t, 4> family_counts{};
  for (const Spec& spec : history) (void)exact_support(data, spec);
  for (const Spec& spec : queries) {
    (void)exact_support(data, spec);
    ++family_counts[static_cast<uint32_t>(spec.family)];
  }
  for (uint32_t count : family_counts)
    require(count == config.timed_per_family,
            "same-pool workload is not family balanced");

  std::cout << "SAMEPOOL_WORKLOAD_READY history=" << history.size()
            << " timed=" << queries.size()
            << " per_family=" << config.timed_per_family
            << " estimand=complete_batch_qps_and_family_service_sum"
            << " aggregation=queries_then_family_then_complete_batch"
            << " oracle=strict_set_membership_leave_one_out"
            << " evidence_class=outcome_exposed_development_same_pool"
            << std::endl;

  SieveEngine sieve(data, history, config.m, config.ef_construction,
                    config.ef_search, config.vector_budget, config.cutoff,
                    config.threads);
  GraphInventory inventory = load_graph_inventory(
      data, config.graph_root, GraphInventoryMode::CleanAll200);
  for (const Spec& spec : queries) {
    if (spec.family == Family::Equality)
      require(inventory.atoms[spec.primary] != nullptr,
              "timed equality row uses rejected/missing graph atom");
  }
  SamePoolHybrid hybrid(sieve, data, inventory, config.graph_beam,
                        config.graph_pool, config.graph_cut);
  std::cout << "SAMEPOOL_HYBRID_READY equality=local_graph_r16_beam256"
            << " conjunction=official_sieve"
            << " range=native_exact_support_and_topk"
            << " dnf2=native_exact_support_and_topk"
            << " exact_support_derivation_timed=true"
            << " formula_specific_materialization=false"
            << " warm=route_native_single_pass"
            << " lanes=8 cpu_set=60,61,63,64,66,67,69,70"
            << " current_rss_kib=" << current_rss_kib(false)
            << " peak_rss_kib=" << current_rss_kib(true)
            << " graph_not_in_v18_16mib_manifest=true"
            << " baseline_core_modified=false development_only=true"
            << std::endl;
  return run_same_pool_fork8(data, sieve, hybrid, inventory, queries);
}
#endif

}  // namespace

#ifndef BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
int main(int argc, char** argv) {
  try {
    return hybrid_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
#endif
