// Prospective LAION1M v19 same-resident paired formal runner.
//
// This project-owned runner composes the released SIEVE implementation through
// the existing adapter; it does not modify baseline source.  The S arm owns a
// saturated 1,000,000-vector-budget SIEVE instance.  The H arm owns a distinct
// 100,000-vector-budget SIEVE instance plus the clean local-graph inventory.
// Both remain resident in one parent process.  The one manifest-injected ef is
// used by every S route and by H's only SIEVE route (conjunction).
//
// This file is a formal skeleton until a later complete manifest and activation
// gate freeze its paths, query pool, ef, resource census, and binary.  It must
// not be used to generate or select those values.

#ifndef BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS
#define BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS "evidence_class=frozen_formal"
#define BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS_DEFINED_HERE
#endif
#define BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
#include "laion1m_v18_sieve_local_graph_hybrid_runner.cpp"
#undef BOOLEANN_LAION_SAME_POOL_TYPES_ONLY
#ifdef BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS_DEFINED_HERE
#undef BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS
#undef BOOLEANN_LAION_GRAPH_EVIDENCE_CLASS_DEFINED_HERE
#endif

#include "laion1m_paired_formal_block_durability.h"
#include "laion1m_v19_same_resident_formal_protocol.h"

#include <cerrno>
#include <charconv>
#include <filesystem>
#include <numeric>
#include <thread>

#ifndef BOOLEANN_LAION_HEARTBEAT_RECORD
#define BOOLEANN_LAION_HEARTBEAT_RECORD \
  "FORMAL_HEARTBEAT schema=laion1m_v19_formal_heartbeat_v1"
#endif
#ifndef BOOLEANN_LAION_QUERY_AUDIT_SCHEMA
#define BOOLEANN_LAION_QUERY_AUDIT_SCHEMA \
  "laion1m-v19-formal-query-audit/v1"
#endif
#ifndef BOOLEANN_LAION_BLOCK_SUMMARY_SCHEMA
#define BOOLEANN_LAION_BLOCK_SUMMARY_SCHEMA \
  "laion1m-v19-formal-block-summary/v1"
#endif
#ifndef BOOLEANN_LAION_RESOURCE_SCHEMA
#define BOOLEANN_LAION_RESOURCE_SCHEMA \
  "laion1m_v19_same_resident_resource_v1"
#endif
#ifndef BOOLEANN_LAION_ACTIVATION_SCHEMA
#define BOOLEANN_LAION_ACTIVATION_SCHEMA \
  "laion1m_v19_same_resident_activation_v1"
#endif
#ifndef BOOLEANN_LAION_FINAL_SCHEMA
#define BOOLEANN_LAION_FINAL_SCHEMA \
  "laion1m_v19_same_resident_final_v1"
#endif

namespace laion1m_v19_same_resident_paired_formal_runner {

namespace fs = std::filesystem;
namespace protocol = laion1m_v19_same_resident_formal_protocol;

constexpr uint32_t kMaximumQueries = 800;

struct Options {
  std::string manifest;
  std::string state_id;
  std::string base;
  std::string spmat;
  std::string endpoints;
  std::string history;
  std::string queries;
  std::string graph_root;
  std::array<std::string, ATTRIBUTES> numeric;
  fs::path raw_directory;
  fs::path ack_directory;
  std::array<int, protocol::kLanes> cpus{};
  size_t frozen_ef = 0;
  uint32_t timed_per_family = 0;
  uint64_t query_permutation_seed = 0;
  uint64_t query_permutation_fingerprint = 0;
  size_t minimum_recall_numerator = 0;
  size_t minimum_recall_denominator = 0;
  size_t expected_reference_partitions = 0;
  size_t expected_reference_memberships = 0;
  size_t expected_saturated_partitions = 0;
  size_t expected_saturated_memberships = 0;
  uint64_t expected_graph_file_bytes = 0;
  uint64_t expected_graph_subset_bytes = 0;
  uint64_t expected_graph_dense_bytes = 0;
  uint64_t expected_graph_memberships = 0;
  uint64_t graph_authority_persistent_bytes = 0;
  uint64_t graph_authority_runtime_bytes = 0;
  std::chrono::milliseconds ack_timeout{0};
  std::chrono::milliseconds ack_poll{0};
};

uint64_t parse_u64(const std::string& value, const std::string& label,
                   int base = 10) {
  size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed, base);
  require(consumed == value.size(), label + " is not an exact unsigned integer");
  return parsed;
}

std::array<int, protocol::kLanes> parse_cpus(const std::string& value) {
  std::array<int, protocol::kLanes> cpus{};
  std::istringstream input(value);
  std::string field;
  size_t count = 0;
  while (std::getline(input, field, ',')) {
    require(count < cpus.size() && !field.empty(),
            "cpu-list must contain exactly eight fields");
    size_t consumed = 0;
    const long parsed = std::stol(field, &consumed);
    require(consumed == field.size() && parsed >= 0 && parsed < CPU_SETSIZE,
            "cpu-list contains an invalid CPU");
    cpus[count++] = static_cast<int>(parsed);
  }
  require(count == cpus.size(), "cpu-list must contain exactly eight CPUs");
  std::array<int, protocol::kLanes> sorted = cpus;
  std::sort(sorted.begin(), sorted.end());
  require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
          "cpu-list must be unique");
  return cpus;
}

Options parse_options(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    require(key != "--self-test", "use the separate query-free protocol fixture");
    require(index + 1 < argc, "missing option value: " + key);
    const std::string value = argv[++index];
    if (key == "--manifest") result.manifest = value;
    else if (key == "--state-id") result.state_id = value;
    else if (key == "--base") result.base = value;
    else if (key == "--spmat") result.spmat = value;
    else if (key == "--endpoints") result.endpoints = value;
    else if (key == "--history") result.history = value;
    else if (key == "--queries") result.queries = value;
    else if (key == "--graph-root") result.graph_root = value;
    else if (key == "--numeric-similarity") result.numeric[0] = value;
    else if (key == "--numeric-original-width") result.numeric[1] = value;
    else if (key == "--numeric-original-height") result.numeric[2] = value;
    else if (key == "--raw-directory") result.raw_directory = value;
    else if (key == "--ack-directory") result.ack_directory = value;
    else if (key == "--cpu-list") result.cpus = parse_cpus(value);
    else if (key == "--frozen-ef")
      result.frozen_ef = parse_u64(value, "frozen-ef");
    else if (key == "--timed-per-family")
      result.timed_per_family = parse_u64(value, "timed-per-family");
    else if (key == "--query-permutation-seed")
      result.query_permutation_seed =
          parse_u64(value, "query-permutation-seed", 0);
    else if (key == "--query-permutation-fingerprint")
      result.query_permutation_fingerprint =
          parse_u64(value, "query-permutation-fingerprint", 0);
    else if (key == "--minimum-recall-numerator")
      result.minimum_recall_numerator =
          parse_u64(value, "minimum-recall-numerator");
    else if (key == "--minimum-recall-denominator")
      result.minimum_recall_denominator =
          parse_u64(value, "minimum-recall-denominator");
    else if (key == "--expected-reference-partitions")
      result.expected_reference_partitions =
          parse_u64(value, "expected-reference-partitions");
    else if (key == "--expected-reference-memberships")
      result.expected_reference_memberships =
          parse_u64(value, "expected-reference-memberships");
    else if (key == "--expected-saturated-partitions")
      result.expected_saturated_partitions =
          parse_u64(value, "expected-saturated-partitions");
    else if (key == "--expected-saturated-memberships")
      result.expected_saturated_memberships =
          parse_u64(value, "expected-saturated-memberships");
    else if (key == "--expected-graph-file-bytes")
      result.expected_graph_file_bytes =
          parse_u64(value, "expected-graph-file-bytes");
    else if (key == "--expected-graph-subset-bytes")
      result.expected_graph_subset_bytes =
          parse_u64(value, "expected-graph-subset-bytes");
    else if (key == "--expected-graph-dense-bytes")
      result.expected_graph_dense_bytes =
          parse_u64(value, "expected-graph-dense-bytes");
    else if (key == "--expected-graph-memberships")
      result.expected_graph_memberships =
          parse_u64(value, "expected-graph-memberships");
    else if (key == "--graph-authority-persistent-bytes")
      result.graph_authority_persistent_bytes =
          parse_u64(value, "graph-authority-persistent-bytes");
    else if (key == "--graph-authority-runtime-bytes")
      result.graph_authority_runtime_bytes =
          parse_u64(value, "graph-authority-runtime-bytes");
    else if (key == "--ack-timeout-seconds")
      result.ack_timeout = std::chrono::seconds(
          parse_u64(value, "ack-timeout-seconds"));
    else if (key == "--ack-poll-milliseconds")
      result.ack_poll = std::chrono::milliseconds(
          parse_u64(value, "ack-poll-milliseconds"));
    else
      fail("unknown option: " + key);
  }

  require(!result.manifest.empty() && !result.state_id.empty() &&
              !result.base.empty() && !result.spmat.empty() &&
              !result.endpoints.empty() && !result.history.empty() &&
              !result.queries.empty() && !result.graph_root.empty() &&
              std::all_of(result.numeric.begin(), result.numeric.end(),
                          [](const std::string& path) { return !path.empty(); }),
          "manifest-injected input/path option is missing");
  for (const std::string* path : {&result.manifest, &result.base, &result.spmat,
                                  &result.endpoints, &result.history,
                                  &result.queries, &result.graph_root,
                                  &result.numeric[0], &result.numeric[1],
                                  &result.numeric[2]})
    require(fs::path(*path).is_absolute(),
            "all manifest-injected input paths must be absolute");
  require(result.raw_directory.is_absolute() && result.ack_directory.is_absolute(),
          "raw and ACK directories must be absolute");
  laion_paired_formal_block_durability::validate_state_id(result.state_id);
  require(result.frozen_ef == protocol::kFrozenEf,
          "formal frozen-ef must equal the upstream-selected 800");
  require(result.timed_per_family == protocol::kTimedPerFamily &&
              uint64_t(result.timed_per_family) * 4U == kMaximumQueries,
          "formal workload must contain exactly q800/200 per family");
  require(result.query_permutation_seed != 0 &&
              result.query_permutation_fingerprint != 0,
          "query permutation seed/fingerprint must be manifest-injected");
  require(result.minimum_recall_numerator ==
                  protocol::kMinimumRecallNumerator &&
              result.minimum_recall_denominator ==
                  protocol::kMinimumRecallDenominator,
          "formal recall threshold must equal 9995/10000");
  require(result.expected_reference_partitions ==
                  protocol::kExpectedReferencePartitions &&
              result.expected_reference_memberships ==
                  protocol::kExpectedReferenceMemberships &&
              result.expected_saturated_partitions ==
                  protocol::kExpectedSaturatedPartitions &&
              result.expected_saturated_memberships ==
                  protocol::kExpectedSaturatedMemberships &&
              result.expected_graph_file_bytes ==
                  protocol::kExpectedGraphFileBytes &&
              result.expected_graph_subset_bytes ==
                  protocol::kExpectedGraphSubsetBytes &&
              result.expected_graph_dense_bytes ==
                  protocol::kExpectedGraphDenseBytes &&
              result.expected_graph_memberships ==
                  protocol::kExpectedGraphMemberships &&
              result.graph_authority_persistent_bytes ==
                  protocol::kGraphPersistentBytes &&
              result.graph_authority_runtime_bytes ==
                  protocol::kGraphDesignPayloadRuntimeBytes,
          "formal resource authority/census differs from frozen grid");
  require(result.ack_timeout == protocol::kAckTimeout &&
              result.ack_poll == protocol::kAckPoll,
          "formal ACK deadline/poll interval differs from frozen contract");
  return result;
}

uint64_t rss_anon_bytes() {
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("RssAnon:", 0) != 0) continue;
    std::istringstream fields(line.substr(8));
    uint64_t kib = 0;
    fields >> kib;
    require(!fields.fail(), "cannot parse RssAnon");
    return kib * 1024ULL;
  }
  fail("RssAnon absent from /proc/self/status");
}

uint64_t monotone_delta(uint64_t after, uint64_t before,
                        const std::string& label) {
  require(after >= before, label + " RssAnon decreased across resident load");
  return after - before;
}

enum class HeartbeatPhase : uint32_t {
  InputContracts,
  BuildReference,
  LoadGraph,
  BuildSaturated,
  ResourceGate,
  QueryActivation,
  Prewarm,
  TimedBlock,
  PostWallAudit,
  PublishBlock,
  AwaitAdapter,
  Complete,
};

const char* heartbeat_phase_name(HeartbeatPhase phase) {
  switch (phase) {
    case HeartbeatPhase::InputContracts: return "input_contracts";
    case HeartbeatPhase::BuildReference: return "build_reference_sieve";
    case HeartbeatPhase::LoadGraph: return "load_clean_graph";
    case HeartbeatPhase::BuildSaturated: return "build_saturated_sieve";
    case HeartbeatPhase::ResourceGate: return "resource_gate";
    case HeartbeatPhase::QueryActivation: return "query_activation";
    case HeartbeatPhase::Prewarm: return "prewarm";
    case HeartbeatPhase::TimedBlock: return "timed_block";
    case HeartbeatPhase::PostWallAudit: return "post_wall_audit";
    case HeartbeatPhase::PublishBlock: return "publish_block";
    case HeartbeatPhase::AwaitAdapter: return "await_adapter_ack_release";
    case HeartbeatPhase::Complete: return "complete";
  }
  return "invalid";
}

class ParentHeartbeat {
 public:
  ParentHeartbeat() : started_(Clock::now()), thread_([this] { loop(); }) {}
  ~ParentHeartbeat() { stop_before_fork(); }

  void stop_before_fork() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    live_.store(false, std::memory_order_release);
  }
  bool thread_live() const {
    return live_.load(std::memory_order_acquire);
  }

  void phase(HeartbeatPhase value, size_t block = 0) {
    block_.store(block, std::memory_order_release);
    phase_.store(static_cast<uint32_t>(value), std::memory_order_release);
  }
  void completed(size_t blocks, uint64_t wall_ns, uint64_t strict_hits,
                 uint64_t strict_denominator, uint64_t tie_hits,
                 uint64_t tie_denominator) {
    latest_wall_ns_.store(wall_ns, std::memory_order_release);
    latest_strict_hits_.store(strict_hits, std::memory_order_release);
    latest_strict_denominator_.store(strict_denominator,
                                     std::memory_order_release);
    latest_tie_hits_.store(tie_hits, std::memory_order_release);
    latest_tie_denominator_.store(tie_denominator,
                                  std::memory_order_release);
    completed_.store(blocks, std::memory_order_release);
  }
  void note_contract_io() {
    contract_io_.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  void loop() {
    while (!stop_.load(std::memory_order_acquire)) {
      for (size_t second = 0; second < 25; ++second) {
        if (stop_.load(std::memory_order_acquire)) return;
        ::sleep(1);
      }
      const auto phase = static_cast<HeartbeatPhase>(
          phase_.load(std::memory_order_acquire));
      const uint64_t elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - started_)
              .count();
      std::ostringstream line;
      line << BOOLEANN_LAION_HEARTBEAT_RECORD
           << " utc_epoch_s=" << std::time(nullptr)
           << " pid=" << ::getpid()
           << " phase=" << heartbeat_phase_name(phase)
           << " block=" << block_.load(std::memory_order_acquire)
           << " completed_blocks=" << completed_.load(std::memory_order_acquire)
           << " total_blocks=" << protocol::kBlocks
           << " elapsed_s=" << elapsed
           << " latest_wall_ns=" << latest_wall_ns_.load(std::memory_order_acquire)
           << " latest_strict_hits="
           << latest_strict_hits_.load(std::memory_order_acquire)
           << " latest_strict_denominator="
           << latest_strict_denominator_.load(std::memory_order_acquire)
           << " latest_tie_hits="
           << latest_tie_hits_.load(std::memory_order_acquire)
           << " latest_tie_denominator="
           << latest_tie_denominator_.load(std::memory_order_acquire)
           << " current_rss_kib=" << current_rss_kib(false)
           << " peak_rss_kib=" << current_rss_kib(true)
           << " contract_io_calls="
           << contract_io_.load(std::memory_order_acquire)
           << " eta=unknown\n";
      const std::string record = line.str();
      size_t offset = 0;
      while (offset < record.size()) {
        const ssize_t written = ::write(
            STDERR_FILENO, record.data() + offset, record.size() - offset);
        if (written > 0) {
          offset += static_cast<size_t>(written);
          continue;
        }
        if (written < 0 && errno == EINTR) continue;
        break;
      }
    }
  }

  Clock::time_point started_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> live_{true};
  std::atomic<uint32_t> phase_{
      static_cast<uint32_t>(HeartbeatPhase::InputContracts)};
  std::atomic<size_t> block_{0};
  std::atomic<size_t> completed_{0};
  std::atomic<uint64_t> latest_wall_ns_{0};
  std::atomic<uint64_t> latest_strict_hits_{0};
  std::atomic<uint64_t> latest_strict_denominator_{0};
  std::atomic<uint64_t> latest_tie_hits_{0};
  std::atomic<uint64_t> latest_tie_denominator_{0};
  std::atomic<uint64_t> contract_io_{0};
  std::thread thread_;
};

size_t process_thread_count() {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator("/proc/self/task")) {
    (void)entry;
    ++count;
  }
  return count;
}

struct HnswCensus {
  uint64_t root_memberships = 0;
  uint64_t partition_memberships = 0;
  uint64_t root_index_bytes = 0;
  uint64_t partition_index_bytes = 0;
  size_t partitions = 0;
};

HnswCensus census(SieveEngine& engine) {
  require(engine.index && engine.index->_root && engine.index->_root->_hnsw,
          "SIEVE root HNSW is absent");
  HnswCensus result;
  result.root_memberships = engine.index->_root->_hnsw->cur_element_count.load();
  require(result.root_memberships == N &&
              engine.index->_root->_predicate.cardinality() == N,
          "SIEVE root membership census differs");
  result.root_index_bytes = engine.index->_root->_hnsw->indexFileSize();
  result.partitions = engine.index->_nodes.size();
  for (auto* node : engine.index->_nodes) {
    require(node && node->_hnsw, "SIEVE partition HNSW is absent");
    const uint64_t actual = node->_hnsw->cur_element_count.load();
    require(actual == node->_predicate.cardinality(),
            "SIEVE partition predicate/actual memberships differ");
    result.partition_memberships += actual;
    result.partition_index_bytes += node->_hnsw->indexFileSize();
  }
  return result;
}

protocol::Family protocol_family(Family family) {
  return static_cast<protocol::Family>(static_cast<uint32_t>(family));
}

Result execute(protocol::Arm arm, SieveEngine& saturated,
               SamePoolHybrid& hybrid, const Spec& spec) {
  return arm == protocol::Arm::SaturatedSieve ? saturated.query(spec)
                                               : hybrid.query(spec);
}

class FrozenDeadline {
 public:
  template <class Rep, class Period>
  FrozenDeadline(std::chrono::duration<Rep, Period> duration,
                 std::string stage)
      : deadline_(Clock::now() + duration), stage_(std::move(stage)) {}

  void check() const {
    bci_pass_boundary::require_not_stopped();
    require(Clock::now() < deadline_, stage_ + " exceeded frozen deadline");
  }

 private:
  Clock::time_point deadline_;
  std::string stage_;
};

void prewarm(protocol::Arm arm, SieveEngine& saturated,
             SamePoolHybrid& hybrid, const std::vector<Spec>& queries,
             const FrozenDeadline& deadline) {
  for (const Spec& spec : queries) {
    deadline.check();
    if (arm == protocol::Arm::SaturatedSieve)
      saturated.prewarm(spec);
    else
      hybrid.prewarm(spec);
    deadline.check();
  }
}

struct QueryAudit {
  Metrics metrics;
  uint64_t tie_hits = 0;
  uint64_t tie_denominator = K;
  uint32_t kth_distance_bits = 0;
  std::vector<uint32_t> strict_truth_ids;
  std::vector<uint32_t> observed_ids;
  std::vector<std::optional<uint32_t>> observed_distance_bits;
};

QueryAudit audit_query(const Data& data, hnswlib::L2Space& space,
                       const Spec& spec, const Result& result,
                       bool nondeterministic,
                       const FrozenDeadline& deadline) {
  deadline.check();
  require(result.size <= K, "result size exceeds fixed top-k storage");
  const std::vector<uint32_t> support = exact_support(data, spec);
  deadline.check();
  const Result truth = exact_top10(data, space, spec, support);
  deadline.check();
  require(truth.size == K, "exact truth is underfull");
  QueryAudit audit;
  audit.metrics.queries = 1;
  audit.metrics.denominator = K;
  audit.metrics.removed_self = result.removed_self;
  audit.metrics.underfull = result.size < K;
  audit.metrics.nondeterministic = nondeterministic;
  audit.strict_truth_ids.assign(truth.ids.begin(), truth.ids.begin() + K);
  std::sort(audit.strict_truth_ids.begin(), audit.strict_truth_ids.end());
  require(std::adjacent_find(audit.strict_truth_ids.begin(),
                             audit.strict_truth_ids.end()) ==
              audit.strict_truth_ids.end(),
          "exact strict truth contains duplicate IDs");

  const auto distance = space.get_dist_func();
  void* parameter = space.get_dist_func_param();
  const float* query = data.base + uint64_t(spec.query_base_id) * D;
  float kth_distance = 0.0f;
  for (const uint32_t id : audit.strict_truth_ids) {
    deadline.check();
    require(id < N, "exact strict truth contains an invalid ID");
    kth_distance = std::max(
        kth_distance,
        distance(query, data.base + uint64_t(id) * D, parameter));
  }
  audit.kth_distance_bits = protocol::float_bits(kth_distance);

  std::set<uint32_t> observed;
  for (uint32_t index = 0; index < result.size; ++index) {
    deadline.check();
    const uint32_t id = result.ids[index];
    audit.observed_ids.push_back(id);
    if (!observed.insert(id).second) ++audit.metrics.duplicate;
    if (id >= N) {
      ++audit.metrics.invalid;
      audit.observed_distance_bits.push_back(std::nullopt);
      continue;
    }
    if (id == spec.query_base_id) ++audit.metrics.forbidden_self;
    if (!matches(data, spec, id)) ++audit.metrics.predicate_fail;
    audit.metrics.hits += std::binary_search(
        audit.strict_truth_ids.begin(), audit.strict_truth_ids.end(), id);
    const float observed_distance =
        distance(query, data.base + uint64_t(id) * D, parameter);
    audit.observed_distance_bits.push_back(
        protocol::float_bits(observed_distance));
    audit.tie_hits += observed_distance <= kth_distance;
  }
  deadline.check();
  return audit;
}

void add_metrics(Metrics& target, const Metrics& value) {
  target.elapsed_ns += value.elapsed_ns;
  target.queries += value.queries;
  target.hits += value.hits;
  target.denominator += value.denominator;
  target.invalid += value.invalid;
  target.predicate_fail += value.predicate_fail;
  target.duplicate += value.duplicate;
  target.forbidden_self += value.forbidden_self;
  target.removed_self += value.removed_self;
  target.underfull += value.underfull;
  target.nondeterministic += value.nondeterministic;
}

bool validity_passes(const Metrics& metrics) {
  return metrics.invalid == 0 && metrics.predicate_fail == 0 &&
      metrics.duplicate == 0 && metrics.forbidden_self == 0 &&
      metrics.underfull == 0 && metrics.nondeterministic == 0;
}

bool recall_fraction_passes(uint64_t hits, uint64_t denominator,
                            const Options& options) {
  require(denominator > 0, "zero recall denominator");
  return static_cast<unsigned __int128>(hits) *
             options.minimum_recall_denominator >=
         static_cast<unsigned __int128>(denominator) *
             options.minimum_recall_numerator;
}

std::string uint_array_json(const std::vector<uint32_t>& values) {
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index];
  }
  output << ']';
  return output.str();
}

std::string optional_uint_array_json(
    const std::vector<std::optional<uint32_t>>& values) {
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    if (values[index].has_value())
      output << *values[index];
    else
      output << "null";
  }
  output << ']';
  return output.str();
}

struct SharedBlock {
  std::atomic<uint32_t> next{0};
  std::atomic<uint32_t> ready{0};
  std::atomic<uint32_t> go{0};
  std::atomic<uint32_t> done{0};
  std::atomic<uint32_t> post_go{0};
  std::atomic<uint32_t> post_done{0};
  std::array<Result, kMaximumQueries> results{};
  std::array<uint64_t, kMaximumQueries> service_ns{};
  std::array<uint32_t, kMaximumQueries> worker_lane{};
  std::array<uint32_t, protocol::kLanes> child_queries{};
  std::array<int32_t, protocol::kLanes> child_status{};
  std::array<uint64_t, protocol::kLanes> child_timed_io_calls{};
  std::array<uint64_t, protocol::kLanes> child_rss_kib{};
  std::array<uint64_t, protocol::kLanes> child_peak_rss_kib{};
};

class ForkChildrenGuard {
 public:
  explicit ForkChildrenGuard(std::vector<pid_t>& children)
      : children_(children) {}
  ~ForkChildrenGuard() {
    if (reaped_) return;
    for (const pid_t child : children_)
      if (child > 0) (void)::kill(child, SIGKILL);
    for (const pid_t child : children_) {
      if (child <= 0) continue;
      int state = 0;
      while (::waitpid(child, &state, 0) < 0 && errno == EINTR) {}
    }
  }
  void mark_reaped() { reaped_ = true; }

 private:
  std::vector<pid_t>& children_;
  bool reaped_ = false;
};

void wait_for_counter(std::atomic<uint32_t>& counter, uint32_t target,
                      const FrozenDeadline& deadline,
                      useconds_t poll_microseconds) {
  while (counter.load(std::memory_order_acquire) != target) {
    deadline.check();
    ::usleep(poll_microseconds);
  }
  deadline.check();
}

struct FamilyAudit {
  Metrics strict;
  uint64_t tie_hits = 0;
  uint64_t tie_denominator = 0;
};

struct BlockOutcome {
  size_t block = 0;
  protocol::Arm arm = protocol::Arm::SaturatedSieve;
  uint64_t wall_ns = 0;
  std::array<FamilyAudit, 4> families{};
  Metrics strict;
  uint64_t tie_hits = 0;
  uint64_t tie_denominator = 0;
  bool validity_gate = false;
  bool recall_gate = false;
  bool block_gate = false;
  std::string results_payload;
  std::string summary_payload;
};

BlockOutcome run_block(size_t block, const Options& options, const Data& data,
                       SieveEngine& saturated, SieveEngine& reference,
                       SamePoolHybrid& hybrid,
                       const std::vector<Spec>& queries,
                       const std::vector<uint32_t>& original_rows,
                       ParentHeartbeat& heartbeat) {
  const protocol::Arm arm = protocol::scheduled_arm(block);
  require(!heartbeat.thread_live() && process_thread_count() == 1,
          "formal runner must be single-threaded before every worker fork");
  require(queries.size() == original_rows.size() && !queries.empty() &&
              queries.size() <= kMaximumQueries,
          "formal query permutation extent differs");
  // Reassert the one injected ef before every prewarm and fork.  H has exactly
  // one SIEVE route: conjunction through `reference`.
  saturated.index->setEf(options.frozen_ef);
  reference.index->setEf(options.frozen_ef);
  const FrozenDeadline prewarm_deadline(
      protocol::kPrewarmTimeout, "formal prewarm");
  prewarm(arm, saturated, hybrid, queries, prewarm_deadline);

  void* address = ::mmap(nullptr, sizeof(SharedBlock), PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  require(address != MAP_FAILED, "formal shared-state mmap failed");
  auto* shared = new(address) SharedBlock();
  std::vector<pid_t> children;
  children.reserve(protocol::kLanes);
  ForkChildrenGuard children_guard(children);
  for (size_t lane = 0; lane < protocol::kLanes; ++lane) {
    const pid_t pid = ::fork();
    require(pid >= 0, "formal timed-worker fork failed");
    if (pid == 0) {
      int32_t status = 0;
      try {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(options.cpus[lane], &set);
        if (::sched_setaffinity(0, sizeof(set), &set) != 0) status = 3;
        shared->ready.fetch_add(1, std::memory_order_release);
        while (shared->go.load(std::memory_order_acquire) == 0) ::sched_yield();
        if (status == 0) {
          bci_pass_boundary::worker_timed_region = true;
          for (;;) {
            const uint32_t row =
                shared->next.fetch_add(1, std::memory_order_relaxed);
            if (row >= queries.size()) break;
            const auto begin = Clock::now();
            shared->results[row] =
                execute(arm, saturated, hybrid, queries[row]);
            const auto end = Clock::now();
            shared->service_ns[row] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                    .count();
            shared->worker_lane[row] = lane;
            ++shared->child_queries[lane];
          }
          bci_pass_boundary::worker_timed_region = false;
        }
      } catch (...) {
        bci_pass_boundary::worker_timed_region = false;
        status = 2;
      }
      // The counter lives in each forked address space, so copy it into the
      // MAP_SHARED record before publishing done; inspecting only the parent's
      // COW copy would be an ineffective telemetry-I/O audit.
      shared->child_timed_io_calls[lane] =
          bci_pass_boundary::explicit_worker_timed_io_calls.load(
              std::memory_order_acquire);
      shared->child_status[lane] = status;
      shared->done.fetch_add(1, std::memory_order_release);
      while (shared->post_go.load(std::memory_order_acquire) == 0)
        ::sched_yield();
      shared->child_rss_kib[lane] = current_rss_kib(false);
      shared->child_peak_rss_kib[lane] = current_rss_kib(true);
      shared->post_done.fetch_add(1, std::memory_order_release);
      _exit(status);
    }
    children.push_back(pid);
  }

  const FrozenDeadline ready_deadline(
      protocol::kWorkerReadyTimeout, "formal worker ready");
  wait_for_counter(shared->ready, protocol::kLanes, ready_deadline, 1000);
  const FrozenDeadline timed_deadline(
      protocol::kTimedBlockTimeout, "formal timed block done");
  const auto batch_begin = Clock::now();
  shared->go.store(1, std::memory_order_release);
  wait_for_counter(shared->done, protocol::kLanes, timed_deadline, 100);
  const auto batch_end = Clock::now();
  shared->post_go.store(1, std::memory_order_release);
  const FrozenDeadline post_done_deadline(
      protocol::kPostDoneTimeout, "formal post-done resource census");
  wait_for_counter(
      shared->post_done, protocol::kLanes, post_done_deadline, 100);

  uint64_t child_query_sum = 0;
  uint64_t child_timed_io_sum = 0;
  for (size_t lane = 0; lane < protocol::kLanes; ++lane) {
    int state = 0;
    const pid_t child = children[lane];
    for (;;) {
      post_done_deadline.check();
      const pid_t observed = ::waitpid(child, &state, WNOHANG);
      if (observed == child) break;
      if (observed < 0 && errno == EINTR) continue;
      require(observed == 0, "formal timed-worker waitpid failed");
      ::usleep(100);
    }
    // A later exception must not make the guard signal an already-reaped PID,
    // which the kernel could in principle have recycled.
    children[lane] = -1;
    require(WIFEXITED(state) && WEXITSTATUS(state) == 0 &&
                shared->child_status[lane] == 0,
            "formal timed worker failed");
    child_query_sum += shared->child_queries[lane];
    child_timed_io_sum += shared->child_timed_io_calls[lane];
  }
  children_guard.mark_reaped();
  require(child_query_sum == queries.size(),
          "formal shared-pull query census differs");
  require(child_timed_io_sum == 0,
          "timed worker performed explicit telemetry/publication I/O");

  BlockOutcome outcome;
  outcome.block = block;
  outcome.arm = arm;
  outcome.wall_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end - batch_begin)
          .count();
  require(outcome.wall_ns > 0, "formal block wall is zero");
  const FrozenDeadline audit_deadline(
      protocol::kPostWallAuditTimeout, "formal post-wall audit");
  std::ostringstream raw;
  for (size_t row = 0; row < queries.size(); ++row) {
    audit_deadline.check();
    const Result repeat = execute(arm, saturated, hybrid, queries[row]);
    audit_deadline.check();
    const bool nondeterministic =
        repeat.ids != shared->results[row].ids ||
        repeat.size != shared->results[row].size ||
        repeat.removed_self != shared->results[row].removed_self;
    QueryAudit audit = audit_query(data, *reference.space, queries[row],
                                   shared->results[row], nondeterministic,
                                   audit_deadline);
    audit.metrics.elapsed_ns = shared->service_ns[row];
    FamilyAudit& family = outcome.families[
        static_cast<size_t>(queries[row].family)];
    add_metrics(family.strict, audit.metrics);
    family.tie_hits += audit.tie_hits;
    family.tie_denominator += audit.tie_denominator;
    add_metrics(outcome.strict, audit.metrics);
    outcome.tie_hits += audit.tie_hits;
    outcome.tie_denominator += audit.tie_denominator;

    const protocol::Family family_name_value =
        protocol_family(queries[row].family);
    const protocol::Backend backend = protocol::route(arm, family_name_value);
    const std::optional<size_t> route_ef =
        protocol::route_ef(arm, family_name_value, options.frozen_ef);
    raw << "{\"arm\":\"" << protocol::arm_name(arm)
        << "\",\"audit_after_wall\":true"
        << ",\"backend\":\"" << protocol::backend_name(backend) << "\""
        << ",\"block\":" << block
        << ",\"family\":\"" << protocol::family_name(family_name_value)
        << "\",\"forbidden_self\":" << audit.metrics.forbidden_self
        << ",\"frozen_ef\":";
    if (route_ef.has_value()) raw << *route_ef; else raw << "null";
    raw << ",\"invalid\":" << audit.metrics.invalid
        << ",\"kth_distance_f32_bits\":" << audit.kth_distance_bits
        << ",\"leave_one_out_gate\":"
        << (audit.metrics.forbidden_self == 0
                ? "true" : "false")
        << ",\"nondeterministic\":" << audit.metrics.nondeterministic
        << ",\"duplicate\":" << audit.metrics.duplicate
        << ",\"observed_distance_f32_bits\":"
        << optional_uint_array_json(audit.observed_distance_bits)
        << ",\"observed_ids\":" << uint_array_json(audit.observed_ids)
        << ",\"original_workload_row\":" << original_rows[row]
        << ",\"pair\":" << block / 2
        << ",\"pair_order\":\"" << protocol::pair_order(block / 2) << "\""
        << ",\"permuted_row\":" << row
        << ",\"predicate_fail\":" << audit.metrics.predicate_fail
        << ",\"query_base_id\":" << queries[row].query_base_id
        << ",\"removed_self\":" << audit.metrics.removed_self
        << ",\"result_size\":" << shared->results[row].size
        << ",\"schema\":\"" BOOLEANN_LAION_QUERY_AUDIT_SCHEMA "\""
        << ",\"service_ns\":" << shared->service_ns[row]
        << ",\"state_id\":\"" << options.state_id << "\""
        << ",\"strict_denominator\":" << K
        << ",\"strict_hits\":" << audit.metrics.hits
        << ",\"strict_truth_ids\":"
        << uint_array_json(audit.strict_truth_ids)
        << ",\"support_after_leave_one_out\":"
        << queries[row].expected_support
        << ",\"tie_denominator\":" << audit.tie_denominator
        << ",\"tie_hits\":" << audit.tie_hits
        << ",\"underfull\":" << audit.metrics.underfull
        << ",\"worker_lane\":" << shared->worker_lane[row] << "}\n";
  }

  outcome.validity_gate = validity_passes(outcome.strict);
  outcome.recall_gate = recall_fraction_passes(
      outcome.strict.hits, outcome.strict.denominator, options) &&
      recall_fraction_passes(outcome.tie_hits, outcome.tie_denominator, options);
  for (size_t family_index = 0; family_index < 4; ++family_index) {
    const FamilyAudit& family = outcome.families[family_index];
    require(family.strict.queries == options.timed_per_family &&
                family.strict.denominator ==
                    uint64_t(options.timed_per_family) * K &&
                family.tie_denominator ==
                    uint64_t(options.timed_per_family) * K,
            "formal family audit census differs");
    outcome.validity_gate =
        outcome.validity_gate && validity_passes(family.strict);
    outcome.recall_gate = outcome.recall_gate &&
        recall_fraction_passes(family.strict.hits,
                               family.strict.denominator, options) &&
        recall_fraction_passes(family.tie_hits,
                               family.tie_denominator, options);
    if (arm == protocol::Arm::Hybrid &&
        (family_index == static_cast<size_t>(Family::Range) ||
         family_index == static_cast<size_t>(Family::Dnf2)
#ifdef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
         || family_index == static_cast<size_t>(Family::Equality)
#endif
         )) {
      outcome.recall_gate = outcome.recall_gate &&
          family.strict.hits == family.strict.denominator &&
          family.tie_hits == family.tie_denominator;
    }
  }
  outcome.block_gate = outcome.validity_gate && outcome.recall_gate;
  outcome.results_payload = raw.str();
  require(!outcome.results_payload.empty(), "formal raw payload is empty");

  uint64_t max_child_rss = 0;
  uint64_t max_child_peak_rss = 0;
  for (size_t lane = 0; lane < protocol::kLanes; ++lane) {
    max_child_rss = std::max(max_child_rss, shared->child_rss_kib[lane]);
    max_child_peak_rss =
        std::max(max_child_peak_rss, shared->child_peak_rss_kib[lane]);
  }
  std::ostringstream summary;
  summary << std::setprecision(17)
          << "{\"arm\":\"" << protocol::arm_name(arm)
          << "\",\"audit_after_wall\":true"
          << ",\"block\":" << block
          << ",\"block_gate\":" << (outcome.block_gate ? "true" : "false")
          << ",\"complete_batch_qps\":"
          << (double(queries.size()) * 1e9 / outcome.wall_ns)
          << ",\"frozen_ef\":" << options.frozen_ef
          << ",\"graph_beam\":256,\"graph_cut_milli\":1350"
          << ",\"graph_pool\":32"
          << ",\"h_sieve_routes\":\"conjunction_only\""
          << ",\"duplicate\":" << outcome.strict.duplicate
          << ",\"forbidden_self\":" << outcome.strict.forbidden_self
          << ",\"invalid\":" << outcome.strict.invalid
          << ",\"max_child_peak_rss_kib\":" << max_child_peak_rss
          << ",\"max_child_rss_kib\":" << max_child_rss
          << ",\"nondeterministic\":" << outcome.strict.nondeterministic
          << ",\"pair\":" << block / 2
          << ",\"pair_order\":\"" << protocol::pair_order(block / 2) << "\""
          << ",\"predicate_fail\":" << outcome.strict.predicate_fail
          << ",\"queries\":" << outcome.strict.queries
          << ",\"recall_gate\":" << (outcome.recall_gate ? "true" : "false")
          << ",\"removed_self\":" << outcome.strict.removed_self
          << ",\"schema\":\"" BOOLEANN_LAION_BLOCK_SUMMARY_SCHEMA "\""
          << ",\"state_id\":\"" << options.state_id << "\""
          << ",\"strict_denominator\":" << outcome.strict.denominator
          << ",\"strict_hits\":" << outcome.strict.hits
          << ",\"tie_denominator\":" << outcome.tie_denominator
          << ",\"tie_hits\":" << outcome.tie_hits
          << ",\"timed_worker_explicit_io_calls\":0"
          << ",\"underfull\":" << outcome.strict.underfull
          << ",\"validity_gate\":"
          << (outcome.validity_gate ? "true" : "false")
          << ",\"wall_ns\":" << outcome.wall_ns << "}\n";
  outcome.summary_payload = summary.str();

  shared->~SharedBlock();
  require(::munmap(address, sizeof(SharedBlock)) == 0,
          "formal shared-state munmap failed");
  return outcome;
}

void validate_empty_output_directories(const Options& options) {
  bci_pass_boundary::validate_empty_contract_directory(
      options.raw_directory, "formal raw directory");
  bci_pass_boundary::validate_empty_contract_directory(
      options.ack_directory, "formal ACK directory");
  require(options.raw_directory != options.ack_directory,
          "formal raw and ACK directories must be distinct");
}

int formal_main(int argc, char** argv) {
  bci_pass_boundary::install_signal_handlers();
  const Options options = parse_options(argc, argv);
  require(fs::is_regular_file(fs::symlink_status(options.manifest)),
          "manifest must be an existing real regular file");
  require(protocol::schedule_string() == "SHHSSHHSSHHSSHHS" &&
              protocol::schedule_fingerprint() ==
                  protocol::kScheduleFingerprint,
          "frozen 8-pair schedule contract differs");
  validate_empty_output_directories(options);
  ParentHeartbeat heartbeat;
  heartbeat.phase(HeartbeatPhase::InputContracts);

  // The query pool path has only been parsed as a string at this point.  It is
  // not opened until the complete data/build/graph/resource gate below passes.
  Data data = load_data(options.base, options.spmat, options.numeric,
                        options.endpoints);
  const std::vector<Spec> history = read_workload(options.history, 0);
  for (const Spec& spec : history) (void)exact_support(data, spec);
  const uint64_t data_ready_rssanon = rss_anon_bytes();

  heartbeat.phase(HeartbeatPhase::BuildReference);
  SieveEngine reference(
      data, history, 16, 40, options.frozen_ef,
      protocol::kReferenceSieveVectorBudget, 512, 1);
  reference.index->setEf(options.frozen_ef);
  const uint64_t reference_ready_rssanon = rss_anon_bytes();
  const uint64_t reference_runtime = monotone_delta(
      reference_ready_rssanon, data_ready_rssanon, "reference SIEVE");

  heartbeat.phase(HeartbeatPhase::LoadGraph);
  GraphInventory graph = load_graph_inventory(
      data, options.graph_root, GraphInventoryMode::CleanAll200);
  const uint64_t graph_ready_rssanon = rss_anon_bytes();
  const uint64_t graph_observed_runtime = monotone_delta(
      graph_ready_rssanon, reference_ready_rssanon, "clean graph");

  heartbeat.phase(HeartbeatPhase::BuildSaturated);
  SieveEngine saturated(
      data, history, 16, 40, options.frozen_ef,
      protocol::kSaturatedSieveVectorBudget, 512, 1);
  saturated.index->setEf(options.frozen_ef);
  const uint64_t saturated_ready_rssanon = rss_anon_bytes();
  const uint64_t saturated_runtime = monotone_delta(
      saturated_ready_rssanon, graph_ready_rssanon, "saturated SIEVE");

  heartbeat.phase(HeartbeatPhase::ResourceGate);
  const HnswCensus reference_census = census(reference);
  const HnswCensus saturated_census = census(saturated);
  const uint64_t reference_persistent =
      reference_census.root_index_bytes + reference_census.partition_index_bytes;
  const uint64_t saturated_persistent =
      saturated_census.root_index_bytes + saturated_census.partition_index_bytes;
  const bool census_gate =
      reference_census.partitions == options.expected_reference_partitions &&
      reference_census.partition_memberships ==
          options.expected_reference_memberships &&
      saturated_census.partitions == options.expected_saturated_partitions &&
      saturated_census.partition_memberships ==
          options.expected_saturated_memberships &&
      graph.graph_file_bytes == options.expected_graph_file_bytes &&
      graph.subset_file_bytes == options.expected_graph_subset_bytes &&
      graph.dense_graph_bytes == options.expected_graph_dense_bytes &&
      graph.indexed_memberships == options.expected_graph_memberships;
  const bool graph_persistent_payload_gate =
      graph.graph_file_bytes + graph.subset_file_bytes <=
          protocol::kGraphPersistentBytes;
  const protocol::ResourceDecision resource_decision =
      protocol::resource_decision(
          saturated_persistent, reference_persistent,
          protocol::kGraphPersistentBytes, saturated_runtime,
          reference_runtime, graph_observed_runtime);
  const bool persistent_fairness_gate =
      resource_decision != protocol::ResourceDecision::StopPersistent;
  const bool runtime_fairness_gate =
      resource_decision != protocol::ResourceDecision::StopRuntime;
  const bool resource_gate = census_gate && graph_persistent_payload_gate &&
      resource_decision == protocol::ResourceDecision::Pass;
  std::cout << "FORMAL_RESOURCE_GATE"
            << " schema=" BOOLEANN_LAION_RESOURCE_SCHEMA
            << " reference_budget="
            << protocol::kReferenceSieveVectorBudget
            << " saturated_budget="
            << protocol::kSaturatedSieveVectorBudget
            << " reference_partitions=" << reference_census.partitions
            << " reference_partition_memberships="
            << reference_census.partition_memberships
            << " saturated_partitions=" << saturated_census.partitions
            << " saturated_partition_memberships="
            << saturated_census.partition_memberships
            << " reference_persistent_bytes=" << reference_persistent
            << " saturated_persistent_bytes=" << saturated_persistent
            << " graph_authority_persistent_bytes="
            << options.graph_authority_persistent_bytes
            << " reference_runtime_delta_bytes=" << reference_runtime
            << " saturated_runtime_delta_bytes=" << saturated_runtime
            << " graph_observed_runtime_delta_bytes=" << graph_observed_runtime
            << " graph_authority_runtime_bytes="
            << options.graph_authority_runtime_bytes
            << " graph_authority_runtime_role=design_payload_diagnostic_only"
            << " runtime_fairness_rhs_observed_bytes="
            << reference_runtime + graph_observed_runtime
            << " census_gate=" << (census_gate ? "PASS" : "STOP")
            << " graph_persistent_payload_gate="
            << (graph_persistent_payload_gate ? "PASS" : "STOP")
            << " persistent_fairness_gate="
            << (persistent_fairness_gate ? "PASS" : "STOP")
            << " runtime_fairness_gate="
            << (runtime_fairness_gate ? "PASS" : "STOP")
            << " resource_gate=" << (resource_gate ? "PASS" : "STOP")
            << " query_file_opened=false baseline_core_modified=false"
#ifdef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
            << " graph_role=resident_matched_resource_control"
            << " graph_equality_routed=false"
#else
            << " graph_role=equality_execution"
            << " graph_equality_routed=true"
#endif
            << std::endl;
  require(resource_gate, "resource/census gate stopped before query open");

  SamePoolHybrid hybrid(reference, data, graph, 256, 32, 1.35);
  heartbeat.phase(HeartbeatPhase::QueryActivation);
  // This is the sole query-pool open and is dominated by the resource gate.
  const std::vector<Spec> file_order =
      read_workload(options.queries, options.timed_per_family);
  require(file_order.size() == uint64_t(options.timed_per_family) * 4,
          "formal query pool family census differs");
  const std::vector<uint32_t> permutation = protocol::frozen_permutation(
      static_cast<uint32_t>(file_order.size()),
      options.query_permutation_seed);
  require(protocol::permutation_fingerprint(permutation) ==
              options.query_permutation_fingerprint,
          "formal query permutation fingerprint differs from manifest");
  std::vector<Spec> queries;
  queries.reserve(file_order.size());
  for (const uint32_t row : permutation) queries.push_back(file_order[row]);
  std::array<uint32_t, 4> family_counts{};
  for (const Spec& spec : queries) {
    (void)exact_support(data, spec);
    ++family_counts[static_cast<size_t>(spec.family)];
  }
  for (const uint32_t count : family_counts)
    require(count == options.timed_per_family,
            "formal permuted family census differs");
  std::cout << "FORMAL_ACTIVATION"
            << " schema=" BOOLEANN_LAION_ACTIVATION_SCHEMA
            << " state_id=" << options.state_id
            << " pairs=" << protocol::kPairs
            << " blocks=" << protocol::kBlocks
            << " schedule=" << protocol::schedule_string()
            << " schedule_fingerprint=" << protocol::kScheduleFingerprint
            << " query_rows=" << queries.size()
            << " permutation_seed=" << options.query_permutation_seed
            << " permutation_fingerprint="
            << options.query_permutation_fingerprint
            << " frozen_ef=" << options.frozen_ef
            << " s_all_routes_ef=frozen_ef"
            << " h_conjunction_ef=frozen_ef"
            << " h_other_sieve_routes=none"
#ifdef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
            << " h_equality_route=complete_support_exact_topk"
            << " h_equality_recall_requirement=exact_1"
            << " graph_resident_control_only=true"
            << " graph_equality_routed=false"
#else
            << " h_equality_route=clean_local_graph_r16"
            << " h_equality_recall_requirement=ge_0p9995"
            << " graph_resident_control_only=false"
            << " graph_equality_routed=true"
#endif
            << " sieve_M=16 sieve_ef_construction=40"
            << " sieve_bitvector_cutoff=512 sieve_build_threads=1"
            << " graph_beam=256 graph_pool=32 graph_cut=1.35"
            << " prewarm_deadline_s=600 worker_ready_deadline_s=60"
            << " timed_done_deadline_s=600 post_done_deadline_s=60"
            << " postwall_audit_deadline_s=1800 ack_deadline_s=600"
            << " cpu_set=";
  for (size_t lane = 0; lane < options.cpus.size(); ++lane) {
    if (lane) std::cout << ',';
    std::cout << options.cpus[lane];
  }
  std::cout
            << " sieve_instances=2 graph_inventory_instances=1"
            << " data_instances=1 query_open_after_resource_gate=true"
            << " baseline_core_modified=false" << std::endl;

  // fork(2) is permitted only after the runner's build heartbeat is stopped
  // and joined.  From this point through all timed blocks, the separately
  // running Python controller is the sole heartbeat/RSS monitor.
  heartbeat.stop_before_fork();
  require(!heartbeat.thread_live() && process_thread_count() == 1,
          "internal heartbeat or another runner thread survived to first fork");
  std::cout << "FORMAL_FORK_BOUNDARY internal_heartbeat_stopped=true"
            << " internal_heartbeat_joined=true process_threads=1"
            << " timed_heartbeat_owner=external_controller" << std::endl;

  std::array<std::optional<BlockOutcome>, protocol::kBlocks> outcomes;
  for (size_t block = 0; block < protocol::kBlocks; ++block) {
    bci_pass_boundary::require_not_stopped();
    BlockOutcome outcome = run_block(
        block, options, data, saturated, reference, hybrid, queries,
        permutation, heartbeat);
    const fs::path results_path =
        options.raw_directory / protocol::block_results_leaf(block);
    const fs::path summary_path =
        options.raw_directory / protocol::block_summary_leaf(block);
    laion_paired_formal_block_durability::
        publish_block_artifact_atomic_no_replace(
            results_path, outcome.results_payload,
            [] { bci_pass_boundary::require_not_stopped(); });
    laion_paired_formal_block_durability::
        publish_block_artifact_atomic_no_replace(
            summary_path, outcome.summary_payload,
            [] { bci_pass_boundary::require_not_stopped(); });
    const durable_pass_release::AckRecord acknowledgement =
        laion_paired_formal_block_durability::
            wait_for_next_block_release_for_files(
                options.ack_directory, options.state_id, block,
                summary_path, results_path, options.ack_timeout,
                options.ack_poll,
                [] { bci_pass_boundary::require_not_stopped(); },
                [] {});
    require(acknowledgement.pass == block &&
                acknowledgement.state_id == options.state_id &&
                acknowledgement.passes_jsonl_bytes ==
                    outcome.summary_payload.size() &&
                acknowledgement.results_jsonl_bytes ==
                    outcome.results_payload.size(),
            "returned adapter AckRecord does not exactly bind block files");
    outcomes[block] = std::move(outcome);
    std::cout << "FORMAL_BLOCK_RELEASED"
              << " block=" << block
              << " arm=" << protocol::arm_name(outcomes[block]->arm)
              << " summary_bytes=" << acknowledgement.passes_jsonl_bytes
              << " results_bytes=" << acknowledgement.results_jsonl_bytes
              << " next_block_authorized=true" << std::endl;
    if (!outcomes[block]->block_gate) {
      std::cout << "FORMAL_EARLY_STOP block=" << block
                << " reason=validity_or_recall_gate"
                << " partial_preserved=true next_block_started=false"
                << std::endl;
      return 2;
    }
  }
  // The controller is the sole owner of the prospectively frozen pairwise
  // futility and final performance inference.  This runner gates only the
  // per-block validity/recall contract and release ordering.
  std::cout << "FORMAL_FINAL schema=" BOOLEANN_LAION_FINAL_SCHEMA
            << " completed_pairs=" << protocol::kPairs
            << " completed_blocks=" << protocol::kBlocks
            << " performance_inference_owner=external_controller"
            << " all_raw_summary_ack_release_bound=true"
            << " runner_validity_recall_gate=PASS" << std::endl;
  return 0;
}

}  // namespace laion1m_v19_same_resident_paired_formal_runner

#ifndef BOOLEANN_LAION_V19_NO_MAIN
int main(int argc, char** argv) {
  try {
    return laion1m_v19_same_resident_paired_formal_runner::formal_main(
        argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
#endif
