#pragma once

// Query-free scientific contract shared by the LAION1M v19 formal runner and
// its controller fixture.  This file deliberately has no dataset, SIEVE,
// graph, filesystem-publication, or performance dependency.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace laion1m_v19_same_resident_formal_protocol {

inline constexpr size_t kPairs = 8;
inline constexpr size_t kBlocks = 2 * kPairs;
inline constexpr size_t kLanes = 8;
inline constexpr size_t kReferenceSieveVectorBudget = 100000;
inline constexpr size_t kSaturatedSieveVectorBudget = 1000000;
inline constexpr size_t kFrozenEf = 800;
inline constexpr size_t kTimedPerFamily = 200;
inline constexpr size_t kMinimumRecallNumerator = 9995;
inline constexpr size_t kMinimumRecallDenominator = 10000;
inline constexpr size_t kExpectedReferencePartitions = 18;
inline constexpr size_t kExpectedReferenceMemberships = 109757;
inline constexpr size_t kExpectedSaturatedPartitions = 42;
inline constexpr size_t kExpectedSaturatedMemberships = 167220;
inline constexpr uint64_t kExpectedGraphFileBytes = 80960588ULL;
inline constexpr uint64_t kExpectedGraphSubsetBytes = 5018016ULL;
inline constexpr uint64_t kExpectedGraphDenseBytes = 85292672ULL;
inline constexpr uint64_t kExpectedGraphMemberships = 1254304ULL;
inline constexpr uint64_t kGraphPersistentBytes = 85981008ULL;
// This is the frozen design-payload accounting number.  Runtime fairness uses
// the same-process observed graph delta, never this diagnostic authority.
inline constexpr uint64_t kGraphDesignPayloadRuntimeBytes = 90309888ULL;
inline constexpr auto kPrewarmTimeout = std::chrono::minutes(10);
inline constexpr auto kWorkerReadyTimeout = std::chrono::seconds(60);
inline constexpr auto kTimedBlockTimeout = std::chrono::minutes(10);
inline constexpr auto kPostDoneTimeout = std::chrono::seconds(60);
inline constexpr auto kPostWallAuditTimeout = std::chrono::minutes(30);
inline constexpr auto kAckTimeout = std::chrono::minutes(10);
inline constexpr auto kAckPoll = std::chrono::milliseconds(10);
inline constexpr uint64_t kScheduleFingerprint = 0x7a8c45ab0f7332dbULL;

enum class Arm : uint8_t { SaturatedSieve = 0, Hybrid = 1 };
enum class Family : uint8_t { Equality = 0, Conjunction = 1, Range = 2, Dnf2 = 3 };
enum class Backend : uint8_t {
  SaturatedSieve,
  CleanLocalGraph,
  ReferenceSieve,
  CompleteSupportExactTopK,
};

inline const char* arm_name(Arm arm) {
  return arm == Arm::SaturatedSieve ? "S" : "H";
}

inline const char* family_name(Family family) {
  static constexpr std::array<const char*, 4> names{
      "equality", "conjunction", "range", "dnf2"};
  const size_t index = static_cast<size_t>(family);
  if (index >= names.size()) throw std::runtime_error("family outside contract");
  return names[index];
}

inline const char* backend_name(Backend backend) {
  switch (backend) {
    case Backend::SaturatedSieve: return "official_sieve_saturated_1m";
    case Backend::CleanLocalGraph: return "clean_local_graph_r16";
    case Backend::ReferenceSieve: return "official_sieve_reference_100k";
    case Backend::CompleteSupportExactTopK:
      return "complete_support_exact_topk";
  }
  throw std::runtime_error("backend outside contract");
}

inline Arm scheduled_arm(size_t block) {
  if (block >= kBlocks) throw std::runtime_error("formal block outside 0..15");
  // Adjacent blocks form one pair.  Pair order alternates SH, HS, ... and is
  // fixed independently of all outcomes.
  const bool even_pair = ((block / 2U) % 2U) == 0U;
  const bool first_in_pair = (block % 2U) == 0U;
  const bool saturated = even_pair == first_in_pair;
  return saturated ? Arm::SaturatedSieve : Arm::Hybrid;
}

inline const char* pair_order(size_t pair) {
  if (pair >= kPairs) throw std::runtime_error("formal pair outside 0..7");
  return pair % 2U == 0U ? "SH" : "HS";
}

inline std::string schedule_string() {
  std::string result;
  result.reserve(kBlocks);
  for (size_t block = 0; block < kBlocks; ++block)
    result += arm_name(scheduled_arm(block));
  return result;
}

inline uint64_t schedule_fingerprint() {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char value : schedule_string()) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline uint64_t splitmix64_next(uint64_t& state) {
  uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline uint64_t bounded_draw(uint64_t& state, uint64_t bound) {
  if (bound == 0) throw std::runtime_error("zero permutation bound");
  const uint64_t threshold = (uint64_t(0) - bound) % bound;
  for (;;) {
    const uint64_t value = splitmix64_next(state);
    if (value >= threshold) return value % bound;
  }
}

inline std::vector<uint32_t> frozen_permutation(uint32_t count,
                                                uint64_t seed) {
  std::vector<uint32_t> result(count);
  std::iota(result.begin(), result.end(), uint32_t(0));
  uint64_t state = seed;
  for (uint32_t remaining = count; remaining > 1; --remaining) {
    const uint32_t selected =
        static_cast<uint32_t>(bounded_draw(state, remaining));
    std::swap(result[remaining - 1], result[selected]);
  }
  return result;
}

inline uint64_t permutation_fingerprint(
    const std::vector<uint32_t>& order) {
  uint64_t hash = 1469598103934665603ULL;
  for (const uint32_t value : order) {
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= (value >> (8 * byte)) & 0xffU;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

inline Backend route(Arm arm, Family family) {
  if (arm == Arm::SaturatedSieve) return Backend::SaturatedSieve;
  switch (family) {
    case Family::Equality:
#ifdef BOOLEANN_LAION_COMPLETE_EQUALITY_EXACT_ROUTE
      return Backend::CompleteSupportExactTopK;
#else
      return Backend::CleanLocalGraph;
#endif
    case Family::Conjunction: return Backend::ReferenceSieve;
    case Family::Range:
    case Family::Dnf2: return Backend::CompleteSupportExactTopK;
  }
  throw std::runtime_error("family outside route contract");
}

inline std::optional<size_t> route_ef(Arm arm, Family family,
                                      size_t frozen_ef) {
  if (frozen_ef == 0) throw std::runtime_error("frozen ef must be positive");
  const Backend backend = route(arm, family);
  if (backend == Backend::SaturatedSieve ||
      backend == Backend::ReferenceSieve)
    return frozen_ef;
  return std::nullopt;
}

inline uint32_t float_bits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct ExactCandidate {
  uint32_t id = 0;
  float distance = 0.0f;
};

struct ExactTruth {
  std::vector<uint32_t> ranked_ids;
  std::vector<uint32_t> strict_set_ids;
  float kth_distance = 0.0f;
};

inline ExactTruth exact_truth(std::vector<ExactCandidate> candidates,
                              size_t k) {
  if (k == 0 || candidates.size() < k)
    throw std::runtime_error("exact truth requires at least k candidates");
  std::unordered_set<uint32_t> candidate_ids;
  candidate_ids.reserve(candidates.size());
  for (const ExactCandidate& candidate : candidates) {
    if (!(candidate.distance >= 0.0f) ||
        candidate.distance == std::numeric_limits<float>::infinity())
      throw std::runtime_error("exact candidate distance is invalid");
    if (!candidate_ids.insert(candidate.id).second)
      throw std::runtime_error("exact candidate IDs must be globally unique");
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const ExactCandidate& left, const ExactCandidate& right) {
              return std::pair<float, uint32_t>{left.distance, left.id} <
                     std::pair<float, uint32_t>{right.distance, right.id};
            });
  ExactTruth result;
  result.ranked_ids.reserve(k);
  for (size_t index = 0; index < k; ++index)
    result.ranked_ids.push_back(candidates[index].id);
  result.strict_set_ids = result.ranked_ids;
  std::sort(result.strict_set_ids.begin(), result.strict_set_ids.end());
  result.kth_distance = candidates[k - 1].distance;
  return result;
}

enum class ResourceDecision : uint8_t {
  Pass,
  StopPersistent,
  StopRuntime,
};

inline ResourceDecision resource_decision(
    uint64_t saturated_persistent, uint64_t reference_persistent,
    uint64_t graph_persistent, uint64_t saturated_runtime,
    uint64_t reference_runtime, uint64_t graph_observed_runtime) {
  if (graph_persistent >
          std::numeric_limits<uint64_t>::max() - reference_persistent ||
      saturated_persistent > reference_persistent + graph_persistent)
    return ResourceDecision::StopPersistent;
  if (graph_observed_runtime >
          std::numeric_limits<uint64_t>::max() - reference_runtime ||
      saturated_runtime > reference_runtime + graph_observed_runtime)
    return ResourceDecision::StopRuntime;
  return ResourceDecision::Pass;
}

inline size_t strict_set_hits(const std::vector<uint32_t>& strict_set_ids,
                              const std::vector<uint32_t>& observed_ids) {
  if (!std::is_sorted(strict_set_ids.begin(), strict_set_ids.end()) ||
      std::adjacent_find(strict_set_ids.begin(), strict_set_ids.end()) !=
          strict_set_ids.end())
    throw std::runtime_error("strict truth IDs must be sorted unique");
  size_t hits = 0;
  for (const uint32_t id : observed_ids)
    hits += std::binary_search(strict_set_ids.begin(), strict_set_ids.end(), id);
  return hits;
}

inline size_t tie_aware_hits(const std::vector<float>& observed_distances,
                             float kth_distance) {
  size_t hits = 0;
  for (const float distance : observed_distances)
    hits += distance <= kth_distance;
  return hits;
}

inline std::string block_results_leaf(size_t block) {
  if (block >= kBlocks) throw std::runtime_error("formal block outside 0..15");
  const std::string number = block < 10 ? "0" + std::to_string(block)
                                        : std::to_string(block);
  return "block_" + number + "_results.jsonl";
}

inline std::string block_summary_leaf(size_t block) {
  if (block >= kBlocks) throw std::runtime_error("formal block outside 0..15");
  const std::string number = block < 10 ? "0" + std::to_string(block)
                                        : std::to_string(block);
  return "block_" + number + "_summary.jsonl";
}

}  // namespace laion1m_v19_same_resident_formal_protocol
