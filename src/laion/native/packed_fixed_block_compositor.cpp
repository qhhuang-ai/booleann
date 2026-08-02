// Project-owned packed exact fixed-block compositor candidate for LAION1M.
//
// The immutable route table contains only logical U24 object positions,
// clause-local boundary flags, and packed 20-byte clause records.  Payload
// addresses and cardinalities are resolved inside the request from the
// compact-U24 owner's selection/rank/offset directory.  All explicit hot-path
// arrays are caller-owned.  This isolated candidate does not modify a baseline
// or the retained decoder/scorer source and exposes no performance driver.

#ifndef LAION1M_RETAINED_PREFETCH_POLICY
#define LAION1M_RETAINED_PREFETCH_POLICY 2
#endif
#include "exact_topk_avx2.cpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" int u24le_decode_v1(
    const uint8_t* src, std::size_t src_bytes, int32_t* dst,
    std::size_t dst_capacity, std::size_t* decoded_count);

namespace {

constexpr std::size_t kMaxObjects = 64;
constexpr std::size_t kFbMaxSegments = 64;
constexpr std::size_t kMaxDecodedRows = 14044;
constexpr std::size_t kBatchIds = 64;
constexpr std::size_t kFbTopK = 10;
constexpr std::size_t kAttributes = 3;
constexpr std::size_t kLogicalPositions = 39100;
constexpr std::size_t kU24Words = 611;
constexpr std::size_t kU24Prefixes = 612;
constexpr std::size_t kDescriptorHeaderBytes = 128;
constexpr std::size_t kClauseRecordBytes = 20;

struct FixedBlockAuditV2 {
  uint64_t decoded_input_rows;
  uint64_t segment_input_rows;
  uint64_t boundary_predicate_checks;
  uint64_t support_before_leave_one_out;
  uint64_t support_after_leave_one_out;
  uint64_t self_fragment_occurrences;
};

// Exactly the arrays formerly charged to the source-level 1,904-byte local
// estimate.  They are now part of the caller-owned lane workspace.
struct FixedBlockLocalScratchV2 {
  uint16_t object_positions[kMaxObjects];
  uint32_t object_starts[kMaxObjects];
  uint8_t segment_local_objects[kFbMaxSegments];
  uint32_t positions[kFbMaxSegments];
  int32_t previous_raw[kFbMaxSegments];
  uint8_t has_previous_raw[kFbMaxSegments];
  int32_t current[kFbMaxSegments];
  uint8_t live[kFbMaxSegments];
  uint8_t heap[kFbMaxSegments];
  int32_t batch_ids[kBatchIds];
  Candidate best[kFbTopK];
  const float* vectors[kRetainedInterleaveRows];
  const float* future[kRetainedInterleaveRows];
  float distances[kRetainedInterleaveRows];
};
static_assert(sizeof(FixedBlockLocalScratchV2) == 1904,
              "caller-owned local scratch ABI drift");

constexpr std::size_t kLaneDecodeScratchBytes =
    kMaxDecodedRows * sizeof(int32_t);
constexpr std::size_t kLaneOutputWorkspaceBytes =
    kFbTopK * (sizeof(int32_t) + sizeof(float)) +
    sizeof(FixedBlockAuditV2);

inline uint16_t load_u16(const uint8_t* source) noexcept {
  uint16_t value;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

inline uint32_t load_u32(const uint8_t* source) noexcept {
  uint32_t value;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

inline uint64_t load_u64(const uint8_t* source) noexcept {
  uint64_t value;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

inline double load_f64(const uint8_t* source) noexcept {
  double value;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

inline std::size_t align64(std::size_t value) noexcept {
  return (value + 63U) & ~std::size_t{63U};
}

struct U24DirectoryV2 {
  const uint8_t* owner;
  uint64_t owner_bytes;
  uint32_t words_offset;
  uint32_t prefix_offset;
  uint32_t offsets_offset;
  uint32_t offset_entries;
  uint32_t arena_offset;
  uint32_t payload_bytes;
  uint32_t selected_objects;
};

inline bool parse_u24_owner(
    const uint8_t* owner, uint64_t owner_bytes,
    U24DirectoryV2& output) noexcept {
  static constexpr uint8_t kMagic[8] = {
      'B', 'A', 'J', '7', 'U', '2', '4', 0};
  if (owner == nullptr || owner_bytes < 64 ||
      std::memcmp(owner, kMagic, sizeof(kMagic)) != 0) {
    return false;
  }
  const uint32_t version = load_u32(owner + 8);
  const uint32_t allocated = load_u32(owner + 12);
  const uint32_t words_offset = load_u32(owner + 16);
  const uint32_t words = load_u32(owner + 20);
  const uint32_t prefix_offset = load_u32(owner + 24);
  const uint32_t prefixes = load_u32(owner + 28);
  const uint32_t offsets_offset = load_u32(owner + 32);
  const uint32_t offset_entries = load_u32(owner + 36);
  const uint32_t arena_offset = load_u32(owner + 40);
  const uint32_t payload_bytes = load_u32(owner + 44);
  const uint32_t selected_objects = load_u32(owner + 48);
  if (version != 1 || allocated != owner_bytes || words_offset != 64 ||
      words != kU24Words ||
      prefix_offset != words_offset + 8U * words ||
      prefixes != kU24Prefixes ||
      offsets_offset != prefix_offset + 2U * prefixes ||
      offset_entries != selected_objects + 1U || selected_objects == 0 ||
      static_cast<uint64_t>(offsets_offset) + 4ULL * offset_entries >
          arena_offset ||
      (arena_offset & 63U) != 0 ||
      static_cast<uint64_t>(arena_offset) + payload_bytes > owner_bytes) {
    return false;
  }
  output = {owner, owner_bytes, words_offset, prefix_offset, offsets_offset,
            offset_entries, arena_offset, payload_bytes, selected_objects};
  return true;
}

inline bool resolve_u24(
    const U24DirectoryV2& directory, uint16_t logical_position,
    const uint8_t*& payload, uint32_t& rows) noexcept {
  if (logical_position >= kLogicalPositions) {
    return false;
  }
  const uint32_t word_index = logical_position / 64U;
  const uint32_t bit = logical_position % 64U;
  const uint64_t word = load_u64(
      directory.owner + directory.words_offset + 8U * word_index);
  if ((word & (uint64_t{1} << bit)) == 0) {
    return false;
  }
  const uint32_t prefix = load_u16(
      directory.owner + directory.prefix_offset + 2U * word_index);
  const uint64_t prior_mask = bit == 0
      ? uint64_t{0}
      : ((uint64_t{1} << bit) - 1U);
  const uint32_t ordinal = prefix + static_cast<uint32_t>(
      __builtin_popcountll(word & prior_mask));
  if (ordinal >= directory.selected_objects) {
    return false;
  }
  const uint32_t begin = load_u32(
      directory.owner + directory.offsets_offset + 4U * ordinal);
  const uint32_t end = load_u32(
      directory.owner + directory.offsets_offset + 4U * (ordinal + 1U));
  if (begin > end || end > directory.payload_bytes || (end - begin) % 3U) {
    return false;
  }
  rows = (end - begin) / 3U;
  payload = rows == 0
      ? nullptr
      : directory.owner + directory.arena_offset + begin;
  return true;
}

struct DescriptorV2 {
  const uint8_t* owner;
  uint32_t routes;
  uint32_t segments;
  uint32_t clauses;
  uint32_t route_begins_offset;
  uint32_t segment_positions_offset;
  uint32_t segment_flags_offset;
  uint32_t clauses_offset;
  uint32_t clauses_per_route;
};

inline bool parse_descriptor(
    const uint8_t* owner, uint64_t owner_bytes,
    DescriptorV2& output) noexcept {
  static constexpr uint8_t kMagic[8] = {
      'L', '1', 'M', 'F', 'B', 'P', '2', 0};
  if (owner == nullptr || owner_bytes < kDescriptorHeaderBytes ||
      std::memcmp(owner, kMagic, sizeof(kMagic)) != 0) {
    return false;
  }
  const uint32_t version = load_u32(owner + 8);
  const uint32_t allocated = load_u32(owner + 12);
  const uint32_t routes = load_u32(owner + 16);
  const uint32_t segments = load_u32(owner + 20);
  const uint32_t clauses = load_u32(owner + 24);
  const uint32_t route_begins = load_u32(owner + 28);
  const uint32_t segment_positions = load_u32(owner + 32);
  const uint32_t segment_flags = load_u32(owner + 36);
  const uint32_t clause_records = load_u32(owner + 40);
  if (version != 2 || allocated != owner_bytes || routes == 0 ||
      segments == 0 || clauses == 0 ||
      (clauses != routes && clauses != 2U * routes) ||
      route_begins != kDescriptorHeaderBytes ||
      segment_positions != route_begins + 4U * routes ||
      segment_flags != segment_positions + 2U * segments ||
      clause_records != segment_flags + segments) {
    return false;
  }
  const uint64_t logical_end =
      static_cast<uint64_t>(clause_records) +
      static_cast<uint64_t>(kClauseRecordBytes) * clauses;
  if (logical_end > owner_bytes || align64(logical_end) != owner_bytes) {
    return false;
  }
  output = {owner, routes, segments, clauses, route_begins,
            segment_positions, segment_flags, clause_records,
            clauses / routes};
  return true;
}

inline bool attribute_descriptor(
    uint32_t attribute,
    uintptr_t value0, uintptr_t valid0, uint8_t kind0,
    uintptr_t value1, uintptr_t valid1, uint8_t kind1,
    uintptr_t value2, uintptr_t valid2, uint8_t kind2,
    uintptr_t& value, uintptr_t& valid, uint8_t& kind) noexcept {
  switch (attribute) {
    case 0: value = value0; valid = valid0; kind = kind0; break;
    case 1: value = value1; valid = valid1; kind = kind1; break;
    case 2: value = value2; valid = valid2; kind = kind2; break;
    default: return false;
  }
  return value != 0 && valid != 0 && (kind == 1 || kind == 2);
}

inline bool head_less(
    uint8_t left, uint8_t right, const int32_t* current) noexcept {
  return current[left] < current[right] ||
         (current[left] == current[right] && left < right);
}

inline void heap_push(
    uint8_t segment, uint8_t* heap, std::size_t& heap_size,
    const int32_t* current) noexcept {
  std::size_t child = heap_size++;
  heap[child] = segment;
  while (child != 0) {
    const std::size_t parent = (child - 1U) / 2U;
    if (!head_less(heap[child], heap[parent], current)) break;
    std::swap(heap[child], heap[parent]);
    child = parent;
  }
}

inline uint8_t heap_pop(
    uint8_t* heap, std::size_t& heap_size,
    const int32_t* current) noexcept {
  const uint8_t result = heap[0];
  --heap_size;
  if (heap_size == 0) return result;
  heap[0] = heap[heap_size];
  std::size_t parent = 0;
  while (true) {
    const std::size_t left = 2U * parent + 1U;
    if (left >= heap_size) break;
    const std::size_t right = left + 1U;
    std::size_t smallest = left;
    if (right < heap_size && head_less(heap[right], heap[left], current)) {
      smallest = right;
    }
    if (!head_less(heap[smallest], heap[parent], current)) break;
    std::swap(heap[smallest], heap[parent]);
    parent = smallest;
  }
  return result;
}

inline void admit_candidate(
    const Candidate& scored, Candidate* best,
    std::size_t& best_size) noexcept {
  if (best_size < kFbTopK) {
    best[best_size++] = scored;
    std::push_heap(best, best + best_size, candidate_less);
  } else if (candidate_less(scored, best[0])) {
    std::pop_heap(best, best + best_size, candidate_less);
    best[best_size - 1] = scored;
    std::push_heap(best, best + best_size, candidate_less);
  }
}

inline int score_batch(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* ids, uint64_t count, Candidate* best,
    std::size_t& best_size, const float** vectors, const float** future,
    float* distances) noexcept {
  int32_t previous = -1;
  for (uint64_t index = 0; index < count; ++index) {
    const int32_t id = ids[index];
    if (id < 0 || static_cast<uint64_t>(id) >= base_rows) return 4;
    if (index != 0 && id <= previous) return 5;
    previous = id;
  }
  uint64_t index = 0;
  for (; index + kRetainedInterleaveRows <= count;
       index += kRetainedInterleaveRows) {
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      vectors[lane] = base +
          static_cast<std::size_t>(ids[index + lane]) * kDimension;
      const uint64_t future_index = index + lane + kPrefetchDistance;
      future[lane] = future_index < count
          ? base + static_cast<std::size_t>(ids[future_index]) * kDimension
          : nullptr;
    }
    squared_l2_f32_512_retained_interleaved8(
        query, vectors, future, distances);
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      if (!std::isfinite(distances[lane])) return 6;
      admit_candidate(
          Candidate{distances[lane], ids[index + lane]}, best, best_size);
    }
  }
  for (; index < count; ++index) {
    const float distance = squared_l2_f32_512_retained(
        query, base + static_cast<std::size_t>(ids[index]) * kDimension);
    if (!std::isfinite(distance)) return 6;
    admit_candidate(Candidate{distance, ids[index]}, best, best_size);
  }
  return 0;
}

}  // namespace

extern "C" {

uint32_t laion1m_lowmem_fixedblock_packed_v2_abi_version() { return 2; }
uint32_t laion1m_lowmem_fixedblock_packed_v2_max_objects() { return kMaxObjects; }
uint32_t laion1m_lowmem_fixedblock_packed_v2_max_segments() { return kFbMaxSegments; }
uint32_t laion1m_lowmem_fixedblock_packed_v2_max_decoded_rows() {
  return kMaxDecodedRows;
}
uint32_t laion1m_lowmem_fixedblock_packed_v2_lane_decode_scratch_bytes() {
  return kLaneDecodeScratchBytes;
}
uint32_t laion1m_lowmem_fixedblock_packed_v2_lane_local_scratch_bytes() {
  return sizeof(FixedBlockLocalScratchV2);
}
uint32_t laion1m_lowmem_fixedblock_packed_v2_lane_output_workspace_bytes() {
  return kLaneOutputWorkspaceBytes;
}
uint32_t laion1m_lowmem_fixedblock_packed_v2_audit_struct_bytes() {
  return sizeof(FixedBlockAuditV2);
}

// Status codes retain v1 meanings; 13 denotes packed descriptor/owner drift.
int laion1m_lowmem_fixedblock_exact_top10_packed_v2(
    const float* base, uint64_t base_rows, const float* query,
    const uint8_t* u24_owner, uint64_t u24_owner_bytes,
    const uint8_t* descriptor_owner, uint64_t descriptor_owner_bytes,
    uint64_t plan_ordinal,
    uintptr_t value0, uintptr_t valid0, uint8_t kind0,
    uintptr_t value1, uintptr_t valid1, uint8_t kind1,
    uintptr_t value2, uintptr_t valid2, uint8_t kind2,
    int32_t self_id,
    int32_t* decode_scratch, uint64_t decode_scratch_capacity,
    uint8_t* local_scratch_bytes, uint64_t local_scratch_capacity,
    int32_t* output_top_ids, float* output_top_squared_l2,
    FixedBlockAuditV2* output_audit,
    int32_t* diagnostic_support_after,
    uint64_t diagnostic_support_capacity) noexcept {
  if (base == nullptr || query == nullptr || u24_owner == nullptr ||
      descriptor_owner == nullptr || decode_scratch == nullptr ||
      local_scratch_bytes == nullptr || output_top_ids == nullptr ||
      output_top_squared_l2 == nullptr || output_audit == nullptr) {
    return 1;
  }
  if (base_rows == 0 || base_rows > 0x7fffffffULL || self_id < 0 ||
      static_cast<uint64_t>(self_id) >= base_rows ||
      local_scratch_capacity < sizeof(FixedBlockLocalScratchV2)) {
    return 2;
  }
  U24DirectoryV2 u24{};
  DescriptorV2 descriptor{};
  if (!parse_u24_owner(u24_owner, u24_owner_bytes, u24) ||
      !parse_descriptor(
          descriptor_owner, descriptor_owner_bytes, descriptor) ||
      plan_ordinal > 0xffffU) {
    return 13;
  }
  // Each route word packs [stable plan ordinal:u16 | segment begin:u16].
  // Thus a dynamically selected subset retains manifest identity without an
  // additional lookup section or a dense 200-route allocation.
  uint32_t low = 0, high = descriptor.routes;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2U;
    const uint32_t raw = load_u32(
        descriptor.owner + descriptor.route_begins_offset + 4U * middle);
    if ((raw >> 16U) < plan_ordinal) low = middle + 1U;
    else high = middle;
  }
  if (low == descriptor.routes) return 13;
  const uint32_t route_word = load_u32(
      descriptor.owner + descriptor.route_begins_offset + 4U * low);
  if ((route_word >> 16U) != plan_ordinal) return 13;
  const uint32_t route_index = low;
  const uint32_t begin = route_word & 0xffffU;
  const uint32_t end = route_index + 1U < descriptor.routes
      ? load_u32(descriptor.owner + descriptor.route_begins_offset +
                 4U * (route_index + 1U)) & 0xffffU
      : descriptor.segments;
  if (begin >= end || end > descriptor.segments ||
      end - begin > kFbMaxSegments) {
    return 7;
  }
  const uint32_t segment_count = end - begin;

  // Validate every route-local descriptor and compute the exact unique-object
  // input cardinality before touching any scratch/output byte.
  uint64_t decoded_total = 0;
  uint64_t unique_object_count = 0;
  for (uint32_t segment = 0; segment < segment_count; ++segment) {
    const uint32_t global_segment = begin + segment;
    const uint16_t position = load_u16(
        descriptor.owner + descriptor.segment_positions_offset +
        2U * global_segment);
    const uint8_t flag = descriptor.owner[
        descriptor.segment_flags_offset + global_segment];
    if (flag > descriptor.clauses_per_route) return 9;
    if (flag != 0) {
      const uint32_t clause = static_cast<uint32_t>(route_index) *
          descriptor.clauses_per_route + flag - 1U;
      const uint8_t* record = descriptor.owner + descriptor.clauses_offset +
          kClauseRecordBytes * clause;
      const uint32_t attribute = load_u32(record);
      const double lower = load_f64(record + 4);
      const double upper = load_f64(record + 12);
      uintptr_t values = 0, valid = 0;
      uint8_t kind = 0;
      if (!attribute_descriptor(
              attribute, value0, valid0, kind0, value1, valid1, kind1,
              value2, valid2, kind2, values, valid, kind) ||
          std::isnan(lower) || std::isnan(upper) || !(lower < upper)) {
        return 9;
      }
    }
    bool first = true;
    for (uint32_t prior = 0; prior < segment; ++prior) {
      const uint16_t prior_position = load_u16(
          descriptor.owner + descriptor.segment_positions_offset +
          2U * (begin + prior));
      if (prior_position == position) {
        first = false;
        break;
      }
    }
    if (first) {
      const uint8_t* payload = nullptr;
      uint32_t rows = 0;
      if (!resolve_u24(u24, position, payload, rows)) return 8;
      if ((rows != 0 && payload == nullptr) || rows > kMaxDecodedRows ||
          decoded_total > kMaxDecodedRows - rows) {
        return 2;
      }
      decoded_total += rows;
      ++unique_object_count;
    }
  }
  if (unique_object_count == 0 || unique_object_count > kMaxObjects ||
      decoded_total == 0 || decoded_total > decode_scratch_capacity ||
      decoded_total > kMaxDecodedRows) {
    return 2;
  }
  if ((diagnostic_support_after == nullptr) !=
      (diagnostic_support_capacity == 0) ||
      (diagnostic_support_after != nullptr &&
       diagnostic_support_capacity < decoded_total)) {
    return 11;
  }

  auto* scratch = reinterpret_cast<FixedBlockLocalScratchV2*>(
      local_scratch_bytes);
  std::fill_n(scratch->has_previous_raw, kFbMaxSegments, uint8_t{0});
  std::fill_n(scratch->live, kFbMaxSegments, uint8_t{0});

  uint32_t stored_objects = 0;
  for (uint32_t segment = 0; segment < segment_count; ++segment) {
    const uint16_t position = load_u16(
        descriptor.owner + descriptor.segment_positions_offset +
        2U * (begin + segment));
    uint32_t local = 0;
    while (local < stored_objects &&
           scratch->object_positions[local] != position) ++local;
    if (local == stored_objects) {
      scratch->object_positions[stored_objects++] = position;
    }
    scratch->segment_local_objects[segment] = static_cast<uint8_t>(local);
  }
  if (stored_objects != unique_object_count) return 13;

  uint64_t cursor = 0;
  for (uint32_t object = 0; object < stored_objects; ++object) {
    const uint8_t* payload = nullptr;
    uint32_t rows = 0;
    if (!resolve_u24(
            u24, scratch->object_positions[object], payload, rows)) {
      return 8;
    }
    scratch->object_starts[object] = static_cast<uint32_t>(cursor);
    if (rows != 0) {
      std::size_t decoded = 0;
      const int status = u24le_decode_v1(
          payload, static_cast<std::size_t>(rows) * 3U,
          decode_scratch + cursor,
          static_cast<std::size_t>(decode_scratch_capacity - cursor),
          &decoded);
      if (status != 0 || decoded != rows) return 12;
    }
    cursor += rows;
  }
  if (cursor != decoded_total) return 12;

  uint64_t segment_input_rows = 0;
  for (uint32_t segment = 0; segment < segment_count; ++segment) {
    const uint32_t local = scratch->segment_local_objects[segment];
    const uint32_t object_end = local + 1U < stored_objects
        ? scratch->object_starts[local + 1U]
        : static_cast<uint32_t>(decoded_total);
    scratch->positions[segment] = scratch->object_starts[local];
    segment_input_rows += object_end - scratch->object_starts[local];
  }

  uint64_t boundary_checks = 0;
  auto advance = [&](uint8_t segment) noexcept -> int {
    const uint32_t global_segment = begin + segment;
    const uint32_t local = scratch->segment_local_objects[segment];
    const uint32_t object_end = local + 1U < stored_objects
        ? scratch->object_starts[local + 1U]
        : static_cast<uint32_t>(decoded_total);
    while (scratch->positions[segment] < object_end) {
      const int32_t id = decode_scratch[scratch->positions[segment]++];
      if (id < 0 || static_cast<uint64_t>(id) >= base_rows) return 4;
      if (scratch->has_previous_raw[segment] != 0 &&
          id <= scratch->previous_raw[segment]) return 5;
      scratch->previous_raw[segment] = id;
      scratch->has_previous_raw[segment] = 1;

      const uint8_t flag = descriptor.owner[
          descriptor.segment_flags_offset + global_segment];
      bool admitted = flag == 0;
      if (flag != 0) {
        const uint32_t clause = static_cast<uint32_t>(route_index) *
            descriptor.clauses_per_route + flag - 1U;
        const uint8_t* record = descriptor.owner +
            descriptor.clauses_offset + kClauseRecordBytes * clause;
        const uint32_t attribute = load_u32(record);
        const double lower = load_f64(record + 4);
        const double upper = load_f64(record + 12);
        uintptr_t values_raw = 0, valid_raw = 0;
        uint8_t kind = 0;
        if (!attribute_descriptor(
                attribute, value0, valid0, kind0, value1, valid1, kind1,
                value2, valid2, kind2,
                values_raw, valid_raw, kind)) return 9;
        ++boundary_checks;
        const auto* valid = reinterpret_cast<const uint8_t*>(valid_raw);
        double value = 0.0;
        if (kind == 1) {
          value = reinterpret_cast<const double*>(values_raw)[id];
        } else {
          value = static_cast<double>(
              reinterpret_cast<const int64_t*>(values_raw)[id]);
        }
        admitted = valid[id] != 0 && std::isfinite(value) &&
                   value >= lower && value < upper;
      }
      if (admitted) {
        scratch->current[segment] = id;
        scratch->live[segment] = 1;
        return 0;
      }
    }
    scratch->live[segment] = 0;
    return 0;
  };

  std::size_t heap_size = 0;
  for (uint32_t segment = 0; segment < segment_count; ++segment) {
    const int status = advance(static_cast<uint8_t>(segment));
    if (status != 0) return status;
    if (scratch->live[segment] != 0) {
      heap_push(static_cast<uint8_t>(segment), scratch->heap, heap_size,
                scratch->current);
    }
  }

  std::size_t batch_size = 0;
  std::size_t best_size = 0;
  uint64_t unique_before = 0;
  uint64_t unique_after = 0;
  uint64_t self_occurrences = 0;
  while (heap_size != 0) {
    const int32_t next_id = scratch->current[scratch->heap[0]];
    uint64_t equal_streams = 0;
    do {
      const uint8_t segment = heap_pop(
          scratch->heap, heap_size, scratch->current);
      ++equal_streams;
      const int status = advance(segment);
      if (status != 0) return status;
      if (scratch->live[segment] != 0) {
        heap_push(segment, scratch->heap, heap_size, scratch->current);
      }
    } while (heap_size != 0 &&
             scratch->current[scratch->heap[0]] == next_id);
    ++unique_before;
    if (next_id == self_id) {
      self_occurrences += equal_streams;
      continue;
    }
    if (diagnostic_support_after != nullptr) {
      diagnostic_support_after[unique_after] = next_id;
    }
    scratch->batch_ids[batch_size++] = next_id;
    ++unique_after;
    if (batch_size == kBatchIds) {
      const int status = score_batch(
          base, base_rows, query, scratch->batch_ids, batch_size,
          scratch->best, best_size, scratch->vectors, scratch->future,
          scratch->distances);
      if (status != 0) return status;
      batch_size = 0;
    }
  }
  if (self_occurrences == 0 || unique_before != unique_after + 1U) return 10;
  if (batch_size != 0) {
    const int status = score_batch(
        base, base_rows, query, scratch->batch_ids, batch_size,
        scratch->best, best_size, scratch->vectors, scratch->future,
        scratch->distances);
    if (status != 0) return status;
  }
  if (unique_after < kFbTopK || best_size != kFbTopK) return 3;

  std::sort(scratch->best, scratch->best + best_size, candidate_less);
  for (std::size_t rank = 0; rank < kFbTopK; ++rank) {
    output_top_ids[rank] = scratch->best[rank].id;
    output_top_squared_l2[rank] = scratch->best[rank].distance;
  }
  output_audit->decoded_input_rows = decoded_total;
  output_audit->segment_input_rows = segment_input_rows;
  output_audit->boundary_predicate_checks = boundary_checks;
  output_audit->support_before_leave_one_out = unique_before;
  output_audit->support_after_leave_one_out = unique_after;
  output_audit->self_fragment_occurrences = self_occurrences;
  return 0;
}

}  // extern "C"
