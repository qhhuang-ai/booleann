// Project-owned exact low-memory fixed-block compositor for LAION1M.
//
// One call decodes immutable U24 objects through the already retained decoder,
// applies clause-local boundary predicates, heap-merges and deduplicates all
// admitted rows, removes the query base exactly once, and scores the complete
// support with the retained ABI-v7 interleave-eight float32 arithmetic.
//
// This is an isolated candidate module.  It does not modify any baseline or
// retained scorer source.  Every array in the hot path is caller-owned or a
// fixed-size stack array; there is no heap allocation, I/O, locking, telemetry,
// or global mutation.

#ifndef LAION1M_RETAINED_PREFETCH_POLICY
#define LAION1M_RETAINED_PREFETCH_POLICY 2
#endif
#include "exact_topk_avx2.cpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

extern "C" int u24le_decode_v1(
    const uint8_t* src, std::size_t src_bytes, int32_t* dst,
    std::size_t dst_capacity, std::size_t* decoded_count);

namespace {

constexpr std::size_t kFbMaxObjects = 64;
constexpr std::size_t kFbMaxSegments = 64;
constexpr std::size_t kFbMaxGlobalObjects = 65535;
constexpr std::size_t kFbMaxDecodedRows = 14044;
constexpr std::size_t kFbBatchIds = 64;
constexpr std::size_t kFbTopK = 10;

struct FixedBlockAuditV1 {
  uint64_t decoded_input_rows;
  uint64_t segment_input_rows;
  uint64_t boundary_predicate_checks;
  uint64_t support_before_leave_one_out;
  uint64_t support_after_leave_one_out;
  uint64_t self_fragment_occurrences;
};

constexpr std::size_t kFbLaneDecodeScratchBytes =
    kFbMaxDecodedRows * sizeof(int32_t);
constexpr std::size_t kFbLaneOutputWorkspaceBytes =
    kFbTopK * (sizeof(int32_t) + sizeof(float)) +
    sizeof(FixedBlockAuditV1);

// Simultaneously live explicit fixed arrays in this translation unit plus the
// retained scorer's interleave-eight pointer/distance arrays.  The compiled
// stack frame is measured separately after the release build.
constexpr std::size_t kFbExplicitLocalArrayBytes =
    kFbMaxObjects * sizeof(uint16_t) +          // global object ordinals
    kFbMaxObjects * sizeof(uint32_t) +          // object_starts
    kFbMaxSegments * sizeof(uint8_t) +          // segment-local object
    kFbMaxSegments * sizeof(uint32_t) +         // positions
    kFbMaxSegments * sizeof(int32_t) +          // previous_raw
    kFbMaxSegments * sizeof(uint8_t) +          // has_previous_raw
    kFbMaxSegments * sizeof(int32_t) +          // current admitted head
    kFbMaxSegments * sizeof(uint8_t) +          // live
    kFbMaxSegments * sizeof(uint8_t) +          // heap
    kFbBatchIds * sizeof(int32_t) +             // score batch
    kFbTopK * sizeof(Candidate) +               // best
    2 * kRetainedInterleaveRows * sizeof(const float*) +
    kRetainedInterleaveRows * sizeof(float);

inline bool fb_head_less(
    uint8_t left, uint8_t right, const int32_t* current) noexcept {
  return current[left] < current[right] ||
         (current[left] == current[right] && left < right);
}

inline void fb_heap_push(
    uint8_t segment, uint8_t* heap, std::size_t& heap_size,
    const int32_t* current) noexcept {
  std::size_t child = heap_size++;
  heap[child] = segment;
  while (child != 0) {
    const std::size_t parent = (child - 1) / 2;
    if (!fb_head_less(heap[child], heap[parent], current)) {
      break;
    }
    std::swap(heap[child], heap[parent]);
    child = parent;
  }
}

inline uint8_t fb_heap_pop(
    uint8_t* heap, std::size_t& heap_size,
    const int32_t* current) noexcept {
  const uint8_t result = heap[0];
  --heap_size;
  if (heap_size == 0) {
    return result;
  }
  heap[0] = heap[heap_size];
  std::size_t parent = 0;
  while (true) {
    const std::size_t left = parent * 2 + 1;
    if (left >= heap_size) {
      break;
    }
    const std::size_t right = left + 1;
    std::size_t smallest = left;
    if (right < heap_size &&
        fb_head_less(heap[right], heap[left], current)) {
      smallest = right;
    }
    if (!fb_head_less(heap[smallest], heap[parent], current)) {
      break;
    }
    std::swap(heap[smallest], heap[parent]);
    parent = smallest;
  }
  return result;
}

inline int fb_score_batch(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* ids, uint64_t count,
    Candidate* best, std::size_t& best_size) noexcept {
  return for_each_retained_score_interleaved8(
      base, base_rows, query, ids, count,
      [&](uint64_t, int32_t id, float distance) {
        const Candidate scored{distance, id};
        if (best_size < kFbTopK) {
          best[best_size++] = scored;
          std::push_heap(best, best + best_size, candidate_less);
        } else if (candidate_less(scored, best[0])) {
          std::pop_heap(best, best + best_size, candidate_less);
          best[best_size - 1] = scored;
          std::push_heap(best, best + best_size, candidate_less);
        }
      });
}

}  // namespace

extern "C" {

uint32_t laion1m_lowmem_fixedblock_v1_abi_version() { return 1; }
uint32_t laion1m_lowmem_fixedblock_v1_max_objects() {
  return static_cast<uint32_t>(kFbMaxObjects);
}
uint32_t laion1m_lowmem_fixedblock_v1_max_segments() {
  return static_cast<uint32_t>(kFbMaxSegments);
}
uint32_t laion1m_lowmem_fixedblock_v1_max_global_objects() {
  return static_cast<uint32_t>(kFbMaxGlobalObjects);
}
uint32_t laion1m_lowmem_fixedblock_v1_max_decoded_rows() {
  return static_cast<uint32_t>(kFbMaxDecodedRows);
}
uint32_t laion1m_lowmem_fixedblock_v1_batch_ids() {
  return static_cast<uint32_t>(kFbBatchIds);
}
uint32_t laion1m_lowmem_fixedblock_v1_lane_decode_scratch_bytes() {
  return static_cast<uint32_t>(kFbLaneDecodeScratchBytes);
}
uint32_t laion1m_lowmem_fixedblock_v1_lane_output_workspace_bytes() {
  return static_cast<uint32_t>(kFbLaneOutputWorkspaceBytes);
}
uint32_t laion1m_lowmem_fixedblock_v1_explicit_local_array_bytes() {
  return static_cast<uint32_t>(kFbExplicitLocalArrayBytes);
}
uint32_t laion1m_lowmem_fixedblock_v1_audit_struct_bytes() {
  return static_cast<uint32_t>(sizeof(FixedBlockAuditV1));
}

// Status codes:
//   0 success
//   1 null pointer / output contract
//   2 base, decoded-input, or scratch-capacity bound
//   3 fewer than ten unique rows after leave-one-out
//   4 decoded global ID outside base domain
//   5 one physical fragment is not strictly increasing
//   6 non-finite retained distance
//   7 object/segment count
//   8 malformed U24 payload or object ordinal
//   9 malformed boundary descriptor
//  10 query base absent before leave-one-out
//  11 diagnostic support buffer contract
//  12 retained U24 decoder failure
int laion1m_lowmem_fixedblock_exact_top10_v1(
    const float* base, uint64_t base_rows, const float* query,
    const uint8_t* const* global_object_payloads,
    const uint32_t* global_object_rows, uint64_t global_object_count,
    const uint16_t* all_segment_global_object_ordinals,
    const uint8_t* all_segment_filter_codes,
    const double* all_lower_bounds, const double* all_upper_bounds,
    uint64_t total_segment_count, uint64_t segment_begin,
    uint64_t segment_count,
    const uintptr_t* attribute_value_ptrs,
    const uintptr_t* attribute_valid_ptrs,
    const uint8_t* attribute_value_kinds, uint64_t attribute_count,
    int32_t self_id,
    int32_t* decode_scratch, uint64_t decode_scratch_capacity,
    int32_t* output_top_ids, float* output_top_squared_l2,
    FixedBlockAuditV1* output_audit,
    int32_t* diagnostic_support_after,
    uint64_t diagnostic_support_capacity) noexcept {
  if (base == nullptr || query == nullptr ||
      global_object_payloads == nullptr || global_object_rows == nullptr ||
      all_segment_global_object_ordinals == nullptr ||
      all_segment_filter_codes == nullptr || all_lower_bounds == nullptr ||
      all_upper_bounds == nullptr || attribute_value_ptrs == nullptr ||
      attribute_valid_ptrs == nullptr || attribute_value_kinds == nullptr ||
      decode_scratch == nullptr ||
      output_top_ids == nullptr || output_top_squared_l2 == nullptr ||
      output_audit == nullptr) {
    return 1;
  }
  if (base_rows == 0 || base_rows > 0x7fffffffULL || self_id < 0 ||
      static_cast<uint64_t>(self_id) >= base_rows) {
    return 2;
  }
  if (global_object_count == 0 ||
      global_object_count > kFbMaxGlobalObjects ||
      segment_count == 0 || segment_count > kFbMaxSegments ||
      attribute_count == 0 || attribute_count > 255 ||
      segment_begin > total_segment_count ||
      segment_count > total_segment_count - segment_begin) {
    return 7;
  }

  uint16_t unique_global_objects[kFbMaxObjects];
  uint8_t segment_local_objects[kFbMaxSegments];
  uint64_t unique_object_count = 0;
  for (uint64_t segment = 0; segment < segment_count; ++segment) {
    const uint64_t descriptor = segment_begin + segment;
    const uint16_t global_object =
        all_segment_global_object_ordinals[descriptor];
    if (global_object >= global_object_count) {
      return 8;
    }
    uint64_t local_object = 0;
    while (local_object < unique_object_count &&
           unique_global_objects[local_object] != global_object) {
      ++local_object;
    }
    if (local_object == unique_object_count) {
      if (unique_object_count == kFbMaxObjects) {
        return 7;
      }
      unique_global_objects[unique_object_count++] = global_object;
    }
    segment_local_objects[segment] = static_cast<uint8_t>(local_object);

    const uint8_t filter_code = all_segment_filter_codes[descriptor];
    if (filter_code > attribute_count) {
      return 9;
    }
    if (filter_code != 0) {
      const uint64_t attribute = filter_code - 1;
      if (attribute_value_ptrs[attribute] == 0 ||
          attribute_valid_ptrs[attribute] == 0 ||
          (attribute_value_kinds[attribute] != 1 &&
           attribute_value_kinds[attribute] != 2) ||
          std::isnan(all_lower_bounds[descriptor]) ||
          std::isnan(all_upper_bounds[descriptor]) ||
          !(all_lower_bounds[descriptor] < all_upper_bounds[descriptor])) {
        return 9;
      }
    }
  }

  // Complete the prospective capacity check before touching scratch or any
  // output.  In particular, 14,045 rows cannot be silently truncated.  A
  // physical object referenced by both DNF clauses is decoded once, although
  // its two segment occurrences retain independent filter descriptors.
  uint64_t decoded_total = 0;
  for (uint64_t object = 0; object < unique_object_count; ++object) {
    const uint16_t global_object = unique_global_objects[object];
    const uint64_t rows = global_object_rows[global_object];
    if ((rows != 0 && global_object_payloads[global_object] == nullptr) ||
        rows > kFbMaxDecodedRows ||
        decoded_total > kFbMaxDecodedRows - rows) {
      return rows != 0 && global_object_payloads[global_object] == nullptr
          ? 8 : 2;
    }
    decoded_total += rows;
  }
  if (decoded_total == 0 || decoded_total > decode_scratch_capacity ||
      decoded_total > kFbMaxDecodedRows) {
    return 2;
  }
  if ((diagnostic_support_after == nullptr) !=
      (diagnostic_support_capacity == 0)) {
    return 11;
  }
  if (diagnostic_support_after != nullptr &&
      diagnostic_support_capacity < decoded_total) {
    return 11;
  }

  uint32_t object_starts[kFbMaxObjects];
  uint64_t cursor = 0;
  for (uint64_t object = 0; object < unique_object_count; ++object) {
    const uint16_t global_object = unique_global_objects[object];
    object_starts[object] = static_cast<uint32_t>(cursor);
    const uint64_t rows = global_object_rows[global_object];
    const uint64_t bytes = rows * 3;
    if (rows != 0) {
      std::size_t decoded = 0;
      const int status = u24le_decode_v1(
          global_object_payloads[global_object],
          static_cast<std::size_t>(bytes),
          decode_scratch + cursor,
          static_cast<std::size_t>(decode_scratch_capacity - cursor),
          &decoded);
      if (status != 0 || decoded != rows) {
        return 12;
      }
    }
    cursor += rows;
  }
  if (cursor != decoded_total) {
    return 12;
  }

  uint32_t positions[kFbMaxSegments];
  int32_t previous_raw[kFbMaxSegments];
  uint8_t has_previous_raw[kFbMaxSegments] = {};
  int32_t current[kFbMaxSegments];
  uint8_t live[kFbMaxSegments] = {};
  uint8_t heap[kFbMaxSegments];
  std::size_t heap_size = 0;
  uint64_t boundary_checks = 0;
  uint64_t segment_input_rows = 0;

  for (uint64_t segment = 0; segment < segment_count; ++segment) {
    const uint8_t local_object = segment_local_objects[segment];
    const uint16_t global_object = unique_global_objects[local_object];
    positions[segment] = object_starts[local_object];
    segment_input_rows += global_object_rows[global_object];
  }

  auto advance = [&](uint8_t segment) noexcept -> int {
    const uint64_t descriptor = segment_begin + segment;
    const uint8_t local_object = segment_local_objects[segment];
    const uint16_t global_object = unique_global_objects[local_object];
    const uint32_t end = object_starts[local_object] +
        global_object_rows[global_object];
    while (positions[segment] < end) {
      const int32_t id = decode_scratch[positions[segment]++];
      if (id < 0 || static_cast<uint64_t>(id) >= base_rows) {
        return 4;
      }
      if (has_previous_raw[segment] != 0 &&
          id <= previous_raw[segment]) {
        return 5;
      }
      previous_raw[segment] = id;
      has_previous_raw[segment] = 1;

      const uint8_t filter_code = all_segment_filter_codes[descriptor];
      bool admitted = filter_code == 0;
      if (filter_code != 0) {
        const uint64_t attribute = filter_code - 1;
        ++boundary_checks;
        const auto* valid = reinterpret_cast<const uint8_t*>(
            attribute_valid_ptrs[attribute]);
        double value = 0.0;
        if (attribute_value_kinds[attribute] == 1) {
          const auto* values = reinterpret_cast<const double*>(
              attribute_value_ptrs[attribute]);
          value = values[static_cast<uint64_t>(id)];
        } else {
          const auto* values = reinterpret_cast<const int64_t*>(
              attribute_value_ptrs[attribute]);
          value = static_cast<double>(values[static_cast<uint64_t>(id)]);
        }
        admitted = valid[static_cast<uint64_t>(id)] != 0 &&
                   std::isfinite(value) &&
                   value >= all_lower_bounds[descriptor] &&
                   value < all_upper_bounds[descriptor];
      }
      if (admitted) {
        current[segment] = id;
        live[segment] = 1;
        return 0;
      }
    }
    live[segment] = 0;
    return 0;
  };

  for (uint64_t segment = 0; segment < segment_count; ++segment) {
    const int status = advance(static_cast<uint8_t>(segment));
    if (status != 0) {
      return status;
    }
    if (live[segment] != 0) {
      fb_heap_push(
          static_cast<uint8_t>(segment), heap, heap_size, current);
    }
  }

  int32_t batch_ids[kFbBatchIds];
  std::size_t batch_size = 0;
  Candidate best[kFbTopK];
  std::size_t best_size = 0;
  uint64_t unique_before = 0;
  uint64_t unique_after = 0;
  uint64_t self_occurrences = 0;

  while (heap_size != 0) {
    const int32_t next_id = current[heap[0]];
    uint64_t equal_streams = 0;
    do {
      const uint8_t segment = fb_heap_pop(heap, heap_size, current);
      ++equal_streams;
      const int status = advance(segment);
      if (status != 0) {
        return status;
      }
      if (live[segment] != 0) {
        fb_heap_push(segment, heap, heap_size, current);
      }
    } while (heap_size != 0 && current[heap[0]] == next_id);

    ++unique_before;
    if (next_id == self_id) {
      self_occurrences += equal_streams;
      continue;
    }
    if (diagnostic_support_after != nullptr) {
      diagnostic_support_after[unique_after] = next_id;
    }
    batch_ids[batch_size++] = next_id;
    ++unique_after;
    if (batch_size == kFbBatchIds) {
      const int status = fb_score_batch(
          base, base_rows, query, batch_ids, batch_size, best, best_size);
      if (status != 0) {
        return status;
      }
      batch_size = 0;
    }
  }

  if (self_occurrences == 0 || unique_before != unique_after + 1) {
    return 10;
  }
  if (batch_size != 0) {
    const int status = fb_score_batch(
        base, base_rows, query, batch_ids, batch_size, best, best_size);
    if (status != 0) {
      return status;
    }
  }
  if (unique_after < kFbTopK || best_size != kFbTopK) {
    return 3;
  }

  std::sort(best, best + best_size, candidate_less);
  for (std::size_t rank = 0; rank < kFbTopK; ++rank) {
    output_top_ids[rank] = best[rank].id;
    output_top_squared_l2[rank] = best[rank].distance;
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
