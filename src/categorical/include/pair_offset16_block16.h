#pragma once

// Allocation-free exact support scan in fixed 16-candidate cache-line blocks.
// Each block exposes memory-level parallelism without prefetching lines that a
// valid 64B/128B lower bound has already rejected.  Survivors are compacted in
// a fixed stack array; no query-sized survivor sweep or allocation is used.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "pair_offset16_kernel.h"

namespace yfcc_pair_offset16_block16_cascade_v2 {

namespace base = yfcc_pair_offset16_kernel_candidate_v2;

constexpr std::size_t kDimensions = base::kDimensions;
constexpr std::size_t kLineBytes = 64;
constexpr std::size_t kBlockCandidates = 16;

struct Candidate {
  std::uint32_t partial_distance = 0;
  std::int32_t global_id = -1;
  std::uint16_t local_offset = 0;
  std::uint16_t reserved = 0;
};
static_assert(sizeof(Candidate) == 12, "block16 candidate layout differs");

struct Counters {
  std::uint64_t seed_exact_reads = 0;
  std::uint64_t line0_reads = 0;
  std::uint64_t line1_reads = 0;
  std::uint64_t line2_reads = 0;
  std::uint64_t line0_survivors = 0;
  std::uint64_t line1_survivors = 0;
  std::uint64_t dynamic_line2_skips = 0;
};

inline std::uint32_t horizontal_sum(__m256i value) noexcept {
  __m128i sum = _mm_add_epi32(
      _mm256_castsi256_si128(value),
      _mm256_extracti128_si256(value, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return static_cast<std::uint32_t>(_mm_cvtsi128_si32(sum));
}

inline std::uint32_t exact_line64(const std::uint8_t* left,
                                 const std::uint8_t* right) noexcept {
#if defined(__AVX2__)
  __m256i accumulator = _mm256_setzero_si256();
  base::accumulate_32_u8_l2(left, right, accumulator);
  base::accumulate_32_u8_l2(left + 32, right + 32, accumulator);
  return horizontal_sum(accumulator);
#else
  std::uint32_t total = 0;
  for (std::size_t dimension = 0; dimension < kLineBytes; ++dimension) {
    const std::int32_t difference =
        static_cast<std::int32_t>(left[dimension]) -
        static_cast<std::int32_t>(right[dimension]);
    total += static_cast<std::uint32_t>(difference * difference);
  }
  return total;
#endif
}

inline void prefetch_line(const std::uint8_t* row,
                          std::size_t line_offset) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(row + line_offset, 0, 3);
#else
  (void)row;
  (void)line_offset;
#endif
}

template <std::size_t K, bool Count = false>
inline std::array<std::int32_t, K> scan_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query, Counters* counters = nullptr) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<base::DistanceId, K> frontier{};
  std::array<Candidate, kBlockCandidates> staged{};
  Counters local{};

  if (pair_offsets == nullptr || primary_global_ids == nullptr ||
      primary_packed == nullptr || query == nullptr) {
    std::array<std::int32_t, K> failure;
    failure.fill(-1);
    return failure;
  }

  const std::size_t seed_count = std::min(pair_size, K);
  for (std::size_t index = 0; index < seed_count; ++index) {
    const std::size_t offset = pair_offsets[index];
    const std::int32_t id = primary_global_ids[offset];
    const std::uint8_t* row = primary_packed + offset * vector_stride;
    base::insert_top_k<K>(
        base::DistanceId{base::squared_l2_u8_192(query, row), id}, frontier);
  }
  if constexpr (Count) {
    local.seed_exact_reads = seed_count;
    local.line0_reads = seed_count;
    local.line1_reads = seed_count;
    local.line2_reads = seed_count;
  }

  for (std::size_t block_begin = seed_count; block_begin < pair_size;
       block_begin += kBlockCandidates) {
    const std::size_t block_size =
        std::min(kBlockCandidates, pair_size - block_begin);

    // Stage 0: issue the first-line requests for the whole fixed block, then
    // compute exact 64B lower bounds and compact survivors in-place.
    for (std::size_t lane = 0; lane < block_size; ++lane) {
      const std::size_t offset = pair_offsets[block_begin + lane];
      prefetch_line(primary_packed + offset * vector_stride, 0);
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(primary_global_ids + offset, 0, 3);
#endif
    }
    std::size_t survivor_count = 0;
    for (std::size_t lane = 0; lane < block_size; ++lane) {
      const std::uint16_t offset = pair_offsets[block_begin + lane];
      const std::int32_t id = primary_global_ids[offset];
      const std::uint8_t* row =
          primary_packed + static_cast<std::size_t>(offset) * vector_stride;
      const std::uint32_t partial = exact_line64(query, row);
      if constexpr (Count) ++local.line0_reads;
      if (base::DistanceId{partial, id} < frontier[K - 1]) {
        staged[survivor_count++] = Candidate{partial, id, offset, 0};
      }
    }
    if constexpr (Count) local.line0_survivors += survivor_count;

    // Stage 1: request only line 1 of line-0 survivors.  Reading from a later
    // slot and writing to an earlier/equal slot makes compaction alias-safe.
    for (std::size_t lane = 0; lane < survivor_count; ++lane) {
      const std::size_t offset = staged[lane].local_offset;
      prefetch_line(primary_packed + offset * vector_stride, kLineBytes);
    }
    std::size_t line1_survivor_count = 0;
    for (std::size_t lane = 0; lane < survivor_count; ++lane) {
      Candidate candidate = staged[lane];
      const std::uint8_t* row =
          primary_packed +
          static_cast<std::size_t>(candidate.local_offset) * vector_stride;
      candidate.partial_distance +=
          exact_line64(query + kLineBytes, row + kLineBytes);
      if constexpr (Count) ++local.line1_reads;
      if (base::DistanceId{candidate.partial_distance,
                           candidate.global_id} < frontier[K - 1]) {
        staged[line1_survivor_count++] = candidate;
      }
    }
    if constexpr (Count) {
      local.line1_survivors += line1_survivor_count;
    }

    // Stage 2: only remaining candidates request line 2.  Earlier lanes may
    // tighten the frontier, so recheck the exact 128B lower bound before use.
    for (std::size_t lane = 0; lane < line1_survivor_count; ++lane) {
      const std::size_t offset = staged[lane].local_offset;
      prefetch_line(primary_packed + offset * vector_stride, 2 * kLineBytes);
    }
    for (std::size_t lane = 0; lane < line1_survivor_count; ++lane) {
      const Candidate candidate = staged[lane];
      if (!(base::DistanceId{candidate.partial_distance,
                             candidate.global_id} < frontier[K - 1])) {
        if constexpr (Count) ++local.dynamic_line2_skips;
        continue;
      }
      const std::uint8_t* row =
          primary_packed +
          static_cast<std::size_t>(candidate.local_offset) * vector_stride;
      const std::uint32_t distance = candidate.partial_distance +
          exact_line64(query + 2 * kLineBytes, row + 2 * kLineBytes);
      if constexpr (Count) ++local.line2_reads;
      base::insert_top_k<K>(
          base::DistanceId{distance, candidate.global_id}, frontier);
    }
  }

  if constexpr (Count) {
    if (counters != nullptr) *counters = local;
  }
  std::array<std::int32_t, K> result;
  result.fill(-1);
  for (std::size_t rank = 0; rank < K; ++rank) {
    if (frontier[rank].distance !=
        std::numeric_limits<std::uint32_t>::max()) {
      result[rank] = frontier[rank].global_id;
    }
  }
  return result;
}

template <std::size_t K, bool Count = false>
inline std::array<std::int32_t, K> scan_checked(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids, std::size_t primary_size,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query, Counters* counters = nullptr) {
  if (pair_size > primary_size ||
      primary_size > base::kMaximumPrimarySupport ||
      vector_stride != kDimensions || pair_offsets == nullptr ||
      primary_global_ids == nullptr || primary_packed == nullptr ||
      query == nullptr) {
    throw std::invalid_argument("block16 cascade input contract differs");
  }
  std::int32_t previous_id = -1;
  for (std::size_t index = 0; index < primary_size; ++index) {
    if (primary_global_ids[index] <= previous_id) {
      throw std::invalid_argument("block16 primary IDs are not strict");
    }
    previous_id = primary_global_ids[index];
  }
  std::size_t previous_offset = 0;
  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t offset = pair_offsets[index];
    if (offset >= primary_size || (index != 0 && offset <= previous_offset)) {
      throw std::invalid_argument("block16 offsets are not strict/in-range");
    }
    previous_offset = offset;
  }
  return scan_prevalidated<K, Count>(
      pair_offsets, pair_size, primary_global_ids, primary_packed,
      vector_stride, query, counters);
}

}  // namespace yfcc_pair_offset16_block16_cascade_v2
