#pragma once

// Allocation-free two-pass exact prefix cascade for an immutable packed
// primary posting.  Pass 1 reads/prefetches only the first 128 bytes of every
// complete-support member.  Pass 2 reads the final 64 bytes only for members
// whose exact prefix can still improve canonical (distance,global-ID) top-K.
// No pair row, query outcome, or baseline state is materialized here.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "pair_offset16_kernel.h"

namespace yfcc_pair_offset16_prefix128_cascade_v1 {

namespace base = yfcc_pair_offset16_kernel_candidate_v2;

constexpr std::size_t kDimensions = base::kDimensions;
constexpr std::size_t kPrefixBytes = 128;
constexpr std::size_t kSuffixBytes = 64;
constexpr std::size_t kPrefetchDistance = 16;

struct Survivor {
  std::uint32_t prefix_distance = 0;
  std::int32_t global_id = -1;
  std::uint16_t local_offset = 0;
  std::uint16_t reserved = 0;
};
static_assert(sizeof(Survivor) == 12, "prefix survivor layout differs");

struct Counters {
  std::uint64_t prefix_reads = 0;
  std::uint64_t suffix_reads = 0;
  std::uint64_t seed_exact_reads = 0;
  std::uint64_t prefix_survivors = 0;
  std::uint64_t dynamic_suffix_skips = 0;
};

inline std::uint32_t horizontal_sum(__m256i value) noexcept {
  __m128i sum = _mm_add_epi32(
      _mm256_castsi256_si128(value),
      _mm256_extracti128_si256(value, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return static_cast<std::uint32_t>(_mm_cvtsi128_si32(sum));
}

inline std::uint32_t exact_prefix128(const std::uint8_t* left,
                                    const std::uint8_t* right) noexcept {
#if defined(__AVX2__)
  __m256i even = _mm256_setzero_si256();
  __m256i odd = _mm256_setzero_si256();
  base::accumulate_32_u8_l2(left, right, even);
  base::accumulate_32_u8_l2(left + 32, right + 32, odd);
  base::accumulate_32_u8_l2(left + 64, right + 64, even);
  base::accumulate_32_u8_l2(left + 96, right + 96, odd);
  return horizontal_sum(_mm256_add_epi32(even, odd));
#else
  std::uint32_t total = 0;
  for (std::size_t i = 0; i < kPrefixBytes; ++i) {
    const std::int32_t difference =
        static_cast<std::int32_t>(left[i]) -
        static_cast<std::int32_t>(right[i]);
    total += static_cast<std::uint32_t>(difference * difference);
  }
  return total;
#endif
}

inline std::uint32_t exact_suffix64(const std::uint8_t* left,
                                   const std::uint8_t* right) noexcept {
#if defined(__AVX2__)
  __m256i accumulator = _mm256_setzero_si256();
  base::accumulate_32_u8_l2(left, right, accumulator);
  base::accumulate_32_u8_l2(left + 32, right + 32, accumulator);
  return horizontal_sum(accumulator);
#else
  std::uint32_t total = 0;
  for (std::size_t i = 0; i < kSuffixBytes; ++i) {
    const std::int32_t difference =
        static_cast<std::int32_t>(left[i]) -
        static_cast<std::int32_t>(right[i]);
    total += static_cast<std::uint32_t>(difference * difference);
  }
  return total;
#endif
}

inline void prefetch_prefix128(const std::uint8_t* row) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(row, 0, 3);
  __builtin_prefetch(row + 64, 0, 3);
#else
  (void)row;
#endif
}

inline void prefetch_suffix64(const std::uint8_t* row) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(row + kPrefixBytes, 0, 3);
#else
  (void)row;
#endif
}

template <std::size_t K>
inline std::array<std::int32_t, K>
scan_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query, Survivor* scratch,
    std::size_t scratch_capacity, Counters* counters = nullptr) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<base::DistanceId, K> frontier{};
  Counters local{};
  const bool count = counters != nullptr;
  if (scratch_capacity < pair_size || pair_offsets == nullptr ||
      primary_global_ids == nullptr ||
      primary_packed == nullptr || query == nullptr || scratch == nullptr) {
    std::array<std::int32_t, K> failure;
    failure.fill(-1);
    return failure;
  }

  // Up to K exact seed rows establish a valid upper bound without an
  // all-support selection/sort.  This also preserves the complete-support
  // result when the intersection contains fewer than K rows.
  const std::size_t seed_count = std::min(pair_size, K);
  for (std::size_t index = 0; index < seed_count; ++index) {
    const std::size_t offset = pair_offsets[index];
    const std::int32_t id = primary_global_ids[offset];
    const std::uint8_t* row = primary_packed + offset * vector_stride;
    base::insert_top_k<K>(
        base::DistanceId{base::squared_l2_u8_192(query, row), id}, frontier);
  }
  if (count) {
    local.seed_exact_reads = seed_count;
    local.prefix_reads = seed_count;
    local.suffix_reads = seed_count;
  }

  const std::size_t initial =
      std::min(pair_size, seed_count + kPrefetchDistance);
  for (std::size_t index = seed_count; index < initial; ++index) {
    const std::size_t offset = pair_offsets[index];
    prefetch_prefix128(primary_packed + offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(primary_global_ids + offset, 0, 3);
#endif
  }

  std::size_t survivor_count = 0;
  for (std::size_t index = seed_count; index < pair_size; ++index) {
    const std::size_t future = index + kPrefetchDistance;
    if (future < pair_size) {
      const std::size_t offset = pair_offsets[future];
      prefetch_prefix128(primary_packed + offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(primary_global_ids + offset, 0, 3);
#endif
    }
    const std::uint16_t offset = pair_offsets[index];
    const std::int32_t id = primary_global_ids[offset];
    const std::uint8_t* row =
        primary_packed + static_cast<std::size_t>(offset) * vector_stride;
    const std::uint32_t partial = exact_prefix128(query, row);
    if (count) ++local.prefix_reads;
    if (base::DistanceId{partial, id} < frontier[K - 1]) {
      scratch[survivor_count++] = Survivor{partial, id, offset, 0};
    }
  }
  if (count) local.prefix_survivors = survivor_count;

  const std::size_t initial_suffix =
      std::min(survivor_count, kPrefetchDistance);
  for (std::size_t index = 0; index < initial_suffix; ++index) {
    prefetch_suffix64(
        primary_packed +
        static_cast<std::size_t>(scratch[index].local_offset) * vector_stride);
  }
  for (std::size_t index = 0; index < survivor_count; ++index) {
    const std::size_t future = index + kPrefetchDistance;
    if (future < survivor_count) {
      prefetch_suffix64(
          primary_packed +
          static_cast<std::size_t>(scratch[future].local_offset) *
              vector_stride);
    }
    const Survivor candidate = scratch[index];
    if (!(base::DistanceId{candidate.prefix_distance, candidate.global_id} <
          frontier[K - 1])) {
      if (count) ++local.dynamic_suffix_skips;
      continue;
    }
    const std::uint8_t* row =
        primary_packed +
        static_cast<std::size_t>(candidate.local_offset) * vector_stride;
    const std::uint32_t distance = candidate.prefix_distance +
        exact_suffix64(query + kPrefixBytes, row + kPrefixBytes);
    if (count) ++local.suffix_reads;
    base::insert_top_k<K>(
        base::DistanceId{distance, candidate.global_id}, frontier);
  }
  if (count) *counters = local;

  std::array<std::int32_t, K> result;
  result.fill(-1);
  for (std::size_t index = 0; index < K; ++index) {
    if (frontier[index].distance !=
        std::numeric_limits<std::uint32_t>::max()) {
      result[index] = frontier[index].global_id;
    }
  }
  return result;
}

template <std::size_t K>
inline std::array<std::int32_t, K>
scan_checked(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids, std::size_t primary_size,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query, Survivor* scratch,
    std::size_t scratch_capacity, Counters* counters = nullptr) {
  if (pair_size > primary_size || primary_size > base::kMaximumPrimarySupport ||
      vector_stride != kDimensions || pair_offsets == nullptr ||
      primary_global_ids == nullptr || primary_packed == nullptr ||
      query == nullptr || scratch == nullptr ||
      scratch_capacity < pair_size) {
    throw std::invalid_argument("prefix128 cascade input contract differs");
  }
  std::int32_t previous_id = -1;
  for (std::size_t index = 0; index < primary_size; ++index) {
    if (primary_global_ids[index] <= previous_id) {
      throw std::invalid_argument("prefix128 primary IDs are not strict");
    }
    previous_id = primary_global_ids[index];
  }
  std::size_t previous_offset = 0;
  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t offset = pair_offsets[index];
    if (offset >= primary_size || (index != 0 && offset <= previous_offset)) {
      throw std::invalid_argument("prefix128 offsets are not strict/in-range");
    }
    previous_offset = offset;
  }
  return scan_prevalidated<K>(
      pair_offsets, pair_size, primary_global_ids, primary_packed,
      vector_stride, query, scratch, scratch_capacity, counters);
}

}  // namespace yfcc_pair_offset16_prefix128_cascade_v1
