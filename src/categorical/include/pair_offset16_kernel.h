#pragma once

// Exact YFCC pair-offset16 scan kernel.
//
// Scientific contract:
//   uint16 local offset -> immutable primary posting global ID
//   -> immutable 192-byte base vector -> exact integer squared L2
//   -> top-K ordered by (distance, global ID).
//
// The catalog admission/orientation rule is intentionally outside this file.
// This kernel receives only a prevalidated pair row.  It has no query-vocabulary,
// query-frequency, ground-truth, or prior-outcome input.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace yfcc_pair_offset16_kernel_candidate_v2 {

constexpr std::size_t kDimensions = 192;
constexpr std::size_t kMaximumPrimarySupport = 65'536;
constexpr std::size_t kHighTagBitmapThreshold = 65'536;
constexpr std::size_t kVectorPrefetchDistance = 16;
constexpr std::size_t kGlobalIdPrefetchDistance = 32;
constexpr std::uint32_t kMaximumSquaredL2 =
    static_cast<std::uint32_t>(kDimensions * 255U * 255U);

static_assert(kMaximumSquaredL2 == 12'484'800U,
              "YFCC exact-distance bound changed");
static_assert(kMaximumSquaredL2 <
                  static_cast<std::uint32_t>(
                      std::numeric_limits<std::int32_t>::max()),
              "AVX2 signed 32-bit accumulation would overflow");

struct DistanceId {
  std::uint32_t distance = std::numeric_limits<std::uint32_t>::max();
  std::int32_t global_id = std::numeric_limits<std::int32_t>::max();

  bool operator<(const DistanceId& other) const noexcept {
    return distance < other.distance ||
           (distance == other.distance && global_id < other.global_id);
  }
};

enum class IntersectionStrategy : std::uint8_t {
  SecondaryBitmap = 0,
  SortedMerge = 1,
  GallopingPrimary = 2,
};

enum class ScratchRepresentation : std::uint8_t {
  LocalOffset16 = 0,
  GlobalId32 = 1,
};

#if defined(__AVX2__)
inline void accumulate_32_u8_l2(const std::uint8_t* left,
                                const std::uint8_t* right,
                                __m256i& accumulator) noexcept {
  const __m256i left_u8 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(left));
  const __m256i right_u8 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(right));
  const __m256i difference_low = _mm256_sub_epi16(
      _mm256_cvtepu8_epi16(_mm256_castsi256_si128(left_u8)),
      _mm256_cvtepu8_epi16(_mm256_castsi256_si128(right_u8)));
  const __m256i difference_high = _mm256_sub_epi16(
      _mm256_cvtepu8_epi16(_mm256_extracti128_si256(left_u8, 1)),
      _mm256_cvtepu8_epi16(_mm256_extracti128_si256(right_u8, 1)));
  accumulator = _mm256_add_epi32(
      accumulator,
      _mm256_add_epi32(
          _mm256_madd_epi16(difference_low, difference_low),
          _mm256_madd_epi16(difference_high, difference_high)));
}
#endif

// Exact for YFCC's fixed 192 uint8 dimensions.  Two independent accumulators
// remove the avoidable single dependency chain in candidate v1.
inline std::uint32_t squared_l2_u8_192(const std::uint8_t* left,
                                      const std::uint8_t* right) noexcept {
#if defined(__AVX2__)
  __m256i accumulator_even = _mm256_setzero_si256();
  __m256i accumulator_odd = _mm256_setzero_si256();
  for (std::size_t offset = 0; offset < kDimensions; offset += 64) {
    accumulate_32_u8_l2(
        left + offset, right + offset, accumulator_even);
    accumulate_32_u8_l2(
        left + offset + 32, right + offset + 32, accumulator_odd);
  }
  const __m256i accumulator =
      _mm256_add_epi32(accumulator_even, accumulator_odd);
  __m128i sum = _mm_add_epi32(
      _mm256_castsi256_si128(accumulator),
      _mm256_extracti128_si256(accumulator, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return static_cast<std::uint32_t>(_mm_cvtsi128_si32(sum));
#else
  std::uint32_t distance = 0;
  for (std::size_t dimension = 0; dimension < kDimensions; ++dimension) {
    const std::int32_t difference =
        static_cast<std::int32_t>(left[dimension]) -
        static_cast<std::int32_t>(right[dimension]);
    distance += static_cast<std::uint32_t>(difference * difference);
  }
  return distance;
#endif
}

// Exact staged evaluation against the current top-K boundary.  Every partial
// squared-L2 sum is a lower bound because the omitted dimensions contribute
// nonnegative terms.  Equality can be rejected only when the candidate ID
// cannot improve the canonical (distance, global-ID) boundary.
inline std::uint32_t squared_l2_u8_192_bounded(
    const std::uint8_t* left, const std::uint8_t* right,
    std::uint32_t cutoff_distance, std::int32_t cutoff_global_id,
    std::int32_t candidate_global_id, bool* rejected) noexcept {
  *rejected = false;
  if (cutoff_distance == std::numeric_limits<std::uint32_t>::max()) {
    return squared_l2_u8_192(left, right);
  }
#if defined(__AVX2__)
  std::uint32_t partial = 0;
  for (std::size_t offset = 0; offset < kDimensions; offset += 64) {
    __m256i accumulator = _mm256_setzero_si256();
    accumulate_32_u8_l2(left + offset, right + offset, accumulator);
    accumulate_32_u8_l2(
        left + offset + 32, right + offset + 32, accumulator);
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(accumulator),
        _mm256_extracti128_si256(accumulator, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    partial += static_cast<std::uint32_t>(_mm_cvtsi128_si32(sum));
    if (offset + 64 < kDimensions &&
        (partial > cutoff_distance ||
         (partial == cutoff_distance &&
          candidate_global_id >= cutoff_global_id))) {
      *rejected = true;
      return partial;
    }
  }
  return partial;
#else
  std::uint32_t partial = 0;
  for (std::size_t dimension = 0; dimension < kDimensions; ++dimension) {
    const std::int32_t difference =
        static_cast<std::int32_t>(left[dimension]) -
        static_cast<std::int32_t>(right[dimension]);
    partial += static_cast<std::uint32_t>(difference * difference);
    if ((dimension + 1 == 64 || dimension + 1 == 128) &&
        (partial > cutoff_distance ||
         (partial == cutoff_distance &&
          candidate_global_id >= cutoff_global_id))) {
      *rejected = true;
      return partial;
    }
  }
  return partial;
#endif
}

// Exact 32-byte-stage variant for a query-independent variance-ordered
// physical layout.  A permutation changes only the order of nonnegative
// squared terms, so every checkpoint remains a valid lower bound and the
// completed distance is bit-identical to the canonical 192D distance.
inline std::uint32_t squared_l2_u8_192_bounded32(
    const std::uint8_t* left, const std::uint8_t* right,
    std::uint32_t cutoff_distance, std::int32_t cutoff_global_id,
    std::int32_t candidate_global_id, bool* rejected) noexcept {
  *rejected = false;
  if (cutoff_distance == std::numeric_limits<std::uint32_t>::max()) {
    return squared_l2_u8_192(left, right);
  }
#if defined(__AVX2__)
  std::uint32_t partial = 0;
  for (std::size_t offset = 0; offset < kDimensions; offset += 32) {
    __m256i accumulator = _mm256_setzero_si256();
    accumulate_32_u8_l2(left + offset, right + offset, accumulator);
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(accumulator),
        _mm256_extracti128_si256(accumulator, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    partial += static_cast<std::uint32_t>(_mm_cvtsi128_si32(sum));
    if (offset + 32 < kDimensions &&
        (partial > cutoff_distance ||
         (partial == cutoff_distance &&
          candidate_global_id >= cutoff_global_id))) {
      *rejected = true;
      return partial;
    }
  }
  return partial;
#else
  std::uint32_t partial = 0;
  for (std::size_t dimension = 0; dimension < kDimensions; ++dimension) {
    const std::int32_t difference =
        static_cast<std::int32_t>(left[dimension]) -
        static_cast<std::int32_t>(right[dimension]);
    partial += static_cast<std::uint32_t>(difference * difference);
    if ((dimension + 1) % 32 == 0 &&
        dimension + 1 < kDimensions &&
        (partial > cutoff_distance ||
         (partial == cutoff_distance &&
          candidate_global_id >= cutoff_global_id))) {
      *rejected = true;
      return partial;
    }
  }
  return partial;
#endif
}

inline void permute_u8_192_prevalidated(
    const std::uint8_t* input, const std::size_t* dimension_order,
    std::uint8_t* output) noexcept {
  for (std::size_t dimension = 0; dimension < kDimensions; ++dimension) {
    output[dimension] = input[dimension_order[dimension]];
  }
}

// A 192-byte vector can span four cache lines when its first byte is not
// cache-line aligned.  Prefetching only the first byte, as candidate v1 did,
// does not cover a random global-base gather.
inline void prefetch_whole_vector_192(const std::uint8_t* vector) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(vector, 0, 3);
  __builtin_prefetch(vector + 64, 0, 3);
  __builtin_prefetch(vector + 128, 0, 3);
  __builtin_prefetch(vector + 191, 0, 3);
#else
  (void)vector;
#endif
}

inline void prefetch_vector_prefix_128(
    const std::uint8_t* vector) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(vector, 0, 3);
  __builtin_prefetch(vector + 64, 0, 3);
  __builtin_prefetch(vector + 127, 0, 3);
#else
  (void)vector;
#endif
}

inline void prefetch_aligned_vector_192(
    const std::uint8_t* vector) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(vector, 0, 3);
  __builtin_prefetch(vector + 64, 0, 3);
  __builtin_prefetch(vector + 128, 0, 3);
#else
  (void)vector;
#endif
}

template <std::size_t K>
inline void insert_top_k(const DistanceId& proposed,
                         std::array<DistanceId, K>& frontier) noexcept {
  static_assert(K > 0, "top-K must be positive");
  if (!(proposed < frontier[K - 1])) {
    return;
  }
  std::size_t position = K - 1;
  while (position > 0 && proposed < frontier[position - 1]) {
    frontier[position] = frontier[position - 1];
    --position;
  }
  frontier[position] = proposed;
}

// Preconditions are checked by the checked wrapper before the timed region.
// In particular, offsets and primary IDs are strict/in-range and stride is at
// least 192 bytes.  The two-stage schedule first fetches random primary-ID
// cells, then uses those IDs to fetch every line of a future random vector.
template <std::size_t K>
inline std::array<std::int32_t, K>
scan_exact_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* base_vectors, std::size_t vector_stride,
    const std::uint8_t* query) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<DistanceId, K> frontier{};

#if defined(__GNUC__) || defined(__clang__)
  const std::size_t initial_id_count =
      std::min(pair_size, kGlobalIdPrefetchDistance);
  for (std::size_t index = 0; index < initial_id_count; ++index) {
    __builtin_prefetch(
        primary_global_ids +
            static_cast<std::size_t>(pair_offsets[index]),
        0, 3);
  }
#endif
  const std::size_t initial_vector_count =
      std::min(pair_size, kVectorPrefetchDistance);
  for (std::size_t index = 0; index < initial_vector_count; ++index) {
    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    prefetch_whole_vector_192(
        base_vectors + static_cast<std::size_t>(global_id) * vector_stride);
  }

  for (std::size_t index = 0; index < pair_size; ++index) {
#if defined(__GNUC__) || defined(__clang__)
    const std::size_t future_id = index + kGlobalIdPrefetchDistance;
    if (future_id < pair_size) {
      __builtin_prefetch(
          primary_global_ids +
              static_cast<std::size_t>(pair_offsets[future_id]),
          0, 3);
    }
#endif
    const std::size_t future_vector = index + kVectorPrefetchDistance;
    if (future_vector < pair_size) {
      const std::size_t future_offset =
          static_cast<std::size_t>(pair_offsets[future_vector]);
      const std::int32_t future_global_id =
          primary_global_ids[future_offset];
      prefetch_whole_vector_192(
          base_vectors +
          static_cast<std::size_t>(future_global_id) * vector_stride);
    }

    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    const std::uint8_t* candidate =
        base_vectors + static_cast<std::size_t>(global_id) * vector_stride;
    insert_top_k<K>(
        DistanceId{squared_l2_u8_192(query, candidate), global_id},
        frontier);
  }

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

// Exact staged counterpart for local offsets that resolve into the immutable
// global base-vector array.
template <std::size_t K>
inline std::array<std::int32_t, K>
scan_exact_bounded_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* base_vectors, std::size_t vector_stride,
    const std::uint8_t* query) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<DistanceId, K> frontier{};

#if defined(__GNUC__) || defined(__clang__)
  const std::size_t initial_id_count =
      std::min(pair_size, kGlobalIdPrefetchDistance);
  for (std::size_t index = 0; index < initial_id_count; ++index) {
    __builtin_prefetch(
        primary_global_ids +
            static_cast<std::size_t>(pair_offsets[index]),
        0, 3);
  }
#endif
  const std::size_t initial_vector_count =
      std::min(pair_size, kVectorPrefetchDistance);
  for (std::size_t index = 0; index < initial_vector_count; ++index) {
    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    prefetch_whole_vector_192(
        base_vectors + static_cast<std::size_t>(global_id) * vector_stride);
  }

  for (std::size_t index = 0; index < pair_size; ++index) {
#if defined(__GNUC__) || defined(__clang__)
    const std::size_t future_id = index + kGlobalIdPrefetchDistance;
    if (future_id < pair_size) {
      __builtin_prefetch(
          primary_global_ids +
              static_cast<std::size_t>(pair_offsets[future_id]),
          0, 3);
    }
#endif
    const std::size_t future_vector = index + kVectorPrefetchDistance;
    if (future_vector < pair_size) {
      const std::size_t future_offset =
          static_cast<std::size_t>(pair_offsets[future_vector]);
      const std::int32_t future_global_id =
          primary_global_ids[future_offset];
      prefetch_whole_vector_192(
          base_vectors +
          static_cast<std::size_t>(future_global_id) * vector_stride);
    }

    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    const std::uint8_t* candidate =
        base_vectors + static_cast<std::size_t>(global_id) * vector_stride;
    bool rejected = false;
    const std::uint32_t distance = squared_l2_u8_192_bounded(
        query, candidate, frontier[K - 1].distance,
        frontier[K - 1].global_id, global_id, &rejected);
    if (!rejected) {
      insert_top_k<K>(DistanceId{distance, global_id}, frontier);
    }
  }

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

// Equivalent exact scan when the independently frozen primary atom already
// has a packed copy.  This preserves the locality used by the successful
// localbits pilot instead of needlessly converting all 2,545 packed-primary
// held-out queries back into random global-base gathers.
template <std::size_t K>
inline std::array<std::int32_t, K>
scan_exact_packed_primary_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<DistanceId, K> frontier{};

  const std::size_t initial_count =
      std::min(pair_size, kVectorPrefetchDistance);
  for (std::size_t index = 0; index < initial_count; ++index) {
    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    prefetch_whole_vector_192(
        primary_packed + local_offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(primary_global_ids + local_offset, 0, 3);
#endif
  }

  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t future = index + kVectorPrefetchDistance;
    if (future < pair_size) {
      const std::size_t future_offset =
          static_cast<std::size_t>(pair_offsets[future]);
      prefetch_whole_vector_192(
          primary_packed + future_offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(primary_global_ids + future_offset, 0, 3);
#endif
    }

    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    const std::uint8_t* candidate =
        primary_packed + local_offset * vector_stride;
    insert_top_k<K>(
        DistanceId{squared_l2_u8_192(query, candidate), global_id},
        frontier);
  }

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

// Isolated exact successor for packed primary atoms.  The support, layout,
// output order, and tie rule are identical to the reference scan; only
// candidates that cannot beat the current canonical top-K boundary omit the
// remaining 64-byte stages.
template <std::size_t K>
inline std::array<std::int32_t, K>
scan_exact_packed_primary_bounded_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<DistanceId, K> frontier{};

  const std::size_t initial_count =
      std::min(pair_size, kVectorPrefetchDistance);
  for (std::size_t index = 0; index < initial_count; ++index) {
    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    prefetch_aligned_vector_192(
        primary_packed + local_offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(primary_global_ids + local_offset, 0, 3);
#endif
  }

  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t future = index + kVectorPrefetchDistance;
    if (future < pair_size) {
      const std::size_t future_offset =
          static_cast<std::size_t>(pair_offsets[future]);
      prefetch_aligned_vector_192(
          primary_packed + future_offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(primary_global_ids + future_offset, 0, 3);
#endif
    }

    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    const std::uint8_t* candidate =
        primary_packed + local_offset * vector_stride;
    bool rejected = false;
    const std::uint32_t distance = squared_l2_u8_192_bounded(
        query, candidate, frontier[K - 1].distance,
        frontier[K - 1].global_id, global_id, &rejected);
    if (!rejected) {
      insert_top_k<K>(DistanceId{distance, global_id}, frontier);
    }
  }

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

// Same exact packed scan with 32-byte checkpoints.  It is intended for the
// variance-ordered physical layout but remains exact for any common
// permutation shared by the query and packed vectors.
template <std::size_t K>
inline std::array<std::int32_t, K>
scan_exact_packed_primary_bounded32_prevalidated(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<DistanceId, K> frontier{};

  const std::size_t initial_count =
      std::min(pair_size, kVectorPrefetchDistance);
  for (std::size_t index = 0; index < initial_count; ++index) {
    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    prefetch_aligned_vector_192(
        primary_packed + local_offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(primary_global_ids + local_offset, 0, 3);
#endif
  }

  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t future = index + kVectorPrefetchDistance;
    if (future < pair_size) {
      const std::size_t future_offset =
          static_cast<std::size_t>(pair_offsets[future]);
      prefetch_aligned_vector_192(
          primary_packed + future_offset * vector_stride);
#if defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(primary_global_ids + future_offset, 0, 3);
#endif
    }

    const std::size_t local_offset =
        static_cast<std::size_t>(pair_offsets[index]);
    const std::int32_t global_id = primary_global_ids[local_offset];
    const std::uint8_t* candidate =
        primary_packed + local_offset * vector_stride;
    bool rejected = false;
    const std::uint32_t distance = squared_l2_u8_192_bounded32(
        query, candidate, frontier[K - 1].distance,
        frontier[K - 1].global_id, global_id, &rejected);
    if (!rejected) {
      insert_top_k<K>(DistanceId{distance, global_id}, frontier);
    }
  }

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

// Query-independent online fallback for primaries without a packed atom.
// The caller owns one scratch vector per worker, reserves it to at least the
// maximum admitted primary support before timing, and reuses it across
// queries.  No pair row or query-frequency state is cached.
inline void intersect_postings_to_global_id_scratch_prevalidated(
    const std::int32_t* primary_ids, std::size_t primary_size,
    const std::int32_t* secondary_ids, std::size_t secondary_size,
    const std::uint64_t* secondary_bitmap,
    IntersectionStrategy strategy,
    std::vector<std::int32_t>& scratch) {
  scratch.clear();
  if (primary_size == 0 ||
      (strategy != IntersectionStrategy::SecondaryBitmap &&
       secondary_size == 0)) {
    return;
  }
  if (strategy == IntersectionStrategy::SecondaryBitmap) {
    for (std::size_t index = 0; index < primary_size; ++index) {
      const std::int32_t global_id = primary_ids[index];
      const std::size_t word =
          static_cast<std::size_t>(global_id) >> 6U;
      const unsigned bit =
          static_cast<unsigned>(global_id) & 63U;
      if ((secondary_bitmap[word] & (std::uint64_t{1} << bit)) != 0) {
        scratch.push_back(global_id);
      }
    }
    return;
  }

  if (strategy == IntersectionStrategy::SortedMerge) {
    std::size_t primary_index = 0;
    std::size_t secondary_index = 0;
    while (primary_index < primary_size &&
           secondary_index < secondary_size) {
      const std::int32_t primary_id = primary_ids[primary_index];
      const std::int32_t secondary_id = secondary_ids[secondary_index];
      if (primary_id < secondary_id) {
        ++primary_index;
      } else if (secondary_id < primary_id) {
        ++secondary_index;
      } else {
        scratch.push_back(primary_id);
        ++primary_index;
        ++secondary_index;
      }
    }
    return;
  }

  const std::int32_t* cursor = secondary_ids;
  const std::int32_t* const end = secondary_ids + secondary_size;
  for (std::size_t index = 0; index < primary_size; ++index) {
    cursor = std::lower_bound(cursor, end, primary_ids[index]);
    if (cursor == end) {
      break;
    }
    if (*cursor == primary_ids[index]) {
      scratch.push_back(*cursor);
      ++cursor;
    }
  }
}

// Fast representation for every oriented pair whose primary posting fits the
// lossless uint16 domain.  Bitmap and posting-merge routes both emit local
// offsets, so a packed primary can be addressed directly while a global-base
// route resolves the immutable primary ID exactly once per survivor.
inline void intersect_postings_to_local_offset16_scratch_prevalidated(
    const std::int32_t* primary_ids, std::size_t primary_size,
    const std::int32_t* secondary_ids, std::size_t secondary_size,
    const std::uint64_t* secondary_bitmap,
    IntersectionStrategy strategy,
    std::vector<std::uint16_t>& scratch) {
  scratch.clear();
  if (primary_size == 0 ||
      (strategy != IntersectionStrategy::SecondaryBitmap &&
       secondary_size == 0)) {
    return;
  }
  if (strategy == IntersectionStrategy::SecondaryBitmap) {
    for (std::size_t index = 0; index < primary_size; ++index) {
      const std::int32_t global_id = primary_ids[index];
      const std::size_t word =
          static_cast<std::size_t>(global_id) >> 6U;
      const unsigned bit =
          static_cast<unsigned>(global_id) & 63U;
      if ((secondary_bitmap[word] & (std::uint64_t{1} << bit)) != 0) {
        scratch.push_back(static_cast<std::uint16_t>(index));
      }
    }
    return;
  }

  if (strategy == IntersectionStrategy::SortedMerge) {
    std::size_t primary_index = 0;
    std::size_t secondary_index = 0;
    while (primary_index < primary_size &&
           secondary_index < secondary_size) {
      const std::int32_t primary_id = primary_ids[primary_index];
      const std::int32_t secondary_id = secondary_ids[secondary_index];
      if (primary_id < secondary_id) {
        ++primary_index;
      } else if (secondary_id < primary_id) {
        ++secondary_index;
      } else {
        scratch.push_back(
            static_cast<std::uint16_t>(primary_index));
        ++primary_index;
        ++secondary_index;
      }
    }
    return;
  }

  const std::int32_t* cursor = secondary_ids;
  const std::int32_t* const end = secondary_ids + secondary_size;
  for (std::size_t primary_index = 0;
       primary_index < primary_size; ++primary_index) {
    cursor = std::lower_bound(
        cursor, end, primary_ids[primary_index]);
    if (cursor == end) {
      break;
    }
    if (*cursor == primary_ids[primary_index]) {
      scratch.push_back(
          static_cast<std::uint16_t>(primary_index));
      ++cursor;
    }
  }
}

inline void intersect_postings_to_local_offset16_scratch_checked(
    const std::int32_t* primary_ids, std::size_t primary_size,
    const std::int32_t* secondary_ids, std::size_t secondary_size,
    const std::uint64_t* secondary_bitmap,
    IntersectionStrategy strategy,
    std::vector<std::uint16_t>& scratch) {
  if (primary_size > kMaximumPrimarySupport) {
    throw std::overflow_error(
        "primary posting does not fit local offset16");
  }
  if (strategy != IntersectionStrategy::SecondaryBitmap &&
      strategy != IntersectionStrategy::SortedMerge &&
      strategy != IntersectionStrategy::GallopingPrimary) {
    throw std::invalid_argument(
        "unknown local-offset intersection strategy");
  }
  if ((primary_size != 0 && primary_ids == nullptr) ||
      (strategy != IntersectionStrategy::SecondaryBitmap &&
       secondary_size != 0 && secondary_ids == nullptr) ||
      (strategy == IntersectionStrategy::SecondaryBitmap &&
       secondary_bitmap == nullptr)) {
    throw std::invalid_argument(
        "null local-offset intersection input");
  }
  if (scratch.capacity() < primary_size) {
    throw std::invalid_argument(
        "local-offset scratch was not reserved before timing");
  }
  intersect_postings_to_local_offset16_scratch_prevalidated(
      primary_ids, primary_size, secondary_ids, secondary_size,
      secondary_bitmap, strategy, scratch);
}

inline ScratchRepresentation choose_scratch_representation(
    std::size_t primary_support) noexcept {
  return primary_support <= kMaximumPrimarySupport
             ? ScratchRepresentation::LocalOffset16
             : ScratchRepresentation::GlobalId32;
}

inline IntersectionStrategy choose_intersection_strategy(
    std::size_t primary_size, std::size_t secondary_size,
    bool has_secondary_bitmap) noexcept {
  if (has_secondary_bitmap) {
    return IntersectionStrategy::SecondaryBitmap;
  }
  return secondary_size > primary_size * 8
             ? IntersectionStrategy::GallopingPrimary
             : IntersectionStrategy::SortedMerge;
}

// Base-only support-complement rule.  Prebuild one membership bitmap for
// every base tag above the fixed threshold.  After orienting a pair by
// nondecreasing base support, a high secondary is guaranteed to have a
// bitmap; otherwise both postings are bounded and a sorted merge is exact.
// This includes high-high pairs and therefore needs no pair-specific cache.
inline IntersectionStrategy choose_base_only_support_complement_strategy(
    std::size_t primary_support, std::size_t secondary_support,
    bool secondary_bitmap_present) {
  if (primary_support > secondary_support) {
    throw std::invalid_argument(
        "pair is not oriented by nondecreasing base support");
  }
  if (secondary_bitmap_present) {
    return IntersectionStrategy::SecondaryBitmap;
  }
  if (secondary_support > kHighTagBitmapThreshold) {
    throw std::invalid_argument(
        "base-only high-tag bitmap is missing");
  }
  return IntersectionStrategy::SortedMerge;
}

template <std::size_t K>
inline std::array<std::int32_t, K>
scan_global_id_scratch_exact_prevalidated(
    const std::int32_t* survivor_ids, std::size_t survivor_count,
    const std::uint8_t* base_vectors, std::size_t vector_stride,
    const std::uint8_t* query) noexcept {
  static_assert(K > 0, "top-K must be positive");
  std::array<DistanceId, K> frontier{};

  const std::size_t initial_count =
      std::min(survivor_count, kVectorPrefetchDistance);
  for (std::size_t index = 0; index < initial_count; ++index) {
    prefetch_whole_vector_192(
        base_vectors +
        static_cast<std::size_t>(survivor_ids[index]) * vector_stride);
  }
  for (std::size_t index = 0; index < survivor_count; ++index) {
    const std::size_t future = index + kVectorPrefetchDistance;
    if (future < survivor_count) {
      prefetch_whole_vector_192(
          base_vectors +
          static_cast<std::size_t>(survivor_ids[future]) * vector_stride);
    }
    const std::int32_t global_id = survivor_ids[index];
    const std::uint8_t* candidate =
        base_vectors + static_cast<std::size_t>(global_id) * vector_stride;
    insert_top_k<K>(
        DistanceId{squared_l2_u8_192(query, candidate), global_id},
        frontier);
  }

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

inline void validate_inputs(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids, std::size_t primary_support,
    const std::uint8_t* base_vectors, std::size_t base_points,
    std::size_t vector_stride, const std::uint8_t* query) {
  if (primary_global_ids == nullptr || base_vectors == nullptr ||
      query == nullptr || (pair_size != 0 && pair_offsets == nullptr)) {
    throw std::invalid_argument("null pair-offset scan input");
  }
  if (primary_support > kMaximumPrimarySupport ||
      vector_stride < kDimensions || base_points == 0) {
    throw std::invalid_argument(
        "invalid primary support, vector stride, or base size");
  }
  if (base_points - 1 >
      std::numeric_limits<std::size_t>::max() / vector_stride) {
    throw std::overflow_error("base vector address arithmetic overflows");
  }

  std::int32_t previous_global_id = -1;
  for (std::size_t index = 0; index < primary_support; ++index) {
    const std::int32_t global_id = primary_global_ids[index];
    if (global_id <= previous_global_id ||
        static_cast<std::size_t>(global_id) >= base_points) {
      throw std::out_of_range(
          "primary global IDs are not strict and in range");
    }
    previous_global_id = global_id;
  }

  std::size_t previous_offset = 0;
  bool have_previous_offset = false;
  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t offset =
        static_cast<std::size_t>(pair_offsets[index]);
    if (offset >= primary_support ||
        (have_previous_offset && offset <= previous_offset)) {
      throw std::out_of_range(
          "pair offsets are not strict and in range");
    }
    previous_offset = offset;
    have_previous_offset = true;
  }
}

inline void validate_packed_primary(
    const std::uint8_t* primary_packed, std::size_t primary_support,
    std::size_t vector_stride) {
  if (primary_packed == nullptr || vector_stride < kDimensions) {
    throw std::invalid_argument(
        "null or narrow packed-primary input");
  }
  if (primary_support != 0 &&
      primary_support - 1 >
          std::numeric_limits<std::size_t>::max() / vector_stride) {
    throw std::overflow_error(
        "packed-primary address arithmetic overflows");
  }
}

template <std::size_t K>
inline std::array<std::int32_t, K> scan_exact_checked(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids, std::size_t primary_support,
    const std::uint8_t* base_vectors, std::size_t base_points,
    std::size_t vector_stride, const std::uint8_t* query) {
  validate_inputs(
      pair_offsets, pair_size, primary_global_ids, primary_support,
      base_vectors, base_points, vector_stride, query);
  return scan_exact_prevalidated<K>(
      pair_offsets, pair_size, primary_global_ids, base_vectors,
      vector_stride, query);
}

template <std::size_t K>
inline std::array<std::int32_t, K> scan_exact_packed_primary_checked(
    const std::uint16_t* pair_offsets, std::size_t pair_size,
    const std::int32_t* primary_global_ids, std::size_t primary_support,
    const std::uint8_t* primary_packed, std::size_t vector_stride,
    const std::uint8_t* query) {
  // Mirror the global wrapper's structural checks for the packed address
  // domain.  Avoid this wrapper in the timed region.
  if (primary_packed == nullptr || query == nullptr ||
      primary_global_ids == nullptr ||
      (pair_size != 0 && pair_offsets == nullptr) ||
      primary_support > kMaximumPrimarySupport ||
      vector_stride < kDimensions) {
    throw std::invalid_argument("invalid packed-primary scan input");
  }
  std::int32_t previous_global_id = -1;
  for (std::size_t index = 0; index < primary_support; ++index) {
    const std::int32_t global_id = primary_global_ids[index];
    if (global_id <= previous_global_id) {
      throw std::out_of_range(
          "packed-primary global IDs are not strict and nonnegative");
    }
    previous_global_id = global_id;
  }
  std::size_t previous_offset = 0;
  bool have_previous_offset = false;
  for (std::size_t index = 0; index < pair_size; ++index) {
    const std::size_t offset =
        static_cast<std::size_t>(pair_offsets[index]);
    if (offset >= primary_support ||
        (have_previous_offset && offset <= previous_offset)) {
      throw std::out_of_range(
          "packed-primary offsets are not strict and in range");
    }
    previous_offset = offset;
    have_previous_offset = true;
  }
  validate_packed_primary(
      primary_packed, primary_support, vector_stride);
  return scan_exact_packed_primary_prevalidated<K>(
      pair_offsets, pair_size, primary_global_ids, primary_packed,
      vector_stride, query);
}

}  // namespace yfcc_pair_offset16_kernel_candidate_v2
