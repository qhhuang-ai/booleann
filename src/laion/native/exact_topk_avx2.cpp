// Project-owned ABI-v7 eight-row interleaved retained scorer for LAION1M.
// The legacy ABI-v6 causal control remains in laion1m_exact_topk_avx2_v1.cpp.
//
// Project-owned fixed-scratch top-64 candidate kernel for LAION1M
// float32[*,512].  Python re-scores the shortlist with the retained NumPy
// arithmetic and falls back to a full retained scan unless a conservative
// roundoff-gap certificate proves that the shortlist contains exact top-10.
// The hot call performs no allocation, locking, I/O, or telemetry.  Its
// support input is one already-deduplicated, strictly increasing int32 span.

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace {

constexpr std::size_t kDimension = 512;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kRetainedCutoff = kTopK + 1;
constexpr std::size_t kShortlist = 64;
constexpr std::size_t kMaxSegments = 64;
constexpr std::size_t kPrefetchDistance = 16;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kPrefetchLines = 2;

#ifndef LAION1M_BUILD_LEGACY_V6
constexpr std::size_t kRetainedInterleaveRows = 8;

// 0: the retained v6 two-line T0 lookahead control.
// 1: one same-offset cache line per future row, interleaved with each
//    16-float group and hinted into T1.
// 2: the same blocked schedule with the NTA hint.
#ifndef LAION1M_RETAINED_PREFETCH_POLICY
#define LAION1M_RETAINED_PREFETCH_POLICY 0
#endif
constexpr uint32_t kRetainedPrefetchPolicy =
    LAION1M_RETAINED_PREFETCH_POLICY;
static_assert(
    kRetainedPrefetchPolicy <= 2,
    "LAION1M_RETAINED_PREFETCH_POLICY must be 0, 1, or 2");
#endif

std::atomic<int> g_active_calls{0};
std::atomic<int> g_maximum_active_calls{0};

class ActiveCall {
 public:
  ActiveCall() {
    const int active =
        g_active_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    int observed = g_maximum_active_calls.load(std::memory_order_relaxed);
    while (active > observed &&
           !g_maximum_active_calls.compare_exchange_weak(
               observed, active, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  ~ActiveCall() {
    g_active_calls.fetch_sub(1, std::memory_order_relaxed);
  }

  ActiveCall(const ActiveCall&) = delete;
  ActiveCall& operator=(const ActiveCall&) = delete;
};

struct Candidate {
  float distance;
  int32_t id;
};

inline bool candidate_less(const Candidate& left,
                           const Candidate& right) noexcept {
  return left.distance < right.distance ||
         (left.distance == right.distance && left.id < right.id);
}

inline float squared_l2_f32_512_avx2(
    const float* __restrict__ query,
    const float* __restrict__ vector) noexcept {
  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  __m256 accumulator2 = _mm256_setzero_ps();
  __m256 accumulator3 = _mm256_setzero_ps();
  for (std::size_t offset = 0; offset < kDimension; offset += 32) {
    const __m256 d0 = _mm256_sub_ps(
        _mm256_loadu_ps(vector + offset),
        _mm256_loadu_ps(query + offset));
    const __m256 d1 = _mm256_sub_ps(
        _mm256_loadu_ps(vector + offset + 8),
        _mm256_loadu_ps(query + offset + 8));
    const __m256 d2 = _mm256_sub_ps(
        _mm256_loadu_ps(vector + offset + 16),
        _mm256_loadu_ps(query + offset + 16));
    const __m256 d3 = _mm256_sub_ps(
        _mm256_loadu_ps(vector + offset + 24),
        _mm256_loadu_ps(query + offset + 24));
    accumulator0 =
        _mm256_add_ps(accumulator0, _mm256_mul_ps(d0, d0));
    accumulator1 =
        _mm256_add_ps(accumulator1, _mm256_mul_ps(d1, d1));
    accumulator2 =
        _mm256_add_ps(accumulator2, _mm256_mul_ps(d2, d2));
    accumulator3 =
        _mm256_add_ps(accumulator3, _mm256_mul_ps(d3, d3));
  }
  const __m256 pair01 = _mm256_add_ps(accumulator0, accumulator1);
  const __m256 pair23 = _mm256_add_ps(accumulator2, accumulator3);
  const __m256 total = _mm256_add_ps(pair01, pair23);
  __m128 sum = _mm_add_ps(
      _mm256_castps256_ps128(total),
      _mm256_extractf128_ps(total, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

// Bitwise reproduction of NumPy 2.0.2 c_einsum's contiguous float32
// sum-of-products kernel after an out-of-place float32 subtraction.  That
// kernel uses one four-lane accumulator and visits each 16-float block in
// lane groups 12, 8, 4, 0 before two horizontal adds.
inline float squared_l2_f32_512_retained(
    const float* __restrict__ query,
    const float* __restrict__ vector) noexcept {
  __m128 accumulator = _mm_setzero_ps();
  for (std::size_t offset = 0; offset < kDimension; offset += 16) {
    const __m128 d12 = _mm_sub_ps(
        _mm_loadu_ps(vector + offset + 12),
        _mm_loadu_ps(query + offset + 12));
    accumulator = _mm_add_ps(accumulator, _mm_mul_ps(d12, d12));
    const __m128 d8 = _mm_sub_ps(
        _mm_loadu_ps(vector + offset + 8),
        _mm_loadu_ps(query + offset + 8));
    accumulator = _mm_add_ps(accumulator, _mm_mul_ps(d8, d8));
    const __m128 d4 = _mm_sub_ps(
        _mm_loadu_ps(vector + offset + 4),
        _mm_loadu_ps(query + offset + 4));
    accumulator = _mm_add_ps(accumulator, _mm_mul_ps(d4, d4));
    const __m128 d0 = _mm_sub_ps(
        _mm_loadu_ps(vector + offset),
        _mm_loadu_ps(query + offset));
    accumulator = _mm_add_ps(accumulator, _mm_mul_ps(d0, d0));
  }
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  return _mm_cvtss_f32(accumulator);
}

#ifndef LAION1M_BUILD_LEGACY_V6
inline __m128 retained_accumulate4(
    __m128 accumulator, const float* vector,
    std::size_t offset, __m128 query4) noexcept {
  const __m128 difference = _mm_sub_ps(
      _mm_loadu_ps(vector + offset), query4);
  return _mm_add_ps(
      accumulator, _mm_mul_ps(difference, difference));
}

inline float retained_reduce4(__m128 accumulator) noexcept {
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  return _mm_cvtss_f32(accumulator);
}

// An empty compiler boundary keeps each retained four-float group distinct.
// Without it GCC legally schedules independent work from the next group
// before finishing the current eight rows, inflating live temporaries enough
// to spill all accumulators.  The boundary emits no instruction, pins the
// eight independent chains in XMM registers, and prevents memory loads from
// crossing the required d12 -> d8 -> d4 -> d0 group order.
inline void retained_group_boundary(
    __m128& accumulator0, __m128& accumulator1,
    __m128& accumulator2, __m128& accumulator3,
    __m128& accumulator4, __m128& accumulator5,
    __m128& accumulator6, __m128& accumulator7) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile(
      ""
      : "+x"(accumulator0), "+x"(accumulator1),
        "+x"(accumulator2), "+x"(accumulator3),
        "+x"(accumulator4), "+x"(accumulator5),
        "+x"(accumulator6), "+x"(accumulator7)
      :
      : "memory");
#endif
}

inline void retained_prefetch_block8(
    const float* const future[kRetainedInterleaveRows],
    std::size_t offset) noexcept {
#if LAION1M_RETAINED_PREFETCH_POLICY == 1
  for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
    if (future[lane] != nullptr) {
      _mm_prefetch(
          reinterpret_cast<const char*>(future[lane] + offset),
          _MM_HINT_T1);
    }
  }
#elif LAION1M_RETAINED_PREFETCH_POLICY == 2
  for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
    if (future[lane] != nullptr) {
      _mm_prefetch(
          reinterpret_cast<const char*>(future[lane] + offset),
          _MM_HINT_NTA);
    }
  }
#else
  (void)future;
  (void)offset;
#endif
}

// Eight independent random rows are advanced one retained 16-float group at
// a time.  For every individual row the observable operation order remains
// exactly d12,d8,d4,d0 for offsets 0,16,...,496, followed by the same two
// horizontal additions as NumPy 2.0.2 c_einsum.  Intervening instructions
// from other rows expose memory-level parallelism without changing a row's
// float32 dependency chain.
inline void squared_l2_f32_512_retained_interleaved8(
    const float* __restrict__ query,
    const float* const vectors[kRetainedInterleaveRows],
    const float* const future[kRetainedInterleaveRows],
    float output[kRetainedInterleaveRows]) noexcept {
  __m128 accumulator0 = _mm_setzero_ps();
  __m128 accumulator1 = _mm_setzero_ps();
  __m128 accumulator2 = _mm_setzero_ps();
  __m128 accumulator3 = _mm_setzero_ps();
  __m128 accumulator4 = _mm_setzero_ps();
  __m128 accumulator5 = _mm_setzero_ps();
  __m128 accumulator6 = _mm_setzero_ps();
  __m128 accumulator7 = _mm_setzero_ps();
  for (std::size_t offset = 0; offset < kDimension; offset += 16) {
    retained_prefetch_block8(future, offset);

    const __m128 query12 = _mm_loadu_ps(query + offset + 12);
    accumulator0 = retained_accumulate4(
        accumulator0, vectors[0], offset + 12, query12);
    accumulator1 = retained_accumulate4(
        accumulator1, vectors[1], offset + 12, query12);
    accumulator2 = retained_accumulate4(
        accumulator2, vectors[2], offset + 12, query12);
    accumulator3 = retained_accumulate4(
        accumulator3, vectors[3], offset + 12, query12);
    accumulator4 = retained_accumulate4(
        accumulator4, vectors[4], offset + 12, query12);
    accumulator5 = retained_accumulate4(
        accumulator5, vectors[5], offset + 12, query12);
    accumulator6 = retained_accumulate4(
        accumulator6, vectors[6], offset + 12, query12);
    accumulator7 = retained_accumulate4(
        accumulator7, vectors[7], offset + 12, query12);
    retained_group_boundary(
        accumulator0, accumulator1, accumulator2, accumulator3,
        accumulator4, accumulator5, accumulator6, accumulator7);

    const __m128 query8 = _mm_loadu_ps(query + offset + 8);
    accumulator0 = retained_accumulate4(
        accumulator0, vectors[0], offset + 8, query8);
    accumulator1 = retained_accumulate4(
        accumulator1, vectors[1], offset + 8, query8);
    accumulator2 = retained_accumulate4(
        accumulator2, vectors[2], offset + 8, query8);
    accumulator3 = retained_accumulate4(
        accumulator3, vectors[3], offset + 8, query8);
    accumulator4 = retained_accumulate4(
        accumulator4, vectors[4], offset + 8, query8);
    accumulator5 = retained_accumulate4(
        accumulator5, vectors[5], offset + 8, query8);
    accumulator6 = retained_accumulate4(
        accumulator6, vectors[6], offset + 8, query8);
    accumulator7 = retained_accumulate4(
        accumulator7, vectors[7], offset + 8, query8);
    retained_group_boundary(
        accumulator0, accumulator1, accumulator2, accumulator3,
        accumulator4, accumulator5, accumulator6, accumulator7);

    const __m128 query4 = _mm_loadu_ps(query + offset + 4);
    accumulator0 = retained_accumulate4(
        accumulator0, vectors[0], offset + 4, query4);
    accumulator1 = retained_accumulate4(
        accumulator1, vectors[1], offset + 4, query4);
    accumulator2 = retained_accumulate4(
        accumulator2, vectors[2], offset + 4, query4);
    accumulator3 = retained_accumulate4(
        accumulator3, vectors[3], offset + 4, query4);
    accumulator4 = retained_accumulate4(
        accumulator4, vectors[4], offset + 4, query4);
    accumulator5 = retained_accumulate4(
        accumulator5, vectors[5], offset + 4, query4);
    accumulator6 = retained_accumulate4(
        accumulator6, vectors[6], offset + 4, query4);
    accumulator7 = retained_accumulate4(
        accumulator7, vectors[7], offset + 4, query4);
    retained_group_boundary(
        accumulator0, accumulator1, accumulator2, accumulator3,
        accumulator4, accumulator5, accumulator6, accumulator7);

    const __m128 query0 = _mm_loadu_ps(query + offset);
    accumulator0 = retained_accumulate4(
        accumulator0, vectors[0], offset, query0);
    accumulator1 = retained_accumulate4(
        accumulator1, vectors[1], offset, query0);
    accumulator2 = retained_accumulate4(
        accumulator2, vectors[2], offset, query0);
    accumulator3 = retained_accumulate4(
        accumulator3, vectors[3], offset, query0);
    accumulator4 = retained_accumulate4(
        accumulator4, vectors[4], offset, query0);
    accumulator5 = retained_accumulate4(
        accumulator5, vectors[5], offset, query0);
    accumulator6 = retained_accumulate4(
        accumulator6, vectors[6], offset, query0);
    accumulator7 = retained_accumulate4(
        accumulator7, vectors[7], offset, query0);
    retained_group_boundary(
        accumulator0, accumulator1, accumulator2, accumulator3,
        accumulator4, accumulator5, accumulator6, accumulator7);
  }
  output[0] = retained_reduce4(accumulator0);
  output[1] = retained_reduce4(accumulator1);
  output[2] = retained_reduce4(accumulator2);
  output[3] = retained_reduce4(accumulator3);
  output[4] = retained_reduce4(accumulator4);
  output[5] = retained_reduce4(accumulator5);
  output[6] = retained_reduce4(accumulator6);
  output[7] = retained_reduce4(accumulator7);
}
#endif

inline void prefetch_row(const float* row) noexcept {
  const char* bytes = reinterpret_cast<const char*>(row);
  // Match the v25 bounded-prefetch contract: prime the first two lines and
  // let the hardware sequential prefetcher follow the 2 KiB row.  Pulling
  // all 32 lines at lookahead 16 fills the 32 KiB L1 with future rows,
  // evicts the query/current row, and multiplies prefetch traffic by 16.
  _mm_prefetch(bytes, _MM_HINT_T0);
  _mm_prefetch(bytes + kCacheLineBytes, _MM_HINT_T0);
}

#ifndef LAION1M_BUILD_LEGACY_V6
template <typename Consumer>
inline int for_each_retained_score_interleaved8(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* support_ids, uint64_t support_count,
    Consumer&& consumer) {
  int32_t previous_id = -1;
  for (uint64_t index = 0; index < support_count; ++index) {
    const int32_t id = support_ids[index];
    if (id < 0 || static_cast<uint64_t>(id) >= base_rows) {
      return 4;
    }
    if (index != 0 && id <= previous_id) {
      return 5;
    }
    previous_id = id;
  }

  uint64_t index = 0;
  for (; index + kRetainedInterleaveRows <= support_count;
       index += kRetainedInterleaveRows) {
    const float* vectors[kRetainedInterleaveRows];
    const float* future[kRetainedInterleaveRows];
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      vectors[lane] =
          base + static_cast<std::size_t>(support_ids[index + lane])
                     * kDimension;
      const uint64_t future_index =
          index + lane + kPrefetchDistance;
      future[lane] =
          future_index < support_count
              ? base
                    + static_cast<std::size_t>(
                          support_ids[future_index])
                          * kDimension
              : nullptr;
      if constexpr (kRetainedPrefetchPolicy == 0) {
        if (future[lane] != nullptr) {
          prefetch_row(future[lane]);
        }
      }
    }
    float distances[kRetainedInterleaveRows];
    squared_l2_f32_512_retained_interleaved8(
        query, vectors, future, distances);
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      if (!std::isfinite(distances[lane])) {
        return 6;
      }
      consumer(index + lane, support_ids[index + lane], distances[lane]);
    }
  }
  for (; index < support_count; ++index) {
    if constexpr (kRetainedPrefetchPolicy == 0) {
      const uint64_t future_index = index + kPrefetchDistance;
      if (future_index < support_count) {
        prefetch_row(
            base + static_cast<std::size_t>(support_ids[future_index])
                       * kDimension);
      }
    }
    const float distance = squared_l2_f32_512_retained(
        query,
        base + static_cast<std::size_t>(support_ids[index]) * kDimension);
    if (!std::isfinite(distance)) {
      return 6;
    }
    consumer(index, support_ids[index], distance);
  }
  return 0;
}
#endif

}  // namespace

extern "C" {

uint32_t laion1m_exact_topk_avx2_v1_abi_version() {
#ifdef LAION1M_BUILD_LEGACY_V6
  return 6;
#else
  return 7;
#endif
}

uint32_t laion1m_exact_topk_avx2_v1_dimension() {
  return static_cast<uint32_t>(kDimension);
}

uint32_t laion1m_exact_topk_avx2_v1_k() {
  return static_cast<uint32_t>(kTopK);
}

uint32_t laion1m_exact_topk_avx2_v1_shortlist_capacity() {
  return static_cast<uint32_t>(kShortlist);
}

uint32_t laion1m_exact_topk_avx2_v1_prefetch_distance() {
  return static_cast<uint32_t>(kPrefetchDistance);
}

uint32_t laion1m_exact_topk_avx2_v1_prefetch_lines() {
  return static_cast<uint32_t>(kPrefetchLines);
}

#ifndef LAION1M_BUILD_LEGACY_V6
uint32_t laion1m_exact_topk_avx2_v1_retained_interleave_rows() {
  return static_cast<uint32_t>(kRetainedInterleaveRows);
}

uint32_t laion1m_exact_topk_avx2_v1_retained_prefetch_policy() {
  return kRetainedPrefetchPolicy;
}
#endif

int laion1m_exact_topk_avx2_v1_cpu_supported() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
  return 1;
#endif
}

// Status: 0=success, 1=null pointer, 2=invalid base/support size,
// 3=support below k, 4=ID out of range, 5=not strictly increasing,
// 6=non-finite distance.
int laion1m_exact_top64_i32_f32_avx2_v1(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* support_ids, uint64_t support_count,
    int32_t* output_ids, float* output_squared_l2) {
  if (base == nullptr || query == nullptr || support_ids == nullptr ||
      output_ids == nullptr || output_squared_l2 == nullptr) {
    return 1;
  }
  if (base_rows == 0 || base_rows > 0x7fffffffULL ||
      support_count > base_rows) {
    return 2;
  }
  if (support_count < kTopK) {
    return 3;
  }

  Candidate best[kShortlist];
  std::size_t best_size = 0;
  int32_t previous_id = -1;

  for (uint64_t index = 0; index < support_count; ++index) {
    const int32_t id = support_ids[index];
    if (id < 0 || static_cast<uint64_t>(id) >= base_rows) {
      return 4;
    }
    if (index != 0 && id <= previous_id) {
      return 5;
    }
    previous_id = id;

    if (index + kPrefetchDistance < support_count) {
      const int32_t future_id =
          support_ids[index + kPrefetchDistance];
      if (future_id >= 0 &&
          static_cast<uint64_t>(future_id) < base_rows) {
        prefetch_row(
            base + static_cast<std::size_t>(future_id) * kDimension);
      }
    }

    const float distance = squared_l2_f32_512_avx2(
        query, base + static_cast<std::size_t>(id) * kDimension);
    if (!std::isfinite(distance)) {
      return 6;
    }
    const Candidate scored{distance, id};
    if (best_size < kShortlist) {
      best[best_size] = scored;
      ++best_size;
      std::push_heap(best, best + best_size, candidate_less);
    } else if (candidate_less(scored, best[0])) {
      std::pop_heap(best, best + best_size, candidate_less);
      best[best_size - 1] = scored;
      std::push_heap(best, best + best_size, candidate_less);
    }
  }

  std::sort(best, best + best_size, candidate_less);
  for (std::size_t rank = 0; rank < best_size; ++rank) {
    output_ids[rank] = best[rank].id;
    output_squared_l2[rank] = best[rank].distance;
  }
  return 0;
}

// Exact top-11 under the retained NumPy float32 arithmetic above.  Rank 11
// lets Python detect a cutoff tie, where NumPy argpartition's selected member
// is implementation-specific and therefore requires a full retained fallback.
int laion1m_exact_top10_i32_f32_retained_v1(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* support_ids, uint64_t support_count,
    int32_t* output_ids, float* output_squared_l2) {
  if (base == nullptr || query == nullptr || support_ids == nullptr ||
      output_ids == nullptr || output_squared_l2 == nullptr) {
    return 1;
  }
  if (base_rows == 0 || base_rows > 0x7fffffffULL ||
      support_count > base_rows) {
    return 2;
  }
  if (support_count < kTopK) {
    return 3;
  }

  Candidate best[kRetainedCutoff];
  std::size_t best_size = 0;
#ifdef LAION1M_BUILD_LEGACY_V6
  int32_t previous_id = -1;
  for (uint64_t index = 0; index < support_count; ++index) {
    const int32_t id = support_ids[index];
    if (id < 0 || static_cast<uint64_t>(id) >= base_rows) {
      return 4;
    }
    if (index != 0 && id <= previous_id) {
      return 5;
    }
    previous_id = id;

    if (index + kPrefetchDistance < support_count) {
      const int32_t future_id =
          support_ids[index + kPrefetchDistance];
      if (future_id >= 0 &&
          static_cast<uint64_t>(future_id) < base_rows) {
        prefetch_row(
            base + static_cast<std::size_t>(future_id) * kDimension);
      }
    }

    const float distance = squared_l2_f32_512_retained(
        query, base + static_cast<std::size_t>(id) * kDimension);
    if (!std::isfinite(distance)) {
      return 6;
    }
    const Candidate scored{distance, id};
    if (best_size < kRetainedCutoff) {
      best[best_size] = scored;
      ++best_size;
      std::push_heap(best, best + best_size, candidate_less);
    } else if (candidate_less(scored, best[0])) {
      std::pop_heap(best, best + best_size, candidate_less);
      best[best_size - 1] = scored;
      std::push_heap(best, best + best_size, candidate_less);
    }
  }
#else
  const int status = for_each_retained_score_interleaved8(
      base, base_rows, query, support_ids, support_count,
      [&](uint64_t, int32_t id, float distance) {
        const Candidate scored{distance, id};
        if (best_size < kRetainedCutoff) {
          best[best_size] = scored;
          ++best_size;
          std::push_heap(best, best + best_size, candidate_less);
        } else if (candidate_less(scored, best[0])) {
          std::pop_heap(best, best + best_size, candidate_less);
          best[best_size - 1] = scored;
          std::push_heap(best, best + best_size, candidate_less);
        }
      });
  if (status != 0) {
    return status;
  }
#endif
  std::sort(best, best + best_size, candidate_less);
  for (std::size_t rank = 0; rank < best_size; ++rank) {
    output_ids[rank] = best[rank].id;
    output_squared_l2[rank] = best[rank].distance;
  }
  return 0;
}

#ifndef LAION1M_BUILD_LEGACY_V6
// Synthetic/full-payload audit ABI.  It exposes the exact score of every
// sorted support row so the interleaved implementation can be compared
// bit-for-bit with retained NumPy, including rows outside top-11.
int laion1m_retained_distances_interleaved8_v1(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* support_ids, uint64_t support_count,
    float* output_squared_l2) {
  if (base == nullptr || query == nullptr || support_ids == nullptr ||
      output_squared_l2 == nullptr) {
    return 1;
  }
  if (base_rows == 0 || base_rows > 0x7fffffffULL ||
      support_count == 0 || support_count > base_rows) {
    return 2;
  }
  return for_each_retained_score_interleaved8(
      base, base_rows, query, support_ids, support_count,
      [&](uint64_t index, int32_t, float distance) {
        output_squared_l2[index] = distance;
      });
}
#endif

// The caller supplies a fresh concatenation of exact sorted fragments.
// Sorting and uniquing it in place avoids NumPy's GIL-visible sort path.
int laion1m_sort_unique_i32_inplace_v1(
    int32_t* values, uint64_t count, uint64_t* unique_count) {
  if (values == nullptr || unique_count == nullptr) {
    return 1;
  }
  if (count == 0 || count > 0x7fffffffULL) {
    return 2;
  }
  std::sort(values, values + count);
  int32_t* const terminal = std::unique(values, values + count);
  *unique_count = static_cast<uint64_t>(terminal - values);
  return 0;
}

// Merge individually sorted exact fragments without materializing their
// union.  Duplicate IDs across DNF clauses are scored once.
int laion1m_exact_top11_segmented_i32_f32_retained_v1(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* concatenated_ids, uint64_t input_count,
    const uint64_t* offsets, uint64_t segment_count,
    int32_t* output_ids, float* output_squared_l2,
    uint64_t* output_unique_count) {
  if (base == nullptr || query == nullptr ||
      concatenated_ids == nullptr || offsets == nullptr ||
      output_ids == nullptr || output_squared_l2 == nullptr ||
      output_unique_count == nullptr) {
    return 1;
  }
  if (base_rows == 0 || base_rows > 0x7fffffffULL ||
      input_count == 0 || input_count > base_rows) {
    return 2;
  }
  if (segment_count == 0 || segment_count > kMaxSegments) {
    return 7;
  }
  if (offsets[0] != 0 || offsets[segment_count] != input_count) {
    return 8;
  }
  uint64_t positions[kMaxSegments];
  for (uint64_t segment = 0; segment < segment_count; ++segment) {
    const uint64_t begin = offsets[segment];
    const uint64_t end = offsets[segment + 1];
    if (begin >= end || end > input_count) {
      return 8;
    }
    positions[segment] = begin;
    int32_t previous = -1;
    for (uint64_t index = begin; index < end; ++index) {
      const int32_t id = concatenated_ids[index];
      if (id < 0 || static_cast<uint64_t>(id) >= base_rows) {
        return 4;
      }
      if (index != begin && id <= previous) {
        return 5;
      }
      previous = id;
    }
  }

  Candidate best[kRetainedCutoff];
  std::size_t best_size = 0;
  uint64_t unique_count = 0;
  while (true) {
    int32_t next_id = 0x7fffffff;
    bool found = false;
    for (uint64_t segment = 0; segment < segment_count; ++segment) {
      if (positions[segment] < offsets[segment + 1]) {
        const int32_t id = concatenated_ids[positions[segment]];
        if (!found || id < next_id) {
          next_id = id;
          found = true;
        }
      }
    }
    if (!found) {
      break;
    }
    for (uint64_t segment = 0; segment < segment_count; ++segment) {
      if (positions[segment] < offsets[segment + 1] &&
          concatenated_ids[positions[segment]] == next_id) {
        ++positions[segment];
      }
    }
    ++unique_count;
    const float distance = squared_l2_f32_512_retained(
        query,
        base + static_cast<std::size_t>(next_id) * kDimension);
    if (!std::isfinite(distance)) {
      return 6;
    }
    const Candidate scored{distance, next_id};
    if (best_size < kRetainedCutoff) {
      best[best_size] = scored;
      ++best_size;
      std::push_heap(best, best + best_size, candidate_less);
    } else if (candidate_less(scored, best[0])) {
      std::pop_heap(best, best + best_size, candidate_less);
      best[best_size - 1] = scored;
      std::push_heap(best, best + best_size, candidate_less);
    }
  }
  if (unique_count < kTopK) {
    return 3;
  }
  std::sort(best, best + best_size, candidate_less);
  for (std::size_t rank = 0; rank < best_size; ++rank) {
    output_ids[rank] = best[rank].id;
    output_squared_l2[rank] = best[rank].distance;
  }
  *output_unique_count = unique_count;
  return 0;
}

void laion1m_exact_topk_avx2_v1_reset_concurrency() {
  g_active_calls.store(0, std::memory_order_relaxed);
  g_maximum_active_calls.store(0, std::memory_order_relaxed);
}

int laion1m_exact_topk_avx2_v1_max_concurrency() {
  return g_maximum_active_calls.load(std::memory_order_relaxed);
}

// Synthetic-only GIL/concurrency probe.  Instrumentation is deliberately
// outside the production function so eight lanes never bounce one global
// atomic cache line in the timed request path.
int laion1m_exact_topk_avx2_v1_concurrency_probe(
    const float* base, uint64_t base_rows, const float* query,
    const int32_t* support_ids, uint64_t support_count,
    uint32_t repetitions, uint64_t* checksum) {
  if (repetitions == 0 || checksum == nullptr) {
    return 7;
  }
  ActiveCall active_call;
  int32_t output_ids[kShortlist];
  float output_distances[kShortlist];
  uint64_t result = 0;
  for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
    const int status = laion1m_exact_top64_i32_f32_avx2_v1(
        base, base_rows, query, support_ids, support_count,
        output_ids, output_distances);
    if (status != 0) {
      return status;
    }
    const std::size_t count =
        support_count < kShortlist
            ? static_cast<std::size_t>(support_count)
            : kShortlist;
    for (std::size_t rank = 0; rank < count; ++rank) {
      uint32_t bits = 0;
      std::memcpy(&bits, output_distances + rank, sizeof(bits));
      result = result * 0x9e3779b185ebca87ULL +
               static_cast<uint32_t>(output_ids[rank]) + bits;
    }
  }
  *checksum = result;
  return 0;
}

}  // extern "C"
