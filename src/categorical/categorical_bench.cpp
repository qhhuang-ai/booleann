// bci_bench: Boole-ANN Cell Index (BCI) full-system benchmark on YFCC10M.
// Loads all 1317 pre-built HAMCG Vamana shards + base + queries + metadata + GT.
// For each query in [qid_lo, qid_hi):
//   - Single equality (filter cardinality 1):
//       if shard exists for that tag -> HAMCG beam_search on shard
//       else (cold tag <0.1%)       -> brute force over posting list (fast)
//   - Conjunction (filter cardinality 2):
//       pick smaller tag T_small, beam_search on its shard, post-filter by other tag
//       if T_small has no shard -> try T_large; if neither -> mark FALLBACK_NEEDED
// Recall@10 vs ground-truth, QPS, latency histogram.

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <malloc.h>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/filters.h"
#include "utils/graph.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/types.h"
#include "pair_offset16_block16.h"
#include "pair_offset16_kernel.h"
#include "pair_offset16_prefix128.h"
#include "shortcut_residual_mask.h"

#include <immintrin.h>

namespace yfcc_support_complement =
    yfcc_pair_offset16_kernel_candidate_v2;

#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BOUNDED_EXACT
template <typename T, std::size_t Alignment>
struct CacheAlignedAllocator {
  using value_type = T;
  using is_always_equal = std::true_type;

  CacheAlignedAllocator() noexcept = default;
  template <typename U>
  CacheAlignedAllocator(
      const CacheAlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(
        ::operator new(count * sizeof(T), std::align_val_t(Alignment)));
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    ::operator delete(pointer, std::align_val_t(Alignment));
  }

  template <typename U>
  struct rebind {
    using other = CacheAlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const CacheAlignedAllocator<T, Alignment>&,
                const CacheAlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const CacheAlignedAllocator<T, Alignment>& left,
                const CacheAlignedAllocator<U, Alignment>& right) noexcept {
  return !(left == right);
}

using CommonPackVector =
    std::vector<uint8_t, CacheAlignedAllocator<uint8_t, 64>>;
#else
using CommonPackVector = std::vector<uint8_t>;
#endif

// SIMD-accelerated L2 squared distance for uint8 vectors (AVX2).
// Replaces the scalar fallback in euclidian_point.h for the brute hot path
// where this is the dominant work. For dim=192 (YFCC10M) this is 6× AVX2
// iterations + final tail; expected speedup vs scalar: 4-8×.
static inline float l2_sq_uint8_avx2(const uint8_t* __restrict__ a,
                                     const uint8_t* __restrict__ b,
                                     unsigned d) {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  unsigned i = 0;
  auto accum32 = [](__m256i& acc, const uint8_t* x, const uint8_t* y) {
    __m256i va = _mm256_loadu_si256((const __m256i*)x);
    __m256i vb = _mm256_loadu_si256((const __m256i*)y);
    __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
    __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
    __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
    __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
    __m256i d_lo = _mm256_sub_epi16(va_lo, vb_lo);
    __m256i d_hi = _mm256_sub_epi16(va_hi, vb_hi);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_lo, d_lo));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_hi, d_hi));
  };
  for (; i + 64 <= d; i += 64) {
    accum32(acc0, a + i,      b + i);
    accum32(acc1, a + i + 32, b + i + 32);
  }
  for (; i + 32 <= d; i += 32) {
    accum32(acc0, a + i, b + i);
  }
  __m256i acc = _mm256_add_epi32(acc0, acc1);
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc),
                                  _mm256_extracti128_si256(acc, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  int result = _mm_cvtsi128_si32(sum128);
  for (; i < d; ++i) {
    int diff = (int)a[i] - (int)b[i];
    result += diff * diff;
  }
  return (float)result;
}

// Score four independent uint8/192-D candidates in one interleaved AVX2
// kernel.  The query lanes are decoded once per 32 dimensions while four
// random candidate streams remain in flight.  Every accumulator is an exact
// 32-bit integer; the maximum 192-D squared distance is below 2^24, so its
// float representation and the reference ordering are exact.
static inline void l2_sq_uint8_four_192_avx2(
    const uint8_t* const candidates[4],
    const uint8_t* query,
    float distances[4]) {
  __m256i accumulators[4] = {
      _mm256_setzero_si256(), _mm256_setzero_si256(),
      _mm256_setzero_si256(), _mm256_setzero_si256()};
  for (unsigned offset = 0; offset < 192; offset += 32) {
    const __m256i query_bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(query + offset));
    const __m256i query_lo = _mm256_cvtepu8_epi16(
        _mm256_castsi256_si128(query_bytes));
    const __m256i query_hi = _mm256_cvtepu8_epi16(
        _mm256_extracti128_si256(query_bytes, 1));
    for (int lane = 0; lane < 4; ++lane) {
      const __m256i candidate_bytes = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(
              candidates[lane] + offset));
      const __m256i candidate_lo = _mm256_cvtepu8_epi16(
          _mm256_castsi256_si128(candidate_bytes));
      const __m256i candidate_hi = _mm256_cvtepu8_epi16(
          _mm256_extracti128_si256(candidate_bytes, 1));
      const __m256i delta_lo =
          _mm256_sub_epi16(candidate_lo, query_lo);
      const __m256i delta_hi =
          _mm256_sub_epi16(candidate_hi, query_hi);
      accumulators[lane] = _mm256_add_epi32(
          accumulators[lane],
          _mm256_madd_epi16(delta_lo, delta_lo));
      accumulators[lane] = _mm256_add_epi32(
          accumulators[lane],
          _mm256_madd_epi16(delta_hi, delta_hi));
    }
  }
  for (int lane = 0; lane < 4; ++lane) {
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(accumulators[lane]),
        _mm256_extracti128_si256(accumulators[lane], 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    distances[lane] =
        static_cast<float>(_mm_cvtsi128_si32(sum128));
  }
}

#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
static int exact_radix_distance_self_test() {
  std::array<std::array<uint8_t, 192>, 4> candidates{};
  std::array<uint8_t, 192> query{};
  const uint8_t* pointers[4]{};
  for (size_t dimension = 0; dimension < query.size(); ++dimension) {
    query[dimension] =
        static_cast<uint8_t>((31 * dimension + 7) & 255);
    for (size_t lane = 0; lane < candidates.size(); ++lane) {
      candidates[lane][dimension] = static_cast<uint8_t>(
          ((17 + 12 * lane) * dimension + 43 + lane) & 255);
    }
  }
  for (size_t lane = 0; lane < candidates.size(); ++lane) {
    pointers[lane] = candidates[lane].data();
  }
  float actual[4]{};
  l2_sq_uint8_four_192_avx2(
      pointers, query.data(), actual);
  for (size_t lane = 0; lane < candidates.size(); ++lane) {
    const float expected = l2_sq_uint8_avx2(
        candidates[lane].data(), query.data(), 192);
    if (actual[lane] != expected) {
      throw std::runtime_error(
          "four-way exact L2 self-test disagrees with reference AVX2");
    }
  }
  std::printf(
      "[exact radix self-test] PASS four_way_l2_exact=1 "
      "distance_integer_bits=24\n");
  return 0;
}
#endif

// Quality-only precursor for a packed 4-bit graph-vector layout.  Each byte is
// quantized to its high nibble before the same AVX2 squared-L2 reduction.  The
// diagnostic target still reads the original bytes, so its timing is not a
// packed-layout claim; it answers only whether b256 navigation plus exact
// reranking retains the required candidates before a 2x-smaller payload is
// built.
static inline float l2_sq_uint4_high_avx2(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    unsigned d) {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  const __m256i high_mask = _mm256_set1_epi8(
      static_cast<char>(0xf0));
  unsigned i = 0;
  auto accum32 = [&](__m256i& acc,
                      const uint8_t* x,
                      const uint8_t* y) {
    __m256i va =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x));
    __m256i vb =
        _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(y));
    va = _mm256_srli_epi16(
        _mm256_and_si256(va, high_mask), 4);
    vb = _mm256_srli_epi16(
        _mm256_and_si256(vb, high_mask), 4);
    const __m256i va_lo =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
    const __m256i va_hi =
        _mm256_cvtepu8_epi16(
            _mm256_extracti128_si256(va, 1));
    const __m256i vb_lo =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
    const __m256i vb_hi =
        _mm256_cvtepu8_epi16(
            _mm256_extracti128_si256(vb, 1));
    const __m256i d_lo = _mm256_sub_epi16(va_lo, vb_lo);
    const __m256i d_hi = _mm256_sub_epi16(va_hi, vb_hi);
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(d_lo, d_lo));
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(d_hi, d_hi));
  };
  for (; i + 64 <= d; i += 64) {
    accum32(acc0, a + i, b + i);
    accum32(acc1, a + i + 32, b + i + 32);
  }
  for (; i + 32 <= d; i += 32) {
    accum32(acc0, a + i, b + i);
  }
  __m256i acc = _mm256_add_epi32(acc0, acc1);
  __m128i sum128 = _mm_add_epi32(
      _mm256_castsi256_si128(acc),
      _mm256_extracti128_si256(acc, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  int result = _mm_cvtsi128_si32(sum128);
  for (; i < d; ++i) {
    const int diff =
        static_cast<int>(a[i] >> 4) -
        static_cast<int>(b[i] >> 4);
    result += diff * diff;
  }
  return static_cast<float>(result);
}

// Strict lower bound from the high-bit interval of each base component.
// With b retained high bits, x lies in [high(x), high(x)+2^(8-b)-1].
// The squared distance from the exact query byte to that interval is a
// component-wise lower bound on the full uint8 L2 distance.
static inline float l2_sq_uint8_high_bits_lower_bound_avx2(
    const uint8_t* __restrict__ base,
    const uint8_t* __restrict__ query,
    unsigned d,
    int high_bits) {
  if (high_bits <= 0 || high_bits >= 8) {
    throw std::runtime_error(
        "high-bit lower bound requires retained bits in [1,7]");
  }
  const int residual_bits = 8 - high_bits;
  const uint8_t residual_mask_value =
      static_cast<uint8_t>((1U << residual_bits) - 1U);
  const uint8_t high_mask_value =
      static_cast<uint8_t>(~residual_mask_value);
  const __m256i high_mask = _mm256_set1_epi8(
      static_cast<char>(high_mask_value));
  const __m256i residual_mask = _mm256_set1_epi8(
      static_cast<char>(residual_mask_value));
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  unsigned i = 0;
  auto accum32 = [&]( __m256i& acc,
                      const uint8_t* x,
                      const uint8_t* q) {
    const __m256i value = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(x));
    const __m256i query_value = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(q));
    const __m256i low =
        _mm256_and_si256(value, high_mask);
    const __m256i high =
        _mm256_add_epi8(low, residual_mask);
    const __m256i below =
        _mm256_subs_epu8(low, query_value);
    const __m256i above =
        _mm256_subs_epu8(query_value, high);
    const __m256i delta =
        _mm256_or_si256(below, above);
    const __m256i delta_lo = _mm256_cvtepu8_epi16(
        _mm256_castsi256_si128(delta));
    const __m256i delta_hi = _mm256_cvtepu8_epi16(
        _mm256_extracti128_si256(delta, 1));
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(delta_lo, delta_lo));
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(delta_hi, delta_hi));
  };
  for (; i + 64 <= d; i += 64) {
    accum32(acc0, base + i, query + i);
    accum32(acc1, base + i + 32, query + i + 32);
  }
  for (; i + 32 <= d; i += 32) {
    accum32(acc0, base + i, query + i);
  }
  __m256i acc = _mm256_add_epi32(acc0, acc1);
  __m128i sum128 = _mm_add_epi32(
      _mm256_castsi256_si128(acc),
      _mm256_extracti128_si256(acc, 1));
  sum128 = _mm_hadd_epi32(sum128, sum128);
  sum128 = _mm_hadd_epi32(sum128, sum128);
  int result = _mm_cvtsi128_si32(sum128);
  for (; i < d; ++i) {
    const int low =
        static_cast<int>(base[i] & high_mask_value);
    const int high =
        low + static_cast<int>(residual_mask_value);
    const int q = static_cast<int>(query[i]);
    const int delta =
        q < low ? low - q : (q > high ? q - high : 0);
    result += delta * delta;
  }
  return static_cast<float>(result);
}

#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
static int bitplane_lower_bound_self_test() {
  std::array<uint8_t, 192> base{};
  std::array<uint8_t, 192> query{};
  for (size_t i = 0; i < base.size(); ++i) {
    base[i] = static_cast<uint8_t>((73 * i + 29) & 255);
    query[i] = static_cast<uint8_t>((31 * i + 211) & 255);
  }
  const float exact = l2_sq_uint8_avx2(
      base.data(), query.data(),
      static_cast<unsigned>(base.size()));
  for (int high_bits = 1; high_bits <= 7; ++high_bits) {
    int scalar = 0;
    const int residual_bits = 8 - high_bits;
    const int residual_mask = (1 << residual_bits) - 1;
    const int high_mask = 255 ^ residual_mask;
    for (size_t i = 0; i < base.size(); ++i) {
      const int low = base[i] & high_mask;
      const int high = low + residual_mask;
      const int q = query[i];
      const int delta =
          q < low ? low - q : (q > high ? q - high : 0);
      scalar += delta * delta;
    }
    const float actual =
        l2_sq_uint8_high_bits_lower_bound_avx2(
            base.data(), query.data(),
            static_cast<unsigned>(base.size()), high_bits);
    if (actual != static_cast<float>(scalar) ||
        actual > exact) {
      throw std::runtime_error(
          "bitplane lower-bound AVX2/scalar/safety self-test failed");
    }
  }
  std::printf(
      "[bitplane lower-bound self-test] PASS bits=1..7 "
      "scalar_exact=1 lower_bound_safe=1\n");
  return 0;
}
#endif

#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
static int q4_graph_distance_self_test() {
  std::array<uint8_t, 192> left{};
  std::array<uint8_t, 192> right{};
  int expected = 0;
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = static_cast<uint8_t>((37 * i + 11) & 255);
    right[i] = static_cast<uint8_t>((19 * i + 203) & 255);
    const int delta =
        static_cast<int>(left[i] >> 4) -
        static_cast<int>(right[i] >> 4);
    expected += delta * delta;
  }
  const float actual = l2_sq_uint4_high_avx2(
      left.data(), right.data(),
      static_cast<unsigned>(left.size()));
  const float prefix = l2_sq_uint4_high_avx2(
      left.data(), right.data(), 64);
  const float suffix = l2_sq_uint4_high_avx2(
      left.data() + 64, right.data() + 64, 128);
  if (actual != static_cast<float>(expected) ||
      prefix + suffix != actual) {
    throw std::runtime_error(
        "q4 AVX2 self-test disagrees with scalar/segmented distance");
  }
  std::printf(
      "[q4 graph distance self-test] PASS distance=%.0f "
      "segmented_exact=1\n",
      actual);
  return 0;
}
#endif

// Exact nonnegative L2 with a safe frontier lower bound.  After each 64-byte
// stage, the accumulated squared distance is a lower bound on the complete
// 192-dimensional distance.  Once it reaches the current beam cutoff the
// caller will reject the candidate, so the remaining cache lines need not be
// fetched or evaluated.  Returning the partial value is sufficient because
// it is used only by the `distance >= cutoff` rejection.
static inline float l2_sq_uint8_avx2_cutoff(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    unsigned d, float cutoff, bool* early_abandoned) {
  int result = 0;
  unsigned i = 0;
  auto accum32 = [](const uint8_t* x, const uint8_t* y) {
    const __m256i va =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i vb =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y));
    const __m256i va_lo =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
    const __m256i va_hi =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
    const __m256i vb_lo =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
    const __m256i vb_hi =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
    const __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
    const __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
    return _mm256_add_epi32(
        _mm256_madd_epi16(diff_lo, diff_lo),
        _mm256_madd_epi16(diff_hi, diff_hi));
  };
  auto horizontal_sum = [](__m256i value) {
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
  };
  for (; i + 64 <= d; i += 64) {
    if (i + 64 < d) {
      __builtin_prefetch(b + i + 64, 0, 1);
    }
    const __m256i block = _mm256_add_epi32(
        accum32(a + i, b + i),
        accum32(a + i + 32, b + i + 32));
    result += horizontal_sum(block);
    if (static_cast<float>(result) >= cutoff) {
      if (early_abandoned != nullptr) *early_abandoned = true;
      return static_cast<float>(result);
    }
  }
  for (; i + 32 <= d; i += 32) {
    result += horizontal_sum(accum32(a + i, b + i));
    if (static_cast<float>(result) >= cutoff) {
      if (early_abandoned != nullptr) *early_abandoned = true;
      return static_cast<float>(result);
    }
  }
  for (; i < d; ++i) {
    const int diff = static_cast<int>(a[i]) -
                     static_cast<int>(b[i]);
    result += diff * diff;
  }
  if (early_abandoned != nullptr) *early_abandoned = false;
  return static_cast<float>(result);
}

using PointT = Euclidian_Point<uint8_t>;
using PR     = PointRange<uint8_t, PointT>;
using Indx   = int32_t;
using GraphI = Graph<Indx>;

static std::pair<int32_t, int32_t> canonical_pair_from_support(
    int32_t a, int32_t b, int64_t support_a, int64_t support_b) {
  const bool a_primary =
      support_a < support_b || (support_a == support_b && a < b);
  return a_primary ? std::make_pair(a, b) : std::make_pair(b, a);
}

static int select_single_beam(
    int32_t tag, int base_beam, int high_beam,
    const std::unordered_set<int32_t>& high_beam_tags) {
  return high_beam_tags.count(tag) ? high_beam : base_beam;
}

static bool select_single_high_effort(
    int32_t tag, int64_t support, int64_t min_support,
    const std::unordered_set<int32_t>& high_beam_tags) {
  return high_beam_tags.count(tag) ||
         (min_support > 0 && support >= min_support);
}

struct SingleSearchTier {
  int beam;
  double cut;
  bool high_effort;
  bool risk_effort;
};

static SingleSearchTier select_single_search_tier(
    int32_t tag, int64_t support,
    int base_beam, double base_cut,
    int high_beam, double high_cut, int64_t high_min_support,
    const std::unordered_set<int32_t>& high_beam_tags,
    int risk_beam, double risk_cut,
    const std::unordered_set<int32_t>& risk_beam_tags) {
  const bool high_effort = select_single_high_effort(
      tag, support, high_min_support, high_beam_tags);
  const bool risk_effort = risk_beam_tags.count(tag) != 0;
  return {
      risk_effort ? risk_beam : (high_effort ? high_beam : base_beam),
      risk_effort ? risk_cut : (high_effort ? high_cut : base_cut),
      high_effort,
      risk_effort};
}

static bool select_single_multi_start(
    int64_t support, int n_starts, int64_t min_support) {
  return n_starts > 1 && min_support > 0 && support >= min_support;
}

#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
struct ResidualLandmarkDirectory {
  std::array<std::vector<int32_t>, 2> landmarks_by_development_half;
};

static int residual_landmark_development_half(int qid) {
  if (qid >= 30000 && qid < 35000) return 0;
  if (qid >= 35000 && qid < 40000) return 1;
  return -1;
}

static const std::vector<int32_t>& residual_landmark_crossfit_entries(
    const ResidualLandmarkDirectory& directory, int qid) {
  const int held_out_half = residual_landmark_development_half(qid);
  if (held_out_half < 0) {
    throw std::runtime_error(
        "residual-landmark cross-fit query is outside q30--40");
  }
  return directory.landmarks_by_development_half[1 - held_out_half];
}

static ResidualLandmarkDirectory load_residual_landmark_directory(
    const std::filesystem::path& coverage_path) {
  std::ifstream input(coverage_path);
  if (!input) {
    throw std::runtime_error(
        "cannot open residual-landmark coverage CSV: " +
        coverage_path.string());
  }
  std::string line;
  if (!std::getline(input, line) ||
      line !=
          "source_local,target_local,qid,residual_local,kind,"
          "development_half") {
    throw std::runtime_error(
        "unexpected residual-landmark coverage header");
  }
  ResidualLandmarkDirectory directory;
  size_t rows = 0;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::array<std::string, 6> fields;
    size_t begin = 0;
    for (size_t column = 0; column < fields.size(); ++column) {
      const size_t comma = line.find(',', begin);
      if (column + 1 == fields.size()) {
        if (comma != std::string::npos) {
          throw std::runtime_error(
              "too many residual-landmark coverage columns");
        }
        fields[column] = line.substr(begin);
      } else {
        if (comma == std::string::npos) {
          throw std::runtime_error(
              "too few residual-landmark coverage columns");
        }
        fields[column] = line.substr(begin, comma - begin);
        begin = comma + 1;
      }
    }
    const int32_t residual_local = std::stoi(fields[3]);
    const int development_half = std::stoi(fields[5]);
    if (residual_local < 0 ||
        (development_half != 0 && development_half != 1)) {
      throw std::runtime_error(
          "invalid residual landmark or development half");
    }
    directory.landmarks_by_development_half[development_half].push_back(
        residual_local);
    ++rows;
  }
  if (rows == 0) {
    throw std::runtime_error("residual-landmark coverage CSV is empty");
  }
  for (auto& entries : directory.landmarks_by_development_half) {
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    if (entries.empty()) {
      throw std::runtime_error(
          "residual-landmark coverage has an empty development half");
    }
  }
  return directory;
}

static int residual_landmark_self_test() {
  ResidualLandmarkDirectory directory;
  directory.landmarks_by_development_half[0] = {3, 7};
  directory.landmarks_by_development_half[1] = {11, 13, 17};
  if (residual_landmark_development_half(29999) != -1 ||
      residual_landmark_development_half(30000) != 0 ||
      residual_landmark_development_half(34999) != 0 ||
      residual_landmark_development_half(35000) != 1 ||
      residual_landmark_development_half(39999) != 1 ||
      residual_landmark_development_half(40000) != -1 ||
      residual_landmark_crossfit_entries(directory, 32346) !=
          std::vector<int32_t>({11, 13, 17}) ||
      residual_landmark_crossfit_entries(directory, 38078) !=
          std::vector<int32_t>({3, 7})) {
    std::fprintf(stderr, "residual-landmark cross-fit self-test failed\n");
    return 1;
  }
  std::printf(
      "[residual landmark self-test] PASS crossfit=1 "
      "forbidden_formal_queries=1\n");
  return 0;
}
#endif

static bool should_retry_single_search(
    size_t frontier_size, size_t effective_beam, double boundary_ratio,
    double max_boundary_ratio) {
  return max_boundary_ratio > 0.0 &&
         frontier_size == effective_beam &&
         boundary_ratio > 0.0 &&
         boundary_ratio <= max_boundary_ratio;
}

static bool single_retry_tag_authorized(
    int32_t tag, bool has_frozen_vocabulary,
    const std::unordered_set<int32_t>& frozen_tags) {
  return !has_frozen_vocabulary || frozen_tags.count(tag) != 0;
}

static bool batched_l2_retry_modes_supported(
    bool graph_retry_enabled, bool alternate_retry_enabled) {
  // The production batched-L2 branch implements a larger-beam retry from the
  // same entry.  Alternate-entry retry still depends on cluster representatives
  // and remains isolated to the reference branch.
  (void)graph_retry_enabled;
  return !alternate_retry_enabled;
}

static int single_adaptive_beam_self_test() {
  const std::unordered_set<int32_t> hard_tags = {89, 173};
  const std::unordered_set<int32_t> risk_tags = {8};
  const auto risk_tier = select_single_search_tier(
      8, 2'000'000, 192, 1.35, 224, 1.30, 800'000,
      hard_tags, 384, 1.40, risk_tags);
  const auto high_tier = select_single_search_tier(
      89, 10, 192, 1.35, 224, 1.30, 800'000,
      hard_tags, 384, 1.40, risk_tags);
  const auto base_tier = select_single_search_tier(
      7, 10, 192, 1.35, 224, 1.30, 800'000,
      hard_tags, 384, 1.40, risk_tags);
  const uint8_t self_test_values[4] = {1, 2, 3, 4};
  PointT ordinary_query(self_test_values, 4, 4, 123);
  PointT external_query(self_test_values, 4, 4, -1);
  if (select_single_beam(89, 128, 256, hard_tags) != 256 ||
      select_single_beam(173, 128, 256, hard_tags) != 256 ||
      select_single_beam(8, 128, 256, hard_tags) != 128 ||
      select_single_beam(-1, 128, 256, hard_tags) != 128 ||
      !select_single_high_effort(89, 10, 1'000'000, hard_tags) ||
      !select_single_high_effort(8, 1'000'000, 1'000'000, {}) ||
      select_single_high_effort(8, 999'999, 1'000'000, {}) ||
      risk_tier.beam != 384 || risk_tier.cut != 1.40 ||
      !risk_tier.high_effort || !risk_tier.risk_effort ||
      high_tier.beam != 224 || high_tier.cut != 1.30 ||
      !high_tier.high_effort || high_tier.risk_effort ||
      base_tier.beam != 192 || base_tier.cut != 1.35 ||
      base_tier.high_effort || base_tier.risk_effort ||
      !select_single_multi_start(1'200'000, 2, 1'200'000) ||
      select_single_multi_start(1'199'999, 2, 1'200'000) ||
      select_single_multi_start(1'200'000, 1, 1'200'000) ||
      !should_retry_single_search(192, 192, 1.009, 1.010) ||
      should_retry_single_search(191, 192, 1.009, 1.010) ||
      should_retry_single_search(192, 192, 1.011, 1.010) ||
      !single_retry_tag_authorized(89, true, hard_tags) ||
      single_retry_tag_authorized(8, true, hard_tags) ||
      !single_retry_tag_authorized(8, false, hard_tags) ||
      !batched_l2_retry_modes_supported(true, false) ||
      batched_l2_retry_modes_supported(false, true) ||
      external_query.id() != -1 ||
      external_query.distance(ordinary_query) != 0.0f) {
    std::fprintf(stderr, "single adaptive beam self-test failed\n");
    return 1;
  }
  const std::array<int32_t, 4> primary = {1, 3, 5, 7};
  const std::array<int32_t, 4> secondary = {0, 3, 7, 9};
  std::vector<uint16_t> offsets;
  offsets.reserve(primary.size());
  yfcc_support_complement::
      intersect_postings_to_local_offset16_scratch_checked(
          primary.data(), primary.size(), secondary.data(), secondary.size(),
          nullptr,
          yfcc_support_complement::IntersectionStrategy::SortedMerge,
          offsets);
  if (offsets != std::vector<uint16_t>({1, 3}) ||
      yfcc_support_complement::choose_scratch_representation(65'536) !=
          yfcc_support_complement::ScratchRepresentation::LocalOffset16 ||
      yfcc_support_complement::choose_scratch_representation(65'537) !=
          yfcc_support_complement::ScratchRepresentation::GlobalId32 ||
      yfcc_support_complement::
              choose_base_only_support_complement_strategy(
                  100, 65'536, false) !=
          yfcc_support_complement::IntersectionStrategy::SortedMerge ||
      yfcc_support_complement::
              choose_base_only_support_complement_strategy(
                  100, 65'537, true) !=
          yfcc_support_complement::IntersectionStrategy::SecondaryBitmap) {
    std::fprintf(stderr, "support-complement self-test failed\n");
    return 1;
  }
  std::printf("single adaptive beam self-test passed\n");
  std::printf("support-complement self-test passed\n");
  return 0;
}

static std::string env_path_or_default(const char* name, const std::string& fallback,
                                       bool trailing_slash) {
  const char* raw = std::getenv(name);
  std::string out = (raw && raw[0] != '\0') ? std::string(raw) : fallback;
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  if (trailing_slash) out.push_back('/');
  return out;
}

static void fsync_file_path(const std::filesystem::path& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("cannot open for fsync: " + path.string());
  }
  if (::fsync(fd) != 0) {
    int saved_errno = errno;
    ::close(fd);
    throw std::runtime_error(
        "fsync failed for " + path.string() + ": " + std::strerror(saved_errno));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("close failed after fsync: " + path.string());
  }
}

static void fsync_directory(const std::filesystem::path& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("cannot open directory for fsync: " + path.string());
  }
  if (::fsync(fd) != 0) {
    int saved_errno = errno;
    ::close(fd);
    throw std::runtime_error(
        "directory fsync failed for " + path.string() + ": " +
        std::strerror(saved_errno));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("directory close failed after fsync: " + path.string());
  }
}

static void durable_publish_no_replace(const std::filesystem::path& partial,
                                       const std::filesystem::path& canonical) {
  fsync_file_path(partial);
  if (::link(partial.c_str(), canonical.c_str()) != 0) {
    int saved_errno = errno;
    throw std::runtime_error(
        "atomic no-replace publish failed for " + canonical.string() + ": " +
        std::strerror(saved_errno));
  }
  fsync_directory(canonical.parent_path());
  if (::unlink(partial.c_str()) != 0) {
    throw std::runtime_error("cannot unlink published partial: " +
                             partial.string());
  }
  fsync_directory(canonical.parent_path());
}

static uint64_t fnv1a64_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open acknowledgement: " + path.string());
  }
  uint64_t value = 14695981039346656037ULL;
  char buffer[1 << 15];
  while (input) {
    input.read(buffer, sizeof(buffer));
    std::streamsize count = input.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      value ^= static_cast<unsigned char>(buffer[i]);
      value *= 1099511628211ULL;
    }
  }
  return value;
}

static std::unordered_map<std::string, std::string> read_key_value_file(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open release marker: " + path.string());
  }
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    size_t split = line.find('=');
    if (split == std::string::npos || split == 0) {
      throw std::runtime_error("malformed release marker: " + path.string());
    }
    if (!values.emplace(line.substr(0, split), line.substr(split + 1)).second) {
      throw std::runtime_error("duplicate release-marker key: " + path.string());
    }
  }
  return values;
}

// Thin SubsetPointRange w/o unordered_map (which dominated cost in v1).
// beam_search only needs operator[], size(), dimension(), aligned_dimension();
// it never calls real_index/subset_index, so the map is dead weight.
struct ThinSubPR {
  PR& pr;
  const parlay::sequence<int32_t>& subset;
  const uint8_t* packed_values;
  size_t n;
  unsigned int dims, aligned_dims;
  ThinSubPR(PR& pr_, const parlay::sequence<int32_t>& s,
            const uint8_t* packed_values_ = nullptr)
    : pr(pr_), subset(s), packed_values(packed_values_), n(s.size()),
      dims((unsigned)pr_.dimension()),
      aligned_dims((unsigned)pr_.aligned_dimension()) {}
  size_t size() const { return n; }
  PointT operator[](long i) {
    if (packed_values != nullptr) {
      return PointT(
          packed_values + static_cast<size_t>(i) * aligned_dims,
          dims, aligned_dims, subset[i]);
    }
    return pr[subset[i]];
  }
  long dimension() const { return dims; }
  long aligned_dimension() const { return aligned_dims; }
};

// Allocation-stable form of ParlayANN's beam_search for the batched single-tag
// route.  It preserves the original frontier/visited ordering and collision
// behavior, but reuses all temporary vectors on the worker that owns a shard
// bucket.  Epoch-stamped hash slots are equivalent to a fresh all--1 table and
// avoid clearing O(beam^2) entries for every query.
template <typename indexType, typename distanceType>
struct ReusableBeamWorkspace {
  std::vector<uint32_t> hash_slots;
  uint32_t hash_epoch = 0;
  std::vector<std::pair<indexType, distanceType>> frontier;
  std::vector<std::pair<indexType, distanceType>> unvisited_frontier;
  std::vector<std::pair<indexType, distanceType>> visited;
  std::vector<std::pair<indexType, distanceType>> new_frontier;
  std::vector<std::pair<indexType, distanceType>> candidates;
  std::vector<indexType> keep;
  size_t staged_distance_candidates = 0;
  size_t early_abandoned_candidates = 0;
};

template <typename indexType, typename distanceType>
struct ReusableBeamSearchView {
  const std::vector<std::pair<indexType, distanceType>>& frontier;
  const std::vector<std::pair<indexType, distanceType>>& visited;
  size_t distance_computations;
};

// Project-local, allocation-stable beam state for the single-tag route.
//
// ParlayANN's reference beam_search keeps both frontier and visited sorted.
// Every graph expansion therefore performs a linear set_union over the whole
// beam, inserts into the growing visited vector, and then performs a second
// linear set_difference even though only the first unvisited item is consumed.
// This state machine implements the same union -> beam trim -> metric-cut trim
// transition with:
//   * the closest k+1 active entries in a tiny sorted array (the cut pivot);
//   * the remaining active entries in a max heap (beam/cut eviction);
//   * active, unvisited entries in a lazy min heap (next expansion); and
//   * a compact exact status table for stale-entry and visited suppression.
//
// The approximate direct-mapped "seen" filter remains in the graph-search
// wrapper below, including its collision behavior.  The exact status table is
// only a different representation of reference frontier/visited state.
template <typename indexType, typename distanceType>
struct DualHeapBeamEntry {
  indexType id;
  distanceType distance;
  uint32_t generation;
};

template <typename indexType, typename distanceType>
static inline bool dual_heap_pair_less(
    const DualHeapBeamEntry<indexType, distanceType>& left,
    const DualHeapBeamEntry<indexType, distanceType>& right) {
  return left.distance < right.distance ||
         (left.distance == right.distance && left.id < right.id);
}

template <typename indexType, typename distanceType>
struct DualHeapBeamMaxCompare {
  bool operator()(
      const DualHeapBeamEntry<indexType, distanceType>& left,
      const DualHeapBeamEntry<indexType, distanceType>& right) const {
    return dual_heap_pair_less(left, right);
  }
};

template <typename indexType, typename distanceType>
struct DualHeapBeamMinCompare {
  bool operator()(
      const DualHeapBeamEntry<indexType, distanceType>& left,
      const DualHeapBeamEntry<indexType, distanceType>& right) const {
    return dual_heap_pair_less(right, left);
  }
};

template <typename indexType>
class DualHeapBeamStatusTable {
 public:
  static constexpr uint8_t kActive = 1;
  static constexpr uint8_t kVisited = 2;

  struct Snapshot {
    uint8_t flags = 0;
    uint32_t generation = 0;
  };

  void reset(size_t beam_size) {
    size_t requested = 1024;
    while (requested < beam_size * 8) requested <<= 1;
    if (slots_.size() != requested) slots_.resize(requested);
    clear_slots();
    next_generation_ = 0;
  }

  Snapshot lookup(indexType id) const {
    const size_t slot = find_existing(id);
    if (slot == npos) return {};
    return {slots_[slot].flags, slots_[slot].generation};
  }

  std::pair<bool, Snapshot> activate(indexType id) {
    maybe_compact();
    size_t slot = find_existing(id);
    if (slot != npos && (slots_[slot].flags & kActive)) {
      return {false, {slots_[slot].flags, slots_[slot].generation}};
    }
    if (slot == npos) slot = find_insert_slot(id);
    Slot& item = slots_[slot];
    if (item.key != id) {
      if (item.key == tombstone_key()) --tombstones_;
      item.key = id;
      item.flags = 0;
      item.generation = 0;
      ++live_keys_;
    }
    ++next_generation_;
    if (next_generation_ == 0) {
      throw std::runtime_error("dual-heap generation counter overflow");
    }
    item.generation = next_generation_;
    item.flags |= kActive;
    return {true, {item.flags, item.generation}};
  }

  void mark_visited(indexType id, uint32_t generation) {
    const size_t slot = find_existing(id);
    if (slot == npos || slots_[slot].generation != generation ||
        !(slots_[slot].flags & kActive)) {
      throw std::runtime_error(
          "dual-heap attempted to visit an inactive frontier entry");
    }
    slots_[slot].flags |= kVisited;
  }

  bool is_active(indexType id, uint32_t generation) const {
    const size_t slot = find_existing(id);
    return slot != npos && slots_[slot].generation == generation &&
           (slots_[slot].flags & kActive);
  }

  bool is_active_unvisited(indexType id, uint32_t generation) const {
    const size_t slot = find_existing(id);
    return slot != npos && slots_[slot].generation == generation &&
           (slots_[slot].flags & kActive) &&
           !(slots_[slot].flags & kVisited);
  }

  bool is_visited(indexType id) const {
    const size_t slot = find_existing(id);
    return slot != npos && (slots_[slot].flags & kVisited);
  }

  void deactivate(indexType id, uint32_t generation) {
    const size_t slot = find_existing(id);
    if (slot == npos || slots_[slot].generation != generation ||
        !(slots_[slot].flags & kActive)) {
      return;
    }
    Slot& item = slots_[slot];
    item.flags &= static_cast<uint8_t>(~kActive);
    if (!(item.flags & kVisited)) {
      item.key = tombstone_key();
      item.flags = 0;
      item.generation = 0;
      --live_keys_;
      ++tombstones_;
    }
  }

 private:
  struct Slot {
    indexType key = empty_key();
    uint32_t generation = 0;
    uint8_t flags = 0;
  };

  static constexpr size_t npos = std::numeric_limits<size_t>::max();
  static constexpr indexType empty_key() {
    return static_cast<indexType>(-1);
  }
  static constexpr indexType tombstone_key() {
    return static_cast<indexType>(-2);
  }

  size_t mask() const { return slots_.size() - 1; }

  size_t initial_slot(indexType id) const {
    return static_cast<size_t>(parlay::hash64_2(id)) & mask();
  }

  size_t find_existing(indexType id) const {
    if (slots_.empty()) return npos;
    size_t slot = initial_slot(id);
    for (size_t probe = 0; probe < slots_.size(); ++probe) {
      const indexType key = slots_[slot].key;
      if (key == empty_key()) return npos;
      if (key == id) return slot;
      slot = (slot + 1) & mask();
    }
    return npos;
  }

  size_t find_insert_slot(indexType id) {
    size_t slot = initial_slot(id);
    size_t first_tombstone = npos;
    for (size_t probe = 0; probe < slots_.size(); ++probe) {
      const indexType key = slots_[slot].key;
      if (key == id) return slot;
      if (key == tombstone_key() && first_tombstone == npos) {
        first_tombstone = slot;
      } else if (key == empty_key()) {
        return first_tombstone == npos ? slot : first_tombstone;
      }
      slot = (slot + 1) & mask();
    }
    if (first_tombstone != npos) return first_tombstone;
    throw std::runtime_error("dual-heap status table is full");
  }

  void clear_slots() {
    std::fill(slots_.begin(), slots_.end(), Slot{});
    live_keys_ = 0;
    tombstones_ = 0;
  }

  void maybe_compact() {
    if (slots_.empty()) reset(1);
    if ((live_keys_ + tombstones_ + 1) * 2 >= slots_.size()) {
      // Inactive visited vertices remain exact members of the reference
      // visited set even after leaving the frontier.  Their count is bounded
      // by QP.limit rather than beamSize, so grow on demand.
      const size_t requested =
          (live_keys_ + 1) * 2 >= slots_.size()
              ? slots_.size() * 2
              : slots_.size();
      rehash(requested);
      return;
    }
    if (tombstones_ * 3 < slots_.size()) return;
    rehash(slots_.size());
  }

  void rehash(size_t requested) {
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(requested, Slot{});
    live_keys_ = 0;
    tombstones_ = 0;
    for (const Slot& item : old) {
      if (item.key < 0 || item.flags == 0) continue;
      const size_t slot = find_insert_slot(item.key);
      slots_[slot] = item;
      ++live_keys_;
    }
  }

  std::vector<Slot> slots_;
  size_t live_keys_ = 0;
  size_t tombstones_ = 0;
  uint32_t next_generation_ = 0;
};

template <typename indexType, typename distanceType>
class DualHeapBeamState {
 public:
  using Entry = DualHeapBeamEntry<indexType, distanceType>;
  using Pair = std::pair<indexType, distanceType>;

  void reset(size_t beam_size, size_t k, double cut) {
    if (beam_size == 0 || k == 0 || beam_size < k ||
        !(cut > 0.0) || !std::isfinite(cut)) {
      throw std::runtime_error("invalid dual-heap beam parameters");
    }
    beam_size_ = beam_size;
    k_ = k;
    cut_ = cut;
    elite_limit_ = std::min(beam_size_, k_ + 1);
    active_count_ = 0;
    elite_.clear();
    visited_.clear();
    tail_ = decltype(tail_)();
    pending_ = decltype(pending_)();
    status_.reset(beam_size_);
    elite_.reserve(elite_limit_);
    visited_.reserve(2 * beam_size_);
  }

  void add_start(indexType id, distanceType distance) {
    const auto existing = status_.lookup(id);
    if (existing.flags != 0) {
      throw std::runtime_error(
          "dual-heap requires distinct starting points");
    }
    insert_active(id, distance);
  }

  bool has_unvisited() {
    clean_pending();
    return !pending_.empty();
  }

  Entry next_unvisited() {
    clean_pending();
    if (pending_.empty()) {
      throw std::runtime_error("dual-heap frontier has no unvisited entry");
    }
    return pending_.top();
  }

  void visit(const Entry& entry) {
    if (!status_.is_active_unvisited(entry.id, entry.generation)) {
      throw std::runtime_error(
          "dual-heap next entry became inactive before visit");
    }
    status_.mark_visited(entry.id, entry.generation);
    visited_.push_back(entry);
    pending_.pop();
  }

  size_t size() const { return active_count_; }

  distanceType premerge_distance_cutoff() {
    if (active_count_ < beam_size_) {
      return static_cast<distanceType>(
          std::numeric_limits<int>::max());
    }
    clean_tail();
    if (!tail_.empty()) return tail_.top().distance;
    if (elite_.empty()) {
      throw std::runtime_error("dual-heap full frontier is empty");
    }
    return elite_.back().distance;
  }

  void merge_candidates(
      const std::vector<Pair>& sorted_candidates, bool is_metric) {
    for (const Pair& candidate : sorted_candidates) {
      insert_active(candidate.first, candidate.second);
      while (active_count_ > beam_size_) remove_farthest();
    }
    if (is_metric && active_count_ > k_) apply_metric_cut();
  }

  std::vector<Pair> sorted_frontier() const {
    std::vector<Pair> output;
    output.reserve(active_count_);
    for (const Entry& entry : elite_) {
      if (status_.is_active(entry.id, entry.generation)) {
        output.push_back({entry.id, entry.distance});
      }
    }
    auto tail_copy = tail_;
    while (!tail_copy.empty()) {
      const Entry entry = tail_copy.top();
      tail_copy.pop();
      if (status_.is_active(entry.id, entry.generation)) {
        output.push_back({entry.id, entry.distance});
      }
    }
    std::sort(output.begin(), output.end(), pair_less);
    if (output.size() != active_count_) {
      throw std::runtime_error(
          "dual-heap active frontier cardinality mismatch");
    }
    return output;
  }

  std::vector<Pair> sorted_visited() const {
    std::vector<Pair> output;
    output.reserve(visited_.size());
    for (const Entry& entry : visited_) {
      output.push_back({entry.id, entry.distance});
    }
    std::sort(output.begin(), output.end(), pair_less);
    return output;
  }

 private:
  static bool pair_less(const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second && left.first < right.first);
  }

  static bool entry_less(const Entry& left, const Entry& right) {
    return dual_heap_pair_less(left, right);
  }

  void clean_pending() {
    while (!pending_.empty() &&
           !status_.is_active_unvisited(
               pending_.top().id, pending_.top().generation)) {
      pending_.pop();
    }
  }

  void clean_tail() {
    while (!tail_.empty() &&
           !status_.is_active(
               tail_.top().id, tail_.top().generation)) {
      tail_.pop();
    }
  }

  void insert_elite(const Entry& entry) {
    elite_.insert(
        std::lower_bound(
            elite_.begin(), elite_.end(), entry, entry_less),
        entry);
  }

  void insert_active(indexType id, distanceType distance) {
    const auto activation = status_.activate(id);
    if (!activation.first) return;
    const Entry entry{id, distance, activation.second.generation};
    ++active_count_;
    if (!(activation.second.flags &
          DualHeapBeamStatusTable<indexType>::kVisited)) {
      pending_.push(entry);
    }
    if (elite_.size() < elite_limit_) {
      insert_elite(entry);
    } else if (entry_less(entry, elite_.back())) {
      tail_.push(elite_.back());
      elite_.pop_back();
      insert_elite(entry);
    } else {
      tail_.push(entry);
    }
  }

  void deactivate(const Entry& entry) {
    if (!status_.is_active(entry.id, entry.generation)) return;
    status_.deactivate(entry.id, entry.generation);
    --active_count_;
  }

  void remove_farthest() {
    clean_tail();
    if (!tail_.empty()) {
      const Entry entry = tail_.top();
      tail_.pop();
      deactivate(entry);
      return;
    }
    if (elite_.empty()) {
      throw std::runtime_error(
          "dual-heap cannot evict from an empty frontier");
    }
    const Entry entry = elite_.back();
    elite_.pop_back();
    deactivate(entry);
  }

  void rebuild_after_exceptional_cut(
      const Pair& threshold) {
    std::vector<Pair> frontier = sorted_frontier();
    for (const Pair& item : frontier) {
      if (!pair_less(threshold, item)) continue;
      const auto snapshot = status_.lookup(item.first);
      if (snapshot.flags &
          DualHeapBeamStatusTable<indexType>::kActive) {
        status_.deactivate(item.first, snapshot.generation);
        --active_count_;
      }
    }
    elite_.clear();
    tail_ = decltype(tail_)();
    for (const Pair& item : frontier) {
      if (pair_less(threshold, item)) continue;
      const auto snapshot = status_.lookup(item.first);
      if (!(snapshot.flags &
            DualHeapBeamStatusTable<indexType>::kActive)) {
        throw std::runtime_error(
            "dual-heap exceptional cut lost an active entry");
      }
      const Entry entry{
          item.first, item.second, snapshot.generation};
      if (elite_.size() < elite_limit_) {
        elite_.push_back(entry);
      } else {
        tail_.push(entry);
      }
    }
  }

  void apply_metric_cut() {
    if (elite_.size() <= k_) {
      throw std::runtime_error(
          "dual-heap metric cut is missing its k+1 pivot");
    }
    const distanceType threshold_distance =
        static_cast<distanceType>(
            cut_ * elite_[k_].distance);
    const Pair threshold{
        static_cast<indexType>(0), threshold_distance};
    const Pair elite_back{
        elite_.back().id, elite_.back().distance};
    if (pair_less(threshold, elite_back)) {
      rebuild_after_exceptional_cut(threshold);
      return;
    }
    clean_tail();
    while (!tail_.empty()) {
      const Entry entry = tail_.top();
      const Pair item{entry.id, entry.distance};
      if (!pair_less(threshold, item)) break;
      tail_.pop();
      deactivate(entry);
      clean_tail();
    }
  }

  size_t beam_size_ = 0;
  size_t k_ = 0;
  size_t elite_limit_ = 0;
  size_t active_count_ = 0;
  double cut_ = 0.0;
  std::vector<Entry> elite_;
  std::vector<Entry> visited_;
  std::priority_queue<
      Entry, std::vector<Entry>,
      DualHeapBeamMaxCompare<indexType, distanceType>>
      tail_;
  std::priority_queue<
      Entry, std::vector<Entry>,
      DualHeapBeamMinCompare<indexType, distanceType>>
      pending_;
  DualHeapBeamStatusTable<indexType> status_;
};

template <typename indexType, typename distanceType>
struct DualHeapBeamWorkspace {
  DualHeapBeamState<indexType, distanceType> state;
  std::vector<std::pair<indexType, distanceType>> candidates;
  std::vector<indexType> keep;
};

template <typename indexType, typename distanceType>
struct DualHeapBeamSearchResult {
  std::vector<std::pair<indexType, distanceType>> frontier;
  std::vector<std::pair<indexType, distanceType>> visited;
  size_t distance_computations;
};

template <typename Point, typename PointRange, typename indexType>
DualHeapBeamSearchResult<indexType, typename Point::distanceType>
beam_search_dual_heap(
    Point query, Graph<indexType>& graph, PointRange& points,
    const parlay::sequence<indexType>& starting_points, QueryParams& params,
    DualHeapBeamWorkspace<indexType, typename Point::distanceType>& workspace) {
  using distanceType = typename Point::distanceType;
  using Pair = std::pair<indexType, distanceType>;
  if (params.k <= 0 || params.beamSize < params.k) {
    throw std::runtime_error(
        "dual-heap query requires 0 < k <= beam");
  }
  workspace.state.reset(
      static_cast<size_t>(params.beamSize),
      static_cast<size_t>(params.k),
      params.cut);
  workspace.candidates.clear();
  workspace.candidates.reserve(graph.max_degree());
  workspace.keep.clear();
  workspace.keep.reserve(graph.max_degree());
  for (indexType start : starting_points) {
    workspace.state.add_start(
        start, points[start].distance(query));
  }

  const int bits = std::max<int>(
      10, std::ceil(std::log2(params.beamSize * params.beamSize)) - 2);
  std::vector<indexType> hash_filter(size_t{1} << bits, -1);
  auto has_been_seen = [&](indexType candidate) {
    const size_t location =
        parlay::hash64_2(candidate) & (hash_filter.size() - 1);
    if (hash_filter[location] == candidate) return true;
    hash_filter[location] = candidate;
    return false;
  };

  size_t distance_computations = starting_points.size();
  int visited_count = 0;
  while (workspace.state.has_unvisited() &&
         visited_count < params.limit) {
    const auto current = workspace.state.next_unvisited();
    graph[current.id].prefetch();
    workspace.state.visit(current);
    ++visited_count;

    workspace.candidates.clear();
    workspace.keep.clear();
    const long neighbor_count = std::min<long>(
        graph[current.id].size(), params.degree_limit);
    for (indexType neighbor_index = 0;
         neighbor_index < neighbor_count; ++neighbor_index) {
      const indexType candidate =
          graph[current.id][neighbor_index];
      if (candidate == query.id() ||
          has_been_seen(candidate)) {
        continue;
      }
      workspace.keep.push_back(candidate);
      points[candidate].prefetch();
    }

    const distanceType cutoff =
        workspace.state.premerge_distance_cutoff();
    for (indexType candidate : workspace.keep) {
      const distanceType distance =
          points[candidate].distance(query);
      ++distance_computations;
      if (distance >= cutoff) continue;
      workspace.candidates.push_back({candidate, distance});
    }
    std::sort(
        workspace.candidates.begin(),
        workspace.candidates.end(),
        [](const Pair& left, const Pair& right) {
          return left.second < right.second ||
                 (left.second == right.second &&
                  left.first < right.first);
        });
    workspace.state.merge_candidates(
        workspace.candidates, points[0].is_metric());
  }
  return {
      workspace.state.sorted_frontier(),
      workspace.state.sorted_visited(),
      distance_computations};
}

// Allocation-stable indexed-heap successor to the dual-heap treatment.
//
// Every active local vertex has exact positions in the small sorted elite,
// tail max-heap, and pending min-heap.  Visit and eviction therefore remove
// entries in place in O(log beam), with no lazy stale entries.  Dense
// epoch-stamped status is prepared before the timed batch and reset without
// clearing or allocation for each query.
template <typename indexType, typename distanceType>
struct IndexedHeapBeamEntry {
  indexType id;
  distanceType distance;
};

template <typename indexType, typename distanceType>
static inline bool indexed_heap_pair_less(
    const IndexedHeapBeamEntry<indexType, distanceType>& left,
    const IndexedHeapBeamEntry<indexType, distanceType>& right) {
  return left.distance < right.distance ||
         (left.distance == right.distance && left.id < right.id);
}

template <typename indexType, typename distanceType>
class IndexedHeapBeamState {
 public:
  using Entry = IndexedHeapBeamEntry<indexType, distanceType>;
  using Pair = std::pair<indexType, distanceType>;

  void prepare(
      size_t maximum_points, size_t maximum_beam,
      size_t maximum_visits) {
    if (status_.size() < maximum_points) {
      status_.resize(maximum_points);
    }
    elite_.reserve(maximum_beam);
    tail_.reserve(maximum_beam);
    pending_.reserve(maximum_beam);
    visited_.reserve(maximum_visits);
    frontier_output_.reserve(maximum_beam);
    maximum_points_ = std::max(maximum_points_, maximum_points);
    maximum_beam_ = std::max(maximum_beam_, maximum_beam);
    maximum_visits_ = std::max(maximum_visits_, maximum_visits);
  }

  void reset(
      size_t point_count, size_t beam_size, size_t k, double cut,
      size_t visit_limit) {
    if (point_count == 0 || point_count > maximum_points_ ||
        beam_size == 0 || beam_size > maximum_beam_ ||
        k == 0 || beam_size < k ||
        visit_limit > maximum_visits_ ||
        !(cut > 0.0) || !std::isfinite(cut)) {
      throw std::runtime_error(
          "indexed-heap query exceeds its prepared envelope");
    }
    ++epoch_;
    if (epoch_ == 0 || epoch_ > kEpochMask) {
      for (Status& status : status_) status.stamp_flags = 0;
      epoch_ = 1;
    }
    point_count_ = point_count;
    beam_size_ = beam_size;
    k_ = k;
    cut_ = cut;
    elite_limit_ = std::min(beam_size_, k_ + 1);
    active_count_ = 0;
    elite_.clear();
    tail_.clear();
    pending_.clear();
    visited_.clear();
    frontier_output_.clear();
  }

  void add_start(indexType id, distanceType distance) {
    Status& status = touch(id);
    if (is_active(status) || is_visited(status)) {
      throw std::runtime_error(
          "indexed-heap requires distinct unvisited starting points");
    }
    insert_active(id, distance);
  }

  bool has_unvisited() const { return !pending_.empty(); }

  Entry next_unvisited() const {
    if (pending_.empty()) {
      throw std::runtime_error(
          "indexed-heap frontier has no unvisited entry");
    }
    return pending_.front();
  }

  void visit(const Entry& entry) {
    Status& status = current(entry.id);
    if (!is_active(status) || is_visited(status) ||
        status.pending_pos != 0) {
      throw std::runtime_error(
          "indexed-heap attempted to visit a non-pending entry");
    }
    const Entry removed = pending_remove_at(0);
    if (removed.id != entry.id ||
        removed.distance != entry.distance) {
      throw std::runtime_error(
          "indexed-heap pending root changed before visit");
    }
    status.stamp_flags |= kVisitedBit;
    visited_.push_back(entry);
  }

  distanceType premerge_distance_cutoff() const {
    if (active_count_ < beam_size_) {
      return static_cast<distanceType>(
          std::numeric_limits<int>::max());
    }
    return farthest().distance;
  }

  void merge_candidates(
      const std::vector<Pair>& sorted_candidates, bool is_metric) {
    for (const Pair& candidate : sorted_candidates) {
      insert_active(candidate.first, candidate.second);
      while (active_count_ > beam_size_) remove_farthest();
    }
    if (is_metric && active_count_ > k_) apply_metric_cut();
  }

  void finalize() {
    frontier_output_.clear();
    for (const Entry& entry : elite_) {
      frontier_output_.push_back({entry.id, entry.distance});
    }
    for (const Entry& entry : tail_) {
      frontier_output_.push_back({entry.id, entry.distance});
    }
    std::sort(
        frontier_output_.begin(), frontier_output_.end(), pair_less);
    std::sort(
        visited_.begin(), visited_.end(), entry_less);
    if (frontier_output_.size() != active_count_) {
      throw std::runtime_error(
          "indexed-heap active frontier cardinality mismatch");
    }
  }

  const std::vector<Pair>& frontier() const {
    return frontier_output_;
  }

  const std::vector<Entry>& visited_entries() const {
    return visited_;
  }

 private:
  static constexpr uint32_t kVisitedBit = uint32_t{1} << 31;
  static constexpr uint32_t kEpochMask = ~kVisitedBit;
  static constexpr int32_t kNoPosition = -1;

  struct Status {
    uint32_t stamp_flags = 0;
    int32_t elite_pos = kNoPosition;
    int32_t tail_pos = kNoPosition;
    int32_t pending_pos = kNoPosition;
  };

  static bool pair_less(const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second && left.first < right.first);
  }

  static bool entry_less(const Entry& left, const Entry& right) {
    return indexed_heap_pair_less(left, right);
  }

  bool is_visited(const Status& status) const {
    return (status.stamp_flags & kVisitedBit) != 0;
  }

  bool is_active(const Status& status) const {
    return status.elite_pos != kNoPosition ||
           status.tail_pos != kNoPosition;
  }

  Status& touch(indexType id) {
    if (id < 0 || static_cast<size_t>(id) >= point_count_) {
      throw std::runtime_error(
          "indexed-heap local id is outside the prepared shard");
    }
    Status& status = status_[static_cast<size_t>(id)];
    if ((status.stamp_flags & kEpochMask) != epoch_) {
      status.stamp_flags = epoch_;
      status.elite_pos = kNoPosition;
      status.tail_pos = kNoPosition;
      status.pending_pos = kNoPosition;
    }
    return status;
  }

  Status& current(indexType id) {
    Status& status = touch(id);
    return status;
  }

  const Entry& farthest() const {
    if (!tail_.empty()) return tail_.front();
    if (elite_.empty()) {
      throw std::runtime_error(
          "indexed-heap cannot read an empty frontier");
    }
    return elite_.back();
  }

  void elite_reindex(size_t begin) {
    for (size_t position = begin; position < elite_.size(); ++position) {
      current(elite_[position].id).elite_pos =
          static_cast<int32_t>(position);
    }
  }

  void elite_insert(const Entry& entry) {
    const size_t position = static_cast<size_t>(
        std::lower_bound(
            elite_.begin(), elite_.end(), entry, entry_less) -
        elite_.begin());
    elite_.insert(elite_.begin() + position, entry);
    elite_reindex(position);
  }

  Entry elite_remove_at(size_t position) {
    if (position >= elite_.size()) {
      throw std::runtime_error(
          "indexed-heap elite removal is out of range");
    }
    const Entry removed = elite_[position];
    current(removed.id).elite_pos = kNoPosition;
    elite_.erase(elite_.begin() + position);
    elite_reindex(position);
    return removed;
  }

  void tail_swap(size_t left, size_t right) {
    std::swap(tail_[left], tail_[right]);
    current(tail_[left].id).tail_pos =
        static_cast<int32_t>(left);
    current(tail_[right].id).tail_pos =
        static_cast<int32_t>(right);
  }

  void tail_sift_up(size_t position) {
    while (position > 0) {
      const size_t parent = (position - 1) / 2;
      if (!entry_less(tail_[parent], tail_[position])) break;
      tail_swap(parent, position);
      position = parent;
    }
  }

  void tail_sift_down(size_t position) {
    while (true) {
      const size_t left = 2 * position + 1;
      if (left >= tail_.size()) break;
      const size_t right = left + 1;
      size_t largest = left;
      if (right < tail_.size() &&
          entry_less(tail_[left], tail_[right])) {
        largest = right;
      }
      if (!entry_less(tail_[position], tail_[largest])) break;
      tail_swap(position, largest);
      position = largest;
    }
  }

  void tail_push(const Entry& entry) {
    Status& status = current(entry.id);
    if (status.tail_pos != kNoPosition) {
      throw std::runtime_error(
          "indexed-heap duplicate tail insertion");
    }
    status.tail_pos = static_cast<int32_t>(tail_.size());
    tail_.push_back(entry);
    tail_sift_up(tail_.size() - 1);
  }

  Entry tail_remove_at(size_t position) {
    if (position >= tail_.size()) {
      throw std::runtime_error(
          "indexed-heap tail removal is out of range");
    }
    const Entry removed = tail_[position];
    current(removed.id).tail_pos = kNoPosition;
    const size_t last = tail_.size() - 1;
    if (position == last) {
      tail_.pop_back();
      return removed;
    }
    tail_[position] = tail_[last];
    tail_.pop_back();
    current(tail_[position].id).tail_pos =
        static_cast<int32_t>(position);
    if (position > 0 &&
        entry_less(tail_[(position - 1) / 2], tail_[position])) {
      tail_sift_up(position);
    } else {
      tail_sift_down(position);
    }
    return removed;
  }

  void pending_swap(size_t left, size_t right) {
    std::swap(pending_[left], pending_[right]);
    current(pending_[left].id).pending_pos =
        static_cast<int32_t>(left);
    current(pending_[right].id).pending_pos =
        static_cast<int32_t>(right);
  }

  void pending_sift_up(size_t position) {
    while (position > 0) {
      const size_t parent = (position - 1) / 2;
      if (!entry_less(pending_[position], pending_[parent])) break;
      pending_swap(parent, position);
      position = parent;
    }
  }

  void pending_sift_down(size_t position) {
    while (true) {
      const size_t left = 2 * position + 1;
      if (left >= pending_.size()) break;
      const size_t right = left + 1;
      size_t smallest = left;
      if (right < pending_.size() &&
          entry_less(pending_[right], pending_[left])) {
        smallest = right;
      }
      if (!entry_less(pending_[smallest], pending_[position])) break;
      pending_swap(position, smallest);
      position = smallest;
    }
  }

  void pending_push(const Entry& entry) {
    Status& status = current(entry.id);
    if (status.pending_pos != kNoPosition) {
      throw std::runtime_error(
          "indexed-heap duplicate pending insertion");
    }
    status.pending_pos = static_cast<int32_t>(pending_.size());
    pending_.push_back(entry);
    pending_sift_up(pending_.size() - 1);
  }

  Entry pending_remove_at(size_t position) {
    if (position >= pending_.size()) {
      throw std::runtime_error(
          "indexed-heap pending removal is out of range");
    }
    const Entry removed = pending_[position];
    current(removed.id).pending_pos = kNoPosition;
    const size_t last = pending_.size() - 1;
    if (position == last) {
      pending_.pop_back();
      return removed;
    }
    pending_[position] = pending_[last];
    pending_.pop_back();
    current(pending_[position].id).pending_pos =
        static_cast<int32_t>(position);
    if (position > 0 &&
        entry_less(
            pending_[position], pending_[(position - 1) / 2])) {
      pending_sift_up(position);
    } else {
      pending_sift_down(position);
    }
    return removed;
  }

  void insert_active(indexType id, distanceType distance) {
    Status& status = touch(id);
    if (is_active(status)) return;
    const Entry entry{id, distance};
    ++active_count_;
    if (!is_visited(status)) pending_push(entry);
    if (elite_.size() < elite_limit_) {
      elite_insert(entry);
    } else if (entry_less(entry, elite_.back())) {
      const Entry demoted =
          elite_remove_at(elite_.size() - 1);
      tail_push(demoted);
      elite_insert(entry);
    } else {
      tail_push(entry);
    }
  }

  void deactivate(const Entry& entry) {
    Status& status = current(entry.id);
    if (status.elite_pos != kNoPosition ||
        status.tail_pos != kNoPosition) {
      throw std::runtime_error(
          "indexed-heap representation was not removed before deactivate");
    }
    if (status.pending_pos != kNoPosition) {
      pending_remove_at(
          static_cast<size_t>(status.pending_pos));
    }
    --active_count_;
  }

  void remove_farthest() {
    Entry removed;
    if (!tail_.empty()) {
      removed = tail_remove_at(0);
    } else {
      removed = elite_remove_at(elite_.size() - 1);
    }
    deactivate(removed);
  }

  void apply_metric_cut() {
    if (elite_.size() <= k_) {
      throw std::runtime_error(
          "indexed-heap metric cut is missing its k+1 pivot");
    }
    const distanceType threshold_distance =
        static_cast<distanceType>(
            cut_ * elite_[k_].distance);
    const Entry threshold{
        static_cast<indexType>(0), threshold_distance};
    while (active_count_ > 0 &&
           entry_less(threshold, farthest())) {
      remove_farthest();
    }
  }

  std::vector<Status> status_;
  uint32_t epoch_ = 0;
  size_t maximum_points_ = 0;
  size_t maximum_beam_ = 0;
  size_t maximum_visits_ = 0;
  size_t point_count_ = 0;
  size_t beam_size_ = 0;
  size_t k_ = 0;
  size_t elite_limit_ = 0;
  size_t active_count_ = 0;
  double cut_ = 0.0;
  std::vector<Entry> elite_;
  std::vector<Entry> tail_;
  std::vector<Entry> pending_;
  std::vector<Entry> visited_;
  std::vector<Pair> frontier_output_;
};

template <typename indexType, typename distanceType>
struct IndexedHeapBeamWorkspace {
  using Pair = std::pair<indexType, distanceType>;

  IndexedHeapBeamState<indexType, distanceType> state;
  std::vector<Pair> candidates;
  std::vector<indexType> keep;
  std::vector<indexType> seen_values;
  std::vector<uint32_t> seen_stamps;
  uint32_t seen_epoch = 0;
  size_t seen_size = 0;

  void prepare(
      size_t maximum_points, size_t maximum_beam,
      size_t maximum_visits, size_t maximum_degree) {
    state.prepare(maximum_points, maximum_beam, maximum_visits);
    candidates.reserve(maximum_degree);
    keep.reserve(maximum_degree);
    const int bits = std::max<int>(
        10,
        static_cast<int>(
            std::ceil(std::log2(maximum_beam * maximum_beam))) -
            2);
    const size_t maximum_seen_size = size_t{1} << bits;
    if (seen_values.size() < maximum_seen_size) {
      seen_values.resize(maximum_seen_size);
      seen_stamps.resize(maximum_seen_size);
    }
  }

  void begin_query(size_t beam_size) {
    const int bits = std::max<int>(
        10,
        static_cast<int>(
            std::ceil(std::log2(beam_size * beam_size))) -
            2);
    seen_size = size_t{1} << bits;
    if (seen_size > seen_values.size()) {
      throw std::runtime_error(
          "indexed-heap seen filter exceeds prepared envelope");
    }
    ++seen_epoch;
    if (seen_epoch == 0) {
      std::fill(seen_stamps.begin(), seen_stamps.end(), 0);
      seen_epoch = 1;
    }
    candidates.clear();
    keep.clear();
  }

  bool has_been_seen(indexType candidate) {
    const size_t location =
        parlay::hash64_2(candidate) & (seen_size - 1);
    if (seen_stamps[location] == seen_epoch &&
        seen_values[location] == candidate) {
      return true;
    }
    seen_stamps[location] = seen_epoch;
    seen_values[location] = candidate;
    return false;
  }
};

template <typename indexType, typename distanceType>
struct IndexedHeapBeamSearchView {
  const std::vector<std::pair<indexType, distanceType>>& frontier;
  const std::vector<
      IndexedHeapBeamEntry<indexType, distanceType>>& visited;
  size_t distance_computations;
};

template <typename Point, typename PointRange, typename indexType>
IndexedHeapBeamSearchView<indexType, typename Point::distanceType>
beam_search_indexed_heap(
    Point query, Graph<indexType>& graph, PointRange& points,
    const parlay::sequence<indexType>& starting_points, QueryParams& params,
    IndexedHeapBeamWorkspace<
        indexType, typename Point::distanceType>& workspace) {
  using distanceType = typename Point::distanceType;
  using Pair = std::pair<indexType, distanceType>;
  if (params.k <= 0 || params.beamSize < params.k) {
    throw std::runtime_error(
        "indexed-heap query requires 0 < k <= beam");
  }
  workspace.begin_query(static_cast<size_t>(params.beamSize));
  workspace.state.reset(
      points.size(),
      static_cast<size_t>(params.beamSize),
      static_cast<size_t>(params.k),
      params.cut,
      static_cast<size_t>(std::min<long>(
          params.limit, static_cast<long>(points.size()))));
  for (indexType start : starting_points) {
    workspace.state.add_start(
        start, points[start].distance(query));
  }

  size_t distance_computations = starting_points.size();
  int visited_count = 0;
  while (workspace.state.has_unvisited() &&
         visited_count < params.limit) {
    const auto current = workspace.state.next_unvisited();
    graph[current.id].prefetch();
    workspace.state.visit(current);
    ++visited_count;

    workspace.candidates.clear();
    workspace.keep.clear();
    const long neighbor_count = std::min<long>(
        graph[current.id].size(), params.degree_limit);
    for (indexType neighbor_index = 0;
         neighbor_index < neighbor_count; ++neighbor_index) {
      const indexType candidate =
          graph[current.id][neighbor_index];
      if (candidate == query.id() ||
          workspace.has_been_seen(candidate)) {
        continue;
      }
      workspace.keep.push_back(candidate);
      points[candidate].prefetch();
    }

    const distanceType cutoff =
        workspace.state.premerge_distance_cutoff();
    for (indexType candidate : workspace.keep) {
      const distanceType distance =
          points[candidate].distance(query);
      ++distance_computations;
      if (distance >= cutoff) continue;
      workspace.candidates.push_back({candidate, distance});
    }
    std::sort(
        workspace.candidates.begin(),
        workspace.candidates.end(),
        [](const Pair& left, const Pair& right) {
          return left.second < right.second ||
                 (left.second == right.second &&
                  left.first < right.first);
        });
    workspace.state.merge_candidates(
        workspace.candidates, points[0].is_metric());
  }
  workspace.state.finalize();
  return {
      workspace.state.frontier(),
      workspace.state.visited_entries(),
      distance_computations};
}

#ifdef BCI_ENABLE_FIXED_FRONTIER_BATCHED_DIAGNOSTIC
// Fresh-semantics high-beam successor for the uint8/192-D single-tag path.
// The sorted frontier and its union/cut transition remain byte-for-byte
// comparable with the vector reference.  Dense epoch/generation marks plus a
// lazy pending min-heap replace sorted visited insertion and the per-expansion
// set_difference over frontier x visited.
template <typename indexType, typename distanceType>
class FixedFrontierBatchedWorkspace {
 public:
  using Pair = std::pair<indexType, distanceType>;

  struct PendingEntry {
    indexType id;
    distanceType distance;
    uint32_t generation;
  };

  void prepare(
      size_t maximum_points, size_t maximum_beam,
      size_t maximum_visits, size_t maximum_degree) {
    if (status_.size() < maximum_points) {
      status_.resize(maximum_points);
    }
    frontier.reserve(maximum_beam);
    frontier_visited_.reserve(maximum_beam);
    scratch.reserve(maximum_beam + maximum_degree);
    scratch_visited_.reserve(maximum_beam + maximum_degree);
    candidates.reserve(maximum_degree);
    prefix_survivors.reserve(maximum_degree);
    keep.reserve(maximum_degree);
    pending_.reserve(maximum_beam * 4 + maximum_degree);
    visited_.reserve(maximum_visits);
    const int bits = std::max<int>(
        10,
        static_cast<int>(
            std::ceil(std::log2(maximum_beam * maximum_beam))) -
            2);
    const size_t maximum_seen = size_t{1} << bits;
    if (seen_values_.size() < maximum_seen) {
      seen_values_.resize(maximum_seen);
      seen_stamps_.resize(maximum_seen);
    }
    maximum_points_ = std::max(maximum_points_, maximum_points);
    maximum_beam_ = std::max(maximum_beam_, maximum_beam);
    maximum_visits_ = std::max(maximum_visits_, maximum_visits);
    maximum_degree_ = std::max(maximum_degree_, maximum_degree);
  }

  void begin_query(
      size_t point_count, size_t beam_size,
      size_t visit_limit) {
    if (point_count == 0 || point_count > maximum_points_ ||
        beam_size == 0 || beam_size > maximum_beam_ ||
        visit_limit > maximum_visits_) {
      throw std::runtime_error(
          "fixed-frontier query exceeds its prepared envelope");
    }
    ++epoch_;
    if (epoch_ == 0 || epoch_ > kEpochMask) {
      for (Status& status : status_) status.stamp_flags = 0;
      epoch_ = 1;
    }
    ++seen_epoch_;
    if (seen_epoch_ == 0) {
      std::fill(seen_stamps_.begin(), seen_stamps_.end(), 0);
      seen_epoch_ = 1;
    }
    const int bits = std::max<int>(
        10,
        static_cast<int>(
            std::ceil(std::log2(beam_size * beam_size))) -
            2);
    seen_size_ = size_t{1} << bits;
    if (seen_size_ > seen_values_.size()) {
      throw std::runtime_error(
          "fixed-frontier seen filter exceeds prepared envelope");
    }
    point_count_ = point_count;
    beam_size_ = beam_size;
    frontier.clear();
    frontier_visited_.clear();
    scratch.clear();
    scratch_visited_.clear();
    candidates.clear();
    prefix_survivors.clear();
    keep.clear();
    pending_.clear();
    visited_.clear();
  }

  bool has_been_seen(indexType candidate) {
    const size_t location =
        parlay::hash64_2(candidate) & (seen_size_ - 1);
    if (seen_stamps_[location] == seen_epoch_ &&
        seen_values_[location] == candidate) {
      return true;
    }
    seen_stamps_[location] = seen_epoch_;
    seen_values_[location] = candidate;
    return false;
  }

  void finalize_starts() {
    std::sort(frontier.begin(), frontier.end(), pair_less);
    frontier_visited_.assign(frontier.size(), uint8_t{0});
  }

  bool pop_next_unvisited(Pair& output) {
    // Frontier is already in the exact (distance,id) order.  Carrying the
    // visited bit beside each entry turns the reference set_difference into
    // one contiguous scan over at most B bytes.
    for (size_t i = 0; i < frontier.size(); ++i) {
      if (!frontier_visited_[i]) {
        Status& status = touch(frontier[i].first);
        if (is_visited(status)) {
          throw std::runtime_error(
              "fixed-frontier visited flag lost across re-entry");
        }
        status.stamp_flags |= kVisitedBit;
        frontier_visited_[i] = uint8_t{1};
        output = frontier[i];
        visited_.push_back(output);
        return true;
      }
    }
    return false;
  }

  void commit_scratch() {
    // Propagate flags for retained entries.  A previously expanded entry may
    // leave the frontier and re-enter after a collision in the deliberately
    // direct-mapped seen filter, so genuinely new entries consult the exact
    // epoch-stamped ever-visited state.
    scratch_visited_.clear();
    scratch_visited_.reserve(scratch.size());
    size_t old_position = 0;
    for (const Pair& entry : scratch) {
      while (old_position < frontier.size() &&
             pair_less(frontier[old_position], entry)) {
        ++old_position;
      }
      if (old_position < frontier.size() &&
          !pair_less(entry, frontier[old_position]) &&
          !pair_less(frontier[old_position], entry)) {
        scratch_visited_.push_back(
            frontier_visited_[old_position]);
        ++old_position;
      } else {
        scratch_visited_.push_back(
            is_visited(touch(entry.first))
                ? uint8_t{1}
                : uint8_t{0});
      }
    }
    frontier.swap(scratch);
    frontier_visited_.swap(scratch_visited_);
    scratch.clear();
    scratch_visited_.clear();
  }

  void finalize() {
    std::sort(visited_.begin(), visited_.end(), pair_less);
  }

  const std::vector<Pair>& visited() const { return visited_; }

  std::vector<Pair> frontier;
  std::vector<Pair> scratch;
  std::vector<Pair> candidates;
  std::vector<Pair> prefix_survivors;
  std::vector<indexType> keep;

 private:
  static constexpr uint32_t kActiveBit = uint32_t{1} << 31;
  static constexpr uint32_t kVisitedBit = uint32_t{1} << 30;
  static constexpr uint32_t kEpochMask =
      ~(kActiveBit | kVisitedBit);

  struct Status {
    uint32_t stamp_flags = 0;
    uint32_t generation = 0;
  };

  struct PendingMinCompare {
    bool operator()(
        const PendingEntry& left,
        const PendingEntry& right) const {
      return pair_less(
          Pair{right.id, right.distance},
          Pair{left.id, left.distance});
    }
  };

  static bool pair_less(const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second &&
            left.first < right.first);
  }

  bool is_active(const Status& status) const {
    return (status.stamp_flags & kActiveBit) != 0;
  }

  bool is_visited(const Status& status) const {
    return (status.stamp_flags & kVisitedBit) != 0;
  }

  Status& touch(indexType id) {
    if (id < 0 || static_cast<size_t>(id) >= point_count_) {
      throw std::runtime_error(
          "fixed-frontier local id is outside prepared shard");
    }
    Status& status = status_[static_cast<size_t>(id)];
    if ((status.stamp_flags & kEpochMask) != epoch_) {
      status.stamp_flags = epoch_;
      status.generation = 0;
    }
    return status;
  }

  void push_pending(
      indexType id, distanceType distance,
      uint32_t generation) {
    pending_.push_back({id, distance, generation});
    std::push_heap(
        pending_.begin(), pending_.end(),
        PendingMinCompare{});
  }

  void activate(const Pair& entry) {
    Status& status = touch(entry.first);
    if (is_active(status)) {
      throw std::runtime_error(
          "fixed-frontier duplicate activation");
    }
    ++status.generation;
    if (status.generation == 0) {
      throw std::runtime_error(
          "fixed-frontier generation overflow");
    }
    status.stamp_flags |= kActiveBit;
    if (!is_visited(status)) {
      push_pending(
          entry.first, entry.second,
          status.generation);
    }
  }

  void deactivate(const Pair& entry) {
    Status& status = touch(entry.first);
    if (!is_active(status)) {
      throw std::runtime_error(
          "fixed-frontier duplicate deactivation");
    }
    status.stamp_flags &= ~kActiveBit;
  }

  void rebuild_pending() {
    pending_.clear();
    for (const Pair& entry : frontier) {
      Status& status = touch(entry.first);
      if (is_active(status) && !is_visited(status)) {
        pending_.push_back(
            {entry.first, entry.second, status.generation});
      }
    }
    std::make_heap(
        pending_.begin(), pending_.end(),
        PendingMinCompare{});
  }

  std::vector<Status> status_;
  std::vector<uint8_t> frontier_visited_;
  std::vector<uint8_t> scratch_visited_;
  std::vector<PendingEntry> pending_;
  std::vector<Pair> visited_;
  std::vector<indexType> seen_values_;
  std::vector<uint32_t> seen_stamps_;
  uint32_t epoch_ = 0;
  uint32_t seen_epoch_ = 0;
  size_t seen_size_ = 0;
  size_t point_count_ = 0;
  size_t beam_size_ = 0;
  size_t maximum_points_ = 0;
  size_t maximum_beam_ = 0;
  size_t maximum_visits_ = 0;
  size_t maximum_degree_ = 0;
};

template <typename indexType, typename distanceType>
struct FixedFrontierBatchedResult {
  const std::vector<std::pair<indexType, distanceType>>& frontier;
  const std::vector<std::pair<indexType, distanceType>>& visited;
  size_t distance_computations;
  size_t staged_candidates;
  size_t prefix_rejected_candidates;
};

template <typename Point, typename PointRange, typename indexType>
FixedFrontierBatchedResult<
    indexType, typename Point::distanceType>
beam_search_fixed_frontier_batched_l2(
    Point query, Graph<indexType>& graph, PointRange& points,
    const parlay::sequence<indexType>& starting_points,
    QueryParams& params, int prefix_dimensions,
    FixedFrontierBatchedWorkspace<
        indexType, typename Point::distanceType>& workspace) {
  using distanceType = typename Point::distanceType;
  using Pair = std::pair<indexType, distanceType>;
  const auto less = [](const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second &&
            left.first < right.first);
  };
  if (params.k <= 0 || params.beamSize < params.k ||
      points.dimension() != 192 ||
      points.aligned_dimension() < 192 ||
      (prefix_dimensions != 64 &&
       prefix_dimensions != 128)) {
    throw std::runtime_error(
        "fixed-frontier batched L2 requires uint8/192-D, "
        "0 < k <= beam, and prefix 64 or 128");
  }
  workspace.begin_query(
      points.size(), static_cast<size_t>(params.beamSize),
      static_cast<size_t>(std::min<long>(
          params.limit, static_cast<long>(points.size()))));
  size_t distance_computations = 0;
  for (indexType start : starting_points) {
    workspace.frontier.push_back(
        {start, points[start].distance(query)});
    ++distance_computations;
  }
  workspace.finalize_starts();

  size_t staged_candidates = 0;
  size_t prefix_rejected_candidates = 0;
  int visited_count = 0;
  Pair current{};
  while (visited_count < params.limit &&
         workspace.pop_next_unvisited(current)) {
    graph[current.first].prefetch();
    ++visited_count;
    workspace.candidates.clear();
    workspace.prefix_survivors.clear();
    workspace.keep.clear();
    const long neighbor_count = std::min<long>(
        graph[current.first].size(), params.degree_limit);
    for (indexType neighbor_index = 0;
         neighbor_index < neighbor_count; ++neighbor_index) {
      const indexType candidate =
          graph[current.first][neighbor_index];
      if (candidate == query.id() ||
          workspace.has_been_seen(candidate)) {
        continue;
      }
      workspace.keep.push_back(candidate);
      const uint8_t* values = points[candidate].get();
      __builtin_prefetch(values, 0, 1);
      if (prefix_dimensions == 128) {
        __builtin_prefetch(values + 64, 0, 1);
      }
    }

    const distanceType cutoff =
        workspace.frontier.size() <
                static_cast<size_t>(params.beamSize)
            ? static_cast<distanceType>(
                  std::numeric_limits<int>::max())
            : workspace.frontier.back().second;
    const uint8_t* query_values = query.get();
    for (indexType candidate : workspace.keep) {
      const uint8_t* candidate_values =
          points[candidate].get();
      const distanceType prefix =
          l2_sq_uint8_avx2(
              candidate_values, query_values,
              static_cast<unsigned>(prefix_dimensions));
      ++distance_computations;
      ++staged_candidates;
      if (prefix >= cutoff) {
        ++prefix_rejected_candidates;
        continue;
      }
      workspace.prefix_survivors.push_back(
          {candidate, prefix});
      for (int offset = prefix_dimensions;
           offset < 192; offset += 64) {
        __builtin_prefetch(
            candidate_values + offset, 0, 1);
      }
    }
    for (const Pair& survivor :
         workspace.prefix_survivors) {
      const uint8_t* candidate_values =
          points[survivor.first].get();
      const distanceType distance =
          survivor.second +
          l2_sq_uint8_avx2(
              candidate_values + prefix_dimensions,
              query_values + prefix_dimensions,
              static_cast<unsigned>(
                  192 - prefix_dimensions));
      if (distance >= cutoff) continue;
      workspace.candidates.push_back(
          {survivor.first, distance});
    }
    std::sort(
        workspace.candidates.begin(),
        workspace.candidates.end(), less);

    workspace.scratch.clear();
    std::set_union(
        workspace.frontier.begin(),
        workspace.frontier.end(),
        workspace.candidates.begin(),
        workspace.candidates.end(),
        std::back_inserter(workspace.scratch), less);
    if (workspace.scratch.size() >
        static_cast<size_t>(params.beamSize)) {
      workspace.scratch.resize(
          static_cast<size_t>(params.beamSize));
    }
    if (workspace.scratch.size() >
            static_cast<size_t>(params.k) &&
        points[0].is_metric()) {
      const Pair threshold{
          0,
          static_cast<distanceType>(
              params.cut *
              workspace.scratch[params.k].second)};
      workspace.scratch.resize(
          static_cast<size_t>(
              std::upper_bound(
                  workspace.scratch.begin(),
                  workspace.scratch.end(),
                  threshold, less) -
              workspace.scratch.begin()));
    }
    workspace.commit_scratch();
  }
  workspace.finalize();
  return {
      workspace.frontier, workspace.visited(),
      distance_computations, staged_candidates,
      prefix_rejected_candidates};
}

struct FixedFrontierToyResult {
  std::vector<std::pair<int32_t, float>> frontier;
  std::vector<std::pair<int32_t, float>> visited;
  size_t logical_computations = 0;
};

static FixedFrontierToyResult fixed_frontier_reference_toy(
    const std::vector<std::vector<int32_t>>& graph,
    const std::vector<float>& distances,
    const std::vector<float>& prefix_distances,
    size_t beam, size_t k, double cut) {
  using Pair = std::pair<int32_t, float>;
  const auto less = [](const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second &&
            left.first < right.first);
  };
  const int bits = std::max<int>(
      10,
      static_cast<int>(
          std::ceil(std::log2(beam * beam))) -
          2);
  std::vector<int32_t> hash_filter(size_t{1} << bits, -1);
  auto seen = [&](int32_t id) {
    const size_t slot =
        parlay::hash64_2(id) & (hash_filter.size() - 1);
    if (hash_filter[slot] == id) return true;
    hash_filter[slot] = id;
    return false;
  };
  std::vector<Pair> frontier{{0, distances[0]}};
  std::vector<Pair> visited;
  std::vector<Pair> unvisited(beam);
  unvisited[0] = frontier[0];
  std::vector<Pair> candidates;
  std::vector<Pair> merged(beam + 512);
  size_t logical = 1;
  int remaining = 1;
  int visit_count = 0;
  while (remaining > 0 && visit_count < 1'000) {
    const Pair current = unvisited[0];
    visited.insert(
        std::upper_bound(
            visited.begin(), visited.end(),
            current, less),
        current);
    ++visit_count;
    candidates.clear();
    const float cutoff_distance =
        frontier.size() < beam
            ? static_cast<float>(
                  std::numeric_limits<int>::max())
            : frontier.back().second;
    for (int32_t id : graph[current.first]) {
      if (seen(id)) continue;
      ++logical;
      if (prefix_distances[id] >= cutoff_distance) continue;
      const float distance = distances[id];
      if (distance >= cutoff_distance) continue;
      candidates.push_back({id, distance});
    }
    std::sort(candidates.begin(), candidates.end(), less);
    size_t merged_size = static_cast<size_t>(
        std::set_union(
            frontier.begin(), frontier.end(),
            candidates.begin(), candidates.end(),
            merged.begin(), less) -
        merged.begin());
    merged_size = std::min(beam, merged_size);
    if (merged_size > k) {
      const Pair threshold{
          0,
          static_cast<float>(
              cut * merged[k].second)};
      merged_size = static_cast<size_t>(
          std::upper_bound(
              merged.begin(), merged.begin() + merged_size,
              threshold, less) -
          merged.begin());
    }
    frontier.assign(merged.begin(), merged.begin() + merged_size);
    remaining = static_cast<int>(
        std::set_difference(
            frontier.begin(), frontier.end(),
            visited.begin(), visited.end(),
            unvisited.begin(), less) -
        unvisited.begin());
  }
  return {
      std::move(frontier), std::move(visited), logical};
}

static FixedFrontierToyResult fixed_frontier_successor_toy(
    const std::vector<std::vector<int32_t>>& graph,
    const std::vector<float>& distances,
    const std::vector<float>& prefix_distances,
    size_t beam, size_t k, double cut) {
  using Pair = std::pair<int32_t, float>;
  const auto less = [](const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second &&
            left.first < right.first);
  };
  size_t maximum_degree = 1;
  for (const auto& neighbors : graph) {
    maximum_degree =
        std::max(maximum_degree, neighbors.size());
  }
  FixedFrontierBatchedWorkspace<int32_t, float> workspace;
  workspace.prepare(
      graph.size(), beam, 1'000, maximum_degree);
  workspace.begin_query(graph.size(), beam, 1'000);
  workspace.frontier.push_back({0, distances[0]});
  workspace.finalize_starts();
  size_t logical = 1;
  int visit_count = 0;
  Pair current{};
  while (visit_count < 1'000 &&
         workspace.pop_next_unvisited(current)) {
    ++visit_count;
    workspace.candidates.clear();
    const float cutoff_distance =
        workspace.frontier.size() < beam
            ? static_cast<float>(
                  std::numeric_limits<int>::max())
            : workspace.frontier.back().second;
    for (int32_t id : graph[current.first]) {
      if (workspace.has_been_seen(id)) continue;
      ++logical;
      if (prefix_distances[id] >= cutoff_distance) continue;
      const float distance = distances[id];
      if (distance >= cutoff_distance) continue;
      workspace.candidates.push_back({id, distance});
    }
    std::sort(
        workspace.candidates.begin(),
        workspace.candidates.end(), less);
    workspace.scratch.clear();
    std::set_union(
        workspace.frontier.begin(),
        workspace.frontier.end(),
        workspace.candidates.begin(),
        workspace.candidates.end(),
        std::back_inserter(workspace.scratch), less);
    if (workspace.scratch.size() > beam) {
      workspace.scratch.resize(beam);
    }
    if (workspace.scratch.size() > k) {
      const Pair threshold{
          0,
          static_cast<float>(
              cut * workspace.scratch[k].second)};
      workspace.scratch.resize(
          static_cast<size_t>(
              std::upper_bound(
                  workspace.scratch.begin(),
                  workspace.scratch.end(),
                  threshold, less) -
              workspace.scratch.begin()));
    }
    workspace.commit_scratch();
  }
  workspace.finalize();
  return {
      workspace.frontier, workspace.visited(), logical};
}

static int fixed_frontier_batched_self_test() {
  constexpr size_t n = 20'500;
  std::vector<std::vector<int32_t>> graph(n);
  std::vector<float> distances(n, 129.0f);
  std::vector<float> prefixes(n, 64.0f);
  distances[0] = 200.0f;
  prefixes[0] = 100.0f;
  for (int32_t id = 1; id < static_cast<int32_t>(n); ++id) {
    distances[id] =
        50.0f + static_cast<float>((id * 37) % 80);
    prefixes[id] =
        std::max(0.0f, distances[id] -
                           static_cast<float>(id % 23));
  }
  for (int32_t id = 1; id <= 300; ++id) {
    graph[0].push_back(id);
  }
  for (int32_t id = 1; id <= 300; ++id) {
    for (int step = 1; step <= 8; ++step) {
      const int32_t next =
          1 + static_cast<int32_t>(
                  (static_cast<uint64_t>(id) * 97 +
                   static_cast<uint64_t>(step) * 193) %
                  (n - 1));
      if (next != id) graph[id].push_back(next);
    }
  }
  // Force a direct-mapped seen-filter collision and later re-observation.
  constexpr size_t mask = (size_t{1} << 14) - 1;
  std::unordered_map<size_t, int32_t> first_by_slot;
  int32_t collision_a = -1;
  int32_t collision_b = -1;
  for (int32_t id = 301;
       id < static_cast<int32_t>(n); ++id) {
    const size_t slot = parlay::hash64_2(id) & mask;
    auto inserted = first_by_slot.emplace(slot, id);
    if (!inserted.second) {
      collision_a = inserted.first->second;
      collision_b = id;
      break;
    }
  }
  if (collision_a < 0 || collision_b < 0) {
    throw std::runtime_error(
        "fixed-frontier self-test could not find seen collision");
  }
  graph[0].push_back(collision_a);
  graph[0].push_back(collision_b);
  graph[1].push_back(collision_a);

  for (const auto& test :
       std::vector<std::pair<size_t, double>>{
           {192, 1.35}, {256, 1.35}, {192, 3.0}}) {
    const auto reference = fixed_frontier_reference_toy(
        graph, distances, prefixes, test.first, 10, test.second);
    const auto successor = fixed_frontier_successor_toy(
        graph, distances, prefixes, test.first, 10, test.second);
    if (reference.frontier != successor.frontier ||
        reference.visited != successor.visited ||
        reference.logical_computations !=
            successor.logical_computations) {
      throw std::runtime_error(
          "fixed-frontier fresh-semantics differential failed "
          "for beam=" + std::to_string(test.first) +
          " cut=" + std::to_string(test.second) +
          " frontier=" +
          std::to_string(reference.frontier.size()) + "/" +
          std::to_string(successor.frontier.size()) +
          " visited=" +
          std::to_string(reference.visited.size()) + "/" +
          std::to_string(successor.visited.size()) +
          " logical=" +
          std::to_string(reference.logical_computations) + "/" +
          std::to_string(successor.logical_computations));
    }
  }
  std::printf(
      "[fixed frontier batched self-test] PASS "
      "beams=192,256 top10=exact visited=exact logical=exact "
      "seen_collision=%d/%d\n",
      collision_a, collision_b);
  return 0;
}
#endif

template <typename indexType, typename distanceType>
struct ExactDistanceMemo {
  ExactDistanceMemo() = default;

  explicit ExactDistanceMemo(size_t domain_size)
      : values(domain_size), stamps(domain_size, 0) {}

  void reset_domain(size_t domain_size) {
    values.resize(domain_size);
    stamps.assign(domain_size, 0);
    epoch = 0;
  }

  void begin_query() {
    ++epoch;
    if (epoch == 0) {
      std::fill(stamps.begin(), stamps.end(), 0);
      epoch = 1;
    }
  }

  bool lookup(indexType candidate, distanceType& distance) const {
    const size_t position = static_cast<size_t>(candidate);
    if (candidate < 0 || position >= stamps.size()) {
      throw std::runtime_error(
          "exact-distance memo candidate is outside local-ID domain");
    }
    if (stamps[position] != epoch) return false;
    distance = values[position];
    return true;
  }

  void insert(indexType candidate, distanceType distance) {
    const size_t position = static_cast<size_t>(candidate);
    if (candidate < 0 || position >= stamps.size()) {
      throw std::runtime_error(
          "exact-distance memo candidate is outside local-ID domain");
    }
    values[position] = distance;
    stamps[position] = epoch;
  }

  bool contains(indexType candidate) const {
    distanceType ignored{};
    return lookup(candidate, ignored);
  }

 private:
  std::vector<distanceType> values;
  std::vector<uint32_t> stamps;
  uint32_t epoch = 0;
};

template <typename indexType, typename distanceType>
struct BatchedL2BeamSearchResult {
  std::vector<std::pair<indexType, distanceType>> frontier;
  std::vector<std::pair<indexType, distanceType>> visited;
  size_t distance_computations;
  size_t staged_candidates;
  size_t prefix_rejected_candidates;
  size_t exact_distance_cache_hits;
  size_t incremental_exact_misses;
  size_t exact_distance_completions;
  size_t uncached_prefix_rejections;
  size_t eviction_spill_triggered = 0;
  size_t eviction_spill_exact_records = 0;
  size_t eviction_spill_width_records = 0;
  size_t eviction_spill_cutoff_records = 0;
  size_t eviction_spill_prefix_records = 0;
  size_t eviction_spill_metric_excluded = 0;
  size_t eviction_spill_handoff_records = 0;
  size_t eviction_spill_extra_distance_computations = 0;
  size_t eviction_spill_new_visits = 0;
  size_t q4_exact_rerank_candidates = 0;
  struct Top10StabilitySummary {
    size_t last_change_visit = 0;
    size_t last_change_distance_computations = 0;
    size_t stability_age = 0;
    size_t changed_steps_last16 = 0;
    size_t entries_last16 = 0;
    size_t total_changed_steps = 0;
    size_t total_entries = 0;
    size_t final_size = 0;
  } top10_stability;
};

#if defined(BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC) || \
    defined(BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC)
// Diagnostic-only, zero-distance-computation certificate.  The trace observes
// the unordered membership of the first k frontier IDs after every completed
// expansion.  It never changes candidate generation, beam/cut transitions, or
// expansion order.  The fixed arrays avoid timed-region allocation and the
// diagnostic is compiled out of the retained production target.
template <typename indexType>
class Top10StabilityTrace {
 public:
  static constexpr size_t kWindow = 16;
  static constexpr size_t kMaxTop = 10;
  using Summary =
      typename BatchedL2BeamSearchResult<indexType, float>::
          Top10StabilitySummary;

  template <typename Frontier>
  void reset(const Frontier& frontier, size_t k) {
    if (k == 0 || k > kMaxTop) {
      throw std::runtime_error(
          "top10 stability diagnostic requires k in [1,10]");
    }
    k_ = k;
    previous_size_ = snapshot(frontier, previous_);
    changed_ring_.fill(0);
    entries_ring_.fill(0);
    summary_ = Summary{};
    summary_.final_size = previous_size_;
  }

  template <typename Frontier>
  void observe(const Frontier& frontier, size_t visit,
               size_t distance_computations) {
    std::array<indexType, kMaxTop> current{};
    const size_t current_size = snapshot(frontier, current);
    const size_t entered = count_entered(
        previous_, previous_size_, current, current_size);
    const bool changed =
        current_size != previous_size_ ||
        !std::equal(
            previous_.begin(), previous_.begin() + previous_size_,
            current.begin());

    if (visit == 0) {
      throw std::runtime_error(
          "top10 stability observations require positive visit indices");
    }
    const size_t slot = (visit - 1) % kWindow;
    summary_.changed_steps_last16 -= changed_ring_[slot];
    summary_.entries_last16 -= entries_ring_[slot];
    changed_ring_[slot] = changed ? 1 : 0;
    entries_ring_[slot] = entered;
    summary_.changed_steps_last16 += changed_ring_[slot];
    summary_.entries_last16 += entries_ring_[slot];

    if (changed) {
      summary_.last_change_visit = visit;
      summary_.last_change_distance_computations =
          distance_computations;
      ++summary_.total_changed_steps;
      summary_.total_entries += entered;
    }
    previous_ = current;
    previous_size_ = current_size;
    summary_.final_size = current_size;
  }

  Summary finish(size_t final_visits) {
    if (summary_.last_change_visit > final_visits) {
      throw std::runtime_error(
          "top10 stability last change exceeds final visits");
    }
    summary_.stability_age =
        final_visits - summary_.last_change_visit;
    return summary_;
  }

 private:
  template <typename Frontier>
  size_t snapshot(
      const Frontier& frontier,
      std::array<indexType, kMaxTop>& output) const {
    const size_t size = std::min(k_, frontier.size());
    for (size_t i = 0; i < size; ++i) {
      output[i] = frontier[i].first;
    }
    std::sort(output.begin(), output.begin() + size);
    return size;
  }

  static size_t count_entered(
      const std::array<indexType, kMaxTop>& previous,
      size_t previous_size,
      const std::array<indexType, kMaxTop>& current,
      size_t current_size) {
    size_t left = 0;
    size_t right = 0;
    size_t entered = 0;
    while (right < current_size) {
      while (left < previous_size &&
             previous[left] < current[right]) {
        ++left;
      }
      if (left == previous_size ||
          current[right] < previous[left]) {
        ++entered;
      }
      ++right;
    }
    return entered;
  }

  size_t k_ = 0;
  size_t previous_size_ = 0;
  std::array<indexType, kMaxTop> previous_{};
  std::array<size_t, kWindow> changed_ring_{};
  std::array<size_t, kWindow> entries_ring_{};
  Summary summary_{};
};

static int top10_stability_self_test() {
  using Pair = std::pair<int32_t, float>;
  auto make_frontier =
      [](std::initializer_list<int32_t> ids) {
        std::vector<Pair> output;
        float distance = 1.0f;
        for (int32_t id : ids) {
          output.push_back({id, distance});
          distance += 1.0f;
        }
        return output;
      };

  Top10StabilityTrace<int32_t> trace;
  auto initial =
      make_frontier({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90});
  trace.reset(initial, 10);

  // Same unordered membership with a different distance order is stable.
  auto reordered =
      make_frontier({10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 90});
  trace.observe(reordered, 1, 100);
  auto summary = trace.finish(1);
  if (summary.total_changed_steps != 0 ||
      summary.stability_age != 1) {
    throw std::runtime_error(
        "top10 stability self-test counted a pure reorder");
  }

  // One replacement contributes one entering ID.
  auto replaced_one =
      make_frontier({1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 90});
  trace.observe(replaced_one, 2, 120);
  summary = trace.finish(2);
  if (summary.last_change_visit != 2 ||
      summary.last_change_distance_computations != 120 ||
      summary.total_changed_steps != 1 ||
      summary.total_entries != 1 ||
      summary.changed_steps_last16 != 1 ||
      summary.entries_last16 != 1 ||
      summary.stability_age != 0) {
    throw std::runtime_error(
        "top10 stability self-test replacement accounting failed");
  }

  // After sixteen later stable transitions, the old change leaves the window.
  for (size_t visit = 3; visit <= 18; ++visit) {
    trace.observe(replaced_one, visit, 120 + visit);
  }
  summary = trace.finish(18);
  if (summary.changed_steps_last16 != 0 ||
      summary.entries_last16 != 0 ||
      summary.stability_age != 16) {
    throw std::runtime_error(
        "top10 stability self-test rolling-window eviction failed");
  }

  // Two entering IDs at the newest transition count as one changed step.
  auto replaced_two =
      make_frontier({1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 90});
  trace.observe(replaced_two, 19, 200);
  summary = trace.finish(19);
  if (summary.changed_steps_last16 != 1 ||
      summary.entries_last16 != 2 ||
      summary.total_changed_steps != 2 ||
      summary.total_entries != 3 ||
      summary.stability_age != 0) {
    throw std::runtime_error(
        "top10 stability self-test multi-entry accounting failed");
  }

  // Filling an incomplete top-k is an entering-ID event.
  Top10StabilityTrace<int32_t> short_trace;
  auto short_initial = make_frontier({1});
  auto short_filled = make_frontier({1, 2});
  short_trace.reset(short_initial, 3);
  short_trace.observe(short_filled, 1, 2);
  const auto short_summary = short_trace.finish(1);
  if (short_summary.total_changed_steps != 1 ||
      short_summary.total_entries != 1 ||
      short_summary.final_size != 2) {
    throw std::runtime_error(
        "top10 stability self-test incomplete-top-k accounting failed");
  }

  std::printf("[top10 stability self-test] PASS\n");
  return 0;
}
#endif

// Project-local staged-distance traversal over the public ParlayANN graph and
// point-range interfaces. The upstream search implementation is not embedded
// here. The first stage evaluates 128 dimensions; candidates whose exact
// nonnegative prefix reaches the current cutoff are rejected, and the
// remaining candidates are completed in a second batch.
template <typename Point, typename PointRange, typename indexType>
BatchedL2BeamSearchResult<indexType, typename Point::distanceType>
beam_search_batched_l2(
    Point query, Graph<indexType>& graph, PointRange& points,
    const parlay::sequence<indexType>& starting_points, QueryParams& params,
    int prefix_dimensions,
    ExactDistanceMemo<
        indexType, typename Point::distanceType>* exact_distance_memo =
        nullptr,
    int neighbor_offset = 0, int neighbor_stride = 1
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
    ,
    bool eviction_spill_eligible = false,
    bool eviction_spill_force_trigger = false
#endif
#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
    ,
    const std::vector<uint64_t>* shortcut_row_masks = nullptr
#endif
    ) {
  using distanceType = typename Point::distanceType;
  using Pair = std::pair<indexType, distanceType>;
  auto less = [](const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second && left.first < right.first);
  };
  if (neighbor_offset < 0 || neighbor_stride <= 0 ||
      neighbor_offset >= neighbor_stride) {
    throw std::runtime_error(
        "batched-L2 neighbor view requires 0 <= offset < stride");
  }

  const int bits = std::max<int>(
      10, std::ceil(std::log2(params.beamSize * params.beamSize)) - 2);
  std::vector<indexType> hash_filter(size_t{1} << bits, -1);
  auto has_been_seen = [&](indexType candidate) {
    const size_t location =
        parlay::hash64_2(candidate) & (hash_filter.size() - 1);
    if (hash_filter[location] == candidate) return true;
    hash_filter[location] = candidate;
    return false;
  };

  const bool uint8_192 =
      points.dimension() == 192 &&
      points.aligned_dimension() >= 192;
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
  if (!uint8_192 || prefix_dimensions != 64 ||
      exact_distance_memo != nullptr ||
      neighbor_offset != 0 || neighbor_stride != 1) {
    throw std::runtime_error(
        "q4 graph diagnostic requires ordinary uint8/192-D "
        "prefix64 traversal without replay or neighbor views");
  }
#endif
#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
  const int bitplane_high_bits =
      std::getenv("BCI_BITPLANE_HIGH_BITS") != nullptr
          ? std::atoi(std::getenv("BCI_BITPLANE_HIGH_BITS"))
          : 4;
  if (!uint8_192 || prefix_dimensions != 64 ||
      exact_distance_memo != nullptr ||
      neighbor_offset != 0 || neighbor_stride != 1 ||
      bitplane_high_bits < 1 || bitplane_high_bits > 7) {
    throw std::runtime_error(
        "bit-plane lower-bound diagnostic requires ordinary "
        "uint8/192-D prefix64 traversal without replay or neighbor "
        "views and retained bits in [1,7]");
  }
#endif
#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
  const char* exact_engine_mode_raw =
      std::getenv("BCI_EXACT_ENGINE_MODE");
  const std::string exact_engine_mode =
      exact_engine_mode_raw != nullptr &&
              exact_engine_mode_raw[0] != '\0'
          ? exact_engine_mode_raw
          : "fourway_radix";
#endif
  auto search_l2_segment =
      [&](const uint8_t* left,
          const uint8_t* right,
          unsigned dimensions) {
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
        return l2_sq_uint4_high_avx2(
            left, right, dimensions);
#else
        return l2_sq_uint8_avx2(
            left, right, dimensions);
#endif
      };
  if (exact_distance_memo != nullptr && !uint8_192) {
    throw std::runtime_error(
        "cached replay diagnostic requires uint8/192-D point storage");
  }
  size_t distance_computations = 0;
  size_t staged_candidates = 0;
  size_t prefix_rejected_candidates = 0;
  size_t exact_distance_cache_hits = 0;
  size_t incremental_exact_misses = 0;
  size_t exact_distance_completions = 0;
  size_t uncached_prefix_rejections = 0;
  auto exact_distance = [&](indexType candidate) {
    ++distance_computations;
    if (exact_distance_memo != nullptr) {
      distanceType cached_distance{};
      if (exact_distance_memo->lookup(
              candidate, cached_distance)) {
        ++exact_distance_cache_hits;
        return cached_distance;
      }
    }
    const distanceType distance =
        exact_distance_memo != nullptr && uint8_192
            ? search_l2_segment(
                  points[candidate].get(), query.get(), 192)
            : points[candidate].distance(query);
    if (exact_distance_memo != nullptr) {
      exact_distance_memo->insert(candidate, distance);
      ++incremental_exact_misses;
      ++exact_distance_completions;
    }
    return distance;
  };

#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  constexpr size_t kEvictionSpillAddedWidth = 26;
  const size_t workspace_beam_size =
      static_cast<size_t>(params.beamSize) +
      kEvictionSpillAddedWidth;
#else
  const size_t workspace_beam_size =
      static_cast<size_t>(params.beamSize);
#endif
  std::vector<Pair> frontier;
  frontier.reserve(workspace_beam_size);
  for (indexType start : starting_points) {
    frontier.push_back({start, exact_distance(start)});
  }
  std::sort(frontier.begin(), frontier.end(), less);

  std::vector<Pair> unvisited_frontier(workspace_beam_size);
  unvisited_frontier[0] = frontier[0];
  std::vector<Pair> visited;
  visited.reserve(2 * params.beamSize);
  std::vector<Pair> new_frontier(
      workspace_beam_size + graph.max_degree());
  std::vector<Pair> candidates;
  candidates.reserve(graph.max_degree());
  std::vector<Pair> prefix_survivors;
  prefix_survivors.reserve(graph.max_degree());
  std::vector<indexType> keep;
  keep.reserve(graph.max_degree());
  std::vector<indexType> uncached_keep;
  uncached_keep.reserve(graph.max_degree());

  int remaining = 1;
  int visited_count = 0;
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  constexpr size_t kEvictionSpillPrefixCompletionCap = 32;
  constexpr int kEvictionSpillVisitCap = 8;
  struct PrefixSpillRecord {
    indexType id;
    distanceType prefix;
    distanceType metric_threshold;
  };
  std::vector<Pair> eviction_spill_exact;
  std::vector<Pair> eviction_spill_width;
  std::vector<Pair> eviction_spill_cutoff;
  std::vector<PrefixSpillRecord> eviction_spill_prefix;
  eviction_spill_exact.reserve(graph.max_degree());
  eviction_spill_width.reserve(graph.max_degree());
  eviction_spill_cutoff.reserve(graph.max_degree());
  eviction_spill_prefix.reserve(graph.max_degree());
  bool eviction_spill_continuation = false;
  bool eviction_spill_triggered = false;
  size_t eviction_spill_metric_excluded = 0;
  size_t eviction_spill_handoff_records = 0;
  size_t eviction_spill_extra_distance_computations = 0;
  int eviction_spill_primary_visits = 0;
  int active_visit_limit = params.limit;
  size_t active_beam_size =
      static_cast<size_t>(params.beamSize);
  std::vector<Pair> eviction_spill_primary_top;
  auto consider_eviction_spill =
      [&](std::vector<Pair>& destination,
          indexType id, distanceType distance) {
        if (!eviction_spill_eligible ||
            eviction_spill_continuation) {
          return;
        }
        destination.push_back({id, distance});
      };
#else
  const int active_visit_limit = params.limit;
  const size_t active_beam_size =
      static_cast<size_t>(params.beamSize);
#endif
#if defined(BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC) || \
    defined(BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC)
  if (params.k != 10) {
    throw std::runtime_error(
        "top10 stability diagnostic requires k=10");
  }
  Top10StabilityTrace<indexType> top10_trace;
  top10_trace.reset(frontier, static_cast<size_t>(params.k));
  typename BatchedL2BeamSearchResult<
      indexType, distanceType>::Top10StabilitySummary
      primary_top10_summary{};
#endif
  while (remaining > 0 &&
         visited_count < active_visit_limit) {
    const Pair current = unvisited_frontier[0];
    graph[current.first].prefetch();
    visited.insert(
        std::upper_bound(
            visited.begin(), visited.end(), current, less),
        current);
    ++visited_count;

    candidates.clear();
    prefix_survivors.clear();
    keep.clear();
    uncached_keep.clear();
    const long neighbor_count = std::min<long>(
        graph[current.first].size(), params.degree_limit);
    for (long neighbor_index = neighbor_offset;
         neighbor_index < neighbor_count;
         neighbor_index += neighbor_stride) {
#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
      if (shortcut_row_masks != nullptr) {
        if (current.first < 0 ||
            static_cast<size_t>(current.first) >=
                shortcut_row_masks->size() ||
            neighbor_index >= 64 ||
            (((*shortcut_row_masks)[
                   static_cast<size_t>(current.first)] >>
               static_cast<unsigned>(neighbor_index)) &
              uint64_t{1}) == 0) {
          continue;
        }
      }
#endif
      const indexType candidate =
          graph[current.first][neighbor_index];
      if (candidate == query.id() || has_been_seen(candidate)) continue;
      keep.push_back(candidate);
      if (exact_distance_memo != nullptr) continue;
      if (uint8_192 && prefix_dimensions > 0) {
        auto candidate_point = points[candidate];
        __builtin_prefetch(candidate_point.get(), 0, 1);
        if (prefix_dimensions == 128) {
          __builtin_prefetch(candidate_point.get() + 64, 0, 1);
        }
      } else {
        auto candidate_point = points[candidate];
        candidate_point.prefetch();
      }
    }

    const distanceType cutoff =
        frontier.size() < active_beam_size
            ? static_cast<distanceType>(
                  std::numeric_limits<int>::max())
            : frontier.back().second;
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
    const distanceType spill_metric_threshold =
        params.k > 0 &&
                frontier.size() >
                    static_cast<size_t>(params.k) &&
                points[0].is_metric()
            ? static_cast<distanceType>(
                  params.cut *
                  frontier[static_cast<size_t>(params.k)]
                      .second)
            : static_cast<distanceType>(
                  std::numeric_limits<int>::max());
#endif
    if (exact_distance_memo != nullptr &&
        uint8_192 && prefix_dimensions > 0) {
      const uint8_t* query_values = query.get();
      for (indexType candidate : keep) {
        ++distance_computations;
        distanceType cached_distance{};
        if (exact_distance_memo->lookup(
                candidate, cached_distance)) {
          ++exact_distance_cache_hits;
          if (cached_distance < cutoff) {
            candidates.push_back(
                {candidate, cached_distance});
          }
          continue;
        }
        ++incremental_exact_misses;
        uncached_keep.push_back(candidate);
        const uint8_t* candidate_values =
            points[candidate].get();
        __builtin_prefetch(candidate_values, 0, 1);
        if (prefix_dimensions == 128) {
          __builtin_prefetch(
              candidate_values + 64, 0, 1);
        }
      }
      for (indexType candidate : uncached_keep) {
        const uint8_t* candidate_values =
            points[candidate].get();
        const distanceType prefix =
            search_l2_segment(
                candidate_values, query_values,
                static_cast<unsigned>(prefix_dimensions));
        ++staged_candidates;
        if (prefix >= cutoff) {
          ++prefix_rejected_candidates;
          ++uncached_prefix_rejections;
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          if (eviction_spill_eligible &&
              !eviction_spill_continuation) {
            eviction_spill_prefix.push_back(
                {candidate, prefix,
                 spill_metric_threshold});
          }
#endif
          continue;
        }
        prefix_survivors.push_back({candidate, prefix});
        for (int offset = prefix_dimensions;
             offset < 192; offset += 64) {
          __builtin_prefetch(candidate_values + offset, 0, 1);
        }
      }
      for (const Pair& survivor : prefix_survivors) {
        const uint8_t* candidate_values =
            points[survivor.first].get();
        const distanceType distance =
            survivor.second +
            search_l2_segment(
                candidate_values + prefix_dimensions,
                query_values + prefix_dimensions,
                static_cast<unsigned>(
                    192 - prefix_dimensions));
        exact_distance_memo->insert(
            survivor.first, distance);
        ++exact_distance_completions;
        if (distance >= cutoff) {
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          if (distance <= spill_metric_threshold) {
            consider_eviction_spill(
                eviction_spill_cutoff,
                survivor.first, distance);
          }
#endif
          continue;
        }
        candidates.push_back({survivor.first, distance});
      }
    } else if (exact_distance_memo != nullptr) {
      for (indexType candidate : keep) {
        const distanceType distance =
            exact_distance(candidate);
        if (distance >= cutoff) {
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          if (distance <= spill_metric_threshold) {
            consider_eviction_spill(
                eviction_spill_cutoff,
                candidate, distance);
          }
#endif
          continue;
        }
        candidates.push_back({candidate, distance});
      }
    } else if (uint8_192 && prefix_dimensions > 0) {
      const uint8_t* query_values = query.get();
#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
      if (exact_engine_mode == "fourway_sorted") {
        size_t candidate_index = 0;
        for (; candidate_index + 4 <= keep.size();
             candidate_index += 4) {
          const uint8_t* pointers[4] = {
              points[keep[candidate_index]].get(),
              points[keep[candidate_index + 1]].get(),
              points[keep[candidate_index + 2]].get(),
              points[keep[candidate_index + 3]].get()};
          float distances[4]{};
          l2_sq_uint8_four_192_avx2(
              pointers, query_values, distances);
          distance_computations += 4;
          staged_candidates += 4;
          for (size_t lane = 0; lane < 4; ++lane) {
            if (distances[lane] < cutoff) {
              candidates.push_back(
                  {keep[candidate_index + lane],
                   static_cast<distanceType>(
                       distances[lane])});
            }
          }
        }
        for (; candidate_index < keep.size();
             ++candidate_index) {
          const distanceType distance =
              l2_sq_uint8_avx2(
                  points[keep[candidate_index]].get(),
                  query_values, 192);
          ++distance_computations;
          ++staged_candidates;
          if (distance < cutoff) {
            candidates.push_back(
                {keep[candidate_index], distance});
          }
        }
      } else {
#endif
      for (indexType candidate : keep) {
        const uint8_t* candidate_values =
            points[candidate].get();
#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
        const distanceType prefix =
            l2_sq_uint8_high_bits_lower_bound_avx2(
                candidate_values, query_values, 192,
                bitplane_high_bits);
#else
        const distanceType prefix =
            search_l2_segment(
                candidate_values, query_values,
                static_cast<unsigned>(prefix_dimensions));
#endif
        ++distance_computations;
        ++staged_candidates;
        if (prefix >= cutoff) {
          ++prefix_rejected_candidates;
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          if (eviction_spill_eligible &&
              !eviction_spill_continuation) {
            eviction_spill_prefix.push_back(
                {candidate, prefix,
                 spill_metric_threshold});
          }
#endif
          continue;
        }
        prefix_survivors.push_back({candidate, prefix});
        for (int offset = prefix_dimensions;
             offset < 192; offset += 64) {
          __builtin_prefetch(candidate_values + offset, 0, 1);
        }
      }
      for (const Pair& survivor : prefix_survivors) {
        const uint8_t* candidate_values =
            points[survivor.first].get();
#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
        const distanceType distance =
            l2_sq_uint8_avx2(
                candidate_values, query_values, 192);
#else
        const distanceType distance =
            survivor.second +
            search_l2_segment(
                candidate_values + prefix_dimensions,
                query_values + prefix_dimensions,
                static_cast<unsigned>(
                    192 - prefix_dimensions));
#endif
        if (distance >= cutoff) {
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          if (distance <= spill_metric_threshold) {
            consider_eviction_spill(
                eviction_spill_cutoff,
                survivor.first, distance);
          }
#endif
          continue;
        }
        candidates.push_back({survivor.first, distance});
      }
#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
      }
#endif
    } else {
      for (indexType candidate : keep) {
        const distanceType distance = uint8_192
            ? search_l2_segment(
                  points[candidate].get(), query.get(), 192)
            : points[candidate].distance(query);
        ++distance_computations;
        if (distance >= cutoff) {
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          if (distance <= spill_metric_threshold) {
            consider_eviction_spill(
                eviction_spill_cutoff,
                candidate, distance);
          }
#endif
          continue;
        }
        candidates.push_back({candidate, distance});
      }
    }
    std::sort(candidates.begin(), candidates.end(), less);

    const size_t union_frontier_size = static_cast<size_t>(
        std::set_union(
            frontier.begin(), frontier.end(),
            candidates.begin(), candidates.end(),
            new_frontier.begin(), less) -
        new_frontier.begin());
    const size_t width_frontier_size = std::min<size_t>(
        active_beam_size, union_frontier_size);
    auto metric_frontier_size = [&](size_t width) {
      if (params.k <= 0 ||
          width <= static_cast<size_t>(params.k) ||
          !points[0].is_metric()) {
        return width;
      }
      return static_cast<size_t>(
          std::upper_bound(
              new_frontier.begin(),
              new_frontier.begin() + width,
              Pair{
                  0,
                  static_cast<distanceType>(
                      params.cut *
                      new_frontier[params.k].second)},
              less) -
          new_frontier.begin());
    };
    size_t new_frontier_size =
        metric_frontier_size(width_frontier_size);
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
    if (eviction_spill_eligible &&
        !eviction_spill_continuation) {
      const size_t counterfactual_width = std::min<size_t>(
          static_cast<size_t>(params.beamSize) +
              kEvictionSpillAddedWidth,
          union_frontier_size);
      const size_t counterfactual_metric =
          metric_frontier_size(counterfactual_width);
      // Capture exactly the same-union interval that b90 retains and b64
      // evicts.  Anything past the counterfactual b90 metric cut is not
      // a width spill and is excluded.
      eviction_spill_metric_excluded +=
          counterfactual_width - counterfactual_metric;
      for (size_t lost = new_frontier_size;
           lost < counterfactual_metric; ++lost) {
        consider_eviction_spill(
            eviction_spill_width,
            new_frontier[lost].first,
            new_frontier[lost].second);
      }
    }
#endif
    frontier.assign(
        new_frontier.begin(),
        new_frontier.begin() + new_frontier_size);
#if defined(BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC) || \
    defined(BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC)
    if (
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
        !eviction_spill_continuation
#else
        true
#endif
    ) {
      top10_trace.observe(
          frontier, static_cast<size_t>(visited_count),
          distance_computations);
    }
#endif
    remaining = static_cast<int>(
        std::set_difference(
            frontier.begin(), frontier.end(),
            visited.begin(), visited.end(),
        unvisited_frontier.begin(), less) -
        unvisited_frontier.begin());

#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
    if (remaining == 0 && !eviction_spill_continuation) {
      primary_top10_summary =
          top10_trace.finish(
              static_cast<size_t>(visited_count));
      const double boundary_ratio =
          frontier.size() > static_cast<size_t>(params.k) &&
                  frontier[static_cast<size_t>(params.k - 1)]
                          .second > 0
              ? static_cast<double>(
                    frontier[static_cast<size_t>(params.k)]
                        .second) /
                    static_cast<double>(
                        frontier[
                            static_cast<size_t>(params.k - 1)]
                                .second)
              : std::numeric_limits<double>::infinity();
      eviction_spill_triggered =
          eviction_spill_eligible &&
          ((eviction_spill_force_trigger &&
            frontier.size() ==
                static_cast<size_t>(params.beamSize)) ||
           (boundary_ratio <= 1.006 &&
            primary_top10_summary.stability_age <= 46 &&
            distance_computations >= 2800));
      if (eviction_spill_triggered) {
        eviction_spill_primary_visits = visited_count;
        eviction_spill_primary_top.assign(
            frontier.begin(),
            frontier.begin() +
                std::min<size_t>(
                    static_cast<size_t>(params.k),
                    frontier.size()));

        auto already_visited = [&](indexType id) {
          return std::any_of(
              visited.begin(), visited.end(),
              [&](const Pair& item) {
                return item.first == id;
              });
        };
        auto dedupe_by_id_keep_best =
            [&](std::vector<Pair>& items) {
              std::sort(
                  items.begin(), items.end(),
                  [](const Pair& left, const Pair& right) {
                    return left.first < right.first ||
                           (left.first == right.first &&
                            left.second < right.second);
                  });
              items.erase(
                  std::unique(
                      items.begin(), items.end(),
                      [](const Pair& left, const Pair& right) {
                        return left.first == right.first;
                      }),
                  items.end());
            };

        // Merge exact width/cutoff victims, then remove candidates that
        // primary b64 eventually visited.  Retaining the complete raw
        // candidate set until this point avoids discarding the best
        // still-unvisited spill behind candidates visited later.
        eviction_spill_exact = eviction_spill_width;
        eviction_spill_exact.insert(
            eviction_spill_exact.end(),
            eviction_spill_cutoff.begin(),
            eviction_spill_cutoff.end());
        dedupe_by_id_keep_best(eviction_spill_exact);
        eviction_spill_exact.erase(
            std::remove_if(
                eviction_spill_exact.begin(),
                eviction_spill_exact.end(),
                [&](const Pair& item) {
                  return already_visited(item.first);
                }),
            eviction_spill_exact.end());

        // Prefix-only victims are deduplicated and lower-bound ranked
        // before at most 32 exact completions.  This bounds continuation
        // work by 8*degree + 32 logical distance completions.
        std::sort(
            eviction_spill_prefix.begin(),
            eviction_spill_prefix.end(),
            [](const PrefixSpillRecord& left,
               const PrefixSpillRecord& right) {
              return left.id < right.id ||
                     (left.id == right.id &&
                      left.prefix < right.prefix);
            });
        {
          std::vector<PrefixSpillRecord> deduped;
          deduped.reserve(eviction_spill_prefix.size());
          for (const PrefixSpillRecord& item :
               eviction_spill_prefix) {
            if (deduped.empty() ||
                deduped.back().id != item.id) {
              deduped.push_back(item);
            } else {
              deduped.back().prefix =
                  std::min(deduped.back().prefix, item.prefix);
              deduped.back().metric_threshold =
                  std::max(
                      deduped.back().metric_threshold,
                      item.metric_threshold);
            }
          }
          eviction_spill_prefix = std::move(deduped);
        }
        std::sort(
            eviction_spill_prefix.begin(),
            eviction_spill_prefix.end(),
            [](const PrefixSpillRecord& left,
               const PrefixSpillRecord& right) {
              return left.prefix < right.prefix ||
                     (left.prefix == right.prefix &&
                      left.id < right.id);
            });
        size_t prefix_completed = 0;
        for (const PrefixSpillRecord& prefix_item :
             eviction_spill_prefix) {
          if (prefix_completed >=
              kEvictionSpillPrefixCompletionCap) {
            break;
          }
          if (already_visited(prefix_item.id) ||
              std::any_of(
                  eviction_spill_exact.begin(),
                  eviction_spill_exact.end(),
                  [&](const Pair& item) {
                    return item.first ==
                           prefix_item.id;
                  })) {
            continue;
          }
          const uint8_t* candidate_values =
              points[prefix_item.id].get();
          const distanceType distance =
              prefix_item.prefix +
              l2_sq_uint8_avx2(
                  candidate_values + prefix_dimensions,
                  query.get() + prefix_dimensions,
                  static_cast<unsigned>(
                      192 - prefix_dimensions));
          ++distance_computations;
          ++eviction_spill_extra_distance_computations;
          ++prefix_completed;
          if (distance >
              prefix_item.metric_threshold) {
            continue;
          }
          consider_eviction_spill(
              eviction_spill_exact,
              prefix_item.id, distance);
        }
        dedupe_by_id_keep_best(eviction_spill_exact);
        std::sort(
            eviction_spill_exact.begin(),
            eviction_spill_exact.end(), less);

        // Preserve the full terminal b64 frontier and extend only its
        // width to b90.  The same hash filter, visited set, distances,
        // and expansion order continue in-place for at most eight new
        // unique visits.
        const size_t handoff_capacity =
            static_cast<size_t>(params.beamSize) +
            kEvictionSpillAddedWidth;
        std::vector<Pair> handoff = frontier;
        handoff.reserve(handoff_capacity);
        for (const Pair& spill : eviction_spill_exact) {
          if (handoff.size() >= handoff_capacity) break;
          if (already_visited(spill.first)) continue;
          const bool already_in_frontier = std::any_of(
              handoff.begin(), handoff.end(),
              [&](const Pair& item) {
                return item.first == spill.first;
              });
          if (already_in_frontier) continue;
          handoff.push_back(spill);
          ++eviction_spill_handoff_records;
        }
        std::sort(handoff.begin(), handoff.end(), less);
        frontier = std::move(handoff);
        eviction_spill_continuation = true;
        active_beam_size = handoff_capacity;
        active_visit_limit = std::min<int>(
            static_cast<int>(params.limit),
            visited_count + kEvictionSpillVisitCap);
        remaining = static_cast<int>(
            std::set_difference(
                frontier.begin(), frontier.end(),
                visited.begin(), visited.end(),
                unvisited_frontier.begin(), less) -
            unvisited_frontier.begin());
      }
    }
#endif
  }
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  if (eviction_spill_triggered) {
    frontier.insert(
        frontier.end(),
        eviction_spill_primary_top.begin(),
        eviction_spill_primary_top.end());
    std::sort(
        frontier.begin(), frontier.end(),
        [](const Pair& left, const Pair& right) {
          return left.first < right.first ||
                 (left.first == right.first &&
                  left.second < right.second);
        });
    frontier.erase(
        std::unique(
            frontier.begin(), frontier.end(),
            [](const Pair& left, const Pair& right) {
              return left.first == right.first;
            }),
        frontier.end());
    std::sort(frontier.begin(), frontier.end(), less);
  }
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
  // Navigation used only the high nibbles.  Exact-rerank the union of the
  // terminal frontier and expanded nodes so the returned order/distances are
  // canonical uint8 L2; the quality gate therefore measures candidate
  // coverage rather than approximate-distance ordering.
  std::vector<Pair> q4_rerank_pool = frontier;
  q4_rerank_pool.insert(
      q4_rerank_pool.end(), visited.begin(), visited.end());
  std::sort(
      q4_rerank_pool.begin(), q4_rerank_pool.end(),
      [](const Pair& left, const Pair& right) {
        return left.first < right.first;
      });
  q4_rerank_pool.erase(
      std::unique(
          q4_rerank_pool.begin(), q4_rerank_pool.end(),
          [](const Pair& left, const Pair& right) {
            return left.first == right.first;
          }),
      q4_rerank_pool.end());
  const size_t q4_exact_rerank_candidates =
      q4_rerank_pool.size();
  for (Pair& item : q4_rerank_pool) {
    item.second = l2_sq_uint8_avx2(
        points[item.first].get(), query.get(), 192);
  }
  std::sort(q4_rerank_pool.begin(), q4_rerank_pool.end(), less);
  if (q4_rerank_pool.size() >
      static_cast<size_t>(params.beamSize)) {
    q4_rerank_pool.resize(
        static_cast<size_t>(params.beamSize));
  }
  frontier = std::move(q4_rerank_pool);
#endif
  BatchedL2BeamSearchResult<indexType, distanceType> result{
      std::move(frontier), std::move(visited),
      distance_computations, staged_candidates,
      prefix_rejected_candidates, exact_distance_cache_hits,
      incremental_exact_misses, exact_distance_completions,
      uncached_prefix_rejections};
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  result.eviction_spill_triggered =
      eviction_spill_triggered ? 1 : 0;
  result.eviction_spill_exact_records =
      eviction_spill_exact.size();
  result.eviction_spill_width_records =
      eviction_spill_width.size();
  result.eviction_spill_cutoff_records =
      eviction_spill_cutoff.size();
  result.eviction_spill_prefix_records =
      eviction_spill_prefix.size();
  result.eviction_spill_metric_excluded =
      eviction_spill_metric_excluded;
  result.eviction_spill_handoff_records =
      eviction_spill_handoff_records;
  result.eviction_spill_extra_distance_computations =
      eviction_spill_extra_distance_computations;
  result.eviction_spill_new_visits =
      eviction_spill_triggered
          ? static_cast<size_t>(
                visited_count - eviction_spill_primary_visits)
          : 0;
  result.top10_stability = primary_top10_summary;
#elif defined(BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC)
  result.top10_stability =
      top10_trace.finish(static_cast<size_t>(visited_count));
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
  result.q4_exact_rerank_candidates =
      q4_exact_rerank_candidates;
#endif
  return result;
}

#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
// Exact beam semantics with two independent structural changes: four random
// vector streams are scored in one AVX2 kernel, and the active beam is kept in
// high-distance radix buckets while a lazy min-heap identifies the next
// unvisited item.  This removes the reference implementation's full
// frontier/visited set_difference after every expansion.  All ranking keys
// remain the exact integer uint8 L2 plus local ID.
template <typename Point, typename PointRange, typename indexType>
BatchedL2BeamSearchResult<indexType, typename Point::distanceType>
beam_search_exact_radix_l2(
    Point query, Graph<indexType>& graph, PointRange& points,
    const parlay::sequence<indexType>& starting_points,
    QueryParams& params, bool use_fourway_distance) {
  using distanceType = typename Point::distanceType;
  using Pair = std::pair<indexType, distanceType>;
  auto less = [](const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second && left.first < right.first);
  };
  auto heap_greater = [&](const Pair& left, const Pair& right) {
    return less(right, left);
  };
  if (points.dimension() != 192 ||
      points.aligned_dimension() < 192 ||
      starting_points.empty()) {
    throw std::runtime_error(
        "exact radix traversal requires uint8/192-D points and a start");
  }

  // A 12-bit high radix leaves only a small sorted vector inside each active
  // bucket.  Empty high buckets are absent, so query setup is proportional to
  // the beam rather than the 24-bit distance universe.
  std::map<uint16_t, std::vector<Pair>> radix;
  std::unordered_map<indexType, distanceType> active;
  active.reserve(
      static_cast<size_t>(params.beamSize) * 2 + 64);
  auto radix_key = [](distanceType distance) {
    return static_cast<uint16_t>(
        static_cast<uint32_t>(distance) >> 12);
  };
  auto active_contains = [&](const Pair& item) {
    const auto found = active.find(item.first);
    return found != active.end() &&
           found->second == item.second;
  };
  auto active_insert = [&](const Pair& item) {
    if (!active.emplace(item.first, item.second).second) {
      return false;
    }
    auto& bucket = radix[radix_key(item.second)];
    bucket.insert(
        std::lower_bound(
            bucket.begin(), bucket.end(), item, less),
        item);
    return true;
  };
  auto active_max = [&]() -> Pair {
    if (radix.empty()) {
      throw std::runtime_error("exact radix active beam is empty");
    }
    return radix.rbegin()->second.back();
  };
  auto active_pop_max = [&]() {
    auto bucket_it = std::prev(radix.end());
    const Pair removed = bucket_it->second.back();
    bucket_it->second.pop_back();
    active.erase(removed.first);
    if (bucket_it->second.empty()) {
      radix.erase(bucket_it);
    }
  };
  auto active_kth = [&](size_t rank) -> Pair {
    for (const auto& bucket : radix) {
      if (rank < bucket.second.size()) {
        return bucket.second[rank];
      }
      rank -= bucket.second.size();
    }
    throw std::runtime_error(
        "exact radix rank exceeds active beam");
  };
  auto active_materialize = [&]() {
    std::vector<Pair> output;
    output.reserve(active.size());
    for (const auto& bucket : radix) {
      output.insert(
          output.end(),
          bucket.second.begin(), bucket.second.end());
    }
    return output;
  };

  std::priority_queue<
      Pair, std::vector<Pair>, decltype(heap_greater)>
      unvisited(heap_greater);
  std::unordered_set<indexType> visited_ids;
  visited_ids.reserve(
      static_cast<size_t>(params.beamSize) * 2 + 64);
  std::vector<Pair> visited;
  visited.reserve(
      static_cast<size_t>(params.beamSize) * 2);

  const int hash_bits = std::max<int>(
      10, std::ceil(std::log2(
              params.beamSize * params.beamSize)) -
              2);
  std::vector<indexType> hash_filter(
      size_t{1} << hash_bits, -1);
  auto has_been_seen = [&](indexType candidate) {
    const size_t location =
        parlay::hash64_2(candidate) &
        (hash_filter.size() - 1);
    if (hash_filter[location] == candidate) return true;
    hash_filter[location] = candidate;
    return false;
  };

  size_t distance_computations = 0;
  for (indexType start : starting_points) {
    const Pair item{
        start, points[start].distance(query)};
    ++distance_computations;
    if (active_insert(item)) {
      unvisited.push(item);
    }
  }

  std::vector<indexType> keep;
  keep.reserve(graph.max_degree());
  std::vector<Pair> candidates;
  candidates.reserve(graph.max_degree());
  int visited_count = 0;
  while (visited_count < params.limit) {
    while (!unvisited.empty() &&
           (!active_contains(unvisited.top()) ||
            visited_ids.count(unvisited.top().first) != 0)) {
      unvisited.pop();
    }
    if (unvisited.empty()) break;
    const Pair current = unvisited.top();
    unvisited.pop();
    visited_ids.insert(current.first);
    visited.push_back(current);
    ++visited_count;
    graph[current.first].prefetch();

    keep.clear();
    candidates.clear();
    const long neighbor_count = std::min<long>(
        graph[current.first].size(), params.degree_limit);
    for (long neighbor_index = 0;
         neighbor_index < neighbor_count;
         ++neighbor_index) {
      const indexType candidate =
          graph[current.first][neighbor_index];
      if (candidate == query.id() ||
          has_been_seen(candidate)) {
        continue;
      }
      keep.push_back(candidate);
      const uint8_t* values = points[candidate].get();
      __builtin_prefetch(values, 0, 1);
      __builtin_prefetch(values + 64, 0, 1);
      __builtin_prefetch(values + 128, 0, 1);
    }

    const distanceType cutoff =
        active.size() <
                static_cast<size_t>(params.beamSize)
            ? static_cast<distanceType>(
                  std::numeric_limits<int>::max())
            : active_max().second;
    size_t candidate_index = 0;
    if (use_fourway_distance) {
      for (; candidate_index + 4 <= keep.size();
           candidate_index += 4) {
        const uint8_t* pointers[4] = {
            points[keep[candidate_index]].get(),
            points[keep[candidate_index + 1]].get(),
            points[keep[candidate_index + 2]].get(),
            points[keep[candidate_index + 3]].get()};
        float distances[4]{};
        l2_sq_uint8_four_192_avx2(
            pointers, query.get(), distances);
        distance_computations += 4;
        for (size_t lane = 0; lane < 4; ++lane) {
          if (distances[lane] < cutoff) {
            candidates.push_back(
                {keep[candidate_index + lane],
                 static_cast<distanceType>(distances[lane])});
          }
        }
      }
    }
    for (; candidate_index < keep.size();
         ++candidate_index) {
      const distanceType distance =
          l2_sq_uint8_avx2(
              points[keep[candidate_index]].get(),
              query.get(), 192);
      ++distance_computations;
      if (distance < cutoff) {
        candidates.push_back(
            {keep[candidate_index], distance});
      }
    }
    std::sort(candidates.begin(), candidates.end(), less);
    for (const Pair& candidate : candidates) {
      if (active_insert(candidate)) {
        unvisited.push(candidate);
      }
    }
    while (active.size() >
           static_cast<size_t>(params.beamSize)) {
      active_pop_max();
    }
    if (params.k > 0 &&
        active.size() > static_cast<size_t>(params.k) &&
        points[0].is_metric()) {
      const Pair kth = active_kth(
          static_cast<size_t>(params.k));
      const Pair boundary{
          0,
          static_cast<distanceType>(
              params.cut * kth.second)};
      while (!radix.empty() &&
             less(boundary, active_max())) {
        active_pop_max();
      }
    }
  }

  std::sort(visited.begin(), visited.end(), less);
  return BatchedL2BeamSearchResult<
      indexType, distanceType>{
      active_materialize(), std::move(visited),
      distance_computations,
      distance_computations - starting_points.size(),
      0, 0, 0, 0, 0};
}
#endif

#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
struct EvictionSpillToyPointRange {
  std::array<std::array<uint8_t, 4>, 66> values{};

  EvictionSpillToyPointRange() {
    values[0] = {{0, 0, 0, 0}};
    for (size_t i = 1; i <= 64; ++i) {
      values[i] = {{2, 2, 2, 2}};
    }
    values[65] = {{0, 0, 0, 1}};
  }

  size_t size() const { return values.size(); }
  long dimension() const { return 4; }
  long aligned_dimension() const { return 4; }
  PointT operator[](long i) {
    return PointT(
        values.at(static_cast<size_t>(i)).data(), 4, 4, i);
  }
};

static int eviction_spill_self_test() {
  EvictionSpillToyPointRange points;
  GraphI graph(64, points.size());
  parlay::sequence<Indx> root_neighbors;
  for (Indx id = 1; id <= 64; ++id) {
    root_neighbors.push_back(id);
  }
  graph[0].update_neighbors(root_neighbors);
  graph[64].update_neighbors(parlay::sequence<Indx>{65});
  const std::array<uint8_t, 4> query_values{{0, 0, 0, 0}};
  PointT query(query_values.data(), 4, 4, -1);
  QueryParams params(
      /*k=*/10, /*beam=*/64, /*cut=*/1.35,
      /*limit=*/100000, /*degree_limit=*/64);
  QueryParams wider_params(
      /*k=*/10, /*beam=*/65, /*cut=*/1.35,
      /*limit=*/100000, /*degree_limit=*/64);

  auto control = beam_search_batched_l2<
      PointT, EvictionSpillToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 0, 1, false, false);
  auto zero_trigger = beam_search_batched_l2<
      PointT, EvictionSpillToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 0, 1, true, false);
  auto treatment = beam_search_batched_l2<
      PointT, EvictionSpillToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 0, 1, true, true);
  auto width_oracle = beam_search_batched_l2<
      PointT, EvictionSpillToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          wider_params, 0, nullptr, 0, 1, false, false);

  auto has_id = [](const auto& frontier, Indx id) {
    return std::any_of(
        frontier.begin(), frontier.end(),
        [&](const auto& item) {
          return item.first == id;
        });
  };
  auto same_pairs = [](const auto& left, const auto& right) {
    return left == right;
  };
  if (!same_pairs(control.frontier, zero_trigger.frontier) ||
      !same_pairs(control.visited, zero_trigger.visited) ||
      control.distance_computations !=
          zero_trigger.distance_computations ||
      control.staged_candidates !=
          zero_trigger.staged_candidates ||
      control.prefix_rejected_candidates !=
          zero_trigger.prefix_rejected_candidates ||
      zero_trigger.eviction_spill_triggered != 0) {
    throw std::runtime_error(
        "eviction-spill zero-trigger path changed b64 semantics");
  }
  if (has_id(control.frontier, 65) ||
      !has_id(width_oracle.frontier, 65) ||
      !has_id(treatment.frontier, 65) ||
      treatment.eviction_spill_triggered != 1 ||
      treatment.eviction_spill_new_visits == 0 ||
      treatment.eviction_spill_new_visits > 8) {
    throw std::runtime_error(
        "eviction-spill toy failed common-spill-common repair/cap");
  }
  size_t gateway_visits = 0;
  for (const auto& item : treatment.visited) {
    gateway_visits += item.first == 64;
  }
  if (gateway_visits != 1) {
    throw std::runtime_error(
        "eviction-spill toy re-expanded or skipped its gateway");
  }
  std::printf(
      "[eviction spill self-test] PASS "
      "zero_trigger_exact=1 b65_oracle_target=65 repaired_target=65 "
      "gateway_visits=%zu new_visits=%zu cap=8\n",
      gateway_visits,
      treatment.eviction_spill_new_visits);
  return 0;
}
#endif

#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
// Merge two independently executed search states under the same exact order
// used by the YFCC ground-truth generator: (distance, global ID).  The primary
// frontier is never exposed to the rescue branch's fixed beam, so rescue
// candidates cannot evict primary candidates before this exact union.
template <typename indexType, typename distanceType, typename Subset>
BatchedL2BeamSearchResult<indexType, distanceType>
merge_r96_degree4_sidecar_results(
    BatchedL2BeamSearchResult<indexType, distanceType> primary,
    BatchedL2BeamSearchResult<indexType, distanceType> rescue,
    const Subset& local_to_global) {
  using Pair = std::pair<indexType, distanceType>;
  const auto by_local_id = [](const Pair& left, const Pair& right) {
    return left.first < right.first ||
           (left.first == right.first && left.second < right.second);
  };
  const auto by_exact_global = [&](const Pair& left, const Pair& right) {
    if (left.second != right.second) {
      return left.second < right.second;
    }
    return local_to_global[static_cast<size_t>(left.first)] <
           local_to_global[static_cast<size_t>(right.first)];
  };
  primary.frontier.insert(
      primary.frontier.end(),
      rescue.frontier.begin(), rescue.frontier.end());
  std::sort(
      primary.frontier.begin(), primary.frontier.end(), by_local_id);
  primary.frontier.erase(
      std::unique(
          primary.frontier.begin(), primary.frontier.end(),
          [](const Pair& left, const Pair& right) {
            return left.first == right.first;
          }),
      primary.frontier.end());
  std::sort(
      primary.frontier.begin(), primary.frontier.end(), by_exact_global);
  primary.visited.insert(
      primary.visited.end(),
      rescue.visited.begin(), rescue.visited.end());
  primary.distance_computations += rescue.distance_computations;
  primary.staged_candidates += rescue.staged_candidates;
  primary.prefix_rejected_candidates +=
      rescue.prefix_rejected_candidates;
  primary.exact_distance_cache_hits +=
      rescue.exact_distance_cache_hits;
  primary.incremental_exact_misses +=
      rescue.incremental_exact_misses;
  primary.exact_distance_completions +=
      rescue.exact_distance_completions;
  primary.uncached_prefix_rejections +=
      rescue.uncached_prefix_rejections;
  return primary;
}

// Data-free monotonicity and mapping counterexample.  Local IDs deliberately
// disagree with global-ID order; sorting ties by local ID would fail.
static int r96_degree4_sidecar_self_test() {
  using Result = BatchedL2BeamSearchResult<int32_t, float>;
  const std::vector<int32_t> subset{50, 10, 30, 20, 40};
  Result primary{};
  primary.frontier = {{0, 1.0f}, {1, 2.0f}, {2, 3.0f}};
  primary.visited = primary.frontier;
  primary.distance_computations = 3;
  Result rescue{};
  rescue.frontier = {
      {3, 1.0f},  // Equal distance, smaller global ID than local 0.
      {1, 2.0f},  // Duplicate must not appear twice.
      {4, 2.5f}};
  rescue.visited = rescue.frontier;
  rescue.distance_computations = 3;
  const size_t expected_work =
      primary.distance_computations + rescue.distance_computations;
  const Result merged = merge_r96_degree4_sidecar_results(
      std::move(primary), std::move(rescue), subset);
  const std::vector<int32_t> expected_local_order{3, 0, 1, 4, 2};
  std::vector<int32_t> actual_local_order;
  for (const auto& item : merged.frontier) {
    actual_local_order.push_back(item.first);
  }
  if (actual_local_order != expected_local_order ||
      merged.distance_computations != expected_work) {
    throw std::runtime_error(
        "degree4 sidecar exact-global merge counterexample failed");
  }

  // Canonical full-support GT for k=3 is {3,0,1}.  The primary has two hits
  // and the independent exact union has three: adding candidates cannot
  // reduce recall under the same total order.
  const std::unordered_set<int32_t> canonical_gt{3, 0, 1};
  const std::vector<int32_t> primary_top{0, 1, 2};
  size_t primary_hits = 0;
  size_t merged_hits = 0;
  for (int32_t local : primary_top) {
    primary_hits += canonical_gt.count(local);
  }
  for (size_t rank = 0; rank < 3; ++rank) {
    merged_hits += canonical_gt.count(
        merged.frontier[rank].first);
  }
  if (merged_hits < primary_hits || primary_hits != 2 ||
      merged_hits != 3) {
    throw std::runtime_error(
        "degree4 sidecar monotonic-recall counterexample failed");
  }
  constexpr size_t kHandoffSeeds = 10;
  constexpr size_t kVisitCap = 64;
  constexpr size_t kDegreeCap = 4;
  constexpr size_t kMaximumScores =
      kHandoffSeeds + kVisitCap * kDegreeCap;
  static_assert(kMaximumScores == 266);
  std::printf(
      "[R96 degree4 sidecar self-test] PASS "
      "primary_immutable=1 independent_state=1 "
      "exact_global_merge=1 score_cap=%zu\n",
      kMaximumScores);
  return 0;
}
#endif

#ifdef BCI_ENABLE_TWO_VIEW_R32_DIAGNOSTIC
template <typename indexType, typename distanceType>
BatchedL2BeamSearchResult<indexType, distanceType>
merge_independent_two_view_results(
    BatchedL2BeamSearchResult<indexType, distanceType> left,
    BatchedL2BeamSearchResult<indexType, distanceType> right) {
  using Result = BatchedL2BeamSearchResult<indexType, distanceType>;
  using Pair = std::pair<indexType, distanceType>;
  const auto by_id = [](const Pair& a, const Pair& b) {
    return a.first < b.first ||
           (a.first == b.first && a.second < b.second);
  };
  const auto by_distance = [](const Pair& a, const Pair& b) {
    return a.second < b.second ||
           (a.second == b.second && a.first < b.first);
  };
  left.frontier.insert(
      left.frontier.end(),
      right.frontier.begin(), right.frontier.end());
  std::sort(left.frontier.begin(), left.frontier.end(), by_id);
  left.frontier.erase(
      std::unique(
          left.frontier.begin(), left.frontier.end(),
          [](const Pair& a, const Pair& b) {
            return a.first == b.first;
          }),
      left.frontier.end());
  std::sort(left.frontier.begin(), left.frontier.end(), by_distance);

  // Keep duplicate visits across the two vectors: they are independent
  // searches and the counter must reflect their summed work.
  left.visited.insert(
      left.visited.end(),
      right.visited.begin(), right.visited.end());
  left.distance_computations += right.distance_computations;
  left.staged_candidates += right.staged_candidates;
  left.prefix_rejected_candidates +=
      right.prefix_rejected_candidates;
  left.exact_distance_cache_hits +=
      right.exact_distance_cache_hits;
  left.incremental_exact_misses +=
      right.incremental_exact_misses;
  left.exact_distance_completions +=
      right.exact_distance_completions;
  left.uncached_prefix_rejections +=
      right.uncached_prefix_rejections;
  return left;
}

// Data-free semantic counterexample.  Even slots lead only to a worse basin;
// odd slots lead to the true nearest point.  The gate fails if either parity
// is ignored, if the stride is silently treated as one, or if the two search
// states are not executed and merged independently.
struct TwoViewR32ToyPointRange {
  std::array<std::array<uint8_t, 4>, 6> values{{
      {{100, 100, 100, 100}},
      {{90, 90, 90, 90}},
      {{80, 80, 80, 80}},
      {{20, 20, 20, 20}},
      {{10, 10, 10, 10}},
      {{1, 1, 1, 1}},
  }};

  size_t size() const { return values.size(); }
  long dimension() const { return 4; }
  long aligned_dimension() const { return 4; }
  PointT operator[](long i) {
    return PointT(
        values.at(static_cast<size_t>(i)).data(), 4, 4, i);
  }
};

static int two_view_r32_self_test() {
  TwoViewR32ToyPointRange points;
  GraphI graph(2, points.size());
  graph[0].update_neighbors(parlay::sequence<Indx>{1, 3});
  graph[1].update_neighbors(parlay::sequence<Indx>{2, 4});
  graph[3].update_neighbors(parlay::sequence<Indx>{1, 4});
  graph[4].update_neighbors(parlay::sequence<Indx>{1, 5});
  const std::array<uint8_t, 4> query_values{{0, 0, 0, 0}};
  PointT query(query_values.data(), 4, 4, -1);
  QueryParams params(
      /*k=*/1, /*beam=*/2, /*cut=*/1.35,
      /*limit=*/100, /*degree_limit=*/2);
  auto even = beam_search_batched_l2<
      PointT, TwoViewR32ToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 0, 2);
  auto odd = beam_search_batched_l2<
      PointT, TwoViewR32ToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 1, 2);
  const size_t expected_work =
      even.distance_computations + odd.distance_computations;
  auto merged = merge_independent_two_view_results(
      std::move(even), std::move(odd));
  if (merged.frontier.empty() ||
      merged.frontier.front().first != 5 ||
      merged.distance_computations != expected_work ||
      merged.visited.size() < 4) {
    throw std::runtime_error(
        "two-view R32 self-test failed to merge independent basins");
  }

  auto even_only = beam_search_batched_l2<
      PointT, TwoViewR32ToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 0, 2);
  if (even_only.frontier.empty() ||
      even_only.frontier.front().first != 2 ||
      !(merged.frontier.front().second <
        even_only.frontier.front().second)) {
    throw std::runtime_error(
        "two-view R32 self-test did not isolate parity");
  }

  auto full = beam_search_batched_l2<
      PointT, TwoViewR32ToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 0, 1);
  auto escape = beam_search_batched_l2<
      PointT, TwoViewR32ToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          params, 0, nullptr, 1, 2);
  const size_t full_escape_work =
      full.distance_computations + escape.distance_computations;
  const float full_best = full.frontier.front().second;
  auto full_escape = merge_independent_two_view_results(
      std::move(full), std::move(escape));
  if (full_escape.frontier.empty() ||
      full_escape.frontier.front().second > full_best ||
      full_escape.distance_computations != full_escape_work) {
    throw std::runtime_error(
        "two-view R32 self-test failed to preserve the full primary view");
  }
  std::printf(
      "[two-view R32 self-test] PASS even_top=%d merged_top=%d "
      "summed_distance_computations=%zu full_primary=preserved\n",
      even_only.frontier.front().first,
      merged.frontier.front().first,
      merged.distance_computations);
  return 0;
}
#endif

#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
struct ShortcutResidualMask {
  yfcc_shortcut_residual_mask_v1::Header header{};
  std::vector<uint8_t> payload;
  std::vector<uint64_t> row_masks;
  uint64_t novel_edges = 0;
  uint64_t file_bytes = 0;
};

static std::vector<uint64_t> decode_shortcut_row_masks(
    GraphI& graph, const std::vector<uint8_t>& payload,
    uint64_t expected_edges, uint64_t& novel_edges) {
  using namespace yfcc_shortcut_residual_mask_v1;
  if (payload.size() != payload_bytes_for(expected_edges)) {
    throw std::runtime_error(
        "shortcut residual mask payload length mismatch");
  }
  if (expected_edges % 8 != 0 && !payload.empty()) {
    const unsigned valid_bits =
        static_cast<unsigned>(expected_edges % 8);
    const uint8_t invalid_mask =
        static_cast<uint8_t>(0xffU << valid_bits);
    if ((payload.back() & invalid_mask) != 0) {
      throw std::runtime_error(
          "shortcut residual mask has nonzero tail bits");
    }
  }
  if (graph.max_degree() <= 0 || graph.max_degree() > 64) {
    throw std::runtime_error(
        "shortcut residual mask requires graph max degree in [1,64]");
  }
  std::vector<uint64_t> row_masks(graph.size(), 0);
  uint64_t edge_position = 0;
  novel_edges = 0;
  for (size_t node = 0; node < graph.size(); ++node) {
    const size_t degree = graph[static_cast<long>(node)].size();
    if (degree > 64 || edge_position + degree > expected_edges) {
      throw std::runtime_error(
          "shortcut residual mask row/edge mapping overflow");
    }
    uint64_t mask = 0;
    for (size_t slot = 0; slot < degree; ++slot) {
      if (bit_is_set(payload.data(), edge_position + slot)) {
        mask |= uint64_t{1} << static_cast<unsigned>(slot);
        ++novel_edges;
      }
    }
    row_masks[node] = mask;
    edge_position += degree;
  }
  if (edge_position != expected_edges) {
    throw std::runtime_error(
        "shortcut residual mask edge count does not match graph rows");
  }
  return row_masks;
}

static ShortcutResidualMask load_shortcut_residual_mask(
    const std::filesystem::path& path, GraphI& graph,
    uint64_t current_graph_bytes) {
  using namespace yfcc_shortcut_residual_mask_v1;
  ShortcutResidualMask output;
  std::error_code error;
  output.file_bytes =
      std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error(
        "cannot stat shortcut residual mask: " + error.message());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "cannot open shortcut residual mask: " + path.string());
  }
  input.read(
      reinterpret_cast<char*>(&output.header),
      static_cast<std::streamsize>(sizeof(output.header)));
  if (!input ||
      !std::equal(
          kMagic.begin(), kMagic.end(),
          output.header.magic) ||
      output.header.version != kVersion ||
      output.header.header_bytes != kHeaderBytes ||
      output.header.flags != kFlagLsbFirst ||
      output.header.rows != graph.size() ||
      output.header.max_degree !=
          static_cast<uint32_t>(graph.max_degree()) ||
      output.header.current_graph_bytes != current_graph_bytes ||
      output.header.payload_bytes !=
          payload_bytes_for(output.header.edge_count) ||
      output.file_bytes !=
          static_cast<uint64_t>(kHeaderBytes) +
              output.header.payload_bytes ||
      output.header.current_graph_bytes + output.file_bytes >
          output.header.control_graph_bytes) {
    throw std::runtime_error(
        "shortcut residual mask header/resource contract failed");
  }
  const uint64_t graph_container_bytes =
      8ULL + 4ULL * output.header.rows +
      4ULL * output.header.edge_count;
  if (graph_container_bytes != current_graph_bytes ||
      output.header.payload_bytes >
          static_cast<uint64_t>(
              std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(
        "shortcut residual mask graph geometry mismatch");
  }
  output.payload.resize(
      static_cast<size_t>(output.header.payload_bytes));
  if (!output.payload.empty()) {
    input.read(
        reinterpret_cast<char*>(output.payload.data()),
        static_cast<std::streamsize>(output.payload.size()));
  }
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error(
        "shortcut residual mask payload is truncated or extended");
  }
  output.row_masks = decode_shortcut_row_masks(
      graph, output.payload, output.header.edge_count,
      output.novel_edges);
  // The query kernel uses fixed-width row masks.  Do not retain the packed
  // file payload as a second in-memory copy after its row axis is validated.
  output.payload.clear();
  output.payload.shrink_to_fit();
  if (output.novel_edges == 0) {
    throw std::runtime_error(
        "shortcut residual mask contains no novel edges");
  }
  return output;
}

struct ShortcutResidualToyPointRange {
  std::array<std::array<uint8_t, 4>, 7> values{{
      {{5, 5, 5, 5}},  // entry zero: distance 100
      {{2, 2, 1, 1}},  // primary seed: distance 10
      {{2, 2, 2, 0}},  // primary decoy: distance 12
      {{2, 2, 2, 1}},  // primary decoy: distance 13
      {{4, 4, 0, 0}},  // unused
      {{4, 4, 0, 0}},  // shortcut bridge: distance 32
      {{0, 0, 0, 1}},  // true nearest: distance 1
  }};

  size_t size() const { return values.size(); }
  long dimension() const { return 4; }
  long aligned_dimension() const { return 4; }
  PointT operator[](long i) {
    return PointT(
        values.at(static_cast<size_t>(i)).data(), 4, 4, i);
  }
};

static int shortcut_residual_self_test() {
  ShortcutResidualToyPointRange points;
  GraphI graph(3, points.size());
  graph[0].update_neighbors(parlay::sequence<Indx>{1, 2, 3});
  graph[1].update_neighbors(parlay::sequence<Indx>{5});
  graph[5].update_neighbors(parlay::sequence<Indx>{6});

  // Flattened positions are row0:[0,3), row1:[3,4), row5:[4,5).
  std::vector<uint8_t> payload(1, 0);
  payload[0] =
      static_cast<uint8_t>((uint8_t{1} << 3) |
                           (uint8_t{1} << 4));
  uint64_t novel_edges = 0;
  const auto row_masks = decode_shortcut_row_masks(
      graph, payload, 5, novel_edges);
  if (novel_edges != 2 || row_masks[0] != 0 ||
      row_masks[1] != 1 || row_masks[5] != 1) {
    throw std::runtime_error(
        "shortcut residual self-test decoded the wrong row masks");
  }

  const std::array<uint8_t, 4> query_values{{0, 0, 0, 0}};
  PointT query(query_values.data(), 4, 4, -1);
  QueryParams primary_params(
      /*k=*/3, /*beam=*/3, /*cut=*/1.35,
      /*limit=*/100, /*degree_limit=*/3);
  auto primary = beam_search_batched_l2<
      PointT, ShortcutResidualToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          primary_params, 0);
  if (primary.frontier.size() < 3 ||
      primary.frontier.front().first == 6) {
    throw std::runtime_error(
        "shortcut residual toy primary unexpectedly found the target");
  }

  parlay::sequence<Indx> primary_seeds;
  for (size_t i = 0;
       i < std::min<size_t>(3, primary.frontier.size()); ++i) {
    primary_seeds.push_back(primary.frontier[i].first);
  }
  QueryParams residual_params(
      /*k=*/3, /*beam=*/4, /*cut=*/1.35,
      /*limit=*/100, /*degree_limit=*/3);
  auto residual = beam_search_batched_l2<
      PointT, ShortcutResidualToyPointRange, Indx>(
          query, graph, points, primary_seeds,
          residual_params, 0, nullptr, 0, 1, &row_masks);
  auto entry_only = beam_search_batched_l2<
      PointT, ShortcutResidualToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0},
          residual_params, 0, nullptr, 0, 1, &row_masks);
  if (residual.frontier.empty() ||
      residual.frontier.front().first != 6 ||
      (!entry_only.frontier.empty() &&
       entry_only.frontier.front().first == 6)) {
    throw std::runtime_error(
        "shortcut residual toy did not require primary-frontier seeds");
  }

  // Final ranking is canonical exact (distance, global id), not provenance.
  std::vector<std::pair<float, int32_t>> candidates{
      {4.0f, 9}, {4.0f, 3}, {1.0f, 7}, {1.0f, 7}};
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto& left, const auto& right) {
        return left.second < right.second ||
               (left.second == right.second &&
                left.first < right.first);
      });
  candidates.erase(
      std::unique(
          candidates.begin(), candidates.end(),
          [](const auto& left, const auto& right) {
            return left.second == right.second;
          }),
      candidates.end());
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto& left, const auto& right) {
        return left.first < right.first ||
               (left.first == right.first &&
                left.second < right.second);
      });
  if (candidates.size() != 3 ||
      candidates[0] != std::pair<float, int32_t>{1.0f, 7} ||
      candidates[1] != std::pair<float, int32_t>{4.0f, 3}) {
    throw std::runtime_error(
        "shortcut residual canonical merge/tie self-test failed");
  }

  bool caught_tail = false;
  try {
    auto invalid_payload = payload;
    invalid_payload[0] |= uint8_t{1} << 7;
    uint64_t ignored = 0;
    (void)decode_shortcut_row_masks(
        graph, invalid_payload, 5, ignored);
  } catch (const std::runtime_error&) {
    caught_tail = true;
  }
  if (!caught_tail) {
    throw std::runtime_error(
        "shortcut residual self-test accepted nonzero tail bits");
  }

  std::printf(
      "[shortcut residual self-test] PASS "
      "primary_top=%d residual_top=%d entry_only_top=%d "
      "novel_edges=%llu canonical_tie_global=3\n",
      primary.frontier.front().first,
      residual.frontier.front().first,
      entry_only.frontier.empty()
          ? -1
          : entry_only.frontier.front().first,
      static_cast<unsigned long long>(novel_edges));
  return 0;
}
#endif

#ifdef BCI_ENABLE_BATCHED_MULTI_START_DIAGNOSTIC
// Data-free counterexample for the exact semantic change authorized by the
// development-only multi-start target.  The two components have no cross
// edge: entry zero can only reach the deliberately worse component, while a
// distinct second entry reaches the true nearest point.  This catches a
// runner that parses two starts but silently executes only entry zero.
struct BatchedMultiStartToyPointRange {
  std::array<std::array<uint8_t, 4>, 6> values{{
      {{100, 100, 100, 100}},
      {{90, 90, 90, 90}},
      {{80, 80, 80, 80}},
      {{20, 20, 20, 20}},
      {{10, 10, 10, 10}},
      {{1, 1, 1, 1}},
  }};

  size_t size() const { return values.size(); }
  long dimension() const { return 4; }
  long aligned_dimension() const { return 4; }
  PointT operator[](long i) {
    return PointT(values.at(static_cast<size_t>(i)).data(), 4, 4, i);
  }
};

static int batched_multi_start_self_test() {
  BatchedMultiStartToyPointRange points;
  GraphI graph(2, points.size());
  graph[0].update_neighbors(parlay::sequence<Indx>{1});
  graph[1].update_neighbors(parlay::sequence<Indx>{2});
  graph[3].update_neighbors(parlay::sequence<Indx>{4});
  graph[4].update_neighbors(parlay::sequence<Indx>{5});
  const std::array<uint8_t, 4> query_values{{0, 0, 0, 0}};
  PointT query(query_values.data(), 4, 4, -1);
  QueryParams params(/*k=*/1, /*beam=*/2, /*cut=*/1.35,
                     /*limit=*/100, /*degree_limit=*/2);
  const auto single = beam_search_batched_l2<PointT,
      BatchedMultiStartToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0}, params, 0);
  const auto multi = beam_search_batched_l2<PointT,
      BatchedMultiStartToyPointRange, Indx>(
          query, graph, points, parlay::sequence<Indx>{0, 3}, params, 0);
  if (single.frontier.empty() || multi.frontier.empty() ||
      single.frontier.front().first != 2 ||
      multi.frontier.front().first != 5 ||
      !(multi.frontier.front().second <
        single.frontier.front().second)) {
    throw std::runtime_error(
        "batched multi-start self-test failed to recover alternate basin");
  }
  std::printf(
      "[batched multi-start self-test] PASS "
      "single_top=%d multi_top=%d\n",
      single.frontier.front().first, multi.frontier.front().first);
  return 0;
}
#endif

#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
struct CachedReplayTop10Churn {
  size_t intersection = 0;
  size_t entered = 0;
  size_t dropped = 0;
  double base_d10 = std::numeric_limits<double>::quiet_NaN();
  double replay_d10 = std::numeric_limits<double>::quiet_NaN();
  double min_entered_margin_abs =
      std::numeric_limits<double>::quiet_NaN();
  double max_entered_margin_abs =
      std::numeric_limits<double>::quiet_NaN();

  bool changed() const {
    return entered != 0 || dropped != 0;
  }
};

template <typename indexType, typename distanceType>
CachedReplayTop10Churn analyze_cached_replay_top10_churn(
    const std::vector<std::pair<indexType, distanceType>>& base,
    const std::vector<std::pair<indexType, distanceType>>& replay,
    size_t k) {
  CachedReplayTop10Churn output;
  const size_t base_size = std::min(k, base.size());
  const size_t replay_size = std::min(k, replay.size());
  std::vector<indexType> base_ids;
  std::vector<indexType> replay_ids;
  base_ids.reserve(base_size);
  replay_ids.reserve(replay_size);
  for (size_t i = 0; i < base_size; ++i) {
    base_ids.push_back(base[i].first);
  }
  for (size_t i = 0; i < replay_size; ++i) {
    replay_ids.push_back(replay[i].first);
  }
  std::sort(base_ids.begin(), base_ids.end());
  std::sort(replay_ids.begin(), replay_ids.end());
  output.intersection = 0;
  size_t left = 0;
  size_t right = 0;
  while (left < base_ids.size() && right < replay_ids.size()) {
    if (base_ids[left] < replay_ids[right]) {
      ++left;
    } else if (replay_ids[right] < base_ids[left]) {
      ++right;
    } else {
      ++output.intersection;
      ++left;
      ++right;
    }
  }
  output.entered = replay_size - output.intersection;
  output.dropped = base_size - output.intersection;
  if (base_size == k) {
    output.base_d10 = static_cast<double>(base[k - 1].second);
  }
  if (replay_size == k) {
    output.replay_d10 =
        static_cast<double>(replay[k - 1].second);
  }
  if (base_size == k && output.entered != 0) {
    double min_margin = std::numeric_limits<double>::infinity();
    double max_margin = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < replay_size; ++i) {
      if (!std::binary_search(
              base_ids.begin(), base_ids.end(),
              replay[i].first)) {
        const double margin =
            output.base_d10 -
            static_cast<double>(replay[i].second);
        min_margin = std::min(min_margin, margin);
        max_margin = std::max(max_margin, margin);
      }
    }
    output.min_entered_margin_abs = min_margin;
    output.max_entered_margin_abs = max_margin;
  }
  return output;
}

struct CachedReplayToyResult {
  std::vector<std::pair<int32_t, float>> frontier;
  std::vector<std::pair<int32_t, float>> visited;
  size_t logical_computations = 0;
  size_t cache_hits = 0;
  size_t exact_misses = 0;
  size_t exact_completions = 0;
  size_t uncached_prefix_rejections = 0;
};

static CachedReplayToyResult cached_replay_toy_beam_search(
    const std::vector<std::vector<int32_t>>& graph,
    const std::vector<float>& distances,
    const std::vector<float>& prefix_distances,
    long beam, long k, double cut,
    ExactDistanceMemo<int32_t, float>* memo) {
  using Pair = std::pair<int32_t, float>;
  const auto less = [](const Pair& left, const Pair& right) {
    return left.second < right.second ||
           (left.second == right.second &&
            left.first < right.first);
  };
  const int bits = std::max<int>(
      10, std::ceil(std::log2(beam * beam)) - 2);
  std::vector<int32_t> hash_filter(size_t{1} << bits, -1);
  auto has_been_seen = [&](int32_t candidate) {
    const size_t location =
        parlay::hash64_2(candidate) & (hash_filter.size() - 1);
    if (hash_filter[location] == candidate) return true;
    hash_filter[location] = candidate;
    return false;
  };

  CachedReplayToyResult output;
  auto starting_distance = [&](int32_t candidate) {
    ++output.logical_computations;
    if (memo != nullptr) {
      float cached_distance = 0.0f;
      if (memo->lookup(candidate, cached_distance)) {
        ++output.cache_hits;
        return cached_distance;
      }
    }
    const float distance = distances.at(candidate);
    ++output.exact_misses;
    ++output.exact_completions;
    if (memo != nullptr) {
      memo->insert(candidate, distance);
    }
    return distance;
  };

  std::vector<Pair> frontier{{0, starting_distance(0)}};
  std::vector<Pair> visited;
  std::vector<Pair> unvisited_frontier(static_cast<size_t>(beam));
  unvisited_frontier[0] = frontier[0];
  size_t max_degree = 0;
  for (const auto& neighbors : graph) {
    max_degree = std::max(max_degree, neighbors.size());
  }
  std::vector<Pair> new_frontier(
      static_cast<size_t>(beam) + max_degree);
  std::vector<Pair> candidates;
  candidates.reserve(max_degree);
  int remaining = 1;
  int visited_count = 0;
  while (remaining > 0 && visited_count < 1'000) {
    const Pair current = unvisited_frontier[0];
    visited.insert(
        std::upper_bound(
            visited.begin(), visited.end(), current, less),
        current);
    ++visited_count;
    candidates.clear();
    const float cutoff_distance =
        frontier.size() < static_cast<size_t>(beam)
            ? static_cast<float>(
                  std::numeric_limits<int>::max())
            : frontier.back().second;
    for (int32_t candidate : graph.at(current.first)) {
      if (has_been_seen(candidate)) continue;
      ++output.logical_computations;
      float distance = 0.0f;
      if (memo != nullptr &&
          memo->lookup(candidate, distance)) {
        ++output.cache_hits;
        if (distance < cutoff_distance) {
          candidates.push_back({candidate, distance});
        }
        continue;
      }
      ++output.exact_misses;
      if (prefix_distances.at(candidate) >= cutoff_distance) {
        ++output.uncached_prefix_rejections;
        continue;
      }
      distance = distances.at(candidate);
      ++output.exact_completions;
      if (memo != nullptr) {
        memo->insert(candidate, distance);
      }
      if (distance >= cutoff_distance) continue;
      candidates.push_back({candidate, distance});
    }
    std::sort(candidates.begin(), candidates.end(), less);
    size_t frontier_size = static_cast<size_t>(
        std::set_union(
            frontier.begin(), frontier.end(),
            candidates.begin(), candidates.end(),
            new_frontier.begin(), less) -
        new_frontier.begin());
    frontier_size =
        std::min(static_cast<size_t>(beam), frontier_size);
    if (k > 0 && frontier_size > static_cast<size_t>(k)) {
      frontier_size = static_cast<size_t>(
          std::upper_bound(
              new_frontier.begin(),
              new_frontier.begin() + frontier_size,
              Pair{
                  0,
                  static_cast<float>(
                      cut * new_frontier[k].second)},
              less) -
          new_frontier.begin());
    }
    frontier.assign(
        new_frontier.begin(),
        new_frontier.begin() + frontier_size);
    remaining = static_cast<int>(
        std::set_difference(
            frontier.begin(), frontier.end(),
            visited.begin(), visited.end(),
            unvisited_frontier.begin(), less) -
        unvisited_frontier.begin());
  }
  output.frontier = std::move(frontier);
  output.visited = std::move(visited);
  return output;
}

static int cached_replay_self_test() {
  constexpr long kBaseBeam = 64;
  constexpr long kProbeBeam = 68;
  constexpr long kRepairBeam = 256;
  constexpr long k = 10;
  constexpr double kCut = 1.35;
  std::vector<std::vector<int32_t>> graph(80);
  std::vector<float> distances(80, 120.0f);
  std::vector<float> prefix_distances(80, 120.0f);
  distances[0] = 200.0f;
  prefix_distances[0] = 150.0f;
  for (int32_t id = 1; id <= 70; ++id) {
    graph[0].push_back(id);
    distances[id] =
        id <= 64
            ? 100.0f + static_cast<float>(id) / 100.0f
            : 120.0f +
                  static_cast<float>(id - 65) / 100.0f;
    prefix_distances[id] = distances[id];
  }
  // b64 trims both gateways.  b68 retains 65 and discovers 71; b256
  // additionally retains 69 and discovers 72.
  graph[65].push_back(71);
  distances[71] = 1.0f;
  prefix_distances[71] = 0.5f;
  graph[69].push_back(72);
  distances[72] = 0.5f;
  prefix_distances[72] = 0.25f;
  // Candidate 73 is a partial-distance poison test: b64 rejects its
  // 64-D lower bound against the tighter cutoff and therefore must not
  // cache 125 as though 110 were exact.  b68 has a wider cutoff, completes
  // the same candidate, and then caches the true exact value.
  graph[1].push_back(73);
  distances[73] = 125.0f;
  prefix_distances[73] = 110.0f;

  const auto fresh64 = cached_replay_toy_beam_search(
      graph, distances, prefix_distances,
      kBaseBeam, k, kCut, nullptr);
  ExactDistanceMemo<int32_t, float> memo(graph.size());
  memo.begin_query();
  const auto cached64 = cached_replay_toy_beam_search(
      graph, distances, prefix_distances,
      kBaseBeam, k, kCut, &memo);
  if (fresh64.frontier != cached64.frontier ||
      fresh64.visited != cached64.visited ||
      fresh64.logical_computations !=
          cached64.logical_computations ||
      memo.contains(73) ||
      cached64.uncached_prefix_rejections == 0 ||
      cached64.exact_misses !=
          cached64.exact_completions +
              cached64.uncached_prefix_rejections) {
    throw std::runtime_error(
        "cached replay self-test b64 fresh/cache/prefix poison failed");
  }

  const auto fresh68 = cached_replay_toy_beam_search(
      graph, distances, prefix_distances,
      kProbeBeam, k, kCut, nullptr);
  const auto cached68 = cached_replay_toy_beam_search(
      graph, distances, prefix_distances,
      kProbeBeam, k, kCut, &memo);
  if (fresh68.frontier != cached68.frontier ||
      fresh68.visited != cached68.visited ||
      fresh68.logical_computations !=
          cached68.logical_computations ||
      cached68.cache_hits + cached68.exact_misses !=
          cached68.logical_computations ||
      cached68.exact_misses !=
          cached68.exact_completions +
              cached68.uncached_prefix_rejections ||
      cached68.cache_hits == 0 ||
      cached68.exact_misses == 0 ||
      !memo.contains(73)) {
    throw std::runtime_error(
        "cached replay self-test b68 fresh/cache differential failed");
  }
  const auto probe_churn =
      analyze_cached_replay_top10_churn(
          cached64.frontier, cached68.frontier, k);
  if (probe_churn.intersection != 9 ||
      probe_churn.entered != 1 ||
      probe_churn.dropped != 1) {
    throw std::runtime_error(
        "cached replay self-test b68 top10 churn failed");
  }

  const auto fresh256 = cached_replay_toy_beam_search(
      graph, distances, prefix_distances,
      kRepairBeam, k, kCut, nullptr);
  const auto cached256 = cached_replay_toy_beam_search(
      graph, distances, prefix_distances,
      kRepairBeam, k, kCut, &memo);
  if (fresh256.frontier != cached256.frontier ||
      fresh256.visited != cached256.visited ||
      fresh256.logical_computations !=
          cached256.logical_computations ||
      cached256.cache_hits + cached256.exact_misses !=
          cached256.logical_computations ||
      cached256.exact_misses !=
          cached256.exact_completions +
              cached256.uncached_prefix_rejections ||
      cached256.cache_hits == 0 ||
      cached256.exact_misses == 0) {
    throw std::runtime_error(
        "cached replay self-test b256 fresh/cache differential failed");
  }
  const auto repair_churn =
      analyze_cached_replay_top10_churn(
          cached64.frontier, cached256.frontier, k);
  if (repair_churn.intersection != 8 ||
      repair_churn.entered != 2 ||
      repair_churn.dropped != 2) {
    throw std::runtime_error(
        "cached replay self-test b256 top10 churn failed");
  }
  std::printf(
      "[cached replay self-test] PASS "
      "b68_logical=%zu b68_hits=%zu b68_misses=%zu "
      "b256_logical=%zu b256_hits=%zu b256_misses=%zu\n",
      cached68.logical_computations, cached68.cache_hits,
      cached68.exact_misses, cached256.logical_computations,
      cached256.cache_hits, cached256.exact_misses);
  return 0;
}
#endif

template <typename Point, typename PointRange, typename indexType>
ReusableBeamSearchView<indexType, typename Point::distanceType>
beam_search_reusable(
    Point query, Graph<indexType>& graph, PointRange& points,
    const parlay::sequence<indexType>& starting_points, QueryParams& params,
    ReusableBeamWorkspace<indexType, typename Point::distanceType>& workspace) {
  using distanceType = typename Point::distanceType;
  auto less = [](const std::pair<indexType, distanceType>& left,
                 const std::pair<indexType, distanceType>& right) {
    return left.second < right.second ||
           (left.second == right.second && left.first < right.first);
  };

  const int bits = std::max<int>(
      10, std::ceil(std::log2(params.beamSize * params.beamSize)) - 2);
  const size_t hash_size = size_t{1} << bits;
  constexpr uint32_t CANDIDATE_BITS = 22;
  constexpr uint32_t CANDIDATE_MASK =
      (uint32_t{1} << CANDIDATE_BITS) - 1;
  constexpr uint32_t MAX_HASH_EPOCH =
      (uint32_t{1} << (32 - CANDIDATE_BITS)) - 1;
  if (points.size() > (size_t{1} << CANDIDATE_BITS)) {
    throw std::runtime_error(
        "packed reusable beam hash requires shard size <= 2^22");
  }
  if (workspace.hash_slots.size() != hash_size) {
    workspace.hash_slots.assign(hash_size, 0);
    workspace.hash_epoch = 0;
  }
  ++workspace.hash_epoch;
  if (workspace.hash_epoch > MAX_HASH_EPOCH) {
    std::fill(
        workspace.hash_slots.begin(), workspace.hash_slots.end(), 0);
    workspace.hash_epoch = 1;
  }
  auto has_been_seen = [&](indexType candidate) {
    if (candidate < 0 ||
        static_cast<uint32_t>(candidate) > CANDIDATE_MASK) {
      throw std::runtime_error(
          "packed reusable beam candidate is out of range");
    }
    const size_t location =
        parlay::hash64_2(candidate) & (hash_size - 1);
    const uint32_t encoded =
        (workspace.hash_epoch << CANDIDATE_BITS) |
        static_cast<uint32_t>(candidate);
    if (workspace.hash_slots[location] == encoded) {
      return true;
    }
    workspace.hash_slots[location] = encoded;
    return false;
  };
  workspace.frontier.clear();
  workspace.frontier.reserve(params.beamSize);
  for (indexType start : starting_points) {
    workspace.frontier.push_back(
        {start, points[start].distance(query)});
  }
  std::sort(workspace.frontier.begin(), workspace.frontier.end(), less);

  workspace.unvisited_frontier.resize(params.beamSize);
  workspace.unvisited_frontier[0] = workspace.frontier[0];
  workspace.visited.clear();
  workspace.visited.reserve(2 * params.beamSize);
  workspace.new_frontier.resize(params.beamSize + graph.max_degree());
  workspace.candidates.clear();
  workspace.candidates.reserve(graph.max_degree());
  workspace.keep.clear();
  workspace.keep.reserve(graph.max_degree());

  size_t distance_computations = starting_points.size();
  int remaining = 1;
  int visited_count = 0;
  while (remaining > 0 && visited_count < params.limit) {
    const auto current = workspace.unvisited_frontier[0];
    graph[current.first].prefetch();
    workspace.visited.insert(
        std::upper_bound(
            workspace.visited.begin(), workspace.visited.end(),
            current, less),
        current);
    ++visited_count;

    workspace.candidates.clear();
    workspace.keep.clear();
    const long neighbor_count = std::min<long>(
        graph[current.first].size(), params.degree_limit);
    for (indexType neighbor_index = 0;
         neighbor_index < neighbor_count; ++neighbor_index) {
      const indexType candidate = graph[current.first][neighbor_index];
      if (candidate == query.id() || has_been_seen(candidate)) continue;
      workspace.keep.push_back(candidate);
      auto candidate_point = points[candidate];
      __builtin_prefetch(candidate_point.get(), 0, 1);
    }

    const distanceType cutoff =
        workspace.frontier.size() <
                static_cast<size_t>(params.beamSize)
            ? static_cast<distanceType>(
                  std::numeric_limits<int>::max())
            : workspace.frontier.back().second;
    for (indexType candidate : workspace.keep) {
      auto candidate_point = points[candidate];
      bool early_abandoned = false;
      const distanceType distance =
          l2_sq_uint8_avx2_cutoff(
              candidate_point.get(), query.get(),
              static_cast<unsigned>(points.dimension()), cutoff,
              &early_abandoned);
      ++workspace.staged_distance_candidates;
      workspace.early_abandoned_candidates +=
          static_cast<size_t>(early_abandoned);
      ++distance_computations;
      if (distance >= cutoff) continue;
      workspace.candidates.push_back({candidate, distance});
    }
    std::sort(
        workspace.candidates.begin(), workspace.candidates.end(), less);

    size_t new_frontier_size = static_cast<size_t>(
        std::set_union(
            workspace.frontier.begin(), workspace.frontier.end(),
            workspace.candidates.begin(), workspace.candidates.end(),
            workspace.new_frontier.begin(), less) -
        workspace.new_frontier.begin());
    new_frontier_size = std::min<size_t>(
        params.beamSize, new_frontier_size);
    if (params.k > 0 &&
        new_frontier_size > static_cast<size_t>(params.k) &&
        points[0].is_metric()) {
      new_frontier_size = static_cast<size_t>(
          std::upper_bound(
              workspace.new_frontier.begin(),
              workspace.new_frontier.begin() + new_frontier_size,
              std::pair<indexType, distanceType>{
                  0,
                  static_cast<distanceType>(
                      params.cut *
                      workspace.new_frontier[params.k].second)},
              less) -
          workspace.new_frontier.begin());
    }
    workspace.frontier.assign(
        workspace.new_frontier.begin(),
        workspace.new_frontier.begin() + new_frontier_size);
    remaining = static_cast<int>(
        std::set_difference(
            workspace.frontier.begin(), workspace.frontier.end(),
            workspace.visited.begin(), workspace.visited.end(),
            workspace.unvisited_frontier.begin(), less) -
        workspace.unvisited_frontier.begin());
  }
  return {
      workspace.frontier, workspace.visited, distance_computations};
}

struct Shard {
  int32_t tag;
  int64_t freq;
  parlay::sequence<int32_t> subset; // local -> global base idx
  GraphI graph;
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
  std::unique_ptr<GraphI> r96_degree4_sidecar;
#endif
  long maxDeg;
};

// IVF²-style per-tag cluster index (cache-friendly working set for warm-cache QPS)
// Design note: centroids fit L1 (< 256KB), members CSR-style.
// Replaces beam_search with: linear centroid scan -> top-nprobe -> collect members -> exact rerank.
struct ClusterIndex {
  int32_t tag;
  int32_t n_clusters;
  int32_t aligned_dim;
  std::vector<uint8_t> centroid_data;     // n_clusters * aligned_dim, contiguous
  std::vector<int64_t> member_offsets;    // n_clusters + 1
  std::vector<int32_t> member_ids;        // flattened global point IDs
  std::vector<int32_t> representative_local_ids;  // centroid-nearest graph entries
};

struct PairPosting {
  std::vector<int32_t> ids;       // exact A ∩ B point ids, sorted by global id
  std::vector<uint32_t> primary_offsets;  // exact A ∩ B offsets in primary CSR
  std::vector<uint8_t> packed;    // ids.size() * aligned_dim bytes, contiguous
  int32_t primary_tag = -1;
};

enum class PairVectorLayout {
  PairPacked,
  PrimaryOffset,
};

static const char* pair_vector_layout_name(PairVectorLayout layout) {
  return layout == PairVectorLayout::PairPacked ? "pair_packed"
                                                 : "primary_offset";
}

static PairVectorLayout parse_pair_vector_layout(const char* raw) {
  const std::string value =
      raw != nullptr && raw[0] != '\0' ? std::string(raw) : "pair_packed";
  if (value == "pair_packed") return PairVectorLayout::PairPacked;
  if (value == "primary_offset") return PairVectorLayout::PrimaryOffset;
  throw std::runtime_error(
      "BCI_PAIR_VECTOR_LAYOUT must be pair_packed or primary_offset");
}

static void validate_primary_pack_alignment(
    const int32_t* primary_global_ids, size_t primary_size,
    const uint8_t* primary_packed, size_t aligned_dim,
    const std::function<const uint8_t*(int32_t)>& base_vector,
    const size_t* dimension_order = nullptr) {
  if ((primary_size > 0 &&
       (primary_global_ids == nullptr || primary_packed == nullptr)) ||
      aligned_dim == 0) {
    throw std::runtime_error("invalid common primary-pack alignment inputs");
  }
  for (size_t offset = 0; offset < primary_size; ++offset) {
    const uint8_t* expected = base_vector(primary_global_ids[offset]);
    bool matches = expected != nullptr;
    const uint8_t* observed =
        primary_packed + offset * aligned_dim;
    if (matches && dimension_order == nullptr) {
      matches =
          std::memcmp(observed, expected, aligned_dim) == 0;
    } else if (matches) {
      for (size_t dimension = 0; dimension < aligned_dim; ++dimension) {
        if (observed[dimension] != expected[dimension_order[dimension]]) {
          matches = false;
          break;
        }
      }
    }
    if (!matches) {
      throw std::runtime_error(
          "common primary pack/CSR/base alignment mismatch");
    }
  }
}

static PairPosting encode_pair_posting(
    int32_t primary_tag, const int32_t* primary_global_ids,
    size_t primary_size, size_t aligned_dim, PairVectorLayout layout,
    const std::function<bool(int32_t)>& keep,
    const std::function<const uint8_t*(int32_t)>& base_vector) {
  if ((primary_size > 0 && primary_global_ids == nullptr) ||
      aligned_dim == 0) {
    throw std::runtime_error("invalid pair-posting encoder inputs");
  }
  PairPosting posting;
  posting.primary_tag = primary_tag;
  if (layout == PairVectorLayout::PairPacked) {
    posting.ids.reserve(primary_size);
  } else {
    posting.primary_offsets.reserve(primary_size);
  }
  int32_t previous_global = -1;
  for (size_t offset = 0; offset < primary_size; ++offset) {
    int32_t global_id = primary_global_ids[offset];
    if (global_id < 0 ||
        (offset > 0 && global_id <= previous_global)) {
      throw std::runtime_error(
          "primary CSR posting is not strictly increasing");
    }
    previous_global = global_id;
    if (!keep(global_id)) continue;
    if (layout == PairVectorLayout::PairPacked) {
      posting.ids.push_back(global_id);
    } else {
      if (offset > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("primary-local pair offset exceeds uint32");
      }
      posting.primary_offsets.push_back((uint32_t)offset);
    }
  }
  if (layout == PairVectorLayout::PairPacked) {
    std::vector<int32_t>(posting.ids.begin(), posting.ids.end())
        .swap(posting.ids);
    posting.packed.resize(posting.ids.size() * aligned_dim);
    for (size_t k = 0; k < posting.ids.size(); ++k) {
      const uint8_t* source = base_vector(posting.ids[k]);
      if (source == nullptr) {
        throw std::runtime_error("pair encoder base-vector lookup failed");
      }
      std::memcpy(posting.packed.data() + k * aligned_dim, source,
                  aligned_dim);
    }
    if (posting.ids.capacity() != posting.ids.size() ||
        posting.packed.capacity() != posting.packed.size() ||
        !posting.primary_offsets.empty() ||
        posting.primary_offsets.capacity() != 0) {
      throw std::runtime_error(
          "pair-packed encoder retained excess capacity");
    }
  } else {
    std::vector<uint32_t>(
        posting.primary_offsets.begin(), posting.primary_offsets.end())
        .swap(posting.primary_offsets);
    if (posting.primary_offsets.capacity() !=
            posting.primary_offsets.size() ||
        !posting.ids.empty() || posting.ids.capacity() != 0 ||
        !posting.packed.empty() || posting.packed.capacity() != 0) {
      throw std::runtime_error(
          "primary-offset encoder retained hidden payload");
    }
  }
  return posting;
}

struct FlatPairPostings {
  static constexpr size_t npos = std::numeric_limits<size_t>::max();
  PairVectorLayout layout = PairVectorLayout::PairPacked;
  size_t aligned_dim = 0;
  std::vector<uint64_t> keys;         // sorted canonical oriented pair keys
  std::vector<uint64_t> row_offsets;  // support offsets, keys.size() + 1
  std::vector<int32_t> ids;            // pair_packed only
  std::vector<uint32_t> primary_offsets;  // primary_offset only
  std::vector<uint8_t> packed;         // pair_packed only

  size_t size() const { return keys.size(); }

  size_t find_index(uint64_t key) const {
    auto it = std::lower_bound(keys.begin(), keys.end(), key);
    return it != keys.end() && *it == key ? (size_t)(it - keys.begin())
                                          : npos;
  }

  bool contains(uint64_t key) const { return find_index(key) != npos; }

  size_t support(size_t index) const {
    if (index >= keys.size() || row_offsets.size() != keys.size() + 1) {
      throw std::runtime_error("flat pair directory shape mismatch");
    }
    return (size_t)(row_offsets[index + 1] - row_offsets[index]);
  }
};

static void validate_flat_pair_postings_structure(
    const FlatPairPostings& postings) {
  if (postings.aligned_dim == 0 ||
      postings.row_offsets.size() != postings.keys.size() + 1 ||
      postings.row_offsets.empty() || postings.row_offsets.front() != 0 ||
      !std::is_sorted(postings.keys.begin(), postings.keys.end()) ||
      std::adjacent_find(postings.keys.begin(), postings.keys.end()) !=
          postings.keys.end() ||
      !std::is_sorted(postings.row_offsets.begin(),
                      postings.row_offsets.end())) {
    throw std::runtime_error("flat pair directory structural mismatch");
  }
  const size_t memberships = (size_t)postings.row_offsets.back();
  if (postings.layout == PairVectorLayout::PairPacked) {
    if (!postings.primary_offsets.empty() ||
        postings.primary_offsets.capacity() != 0 ||
        postings.ids.size() != memberships ||
        postings.packed.size() != memberships * postings.aligned_dim) {
      throw std::runtime_error("flat pair-packed payload structural mismatch");
    }
  } else if (!postings.ids.empty() || postings.ids.capacity() != 0 ||
             !postings.packed.empty() || postings.packed.capacity() != 0 ||
             postings.primary_offsets.size() != memberships) {
    throw std::runtime_error(
        "flat primary-offset payload structural mismatch");
  }
}

// Production-coupled exact pair scan.  The primary-offset layout factorizes a
// pair view over the already resident, immutable packed primary atom.  The
// recovered global ID still controls deterministic tie-breaking.
static void scan_pair_posting_exact(
    const FlatPairPostings& postings, size_t posting_index,
    const int32_t* primary_global_ids, size_t primary_size,
    const uint8_t* primary_packed, const uint8_t* query_data,
    unsigned query_dim, size_t aligned_dim,
    std::pair<float, int32_t>* frontier, int k_result) {
  if (frontier == nullptr || k_result <= 0 || query_data == nullptr) {
    throw std::runtime_error("invalid exact pair-scan arguments");
  }
  for (int k = 0; k < k_result; ++k) {
    frontier[k] = {std::numeric_limits<float>::max(), -1};
  }
  auto insert = [&](float distance, int32_t global_id) {
    if (distance < frontier[k_result - 1].first ||
        (distance == frontier[k_result - 1].first &&
         global_id < frontier[k_result - 1].second)) {
      int position = k_result - 1;
      while (position > 0 &&
             (frontier[position - 1].first > distance ||
              (frontier[position - 1].first == distance &&
               frontier[position - 1].second > global_id))) {
        frontier[position] = frontier[position - 1];
        --position;
      }
      frontier[position] = {distance, global_id};
    }
  };

  if (posting_index >= postings.size() ||
      postings.aligned_dim != aligned_dim ||
      postings.row_offsets.size() != postings.size() + 1) {
    throw std::runtime_error("flat pair scan directory mismatch");
  }
  const size_t begin = (size_t)postings.row_offsets[posting_index];
  const size_t end = (size_t)postings.row_offsets[posting_index + 1];
  if (end < begin) {
    throw std::runtime_error("flat pair row offsets are not monotone");
  }

  if (postings.layout == PairVectorLayout::PairPacked) {
    if (!postings.primary_offsets.empty() ||
        postings.ids.size() != postings.row_offsets.back() ||
        postings.packed.size() != postings.ids.size() * aligned_dim) {
      throw std::runtime_error("pair-packed posting shape mismatch");
    }
    int32_t previous_global_id = -1;
    for (size_t k = begin; k < end; ++k) {
      if (postings.ids[k] <= previous_global_id) {
        throw std::runtime_error(
            "pair-packed IDs are not strictly increasing");
      }
      previous_global_id = postings.ids[k];
      float distance = l2_sq_uint8_avx2(
          query_data, postings.packed.data() + k * aligned_dim, query_dim);
      insert(distance, postings.ids[k]);
    }
    return;
  }

  if (!postings.ids.empty() || !postings.packed.empty() ||
      postings.primary_offsets.size() != postings.row_offsets.back() ||
      (begin < end &&
       (primary_global_ids == nullptr || primary_packed == nullptr))) {
    throw std::runtime_error("primary-offset posting shape mismatch");
  }
  uint32_t previous_offset = 0;
  bool have_previous_offset = false;
  int32_t previous_global_id = -1;
  for (size_t k = begin; k < end; ++k) {
    uint32_t offset = postings.primary_offsets[k];
    if (offset >= primary_size ||
        (have_previous_offset && offset <= previous_offset)) {
      throw std::runtime_error("primary-offset posting is out of bounds");
    }
    int32_t global_id = primary_global_ids[offset];
    if (global_id <= previous_global_id) {
      throw std::runtime_error(
          "primary-offset global IDs are not strictly increasing");
    }
    previous_offset = offset;
    previous_global_id = global_id;
    have_previous_offset = true;
    float distance = l2_sq_uint8_avx2(
        query_data, primary_packed + (size_t)offset * aligned_dim, query_dim);
    insert(distance, global_id);
  }
}

static int pair_layout_contract_self_test() {
  constexpr unsigned dim = 32;
  constexpr int k_result = 4;
  if (sizeof(FlatPairPostings) != 136) {
    throw std::runtime_error(
        "pair layout self-test: FlatPairPostings ABI accounting changed");
  }
  const int32_t primary_ids[] = {0, 2, 7, 9};
  std::vector<uint8_t> primary_packed(4 * dim, 0);
  for (unsigned d = 0; d < dim; ++d) {
    primary_packed[0 * dim + d] = 3;
    primary_packed[1 * dim + d] = 9;
    primary_packed[2 * dim + d] = 3;  // tie with global 0; ID 0 must win.
    primary_packed[3 * dim + d] = 21;
  }
  std::vector<uint8_t> query(dim, 4);

  auto base_vector = [&](int32_t global_id) -> const uint8_t* {
    for (size_t offset = 0; offset < 4; ++offset) {
      if (primary_ids[offset] == global_id) {
        return primary_packed.data() + offset * dim;
      }
    }
    return nullptr;
  };
  validate_primary_pack_alignment(
      primary_ids, 4, primary_packed.data(), dim, base_vector);
  std::vector<uint8_t> swapped_pack = primary_packed;
  std::swap_ranges(swapped_pack.begin(), swapped_pack.begin() + dim,
                   swapped_pack.begin() + dim);
  bool swapped_rejected = false;
  try {
    validate_primary_pack_alignment(
        primary_ids, 4, swapped_pack.data(), dim, base_vector);
  } catch (const std::runtime_error&) {
    swapped_rejected = true;
  }
  if (!swapped_rejected) {
    throw std::runtime_error(
        "pair layout self-test: swapped primary pack was accepted");
  }
  auto keep = [](int32_t global_id) {
    return global_id == 0 || global_id == 7 || global_id == 9;
  };
  PairPosting encoded_packed = encode_pair_posting(
      5, primary_ids, 4, dim, PairVectorLayout::PairPacked, keep,
      base_vector);
  PairPosting encoded_offset = encode_pair_posting(
      5, primary_ids, 4, dim, PairVectorLayout::PrimaryOffset, keep,
      base_vector);
  if (encoded_offset.primary_offsets !=
          std::vector<uint32_t>({0, 2, 3}) ||
      canonical_pair_from_support(9, 3, 10, 10) !=
          std::make_pair<int32_t, int32_t>(3, 9) ||
      canonical_pair_from_support(3, 9, 10, 10) !=
          std::make_pair<int32_t, int32_t>(3, 9)) {
    throw std::runtime_error(
        "pair layout self-test: encode/orientation invariant failed");
  }

  FlatPairPostings pair_packed;
  pair_packed.layout = PairVectorLayout::PairPacked;
  pair_packed.aligned_dim = dim;
  pair_packed.keys = {17};
  pair_packed.row_offsets = {0, 3};
  pair_packed.ids = std::move(encoded_packed.ids);
  pair_packed.packed = std::move(encoded_packed.packed);

  FlatPairPostings primary_offset;
  primary_offset.layout = PairVectorLayout::PrimaryOffset;
  primary_offset.aligned_dim = dim;
  primary_offset.keys = {17};
  primary_offset.row_offsets = {0, 3};
  primary_offset.primary_offsets =
      std::move(encoded_offset.primary_offsets);
  validate_flat_pair_postings_structure(pair_packed);
  validate_flat_pair_postings_structure(primary_offset);
  std::pair<float, int32_t> packed_frontier[k_result];
  std::pair<float, int32_t> offset_frontier[k_result];
  scan_pair_posting_exact(
      pair_packed, 0, nullptr, 0, nullptr, query.data(), dim, dim,
      packed_frontier, k_result);
  scan_pair_posting_exact(
      primary_offset, 0, primary_ids, 4, primary_packed.data(), query.data(),
      dim, dim, offset_frontier, k_result);
  for (int k = 0; k < k_result; ++k) {
    if (packed_frontier[k] != offset_frontier[k]) {
      throw std::runtime_error(
          "pair layout self-test: packed/offset result mismatch");
    }
  }
  if (offset_frontier[0].second != 0 || offset_frontier[1].second != 7 ||
      offset_frontier[3].second != -1 ||
      !primary_offset.ids.empty() || primary_offset.ids.capacity() != 0 ||
      !primary_offset.packed.empty() ||
      primary_offset.packed.capacity() != 0) {
    throw std::runtime_error(
        "pair layout self-test: tie or zero-copy invariant failed");
  }

  // Local offsets {0,2,3} numerically collide with valid global IDs 0 and 2.
  // A local/global confusion would emit {0,2,3}, not the expected {0,7,9}.
  const int32_t remapped_primary_ids[] = {100, 102, 107, 109};
  scan_pair_posting_exact(
      primary_offset, 0, remapped_primary_ids, 4, primary_packed.data(),
      query.data(), dim, dim, offset_frontier, k_result);
  if (offset_frontier[0].second != 100 ||
      offset_frontier[1].second != 107) {
    throw std::runtime_error(
        "pair layout self-test: primary/global remapping was ignored");
  }
  if (!primary_offset.contains(17) ||
      primary_offset.contains(18) ||
      primary_offset.find_index(18) != FlatPairPostings::npos) {
    throw std::runtime_error(
        "pair layout self-test: foreign/unseen pair lookup was accepted");
  }

  FlatPairPostings out_of_bounds;
  out_of_bounds.layout = PairVectorLayout::PrimaryOffset;
  out_of_bounds.aligned_dim = dim;
  out_of_bounds.keys = {17};
  out_of_bounds.row_offsets = {0, 1};
  out_of_bounds.primary_offsets = {4};
  bool out_of_bounds_rejected = false;
  try {
    scan_pair_posting_exact(
        out_of_bounds, 0, primary_ids, 4, primary_packed.data(), query.data(),
        dim, dim, offset_frontier, k_result);
  } catch (const std::runtime_error&) {
    out_of_bounds_rejected = true;
  }
  if (!out_of_bounds_rejected) {
    throw std::runtime_error(
        "pair layout self-test: out-of-bounds offset was accepted");
  }
  FlatPairPostings duplicate_offset = primary_offset;
  duplicate_offset.primary_offsets = {0, 0, 3};
  bool duplicate_rejected = false;
  try {
    scan_pair_posting_exact(
        duplicate_offset, 0, primary_ids, 4, primary_packed.data(),
        query.data(), dim, dim, offset_frontier, k_result);
  } catch (const std::runtime_error&) {
    duplicate_rejected = true;
  }
  if (!duplicate_rejected) {
    throw std::runtime_error(
        "pair layout self-test: duplicate offset was accepted");
  }
  FlatPairPostings hidden_wrong_layout_copy = primary_offset;
  hidden_wrong_layout_copy.ids.reserve(1);
  hidden_wrong_layout_copy.ids.clear();
  bool hidden_copy_rejected = false;
  try {
    validate_flat_pair_postings_structure(hidden_wrong_layout_copy);
  } catch (const std::runtime_error&) {
    hidden_copy_rejected = true;
  }
  if (!hidden_copy_rejected) {
    throw std::runtime_error(
        "pair layout self-test: hidden wrong-layout capacity was accepted");
  }
  FlatPairPostings empty = primary_offset;
  empty.row_offsets = {0, 0};
  std::vector<uint32_t>().swap(empty.primary_offsets);
  validate_flat_pair_postings_structure(empty);
  scan_pair_posting_exact(
      empty, 0, primary_ids, 4, primary_packed.data(), query.data(), dim, dim,
      offset_frontier, k_result);
  for (int k = 0; k < k_result; ++k) {
    if (offset_frontier[k].second != -1) {
      throw std::runtime_error(
          "pair layout self-test: empty intersection returned an ID");
    }
  }
  for (int malformed_case = 0; malformed_case < 3; ++malformed_case) {
    FlatPairPostings malformed = primary_offset;
    if (malformed_case == 0) {
      malformed.row_offsets = {1, 3};
    } else if (malformed_case == 1) {
      malformed.row_offsets = {0, 2};
    } else {
      malformed.keys = {17, 19};
      malformed.row_offsets = {0, 3, 2};
    }
    bool rejected = false;
    try {
      validate_flat_pair_postings_structure(malformed);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error(
          "pair layout self-test: malformed flat CSR was accepted");
    }
  }
  std::printf("[pair layout self-test] PASS\n");
  return 0;
}

struct NegTask {
  int32_t anchor = -1;
  std::vector<int32_t> neg;
  std::vector<int32_t> qids;
};

struct NegGt {
  bool available = false;
  std::vector<int32_t> ids;  // rows * 10
};

struct NegObservable {
  int64_t anchor_size = 0;
  std::vector<int32_t> survivors;
  double observable_us = 0.0;
};

class StdoutRedirectToStderr {
 public:
  explicit StdoutRedirectToStderr(bool enable) {
    if (!enable) return;
    std::cout.flush();
    std::fflush(stdout);
    saved_fd_ = dup(STDOUT_FILENO);
    if (saved_fd_ >= 0) {
      dup2(STDERR_FILENO, STDOUT_FILENO);
      saved_cout_ = std::cout.rdbuf(std::cerr.rdbuf());
      active_ = true;
    }
  }

  ~StdoutRedirectToStderr() { restore(); }

  void restore() {
    if (!active_) return;
    std::cout.flush();
    std::fflush(stdout);
    dup2(saved_fd_, STDOUT_FILENO);
    close(saved_fd_);
    std::cout.rdbuf(saved_cout_);
    active_ = false;
    saved_fd_ = -1;
    saved_cout_ = nullptr;
  }

 private:
  bool active_ = false;
  int saved_fd_ = -1;
  std::streambuf* saved_cout_ = nullptr;
};

static inline uint64_t pair_key(int32_t a, int32_t b) {
  return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

static inline int32_t pair_key_first(uint64_t key) {
  return (int32_t)(uint32_t)(key >> 32);
}

static inline int32_t pair_key_second(uint64_t key) {
  return (int32_t)(uint32_t)key;
}

static std::unique_ptr<ClusterIndex> load_cluster_index(const std::string& clust_dir, int32_t tag) {
  char cp[1024], mp[1024];
  snprintf(cp, sizeof(cp), "%s/%d_centroids.bin", clust_dir.c_str(), tag);
  snprintf(mp, sizeof(mp), "%s/%d_members.bin",   clust_dir.c_str(), tag);
  if (!std::filesystem::exists(cp) || !std::filesystem::exists(mp)) return nullptr;
  auto ci = std::make_unique<ClusterIndex>();
  ci->tag = tag;
  {
    FILE* f = fopen(cp, "rb");
    size_t got = fread(&ci->n_clusters, sizeof(int32_t), 1, f);
    got = fread(&ci->aligned_dim, sizeof(int32_t), 1, f);
    ci->centroid_data.resize((size_t)ci->n_clusters * ci->aligned_dim);
    got = fread(ci->centroid_data.data(), 1, ci->centroid_data.size(), f);
    (void)got;
    fclose(f);
  }
  {
    FILE* f = fopen(mp, "rb");
    int32_t nc;
    size_t got = fread(&nc, sizeof(int32_t), 1, f);
    ci->member_offsets.resize(nc + 1);
    got = fread(ci->member_offsets.data(), sizeof(int64_t), nc + 1, f);
    int64_t total = ci->member_offsets.back();
    ci->member_ids.resize(total);
    got = fread(ci->member_ids.data(), sizeof(int32_t), total, f);
    (void)got;
    fclose(f);
  }
  return ci;
}

static parlay::sequence<int32_t> load_subset_idx(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return {};
  int32_t n;
  size_t got = fread(&n, sizeof(int32_t), 1, f); (void)got;
  parlay::sequence<int32_t> v(n);
  got = fread(v.data(), sizeof(int32_t), n, f); (void)got;
  fclose(f); return v;
}

// GT format: header [N:uint32, K:uint32], body N*K * (uint32 idx + float32 dist)
struct GroundTruth {
  uint32_t N, K;
  std::vector<uint32_t> indices;     // N*K
  std::vector<float>    distances;   // N*K
};

static GroundTruth load_gt(const std::string& path, uint32_t expected_n,
                           uint32_t expected_k, uint32_t base_size) {
  GroundTruth gt{};
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("GT is not a regular file: " + path);
  }
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    throw std::runtime_error("GT open failed: " + path);
  }
  auto require_read = [&](void* destination, size_t width, size_t count,
                          const char* label) {
    size_t got = fread(destination, width, count, f);
    if (got != count) {
      fclose(f);
      throw std::runtime_error(std::string("truncated GT ") + label + ": " + path);
    }
  };
  require_read(&gt.N, sizeof(uint32_t), 1, "N");
  require_read(&gt.K, sizeof(uint32_t), 1, "K");
  if (gt.N != expected_n || gt.K != expected_k) {
    fclose(f);
    throw std::runtime_error(
        "GT shape mismatch: expected " + std::to_string(expected_n) + "x" +
        std::to_string(expected_k) + ", got " + std::to_string(gt.N) + "x" +
        std::to_string(gt.K));
  }
  const uint64_t cells = static_cast<uint64_t>(gt.N) * gt.K;
  const uint64_t expected_bytes = 8ULL + cells * 8ULL;
  if (std::filesystem::file_size(path) != expected_bytes) {
    fclose(f);
    throw std::runtime_error("GT exact-byte-size mismatch: " + path);
  }
  gt.indices.resize((size_t)gt.N * gt.K);
  gt.distances.resize((size_t)gt.N * gt.K);
  require_read(gt.indices.data(), sizeof(uint32_t), gt.indices.size(), "IDs");
  require_read(gt.distances.data(), sizeof(float), gt.distances.size(), "distances");
  if (fgetc(f) != EOF) {
    fclose(f);
    throw std::runtime_error("trailing GT bytes: " + path);
  }
  if (fclose(f) != 0) {
    throw std::runtime_error("GT close failed: " + path);
  }
  for (uint32_t row = 0; row < gt.N; ++row) {
    bool saw_padding = false;
    std::pair<float, uint32_t> previous{-1.0f, 0};
    bool have_previous = false;
    std::unordered_set<uint32_t> row_ids;
    for (uint32_t column = 0; column < gt.K; ++column) {
      const size_t offset = static_cast<size_t>(row) * gt.K + column;
      const uint32_t id = gt.indices[offset];
      const float distance = gt.distances[offset];
      if (id == std::numeric_limits<uint32_t>::max()) {
        saw_padding = true;
        if (!std::isinf(distance) || distance < 0) {
          throw std::runtime_error("invalid GT padding distance");
        }
        continue;
      }
      if (saw_padding || id >= base_size || !std::isfinite(distance) ||
          distance < 0 || !row_ids.insert(id).second) {
        throw std::runtime_error("invalid GT ID/distance contract");
      }
      const std::pair<float, uint32_t> current{distance, id};
      if (have_previous && current <= previous) {
        throw std::runtime_error(
            "GT row violates strict distance/global-ID ordering or repeats an ID");
      }
      previous = current;
      have_previous = true;
    }
  }
  return gt;
}

class NegJsonParser {
 public:
  explicit NegJsonParser(std::string text) : s_(std::move(text)) {}

  std::vector<NegTask> parse_tasks() {
    std::vector<NegTask> tasks;
    skip_ws();
    expect('[');
    skip_ws();
    if (consume(']')) return tasks;
    while (true) {
      tasks.push_back(parse_task());
      skip_ws();
      if (consume(']')) break;
      expect(',');
    }
    skip_ws();
    if (pos_ != s_.size()) throw std::runtime_error("trailing content after JSON array");
    return tasks;
  }

 private:
  std::string s_;
  size_t pos_ = 0;

  void skip_ws() {
    while (pos_ < s_.size() && std::isspace((unsigned char)s_[pos_])) ++pos_;
  }

  bool consume(char c) {
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char c) {
    if (!consume(c)) {
      std::ostringstream oss;
      oss << "expected '" << c << "' at byte " << pos_;
      throw std::runtime_error(oss.str());
    }
  }

  std::string parse_string() {
    skip_ws();
    expect('"');
    std::string out;
    while (pos_ < s_.size()) {
      char c = s_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= s_.size()) throw std::runtime_error("unterminated string escape");
        char e = s_[pos_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          default: throw std::runtime_error("unsupported string escape");
        }
      } else {
        out.push_back(c);
      }
    }
    throw std::runtime_error("unterminated string");
  }

  int32_t parse_int32() {
    skip_ws();
    bool neg = false;
    if (pos_ < s_.size() && s_[pos_] == '-') {
      neg = true;
      ++pos_;
    }
    if (pos_ >= s_.size() || !std::isdigit((unsigned char)s_[pos_])) {
      throw std::runtime_error("expected integer");
    }
    int64_t val = 0;
    while (pos_ < s_.size() && std::isdigit((unsigned char)s_[pos_])) {
      val = val * 10 + (s_[pos_++] - '0');
      int64_t lim = neg ? -(int64_t)std::numeric_limits<int32_t>::min()
                        : (int64_t)std::numeric_limits<int32_t>::max();
      if (val > lim) throw std::runtime_error("integer out of int32 range");
    }
    int64_t signed_val = neg ? -val : val;
    return (int32_t)signed_val;
  }

  std::vector<int32_t> parse_int_array() {
    std::vector<int32_t> out;
    expect('[');
    skip_ws();
    if (consume(']')) return out;
    while (true) {
      out.push_back(parse_int32());
      skip_ws();
      if (consume(']')) break;
      expect(',');
    }
    return out;
  }

  void skip_number_or_literal() {
    skip_ws();
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == ',' || c == ']' || c == '}' || std::isspace((unsigned char)c)) break;
      ++pos_;
    }
  }

  void skip_value() {
    skip_ws();
    if (pos_ >= s_.size()) throw std::runtime_error("unexpected end while skipping value");
    if (s_[pos_] == '"') {
      (void)parse_string();
    } else if (s_[pos_] == '[') {
      expect('[');
      skip_ws();
      if (consume(']')) return;
      while (true) {
        skip_value();
        skip_ws();
        if (consume(']')) break;
        expect(',');
      }
    } else if (s_[pos_] == '{') {
      expect('{');
      skip_ws();
      if (consume('}')) return;
      while (true) {
        (void)parse_string();
        expect(':');
        skip_value();
        skip_ws();
        if (consume('}')) break;
        expect(',');
      }
    } else {
      skip_number_or_literal();
    }
  }

  NegTask parse_task() {
    NegTask task;
    bool saw_anchor = false, saw_neg = false, saw_qids = false;
    expect('{');
    skip_ws();
    if (consume('}')) throw std::runtime_error("empty negation task");
    while (true) {
      std::string key = parse_string();
      expect(':');
      if (key == "anchor") {
        task.anchor = parse_int32();
        saw_anchor = true;
      } else if (key == "neg") {
        task.neg = parse_int_array();
        saw_neg = true;
      } else if (key == "qids") {
        task.qids = parse_int_array();
        saw_qids = true;
      } else {
        skip_value();
      }
      skip_ws();
      if (consume('}')) break;
      expect(',');
    }
    if (!saw_anchor || !saw_neg || !saw_qids) {
      throw std::runtime_error("negation task missing anchor, neg, or qids");
    }
    if (task.neg.empty()) throw std::runtime_error("negation task has empty neg list");
    if (task.qids.empty()) throw std::runtime_error("negation task has empty qids list");
    return task;
  }
};

static std::vector<NegTask> load_neg_workload_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("could not open BCI_NEG_WORKLOAD=" + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  NegJsonParser parser(ss.str());
  return parser.parse_tasks();
}

static int64_t tag_posting_size(const csr_filters& bmt, int32_t tag) {
  if (tag < 0 || (int64_t)tag + 1 > bmt.n_points) return 0;
  return bmt.row_offsets[tag + 1] - bmt.row_offsets[tag];
}

static bool bitvec_has(const uint64_t* bv, int32_t g) {
  return (bv[g >> 6] & (1ULL << (g & 63))) != 0ULL;
}

static std::unordered_map<int32_t, std::vector<uint64_t>>
build_neg_bitvecs(const std::unordered_set<int32_t>& tags,
                  const csr_filters& bm,
                  const csr_filters& bmt) {
  std::unordered_map<int32_t, std::vector<uint64_t>> bitvecs;
  size_t bv_words = (bm.n_points + 63) / 64;
  bitvecs.reserve(tags.size() * 2 + 1);
  for (int32_t tag : tags) {
    if (tag < 0 || (int64_t)tag + 1 > bmt.n_points) continue;
    auto& bv = bitvecs[tag];
    bv.assign(bv_words, 0ULL);
    int64_t lo = bmt.row_offsets[tag];
    int64_t hi = bmt.row_offsets[tag + 1];
    for (int64_t j = lo; j < hi; ++j) {
      int32_t g = bmt.row_indices[j];
      bv[g >> 6] |= (1ULL << (g & 63));
    }
  }
  return bitvecs;
}

static bool survives_neg_filter(int32_t g,
                                const std::vector<int32_t>& neg_tags,
                                const std::vector<const uint64_t*>& neg_bvs,
                                const csr_filters& bm) {
  for (size_t i = 0; i < neg_tags.size(); ++i) {
    const uint64_t* bv = neg_bvs[i];
    bool has = bv ? bitvec_has(bv, g) : (bool)bm.match(g, neg_tags[i]);
    if (has) return false;
  }
  return true;
}

static NegObservable observe_neg_survivors(const NegTask& task,
                                           const csr_filters& bm,
                                           const csr_filters& bmt,
                                           const std::unordered_map<int32_t, std::vector<uint64_t>>& bitvecs) {
  NegObservable obs;
  obs.anchor_size = tag_posting_size(bmt, task.anchor);
  std::vector<const uint64_t*> neg_bvs;
  neg_bvs.reserve(task.neg.size());
  for (int32_t t : task.neg) {
    auto it = bitvecs.find(t);
    neg_bvs.push_back(it == bitvecs.end() ? nullptr : it->second.data());
  }

  auto t0 = std::chrono::steady_clock::now();
  obs.survivors.reserve((size_t)obs.anchor_size);
  if (obs.anchor_size > 0) {
    int64_t lo = bmt.row_offsets[task.anchor];
    int64_t hi = bmt.row_offsets[task.anchor + 1];
    for (int64_t j = lo; j < hi; ++j) {
      int32_t g = bmt.row_indices[j];
      if (survives_neg_filter(g, task.neg, neg_bvs, bm)) obs.survivors.push_back(g);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  obs.observable_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  return obs;
}

static std::string csv_neg_tags(const std::vector<int32_t>& neg) {
  std::ostringstream oss;
  oss << '"';
  for (size_t i = 0; i < neg.size(); ++i) {
    if (i) oss << '|';
    oss << neg[i];
  }
  oss << '"';
  return oss.str();
}

static NegGt load_optional_neg_gt(const char* raw_path, size_t expected_rows) {
  NegGt gt;
  if (!raw_path || raw_path[0] == '\0') return gt;
  std::string path(raw_path);
  std::error_code ec;
  uintmax_t bytes = std::filesystem::file_size(path, ec);
  if (ec) {
    std::fprintf(stderr, "[neg] GT unavailable, skipping recall: %s\n", path.c_str());
    return gt;
  }
  uintmax_t expected_bytes = expected_rows * 10ULL * sizeof(int32_t);
  if (bytes != expected_bytes) {
    std::fprintf(stderr,
                 "[neg] GT size mismatch for %s: got %llu bytes, expected %llu; skipping recall\n",
                 path.c_str(), (unsigned long long)bytes, (unsigned long long)expected_bytes);
    return gt;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "[neg] GT unreadable, skipping recall: %s\n", path.c_str());
    return gt;
  }
  gt.ids.resize(expected_rows * 10ULL);
  in.read(reinterpret_cast<char*>(gt.ids.data()), (std::streamsize)(gt.ids.size() * sizeof(int32_t)));
  if (!in) {
    std::fprintf(stderr, "[neg] GT read failed, skipping recall: %s\n", path.c_str());
    gt.ids.clear();
    return gt;
  }
  gt.available = true;
  return gt;
}

static double neg_recall10(const std::vector<int32_t>& results,
                           const NegGt& gt,
                           size_t row_idx) {
  if (!gt.available) return std::numeric_limits<double>::quiet_NaN();
  const int32_t* row = gt.ids.data() + row_idx * 10ULL;
  std::vector<int32_t> truth;
  truth.reserve(10);
  for (int i = 0; i < 10; ++i) {
    if (row[i] >= 0) truth.push_back(row[i]);
  }
  if (truth.empty()) {
    return results.empty() ? 1.0 : 0.0;
  }
  std::sort(truth.begin(), truth.end());
  int hit = 0;
  for (int32_t v : results) {
    if (v >= 0 && std::binary_search(truth.begin(), truth.end(), v)) ++hit;
  }
  return (double)hit / (double)truth.size();
}

static std::vector<int32_t> neg_brute_topk(PR& base,
                                           PointT& q,
                                           const std::vector<int32_t>& candidates,
                                           int K) {
  std::vector<std::pair<float, int32_t>> frontier((size_t)K,
      {std::numeric_limits<float>::max(), -1});
  auto heap_insert_brute = [&](float d, int32_t g) {
    if (d < frontier[K - 1].first ||
        (d == frontier[K - 1].first && g < frontier[K - 1].second)) {
      int p = K - 1;
      while (p > 0 && (frontier[p - 1].first > d ||
                       (frontier[p - 1].first == d &&
                        frontier[p - 1].second > g))) {
        frontier[p] = frontier[p - 1];
        --p;
      }
      frontier[p] = {d, g};
    }
  };

  const uint8_t* q_data = q.get();
  unsigned q_dim = (unsigned)base.dimension();
  for (int32_t g : candidates) {
    const uint8_t* bp_data = base[g].get();
    float d = l2_sq_uint8_avx2(q_data, bp_data, q_dim);
    heap_insert_brute(d, g);
  }

  std::vector<int32_t> top;
  top.reserve(K);
  for (int k = 0; k < K; ++k) {
    if (frontier[k].second >= 0) top.push_back(frontier[k].second);
  }
  return top;
}

static std::vector<int32_t> neg_graph_topk(PR& base,
                                           PointT& q,
                                           Shard& sh,
                                           const std::vector<int32_t>& neg_tags,
                                           const std::vector<const uint64_t*>& neg_bvs,
                                           const csr_filters& bm,
                                           int K,
                                           int beam,
                                           int post_filter_pool,
                                           bool exact_scaled,
                                           int64_t anchor_size,
                                           int64_t n_eff) {
  ThinSubPR sub_pr(base, sh.subset);
  double cut_val = std::getenv("BCI_HAMCG_CUT") ? std::atof(std::getenv("BCI_HAMCG_CUT")) : 1.35;
  long graph_k = 0;  // untrimmed beam frontier; beam width controls candidate pool
  long graph_beam = beam;
  if (exact_scaled && n_eff > 0 && anchor_size > 0) {
    double scale = (double)anchor_size / (double)n_eff;
    int64_t scaled_pool = (int64_t)std::ceil((double)post_filter_pool * scale);
    scaled_pool = std::max<int64_t>(scaled_pool, K);
    scaled_pool = std::min<int64_t>(scaled_pool, (int64_t)sh.graph.size());
    graph_beam = std::max<long>((long)beam, (long)scaled_pool);
  }
  long bounded_limit = std::min<long>((long)sh.graph.size(), (long)std::max<long>(100L * beam, 100000L));
  if (exact_scaled) {
    bounded_limit = std::min<long>((long)sh.graph.size(), std::max<long>(bounded_limit, 4L * graph_beam));
  }
  QueryParams QP(graph_k, graph_beam, /*cut=*/cut_val, bounded_limit, sh.maxDeg);
  auto res = beam_search<PointT, ThinSubPR, Indx>(
      q, sh.graph, sub_pr, /*start=*/0, QP);
  auto& frontier = res.first.first;

  std::vector<std::pair<float, int32_t>> cands;
  cands.reserve(std::min<size_t>(frontier.size(), (size_t)std::max(K, post_filter_pool)));
  for (size_t j = 0; j < frontier.size(); ++j) {
    int32_t local = frontier[j].first;
    float dist = frontier[j].second;
    int32_t global = sh.subset[local];
    if (survives_neg_filter(global, neg_tags, neg_bvs, bm)) {
      cands.push_back({dist, global});
    }
  }

  int kk = std::min<int>(K, (int)cands.size());
  if (kk > 0) {
    std::partial_sort(cands.begin(), cands.begin() + kk, cands.end(),
      [](auto& a, auto& b) {
        return a.first < b.first ||
               (a.first == b.first && a.second < b.second);
      });
  }
  std::vector<int32_t> top;
  top.reserve(K);
  for (int i = 0; i < kk; ++i) top.push_back(cands[i].second);
  return top;
}

static int run_neg_workload(const std::string& workload_path,
                            const std::string& planner,
                            PR& base,
                            PR& query,
                            const csr_filters& bm,
                            const csr_filters& bmt,
                            std::unordered_map<int32_t, std::unique_ptr<Shard>>& shards,
                            int K,
                            int beam,
                            int post_filter_pool,
                            int64_t brute_conj_thresh) {
  if (planner != "exact" && planner != "mono" && planner != "indep") {
    std::fprintf(stderr, "[neg] unsupported BCI_PLANNER=%s (expected exact|mono|indep)\n",
                 planner.c_str());
    return 1;
  }

  std::vector<NegTask> tasks;
  try {
    tasks = load_neg_workload_json(workload_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[neg] workload parse error: %s\n", e.what());
    return 1;
  }

  size_t total_rows = 0;
  std::unordered_set<int32_t> bitvec_tags;
  for (const auto& task : tasks) {
    total_rows += task.qids.size();
    for (int32_t t : task.neg) bitvec_tags.insert(t);
  }

  auto t_bv0 = std::chrono::steady_clock::now();
  auto neg_bitvecs = build_neg_bitvecs(bitvec_tags, bm, bmt);
  auto t_bv1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "[neg] loaded %zu tasks, built %zu neg bitvectors in %.2fs\n",
               tasks.size(), neg_bitvecs.size(),
               std::chrono::duration<double>(t_bv1 - t_bv0).count());

  NegGt neg_gt = load_optional_neg_gt(std::getenv("BCI_NEG_GT"), total_rows);

  std::ofstream out_file;
  std::ostream* out = &std::cout;
  const char* out_path = std::getenv("BCI_NEG_OUT");
  if (out_path && out_path[0] != '\0') {
    out_file.open(out_path);
    if (!out_file) {
      std::fprintf(stderr, "[neg] could not open BCI_NEG_OUT=%s\n", out_path);
      return 1;
    }
    out = &out_file;
  }

  *out << "qid,anchor,neg_tags,anchor_size,n_eff,r,planner,route,observable_us,search_us,total_us,n_returned,recall10\n";
  out->setf(std::ios::fixed);
  *out << std::setprecision(6);

  size_t row_idx = 0;
  for (const auto& task : tasks) {
    if (task.anchor < 0 || (int64_t)task.anchor + 1 > bmt.n_points) {
      std::fprintf(stderr, "[neg] anchor tag out of range: %d\n", task.anchor);
    }
    for (int32_t qid : task.qids) {
      if (qid < 0 || (int64_t)qid >= query.size()) {
        std::fprintf(stderr, "[neg] qid out of range: %d\n", qid);
        return 1;
      }
    }

    NegObservable obs = observe_neg_survivors(task, bm, bmt, neg_bitvecs);
    int64_t n_eff = (int64_t)obs.survivors.size();
    double r = obs.anchor_size > 0
             ? 1.0 - ((double)n_eff / (double)obs.anchor_size)
             : 1.0;

    bool route_brute = false;
    bool route_empty = false;
    if (planner == "exact") {
      if (n_eff == 0) route_empty = true;
      else route_brute = (n_eff <= brute_conj_thresh);
    } else if (planner == "mono") {
      route_brute = (obs.anchor_size <= brute_conj_thresh);
    } else {
      double n_hat = (double)obs.anchor_size;
      for (int32_t t : task.neg) {
        double f = (double)tag_posting_size(bmt, t);
        n_hat *= (1.0 - f / (double)bm.n_points);
      }
      route_brute = (n_hat <= (double)brute_conj_thresh);
    }

    std::vector<const uint64_t*> neg_bvs;
    neg_bvs.reserve(task.neg.size());
    for (int32_t t : task.neg) {
      auto it = neg_bitvecs.find(t);
      neg_bvs.push_back(it == neg_bitvecs.end() ? nullptr : it->second.data());
    }

    for (int32_t qid : task.qids) {
      PointT q = query[qid];
      std::vector<int32_t> top;
      const char* route = "graph";
      double search_us = 0.0;
      auto t_s0 = std::chrono::steady_clock::now();
      if (route_empty) {
        route = "empty";
      } else if (route_brute) {
        route = "brute";
        top = neg_brute_topk(base, q, obs.survivors, K);
      } else {
        auto sh_it = shards.find(task.anchor);
        if (sh_it == shards.end()) {
          std::fprintf(stderr,
                       "[neg] missing shard for graph-routed anchor %d; falling back to exact brute\n",
                       task.anchor);
          route = "brute";
          top = neg_brute_topk(base, q, obs.survivors, K);
        } else {
          route = "graph";
          bool exact_scaled = (planner == "exact");
          top = neg_graph_topk(base, q, *sh_it->second, task.neg, neg_bvs, bm, K, beam,
                               post_filter_pool, exact_scaled, obs.anchor_size, n_eff);
        }
      }
      auto t_s1 = std::chrono::steady_clock::now();
      search_us = std::chrono::duration<double, std::micro>(t_s1 - t_s0).count();
      double total_us = obs.observable_us + search_us;
      double recall = neg_recall10(top, neg_gt, row_idx);

      *out << qid << ','
           << task.anchor << ','
           << csv_neg_tags(task.neg) << ','
           << obs.anchor_size << ','
           << n_eff << ','
           << r << ','
           << planner << ','
           << route << ','
           << obs.observable_us << ','
           << search_us << ','
           << total_us << ','
           << top.size() << ',';
      if (std::isnan(recall)) *out << "NaN\n";
      else *out << recall << '\n';
      ++row_idx;
    }
  }

  return 0;
}

int main(int argc, char** argv) {
#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
  const char* exact_radix_self_test_raw =
      std::getenv("BCI_EXACT_RADIX_SELF_TEST");
  if (exact_radix_self_test_raw != nullptr &&
      std::string(exact_radix_self_test_raw) == "1") {
    return exact_radix_distance_self_test();
  }
#endif
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
  const char* residual_landmark_self_test_raw =
      std::getenv("BCI_RESIDUAL_LANDMARK_SELF_TEST");
  if (residual_landmark_self_test_raw != nullptr &&
      std::string(residual_landmark_self_test_raw) == "1") {
    return residual_landmark_self_test();
  }
#endif
#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
  const char* bitplane_self_test_raw =
      std::getenv("BCI_BITPLANE_LB_SELF_TEST");
  if (bitplane_self_test_raw != nullptr &&
      std::string(bitplane_self_test_raw) == "1") {
    return bitplane_lower_bound_self_test();
  }
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
  const char* q4_self_test_raw =
      std::getenv("BCI_Q4_GRAPH_SELF_TEST");
  if (q4_self_test_raw != nullptr &&
      std::string(q4_self_test_raw) == "1") {
    return q4_graph_distance_self_test();
  }
#endif
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  const char* eviction_spill_self_test_raw =
      std::getenv("BCI_EVICTION_SPILL_SELF_TEST");
  if (eviction_spill_self_test_raw != nullptr &&
      std::string(eviction_spill_self_test_raw) == "1") {
    return eviction_spill_self_test();
  }
#endif
#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
  const char* shortcut_residual_self_test_raw =
      std::getenv("BCI_SHORTCUT_RESIDUAL_SELF_TEST");
  if (shortcut_residual_self_test_raw != nullptr &&
      std::string(shortcut_residual_self_test_raw) == "1") {
    return shortcut_residual_self_test();
  }
#endif
#ifdef BCI_ENABLE_TWO_VIEW_R32_DIAGNOSTIC
  const char* two_view_r32_self_test_raw =
      std::getenv("BCI_TWO_VIEW_R32_SELF_TEST");
  if (two_view_r32_self_test_raw != nullptr &&
      std::string(two_view_r32_self_test_raw) == "1") {
    return two_view_r32_self_test();
  }
#endif
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
  const char* r96_degree4_sidecar_self_test_raw =
      std::getenv("BCI_R96_DEGREE4_SIDECAR_SELF_TEST");
  if (r96_degree4_sidecar_self_test_raw != nullptr &&
      std::string(r96_degree4_sidecar_self_test_raw) == "1") {
    return r96_degree4_sidecar_self_test();
  }
#endif
#ifdef BCI_ENABLE_BATCHED_MULTI_START_DIAGNOSTIC
  const char* batched_multi_start_self_test_raw =
      std::getenv("BCI_BATCHED_MULTI_START_SELF_TEST");
  if (batched_multi_start_self_test_raw != nullptr &&
      std::string(batched_multi_start_self_test_raw) == "1") {
    return batched_multi_start_self_test();
  }
#endif
#ifdef BCI_ENABLE_FIXED_FRONTIER_BATCHED_DIAGNOSTIC
  const char* fixed_frontier_self_test_raw =
      std::getenv("BCI_FIXED_FRONTIER_BATCHED_SELF_TEST");
  if (fixed_frontier_self_test_raw != nullptr &&
      std::string(fixed_frontier_self_test_raw) == "1") {
    return fixed_frontier_batched_self_test();
  }
#endif
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
  const char* cached_replay_self_test_raw =
      std::getenv("BCI_CACHED_REPLAY_SELF_TEST");
  if (cached_replay_self_test_raw != nullptr &&
      std::string(cached_replay_self_test_raw) == "1") {
    return cached_replay_self_test();
  }
#endif
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
  const char* top10_stability_self_test_raw =
      std::getenv("BCI_TOP10_STABILITY_SELF_TEST");
  if (top10_stability_self_test_raw != nullptr &&
      std::string(top10_stability_self_test_raw) == "1") {
    return top10_stability_self_test();
  }
#endif
  const char* single_beam_self_test_raw =
      std::getenv("BCI_SINGLE_BEAM_SELF_TEST");
  if (single_beam_self_test_raw != nullptr &&
      std::string(single_beam_self_test_raw) == "1") {
    return single_adaptive_beam_self_test();
  }
  const char* pair_layout_self_test_raw =
      std::getenv("BCI_PAIR_LAYOUT_SELF_TEST");
  if (pair_layout_self_test_raw != nullptr &&
      std::string(pair_layout_self_test_raw) == "1") {
    return pair_layout_contract_self_test();
  }
  const char* neg_workload_raw = std::getenv("BCI_NEG_WORKLOAD");
  const bool NEG_MODE = neg_workload_raw != nullptr && neg_workload_raw[0] != '\0';
  StdoutRedirectToStderr neg_stdout_redirect(NEG_MODE);

  const std::string DATA = env_path_or_default(
      "BCI_YFCC_DATA_ROOT",
      "./data/yfcc10m",
      true);
  const std::string DRT  = env_path_or_default(
      "BCI_DRT_ROOT",
      "./indexes/yfcc10m",
      false);

  // Args: qid_lo qid_hi beam K
  int qid_lo = argc > 1 ? atoi(argv[1]) : 60000;
  int qid_hi = argc > 2 ? atoi(argv[2]) : 70000;
  int beam   = argc > 3 ? atoi(argv[3]) : 128;
  int K      = argc > 4 ? atoi(argv[4]) : 10;
  int post_filter_pool = argc > 5 ? atoi(argv[5]) : 4 * beam;
  int64_t brute_conj_thresh = argc > 6 ? atoll(argv[6]) : 200000;
  int n_runs = argc > 7 ? atoi(argv[7]) : 1;  // run query batch N times for warm-cache
  int use_clusters = argc > 8 ? atoi(argv[8]) : 0;  // 0=disable, 1=use IVF² cluster path for conj
  int nprobe = argc > 9 ? atoi(argv[9]) : 3;        // top-N centroids to probe
  int target_pts = argc > 10 ? atoi(argv[10]) : 15000;  // PIVF default
  int use_bitvec = argc > 11 ? atoi(argv[11]) : 1;  // 0=disable bitvec, use bm.match (faster but lower recall)
  int use_pach   = argc > 12 ? atoi(argv[12]) : 1;  // 0=no PACH (baseline), 1=PACH cluster pruning ON
  const double hamcg_cut = std::getenv("BCI_HAMCG_CUT") ?
      std::atof(std::getenv("BCI_HAMCG_CUT")) : 1.35;
  int single_pool = std::getenv("BCI_SINGLE_POOL") ?
      std::atoi(std::getenv("BCI_SINGLE_POOL")) : K;
  if (single_pool < K) single_pool = K;
  const int single_high_beam = std::getenv("BCI_SINGLE_HIGH_BEAM") ?
      std::atoi(std::getenv("BCI_SINGLE_HIGH_BEAM")) : beam;
  const double single_high_cut = std::getenv("BCI_SINGLE_HIGH_CUT") ?
      std::atof(std::getenv("BCI_SINGLE_HIGH_CUT")) : hamcg_cut;
  const int64_t single_high_beam_min_support =
      std::getenv("BCI_SINGLE_HIGH_BEAM_MIN_SUPPORT") ?
      std::atoll(std::getenv("BCI_SINGLE_HIGH_BEAM_MIN_SUPPORT")) : 0;
  const int single_risk_beam = std::getenv("BCI_SINGLE_RISK_BEAM") ?
      std::atoi(std::getenv("BCI_SINGLE_RISK_BEAM")) : single_high_beam;
  const double single_risk_cut = std::getenv("BCI_SINGLE_RISK_CUT") ?
      std::atof(std::getenv("BCI_SINGLE_RISK_CUT")) : single_high_cut;
  const int single_multi_start = std::getenv("BCI_SINGLE_MULTI_START") ?
      std::atoi(std::getenv("BCI_SINGLE_MULTI_START")) : 1;
  const int64_t single_multi_start_min_support =
      std::getenv("BCI_SINGLE_MULTI_START_MIN_SUPPORT") ?
      std::atoll(std::getenv("BCI_SINGLE_MULTI_START_MIN_SUPPORT")) : 0;
  const char* single_entry_mode_raw =
      std::getenv("BCI_SINGLE_ENTRY_MODE");
  const std::string single_entry_mode =
      (single_entry_mode_raw != nullptr &&
       single_entry_mode_raw[0] != '\0')
          ? single_entry_mode_raw
          : "zero";
  const int single_entry_centroids =
      std::getenv("BCI_SINGLE_ENTRY_CENTROIDS")
          ? std::atoi(std::getenv("BCI_SINGLE_ENTRY_CENTROIDS"))
          : 1;
  const double single_retry_boundary_ratio =
      std::getenv("BCI_SINGLE_RETRY_BOUNDARY_RATIO") ?
      std::atof(std::getenv("BCI_SINGLE_RETRY_BOUNDARY_RATIO")) : 0.0;
  const int single_retry_beam = std::getenv("BCI_SINGLE_RETRY_BEAM") ?
      std::atoi(std::getenv("BCI_SINGLE_RETRY_BEAM")) : beam;
  const double single_retry_cut = std::getenv("BCI_SINGLE_RETRY_CUT") ?
      std::atof(std::getenv("BCI_SINGLE_RETRY_CUT")) : hamcg_cut;
  const int64_t single_retry_min_dist_comps =
      std::getenv("BCI_SINGLE_RETRY_MIN_DIST_COMPS") ?
      std::atoll(std::getenv("BCI_SINGLE_RETRY_MIN_DIST_COMPS")) : 0;
  const int64_t single_retry_max_dist_comps =
      std::getenv("BCI_SINGLE_RETRY_MAX_DIST_COMPS") ?
      std::atoll(std::getenv("BCI_SINGLE_RETRY_MAX_DIST_COMPS")) :
      std::numeric_limits<int64_t>::max();
  const int64_t single_retry_min_support =
      std::getenv("BCI_SINGLE_RETRY_MIN_SUPPORT") ?
      std::atoll(std::getenv("BCI_SINGLE_RETRY_MIN_SUPPORT")) : 0;
  const int64_t single_retry_max_support =
      std::getenv("BCI_SINGLE_RETRY_MAX_SUPPORT") ?
      std::atoll(std::getenv("BCI_SINGLE_RETRY_MAX_SUPPORT")) :
      std::numeric_limits<int64_t>::max();
  const bool single_retry_distance_cache =
      std::getenv("BCI_SINGLE_RETRY_DISTANCE_CACHE") != nullptr &&
      std::atoi(std::getenv("BCI_SINGLE_RETRY_DISTANCE_CACHE")) != 0;
  const double single_exact_retry_boundary_ratio =
      std::getenv("BCI_SINGLE_EXACT_RETRY_BOUNDARY_RATIO") ?
      std::atof(std::getenv("BCI_SINGLE_EXACT_RETRY_BOUNDARY_RATIO")) : 0.0;
  const int64_t single_exact_retry_max_support =
      std::getenv("BCI_SINGLE_EXACT_RETRY_MAX_SUPPORT") ?
      std::atoll(std::getenv("BCI_SINGLE_EXACT_RETRY_MAX_SUPPORT")) : 0;
  const int64_t single_exact_retry_max_dist_comps =
      std::getenv("BCI_SINGLE_EXACT_RETRY_MAX_DIST_COMPS") ?
      std::atoll(std::getenv("BCI_SINGLE_EXACT_RETRY_MAX_DIST_COMPS")) :
      std::numeric_limits<int64_t>::max();
  const int single_exact_retry_pack =
      std::getenv("BCI_SINGLE_EXACT_RETRY_PACK") ?
      std::atoi(std::getenv("BCI_SINGLE_EXACT_RETRY_PACK")) : 1;
  const double single_alt_retry_boundary_ratio =
      std::getenv("BCI_SINGLE_ALT_RETRY_BOUNDARY_RATIO") ?
      std::atof(std::getenv("BCI_SINGLE_ALT_RETRY_BOUNDARY_RATIO")) : 0.0;
  const int64_t single_alt_retry_max_support =
      std::getenv("BCI_SINGLE_ALT_RETRY_MAX_SUPPORT") ?
      std::atoll(std::getenv("BCI_SINGLE_ALT_RETRY_MAX_SUPPORT")) : 0;
  if (single_high_beam < beam) {
    throw std::runtime_error("BCI_SINGLE_HIGH_BEAM must be >= base beam");
  }
  if (!(single_high_cut > 0.0) || !std::isfinite(single_high_cut)) {
    throw std::runtime_error("BCI_SINGLE_HIGH_CUT must be finite and positive");
  }
  if (single_risk_beam < beam || !(single_risk_cut > 0.0) ||
      !std::isfinite(single_risk_cut)) {
    throw std::runtime_error(
        "BCI_SINGLE_RISK_BEAM/CUT must be at least the base beam and "
        "finite positive");
  }
  if (single_multi_start < 1 || single_multi_start > 2) {
    throw std::runtime_error("BCI_SINGLE_MULTI_START must be 1 or 2");
  }
  if (single_entry_mode != "zero" &&
      single_entry_mode != "nearest_centroid") {
    throw std::runtime_error(
        "BCI_SINGLE_ENTRY_MODE must be zero or nearest_centroid");
  }
  if (single_entry_mode == "nearest_centroid" &&
      single_multi_start != 1) {
    throw std::runtime_error(
        "nearest-centroid single entry must be isolated from multi-start");
  }
  if (single_entry_centroids < 1 || single_entry_centroids > 16 ||
      (single_entry_mode == "zero" && single_entry_centroids != 1)) {
    throw std::runtime_error(
        "BCI_SINGLE_ENTRY_CENTROIDS must be one in zero mode and in [1,16] "
        "in nearest-centroid mode");
  }
  if (single_multi_start > 1 && single_multi_start_min_support <= 0) {
    throw std::runtime_error(
        "BCI_SINGLE_MULTI_START_MIN_SUPPORT must be positive when enabled");
  }
  if (single_retry_boundary_ratio < 0.0 ||
      !std::isfinite(single_retry_boundary_ratio)) {
    throw std::runtime_error(
        "BCI_SINGLE_RETRY_BOUNDARY_RATIO must be finite and nonnegative");
  }
  if (single_retry_boundary_ratio > 0.0 &&
      (single_retry_beam < beam || !(single_retry_cut > 0.0) ||
       !std::isfinite(single_retry_cut) ||
       single_retry_min_dist_comps < 0 ||
       single_retry_max_dist_comps < single_retry_min_dist_comps ||
       single_retry_min_support < 0 ||
       single_retry_max_support < single_retry_min_support)) {
    throw std::runtime_error("invalid single-search retry parameters");
  }
  if (single_exact_retry_boundary_ratio < 0.0 ||
      !std::isfinite(single_exact_retry_boundary_ratio) ||
      (single_exact_retry_boundary_ratio > 0.0 &&
       (single_exact_retry_max_support <= 0 ||
        single_exact_retry_max_dist_comps <= 0))) {
    throw std::runtime_error("invalid single exact-retry parameters");
  }
  if (single_exact_retry_pack != 0 &&
      single_exact_retry_pack != 1) {
    throw std::runtime_error(
        "BCI_SINGLE_EXACT_RETRY_PACK must be 0 or 1");
  }
  if (single_alt_retry_boundary_ratio < 0.0 ||
      !std::isfinite(single_alt_retry_boundary_ratio) ||
      (single_alt_retry_boundary_ratio > 0.0 &&
       single_alt_retry_max_support <= 0)) {
    throw std::runtime_error("invalid single alternate-retry parameters");
  }
  const int active_retry_modes =
      (single_alt_retry_boundary_ratio > 0.0);
  if ((active_retry_modes > 0 &&
       (single_retry_boundary_ratio > 0.0 ||
        single_exact_retry_boundary_ratio > 0.0)) ||
      (single_multi_start > 1 &&
       single_alt_retry_boundary_ratio > 0.0)) {
    throw std::runtime_error(
        "alternate-entry retry must be isolated from graph/exact retry");
  }
  const int expected_graph_degree = std::getenv("BCI_EXPECT_GRAPH_DEGREE") ?
      std::atoi(std::getenv("BCI_EXPECT_GRAPH_DEGREE")) : 0;
  const int query_graph_degree_limit =
      std::getenv("BCI_QUERY_GRAPH_DEGREE_LIMIT") ?
      std::atoi(std::getenv("BCI_QUERY_GRAPH_DEGREE_LIMIT")) : 0;
  if (query_graph_degree_limit < 0) {
    throw std::runtime_error(
        "BCI_QUERY_GRAPH_DEGREE_LIMIT must be nonnegative");
  }
  const bool reuse_beam_workspace =
      std::getenv("BCI_REUSE_BEAM_WORKSPACE") != nullptr &&
      std::atoi(std::getenv("BCI_REUSE_BEAM_WORKSPACE")) != 0;
  const bool dual_heap_beam =
      std::getenv("BCI_DUAL_HEAP_BEAM") != nullptr &&
      std::atoi(std::getenv("BCI_DUAL_HEAP_BEAM")) != 0;
  const bool indexed_heap_beam =
      std::getenv("BCI_INDEXED_HEAP_BEAM") != nullptr &&
      std::atoi(std::getenv("BCI_INDEXED_HEAP_BEAM")) != 0;
  const bool batched_l2_beam =
      std::getenv("BCI_BATCHED_L2_BEAM") != nullptr &&
      std::atoi(std::getenv("BCI_BATCHED_L2_BEAM")) != 0;
  const bool r96_degree4_sidecar =
      std::getenv("BCI_R96_DEGREE4_SIDECAR") != nullptr &&
      std::atoi(std::getenv("BCI_R96_DEGREE4_SIDECAR")) != 0;
  const char* r96_degree4_sidecar_root_raw =
      std::getenv("BCI_R96_DEGREE4_SIDECAR_ROOT");
  const bool two_view_r32 =
      std::getenv("BCI_TWO_VIEW_R32") != nullptr &&
      std::atoi(std::getenv("BCI_TWO_VIEW_R32")) != 0;
  const bool two_view_primary_full =
      std::getenv("BCI_TWO_VIEW_R32_PRIMARY_FULL") != nullptr &&
      std::atoi(
          std::getenv("BCI_TWO_VIEW_R32_PRIMARY_FULL")) != 0;
  const int two_view_escape_offset =
      std::getenv("BCI_TWO_VIEW_R32_ESCAPE_OFFSET") != nullptr
          ? std::atoi(
                std::getenv("BCI_TWO_VIEW_R32_ESCAPE_OFFSET"))
          : 1;
  const bool fixed_frontier_batched_l2 =
      std::getenv("BCI_FIXED_FRONTIER_BATCHED_L2") != nullptr &&
      std::atoi(
          std::getenv("BCI_FIXED_FRONTIER_BATCHED_L2")) != 0;
  const bool eviction_spill =
      std::getenv("BCI_EVICTION_SPILL") != nullptr &&
      std::atoi(std::getenv("BCI_EVICTION_SPILL")) != 0;
  const bool q4_graph =
      std::getenv("BCI_Q4_GRAPH") != nullptr &&
      std::atoi(std::getenv("BCI_Q4_GRAPH")) != 0;
  const bool bitplane_lb =
      std::getenv("BCI_BITPLANE_LB") != nullptr &&
      std::atoi(std::getenv("BCI_BITPLANE_LB")) != 0;
  const int bitplane_high_bits =
      std::getenv("BCI_BITPLANE_HIGH_BITS") != nullptr
          ? std::atoi(std::getenv("BCI_BITPLANE_HIGH_BITS"))
          : 4;
  const bool exact_radix =
      std::getenv("BCI_EXACT_RADIX") != nullptr &&
      std::atoi(std::getenv("BCI_EXACT_RADIX")) != 0;
  const char* exact_engine_mode_raw =
      std::getenv("BCI_EXACT_ENGINE_MODE");
  const std::string exact_engine_mode =
      exact_engine_mode_raw != nullptr &&
              exact_engine_mode_raw[0] != '\0'
          ? exact_engine_mode_raw
          : "fourway_radix";
  const bool shortcut_residual =
      std::getenv("BCI_SHORTCUT_RESIDUAL") != nullptr &&
      std::atoi(std::getenv("BCI_SHORTCUT_RESIDUAL")) != 0;
  const int shortcut_residual_tag =
      std::getenv("BCI_SHORTCUT_RESIDUAL_TAG") != nullptr
          ? std::atoi(std::getenv("BCI_SHORTCUT_RESIDUAL_TAG"))
          : -1;
  const char* shortcut_residual_mask_path =
      std::getenv("BCI_SHORTCUT_RESIDUAL_MASK");
  const bool residual_landmark_crossfit =
      std::getenv("BCI_RESIDUAL_LANDMARK_CROSSFIT") != nullptr &&
      std::atoi(std::getenv("BCI_RESIDUAL_LANDMARK_CROSSFIT")) != 0;
  const char* residual_landmark_coverage_path =
      std::getenv("BCI_RESIDUAL_LANDMARK_COVERAGE");
  const int batched_l2_prefix_dimensions =
      std::getenv("BCI_BATCHED_L2_PREFIX_DIMS") != nullptr
          ? std::atoi(std::getenv("BCI_BATCHED_L2_PREFIX_DIMS"))
          : 0;
  if (batched_l2_prefix_dimensions != 0 &&
      batched_l2_prefix_dimensions != 64 &&
      batched_l2_prefix_dimensions != 128) {
    throw std::runtime_error(
        "BCI_BATCHED_L2_PREFIX_DIMS must be 0, 64, or 128");
  }
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
  if (!r96_degree4_sidecar ||
      r96_degree4_sidecar_root_raw == nullptr ||
      r96_degree4_sidecar_root_raw[0] == '\0' ||
      !batched_l2_beam || fixed_frontier_batched_l2 ||
      dual_heap_beam || indexed_heap_beam ||
      reuse_beam_workspace || two_view_r32 ||
      beam != 64 || K != 10 || single_pool != 10 ||
      single_high_beam != 64 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      expected_graph_degree != 64 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != 64) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "R96 degree4 sidecar treatment requires the immutable ordinary "
        "R64/b64 prefix64 primary, K=single_pool=10, one zero entry, "
        "no adaptive/retry/alternate graph engine, and a sidecar root");
  }
#else
  if (r96_degree4_sidecar ||
      (r96_degree4_sidecar_root_raw != nullptr &&
       r96_degree4_sidecar_root_raw[0] != '\0')) {
    throw std::runtime_error(
        "R96 degree4 sidecar controls require the isolated treatment target");
  }
#endif
#ifdef BCI_ENABLE_TWO_VIEW_R32_DIAGNOSTIC
  if (two_view_r32 &&
      (!batched_l2_beam || fixed_frontier_batched_l2 ||
       beam != 64 || K != 10 || single_pool != 10 ||
       single_high_beam != 64 ||
       std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
       (two_view_primary_full &&
        (two_view_escape_offset < 0 ||
         two_view_escape_offset > 1)) ||
       (query_graph_degree_limit != 0 &&
        query_graph_degree_limit != 64))) {
    throw std::runtime_error(
        "two-view R32 diagnostic requires ordinary batched-L2, "
        "beam=64, K=single_pool=10, no adaptive high beam/cut, "
        "and full R64 row visibility");
  }
#else
  if (two_view_r32) {
    throw std::runtime_error(
        "BCI_TWO_VIEW_R32 requires the isolated two-view diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
  if (!shortcut_residual || shortcut_residual_tag != 35 ||
      shortcut_residual_mask_path == nullptr ||
      shortcut_residual_mask_path[0] == '\0' ||
      !batched_l2_beam || fixed_frontier_batched_l2 ||
      two_view_r32 || beam != 64 || K != 10 ||
      single_pool != 10 || single_high_beam != 64 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != 64) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "shortcut residual diagnostic requires tag35, ordinary "
        "batched-L2 primary b64, secondary b32, K=single_pool=10, "
        "prefix64, zero entry, one start, full row visibility, and "
        "all retry/two-view modes disabled");
  }
#else
  if (shortcut_residual ||
      shortcut_residual_tag != -1 ||
      shortcut_residual_mask_path != nullptr) {
    throw std::runtime_error(
        "BCI_SHORTCUT_RESIDUAL requires the isolated diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
  if (!residual_landmark_crossfit ||
      residual_landmark_coverage_path == nullptr ||
      residual_landmark_coverage_path[0] == '\0' ||
      qid_lo != 30000 || qid_hi != 40000 ||
      !batched_l2_beam || fixed_frontier_batched_l2 ||
      two_view_r32 || beam != 64 || K != 10 ||
      single_pool != 10 || single_high_beam != 64 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != 64) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "residual-landmark diagnostic requires q30--40, ordinary "
        "batched-L2 primary b64, K=single_pool=10, prefix64, zero "
        "entry, one configured start, full R64 rows, and all retry/"
        "two-view modes disabled");
  }
#else
  if (residual_landmark_crossfit ||
      residual_landmark_coverage_path != nullptr) {
    throw std::runtime_error(
        "BCI_RESIDUAL_LANDMARK_CROSSFIT requires the isolated "
        "residual-landmark diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  if (!eviction_spill ||
      !batched_l2_beam || fixed_frontier_batched_l2 ||
      two_view_r32 || beam != 64 || K != 10 ||
      single_pool != 10 || single_high_beam != 64 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != 64) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "eviction-spill diagnostic requires ordinary batched-L2 "
        "b64, K=single_pool=10, prefix64, zero entry, one start, "
        "full R64 rows, and all retry/two-view modes disabled");
  }
#else
  if (eviction_spill) {
    throw std::runtime_error(
        "BCI_EVICTION_SPILL requires the isolated diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
  if (!q4_graph || !batched_l2_beam ||
      fixed_frontier_batched_l2 || two_view_r32 ||
      beam != 256 || K != 10 || single_pool != 10 ||
      single_high_beam != 256 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != expected_graph_degree) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "q4 graph diagnostic requires ordinary batched-L2 b256, "
        "K=single_pool=10, prefix64, zero entry, one start, full "
        "graph rows, and all retry/two-view modes disabled");
  }
#else
  if (q4_graph) {
    throw std::runtime_error(
        "BCI_Q4_GRAPH requires the isolated q4 diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
  if (!bitplane_lb || !batched_l2_beam ||
      fixed_frontier_batched_l2 || two_view_r32 ||
      beam != 256 || K != 10 || single_pool != 10 ||
      single_high_beam != 256 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      bitplane_high_bits < 1 || bitplane_high_bits > 7 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != expected_graph_degree) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "bit-plane lower-bound diagnostic requires ordinary "
        "batched-L2 b256, K=single_pool=10, prefix64, zero entry, "
        "one start, full graph rows, retained bits in [1,7], and "
        "all retry/two-view modes disabled");
  }
#else
  if (bitplane_lb ||
      std::getenv("BCI_BITPLANE_HIGH_BITS") != nullptr) {
    throw std::runtime_error(
        "BCI_BITPLANE_LB requires the isolated bit-plane "
        "lower-bound diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
  if (!exact_radix ||
      (exact_engine_mode != "reference_sorted" &&
       exact_engine_mode != "fourway_sorted" &&
       exact_engine_mode != "single_radix" &&
       exact_engine_mode != "fourway_radix") ||
      !batched_l2_beam ||
      fixed_frontier_batched_l2 || two_view_r32 ||
      beam != 256 || K != 10 || single_pool != 10 ||
      single_high_beam != 256 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      single_multi_start != 1 ||
      single_entry_mode != "zero" ||
      batched_l2_prefix_dimensions != 64 ||
      (query_graph_degree_limit != 0 &&
       query_graph_degree_limit != expected_graph_degree) ||
      single_retry_boundary_ratio != 0.0 ||
      single_exact_retry_boundary_ratio != 0.0 ||
      single_alt_retry_boundary_ratio != 0.0) {
    throw std::runtime_error(
        "exact radix diagnostic requires ordinary batched-L2 b256, "
        "K=single_pool=10, prefix64, zero entry, one start, full "
        "graph rows, and all retry/two-view modes disabled");
  }
#else
  if (exact_radix ||
      std::getenv("BCI_EXACT_ENGINE_MODE") != nullptr) {
    throw std::runtime_error(
        "BCI_EXACT_RADIX requires the isolated exact-radix target");
  }
#endif
#ifdef BCI_ENABLE_FIXED_FRONTIER_BATCHED_DIAGNOSTIC
  if (!fixed_frontier_batched_l2 || batched_l2_beam ||
      (beam != 192 && beam != 256) || K != 10 ||
      single_pool != 10 ||
      batched_l2_prefix_dimensions != 64) {
    throw std::runtime_error(
        "fixed-frontier diagnostic requires its isolated kernel, "
        "beam 192 or 256, K=single_pool=10, and prefix_dims=64");
  }
#else
  if (fixed_frontier_batched_l2) {
    throw std::runtime_error(
        "BCI_FIXED_FRONTIER_BATCHED_L2 requires the versioned "
        "fixed-frontier diagnostic target");
  }
#endif
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
  const int cached_replay_probe_beam =
      std::getenv("BCI_CACHED_REPLAY_PROBE_BEAM") != nullptr
          ? std::atoi(std::getenv("BCI_CACHED_REPLAY_PROBE_BEAM"))
          : 68;
  const int cached_replay_repair_beam =
      std::getenv("BCI_CACHED_REPLAY_REPAIR_BEAM") != nullptr
          ? std::atoi(std::getenv("BCI_CACHED_REPLAY_REPAIR_BEAM"))
          : 256;
  if (!batched_l2_beam || beam != 64 || K != 10 ||
      single_pool != 10 || single_high_beam != 64 ||
      std::abs(single_high_cut - hamcg_cut) > 1e-12 ||
      cached_replay_probe_beam != 68 ||
      cached_replay_repair_beam != 256 ||
      n_runs != 1) {
    throw std::runtime_error(
        "cached replay diagnostic requires batched uint8/192-D "
        "base/probe/repair beams=64/68/256, K=single_pool=10, "
        "one pass, and no adaptive beam/cut");
  }
#endif
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
  if (!batched_l2_beam || K != 10) {
    throw std::runtime_error(
        "top10 stability diagnostic requires batched-L2 beam and K=10");
  }
#endif
  if ((static_cast<int>(dual_heap_beam) +
       static_cast<int>(indexed_heap_beam) +
       static_cast<int>(reuse_beam_workspace) +
       static_cast<int>(batched_l2_beam) +
       static_cast<int>(fixed_frontier_batched_l2)) > 1) {
    throw std::runtime_error(
        "dual-heap, indexed-heap, reusable-workspace, batched-L2, and "
        "fixed-frontier beam are isolated treatments");
  }
  if ((dual_heap_beam || indexed_heap_beam || batched_l2_beam ||
       fixed_frontier_batched_l2) &&
      NEG_MODE) {
    throw std::runtime_error(
        "development beam treatments are not connected to the negative "
        "workload path");
  }
  const bool graph_use_common_pack =
      std::getenv("BCI_GRAPH_USE_COMMON_PACK") != nullptr &&
      std::atoi(std::getenv("BCI_GRAPH_USE_COMMON_PACK")) != 0;
  const bool single_cluster_scan =
      std::getenv("BCI_SINGLE_CLUSTER_SCAN") != nullptr &&
      std::atoi(std::getenv("BCI_SINGLE_CLUSTER_SCAN")) != 0;
  const bool common_pack_hugepage =
      std::getenv("BCI_COMMON_PACK_HUGEPAGE") != nullptr &&
      std::atoi(std::getenv("BCI_COMMON_PACK_HUGEPAGE")) != 0;
  const bool skip_cluster_load = std::getenv("BCI_SKIP_CLUSTER_LOAD") != nullptr &&
      std::atoi(std::getenv("BCI_SKIP_CLUSTER_LOAD")) != 0;
  if (single_entry_mode == "nearest_centroid" && skip_cluster_load) {
    throw std::runtime_error(
        "nearest-centroid single entry requires cluster loading");
  }
  if (single_cluster_scan && skip_cluster_load) {
    throw std::runtime_error(
        "single-cluster scan requires cluster loading");
  }
  if (single_cluster_scan &&
      single_entry_mode != "zero") {
    throw std::runtime_error(
        "single-cluster scan must be isolated from graph entry treatments");
  }
  const bool skip_tag_bitvecs = std::getenv("BCI_SKIP_TAG_BITVECS") != nullptr &&
      std::atoi(std::getenv("BCI_SKIP_TAG_BITVECS")) != 0;
  const char* tag_bitvec_mode_raw = std::getenv("BCI_TAG_BITVEC_MODE");
  const std::string tag_bitvec_mode = skip_tag_bitvecs ? "none" :
      ((tag_bitvec_mode_raw && tag_bitvec_mode_raw[0] != '\0') ?
       tag_bitvec_mode_raw : "all");
  if (tag_bitvec_mode != "all" && tag_bitvec_mode != "secondary_only" &&
      tag_bitvec_mode != "frozen_vocab" &&
      tag_bitvec_mode != "support_complement" &&
      tag_bitvec_mode != "none") {
    throw std::runtime_error(
        "BCI_TAG_BITVEC_MODE must be all, secondary_only, frozen_vocab, "
        "support_complement, or none");
  }
  const char* tag_bitvec_vocab_path = std::getenv("BCI_TAG_BITVEC_VOCAB_FILE");
  const char* support_complement_bitmap_vocab_path =
      std::getenv("BCI_SUPPORT_COMPLEMENT_BITMAP_VOCAB_FILE");
  if (tag_bitvec_mode == "frozen_vocab" &&
      (tag_bitvec_vocab_path == nullptr || tag_bitvec_vocab_path[0] == '\0')) {
    throw std::runtime_error(
        "BCI_TAG_BITVEC_VOCAB_FILE is required in frozen_vocab mode");
  }
  if (support_complement_bitmap_vocab_path != nullptr &&
      support_complement_bitmap_vocab_path[0] != '\0' &&
      tag_bitvec_mode != "support_complement") {
    throw std::runtime_error(
        "BCI_SUPPORT_COMPLEMENT_BITMAP_VOCAB_FILE requires "
        "support_complement mode");
  }
  const char* secondary_membership_raw = std::getenv("BCI_SECONDARY_MEMBERSHIP");
  const std::string secondary_membership =
      (secondary_membership_raw && secondary_membership_raw[0] != '\0') ?
      secondary_membership_raw : "point_csr";
  if (secondary_membership != "point_csr" &&
      secondary_membership != "posting_merge") {
    throw std::runtime_error(
        "BCI_SECONDARY_MEMBERSHIP must be point_csr or posting_merge");
  }
  const bool conjunction_support_complement =
      std::getenv("BCI_CONJ_SUPPORT_COMPLEMENT") != nullptr &&
      std::atoi(std::getenv("BCI_CONJ_SUPPORT_COMPLEMENT")) != 0;
#if defined(BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE)
  const char* support_exact_mode_raw =
      std::getenv("BCI_SUPPORT_COMPLEMENT_EXACT_MODE");
  const std::string support_exact_mode =
      support_exact_mode_raw == nullptr ? "" : support_exact_mode_raw;
  if (support_exact_mode != "direct" &&
      support_exact_mode != "block16") {
    throw std::runtime_error(
        "block16 target requires BCI_SUPPORT_COMPLEMENT_EXACT_MODE="
        "direct or block16");
  }
#elif defined(BCI_ENABLE_SUPPORT_COMPLEMENT_PREFIX128_CASCADE)
  const char* support_exact_mode_raw =
      std::getenv("BCI_SUPPORT_COMPLEMENT_EXACT_MODE");
  const std::string support_exact_mode =
      support_exact_mode_raw == nullptr ? "" : support_exact_mode_raw;
  if (support_exact_mode != "direct" &&
      support_exact_mode != "prefix128") {
    throw std::runtime_error(
        "prefix128 target requires BCI_SUPPORT_COMPLEMENT_EXACT_MODE="
        "direct or prefix128");
  }
#else
  const std::string support_exact_mode = "retained";
#endif
  if (conjunction_support_complement &&
      (tag_bitvec_mode != "support_complement" ||
       secondary_membership != "posting_merge")) {
    throw std::runtime_error(
        "BCI_CONJ_SUPPORT_COMPLEMENT requires support_complement bitvectors "
        "and posting_merge membership");
  }
  if (conjunction_support_complement && K != 10) {
    throw std::runtime_error(
        "BCI_CONJ_SUPPORT_COMPLEMENT currently requires K=10");
  }
  if (conjunction_support_complement &&
      (std::getenv("BCI_PAIR_POSTING_CACHE") == nullptr ||
       std::atoi(std::getenv("BCI_PAIR_POSTING_CACHE")) != 0)) {
    throw std::runtime_error(
        "BCI_CONJ_SUPPORT_COMPLEMENT requires BCI_PAIR_POSTING_CACHE=0");
  }
  const PairVectorLayout pair_vector_layout =
      parse_pair_vector_layout(std::getenv("BCI_PAIR_VECTOR_LAYOUT"));

#ifdef BCI_REQUIRE_YFCC_PAIR_TIER_FORMAL
  auto require_env = [](const char* name) -> const char* {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      throw std::runtime_error(std::string("formal target requires ") + name);
    }
    return value;
  };
  if (NEG_MODE || std::string(require_env("BCI_FORMAL_PAIR_TIER")) != "1" ||
      std::string(require_env("BCI_QUERY_MODE")) != "two_tag" ||
      std::string(require_env("BCI_PAIR_POSTING_CACHE")) != "1" ||
      std::string(require_env("BCI_TIE_AWARE")) != "0" ||
      qid_hi - qid_lo != 10'000 || beam != 128 || K != 10 ||
      post_filter_pool != 512 || brute_conj_thresh != 100'000'000 ||
      n_runs != 6 || use_clusters != 0 || use_bitvec != 0 || use_pach != 0 ||
      parlay::num_workers() != 8 || !skip_cluster_load ||
      !skip_tag_bitvecs || tag_bitvec_mode != "none" ||
      secondary_membership != "posting_merge") {
    throw std::runtime_error("formal pair-tier execution contract mismatch");
  }
  (void)require_env("BCI_GT_PATH");
  (void)require_env("BCI_GRAPH_VOCAB_FILE");
  (void)require_env("BCI_PACK_VOCAB_FILE");
  (void)require_env("BCI_PAIR_VOCAB_FILE");
  (void)require_env("BCI_PASS_CSV");
  (void)require_env("BCI_PERQUERY_CSV");
  (void)require_env("BCI_PASS_ARTIFACT_DIR");
  (void)require_env("BCI_RUN_TOKEN");
  (void)require_env("BCI_GT_SHA256");
  (void)require_env("BCI_ACTIVE_QIDS_SHA256");
#endif

#ifdef BCI_REQUIRE_YFCC_PAIR_DENSE_CALIBRATION
  auto require_dense_env = [](const char* name) -> const char* {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      throw std::runtime_error(
          std::string("dense calibration target requires ") + name);
    }
    return value;
  };
  if (NEG_MODE ||
      std::string(require_dense_env("BCI_DENSE_PAIR_CALIBRATION")) != "1" ||
      std::string(require_dense_env("BCI_QUERY_MODE")) != "two_tag" ||
      std::string(require_dense_env("BCI_PAIR_POSTING_CACHE")) != "1" ||
      std::string(require_dense_env("BCI_TIE_AWARE")) != "0" ||
      qid_lo != 60'000 || qid_hi != 70'000 || beam != 128 || K != 10 ||
      post_filter_pool != 512 || brute_conj_thresh != 100'000'000 ||
      n_runs != 6 || use_clusters != 0 || use_bitvec != 0 || use_pach != 0 ||
      parlay::num_workers() != 8 || !skip_cluster_load ||
      !skip_tag_bitvecs || tag_bitvec_mode != "none" ||
      secondary_membership != "posting_merge") {
    throw std::runtime_error("dense pair calibration execution contract mismatch");
  }
  (void)require_dense_env("BCI_PAIR_VECTOR_LAYOUT");
  (void)require_dense_env("BCI_GT_PATH");
  (void)require_dense_env("BCI_GRAPH_VOCAB_FILE");
  (void)require_dense_env("BCI_PACK_VOCAB_FILE");
  (void)require_dense_env("BCI_PAIR_VOCAB_FILE");
  (void)require_dense_env("BCI_PASS_CSV");
  (void)require_dense_env("BCI_PERQUERY_CSV");
  (void)require_dense_env("BCI_PASS_ARTIFACT_DIR");
  (void)require_dense_env("BCI_RUN_TOKEN");
  (void)require_dense_env("BCI_GT_SHA256");
  (void)require_dense_env("BCI_ACTIVE_QIDS_SHA256");
#endif

  printf("=== BCI bench (Arch A: HAMCG shards + brute-fallback) ===\n");
  printf("qid range = [%d, %d), beam = %d, K = %d, post_filter_pool = %d\n",
         qid_lo, qid_hi, beam, K, post_filter_pool);
  printf("brute_conj_thresh=%ld n_runs=%d use_clusters=%d nprobe=%d target_pts=%d use_bitvec=%d use_pach=%d\n",
         (long)brute_conj_thresh, n_runs, use_clusters, nprobe, target_pts, use_bitvec, use_pach);
  printf("[controls] DATA=%s DRT=%s hamcg_cut=%.4f single_pool=%d expected_graph_degree=%d\n",
         DATA.c_str(), DRT.c_str(), hamcg_cut, single_pool, expected_graph_degree);
  printf("[resource controls] skip_cluster_load=%d skip_tag_bitvecs=%d tag_bitvec_mode=%s secondary_membership=%s\n",
         (int)skip_cluster_load, (int)skip_tag_bitvecs, tag_bitvec_mode.c_str(),
         secondary_membership.c_str());
  printf("[conjunction support complement] enabled=%d threshold=%zu\n",
         (int)conjunction_support_complement,
         yfcc_support_complement::kHighTagBitmapThreshold);
#if defined(BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE)
  printf("[support-complement exact mode] %s same_binary=1 "
         "block_candidates=16 line_bytes=64 counters_compiled_out=1\n",
         support_exact_mode.c_str());
#elif defined(BCI_ENABLE_SUPPORT_COMPLEMENT_PREFIX128_CASCADE)
  printf("[support-complement exact mode] %s same_binary=1 "
         "prefix_bytes=128 suffix_bytes=64\n",
         support_exact_mode.c_str());
#endif
  printf("[pair vector layout] %s\n",
         pair_vector_layout_name(pair_vector_layout));
  printf("parlay workers = %ld\n", parlay::num_workers());

  // Physical-state admission controls.  When a vocabulary file is present,
  // load/materialise every listed unit and no unlisted unit, independently of
  // the active query batch.  Comment-only files intentionally mean an empty
  // vocabulary and are valid for the zero-budget arm.
  auto load_tag_vocabulary = [&](const char* env_name) {
    std::unordered_set<int32_t> tags;
    const char* path = std::getenv(env_name);
    if (path == nullptr || path[0] == '\0') {
      return std::make_pair(false, tags);
    }
    std::ifstream vf(path);
    if (!vf) {
      throw std::runtime_error(std::string("cannot open ") + env_name + ": " + path);
    }
    std::string line;
    while (std::getline(vf, line)) {
      auto first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos || line[first] == '#') continue;
      std::istringstream iss(line);
      int64_t tag64 = -1;
      if (!(iss >> tag64) || tag64 < 0 ||
          tag64 > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("invalid tag in ") + env_name + ": " + line);
      }
      tags.insert((int32_t)tag64);
    }
    return std::make_pair(true, tags);
  };
  auto graph_vocab_result = load_tag_vocabulary("BCI_GRAPH_VOCAB_FILE");
  const bool has_graph_vocab = graph_vocab_result.first;
  const auto& graph_vocab = graph_vocab_result.second;
  auto pack_vocab_result = load_tag_vocabulary("BCI_PACK_VOCAB_FILE");
  const bool has_pack_vocab = pack_vocab_result.first;
  const auto& pack_vocab = pack_vocab_result.second;
  auto high_beam_vocab_result =
      load_tag_vocabulary("BCI_SINGLE_HIGH_BEAM_TAGS_FILE");
  const bool has_high_beam_vocab = high_beam_vocab_result.first;
  const auto& high_beam_tags = high_beam_vocab_result.second;
  auto single_risk_beam_vocab_result =
      load_tag_vocabulary("BCI_SINGLE_RISK_BEAM_TAGS_FILE");
  const bool has_single_risk_beam_vocab =
      single_risk_beam_vocab_result.first;
  const auto& single_risk_beam_tags =
      single_risk_beam_vocab_result.second;
  if (has_single_risk_beam_vocab && single_risk_beam_tags.empty()) {
    throw std::runtime_error(
        "BCI_SINGLE_RISK_BEAM_TAGS_FILE must be nonempty when set");
  }
  auto single_retry_vocab_result =
      load_tag_vocabulary("BCI_SINGLE_RETRY_TAGS_FILE");
  const bool has_single_retry_vocab = single_retry_vocab_result.first;
  const auto& single_retry_tags = single_retry_vocab_result.second;
  if (single_retry_distance_cache &&
      (!batched_l2_beam || single_retry_boundary_ratio <= 0.0 ||
       !has_single_retry_vocab || single_retry_tags.empty())) {
    throw std::runtime_error(
        "single-retry distance cache requires batched-L2, an enabled "
        "graph retry, and a nonempty frozen retry-tag vocabulary");
  }
  auto two_view_vocab_result =
      load_tag_vocabulary("BCI_TWO_VIEW_R32_TAGS_FILE");
  const bool has_two_view_vocab = two_view_vocab_result.first;
  const auto& two_view_tags = two_view_vocab_result.second;
#ifdef BCI_ENABLE_TWO_VIEW_R32_DIAGNOSTIC
  if (two_view_r32 &&
      (!has_two_view_vocab || two_view_tags.empty())) {
    throw std::runtime_error(
        "two-view R32 diagnostic requires a nonempty "
        "BCI_TWO_VIEW_R32_TAGS_FILE");
  }
  if (!two_view_r32 && has_two_view_vocab) {
    throw std::runtime_error(
        "BCI_TWO_VIEW_R32_TAGS_FILE is set while the treatment is disabled");
  }
  if (!two_view_r32 &&
      (two_view_primary_full ||
       std::getenv("BCI_TWO_VIEW_R32_ESCAPE_OFFSET") != nullptr)) {
    throw std::runtime_error(
        "two-view primary/escape controls require BCI_TWO_VIEW_R32=1");
  }
#endif
  printf("[physical vocabularies] graph_mode=%s graph_tags=%zu pack_mode=%s pack_tags=%zu\n",
         has_graph_vocab ? "frozen" : "legacy", graph_vocab.size(),
         has_pack_vocab ? "frozen" : "legacy", pack_vocab.size());
  printf("[single adaptive beam] mode=%s base_beam=%d high_beam=%d "
         "base_cut=%.4f high_cut=%.4f tags=%zu min_support=%ld\n",
         (has_high_beam_vocab || single_high_beam_min_support > 0 ||
          has_single_risk_beam_vocab) ?
             "frozen" : "disabled", beam,
         single_high_beam, hamcg_cut, single_high_cut,
         high_beam_tags.size(),
         (long)single_high_beam_min_support);
  printf("[single risk beam] mode=%s beam=%d cut=%.4f tags=%zu "
         "precedence=over_adaptive_high\n",
         has_single_risk_beam_vocab ? "frozen" : "disabled",
         single_risk_beam, single_risk_cut,
         single_risk_beam_tags.size());
  printf("[single multi-start] n_starts=%d min_support=%ld\n",
         single_multi_start, (long)single_multi_start_min_support);
  printf("[single graph entry] mode=%s centroids=%d\n",
         single_entry_mode.c_str(), single_entry_centroids);
  printf("[single graph state] dual_heap=%d indexed_heap=%d "
         "reusable_workspace=%d batched_l2=%d "
         "fixed_frontier_batched_l2=%d two_view_r32=%d "
         "two_view_tags=%zu two_view_primary_full=%d "
         "two_view_escape_offset=%d distance=%s prefix_dims=%d\n",
         dual_heap_beam, indexed_heap_beam, reuse_beam_workspace,
         batched_l2_beam, fixed_frontier_batched_l2,
         two_view_r32, two_view_tags.size(),
         two_view_primary_full, two_view_escape_offset,
         (batched_l2_beam || fixed_frontier_batched_l2)
             ? (batched_l2_prefix_dimensions == 0
                    ? "avx2_full"
                    : "batched_prefix_then_suffix")
             : "reference",
         batched_l2_prefix_dimensions);
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
  printf("[R96 degree4 sidecar] enabled=%d root=%s "
         "policy=canonical_R96_minus_R64_first4 "
         "handoff=primary_exact_top10 beam=64 visit_cap=64 "
         "degree_cap=4 score_cap=266 merge=exact_distance_global_id\n",
         (int)r96_degree4_sidecar,
         r96_degree4_sidecar_root_raw);
#endif
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
  printf("[single graph top10 stability] enabled=1 window_expansions=16 "
         "membership=unordered instrumentation=diagnostic_only\n");
#endif
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
  printf("[single graph cached replay] enabled=1 base_beam=64 "
         "probe_beam=%d repair_beam=%d trigger=top10_membership_churn "
         "distance=exact_uint8_192 memo_scope=query_local "
         "traversal_state=fresh_each_stage\n",
         cached_replay_probe_beam, cached_replay_repair_beam);
#endif
  printf("[single retry] boundary_ratio=%.6f beam=%d cut=%.4f "
         "dist_comps=[%ld,%ld] support=[%ld,%ld] "
         "tag_mode=%s tags=%zu distance_cache=%s\n",
         single_retry_boundary_ratio, single_retry_beam,
         single_retry_cut, (long)single_retry_min_dist_comps,
         (long)single_retry_max_dist_comps,
         (long)single_retry_min_support,
         (long)single_retry_max_support,
         has_single_retry_vocab ? "frozen" : "all",
         single_retry_tags.size(),
         single_retry_distance_cache ? "exact192_query_epoch" : "off");
  printf("[single exact retry] boundary_ratio=%.6f max_support=%ld "
         "max_dist_comps=%ld vector_layout=%s\n",
         single_exact_retry_boundary_ratio,
         (long)single_exact_retry_max_support,
         (long)single_exact_retry_max_dist_comps,
         single_exact_retry_pack ? "packed_shard" : "base_gather");
  printf("[single alternate retry] boundary_ratio=%.6f max_support=%ld\n",
         single_alt_retry_boundary_ratio,
         (long)single_alt_retry_max_support);
  printf("[single graph workspace] reusable=%d\n",
         (int)reuse_beam_workspace);
  printf("[single graph vector layout] common_primary_pack=%d\n",
         (int)graph_use_common_pack);
  printf("[single cluster scan] enabled=%d nprobe=%d target_points=%d\n",
         (int)single_cluster_scan, nprobe, target_pts);
  printf("[query graph edge budget] degree_limit=%d\n",
         query_graph_degree_limit);
  printf("[common primary-pack memory advice] hugepage=%d\n",
         (int)common_pack_hugepage);

  // -- Load datasets ---------------------------------------------------------
  auto t0 = std::chrono::steady_clock::now();

  PR base((DATA + "base.10M.u8bin").c_str());
  PR query((DATA + "query.public.100K.u8bin").c_str());
  csr_filters qm(DATA + "query.metadata.public.100K.spmat");
  csr_filters bm(DATA + "base.metadata.10M.spmat");
  csr_filters bmt = bm.transpose();
  if (canonical_pair_from_support(9, 3, 10, 10) !=
          std::make_pair<int32_t, int32_t>(3, 9) ||
      canonical_pair_from_support(9, 3, 2, 10) !=
          std::make_pair<int32_t, int32_t>(9, 3)) {
    throw std::runtime_error("canonical pair orientation self-test failed");
  }
  auto canonical_pair_orientation = [&](int32_t a, int32_t b) {
    int64_t fa = bmt.row_offsets[a + 1] - bmt.row_offsets[a];
    int64_t fb = bmt.row_offsets[b + 1] - bmt.row_offsets[b];
    return canonical_pair_from_support(a, b, fa, fb);
  };
  printf("[loaded] base=%ld dim=%ld query=%ld qm=%ld base_meta=%ld\n",
         base.size(), base.dimension(), query.size(),
         qm.n_points, bm.n_points);
  if (base.size() != 10'000'000 || query.size() != 100'000 ||
      base.dimension() != 192 || query.dimension() != 192 ||
      base.aligned_dimension() != 192 ||
      query.aligned_dimension() != 192 ||
      bm.n_points != 10'000'000 || qm.n_points != 100'000 ||
      bm.n_filters != 200'386 ||
      bmt.n_points != 200'386 || bmt.n_filters != 10'000'000) {
    throw std::runtime_error(
        "YFCC base/query/vector/metadata dimensions violate the frozen contract");
  }

  // GT for [qid_lo, qid_hi)
  GroundTruth gt{};
  if (!NEG_MODE) {
    int gt_hi_bin = qid_hi / 1000; int gt_lo_bin = qid_lo / 1000;
    const char* gt_override = std::getenv("BCI_GT_PATH");
    std::string gt_path =
        (gt_override != nullptr && gt_override[0] != '\0')
            ? std::string(gt_override)
            : DATA + "yfcc-10M-gt-" + std::to_string(gt_lo_bin) + "-" +
                  std::to_string(gt_hi_bin) + "-AND.bin";
    gt = load_gt(gt_path, static_cast<uint32_t>(qid_hi - qid_lo),
                 static_cast<uint32_t>(K), static_cast<uint32_t>(base.size()));
    printf("[GT loaded] %s N=%u K=%u\n", gt_path.c_str(), gt.N, gt.K);
  }

  // -- Load shards ------------------------------------------------------------
  auto t_shard0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::unique_ptr<Shard>> shards;
  uint64_t loaded_graph_bytes = 0;
  uint64_t loaded_subset_bytes = 0;
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
  uint64_t loaded_r96_degree4_sidecar_bytes = 0;
#endif

  for (const auto& entry : std::filesystem::directory_iterator(DRT + "/shards")) {
    auto name = entry.path().filename().string();
    if (name.find("vamana_tag_") != 0 || entry.path().extension() != ".bin") continue;
    int tag = atoi(name.substr(11, name.size()-15).c_str());
    if (has_graph_vocab && !graph_vocab.count(tag)) continue;
    std::string subset_path = DRT + "/subset_idx/subset_idx_" + std::to_string(tag) + ".bin";
    auto sub = load_subset_idx(subset_path);
    if (sub.empty()) continue;
    std::string gp = DRT + "/shards/vamana_tag_" + std::to_string(tag) + ".bin";
    auto sh = std::make_unique<Shard>();
    sh->tag = tag;
    sh->freq = (int64_t)sub.size();
    sh->subset = std::move(sub);
    sh->graph = GraphI((char*)gp.c_str());
    sh->maxDeg = sh->graph.max_degree();
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
    {
      const std::string sidecar_root(r96_degree4_sidecar_root_raw);
      const std::string sidecar_subset_path =
          sidecar_root + "/subset_idx/subset_idx_" +
          std::to_string(tag) + ".bin";
      const std::string sidecar_graph_path =
          sidecar_root + "/shards/vamana_tag_" +
          std::to_string(tag) + ".bin";
      const auto sidecar_subset =
          load_subset_idx(sidecar_subset_path);
      if (sidecar_subset.size() != sh->subset.size() ||
          !std::equal(
              sidecar_subset.begin(), sidecar_subset.end(),
              sh->subset.begin())) {
        throw std::runtime_error(
            "R96 degree4 sidecar ordered local/global mapping differs "
            "for tag " + std::to_string(tag));
      }
      sh->r96_degree4_sidecar =
          std::make_unique<GraphI>(
              const_cast<char*>(sidecar_graph_path.c_str()));
      GraphI& sidecar = *sh->r96_degree4_sidecar;
      if (sidecar.size() != sh->graph.size() ||
          sidecar.max_degree() != 4) {
        throw std::runtime_error(
            "R96 degree4 sidecar row/degree contract differs for tag " +
            std::to_string(tag));
      }
      uint64_t sidecar_edges = 0;
      for (size_t local = 0; local < sidecar.size(); ++local) {
        auto row = sidecar[static_cast<Indx>(local)];
        if (row.size() > 4) {
          throw std::runtime_error(
              "R96 degree4 sidecar row exceeds four");
        }
        std::array<Indx, 4> seen{};
        for (size_t position = 0; position < row.size(); ++position) {
          const Indx target = row[static_cast<Indx>(position)];
          if (target < 0 ||
              static_cast<size_t>(target) >= sidecar.size() ||
              static_cast<size_t>(target) == local) {
            throw std::runtime_error(
                "R96 degree4 sidecar has self/out-of-range edge");
          }
          for (size_t prior = 0; prior < position; ++prior) {
            if (seen[prior] == target) {
              throw std::runtime_error(
                  "R96 degree4 sidecar row has duplicate edge");
            }
          }
          seen[position] = target;
        }
        sidecar_edges += row.size();
      }
      const uint64_t expected_sidecar_bytes =
          8ULL + 4ULL * static_cast<uint64_t>(sidecar.size()) +
          4ULL * sidecar_edges;
      const uint64_t actual_sidecar_bytes =
          std::filesystem::file_size(sidecar_graph_path);
      if (actual_sidecar_bytes != expected_sidecar_bytes) {
        throw std::runtime_error(
            "R96 degree4 sidecar exact-length contract failed");
      }
      loaded_r96_degree4_sidecar_bytes += actual_sidecar_bytes;
    }
#endif
    shards[tag] = std::move(sh);
    loaded_graph_bytes += std::filesystem::file_size(entry.path());
    loaded_subset_bytes += std::filesystem::file_size(subset_path);
  }
  auto t_shard1 = std::chrono::steady_clock::now();
  printf("[loaded %zu shards, graph=%.3fGB subset=%.3fGB in %.2fs]\n", shards.size(),
         loaded_graph_bytes / 1e9, loaded_subset_bytes / 1e9,
         std::chrono::duration<double>(t_shard1-t_shard0).count());
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
  printf("[loaded R96 degree4 sidecars] graph=%.3fGB "
         "mapping=ordered_identity\n",
         loaded_r96_degree4_sidecar_bytes / 1e9);
#endif
  if (shards.empty() && !has_graph_vocab) throw std::runtime_error("no graph shards loaded");
  if (has_graph_vocab && shards.size() != graph_vocab.size()) {
    throw std::runtime_error("frozen graph vocabulary contains a missing or empty shard");
  }
  long min_graph_degree = std::numeric_limits<long>::max();
  long max_graph_degree = std::numeric_limits<long>::min();
  for (const auto& kv : shards) {
    min_graph_degree = std::min(min_graph_degree, kv.second->maxDeg);
    max_graph_degree = std::max(max_graph_degree, kv.second->maxDeg);
  }
  printf("[graph degree audit] loaded=%zu min=%ld max=%ld expected=%d\n",
         shards.size(), shards.empty() ? 0 : min_graph_degree,
         shards.empty() ? 0 : max_graph_degree, expected_graph_degree);
  if (!shards.empty() && expected_graph_degree > 0 &&
      (min_graph_degree != expected_graph_degree ||
       max_graph_degree != expected_graph_degree)) {
    throw std::runtime_error(
        "graph-degree mismatch: wrong BCI_DRT_ROOT or mixed shard directory");
  }

#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
  std::unique_ptr<ShortcutResidualMask> shortcut_mask;
  {
    auto shard_it = shards.find(shortcut_residual_tag);
    if (shard_it == shards.end()) {
      throw std::runtime_error(
          "shortcut residual tag35 shard is not loaded");
    }
    const std::filesystem::path current_graph_path =
        std::filesystem::path(DRT) / "shards" /
        ("vamana_tag_" +
         std::to_string(shortcut_residual_tag) + ".bin");
    const uint64_t current_graph_bytes =
        std::filesystem::file_size(current_graph_path);
    shortcut_mask = std::make_unique<ShortcutResidualMask>(
        load_shortcut_residual_mask(
            shortcut_residual_mask_path,
            shard_it->second->graph,
            current_graph_bytes));
    printf(
        "[shortcut residual mask] tag=%d rows=%llu edges=%llu "
        "novel_edges=%llu payload_bytes=%llu sidecar_bytes=%llu "
        "current_graph_bytes=%llu control_graph_bytes=%llu "
        "combined_bytes=%llu budget_ok=1 starts=primary_top10 "
        "secondary_beam=32 secondary_degree_limit=64\n",
        shortcut_residual_tag,
        static_cast<unsigned long long>(
            shortcut_mask->header.rows),
        static_cast<unsigned long long>(
            shortcut_mask->header.edge_count),
        static_cast<unsigned long long>(
            shortcut_mask->novel_edges),
        static_cast<unsigned long long>(
            shortcut_mask->header.payload_bytes),
        static_cast<unsigned long long>(
            shortcut_mask->file_bytes),
        static_cast<unsigned long long>(
            shortcut_mask->header.current_graph_bytes),
        static_cast<unsigned long long>(
            shortcut_mask->header.control_graph_bytes),
        static_cast<unsigned long long>(
            shortcut_mask->header.current_graph_bytes +
            shortcut_mask->file_bytes));
  }
#endif

  if (NEG_MODE) {
    const char* planner_raw = std::getenv("BCI_PLANNER");
    std::string planner = (planner_raw && planner_raw[0] != '\0') ? planner_raw : "exact";
    neg_stdout_redirect.restore();
    return run_neg_workload(neg_workload_raw, planner, base, query, bm, bmt, shards,
                            K, beam, post_filter_pool, brute_conj_thresh);
  }

  // -- Load IVF² cluster indices (IVF² absorb fast path) ----------------------
  auto t_clust0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::unique_ptr<ClusterIndex>> clusters;
  std::string clust_dir = DRT + "/clusters";
  if (!skip_cluster_load && std::filesystem::exists(clust_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(clust_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("_centroids.bin") == std::string::npos) continue;
      int tag = atoi(name.c_str());
      if (has_graph_vocab && !graph_vocab.count(tag)) continue;
      auto ci = load_cluster_index(clust_dir, tag);
      if (ci) clusters[tag] = std::move(ci);
    }
  }
  auto t_clust1 = std::chrono::steady_clock::now();
  printf("[loaded %zu cluster indices in %.2fs]\n", clusters.size(),
         std::chrono::duration<double>(t_clust1-t_clust0).count());

  // Derive one graph-local medoid entry for every relevant centroid
  // partition. This state depends only on the base vectors and frozen index.
  if (single_entry_mode == "nearest_centroid" ||
      single_multi_start > 1 ||
      single_alt_retry_boundary_ratio > 0.0) {
    auto t_rep0 = std::chrono::steady_clock::now();
    std::vector<int32_t> global_to_local((size_t)base.size(), -1);
    size_t prepared_tags = 0;
    size_t prepared_representatives = 0;
    for (auto& kv : clusters) {
      auto shard_it = shards.find(kv.first);
      if (shard_it == shards.end()) continue;
      Shard& shard = *shard_it->second;
      const bool needed_by_multi_start =
          select_single_multi_start(
              shard.freq, single_multi_start,
              single_multi_start_min_support);
      const bool needed_by_alt_retry =
          single_alt_retry_boundary_ratio > 0.0 &&
          shard.freq < single_alt_retry_max_support;
      const bool needed_by_nearest_entry =
          single_entry_mode == "nearest_centroid";
      if (!needed_by_nearest_entry &&
          !needed_by_multi_start && !needed_by_alt_retry) {
        continue;
      }
      ClusterIndex& ci = *kv.second;
      ci.representative_local_ids.assign(ci.n_clusters, -1);
      for (size_t local = 0; local < shard.subset.size(); ++local) {
        const int32_t global = shard.subset[local];
        if (global < 0 || global >= base.size()) {
          throw std::runtime_error(
              "shard subset contains an invalid global id");
        }
        global_to_local[(size_t)global] = (int32_t)local;
      }
      for (int c = 0; c < ci.n_clusters; ++c) {
        const int64_t lo = ci.member_offsets[c];
        const int64_t hi = ci.member_offsets[c + 1];
        float best_distance = std::numeric_limits<float>::max();
        int32_t best_local = -1;
        const uint8_t* centroid =
            ci.centroid_data.data() + (size_t)c * ci.aligned_dim;
        for (int64_t j = lo; j < hi; ++j) {
          const int32_t global = ci.member_ids[j];
          if (global < 0 || global >= base.size()) {
            throw std::runtime_error(
                "cluster member contains an invalid global id");
          }
          const int32_t local = global_to_local[(size_t)global];
          if (local < 0) {
            throw std::runtime_error(
                "cluster member is absent from its graph shard");
          }
          const float distance = l2_sq_uint8_avx2(
              centroid, base[global].get(), (unsigned)ci.aligned_dim);
          if (distance < best_distance ||
              (distance == best_distance && local < best_local)) {
            best_distance = distance;
            best_local = local;
          }
        }
        if (best_local >= 0) {
          ci.representative_local_ids[c] = best_local;
          ++prepared_representatives;
        }
      }
      for (const int32_t global : shard.subset) {
        global_to_local[(size_t)global] = -1;
      }
      ++prepared_tags;
    }
    auto t_rep1 = std::chrono::steady_clock::now();
    printf("[single multi-start representatives] tags=%zu reps=%zu "
           "source=index_base_only seconds=%.3f\n",
           prepared_tags, prepared_representatives,
           std::chrono::duration<double>(t_rep1 - t_rep0).count());
  }

  // PACK COLD-TAG POINT DATA contiguous for sequential brute scan.
  // Eliminates random base[g] access in brute path (largest wall-time component).
  // For ~1600 unique cold tags × ~7K points × 192B = ~2.2GB upfront. Sequential
  // memory access = full prefetcher utilization, no DRAM-latency stalls.
  auto t_pack0 = std::chrono::steady_clock::now();
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_VARIANCE_ORDER
  // Query-independent physical layout: estimate per-coordinate dispersion
  // from a deterministic 1% base-only sample, then place high-dispersion
  // coordinates first.  Reordering nonnegative squared-distance terms makes
  // early exact bounds tighter without changing the completed distance.
  std::array<size_t, yfcc_support_complement::kDimensions>
      support_complement_dimension_order{};
  std::iota(
      support_complement_dimension_order.begin(),
      support_complement_dimension_order.end(), size_t{0});
  std::array<uint64_t, yfcc_support_complement::kDimensions> dimension_sum{};
  std::array<uint64_t, yfcc_support_complement::kDimensions>
      dimension_sum_squares{};
  constexpr size_t kVarianceSampleStride = 100;
  size_t variance_samples = 0;
  for (size_t point = 0; point < static_cast<size_t>(base.size());
       point += kVarianceSampleStride) {
    const uint8_t* vector = base[point].get();
    for (size_t dimension = 0;
         dimension < yfcc_support_complement::kDimensions; ++dimension) {
      const uint64_t value = vector[dimension];
      dimension_sum[dimension] += value;
      dimension_sum_squares[dimension] += value * value;
    }
    ++variance_samples;
  }
  auto variance_numerator = [&](size_t dimension) {
    return static_cast<unsigned __int128>(variance_samples) *
               dimension_sum_squares[dimension] -
           static_cast<unsigned __int128>(dimension_sum[dimension]) *
               dimension_sum[dimension];
  };
  std::stable_sort(
      support_complement_dimension_order.begin(),
      support_complement_dimension_order.end(),
      [&](size_t left, size_t right) {
        const auto left_score = variance_numerator(left);
        const auto right_score = variance_numerator(right);
        return left_score != right_score ? left_score > right_score
                                         : left < right;
      });
  std::array<bool, yfcc_support_complement::kDimensions> dimension_seen{};
  for (size_t dimension : support_complement_dimension_order) {
    if (dimension >= yfcc_support_complement::kDimensions ||
        dimension_seen[dimension]) {
      throw std::runtime_error(
          "base-only variance order is not a permutation");
    }
    dimension_seen[dimension] = true;
  }
  printf("[support-complement dimension order] source=base_only "
         "sample_stride=%zu samples=%zu first8=%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
         kVarianceSampleStride, variance_samples,
         support_complement_dimension_order[0],
         support_complement_dimension_order[1],
         support_complement_dimension_order[2],
         support_complement_dimension_order[3],
         support_complement_dimension_order[4],
         support_complement_dimension_order[5],
         support_complement_dimension_order[6],
         support_complement_dimension_order[7]);
#endif
  std::unordered_map<int32_t, CommonPackVector> packed_cold;
  // First pass: which cold tags are actually queried?
  std::unordered_set<int32_t> cold_tags_queried;
  int n_q_tmp = qid_hi - qid_lo;
  if (has_pack_vocab) {
    // Materialise the entire frozen physical vocabulary, including dormant
    // units, so preparation and RSS never depend on test predicate coverage.
    cold_tags_queried = pack_vocab;
  } else for (int i = 0; i < n_q_tmp; ++i) {
    int qid = qid_lo + i;
    int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
    int n_tags = (int)(e - s);
    if (n_tags == 1) {
      int32_t t = qm.row_indices[s];
      if (!shards.count(t)) cold_tags_queried.insert(t);
    } else if (n_tags == 2) {
      int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
      int64_t f1 = bmt.row_offsets[t1+1] - bmt.row_offsets[t1];
      int64_t f2 = bmt.row_offsets[t2+1] - bmt.row_offsets[t2];
      auto orientation = canonical_pair_orientation(t1, t2);
      int32_t small_t = orientation.first;
      int32_t large_t = orientation.second;
      if (std::min(f1, f2) <= brute_conj_thresh) {
        cold_tags_queried.insert(small_t);
      } else if (!shards.count(small_t) && !shards.count(large_t)) {
        cold_tags_queried.insert(small_t);
      }
    }
  }
  for (int32_t tag : cold_tags_queried) {
    if (tag < 0 || tag >= bmt.n_filters) {
      throw std::runtime_error("packed-layout vocabulary tag out of range");
    }
  }
  size_t aligned_dim = base.aligned_dimension();
  for (int32_t tag : cold_tags_queried) {
    int64_t lo = bmt.row_offsets[tag], hi = bmt.row_offsets[tag+1];
    size_t n = hi - lo;
    if (n == 0) continue;
    auto& packed = packed_cold[tag];
    packed.resize(n * aligned_dim);
    for (int64_t j = lo; j < hi; ++j) {
      int32_t g = bmt.row_indices[j];
      uint8_t* src = base[g].get();
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_VARIANCE_ORDER
      yfcc_support_complement::permute_u8_192_prevalidated(
          src, support_complement_dimension_order.data(),
          packed.data() + (j - lo) * aligned_dim);
#else
      std::memcpy(packed.data() + (j - lo) * aligned_dim, src, aligned_dim);
#endif
    }
  }
  size_t packed_bytes = 0;
  size_t packed_memberships = 0;
  size_t packed_misaligned_64 = 0;
  for (auto& kv : packed_cold) {
    packed_bytes += kv.second.size();
    packed_misaligned_64 +=
        reinterpret_cast<uintptr_t>(kv.second.data()) % 64U != 0U;
    if (kv.second.size() % aligned_dim != 0) {
      throw std::runtime_error("common primary pack byte alignment mismatch");
    }
    packed_memberships += kv.second.size() / aligned_dim;
  }
  size_t common_pack_advised_bytes = 0;
  size_t common_pack_advice_failures = 0;
  if (common_pack_hugepage) {
    const long page_size_raw = ::sysconf(_SC_PAGESIZE);
    if (page_size_raw <= 0) {
      throw std::runtime_error(
          "cannot determine system page size for common pack");
    }
    const uintptr_t page_size =
        static_cast<uintptr_t>(page_size_raw);
    for (auto& item : packed_cold) {
      if (item.second.empty()) continue;
      const uintptr_t raw_begin =
          reinterpret_cast<uintptr_t>(item.second.data());
      const uintptr_t raw_end =
          raw_begin + item.second.size();
      const uintptr_t begin = raw_begin & ~(page_size - 1);
      const uintptr_t end =
          (raw_end + page_size - 1) & ~(page_size - 1);
      const size_t bytes = static_cast<size_t>(end - begin);
      if (::madvise(
              reinterpret_cast<void*>(begin), bytes,
              MADV_HUGEPAGE) != 0) {
        ++common_pack_advice_failures;
      } else {
        common_pack_advised_bytes += bytes;
      }
    }
    if (common_pack_advice_failures != 0) {
      throw std::runtime_error(
          "MADV_HUGEPAGE failed for common primary pack");
    }
  }
  auto t_pack1 = std::chrono::steady_clock::now();
  printf("[packed %zu cold-tag arrays in %.2fs, %.1fMB total]\n",
         packed_cold.size(),
         std::chrono::duration<double>(t_pack1-t_pack0).count(),
         packed_bytes / 1e6);
  printf("[common pack physical] tags=%zu memberships=%zu bytes=%zu "
         "aligned_dim=%zu base_alignment=64 misaligned=%zu\n",
         packed_cold.size(), packed_memberships, packed_bytes, aligned_dim,
         packed_misaligned_64);
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BOUNDED_EXACT
  if (packed_misaligned_64 != 0) {
    throw std::runtime_error(
        "bounded-exact common primary pack is not 64-byte aligned");
  }
#endif
  printf("[common primary-pack hugepage] enabled=%d advised_bytes=%zu "
         "failures=%zu\n",
         (int)common_pack_hugepage, common_pack_advised_bytes,
         common_pack_advice_failures);
  if (pair_vector_layout == PairVectorLayout::PrimaryOffset ||
      conjunction_support_complement) {
    auto t_alignment0 = std::chrono::steady_clock::now();
    std::vector<int32_t> packed_tags;
    packed_tags.reserve(packed_cold.size());
    for (const auto& item : packed_cold) packed_tags.push_back(item.first);
    std::sort(packed_tags.begin(), packed_tags.end());
    parlay::parallel_for(0, packed_tags.size(), [&](size_t index) {
      int32_t tag = packed_tags[index];
      int64_t start = bmt.row_offsets[tag];
      int64_t end = bmt.row_offsets[tag + 1];
      const auto& packed = packed_cold.at(tag);
      if (packed.size() != (size_t)(end - start) * aligned_dim) {
        throw std::runtime_error("common primary pack exact-size mismatch");
      }
      auto base_vector = [&](int32_t global_id) -> const uint8_t* {
        return base[global_id].get();
      };
      validate_primary_pack_alignment(
          bmt.row_indices.get() + start, (size_t)(end - start),
          packed.data(), aligned_dim, base_vector
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_VARIANCE_ORDER
          , support_complement_dimension_order.data()
#endif
          );
    });
    auto t_alignment1 = std::chrono::steady_clock::now();
    printf("[common primary-pack alignment audit] tags=%zu bytes=%zu exact=1 "
           "seconds=%.3f\n",
           packed_tags.size(), packed_bytes,
           std::chrono::duration<double>(t_alignment1 - t_alignment0).count());
  }
  std::unordered_set<int32_t> graph_common_pack_tags;
  size_t graph_common_pack_bytes = 0;
  size_t graph_common_pack_memberships = 0;
  if (graph_use_common_pack) {
    for (const auto& item : packed_cold) {
      const int32_t tag = item.first;
      auto shard_it = shards.find(tag);
      if (shard_it == shards.end()) continue;
      const auto& subset = shard_it->second->subset;
      const int64_t start = bmt.row_offsets[tag];
      const int64_t end = bmt.row_offsets[tag + 1];
      const size_t posting_size = static_cast<size_t>(end - start);
      if (subset.size() != posting_size ||
          item.second.size() != posting_size * aligned_dim ||
          !std::equal(
              subset.begin(), subset.end(),
              bmt.row_indices.get() + start)) {
        throw std::runtime_error(
            "graph/common-primary pack local-ID order mismatch");
      }
      graph_common_pack_tags.insert(tag);
      graph_common_pack_memberships += posting_size;
      graph_common_pack_bytes += item.second.size();
    }
  }
  printf("[single graph common-pack audit] enabled=%d tags=%zu "
         "memberships=%zu bytes=%zu local_order_exact=1\n",
         (int)graph_use_common_pack, graph_common_pack_tags.size(),
         graph_common_pack_memberships, graph_common_pack_bytes);

  // Query-independent packed layout for the sparse uncertainty-triggered
  // complete-support fallback. Materialise every graph shard admitted by the
  // support rule, not merely shards seen in the active query batch.
  std::unordered_map<int32_t, std::vector<uint8_t>> packed_single_exact;
  if (single_exact_retry_boundary_ratio > 0.0 &&
      single_exact_retry_pack) {
    auto t_exact_pack0 = std::chrono::steady_clock::now();
    std::vector<int32_t> exact_pack_tags;
    exact_pack_tags.reserve(shards.size());
    size_t exact_pack_memberships = 0;
    for (const auto& kv : shards) {
      if (kv.second->freq < single_exact_retry_max_support) {
        exact_pack_tags.push_back(kv.first);
        exact_pack_memberships += kv.second->subset.size();
      }
    }
    std::sort(exact_pack_tags.begin(), exact_pack_tags.end());
    for (const int32_t tag : exact_pack_tags) {
      packed_single_exact[tag].resize(
          shards.at(tag)->subset.size() * aligned_dim);
    }
    parlay::parallel_for(0, exact_pack_tags.size(), [&](size_t ti) {
      const int32_t tag = exact_pack_tags[ti];
      const auto& subset = shards.at(tag)->subset;
      auto& packed = packed_single_exact.at(tag);
      for (size_t local = 0; local < subset.size(); ++local) {
        std::memcpy(
            packed.data() + local * aligned_dim,
            base[subset[local]].get(), aligned_dim);
      }
    });
    const size_t exact_pack_bytes =
        exact_pack_memberships * aligned_dim;
    auto t_exact_pack1 = std::chrono::steady_clock::now();
    printf("[single exact-retry pack] tags=%zu memberships=%zu "
           "bytes=%zu source=index_base_only seconds=%.3f\n",
           exact_pack_tags.size(), exact_pack_memberships,
           exact_pack_bytes,
           std::chrono::duration<double>(
               t_exact_pack1 - t_exact_pack0).count());
  }

  // PACH-in-brute (NOVEL) — partition each packed_cold tag's posting into
  // chunks of CHUNK_SIZE points. Per chunk, store a bitvec over secondary
  // predicate IDs: bit B set iff ≥1 point in chunk has tag B. At query time
  // for conjunction A∧B (brute path on primary A), skip whole chunks where
  // chunk-bitvec[B]=0. Skip ratio = (1-ρ)^c where ρ = secondary density.
  // For c=64, ρ=0.01: ~47% skip. Activates PACH on dense workloads
  // unlike cluster-level PACH which requires c·ρ ≪ 1.
  constexpr int CHUNK_SIZE = 16;
  auto t_chunk0 = std::chrono::steady_clock::now();
  // chunk_bvs[tag] = flattened (n_chunks × bv_words) bitvecs over secondary tags
  std::unordered_map<int32_t, std::vector<uint64_t>> chunk_bvs;
  size_t pach_bv_words = (bm.n_filters + 63) / 64;
  size_t chunk_bvs_bytes = 0;
  std::vector<int32_t> ct_tags_vec(cold_tags_queried.begin(), cold_tags_queried.end());
  // Allocate
  if (!(std::getenv("BCI_SKIP_PACH_BUILD")!=nullptr && std::atoi(std::getenv("BCI_SKIP_PACH_BUILD"))!=0)) {  // build PACH-in-brute unless BCI_SKIP_PACH_BUILD=1 (resource-fairness: skip the dormant ~40GB structure)
  for (int32_t tag : ct_tags_vec) {
    int64_t lo = bmt.row_offsets[tag], hi = bmt.row_offsets[tag+1];
    int n_pts = (int)(hi - lo);
    int n_chunks = (n_pts + CHUNK_SIZE - 1) / CHUNK_SIZE;
    chunk_bvs[tag].assign((size_t)n_chunks * pach_bv_words, 0ULL);
    chunk_bvs_bytes += (size_t)n_chunks * pach_bv_words * 8;
  }
  // Fill in parallel
  parlay::parallel_for(0, ct_tags_vec.size(), [&](size_t ti) {
    int32_t tag = ct_tags_vec[ti];
    int64_t lo = bmt.row_offsets[tag], hi = bmt.row_offsets[tag+1];
    int n_pts = (int)(hi - lo);
    auto& cbv = chunk_bvs[tag];
    for (int k = 0; k < n_pts; ++k) {
      int chunk_id = k / CHUNK_SIZE;
      int32_t g = bmt.row_indices[lo + k];
      int64_t s = bm.row_offsets[g];
      int64_t e = bm.row_offsets[g+1];
      uint64_t* bv = cbv.data() + (size_t)chunk_id * pach_bv_words;
      for (int64_t j = s; j < e; ++j) {
        int32_t t = bm.row_indices[j];
        bv[t >> 6] |= (1ULL << (t & 63));
      }
    }
  });
  auto t_chunk1 = std::chrono::steady_clock::now();
  printf("[built PACH-in-brute chunk bitvecs for %zu tags in %.2fs, %.1fMB]\n",
         chunk_bvs.size(),
         std::chrono::duration<double>(t_chunk1-t_chunk0).count(),
         chunk_bvs_bytes / 1e6);
  }  // end if(use_pach)

  // Pre-build per-tag bitvectors for conjunction post-filter (eliminates per-query
  // bm.match cache misses). Each bitvector = N_total bits = 1.25MB packed.
  // EXPANDED COVERAGE: build bitvecs for ALL queried secondary tags (not just
  // sweet-spot tags). Previously fallback to bm.match (linear scan) for non-band
  // secondary tags dominated brute_cold latency.
  auto t_bv0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::vector<uint64_t>> bitvecs;
  size_t bv_words = (bm.n_points + 63) / 64;

  // The default preserves the original behavior and materializes every queried
  // atom.  The resource-audit mode materializes only the atom that the frozen
  // router uses as a conjunction's secondary membership predicate.
  std::unordered_set<int32_t> all_query_tags;
  if (tag_bitvec_mode != "none") {
    if (tag_bitvec_mode == "frozen_vocab") {
      std::ifstream vf(tag_bitvec_vocab_path);
      if (!vf) {
        throw std::runtime_error(
            std::string("cannot open BCI_TAG_BITVEC_VOCAB_FILE: ") +
            tag_bitvec_vocab_path);
      }
      std::string line;
      while (std::getline(vf, line)) {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream iss(line);
        int64_t tag64 = -1;
        if (!(iss >> tag64) || tag64 < 0 || tag64 >= bm.n_filters) {
          throw std::runtime_error("invalid tag in frozen bitvector vocabulary");
        }
        all_query_tags.insert((int32_t)tag64);
      }
      if (all_query_tags.empty()) {
        throw std::runtime_error("empty frozen bitvector vocabulary");
      }
    } else if (tag_bitvec_mode == "support_complement") {
      // bmt is the tag-to-point transpose: rows are tags and columns are
      // points.  Its valid posting-row domain is therefore n_points.
      for (int32_t tag = 0; tag < bmt.n_points; ++tag) {
        const size_t support = static_cast<size_t>(
            bmt.row_offsets[tag + 1] - bmt.row_offsets[tag]);
        if (support >
            yfcc_support_complement::kHighTagBitmapThreshold) {
          all_query_tags.insert(tag);
        }
      }
      if (all_query_tags.empty()) {
        throw std::runtime_error(
            "support-complement bitmap family is empty");
      }
      if (support_complement_bitmap_vocab_path != nullptr &&
          support_complement_bitmap_vocab_path[0] != '\0') {
        std::ifstream vf(support_complement_bitmap_vocab_path);
        if (!vf) {
          throw std::runtime_error(
              std::string(
                  "cannot open BCI_SUPPORT_COMPLEMENT_BITMAP_VOCAB_FILE: ") +
              support_complement_bitmap_vocab_path);
        }
        std::string line;
        while (std::getline(vf, line)) {
          auto first = line.find_first_not_of(" \t\r\n");
          if (first == std::string::npos || line[first] == '#') continue;
          std::istringstream iss(line);
          int64_t tag64 = -1;
          if (!(iss >> tag64) || tag64 < 0 || tag64 >= bmt.n_points) {
            throw std::runtime_error(
                "invalid tag in support-complement bitmap vocabulary");
          }
          all_query_tags.insert(static_cast<int32_t>(tag64));
        }
      }
    } else if (tag_bitvec_mode == "secondary_only") {
      for (int i = 0; i < n_q_tmp; ++i) {
        int qid = qid_lo + i;
        int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
        if (e - s != 2) continue;
        int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
        int64_t f1 = bmt.row_offsets[t1+1] - bmt.row_offsets[t1];
        int64_t f2 = bmt.row_offsets[t2+1] - bmt.row_offsets[t2];
        auto orientation = canonical_pair_orientation(t1, t2);
        int32_t small_t = orientation.first;
        int32_t large_t = orientation.second;
        int32_t secondary;
        if (std::min(f1, f2) <= brute_conj_thresh) {
          secondary = large_t;
        } else if (shards.count(small_t)) {
          secondary = large_t;
        } else if (shards.count(large_t)) {
          secondary = small_t;
        } else {
          secondary = large_t;
        }
        all_query_tags.insert(secondary);
      }
    } else {
      for (int i = 0; i < n_q_tmp; ++i) {
        int qid = qid_lo + i;
        int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
        for (int64_t j = s; j < e; ++j) all_query_tags.insert(qm.row_indices[j]);
      }
      // Cluster operators can use either atom as a membership predicate.
      for (auto& tc : clusters) all_query_tags.insert(tc.first);
    }

    for (int32_t tag : all_query_tags) {
      auto& bv = bitvecs[tag];
      bv.assign(bv_words, 0ULL);
      int64_t lo = bmt.row_offsets[tag];
      int64_t hi = bmt.row_offsets[tag+1];
      for (int64_t j = lo; j < hi; ++j) {
        int32_t g = bmt.row_indices[j];
        bv[g >> 6] |= (1ULL << (g & 63));
      }
      if (tag_bitvec_mode == "support_complement") {
        size_t cardinality = 0;
        for (uint64_t word : bv) {
          cardinality += static_cast<size_t>(__builtin_popcountll(word));
        }
        if (cardinality != static_cast<size_t>(hi - lo)) {
          throw std::runtime_error(
              "support-complement bitmap cardinality mismatch");
        }
      }
    }
  }
  if (tag_bitvec_mode == "support_complement") {
    size_t required_high_tags = 0;
    for (int32_t tag = 0; tag < bmt.n_points; ++tag) {
      const size_t support = static_cast<size_t>(
          bmt.row_offsets[tag + 1] - bmt.row_offsets[tag]);
      if (support >
          yfcc_support_complement::kHighTagBitmapThreshold) {
        ++required_high_tags;
        if (!bitvecs.count(tag)) {
          throw std::runtime_error(
              "required high-support bitmap is absent");
        }
      }
    }
    if (required_high_tags != 53 || bitvecs.size() < required_high_tags ||
        bv_words != 156'250) {
      throw std::runtime_error(
          "support-complement bitmap family violates the frozen YFCC contract");
    }
  }
  auto t_bv1 = std::chrono::steady_clock::now();
  printf("[built %zu tag bitvectors in %.2fs, %.1fMB total]\n", bitvecs.size(),
         std::chrono::duration<double>(t_bv1-t_bv0).count(),
         bitvecs.size() * bv_words * 8 / 1e6);

  // PACH (Predicate-Aware Cluster Hierarchy) — NOVEL BCI contribution.
  // For each primary tag T and each cluster c of T, pre-compute a bitvec
  // over secondary tag IDs: bit B set iff cluster c contains ≥1 point with tag B.
  // At query time for A ∧ B with primary=A: scan only clusters c of A whose
  // pach_bitvec[A][c] has bit B set — prunes clusters guaranteed to yield 0
  // post-filter survivors. Expected pruning: 60-90% on selective conjunctions.
  // Distinguishes BCI from PIVF's IVF² (which has no predicate-aware cluster pruning).
  auto t_pach0 = std::chrono::steady_clock::now();
  std::unordered_map<int32_t, std::vector<std::vector<uint64_t>>> pach_bitvecs;
  size_t pach_words = (bm.n_filters + 63) / 64;
  size_t pach_bytes_total = 0;
  size_t pach_clusters = 0;
  parlay::sequence<int32_t> tags_for_pach;
  for (auto& tc : clusters) tags_for_pach.push_back(tc.first);
  // pre-allocate to allow parallel writes
  for (int32_t tag : tags_for_pach) {
    ClusterIndex& ci = *clusters[tag];
    pach_bitvecs[tag].assign(ci.n_clusters, std::vector<uint64_t>(pach_words, 0ULL));
    pach_clusters += ci.n_clusters;
    pach_bytes_total += (size_t)ci.n_clusters * pach_words * 8;
  }
  parlay::parallel_for(0, tags_for_pach.size(), [&](size_t ti) {
    int32_t tag = tags_for_pach[ti];
    ClusterIndex& ci = *clusters[tag];
    auto& cluster_bvs = pach_bitvecs[tag];
    for (int c = 0; c < ci.n_clusters; ++c) {
      auto& bv = cluster_bvs[c];
      int64_t lo = ci.member_offsets[c];
      int64_t hi = ci.member_offsets[c+1];
      for (int64_t j = lo; j < hi; ++j) {
        int32_t g = ci.member_ids[j];
        int64_t s = bm.row_offsets[g];
        int64_t e = bm.row_offsets[g+1];
        for (int64_t k = s; k < e; ++k) {
          int32_t t = bm.row_indices[k];
          bv[t >> 6] |= (1ULL << (t & 63));
        }
      }
    }
  });
  auto t_pach1 = std::chrono::steady_clock::now();
  printf("[built PACH bitvecs for %zu tags, %zu clusters total in %.2fs, %.1fMB]\n",
         pach_bitvecs.size(), pach_clusters,
         std::chrono::duration<double>(t_pach1-t_pach0).count(),
         pach_bytes_total / 1e6);

  // PACH instrumentation: count clusters considered vs kept across all conjunctions
  std::atomic<int64_t> pach_total_clusters{0};
  std::atomic<int64_t> pach_kept_clusters{0};

  // -- Build per-query route info --------------------------------------------
  int n_q = qid_hi - qid_lo;
  std::vector<int>  q_route(n_q, -1);   // 0=HAMCG_single, 1=HAMCG_conj, 2=brute_cold, -1=skip
  std::vector<int32_t> q_primary(n_q, -1);
  std::vector<int32_t> q_secondary(n_q, -1);
  // PROXY ROUTER (paper's central idea): per-query adaptive target_points based
  // on predicate-selectivity proxy. Easy queries (low joint selectivity) use few
  // candidates; hard queries (high selectivity boundary) use more.
  std::vector<int> q_tpts(n_q, target_pts);

  // BCI_FORCE_BRUTE_SINGLE: if set non-zero, route ALL single-tag queries to
  // brute (exact scan over primary's posting). Trades QPS for recall; recommended
  // when HAMCG_single beam-search's order-noise gap limits achievable recall.
  const bool FORCE_BRUTE_SINGLE = std::getenv("BCI_FORCE_BRUTE_SINGLE") != nullptr &&
                                   std::atoi(std::getenv("BCI_FORCE_BRUTE_SINGLE")) != 0;
  const char* query_mode_raw = std::getenv("BCI_QUERY_MODE");
  const std::string query_mode =
      (query_mode_raw != nullptr && query_mode_raw[0] != '\0')
          ? std::string(query_mode_raw)
          : "mixed";
  if (query_mode != "mixed" && query_mode != "two_tag" &&
      query_mode != "single_tag") {
    throw std::runtime_error("BCI_QUERY_MODE must be mixed, two_tag, or single_tag");
  }
  printf("[query mode] %s\n", query_mode.c_str());
  // BCI_BRUTE_SINGLE_THRESH: primary-tag size threshold below which single-tag
  // queries are routed to brute. Default 200000 (matches brute_conj_thresh).
  int64_t brute_single_thresh = std::getenv("BCI_BRUTE_SINGLE_THRESH") ?
      std::atoll(std::getenv("BCI_BRUTE_SINGLE_THRESH")) : 200000LL;
  printf("[single route] brute_single_thresh=%ld force_brute=%d\n",
         (long)brute_single_thresh, FORCE_BRUTE_SINGLE);
  for (int i = 0; i < n_q; ++i) {
    int qid = qid_lo + i;
    int64_t s = qm.row_offsets[qid], e = qm.row_offsets[qid+1];
    int n_tags = (int)(e - s);
    if (n_tags == 1) {
      if (query_mode == "two_tag") {
        q_route[i] = -1;
        continue;
      }
      int32_t t = qm.row_indices[s];
      int64_t ft = bmt.row_offsets[t+1] - bmt.row_offsets[t];
      bool small_enough = (ft <= brute_single_thresh);
      if (shards.count(t) && !(FORCE_BRUTE_SINGLE || small_enough)) {
        q_route[i] = 0; q_primary[i] = t;  // HAMCG beam_search
      } else {
        q_route[i] = 2; q_primary[i] = t;  // brute (q_secondary stays -1 → no filter)
      }
    } else if (n_tags == 2) {
      if (query_mode == "single_tag") {
        q_route[i] = -1;
        continue;
      }
      int32_t t1 = qm.row_indices[s], t2 = qm.row_indices[s+1];
      int64_t f1 = bmt.row_offsets[t1+1] - bmt.row_offsets[t1];
      int64_t f2 = bmt.row_offsets[t2+1] - bmt.row_offsets[t2];
      auto orientation = canonical_pair_orientation(t1, t2);
      int32_t small_t = orientation.first;
      int32_t large_t = orientation.second;
      int64_t small_size = std::min(f1, f2);
      // KEY FIX (per per-route diag: HAMCG_conj recall 0.79 — catastrophic):
      // If smaller tag's posting is small enough to brute, do exact intersection
      // scan instead of imprecise HAMCG_conj (sub_via_single + post-filter).
      // Threshold 200K = brute cost ~40ms per query but recall ~1.0.
      // Trade QPS for recall to surpass ParlayIVF.
      if (small_size <= brute_conj_thresh) {
        q_route[i] = 2; q_primary[i] = small_t; q_secondary[i] = large_t;  // brute exact
      } else if (shards.count(small_t)) {
        q_route[i] = 1; q_primary[i] = small_t; q_secondary[i] = large_t;
      } else if (shards.count(large_t)) {
        q_route[i] = 1; q_primary[i] = large_t; q_secondary[i] = small_t;
      } else {
        q_route[i] = 2; q_primary[i] = small_t; q_secondary[i] = large_t;
      }
      // PROXY: per-query adaptive target_points.
      // For SMALL primary: use all points (no artificial cap, scan is small anyway).
      // For MEDIUM primary: standard global target_pts.
      // For LARGE primary (>500K): scale up to catch boundary tail.
      int64_t primary_size = (q_route[i] == 1) ?
                             (bmt.row_offsets[q_primary[i]+1] - bmt.row_offsets[q_primary[i]]) : small_size;
      if (primary_size < 50000)        q_tpts[i] = (int)primary_size;  // use all
      else if (primary_size < 500000)  q_tpts[i] = target_pts;
      else                              q_tpts[i] = (int)std::min((int64_t)100000, (int64_t)target_pts * 2);
    } else {
      q_route[i] = -1;
    }
  }
  int n_h_single = 0, n_h_conj = 0, n_brute = 0, n_skip = 0;
  for (int i = 0; i < n_q; ++i) {
    if      (q_route[i] == 0) ++n_h_single;
    else if (q_route[i] == 1) ++n_h_conj;
    else if (q_route[i] == 2) ++n_brute;
    else                       ++n_skip;
  }
  printf("[route] HAMCG_single=%d HAMCG_conj=%d brute=%d skip=%d (of %d)\n",
         n_h_single, n_h_conj, n_brute, n_skip, n_q);
  if (reuse_beam_workspace &&
      (n_h_conj != 0 || use_clusters != 0 ||
       single_retry_boundary_ratio != 0.0 ||
       single_alt_retry_boundary_ratio != 0.0 ||
       single_multi_start != 1)) {
    throw std::runtime_error(
        "reusable beam workspace currently requires single graph routes, "
        "no cluster route, no graph/alternate retry, and one start");
  }
  if (dual_heap_beam &&
      (n_h_conj != 0 || use_clusters != 0 ||
       single_retry_boundary_ratio != 0.0 ||
       single_alt_retry_boundary_ratio != 0.0 ||
       single_multi_start != 1 || K <= 0)) {
    throw std::runtime_error(
        "dual-heap beam currently requires k>0, single graph routes, "
        "no cluster route, no graph/alternate retry, and one start");
  }
  if (indexed_heap_beam &&
      (n_h_conj != 0 || use_clusters != 0 ||
       single_retry_boundary_ratio != 0.0 ||
       single_alt_retry_boundary_ratio != 0.0 ||
       single_multi_start != 1 || K <= 0)) {
    throw std::runtime_error(
        "indexed-heap beam currently requires k>0, single graph routes, "
        "no cluster route, no graph/alternate retry, and one start");
  }
  if (fixed_frontier_batched_l2 &&
      (n_h_conj != 0 || use_clusters != 0 ||
       single_retry_boundary_ratio != 0.0 ||
       single_alt_retry_boundary_ratio != 0.0 ||
       single_multi_start != 1 || K <= 0)) {
    throw std::runtime_error(
        "fixed-frontier batched beam currently requires k>0, single graph "
        "routes, no cluster route, no graph/alternate retry, and one start");
  }
  if (batched_l2_beam &&
      (n_h_conj != 0 || use_clusters != 0 ||
       !batched_l2_retry_modes_supported(
           single_retry_boundary_ratio != 0.0,
           single_alt_retry_boundary_ratio != 0.0) ||
#ifndef BCI_ENABLE_BATCHED_MULTI_START_DIAGNOSTIC
       single_multi_start != 1 || K <= 0)) {
#else
       K <= 0)) {
#endif
    throw std::runtime_error(
        "batched-L2 beam currently requires k>0, single graph routes, "
        "no cluster route, and no alternate-entry retry");
  }
  std::vector<int> q_effective_beam(n_q, 0);
  std::vector<double> q_effective_cut(n_q, 0.0);
  std::vector<int> q_effective_starts(n_q, 0);
  int n_high_beam_queries = 0;
  int n_risk_beam_queries = 0;
  int n_multi_start_queries = 0;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == 0) {
      const int64_t support =
          bmt.row_offsets[q_primary[i] + 1] - bmt.row_offsets[q_primary[i]];
      const auto tier = select_single_search_tier(
          q_primary[i], support, beam, hamcg_cut,
          single_high_beam, single_high_cut,
          single_high_beam_min_support, high_beam_tags,
          single_risk_beam, single_risk_cut,
          single_risk_beam_tags);
      q_effective_beam[i] = tier.beam;
      q_effective_cut[i] = tier.cut;
      q_effective_starts[i] =
          select_single_multi_start(
              support, single_multi_start,
              single_multi_start_min_support) ? single_multi_start : 1;
      n_high_beam_queries += q_effective_beam[i] != beam ||
                             q_effective_cut[i] != hamcg_cut;
      n_risk_beam_queries += tier.risk_effort;
      n_multi_start_queries += q_effective_starts[i] > 1;
    } else if (q_route[i] == 1) {
      q_effective_beam[i] = beam;
      q_effective_cut[i] = hamcg_cut;
      q_effective_starts[i] = 1;
    }
  }
  printf("[single adaptive beam routes] active_high_effort_queries=%d\n",
         n_high_beam_queries);
  printf("[single risk beam routes] active_risk_queries=%d\n",
         n_risk_beam_queries);
  printf("[single multi-start routes] active_multi_start_queries=%d\n",
         n_multi_start_queries);
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
  const ResidualLandmarkDirectory residual_landmark_directory =
      load_residual_landmark_directory(
          residual_landmark_coverage_path);
  printf(
      "[residual landmark directory] tag=35 crossfit=1 "
      "half0_entries=%zu half1_entries=%zu\n",
      residual_landmark_directory.landmarks_by_development_half[0].size(),
      residual_landmark_directory.landmarks_by_development_half[1].size());
#endif
  const int n_active = n_q - n_skip;
  if (n_active <= 0) {
    throw std::runtime_error("query mode selected no active queries");
  }

  // The transposed CSR is constructed by visiting points in increasing global
  // ID, so every posting is sorted.  Verify exactly the postings on which the
  // optional merge semi-join relies before entering the timed region.
  if (secondary_membership == "posting_merge") {
    std::unordered_set<int32_t> merge_primary_tags;
    std::unordered_set<int32_t> merge_secondary_tags;
    for (int i = 0; i < n_q; ++i) {
      if (q_route[i] == 2 && q_secondary[i] >= 0) {
        merge_primary_tags.insert(q_primary[i]);
        merge_secondary_tags.insert(q_secondary[i]);
      }
    }
    auto assert_sorted_posting = [&](int32_t tag) {
      const int32_t* begin = bmt.row_indices.get() + bmt.row_offsets[tag];
      const int32_t* end = bmt.row_indices.get() + bmt.row_offsets[tag + 1];
      if (!std::is_sorted(begin, end)) {
        throw std::runtime_error("posting-merge requires sorted global IDs");
      }
      for (const int32_t* cursor = begin + (begin != end); cursor < end;
           ++cursor) {
        if (*cursor <= *(cursor - 1)) {
          throw std::runtime_error(
              "posting-merge requires strictly increasing global IDs");
        }
      }
    };
    for (int32_t tag : merge_primary_tags) assert_sorted_posting(tag);
    for (int32_t tag : merge_secondary_tags) assert_sorted_posting(tag);
    printf("[posting merge audit] primary_tags=%zu secondary_tags=%zu "
           "sorted=1 strictly_increasing=1\n",
           merge_primary_tags.size(), merge_secondary_tags.size());
  }

  // Exact two-tag posting cache for brute conjunctions.
  //
  // The tested YFCC two-tag workload has small primary postings, so the old
  // brute path scanned every point in the smaller tag and rejected most of them
  // by secondary bitvec.  Under a frozen vocabulary, materialise and pack every
  // admitted A∩B before serving the test batch; an unseen pair takes the ordinary
  // exact smallest-posting scan.  This is exact, two-tag-only, and leaves
  // single-tag brute scans on their original path.
  const bool USE_PAIR_POSTING_CACHE =
      std::getenv("BCI_PAIR_POSTING_CACHE") == nullptr ||
      std::atoi(std::getenv("BCI_PAIR_POSTING_CACHE")) != 0;
  // Optional frozen workload vocabulary.  When set, only pairs listed in this
  // file may be materialised; test-query pairs absent from the file fall back
  // to the ordinary smallest-posting scan.  Each non-comment line is "tag_a
  // tag_b".  Both orientations are admitted because the runtime chooses the
  // primary atom by base cardinality.  Leaving the variable unset preserves
  // legacy retrospective runs; prospective evaluations must set it.
  const char* pair_vocab_path = std::getenv("BCI_PAIR_VOCAB_FILE");
  const bool HAS_FROZEN_PAIR_VOCAB = pair_vocab_path != nullptr && pair_vocab_path[0] != '\0';
  std::unordered_set<uint64_t> logical_pair_keys;
  std::vector<std::pair<int32_t, int32_t>> frozen_logical_pairs;
  size_t allowed_logical_pairs = 0;
  if (HAS_FROZEN_PAIR_VOCAB) {
    std::ifstream vf(pair_vocab_path);
    if (!vf) throw std::runtime_error(std::string("cannot open BCI_PAIR_VOCAB_FILE: ") + pair_vocab_path);
    std::string line;
    while (std::getline(vf, line)) {
      auto first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos || line[first] == '#') continue;
      std::istringstream iss(line);
      int64_t a64 = -1, b64 = -1;
      if (!(iss >> a64 >> b64) || a64 < 0 || b64 < 0 ||
          a64 > std::numeric_limits<int32_t>::max() ||
          b64 > std::numeric_limits<int32_t>::max() || a64 == b64) {
        throw std::runtime_error(std::string("invalid pair-vocabulary line: ") + line);
      }
      if (a64 >= bmt.n_filters || b64 >= bmt.n_filters) {
        throw std::runtime_error(std::string("pair-vocabulary tag out of range: ") + line);
      }
      int32_t a = (int32_t)a64, b = (int32_t)b64;
      int32_t lo_id = std::min(a, b), hi_id = std::max(a, b);
      if (logical_pair_keys.insert(pair_key(lo_id, hi_id)).second) {
        ++allowed_logical_pairs;
        frozen_logical_pairs.push_back({lo_id, hi_id});
      }
    }
    printf("[pair vocabulary] frozen file=%s logical_pairs=%zu\n",
           pair_vocab_path, allowed_logical_pairs);
  } else {
    printf("[pair vocabulary] legacy query-batch discovery (not prospective)\n");
  }
	  FlatPairPostings pair_postings;
	  pair_postings.layout = pair_vector_layout;
	  pair_postings.aligned_dim = aligned_dim;
	  if (USE_PAIR_POSTING_CACHE) {
    auto t_pair0 = std::chrono::steady_clock::now();
    std::unordered_set<uint64_t> pair_key_set;
    if (HAS_FROZEN_PAIR_VOCAB) {
      // Materialise the complete frozen vocabulary, not merely the subset
      // requested by this test batch.  This prevents query-batch-dependent
      // build work or memory accounting.
      for (const auto& pair : frozen_logical_pairs) {
        int32_t a = pair.first, b = pair.second;
        auto orientation = canonical_pair_orientation(a, b);
        pair_key_set.insert(pair_key(orientation.first, orientation.second));
      }
      printf("[pair vocabulary materialization] all %zu frozen pairs\n",
             pair_key_set.size());
    } else {
      for (int i = 0; i < n_q; ++i) {
        if (q_route[i] == 2 && q_secondary[i] >= 0) {
          pair_key_set.insert(pair_key(q_primary[i], q_secondary[i]));
        }
      }
    }
    std::vector<uint64_t> pair_keys(pair_key_set.begin(), pair_key_set.end());
    std::sort(pair_keys.begin(), pair_keys.end());
    std::vector<PairPosting> pair_values(pair_keys.size());

	    parlay::parallel_for(0, pair_keys.size(), [&](size_t pi) {
	      int32_t primary = pair_key_first(pair_keys[pi]);
	      int32_t secondary = pair_key_second(pair_keys[pi]);
	      int64_t lo = bmt.row_offsets[primary];
	      int64_t hi = bmt.row_offsets[primary + 1];
	      const size_t primary_size = (size_t)(hi - lo);
	      if (pair_vector_layout == PairVectorLayout::PrimaryOffset) {
	        auto primary_pack_it = packed_cold.find(primary);
	        if (primary_pack_it == packed_cold.end() ||
	            primary_pack_it->second.size() !=
	                primary_size * aligned_dim) {
	          throw std::runtime_error(
	              "primary-offset layout requires an exact common primary pack");
	        }
	      }
	      auto bv_it = bitvecs.find(secondary);
	      const uint64_t* secondary_bv =
	          bv_it == bitvecs.end() ? nullptr : bv_it->second.data();
	      int64_t secondary_cursor = bmt.row_offsets[secondary];
	      const int64_t secondary_end = bmt.row_offsets[secondary + 1];
	      auto keep = [&](int32_t global_id) {
	        if (secondary_bv != nullptr) {
	          return (secondary_bv[global_id >> 6] &
	                  (1ULL << (global_id & 63))) != 0ULL;
	        }
	        if (secondary_membership == "posting_merge") {
	          while (secondary_cursor < secondary_end &&
	                 bmt.row_indices[secondary_cursor] < global_id) {
	            ++secondary_cursor;
	          }
	          return secondary_cursor < secondary_end &&
	                 bmt.row_indices[secondary_cursor] == global_id;
	        }
	        return (bool)bm.match(global_id, secondary);
	      };
	      auto base_vector = [&](int32_t global_id) -> const uint8_t* {
	        return base[global_id].get();
	      };
	      pair_values[pi] = encode_pair_posting(
	          primary, bmt.row_indices.get() + lo, primary_size, aligned_dim,
	          pair_vector_layout, keep, base_vector);
	    });

	    size_t pair_support_total = 0;
	    pair_postings.keys = pair_keys;
	    pair_postings.row_offsets.assign(pair_keys.size() + 1, 0);
	    for (size_t pi = 0; pi < pair_keys.size(); ++pi) {
	      size_t support = pair_values[pi].ids.size() +
	                       pair_values[pi].primary_offsets.size();
	      pair_support_total += support;
	      pair_postings.row_offsets[pi + 1] = pair_support_total;
	    }
	    if (pair_vector_layout == PairVectorLayout::PairPacked) {
	      pair_postings.ids.resize(pair_support_total);
	      pair_postings.packed.resize(pair_support_total * aligned_dim);
	    } else {
	      pair_postings.primary_offsets.resize(pair_support_total);
	    }
	    for (size_t pi = 0; pi < pair_keys.size(); ++pi) {
	      size_t begin = (size_t)pair_postings.row_offsets[pi];
	      size_t end = (size_t)pair_postings.row_offsets[pi + 1];
	      const PairPosting& source = pair_values[pi];
	      if (pair_vector_layout == PairVectorLayout::PairPacked) {
	        if (source.ids.size() != end - begin ||
	            source.packed.size() != (end - begin) * aligned_dim) {
	          throw std::runtime_error("flat pair-packed source shape mismatch");
	        }
	        std::copy(source.ids.begin(), source.ids.end(),
	                  pair_postings.ids.begin() + begin);
	        std::memcpy(
	            pair_postings.packed.data() + begin * aligned_dim,
	            source.packed.data(), source.packed.size());
	      } else {
	        if (source.primary_offsets.size() != end - begin) {
	          throw std::runtime_error(
	              "flat primary-offset source shape mismatch");
	        }
	        std::copy(source.primary_offsets.begin(),
	                  source.primary_offsets.end(),
	                  pair_postings.primary_offsets.begin() + begin);
	      }
	    }
	    std::vector<PairPosting>().swap(pair_values);
	    const size_t pair_ids_total = pair_postings.ids.size();
	    const size_t pair_id_capacity_total = pair_postings.ids.capacity();
	    const size_t pair_offsets_total =
	        pair_postings.primary_offsets.size();
	    const size_t pair_offset_capacity_total =
	        pair_postings.primary_offsets.capacity();
	    const size_t pair_bytes_total = pair_postings.packed.size();
	    const size_t pair_packed_capacity_total =
	        pair_postings.packed.capacity();
	    if (pair_postings.keys.capacity() != pair_postings.keys.size() ||
	        pair_postings.row_offsets.capacity() !=
	            pair_postings.row_offsets.size() ||
	        pair_id_capacity_total != pair_ids_total ||
	        pair_offset_capacity_total != pair_offsets_total ||
	        pair_packed_capacity_total != pair_bytes_total) {
	      throw std::runtime_error(
	          "flat pair directory did not use exact vector capacities");
	    }
	    validate_flat_pair_postings_structure(pair_postings);
	    if (!std::is_sorted(pair_postings.keys.begin(),
	                        pair_postings.keys.end()) ||
	        std::adjacent_find(pair_postings.keys.begin(),
	                           pair_postings.keys.end()) !=
	            pair_postings.keys.end() ||
	        pair_postings.row_offsets.empty() ||
	        pair_postings.row_offsets.front() != 0 ||
	        pair_postings.row_offsets.back() != pair_support_total) {
	      throw std::runtime_error("flat pair directory structural audit failed");
	    }
	    std::atomic<uint64_t> audited_pair_memberships{0};
	    parlay::parallel_for(0, pair_postings.size(), [&](size_t pi) {
	      uint64_t key = pair_postings.keys[pi];
	      int32_t primary = pair_key_first(key);
	      int32_t secondary = pair_key_second(key);
	      auto expected_orientation =
	          canonical_pair_orientation(primary, secondary);
	      if (expected_orientation.first != primary ||
	          expected_orientation.second != secondary) {
	        throw std::runtime_error(
	            "flat pair key does not use canonical support orientation");
	      }
	      size_t begin = (size_t)pair_postings.row_offsets[pi];
	      size_t end = (size_t)pair_postings.row_offsets[pi + 1];
	      int64_t primary_start = bmt.row_offsets[primary];
	      size_t primary_size =
	          (size_t)(bmt.row_offsets[primary + 1] - primary_start);
	      int32_t previous_global = -1;
	      for (size_t k = begin; k < end; ++k) {
	        int32_t global_id = -1;
	        if (pair_vector_layout == PairVectorLayout::PairPacked) {
	          global_id = pair_postings.ids[k];
	          if (global_id < 0 || (size_t)global_id >= base.size()) {
	            throw std::runtime_error(
	                "flat pair-packed global ID is out of range");
	          }
	          if (std::memcmp(
	                  pair_postings.packed.data() + k * aligned_dim,
	                  base[global_id].get(), aligned_dim) != 0) {
	            throw std::runtime_error(
	                "flat pair-packed vector/base audit failed");
	          }
	        } else {
	          uint32_t offset = pair_postings.primary_offsets[k];
	          if (offset >= primary_size) {
	            throw std::runtime_error(
	                "flat primary-offset is out of bounds");
	          }
	          global_id = bmt.row_indices[primary_start + offset];
	        }
	        if (global_id <= previous_global ||
	            !bm.match(global_id, primary) ||
	            !bm.match(global_id, secondary)) {
	          throw std::runtime_error(
	              "flat pair membership/order/predicate audit failed");
	        }
	        previous_global = global_id;
	      }
	      audited_pair_memberships.fetch_add(
	          end - begin, std::memory_order_relaxed);
	    });
	    if (audited_pair_memberships.load(std::memory_order_relaxed) !=
	        pair_support_total) {
	      throw std::runtime_error("flat pair audit support total mismatch");
	    }
	    const size_t pair_directory_bytes =
	        pair_postings.keys.capacity() * sizeof(uint64_t) +
	        pair_postings.row_offsets.capacity() * sizeof(uint64_t);
	    const size_t pair_payload_capacity_bytes =
	        pair_id_capacity_total * sizeof(int32_t) +
	        pair_offset_capacity_total * sizeof(uint32_t) +
	        pair_packed_capacity_total;
	    const size_t pair_derived_capacity_bytes =
	        pair_directory_bytes + pair_payload_capacity_bytes;
	    const size_t pair_container_header_bytes =
	        sizeof(pair_postings);
	    auto usable_bytes = [](const auto& values) -> size_t {
	      return values.capacity() == 0
	                 ? 0
	                 : malloc_usable_size(
	                       const_cast<void*>(
	                           static_cast<const void*>(values.data())));
	    };
	    const size_t pair_allocator_usable_bytes =
	        usable_bytes(pair_postings.keys) +
	        usable_bytes(pair_postings.row_offsets) +
	        usable_bytes(pair_postings.ids) +
	        usable_bytes(pair_postings.primary_offsets) +
	        usable_bytes(pair_postings.packed);
	    if (pair_allocator_usable_bytes < pair_derived_capacity_bytes) {
	      throw std::runtime_error(
	          "allocator usable bytes are below vector capacities");
	    }
	    const size_t pair_physical_derived_bytes =
	        pair_allocator_usable_bytes + pair_container_header_bytes;
	    printf("[flat pair physical audit] pairs=%zu memberships=%zu "
	           "predicates=1 order=1 mapping=1\n",
	           pair_postings.size(), pair_support_total);
	    auto t_pair1 = std::chrono::steady_clock::now();
	    printf("[built %zu exact pair postings in %.2fs, %.1fM support, "
	           "%.1fMB packed, layout=%s]\n",
	           pair_postings.size(),
	           std::chrono::duration<double>(t_pair1-t_pair0).count(),
	           pair_support_total / 1e6, pair_bytes_total / 1e6,
	           pair_vector_layout_name(pair_vector_layout));
	    printf("[pair allocation] ids_size_bytes=%zu ids_capacity_bytes=%zu "
	           "offsets_size_bytes=%zu offsets_capacity_bytes=%zu "
	           "packed_size_bytes=%zu packed_capacity_bytes=%zu "
	           "directory_bytes=%zu derived_capacity_bytes=%zu "
	           "container_header_bytes=%zu allocator_usable_bytes=%zu "
	           "physical_derived_bytes=%zu "
	           "layout=%s\n",
	           pair_ids_total * sizeof(int32_t),
	           pair_id_capacity_total * sizeof(int32_t),
	           pair_offsets_total * sizeof(uint32_t),
	           pair_offset_capacity_total * sizeof(uint32_t),
	           pair_bytes_total, pair_packed_capacity_total,
	           pair_directory_bytes, pair_derived_capacity_bytes,
	           pair_container_header_bytes, pair_allocator_usable_bytes,
	           pair_physical_derived_bytes,
	           pair_vector_layout_name(pair_vector_layout));
	    const char* pair_inventory_path = std::getenv("BCI_PAIR_INVENTORY_CSV");
    if (pair_inventory_path != nullptr && pair_inventory_path[0] != '\0') {
      if (std::filesystem::exists(pair_inventory_path)) {
        throw std::runtime_error("refusing to overwrite pair inventory");
      }
      std::ofstream inventory(pair_inventory_path);
      if (!inventory) throw std::runtime_error("cannot create pair inventory");
	      inventory << "primary_tag,secondary_tag,support,ids_bytes,"
	                   "ids_capacity_bytes,offsets_bytes,"
	                   "offsets_capacity_bytes,packed_bytes,"
	                   "packed_capacity_bytes,total_bytes,layout,"
	                   "flat_index\n";
	      for (size_t pi = 0; pi < pair_postings.keys.size(); ++pi) {
	        uint64_t key = pair_postings.keys[pi];
	        int32_t primary = pair_key_first(key);
	        int32_t secondary = pair_key_second(key);
	        size_t support = pair_postings.support(pi);
	        uint64_t ids_bytes =
	            pair_vector_layout == PairVectorLayout::PairPacked
	                ? support * sizeof(int32_t)
	                : 0;
	        uint64_t ids_capacity_bytes = ids_bytes;
	        uint64_t offsets_bytes =
	            pair_vector_layout == PairVectorLayout::PrimaryOffset
	                ? support * sizeof(uint32_t)
	                : 0;
	        uint64_t offsets_capacity_bytes = offsets_bytes;
	        uint64_t packed_bytes =
	            pair_vector_layout == PairVectorLayout::PairPacked
	                ? support * aligned_dim
	                : 0;
	        uint64_t packed_capacity_bytes = packed_bytes;
	        inventory << primary << ',' << secondary << ',' << support << ','
	                  << ids_bytes << ',' << ids_capacity_bytes << ','
	                  << offsets_bytes << ',' << offsets_capacity_bytes << ','
	                  << packed_bytes << ',' << packed_capacity_bytes << ','
	                  << (ids_bytes + offsets_bytes + packed_bytes) << ','
	                  << pair_vector_layout_name(pair_vector_layout) << ','
	                  << pi << '\n';
      }
      inventory.close();
      printf("[pair inventory] wrote %s rows=%zu\n",
             pair_inventory_path, pair_keys.size());
    }
    if (HAS_FROZEN_PAIR_VOCAB) {
      size_t eligible_queries = 0, covered_queries = 0;
      for (int i = 0; i < n_q; ++i) {
        if (q_route[i] == 2 && q_secondary[i] >= 0) {
          ++eligible_queries;
	          covered_queries += pair_postings.contains(
	              pair_key(q_primary[i], q_secondary[i]));
        }
      }
      printf("[pair vocabulary coverage] cached_queries=%zu eligible_conjunctions=%zu fraction=%.6f\n",
             covered_queries, eligible_queries,
             eligible_queries ? (double)covered_queries / (double)eligible_queries : 0.0);
    }
  }
  std::unordered_set<uint64_t>().swap(logical_pair_keys);
  std::vector<std::pair<int32_t, int32_t>>().swap(frozen_logical_pairs);
  if (!logical_pair_keys.empty() || logical_pair_keys.bucket_count() > 1 ||
      !frozen_logical_pairs.empty() ||
      frozen_logical_pairs.capacity() != 0) {
    throw std::runtime_error(
        "pair vocabulary staging state survived materialization");
  }
  printf("[pair vocabulary staging released] logical_pairs=0 "
         "vector_capacity=0 buckets=%zu\n",
         logical_pair_keys.bucket_count());

  // -- Run queries -----------------------------------------------------------
  std::vector<std::vector<int32_t>> results(n_q);
  std::vector<double> latencies(n_q, 0.0);
  std::vector<size_t> q_graph_dist_comps(n_q, 0);
  std::vector<size_t> q_graph_visited(n_q, 0);
  std::vector<size_t> q_graph_frontier_size(n_q, 0);
  std::vector<size_t> q_graph_staged_candidates(n_q, 0);
  std::vector<size_t> q_graph_prefix_rejected(n_q, 0);
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
  std::vector<size_t> q_residual_landmark_scores(n_q, 0);
  std::vector<int32_t> q_residual_landmark_selected(n_q, -1);
#endif
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
  std::vector<size_t> q_eviction_spill_triggered(n_q, 0);
  std::vector<size_t> q_eviction_spill_exact_records(n_q, 0);
  std::vector<size_t> q_eviction_spill_width_records(n_q, 0);
  std::vector<size_t> q_eviction_spill_cutoff_records(n_q, 0);
  std::vector<size_t> q_eviction_spill_prefix_records(n_q, 0);
  std::vector<size_t> q_eviction_spill_metric_excluded(n_q, 0);
  std::vector<size_t> q_eviction_spill_handoff_records(n_q, 0);
  std::vector<size_t> q_eviction_spill_extra_comps(n_q, 0);
  std::vector<size_t> q_eviction_spill_new_visits(n_q, 0);
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
  std::vector<size_t> q_q4_exact_rerank_candidates(n_q, 0);
#endif
  std::vector<double> q_graph_boundary_ratio(n_q, 0.0);
  std::vector<int> q_graph_retried(n_q, 0);
  std::vector<size_t> q_graph_retry_dist_comps(n_q, 0);
  std::vector<size_t> q_graph_retry_cache_hits(n_q, 0);
  std::vector<size_t> q_graph_retry_uncached_candidates(n_q, 0);
  std::vector<size_t> q_graph_retry_exact_completions(n_q, 0);
  std::vector<size_t> q_graph_retry_uncached_prefix_rejections(n_q, 0);
  std::vector<int> q_graph_exact_retried(n_q, 0);
  std::vector<int> q_graph_alt_retried(n_q, 0);
  std::vector<size_t> q_graph_alt_retry_dist_comps(n_q, 0);
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
  std::vector<size_t> q_graph_top10_last_change_visit(n_q, 0);
  std::vector<size_t> q_graph_top10_last_change_dist_comps(n_q, 0);
  std::vector<size_t> q_graph_top10_stability_age(n_q, 0);
  std::vector<size_t> q_graph_top10_changed_steps_last16(n_q, 0);
  std::vector<size_t> q_graph_top10_entries_last16(n_q, 0);
  std::vector<size_t> q_graph_top10_total_changed_steps(n_q, 0);
  std::vector<size_t> q_graph_top10_total_entries(n_q, 0);
  std::vector<size_t> q_graph_top10_final_size(n_q, 0);
  std::vector<double> q_graph_top10_margin_abs(n_q, 0.0);
#endif
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
  std::vector<size_t> q_cr_base_cache_hits(n_q, 0);
  std::vector<size_t> q_cr_base_exact_misses(n_q, 0);
  std::vector<size_t> q_cr_base_exact_completions(n_q, 0);
  std::vector<size_t> q_cr_base_uncached_prefix_rejections(n_q, 0);
  std::vector<size_t> q_cr_probe_logical_comps(n_q, 0);
  std::vector<size_t> q_cr_probe_cache_hits(n_q, 0);
  std::vector<size_t> q_cr_probe_exact_misses(n_q, 0);
  std::vector<size_t> q_cr_probe_exact_completions(n_q, 0);
  std::vector<size_t> q_cr_probe_uncached_prefix_rejections(n_q, 0);
  std::vector<size_t> q_cr_probe_visited(n_q, 0);
  std::vector<size_t> q_cr_probe_top10_intersection(n_q, 0);
  std::vector<size_t> q_cr_probe_top10_entered(n_q, 0);
  std::vector<size_t> q_cr_probe_top10_dropped(n_q, 0);
  std::vector<double> q_cr_base_d10(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> q_cr_base_frontier_margin_abs(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> q_cr_probe_d10(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> q_cr_probe_frontier_margin_abs(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> q_cr_probe_min_entered_margin_abs(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> q_cr_probe_max_entered_margin_abs(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<int> q_cr_repair_ran(n_q, 0);
  std::vector<size_t> q_cr_repair_logical_comps(n_q, 0);
  std::vector<size_t> q_cr_repair_cache_hits(n_q, 0);
  std::vector<size_t> q_cr_repair_exact_misses(n_q, 0);
  std::vector<size_t> q_cr_repair_exact_completions(n_q, 0);
  std::vector<size_t> q_cr_repair_uncached_prefix_rejections(n_q, 0);
  std::vector<size_t> q_cr_repair_visited(n_q, 0);
  std::vector<size_t> q_cr_repair_top10_intersection(n_q, 0);
  std::vector<size_t> q_cr_repair_top10_entered(n_q, 0);
  std::vector<size_t> q_cr_repair_top10_dropped(n_q, 0);
  std::vector<double> q_cr_repair_d10(
      n_q, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> q_cr_repair_frontier_margin_abs(
      n_q, std::numeric_limits<double>::quiet_NaN());
#endif

  // Batch execution: batch queries per
  // shard so each shard graph + ThinSubPR + subset stay cache-hot across
  // its bucket. Restructure parallelism: parallel_for over shard buckets
  // (sorted by descending bucket size for load balance), each bucket
  // processes its queries serially.
  std::unordered_map<int32_t, std::vector<int>> shard_buckets;
  std::vector<int> pair_brute_ids;
  std::vector<int> brute_ids;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == 0 || q_route[i] == 1) shard_buckets[q_primary[i]].push_back(i);
    else if (q_route[i] == 2) {
      // Keep one state-independent route-2 scheduling phase in input-qid
      // order.  The per-query body chooses pair materialisation or the common
      // atom-posting fallback; selected policies must not change batch
      // partition/order or the number of fork/join barriers.
      brute_ids.push_back(i);
    }
  }
  const char* brute_query_order_raw =
      std::getenv("BCI_BRUTE_QUERY_ORDER");
  const std::string brute_query_order =
      (brute_query_order_raw && brute_query_order_raw[0] != '\0')
          ? brute_query_order_raw
          : "input";
  if (brute_query_order != "input" &&
      brute_query_order != "primary") {
    throw std::runtime_error(
        "BCI_BRUTE_QUERY_ORDER must be input or primary");
  }
  if (brute_query_order == "primary") {
    // Schedule-only development intervention: keep queries sharing the same
    // primary posting adjacent so the posting IDs and packed vectors can
    // remain cache-hot.  Query routes, support construction, distance
    // evaluation, tie-breaking, and result slots are unchanged.
    std::stable_sort(
        brute_ids.begin(), brute_ids.end(), [&](int lhs, int rhs) {
          if (q_primary[lhs] != q_primary[rhs]) {
            return q_primary[lhs] < q_primary[rhs];
          }
          return lhs < rhs;
        });
  }
  std::vector<std::pair<int32_t, std::vector<int>>> buckets;
  buckets.reserve(shard_buckets.size());
  for (auto& kv : shard_buckets) buckets.emplace_back(kv.first, std::move(kv.second));
  std::sort(buckets.begin(), buckets.end(),
    [](const auto& a, const auto& b) {
      if (a.second.size() != b.second.size()) {
        return a.second.size() > b.second.size();
      }
      return a.first < b.first;
    });
  const char* bucket_schedule_raw =
      std::getenv("BCI_SHARD_BUCKET_SCHEDULE");
  const std::string bucket_schedule =
      (bucket_schedule_raw && bucket_schedule_raw[0] != '\0')
          ? bucket_schedule_raw
          : "sorted";
  if (bucket_schedule != "sorted" && bucket_schedule != "lpt") {
    throw std::runtime_error(
        "BCI_SHARD_BUCKET_SCHEDULE must be sorted or lpt");
  }
  const char* bucket_weight_raw =
      std::getenv("BCI_SHARD_BUCKET_WEIGHT");
  const std::string bucket_weight =
      (bucket_weight_raw && bucket_weight_raw[0] != '\0')
          ? bucket_weight_raw
          : "query_count";
  if (bucket_weight != "query_count" &&
      bucket_weight != "effective_beam" &&
      bucket_weight != "calibrated_dist_comps") {
    throw std::runtime_error(
        "BCI_SHARD_BUCKET_WEIGHT must be query_count, effective_beam, "
        "or calibrated_dist_comps");
  }
  if (bucket_schedule != "lpt" && bucket_weight != "query_count") {
    throw std::runtime_error(
        "BCI_SHARD_BUCKET_WEIGHT is only active with LPT scheduling");
  }
  const bool split_overweight_buckets =
      std::getenv("BCI_SHARD_BUCKET_SPLIT_OVERWEIGHT") != nullptr &&
      std::atoi(std::getenv("BCI_SHARD_BUCKET_SPLIT_OVERWEIGHT")) != 0;
  if (split_overweight_buckets) {
#ifndef BCI_ENABLE_BUCKET_MICROBATCH_DIAGNOSTIC
    throw std::runtime_error(
        "BCI_SHARD_BUCKET_SPLIT_OVERWEIGHT requires the isolated "
        "bucket-microbatch build");
#else
    if (bucket_schedule != "lpt") {
      throw std::runtime_error(
          "BCI_SHARD_BUCKET_SPLIT_OVERWEIGHT requires LPT scheduling");
    }
    if (single_retry_distance_cache) {
      throw std::runtime_error(
          "bucket microbatching is incompatible with the tag-local retry "
          "distance cache");
    }
#endif
  }
  std::unordered_map<int32_t, double> calibrated_shard_costs;
  double calibrated_shard_cost_fallback = 0.0;
  int calibrated_shard_cost_base_beam = 0;
  if (bucket_weight == "calibrated_dist_comps") {
    const char* cost_path = std::getenv("BCI_SHARD_BUCKET_COST_FILE");
    if (cost_path == nullptr || cost_path[0] == '\0') {
      throw std::runtime_error(
          "calibrated_dist_comps requires BCI_SHARD_BUCKET_COST_FILE");
    }
    const char* cost_base_beam_raw =
        std::getenv("BCI_SHARD_BUCKET_COST_BASE_BEAM");
    if (cost_base_beam_raw == nullptr ||
        cost_base_beam_raw[0] == '\0') {
      throw std::runtime_error(
          "calibrated_dist_comps requires "
          "BCI_SHARD_BUCKET_COST_BASE_BEAM");
    }
    calibrated_shard_cost_base_beam =
        std::atoi(cost_base_beam_raw);
    if (calibrated_shard_cost_base_beam <= 0) {
      throw std::runtime_error(
          "BCI_SHARD_BUCKET_COST_BASE_BEAM must be positive");
    }
    std::ifstream cost_file(cost_path);
    if (!cost_file) {
      throw std::runtime_error(
          std::string("cannot open BCI_SHARD_BUCKET_COST_FILE: ") +
          cost_path);
    }
    std::vector<double> costs_for_fallback;
    std::string line;
    while (std::getline(cost_file, line)) {
      const auto first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos || line[first] == '#') continue;
      std::istringstream row(line);
      int64_t tag64 = -1;
      double mean_dist_comps = 0.0;
      if (!(row >> tag64 >> mean_dist_comps) ||
          tag64 < 0 || tag64 > std::numeric_limits<int32_t>::max() ||
          !(mean_dist_comps > 0.0) || !std::isfinite(mean_dist_comps)) {
        throw std::runtime_error(
            std::string("invalid calibrated shard cost row: ") + line);
      }
      const int32_t tag = static_cast<int32_t>(tag64);
      if (!calibrated_shard_costs.emplace(tag, mean_dist_comps).second) {
        throw std::runtime_error(
            "duplicate tag in BCI_SHARD_BUCKET_COST_FILE");
      }
      costs_for_fallback.push_back(mean_dist_comps);
    }
    if (costs_for_fallback.empty()) {
      throw std::runtime_error(
          "BCI_SHARD_BUCKET_COST_FILE contains no cost rows");
    }
    const size_t middle = costs_for_fallback.size() / 2;
    std::nth_element(
        costs_for_fallback.begin(),
        costs_for_fallback.begin() + middle,
        costs_for_fallback.end());
    calibrated_shard_cost_fallback = costs_for_fallback[middle];
    printf("[calibrated shard costs] tags=%zu fallback=median %.3f "
           "base_beam=%d source=development_distance_computations\n",
           calibrated_shard_costs.size(),
           calibrated_shard_cost_fallback,
           calibrated_shard_cost_base_beam);
  }
  const size_t graph_lane_count = std::min<size_t>(
      buckets.size(), std::max<long>(1, parlay::num_workers()));
  printf("[batched %zu shard buckets + %zu pair-brute queries + %zu brute queries]\n",
         buckets.size(), pair_brute_ids.size(), brute_ids.size());
  printf("[brute query order] mode=%s planning=outside_batch\n",
         brute_query_order.c_str());
  printf("[shard bucket schedule] mode=%s weight=%s lanes=%zu planning=timed\n",
         bucket_schedule.c_str(), bucket_weight.c_str(), graph_lane_count);
  printf("[shard bucket microbatch] split_overweight=%d "
         "threshold=prospective_mean_lane_load semantics=query_disjoint\n",
         (int)split_overweight_buckets);
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE
  // The isolated production adapter owns its output buffers across passes.
  // Reserve them before timing so neither exact arm performs a per-query heap
  // allocation; query semantics and the retained distance kernel are unchanged.
  for (int query_index : brute_ids) {
    results[query_index].reserve(static_cast<std::size_t>(K));
  }
  printf("[block16 output workspace] preallocated_queries=%zu capacity=%d\n",
         brute_ids.size(), K);
#endif

  constexpr size_t support_complement_fixed_scratch_capacity =
      yfcc_support_complement::kMaximumPrimarySupport;
  struct SupportComplementScratch {
    std::vector<uint16_t> local_offsets;
    std::vector<int32_t> global_ids;
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_PREFIX128_CASCADE
    std::vector<yfcc_pair_offset16_prefix128_cascade_v1::Survivor>
        prefix128_survivors;
    yfcc_pair_offset16_prefix128_cascade_v1::Counters prefix128_counters;
#endif
  };
  std::vector<SupportComplementScratch> support_complement_scratch(
      parlay::num_workers());
  std::vector<ReusableBeamWorkspace<Indx, float>>
      reusable_beam_workspaces(parlay::num_workers());
  std::vector<DualHeapBeamWorkspace<Indx, float>>
      dual_heap_beam_workspaces(parlay::num_workers());
  std::vector<IndexedHeapBeamWorkspace<Indx, float>>
      indexed_heap_beam_workspaces(parlay::num_workers());
#ifdef BCI_ENABLE_FIXED_FRONTIER_BATCHED_DIAGNOSTIC
  std::vector<FixedFrontierBatchedWorkspace<Indx, float>>
      fixed_frontier_batched_workspaces(parlay::num_workers());
#endif
  for (auto& scratch : support_complement_scratch) {
    scratch.local_offsets.reserve(
        support_complement_fixed_scratch_capacity);
    scratch.global_ids.reserve(
        support_complement_fixed_scratch_capacity);
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_PREFIX128_CASCADE
    scratch.prefix128_survivors.resize(
        support_complement_fixed_scratch_capacity);
#endif
  }
  if (conjunction_support_complement) {
    const uint8_t* base_begin = base[0].get();
    if (base.size() > 1 &&
        base[1].get() != base_begin + base.aligned_dimension()) {
      throw std::runtime_error(
          "support-complement base vectors are not contiguous");
    }
    printf("[support-complement scratch] workers=%zu "
           "fixed_query_independent_capacity=%zu\n",
           support_complement_scratch.size(),
           support_complement_fixed_scratch_capacity);
  }
  if (indexed_heap_beam) {
    size_t maximum_points = 1;
    size_t maximum_beam = 1;
    size_t maximum_visits = 1;
    size_t maximum_degree = 1;
    for (const auto& bucket : buckets) {
      const int32_t tag = bucket.first;
      const Shard& shard = *shards.at(tag);
      const int64_t support =
          bmt.row_offsets[tag + 1] - bmt.row_offsets[tag];
      const auto tier = select_single_search_tier(
          tag, support, beam, hamcg_cut,
          single_high_beam, single_high_cut,
          single_high_beam_min_support, high_beam_tags,
          single_risk_beam, single_risk_cut,
          single_risk_beam_tags);
      const long effective_beam = std::max<long>(
          single_pool, tier.beam);
      const long visit_limit = std::min<long>(
          static_cast<long>(shard.graph.size()),
          std::max<long>(100L * effective_beam, 100000L));
      maximum_points =
          std::max(maximum_points, shard.graph.size());
      maximum_beam = std::max(
          maximum_beam, static_cast<size_t>(effective_beam));
      maximum_visits = std::max(
          maximum_visits, static_cast<size_t>(visit_limit));
      maximum_degree = std::max(
          maximum_degree,
          static_cast<size_t>(shard.graph.max_degree()));
    }
    for (auto& workspace : indexed_heap_beam_workspaces) {
      workspace.prepare(
          maximum_points, maximum_beam,
          maximum_visits, maximum_degree);
    }
    printf("[indexed-heap prepared] workers=%zu max_points=%zu "
           "max_beam=%zu max_visits=%zu max_degree=%zu\n",
           indexed_heap_beam_workspaces.size(), maximum_points,
           maximum_beam, maximum_visits, maximum_degree);
  }
#ifdef BCI_ENABLE_FIXED_FRONTIER_BATCHED_DIAGNOSTIC
  if (fixed_frontier_batched_l2) {
    size_t maximum_points = 1;
    size_t maximum_beam = 1;
    size_t maximum_visits = 1;
    size_t maximum_degree = 1;
    for (const auto& bucket : buckets) {
      const int32_t tag = bucket.first;
      const Shard& shard = *shards.at(tag);
      const int64_t support =
          bmt.row_offsets[tag + 1] - bmt.row_offsets[tag];
      const auto tier = select_single_search_tier(
          tag, support, beam, hamcg_cut,
          single_high_beam, single_high_cut,
          single_high_beam_min_support, high_beam_tags,
          single_risk_beam, single_risk_cut,
          single_risk_beam_tags);
      const long effective_beam = std::max<long>(
          single_pool, tier.beam);
      const long visit_limit = std::min<long>(
          static_cast<long>(shard.graph.size()),
          std::max<long>(100L * effective_beam, 100000L));
      maximum_points =
          std::max(maximum_points, shard.graph.size());
      maximum_beam = std::max(
          maximum_beam, static_cast<size_t>(effective_beam));
      maximum_visits = std::max(
          maximum_visits, static_cast<size_t>(visit_limit));
      maximum_degree = std::max(
          maximum_degree,
          static_cast<size_t>(shard.graph.max_degree()));
    }
    for (auto& workspace : fixed_frontier_batched_workspaces) {
      workspace.prepare(
          maximum_points, maximum_beam,
          maximum_visits, maximum_degree);
    }
    printf("[fixed-frontier batched prepared] workers=%zu "
           "max_points=%zu max_beam=%zu max_visits=%zu max_degree=%zu\n",
           fixed_frontier_batched_workspaces.size(), maximum_points,
           maximum_beam, maximum_visits, maximum_degree);
  }
#endif

  // Query-local exact-distance replay is allocated once, before the timed
  // batch.  Only frozen retry tags receive a memo.  A bucket is processed by
  // one lane at a time, so its tag-local epoch is never shared concurrently.
  std::unordered_map<
      int32_t, std::unique_ptr<ExactDistanceMemo<Indx, float>>>
      single_retry_distance_memos;
  size_t single_retry_memo_domain_ids = 0;
  if (single_retry_distance_cache) {
    single_retry_distance_memos.reserve(single_retry_tags.size());
    for (int32_t tag : single_retry_tags) {
      const auto shard = shards.find(tag);
      if (shard == shards.end()) continue;
      const size_t domain_size = shard->second->graph.size();
      single_retry_memo_domain_ids += domain_size;
      single_retry_distance_memos.emplace(
          tag,
          std::make_unique<ExactDistanceMemo<Indx, float>>(
              domain_size));
    }
  }
  printf("[single retry distance-cache prepared] enabled=%d tags=%zu "
         "domain_ids=%zu payload_bytes=%zu allocation=outside_timed_batch\n",
         (int)single_retry_distance_cache,
         single_retry_distance_memos.size(),
         single_retry_memo_domain_ids,
         single_retry_memo_domain_ids *
             (sizeof(float) + sizeof(uint32_t)));

  printf("[bench start v3: batched per-shard parallel + serial-in-bucket]\n");
  if (n_runs > 1) printf("[in-process warmup: running query batch %d times back-to-back for warm-cache measurement]\n", n_runs);
  std::ofstream pass_csv;
  const char* pass_csv_path = std::getenv("BCI_PASS_CSV");
  if (pass_csv_path != nullptr && pass_csv_path[0] != '\0') {
    if (std::filesystem::exists(pass_csv_path)) {
      throw std::runtime_error("refusing to overwrite BCI_PASS_CSV");
    }
    pass_csv.open(pass_csv_path, std::ios::out | std::ios::trunc);
    if (!pass_csv) throw std::runtime_error("cannot create BCI_PASS_CSV");
    pass_csv << "pass,queries,threads,wall_seconds,qps,p50_ms,p95_ms,p99_ms,"
                "predictions_identical_to_first\n";
  }
  const char* pass_artifact_dir_raw = std::getenv("BCI_PASS_ARTIFACT_DIR");
  const bool has_pass_barrier =
      pass_artifact_dir_raw != nullptr && pass_artifact_dir_raw[0] != '\0';
  const char* defer_pass_reporting_raw =
      std::getenv("BCI_DEFER_PASS_REPORTING");
  const bool defer_pass_reporting =
      defer_pass_reporting_raw != nullptr &&
      std::atoi(defer_pass_reporting_raw) != 0;
  int deferred_warmup_passes = 0;
  if (defer_pass_reporting) {
    const char* warmup_raw = std::getenv("BCI_DEFER_WARMUP_PASSES");
    if (warmup_raw == nullptr || warmup_raw[0] == '\0') {
      throw std::runtime_error(
          "BCI_DEFER_PASS_REPORTING requires BCI_DEFER_WARMUP_PASSES");
    }
    deferred_warmup_passes = std::atoi(warmup_raw);
    if (deferred_warmup_passes < 1 || deferred_warmup_passes >= n_runs) {
      throw std::runtime_error("invalid BCI_DEFER_WARMUP_PASSES");
    }
  }
  if (defer_pass_reporting && has_pass_barrier) {
    throw std::runtime_error(
        "BCI_DEFER_PASS_REPORTING is incompatible with pass barrier");
  }
  std::filesystem::path pass_artifact_dir;
  std::string run_token;
  std::string gt_sha256_binding;
  std::string active_qids_sha256_binding;
  int pass_barrier_timeout_seconds = 900;
  if (has_pass_barrier) {
    if (pass_csv_path == nullptr || pass_csv_path[0] == '\0') {
      throw std::runtime_error("BCI_PASS_ARTIFACT_DIR requires BCI_PASS_CSV");
    }
    pass_artifact_dir = std::filesystem::path(pass_artifact_dir_raw);
    const char* run_token_raw = std::getenv("BCI_RUN_TOKEN");
    if (run_token_raw == nullptr || run_token_raw[0] == '\0') {
      throw std::runtime_error("BCI_PASS_ARTIFACT_DIR requires BCI_RUN_TOKEN");
    }
    run_token = run_token_raw;
    if (run_token.find('\n') != std::string::npos ||
        run_token.find('=') != std::string::npos) {
      throw std::runtime_error("invalid BCI_RUN_TOKEN");
    }
    const char* gt_sha256_raw = std::getenv("BCI_GT_SHA256");
    const char* active_qids_sha256_raw =
        std::getenv("BCI_ACTIVE_QIDS_SHA256");
    if (gt_sha256_raw == nullptr || std::strlen(gt_sha256_raw) != 64 ||
        active_qids_sha256_raw == nullptr ||
        std::strlen(active_qids_sha256_raw) != 64) {
      throw std::runtime_error(
          "pass barrier requires 64-hex GT and active-qid SHA256 bindings");
    }
    gt_sha256_binding = gt_sha256_raw;
    active_qids_sha256_binding = active_qids_sha256_raw;
    const char* timeout_raw = std::getenv("BCI_PASS_BARRIER_TIMEOUT_SECONDS");
    if (timeout_raw != nullptr && timeout_raw[0] != '\0') {
      pass_barrier_timeout_seconds = std::atoi(timeout_raw);
    }
    if (pass_barrier_timeout_seconds < 1) {
      throw std::runtime_error("invalid BCI_PASS_BARRIER_TIMEOUT_SECONDS");
    }
    if (!std::filesystem::is_directory(pass_artifact_dir)) {
      throw std::runtime_error("BCI_PASS_ARTIFACT_DIR must already exist");
    }
    printf("[pass barrier] enabled dir=%s token=%s timeout=%ds\n",
           pass_artifact_dir.c_str(), run_token.c_str(),
           pass_barrier_timeout_seconds);
  }

  // OUTER LOOP for warm-cache measurement (matches PIVF Python bench pattern).
  // Each run uses same query batch; cache evolves across runs.
  std::vector<double> run_walls;
  std::vector<std::vector<int32_t>> first_pass_results;
  bool all_pass_predictions_identical = true;
  std::vector<int32_t> deferred_measured_ids;
  if (defer_pass_reporting) {
    deferred_measured_ids.resize(
        static_cast<size_t>(n_runs) * static_cast<size_t>(n_active) *
        static_cast<size_t>(K));
  }
  struct DeferredPassCsvRow {
    int pass;
    double wall_seconds;
    double qps;
    double p50_ms;
    double p95_ms;
    double p99_ms;
    int predictions_identical;
  };
  std::vector<DeferredPassCsvRow> deferred_pass_csv_rows;
  deferred_pass_csv_rows.reserve(static_cast<size_t>(std::max(0, n_runs)));
  for (int run_iter = 0; run_iter < n_runs; ++run_iter) {
  if (reuse_beam_workspace) {
    for (auto& workspace : reusable_beam_workspaces) {
      workspace.staged_distance_candidates = 0;
      workspace.early_abandoned_candidates = 0;
    }
  }
  auto t_q0 = std::chrono::steady_clock::now();
  struct GraphBucketTask {
    size_t bucket_index;
    size_t query_lo;
    size_t query_hi;
    uint64_t prospective_weight;
  };
  std::vector<GraphBucketTask> graph_bucket_tasks;
  std::vector<std::vector<size_t>> graph_bucket_lanes(graph_lane_count);
  std::vector<uint64_t> graph_lane_weight_loads(graph_lane_count, 0);
  std::vector<size_t> graph_lane_query_loads(graph_lane_count, 0);
  if (bucket_schedule == "lpt") {
    // Longest-processing-time scheduling keeps each shard cache-hot while
    // distributing prospective graph work across the fixed worker budget.
    // Adaptive beams make query count a poor proxy: a lane with many high-beam
    // requests can have the same query count but substantially more graph
    // work.  The effective_beam proxy therefore sums beam * starts per bucket;
    // it is fixed before execution and uses neither latency nor ground truth.
    // Weight construction, sorting, and lane assignment are all deliberately
    // inside the timed batch wall.
    auto prospective_query_weight = [&](size_t bi, int qi) -> double {
      if (bucket_weight == "query_count") return 1.0;
      if (bucket_weight == "effective_beam") {
        return static_cast<double>(std::max(1, q_effective_beam[qi])) *
               static_cast<double>(std::max(1, q_effective_starts[qi]));
      }
      const int32_t tag = buckets[bi].first;
      const auto calibrated = calibrated_shard_costs.find(tag);
      const double base_cost =
          calibrated != calibrated_shard_costs.end()
              ? calibrated->second
              : calibrated_shard_cost_fallback;
      return base_cost *
             static_cast<double>(std::max(1, q_effective_beam[qi])) /
             static_cast<double>(calibrated_shard_cost_base_beam) *
             static_cast<double>(std::max(1, q_effective_starts[qi]));
    };
    auto prospective_range_weight =
        [&](size_t bi, size_t lo, size_t hi) -> uint64_t {
      double weight = 0.0;
      for (size_t pos = lo; pos < hi; ++pos) {
        weight += prospective_query_weight(bi, buckets[bi].second[pos]);
      }
      if (!(weight > 0.0) ||
          weight > static_cast<double>(
                       std::numeric_limits<uint64_t>::max())) {
        throw std::runtime_error("invalid prospective shard-task weight");
      }
      return static_cast<uint64_t>(std::llround(weight));
    };
    std::vector<uint64_t> graph_bucket_weights(buckets.size(), 0);
    uint64_t total_graph_weight = 0;
    for (size_t bi = 0; bi < buckets.size(); ++bi) {
      graph_bucket_weights[bi] = prospective_range_weight(
          bi, 0, buckets[bi].second.size());
      total_graph_weight += graph_bucket_weights[bi];
    }
    const uint64_t mean_lane_weight =
        (total_graph_weight + graph_lane_count - 1) / graph_lane_count;
    size_t split_bucket_count = 0;
    for (size_t bi = 0; bi < buckets.size(); ++bi) {
      const size_t query_count = buckets[bi].second.size();
      size_t chunks = 1;
      if (split_overweight_buckets &&
          graph_bucket_weights[bi] > mean_lane_weight) {
        chunks = static_cast<size_t>(
            (graph_bucket_weights[bi] + mean_lane_weight - 1) /
            mean_lane_weight);
        chunks = std::min(chunks, query_count);
      }
      if (chunks > 1) ++split_bucket_count;
      uint64_t assigned_bucket_weight = 0;
      for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t lo = query_count * chunk / chunks;
        const size_t hi = query_count * (chunk + 1) / chunks;
        uint64_t task_weight = prospective_range_weight(bi, lo, hi);
        if (chunk + 1 == chunks) {
          if (assigned_bucket_weight >= graph_bucket_weights[bi]) {
            throw std::runtime_error(
                "prospective task weights exhausted before final chunk");
          }
          task_weight = graph_bucket_weights[bi] - assigned_bucket_weight;
        } else {
          assigned_bucket_weight += task_weight;
        }
        graph_bucket_tasks.push_back({
            bi, lo, hi, task_weight});
      }
    }
    uint64_t task_weight_sum = 0;
    std::vector<size_t> next_query_position(buckets.size(), 0);
    for (const auto& task : graph_bucket_tasks) {
      if (task.query_lo != next_query_position[task.bucket_index] ||
          task.query_hi <= task.query_lo ||
          task.query_hi > buckets[task.bucket_index].second.size()) {
        throw std::runtime_error(
            "bucket microbatch tasks are not an exact contiguous partition");
      }
      next_query_position[task.bucket_index] = task.query_hi;
      task_weight_sum += task.prospective_weight;
    }
    for (size_t bi = 0; bi < buckets.size(); ++bi) {
      if (next_query_position[bi] != buckets[bi].second.size()) {
        throw std::runtime_error(
            "bucket microbatch tasks do not cover every query exactly once");
      }
    }
    if (task_weight_sum != total_graph_weight) {
      throw std::runtime_error(
          "bucket microbatch prospective weights are not conserved");
    }
    std::vector<size_t> graph_task_order(graph_bucket_tasks.size(), 0);
    std::iota(graph_task_order.begin(), graph_task_order.end(), 0);
    std::sort(
        graph_task_order.begin(), graph_task_order.end(),
        [&](size_t lhs, size_t rhs) {
          if (graph_bucket_tasks[lhs].prospective_weight !=
              graph_bucket_tasks[rhs].prospective_weight) {
            return graph_bucket_tasks[lhs].prospective_weight >
                   graph_bucket_tasks[rhs].prospective_weight;
          }
          const auto& lhs_task = graph_bucket_tasks[lhs];
          const auto& rhs_task = graph_bucket_tasks[rhs];
          if (buckets[lhs_task.bucket_index].first !=
              buckets[rhs_task.bucket_index].first) {
            return buckets[lhs_task.bucket_index].first <
                   buckets[rhs_task.bucket_index].first;
          }
          return lhs_task.query_lo < rhs_task.query_lo;
        });
    for (size_t task_index : graph_task_order) {
      const auto& task = graph_bucket_tasks[task_index];
      const size_t lane = static_cast<size_t>(std::distance(
          graph_lane_weight_loads.begin(),
          std::min_element(
              graph_lane_weight_loads.begin(),
              graph_lane_weight_loads.end())));
      graph_bucket_lanes[lane].push_back(task_index);
      graph_lane_weight_loads[lane] += task.prospective_weight;
      graph_lane_query_loads[lane] += task.query_hi - task.query_lo;
    }
    if (std::accumulate(
            graph_lane_weight_loads.begin(),
            graph_lane_weight_loads.end(), uint64_t{0}) !=
        total_graph_weight) {
      throw std::runtime_error(
          "LPT lane assignment does not conserve prospective graph work");
    }
    if (run_iter == 0 && !defer_pass_reporting) {
      printf("[shard bucket tasks] buckets=%zu tasks=%zu split_buckets=%zu "
             "mean_lane_weight=%llu\n",
             buckets.size(), graph_bucket_tasks.size(), split_bucket_count,
             static_cast<unsigned long long>(mean_lane_weight));
      printf("[shard bucket LPT weight loads]");
      for (size_t lane = 0; lane < graph_lane_weight_loads.size(); ++lane) {
        printf("%s%llu", lane ? "," : " ",
               static_cast<unsigned long long>(
                   graph_lane_weight_loads[lane]));
      }
      printf("\n");
      printf("[shard bucket LPT query loads]");
      for (size_t lane = 0; lane < graph_lane_query_loads.size(); ++lane) {
        printf("%s%zu", lane ? "," : " ",
               graph_lane_query_loads[lane]);
      }
      printf("\n");
    }
  }

  // PHASE A: one cache-friendly serial loop per shard bucket.  The optional
  // LPT schedule assigns whole buckets to balanced worker lanes.
  auto process_shard_bucket_range =
      [&](size_t bi, size_t query_lo, size_t query_hi) {
    int32_t T = buckets[bi].first;
    auto& q_list = buckets[bi].second;
    auto& sh = *shards.at(T);
    const uint8_t* graph_packed_values = nullptr;
    if (graph_common_pack_tags.find(T) !=
        graph_common_pack_tags.end()) {
      graph_packed_values = packed_cold.at(T).data();
    }
    ThinSubPR sub_pr(
        base, sh.subset,
        graph_packed_values);  // constructed ONCE per shard
    ExactDistanceMemo<Indx, float>* tag_retry_distance_memo = nullptr;
    if (single_retry_distance_cache) {
      const auto memo = single_retry_distance_memos.find(T);
      if (memo != single_retry_distance_memos.end()) {
        tag_retry_distance_memo = memo->second.get();
      }
    }
    // Iter (per BCI ceiling at 0.93 finding): boost limit to allow beam search
    // to actually explore the graph. ParlayIVF uses limit = 100K-3M. We were
    // starving at 8*beam. Try 100x beam as a balance between budget and reach.
    long bounded_limit = std::min<long>((long)sh.graph.size(), (long)std::max<long>(100L * beam, 100000L));
    const int64_t tag_support =
        bmt.row_offsets[T + 1] - bmt.row_offsets[T];
    const bool tag_high_effort =
        select_single_high_effort(
            T, tag_support, single_high_beam_min_support, high_beam_tags);
    const bool tag_risk_effort =
        single_risk_beam_tags.count(T) != 0;
    const int tag_single_beam =
        tag_risk_effort ? single_risk_beam
                        : (tag_high_effort ? single_high_beam : beam);
    const double tag_single_cut =
        tag_risk_effort ? single_risk_cut
                        : (tag_high_effort ? single_high_cut : hamcg_cut);
    const long effective_degree_limit =
        query_graph_degree_limit > 0
            ? std::min<long>(
                  sh.maxDeg, query_graph_degree_limit)
            : sh.maxDeg;
    long bounded_limit_single =
        std::min<long>((long)sh.graph.size(),
                       (long)std::max<long>(100L * tag_single_beam, 100000L));
    // BCI_SINGLE_POOL expands HAMCG_single candidate pool returned by beam_search
    // for downstream exact-distance rerank (default K=10; recommended 50-80).
    QueryParams QP(
        (long)K, (long)beam, /*cut=*/hamcg_cut, bounded_limit,
        effective_degree_limit);
    QueryParams QP_single(
        (long)single_pool,
        (long)std::max((long)tag_single_beam, (long)single_pool),
        /*cut=*/tag_single_cut, bounded_limit_single,
        effective_degree_limit);
    long bounded_limit_retry =
        std::min<long>((long)sh.graph.size(),
                       (long)std::max<long>(
                           100L * single_retry_beam, 100000L));
    QueryParams QP_single_retry(
        (long)single_pool,
        (long)std::max((long)single_retry_beam, (long)single_pool),
        /*cut=*/single_retry_cut, bounded_limit_retry,
        effective_degree_limit);
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
    const long cached_probe_limit =
        std::min<long>(
            static_cast<long>(sh.graph.size()),
            std::max<long>(
                100L * cached_replay_probe_beam, 100000L));
    const long cached_repair_limit =
        std::min<long>(
            static_cast<long>(sh.graph.size()),
            std::max<long>(
                100L * cached_replay_repair_beam, 100000L));
    QueryParams QP_cached_probe(
        static_cast<long>(single_pool),
        std::max<long>(
            cached_replay_probe_beam, single_pool),
        tag_single_cut, cached_probe_limit,
        effective_degree_limit);
    QueryParams QP_cached_repair(
        static_cast<long>(single_pool),
        std::max<long>(
            cached_replay_repair_beam, single_pool),
        tag_single_cut, cached_repair_limit,
        effective_degree_limit);
    ExactDistanceMemo<Indx, float> exact_distance_memo(
        sh.graph.size());
#endif

    for (size_t query_pos = query_lo; query_pos < query_hi; ++query_pos) {
      const int i = q_list[query_pos];
      int qid = qid_lo + i;
      PointT q = query[qid];
      PointT graph_query(q.get(), (unsigned)query.dimension(),
                         (unsigned)query.aligned_dimension(), -1);
      auto t_a = std::chrono::steady_clock::now();

      // IVF² CLUSTER FAST PATH: if enabled and this is
      // a conjunction query AND primary tag has cluster index, scan centroids
      // (fits L1) + collect top-nprobe clusters + post-filter + exact rerank.
      // Skips beam_search entirely. Replicates PIVF and_query for warm-cache QPS.
      std::vector<std::pair<float, int32_t>> cands;
      cands.reserve(K * 2);
      // HAMCG_single (route 0) uses expanded pool for exact-distance rerank.
      int pool_cap = (q_route[i] == 1) ? post_filter_pool
                   : (q_route[i] == 0) ? single_pool : (int)K;
      int got = 0;
      // OPTIMIZATION: cache bitvector pointers per query (no unordered_map lookup
      // in inner candidate loop). Bitvector lookup = 1 shift + 1 AND + 1 load ~1ns.
      const uint64_t* sec_bv = nullptr;
      const uint64_t* pri_bv = nullptr;
      if (use_bitvec && q_route[i] == 1) {
        auto it_sec = bitvecs.find(q_secondary[i]);
        if (it_sec != bitvecs.end()) sec_bv = it_sec->second.data();
        auto it_pri = bitvecs.find(q_primary[i]);
        if (it_pri != bitvecs.end()) pri_bv = it_pri->second.data();
      }
      auto has_sec = [&](int32_t g) {
        if (sec_bv) return (sec_bv[g >> 6] & (1ULL << (g & 63))) != 0ULL;
        return (bool)bm.match(g, q_secondary[i]);
      };
      auto has_pri = [&](int32_t g) {
        if (pri_bv) return (pri_bv[g >> 6] & (1ULL << (g & 63))) != 0ULL;
        return (bool)bm.match(g, q_primary[i]);
      };

      // use_clusters: 1=PIVF-style JOIN, 2=primary-only sorted_near + bitvec filter
      bool use_ivf2 = (use_clusters >= 1) && q_route[i] == 1 && clusters.count(T);
      bool use_ivf2_simple = (use_clusters == 2);  // skip secondary sorted_near
      bool use_single_cluster =
          single_cluster_scan && q_route[i] == 0;
      if (use_single_cluster) {
        auto cluster_it = clusters.find(T);
        if (cluster_it == clusters.end()) {
          throw std::runtime_error(
              "single-cluster scan is missing a primary cluster index");
        }
        const ClusterIndex& ci = *cluster_it->second;
        std::vector<std::pair<float, int32_t>> centroid_distances;
        centroid_distances.reserve(ci.n_clusters);
        for (int c = 0; c < ci.n_clusters; ++c) {
          PointT centroid(
              ci.centroid_data.data() + (size_t)c * ci.aligned_dim,
              ci.aligned_dim, ci.aligned_dim, c);
          centroid_distances.push_back(
              {graph_query.distance(centroid), c});
        }
        const int selected_clusters =
            std::min<int>(nprobe, centroid_distances.size());
        if (selected_clusters < (int)centroid_distances.size()) {
          std::nth_element(
              centroid_distances.begin(),
              centroid_distances.begin() + selected_clusters,
              centroid_distances.end());
        }
        std::sort(
            centroid_distances.begin(),
            centroid_distances.begin() + selected_clusters);
        std::pair<float, int32_t> frontier[K + 1];
        for (int k = 0; k < K; ++k) {
          frontier[k] = {
              std::numeric_limits<float>::max(), -1};
        }
        int scanned = 0;
        for (int p = 0;
             p < selected_clusters && scanned < target_pts; ++p) {
          const int c = centroid_distances[p].second;
          const int64_t lo = ci.member_offsets[c];
          const int64_t hi = ci.member_offsets[c + 1];
          for (int64_t j = lo;
               j < hi && scanned < target_pts; ++j, ++scanned) {
            const int32_t global = ci.member_ids[j];
            const float distance =
                graph_query.distance(base[global]);
            if (distance < frontier[K - 1].first ||
                (distance == frontier[K - 1].first &&
                 global < frontier[K - 1].second)) {
              int position = K - 1;
              while (
                  position > 0 &&
                  (frontier[position - 1].first > distance ||
                   (frontier[position - 1].first == distance &&
                    frontier[position - 1].second > global))) {
                frontier[position] = frontier[position - 1];
                --position;
              }
              frontier[position] = {distance, global};
            }
          }
        }
        q_graph_dist_comps[i] =
            (uint64_t)ci.n_clusters + (uint64_t)scanned;
        q_graph_visited[i] = (uint64_t)scanned;
        q_graph_frontier_size[i] = (uint64_t)K;
        for (int k = 0; k < K; ++k) {
          if (frontier[k].second >= 0) {
            cands.push_back(frontier[k]);
          }
        }
      } else if (use_ivf2) {
        // PIVF-style sorted_near × 2 + JOIN + exact rerank
        // Step 1: sorted_near on primary → candidate IDs (no distance compute)
        int adaptive_tpts = q_tpts[i];  // proxy-routed per-query target_points
        // PACH: secondary tag for predicate-aware cluster pruning. Only active
        // when use_pach=1 AND query has a secondary tag AND we have pach bitvecs.
        int32_t pach_sec = q_secondary[i];
        const std::vector<std::vector<uint64_t>>* pach_pri = nullptr;
        if (use_pach && pach_sec >= 0) {
          auto it_p = pach_bitvecs.find(T);
          if (it_p != pach_bitvecs.end()) pach_pri = &it_p->second;
        }
        auto sorted_near = [&](const ClusterIndex& ci, std::vector<int32_t>& out, int32_t pach_for_tag) {
          // Build pach lookup for THIS specific tag (primary uses pach_pri; secondary uses its own)
          const std::vector<std::vector<uint64_t>>* my_pach = nullptr;
          int32_t my_filter_tag = -1;
          if (use_pach && pach_sec >= 0) {
            if (pach_for_tag == T) {
              my_pach = pach_pri;
              my_filter_tag = pach_sec;  // primary tag's clusters filtered by secondary
            } else {
              auto it_p = pach_bitvecs.find(pach_for_tag);
              if (it_p != pach_bitvecs.end()) {
                my_pach = &it_p->second;
                my_filter_tag = T;  // secondary tag's clusters filtered by primary
              }
            }
          }
          auto keeps = [&](int c) {
            if (!my_pach) return true;
            const auto& bv = (*my_pach)[c];
            return (bv[my_filter_tag >> 6] & (1ULL << (my_filter_tag & 63))) != 0ULL;
          };
          // Compute centroid distances ONLY for clusters that survive PACH.
          std::vector<std::pair<float, int32_t>> cent_dists;
          cent_dists.reserve(ci.n_clusters);
          int kept = 0;
          for (int c = 0; c < ci.n_clusters; ++c) {
            if (!keeps(c)) continue;
            ++kept;
            PointT cpt(ci.centroid_data.data() + (size_t)c * ci.aligned_dim,
                       ci.aligned_dim, ci.aligned_dim, c);
            cent_dists.push_back({q.distance(cpt), c});
          }
          // PACH instrumentation (atomic to avoid races; cheap)
          pach_total_clusters.fetch_add(ci.n_clusters, std::memory_order_relaxed);
          pach_kept_clusters.fetch_add(kept, std::memory_order_relaxed);
          int np = std::min<int>(nprobe, (int)cent_dists.size());
          if (np == 0) { out.clear(); return; }
          if (np < (int)cent_dists.size()) {
            std::nth_element(cent_dists.begin(), cent_dists.begin() + np, cent_dists.end(),
              [](auto&a, auto&b){ return a.first < b.first; });
          }
          out.clear(); out.reserve(adaptive_tpts);
          for (int p = 0; p < np && (int)out.size() < adaptive_tpts; ++p) {
            int c = cent_dists[p].second;
            int64_t lo = ci.member_offsets[c];
            int64_t hi = ci.member_offsets[c + 1];
            for (int64_t j = lo; j < hi && (int)out.size() < adaptive_tpts; ++j) {
              out.push_back(ci.member_ids[j]);
            }
          }
          std::sort(out.begin(), out.end()); // for sorted-list join
        };
        std::vector<int32_t> ids_a;
        sorted_near(*clusters.at(T), ids_a, T);

        // Step 2: if secondary has cluster AND not use_ivf2_simple, sorted_near + INTERSECT
        // Else: filter ids_a by bitvec(secondary) — simpler, lower overhead
        if (!use_ivf2_simple && clusters.count(q_secondary[i])) {
          std::vector<int32_t> ids_b;
          sorted_near(
              *clusters.at(q_secondary[i]), ids_b, q_secondary[i]);
          // Sorted-list intersection
          std::vector<int32_t> intersect;
          intersect.reserve(std::min(ids_a.size(), ids_b.size()));
          std::set_intersection(ids_a.begin(), ids_a.end(), ids_b.begin(), ids_b.end(),
                                std::back_inserter(intersect));
          // Step 3: exact distance on intersection — PIVF-style streaming top-K
          // with early reject. Skips ~90% of pushes since we only insert when better
          // than current worst. Frontier is fixed K-sized.
          std::pair<float, int32_t> frontier[K + 1];
          for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};
          for (int32_t g : intersect) {
            PointT bp = base[g];
            float d = q.distance(bp);
            if (d < frontier[K-1].first ||
                (d == frontier[K-1].first && g < frontier[K-1].second)) {
              // insertion sort (K=10, ~10 ops per insert)
              int p = K - 1;
              while (p > 0 && (frontier[p-1].first > d ||
                               (frontier[p-1].first == d && frontier[p-1].second > g))) {
                frontier[p] = frontier[p-1];
                --p;
              }
              frontier[p] = {d, g};
            }
          }
          for (int k = 0; k < K; ++k) {
            if (frontier[k].second >= 0) cands.push_back(frontier[k]);
          }
        } else {
          // Single-shard fallback: streaming top-K with bitvec filter
          std::pair<float, int32_t> frontier[K + 1];
          for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};
          for (int32_t g : ids_a) {
            if (!has_sec(g)) continue;
            PointT bp = base[g];
            float d = q.distance(bp);
            if (d < frontier[K-1].first ||
                (d == frontier[K-1].first && g < frontier[K-1].second)) {
              int p = K - 1;
              while (p > 0 && (frontier[p-1].first > d ||
                               (frontier[p-1].first == d && frontier[p-1].second > g))) {
                frontier[p] = frontier[p-1];
                --p;
              }
              frontier[p] = {d, g};
            }
          }
          for (int k = 0; k < K; ++k) {
            if (frontier[k].second >= 0) cands.push_back(frontier[k]);
          }
        }
      } else {
        // PRIMARY beam_search + post-filter (existing path) — use pre-built bitset.
        // HAMCG_single (route 0): use QP_single with expanded k (single_pool) so
        // beam_search returns a larger pool whose top-K-by-exact-distance can be
        // selected; reduces order-noise vs the K=10 default truncation.
        QueryParams& QP_use = (q_route[i] == 0) ? QP_single : QP;
        auto nearest_representatives =
            [&](int count, bool require_independent)
                -> parlay::sequence<Indx> {
          auto cluster_it = clusters.find(T);
          if (cluster_it == clusters.end() ||
              cluster_it->second->representative_local_ids.empty()) {
            throw std::runtime_error(
                "alternate-entry query is missing prepared cluster entries");
          }
          ClusterIndex& ci = *cluster_it->second;
          std::vector<std::pair<float, int32_t>> best;
          best.reserve((size_t)count);
          for (int c = 0; c < ci.n_clusters; ++c) {
            const int32_t representative =
                ci.representative_local_ids[c];
            // Entry zero is the ordinary search basin, so it cannot provide
            // an independent retry.
            if (representative < 0 ||
                (require_independent && representative == 0)) {
              continue;
            }
            PointT centroid(
                ci.centroid_data.data() + (size_t)c * ci.aligned_dim,
                ci.aligned_dim, ci.aligned_dim, c);
            const float distance = graph_query.distance(centroid);
            const std::pair<float, int32_t> candidate{
                distance, representative};
            auto position =
                std::lower_bound(best.begin(), best.end(), candidate);
            if (position != best.end() ||
                best.size() < (size_t)count) {
              best.insert(position, candidate);
              if (best.size() > (size_t)count) {
                best.pop_back();
              }
            }
          }
          if (best.empty()) {
            throw std::runtime_error(
                "alternate-entry query has no independent representative");
          }
          parlay::sequence<Indx> representatives(best.size());
          for (size_t j = 0; j < best.size(); ++j) {
            representatives[j] = best[j].second;
          }
          return representatives;
        };
        parlay::sequence<Indx> start_points =
            single_entry_mode == "nearest_centroid"
                ? nearest_representatives(
                      single_entry_centroids, false)
                : parlay::sequence<Indx>{0};
        if (q_route[i] == 0 && q_effective_starts[i] > 1) {
          auto independent = nearest_representatives(1, true);
          start_points.push_back(independent[0]);
        }
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
        if (q_route[i] == 0 && T == 35) {
          const auto& landmarks =
              residual_landmark_crossfit_entries(
                  residual_landmark_directory, qid);
          std::pair<float, int32_t> best{
              std::numeric_limits<float>::infinity(), -1};
          for (int32_t landmark : landmarks) {
            if (landmark < 0 ||
                static_cast<size_t>(landmark) >= sub_pr.size()) {
              throw std::runtime_error(
                  "residual landmark is outside the tag35 shard");
            }
            const std::pair<float, int32_t> candidate{
                graph_query.distance(sub_pr[landmark]), landmark};
            if (candidate < best) best = candidate;
          }
          if (best.second <= 0) {
            throw std::runtime_error(
                "residual-landmark directory produced no independent entry");
          }
          start_points.push_back(best.second);
          q_residual_landmark_scores[i] = landmarks.size();
          q_residual_landmark_selected[i] = best.second;
        }
#endif
#ifdef BCI_ENABLE_FIXED_FRONTIER_BATCHED_DIAGNOSTIC
        if (fixed_frontier_batched_l2) {
          const size_t worker = parlay::worker_id();
          if (worker >= fixed_frontier_batched_workspaces.size()) {
            throw std::runtime_error(
                "fixed-frontier batched workspace worker id is out of range");
          }
          auto result = beam_search_fixed_frontier_batched_l2<
              PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use,
              batched_l2_prefix_dimensions,
              fixed_frontier_batched_workspaces[worker]);
          const auto& frontier = result.frontier;
          q_graph_dist_comps[i] = result.distance_computations;
          q_graph_visited[i] = result.visited.size();
          q_graph_frontier_size[i] = frontier.size();
          q_graph_staged_candidates[i] =
              result.staged_candidates;
          q_graph_prefix_rejected[i] =
              result.prefix_rejected_candidates;
          if (frontier.size() > (size_t)K &&
              frontier[K - 1].second > 0.0f) {
            q_graph_boundary_ratio[i] =
                (double)frontier[K].second /
                (double)frontier[K - 1].second;
          }
          q_graph_exact_retried[i] =
              should_retry_single_search(
                  frontier.size(), (size_t)q_effective_beam[i],
                  q_graph_boundary_ratio[i],
                  single_exact_retry_boundary_ratio) &&
              tag_support < single_exact_retry_max_support &&
              (int64_t)q_graph_dist_comps[i] <=
                  single_exact_retry_max_dist_comps;
          for (size_t j = 0;
               j < frontier.size() && got < pool_cap; ++j) {
            const int32_t local = frontier[j].first;
            const float dist = frontier[j].second;
            const int32_t global = sh.subset[local];
            cands.push_back({dist, global});
            ++got;
          }
        } else
#endif
        if (indexed_heap_beam) {
          const size_t worker = parlay::worker_id();
          if (worker >= indexed_heap_beam_workspaces.size()) {
            throw std::runtime_error(
                "indexed-heap beam workspace worker id is out of range");
          }
          auto view = beam_search_indexed_heap<
              PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use,
              indexed_heap_beam_workspaces[worker]);
          const auto& frontier = view.frontier;
          q_graph_dist_comps[i] = view.distance_computations;
          q_graph_visited[i] = view.visited.size();
          q_graph_frontier_size[i] = frontier.size();
          if (frontier.size() > (size_t)K &&
              frontier[K - 1].second > 0.0f) {
            q_graph_boundary_ratio[i] =
                (double)frontier[K].second /
                (double)frontier[K - 1].second;
          }
          q_graph_exact_retried[i] =
              should_retry_single_search(
                  frontier.size(), (size_t)q_effective_beam[i],
                  q_graph_boundary_ratio[i],
                  single_exact_retry_boundary_ratio) &&
              tag_support < single_exact_retry_max_support &&
              (int64_t)q_graph_dist_comps[i] <=
                  single_exact_retry_max_dist_comps;
          for (size_t j = 0;
               j < frontier.size() && got < pool_cap; ++j) {
            const int32_t local = frontier[j].first;
            const float dist = frontier[j].second;
            const int32_t global = sh.subset[local];
            cands.push_back({dist, global});
            ++got;
          }
        } else if (batched_l2_beam) {
          ExactDistanceMemo<Indx, float>* query_retry_distance_memo =
              q_route[i] == 0 ? tag_retry_distance_memo : nullptr;
          if (query_retry_distance_memo != nullptr) {
            query_retry_distance_memo->begin_query();
          }
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
          exact_distance_memo.begin_query();
          auto result = beam_search_batched_l2<PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use,
              batched_l2_prefix_dimensions, &exact_distance_memo);
#else
#ifdef BCI_ENABLE_SHORTCUT_RESIDUAL_DIAGNOSTIC
          auto result = beam_search_batched_l2<
              PointT, ThinSubPR, Indx>(
                  graph_query, sh.graph, sub_pr,
                  start_points, QP_use,
                  batched_l2_prefix_dimensions);
          if (shortcut_residual &&
              q_route[i] == 0 &&
              T == shortcut_residual_tag) {
            parlay::sequence<Indx> residual_starts;
            const size_t seed_count = std::min<size_t>(
                static_cast<size_t>(single_pool),
                result.frontier.size());
            residual_starts.reserve(seed_count);
            for (size_t seed = 0; seed < seed_count; ++seed) {
              residual_starts.push_back(
                  result.frontier[seed].first);
            }
            if (residual_starts.empty()) {
              throw std::runtime_error(
                  "shortcut residual primary produced no handoff seeds");
            }
            const long residual_beam = 32;
            const long residual_limit = std::min<long>(
                static_cast<long>(sh.graph.size()),
                std::max<long>(
                    100L * residual_beam, 100000L));
            QueryParams residual_params(
                static_cast<long>(single_pool),
                residual_beam, tag_single_cut,
                residual_limit, effective_degree_limit);
            auto residual = beam_search_batched_l2<
                PointT, ThinSubPR, Indx>(
                    graph_query, sh.graph, sub_pr,
                    residual_starts, residual_params,
                    batched_l2_prefix_dimensions,
                    nullptr, 0, 1,
                    &shortcut_mask->row_masks);

            using FrontierPair =
                std::pair<Indx, typename PointT::distanceType>;
            const auto by_local_id =
                [](const FrontierPair& left,
                   const FrontierPair& right) {
                  return left.first < right.first ||
                         (left.first == right.first &&
                          left.second < right.second);
                };
            const auto by_distance_local =
                [](const FrontierPair& left,
                   const FrontierPair& right) {
                  return left.second < right.second ||
                         (left.second == right.second &&
                          left.first < right.first);
                };
            result.frontier.insert(
                result.frontier.end(),
                residual.frontier.begin(),
                residual.frontier.end());
            std::sort(
                result.frontier.begin(),
                result.frontier.end(), by_local_id);
            result.frontier.erase(
                std::unique(
                    result.frontier.begin(),
                    result.frontier.end(),
                    [](const FrontierPair& left,
                       const FrontierPair& right) {
                      return left.first == right.first;
                    }),
                result.frontier.end());
            std::sort(
                result.frontier.begin(),
                result.frontier.end(),
                by_distance_local);
            result.visited.insert(
                result.visited.end(),
                residual.visited.begin(),
                residual.visited.end());
            result.distance_computations +=
                residual.distance_computations;
            result.staged_candidates +=
                residual.staged_candidates;
            result.prefix_rejected_candidates +=
                residual.prefix_rejected_candidates;
          }
#else
#ifdef BCI_ENABLE_TWO_VIEW_R32_DIAGNOSTIC
          auto result = [&]() {
            if (two_view_r32 &&
                two_view_tags.count(T) != 0) {
              auto view_a = beam_search_batched_l2<
                  PointT, ThinSubPR, Indx>(
                      graph_query, sh.graph, sub_pr,
                      start_points, QP_use,
                      batched_l2_prefix_dimensions,
                      nullptr, 0,
                      two_view_primary_full ? 1 : 2);
              auto view_b = beam_search_batched_l2<
                  PointT, ThinSubPR, Indx>(
                      graph_query, sh.graph, sub_pr,
                      start_points, QP_use,
                      batched_l2_prefix_dimensions,
                      nullptr,
                      two_view_primary_full
                          ? two_view_escape_offset
                          : 1,
                      2);
              return merge_independent_two_view_results(
                  std::move(view_a), std::move(view_b));
            }
            return beam_search_batched_l2<
                PointT, ThinSubPR, Indx>(
                    graph_query, sh.graph, sub_pr,
                    start_points, QP_use,
                    batched_l2_prefix_dimensions);
          }();
#else
#ifdef BCI_ENABLE_R96_DEGREE4_SIDECAR_DIAGNOSTIC
          auto result = [&]() {
            auto primary = beam_search_batched_l2<
                PointT, ThinSubPR, Indx>(
                    graph_query, sh.graph, sub_pr,
                    start_points, QP_use,
                    batched_l2_prefix_dimensions);
            if (!r96_degree4_sidecar || q_route[i] != 0) {
              return primary;
            }
            if (sh.r96_degree4_sidecar == nullptr ||
                primary.frontier.empty()) {
              throw std::runtime_error(
                  "R96 degree4 sidecar treatment has no graph/frontier");
            }
            constexpr size_t kHandoffSeeds = 10;
            constexpr long kRescueBeam = 64;
            constexpr long kRescueVisitCap = 64;
            constexpr long kRescueDegreeCap = 4;
            constexpr size_t kRescueScoreCap =
                kHandoffSeeds +
                static_cast<size_t>(kRescueVisitCap) *
                    static_cast<size_t>(kRescueDegreeCap);
            parlay::sequence<Indx> rescue_starts;
            const size_t seed_count = std::min<size_t>(
                kHandoffSeeds, primary.frontier.size());
            rescue_starts.reserve(seed_count);
            for (size_t seed = 0; seed < seed_count; ++seed) {
              rescue_starts.push_back(
                  primary.frontier[seed].first);
            }
            QueryParams rescue_params(
                /*k=*/10L, kRescueBeam, tag_single_cut,
                std::min<long>(
                    static_cast<long>(
                        sh.r96_degree4_sidecar->size()),
                    kRescueVisitCap),
                kRescueDegreeCap);
            auto rescue = beam_search_batched_l2<
                PointT, ThinSubPR, Indx>(
                    graph_query, *sh.r96_degree4_sidecar,
                    sub_pr, rescue_starts, rescue_params,
                    batched_l2_prefix_dimensions);
            if (rescue.visited.size() >
                    static_cast<size_t>(kRescueVisitCap) ||
                rescue.distance_computations >
                    kRescueScoreCap) {
              throw std::runtime_error(
                  "R96 degree4 sidecar exceeded frozen "
                  "visit/distance-computation cap");
            }
            return merge_r96_degree4_sidecar_results(
                std::move(primary), std::move(rescue),
                sh.subset);
          }();
#else
#ifdef BCI_ENABLE_EXACT_RADIX_DIAGNOSTIC
          auto result = [&]() {
            if (exact_engine_mode == "single_radix" ||
                exact_engine_mode == "fourway_radix") {
              return beam_search_exact_radix_l2<
                  PointT, ThinSubPR, Indx>(
                  graph_query, sh.graph, sub_pr,
                  start_points, QP_use,
                  exact_engine_mode == "fourway_radix");
            }
            return beam_search_batched_l2<
                PointT, ThinSubPR, Indx>(
                graph_query, sh.graph, sub_pr,
                start_points, QP_use,
                batched_l2_prefix_dimensions);
          }();
#else
          auto result = beam_search_batched_l2<PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use,
              batched_l2_prefix_dimensions,
              query_retry_distance_memo
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
              , 0, 1,
              eviction_spill && q_route[i] == 0 && T == 35,
              true
#endif
              );
#endif
#endif
#endif
#endif
#endif
          const auto& frontier = result.frontier;
          q_graph_dist_comps[i] = result.distance_computations;
          q_graph_visited[i] = result.visited.size();
          q_graph_frontier_size[i] = frontier.size();
          q_graph_staged_candidates[i] =
              result.staged_candidates;
          q_graph_prefix_rejected[i] =
              result.prefix_rejected_candidates;
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          q_eviction_spill_triggered[i] =
              result.eviction_spill_triggered;
          q_eviction_spill_exact_records[i] =
              result.eviction_spill_exact_records;
          q_eviction_spill_width_records[i] =
              result.eviction_spill_width_records;
          q_eviction_spill_cutoff_records[i] =
              result.eviction_spill_cutoff_records;
          q_eviction_spill_prefix_records[i] =
              result.eviction_spill_prefix_records;
          q_eviction_spill_metric_excluded[i] =
              result.eviction_spill_metric_excluded;
          q_eviction_spill_handoff_records[i] =
              result.eviction_spill_handoff_records;
          q_eviction_spill_extra_comps[i] =
              result.eviction_spill_extra_distance_computations;
          q_eviction_spill_new_visits[i] =
              result.eviction_spill_new_visits;
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
          q_q4_exact_rerank_candidates[i] =
              result.q4_exact_rerank_candidates;
#endif
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
          q_graph_top10_last_change_visit[i] =
              result.top10_stability.last_change_visit;
          q_graph_top10_last_change_dist_comps[i] =
              result.top10_stability.last_change_distance_computations;
          q_graph_top10_stability_age[i] =
              result.top10_stability.stability_age;
          q_graph_top10_changed_steps_last16[i] =
              result.top10_stability.changed_steps_last16;
          q_graph_top10_entries_last16[i] =
              result.top10_stability.entries_last16;
          q_graph_top10_total_changed_steps[i] =
              result.top10_stability.total_changed_steps;
          q_graph_top10_total_entries[i] =
              result.top10_stability.total_entries;
          q_graph_top10_final_size[i] =
              result.top10_stability.final_size;
          if (frontier.size() > (size_t)K) {
            q_graph_top10_margin_abs[i] =
                static_cast<double>(frontier[K].second) -
                static_cast<double>(frontier[K - 1].second);
          }
#endif
          if (frontier.size() > (size_t)K &&
              frontier[K - 1].second > 0.0f) {
            q_graph_boundary_ratio[i] =
                (double)frontier[K].second /
                (double)frontier[K - 1].second;
          }
          q_graph_exact_retried[i] =
              should_retry_single_search(
                  frontier.size(), (size_t)q_effective_beam[i],
                  q_graph_boundary_ratio[i],
                  single_exact_retry_boundary_ratio) &&
              tag_support < single_exact_retry_max_support &&
              (int64_t)q_graph_dist_comps[i] <=
                  single_exact_retry_max_dist_comps;
          for (size_t j = 0;
               j < frontier.size() && got < pool_cap; ++j) {
            const int32_t local = frontier[j].first;
            const float dist = frontier[j].second;
            const int32_t global = sh.subset[local];
            cands.push_back({dist, global});
            ++got;
          }
          const bool retry_single =
              q_route[i] == 0 &&
              single_retry_tag_authorized(
                  T, has_single_retry_vocab, single_retry_tags) &&
              should_retry_single_search(
                  frontier.size(), (size_t)q_effective_beam[i],
                  q_graph_boundary_ratio[i],
                  single_retry_boundary_ratio) &&
              (single_retry_beam > q_effective_beam[i] ||
               std::abs(single_retry_cut - q_effective_cut[i]) > 1e-6) &&
              (int64_t)q_graph_dist_comps[i] >=
                  single_retry_min_dist_comps &&
              (int64_t)q_graph_dist_comps[i] <=
                  single_retry_max_dist_comps &&
              tag_support >= single_retry_min_support &&
              tag_support <= single_retry_max_support;
          if (retry_single) {
            auto retry_result = beam_search_batched_l2<
                PointT, ThinSubPR, Indx>(
                    graph_query, sh.graph, sub_pr, start_points,
                    QP_single_retry, batched_l2_prefix_dimensions,
                    query_retry_distance_memo);
            q_graph_retried[i] = 1;
            q_graph_retry_dist_comps[i] =
                retry_result.distance_computations;
            q_graph_retry_cache_hits[i] =
                retry_result.exact_distance_cache_hits;
            q_graph_retry_uncached_candidates[i] =
                retry_result.incremental_exact_misses;
            q_graph_retry_exact_completions[i] =
                retry_result.exact_distance_completions;
            q_graph_retry_uncached_prefix_rejections[i] =
                retry_result.uncached_prefix_rejections;
            if (query_retry_distance_memo != nullptr &&
                q_graph_retry_cache_hits[i] +
                        q_graph_retry_uncached_candidates[i] !=
                    q_graph_retry_dist_comps[i] ||
                q_graph_retry_exact_completions[i] +
                        q_graph_retry_uncached_prefix_rejections[i] !=
                    q_graph_retry_uncached_candidates[i]) {
              throw std::runtime_error(
                  "single-retry distance-cache accounting differs");
            }
            int retry_added = 0;
            for (size_t j = 0;
                 j < retry_result.frontier.size() &&
                 retry_added < single_pool; ++j) {
              const int32_t local = retry_result.frontier[j].first;
              const float dist = retry_result.frontier[j].second;
              const int32_t global = sh.subset[local];
              auto existing = std::find_if(
                  cands.begin(), cands.end(),
                  [&](const auto& candidate) {
                    return candidate.second == global;
                  });
              if (existing == cands.end()) {
                cands.push_back({dist, global});
              } else if (dist < existing->first) {
                existing->first = dist;
              }
              ++retry_added;
            }
          }
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
          q_cr_base_cache_hits[i] =
              result.exact_distance_cache_hits;
          q_cr_base_exact_misses[i] =
              result.incremental_exact_misses;
          q_cr_base_exact_completions[i] =
              result.exact_distance_completions;
          q_cr_base_uncached_prefix_rejections[i] =
              result.uncached_prefix_rejections;
          if (frontier.size() >= static_cast<size_t>(K)) {
            q_cr_base_d10[i] =
                static_cast<double>(frontier[K - 1].second);
          }
          if (frontier.size() > static_cast<size_t>(K)) {
            q_cr_base_frontier_margin_abs[i] =
                static_cast<double>(frontier[K].second) -
                static_cast<double>(frontier[K - 1].second);
          }

          auto probe_result =
              beam_search_batched_l2<PointT, ThinSubPR, Indx>(
                  graph_query, sh.graph, sub_pr, start_points,
                  QP_cached_probe, batched_l2_prefix_dimensions,
                  &exact_distance_memo);
          q_cr_probe_logical_comps[i] =
              probe_result.distance_computations;
          q_cr_probe_cache_hits[i] =
              probe_result.exact_distance_cache_hits;
          q_cr_probe_exact_misses[i] =
              probe_result.incremental_exact_misses;
          q_cr_probe_exact_completions[i] =
              probe_result.exact_distance_completions;
          q_cr_probe_uncached_prefix_rejections[i] =
              probe_result.uncached_prefix_rejections;
          q_cr_probe_visited[i] = probe_result.visited.size();
          const auto probe_churn =
              analyze_cached_replay_top10_churn(
                  frontier, probe_result.frontier,
                  static_cast<size_t>(K));
          q_cr_probe_top10_intersection[i] =
              probe_churn.intersection;
          q_cr_probe_top10_entered[i] = probe_churn.entered;
          q_cr_probe_top10_dropped[i] = probe_churn.dropped;
          q_cr_probe_d10[i] = probe_churn.replay_d10;
          q_cr_probe_min_entered_margin_abs[i] =
              probe_churn.min_entered_margin_abs;
          q_cr_probe_max_entered_margin_abs[i] =
              probe_churn.max_entered_margin_abs;
          if (probe_result.frontier.size() >
              static_cast<size_t>(K)) {
            q_cr_probe_frontier_margin_abs[i] =
                static_cast<double>(
                    probe_result.frontier[K].second) -
                static_cast<double>(
                    probe_result.frontier[K - 1].second);
          }

          if (probe_churn.changed()) {
            auto repair_result =
                beam_search_batched_l2<PointT, ThinSubPR, Indx>(
                    graph_query, sh.graph, sub_pr, start_points,
                    QP_cached_repair,
                    batched_l2_prefix_dimensions,
                    &exact_distance_memo);
            q_cr_repair_ran[i] = 1;
            q_cr_repair_logical_comps[i] =
                repair_result.distance_computations;
            q_cr_repair_cache_hits[i] =
                repair_result.exact_distance_cache_hits;
            q_cr_repair_exact_misses[i] =
                repair_result.incremental_exact_misses;
            q_cr_repair_exact_completions[i] =
                repair_result.exact_distance_completions;
            q_cr_repair_uncached_prefix_rejections[i] =
                repair_result.uncached_prefix_rejections;
            q_cr_repair_visited[i] =
                repair_result.visited.size();
            const auto repair_churn =
                analyze_cached_replay_top10_churn(
                    frontier, repair_result.frontier,
                    static_cast<size_t>(K));
            q_cr_repair_top10_intersection[i] =
                repair_churn.intersection;
            q_cr_repair_top10_entered[i] =
                repair_churn.entered;
            q_cr_repair_top10_dropped[i] =
                repair_churn.dropped;
            q_cr_repair_d10[i] = repair_churn.replay_d10;
            if (repair_result.frontier.size() >
                static_cast<size_t>(K)) {
              q_cr_repair_frontier_margin_abs[i] =
                  static_cast<double>(
                      repair_result.frontier[K].second) -
                  static_cast<double>(
                      repair_result.frontier[K - 1].second);
            }

            int repair_added = 0;
            for (size_t j = 0;
                 j < repair_result.frontier.size() &&
                 repair_added < single_pool; ++j) {
              const int32_t local =
                  repair_result.frontier[j].first;
              const float dist =
                  repair_result.frontier[j].second;
              const int32_t global = sh.subset[local];
              auto existing = std::find_if(
                  cands.begin(), cands.end(),
                  [&](const auto& candidate) {
                    return candidate.second == global;
                  });
              if (existing == cands.end()) {
                cands.push_back({dist, global});
              } else if (dist < existing->first) {
                existing->first = dist;
              }
              ++repair_added;
            }
          }
#endif
        } else if (dual_heap_beam) {
          const size_t worker = parlay::worker_id();
          if (worker >= dual_heap_beam_workspaces.size()) {
            throw std::runtime_error(
                "dual-heap beam workspace worker id is out of range");
          }
          auto result = beam_search_dual_heap<PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use,
              dual_heap_beam_workspaces[worker]);
          const auto& frontier = result.frontier;
          q_graph_dist_comps[i] = result.distance_computations;
          q_graph_visited[i] = result.visited.size();
          q_graph_frontier_size[i] = frontier.size();
          if (frontier.size() > (size_t)K &&
              frontier[K - 1].second > 0.0f) {
            q_graph_boundary_ratio[i] =
                (double)frontier[K].second /
                (double)frontier[K - 1].second;
          }
          q_graph_exact_retried[i] =
              should_retry_single_search(
                  frontier.size(), (size_t)q_effective_beam[i],
                  q_graph_boundary_ratio[i],
                  single_exact_retry_boundary_ratio) &&
              tag_support < single_exact_retry_max_support &&
              (int64_t)q_graph_dist_comps[i] <=
                  single_exact_retry_max_dist_comps;
          for (size_t j = 0;
               j < frontier.size() && got < pool_cap; ++j) {
            const int32_t local = frontier[j].first;
            const float dist = frontier[j].second;
            const int32_t global = sh.subset[local];
            cands.push_back({dist, global});
            ++got;
          }
        } else if (reuse_beam_workspace) {
          const size_t worker = parlay::worker_id();
          if (worker >= reusable_beam_workspaces.size()) {
            throw std::runtime_error(
                "reusable beam workspace worker id is out of range");
          }
          auto view = beam_search_reusable<PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use,
              reusable_beam_workspaces[worker]);
          const auto& frontier = view.frontier;
          q_graph_dist_comps[i] = view.distance_computations;
          q_graph_visited[i] = view.visited.size();
          q_graph_frontier_size[i] = frontier.size();
          if (frontier.size() > (size_t)K &&
              frontier[K - 1].second > 0.0f) {
            q_graph_boundary_ratio[i] =
                (double)frontier[K].second /
                (double)frontier[K - 1].second;
          }
          q_graph_exact_retried[i] =
              should_retry_single_search(
                  frontier.size(), (size_t)q_effective_beam[i],
                  q_graph_boundary_ratio[i],
                  single_exact_retry_boundary_ratio) &&
              tag_support < single_exact_retry_max_support &&
              (int64_t)q_graph_dist_comps[i] <=
                  single_exact_retry_max_dist_comps;
          for (size_t j = 0;
               j < frontier.size() && got < pool_cap; ++j) {
            const int32_t local = frontier[j].first;
            const float dist = frontier[j].second;
            const int32_t global = sh.subset[local];
            cands.push_back({dist, global}); ++got;
          }
        } else {
          auto res = beam_search<PointT, ThinSubPR, Indx>(
              graph_query, sh.graph, sub_pr, start_points, QP_use);
          auto& frontier = res.first.first;
          bool retry_single = false;
          bool alternate_retry_single = false;
          if (q_route[i] == 0) {
            q_graph_dist_comps[i] = res.second;
            q_graph_visited[i] = res.first.second.size();
            q_graph_frontier_size[i] = frontier.size();
            if (frontier.size() > (size_t)K &&
                frontier[K - 1].second > 0.0f) {
              q_graph_boundary_ratio[i] =
                  (double)frontier[K].second /
                  (double)frontier[K - 1].second;
            }
            retry_single = should_retry_single_search(
                frontier.size(), (size_t)q_effective_beam[i],
                q_graph_boundary_ratio[i],
                single_retry_boundary_ratio) &&
                single_retry_tag_authorized(
                    T, has_single_retry_vocab, single_retry_tags) &&
                (single_retry_beam > q_effective_beam[i] ||
                 std::abs(single_retry_cut - q_effective_cut[i]) >
                     1e-6) &&
                (int64_t)q_graph_dist_comps[i] >=
                    single_retry_min_dist_comps &&
                (int64_t)q_graph_dist_comps[i] <=
                    single_retry_max_dist_comps &&
                tag_support >= single_retry_min_support &&
                tag_support <= single_retry_max_support;
            q_graph_exact_retried[i] =
                should_retry_single_search(
                    frontier.size(), (size_t)q_effective_beam[i],
                    q_graph_boundary_ratio[i],
                    single_exact_retry_boundary_ratio) &&
                tag_support < single_exact_retry_max_support &&
                (int64_t)q_graph_dist_comps[i] <=
                    single_exact_retry_max_dist_comps;
            alternate_retry_single =
                should_retry_single_search(
                    frontier.size(), (size_t)q_effective_beam[i],
                    q_graph_boundary_ratio[i],
                    single_alt_retry_boundary_ratio) &&
                tag_support < single_alt_retry_max_support;
          }
          for (size_t j = 0;
               j < frontier.size() && got < pool_cap; ++j) {
            int32_t local = frontier[j].first;
            float dist = frontier[j].second;
            const int32_t global = sh.subset[local];
            if (q_route[i] == 1) {
              if (has_sec(global)) {
                cands.push_back({dist, global}); ++got;
              }
            } else {
              cands.push_back({dist, global}); ++got;
            }
          }
          if (retry_single) {
            auto retry_res = beam_search<PointT, ThinSubPR, Indx>(
                graph_query, sh.graph, sub_pr, start_points,
                QP_single_retry);
            q_graph_retried[i] = 1;
            q_graph_retry_dist_comps[i] = retry_res.second;
            auto& retry_frontier = retry_res.first.first;
            int retry_added = 0;
            for (size_t j = 0;
                 j < retry_frontier.size() &&
                 retry_added < single_pool; ++j) {
              const int32_t local = retry_frontier[j].first;
              const float dist = retry_frontier[j].second;
              const int32_t global = sh.subset[local];
              auto existing = std::find_if(
                  cands.begin(), cands.end(),
                  [&](const auto& candidate) {
                    return candidate.second == global;
                  });
              if (existing == cands.end()) {
                cands.push_back({dist, global});
              } else if (dist < existing->first) {
                existing->first = dist;
              }
              ++retry_added;
            }
          }
          if (alternate_retry_single) {
            parlay::sequence<Indx> alternate_start =
                nearest_representatives(1, true);
            auto alternate_res = beam_search<PointT, ThinSubPR, Indx>(
                graph_query, sh.graph, sub_pr, alternate_start,
                QP_single);
            q_graph_alt_retried[i] = 1;
            q_graph_alt_retry_dist_comps[i] =
                alternate_res.second;
            auto& alternate_frontier =
                alternate_res.first.first;
            int alternate_added = 0;
            for (size_t j = 0;
                 j < alternate_frontier.size() &&
                 alternate_added < single_pool; ++j) {
              const int32_t local =
                  alternate_frontier[j].first;
              const float dist = alternate_frontier[j].second;
              const int32_t global = sh.subset[local];
              auto existing = std::find_if(
                  cands.begin(), cands.end(),
                  [&](const auto& candidate) {
                    return candidate.second == global;
                  });
              if (existing == cands.end()) {
                cands.push_back({dist, global});
              } else if (dist < existing->first) {
                existing->first = dist;
              }
              ++alternate_added;
            }
          }
        }
      }

      // ALWAYS DUAL-SHARD (iter-2 after adaptive was too marginal): for ALL
      // conjunction queries where secondary tag has a shard, also beam_search
      // on secondary. Union with primary's filtered candidates. Higher cost,
      // higher recall. (Skipped when use_ivf2 cluster path already used.)
      if (!use_ivf2 && q_route[i] == 1 && shards.count(q_secondary[i])) {
        auto& sh2 = *shards.at(q_secondary[i]);
        const int32_t secondary_tag = q_secondary[i];
        const uint8_t* graph_packed_values2 = nullptr;
        if (graph_common_pack_tags.find(secondary_tag) !=
            graph_common_pack_tags.end()) {
          graph_packed_values2 =
              packed_cold.at(secondary_tag).data();
        }
        ThinSubPR sub_pr2(
            base, sh2.subset, graph_packed_values2);
        long bounded_limit2 = std::min<long>((long)sh2.graph.size(), (long)std::max<long>(100L * beam, 100000L));
        const long secondary_degree_limit =
            query_graph_degree_limit > 0
                ? std::min<long>(
                      sh2.maxDeg, query_graph_degree_limit)
                : sh2.maxDeg;
        QueryParams QP2(
            (long)K, (long)beam, /*cut=*/hamcg_cut,
            bounded_limit2, secondary_degree_limit);
        auto res2 = beam_search<PointT, ThinSubPR, Indx>(
            graph_query, sh2.graph, sub_pr2, /*start=*/0, QP2);
        auto& frontier2 = res2.first.first;
        // dedup
        std::vector<int32_t> seen;
        for (auto& c : cands) seen.push_back(c.second);
        std::sort(seen.begin(), seen.end());
        // build primary bitset (separately from secondary — using the same buffer
        // would clobber). For now, fall back to bm.match for this branch (rare path).
        for (size_t j = 0; j < frontier2.size() && (int)cands.size() < pool_cap; ++j) {
          int32_t local = frontier2[j].first;
          float dist = frontier2[j].second;
          int32_t global = sh2.subset[local];
          if (std::binary_search(seen.begin(), seen.end(), global)) continue;
          if (has_pri(global)) {
            cands.push_back({dist, global});
          }
        }
      }

      // Final top-K by distance with (dist, global_index) tie-breaking to match
      // GT's canonical ordering (asc dist, then asc index — same as numpy
      // stable-sort on sorted-ascending input and the brute_cold heap path).
      int kk = std::min<int>(K, (int)cands.size());
      if (kk > 0) {
        std::partial_sort(cands.begin(), cands.begin()+kk, cands.end(),
          [](auto&a, auto&b){
            return a.first < b.first ||
                   (a.first == b.first && a.second < b.second);
          });
      }
      std::vector<int32_t> top;
      top.reserve(K);
      for (int j = 0; j < kk; ++j) top.push_back(cands[j].second);
      while ((int)top.size() < K) top.push_back(-1);
      results[i] = std::move(top);
      auto t_b = std::chrono::steady_clock::now();
      latencies[i] = std::chrono::duration<double>(t_b-t_a).count() * 1000.0;
    }
  };
  if (bucket_schedule == "lpt") {
    parlay::parallel_for(0, graph_bucket_lanes.size(), [&](size_t lane) {
      for (size_t task_index : graph_bucket_lanes[lane]) {
        const auto& task = graph_bucket_tasks[task_index];
        process_shard_bucket_range(
            task.bucket_index, task.query_lo, task.query_hi);
      }
    }, 1);
  } else {
    parlay::parallel_for(0, buckets.size(), [&](size_t bi) {
      process_shard_bucket_range(bi, 0, buckets[bi].second.size());
    });
  }
  const auto t_phase_graph = std::chrono::steady_clock::now();

  // Sparse complete-support fallback for graph searches whose returned
  // frontier is both width-saturated and nearly tied at the top-k boundary.
  // Split each rare scan into contiguous chunks and use the full fixed worker
  // budget, then merge the local top-k lists exactly. This avoids leaving six
  // workers idle when only one or two requests require the fallback.
  std::vector<int> single_exact_retry_ids;
  for (int i = 0; i < n_q; ++i) {
    if (q_graph_exact_retried[i]) {
      single_exact_retry_ids.push_back(i);
    }
  }
  const size_t exact_workers =
      (size_t)std::max<long>(1, parlay::num_workers());
  const size_t exact_chunks_per_query =
      single_exact_retry_ids.empty() ? 0 : exact_workers;
  const size_t exact_task_count =
      single_exact_retry_ids.size() * exact_chunks_per_query;
  std::vector<std::pair<float, int32_t>> exact_partial_top(
      exact_task_count * (size_t)K,
      {std::numeric_limits<float>::max(), -1});
  std::vector<double> exact_task_ms(exact_task_count, 0.0);
  parlay::parallel_for(0, exact_task_count, [&](size_t task) {
    const auto t_exact0 = std::chrono::steady_clock::now();
    const size_t request_index = task / exact_chunks_per_query;
    const size_t chunk = task % exact_chunks_per_query;
    const int i = single_exact_retry_ids[request_index];
    const int qid = qid_lo + i;
    const uint8_t* q_data = query[qid].get();
    const unsigned q_dim = (unsigned)base.dimension();
    const int32_t tag = q_primary[i];
    const auto& subset = shards.at(tag)->subset;
    const uint8_t* packed = nullptr;
    if (single_exact_retry_pack) {
      auto pack_it = packed_single_exact.find(tag);
      if (pack_it == packed_single_exact.end() ||
          pack_it->second.size() != subset.size() * aligned_dim) {
        throw std::runtime_error(
            "single exact-retry pack is missing or misaligned");
      }
      packed = pack_it->second.data();
    }
    auto* frontier =
        exact_partial_top.data() + task * (size_t)K;
    const size_t lo =
        subset.size() * chunk / exact_chunks_per_query;
    const size_t hi =
        subset.size() * (chunk + 1) / exact_chunks_per_query;
    constexpr size_t EXACT_PREFETCH_AHEAD = 16;
    for (size_t local = lo; local < hi; ++local) {
      if (!packed && local + EXACT_PREFETCH_AHEAD < hi) {
        base[subset[local + EXACT_PREFETCH_AHEAD]].prefetch();
      }
      const int32_t global = subset[local];
      const float distance =
          l2_sq_uint8_avx2(
              q_data,
              packed ? packed + local * aligned_dim :
                       base[global].get(),
              q_dim);
      if (distance < frontier[K - 1].first ||
          (distance == frontier[K - 1].first &&
           global < frontier[K - 1].second)) {
        int p = K - 1;
        while (p > 0 &&
               (frontier[p - 1].first > distance ||
                (frontier[p - 1].first == distance &&
                 frontier[p - 1].second > global))) {
          frontier[p] = frontier[p - 1];
          --p;
        }
        frontier[p] = {distance, global};
      }
    }
    const auto t_exact1 = std::chrono::steady_clock::now();
    exact_task_ms[task] =
        std::chrono::duration<double>(t_exact1 - t_exact0).count() *
        1000.0;
  }, 1);
  parlay::parallel_for(
      0, single_exact_retry_ids.size(), [&](size_t request_index) {
    const auto t_merge0 = std::chrono::steady_clock::now();
    std::pair<float, int32_t> frontier[K + 1];
    for (int k = 0; k < K; ++k) {
      frontier[k] = {std::numeric_limits<float>::max(), -1};
    }
    double max_chunk_ms = 0.0;
    for (size_t chunk = 0; chunk < exact_chunks_per_query; ++chunk) {
      const size_t task =
          request_index * exact_chunks_per_query + chunk;
      max_chunk_ms = std::max(max_chunk_ms, exact_task_ms[task]);
      const auto* partial =
          exact_partial_top.data() + task * (size_t)K;
      for (int k = 0; k < K && partial[k].second >= 0; ++k) {
        const float distance = partial[k].first;
        const int32_t global = partial[k].second;
        if (distance < frontier[K - 1].first ||
            (distance == frontier[K - 1].first &&
             global < frontier[K - 1].second)) {
          int p = K - 1;
          while (p > 0 &&
                 (frontier[p - 1].first > distance ||
                  (frontier[p - 1].first == distance &&
                   frontier[p - 1].second > global))) {
            frontier[p] = frontier[p - 1];
            --p;
          }
          frontier[p] = {distance, global};
        }
      }
    }
    const int i = single_exact_retry_ids[request_index];
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE
    auto& top = results[i];
    top.clear();
#else
    std::vector<int32_t> top;
    top.reserve(K);
#endif
    for (int k = 0; k < K; ++k) {
      if (frontier[k].second >= 0) top.push_back(frontier[k].second);
    }
    while ((int)top.size() < K) top.push_back(-1);
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE
    // `top` aliases results[i] in this build.  Moving a vector into itself is
    // valid-but-unspecified and libstdc++ clears it; the alias is already the
    // destination, so no assignment is required.
    if (results[i].size() != static_cast<size_t>(K)) {
      throw std::runtime_error("exact-retry aliased result width differs");
    }
#else
    results[i] = std::move(top);
#endif
    const auto t_merge1 = std::chrono::steady_clock::now();
    latencies[i] +=
        max_chunk_ms +
        std::chrono::duration<double>(t_merge1 - t_merge0).count() *
            1000.0;
  }, 1);
  const auto t_phase_exact = std::chrono::steady_clock::now();

  // PHASE B0: exact cached pair postings for two-tag brute conjunctions.
  // Keep these separate from long single-tag scans so conjunction latency/QPS
  // is not dominated by concurrent memory-bandwidth pressure from large postings.
  parlay::parallel_for(0, pair_brute_ids.size(), [&](size_t bi) {
    int i = pair_brute_ids[bi];
    int qid = qid_lo + i;
    PointT q = query[qid];
    auto t_a = std::chrono::steady_clock::now();

    std::pair<float, int32_t> frontier[K + 1];

    size_t pair_index =
        pair_postings.find_index(pair_key(q_primary[i], q_secondary[i]));
    if (pair_index != FlatPairPostings::npos) {
      const uint8_t* q_data = q.get();
      unsigned q_dim = (unsigned)base.dimension();
      const int32_t* primary_ids = nullptr;
      const uint8_t* primary_packed = nullptr;
      size_t primary_size = 0;
      if (pair_vector_layout == PairVectorLayout::PrimaryOffset) {
        if (pair_key_first(pair_postings.keys[pair_index]) != q_primary[i]) {
          throw std::runtime_error("pair posting primary-tag mismatch");
        }
        auto pack_it = packed_cold.find(q_primary[i]);
        if (pack_it == packed_cold.end()) {
          throw std::runtime_error("primary-offset serving pack is missing");
        }
        int64_t start = bmt.row_offsets[q_primary[i]];
        int64_t end = bmt.row_offsets[q_primary[i] + 1];
        primary_size = (size_t)(end - start);
        if (pack_it->second.size() != primary_size * aligned_dim) {
          throw std::runtime_error("primary-offset serving pack is misaligned");
        }
        primary_ids = bmt.row_indices.get() + start;
        primary_packed = pack_it->second.data();
      }
      scan_pair_posting_exact(
          pair_postings, pair_index, primary_ids, primary_size,
          primary_packed, q_data, q_dim, aligned_dim, frontier, K);
    } else {
      throw std::runtime_error(
          "pair-brute scheduler referenced an absent flat pair posting");
    }

    std::vector<int32_t> top;
    top.reserve(K);
    for (int k = 0; k < K; ++k) {
      if (frontier[k].second >= 0) top.push_back(frontier[k].second);
    }
    while ((int)top.size() < K) top.push_back(-1);
#ifndef BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE
    results[i] = std::move(top);
#endif

    auto t_b = std::chrono::steady_clock::now();
    latencies[i] = std::chrono::duration<double>(t_b-t_a).count() * 1000.0;
  });
  const auto t_phase_pair = std::chrono::steady_clock::now();

  // PHASE B: parallel_for over remaining brute fallback queries (cold tags, small)
  parlay::parallel_for(0, brute_ids.size(), [&](size_t bi) {
    int i = brute_ids[bi];
    int qid = qid_lo + i;
    PointT q = query[qid];
    auto t_a = std::chrono::steady_clock::now();

    std::vector<int32_t> top;
    top.reserve(K);
    if (q_route[i] == 2) {
      // Brute force on posting list of primary tag — PACKED LAYOUT path:
      // If primary tag's points are pre-packed contiguously (packed_cold), scan
      // sequentially (full prefetcher, no random base[g]). 3-5× speedup vs random.
      // Falls back to random base[g] if not packed.
      int32_t T = q_primary[i];
      int64_t start = bmt.row_offsets[T];
      int64_t end   = bmt.row_offsets[T+1];
      int32_t sec = q_secondary[i];
      size_t n_tag = end - start;

      // Secondary bitvec (for fast filter)
      const uint64_t* sec_bv_brute = nullptr;
      if (sec >= 0) {
        auto it = bitvecs.find(sec);
        if (it != bitvecs.end()) sec_bv_brute = it->second.data();
      }
      int64_t sec_merge_cursor = (sec >= 0) ? bmt.row_offsets[sec] : 0;
      const int64_t sec_merge_end = (sec >= 0) ? bmt.row_offsets[sec + 1] : 0;
      auto has_sec_brute = [&](int32_t g){
        if (sec < 0) return true;
        if (sec_bv_brute) return (sec_bv_brute[g >> 6] & (1ULL << (g & 63))) != 0ULL;
        if (secondary_membership == "posting_merge") {
          while (sec_merge_cursor < sec_merge_end &&
                 bmt.row_indices[sec_merge_cursor] < g) {
            ++sec_merge_cursor;
          }
          return sec_merge_cursor < sec_merge_end &&
                 bmt.row_indices[sec_merge_cursor] == g;
        }
        return (bool)bm.match(g, sec);
      };

      std::pair<float, int32_t> frontier[K + 1];
      for (int k = 0; k < K; ++k) frontier[k] = {std::numeric_limits<float>::max(), -1};

      // Deterministic exact-GT tie rule: on equal distance, prefer the lower
      // global ID.  This keeps exact routes exact under strict-ID scoring.
      auto heap_insert_brute = [&](float d, int32_t g) {
        if (d < frontier[K-1].first ||
            (d == frontier[K-1].first && g < frontier[K-1].second)) {
          int p = K - 1;
          while (p > 0 && (frontier[p-1].first > d ||
                           (frontier[p-1].first == d &&
                            frontier[p-1].second > g))) {
            frontier[p] = frontier[p-1]; --p;
          }
          frontier[p] = {d, g};
        }
      };

      const uint8_t* q_data = q.get();
      const uint8_t* packed_q_data = q_data;
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BOUNDED_EXACT
      alignas(64) std::array<uint8_t, 192> aligned_query{};
      if (conjunction_support_complement && sec >= 0) {
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_VARIANCE_ORDER
        yfcc_support_complement::permute_u8_192_prevalidated(
            q_data, support_complement_dimension_order.data(),
            aligned_query.data());
#else
        std::memcpy(aligned_query.data(), q_data, aligned_query.size());
#endif
        packed_q_data = aligned_query.data();
      }
#endif
      unsigned q_dim = (unsigned)base.dimension();
      size_t pair_index = sec >= 0
                              ? pair_postings.find_index(pair_key(T, sec))
                              : FlatPairPostings::npos;
      bool direct_support_complement_result = false;
      if (pair_index != FlatPairPostings::npos) {
        const int32_t* primary_ids = nullptr;
        const uint8_t* primary_packed = nullptr;
        size_t primary_size = 0;
        if (pair_vector_layout == PairVectorLayout::PrimaryOffset) {
          if (pair_key_first(pair_postings.keys[pair_index]) != T) {
            throw std::runtime_error("pair posting primary-tag mismatch");
          }
          auto pack_it = packed_cold.find(T);
          if (pack_it == packed_cold.end()) {
            throw std::runtime_error("primary-offset serving pack is missing");
          }
          primary_size = n_tag;
          if (pack_it->second.size() != primary_size * aligned_dim) {
            throw std::runtime_error(
                "primary-offset serving pack is misaligned");
          }
          primary_ids = bmt.row_indices.get() + start;
          primary_packed = pack_it->second.data();
        }
        scan_pair_posting_exact(
            pair_postings, pair_index, primary_ids, primary_size,
            primary_packed, q_data, q_dim, aligned_dim, frontier, K);
      } else if (conjunction_support_complement && sec >= 0) {
        const int32_t* primary_ids =
            bmt.row_indices.get() + start;
        const int64_t secondary_start = bmt.row_offsets[sec];
        const int64_t secondary_end = bmt.row_offsets[sec + 1];
        const int32_t* secondary_ids =
            bmt.row_indices.get() + secondary_start;
        const size_t secondary_size =
            static_cast<size_t>(secondary_end - secondary_start);
        auto bitmap_it = bitvecs.find(sec);
        const uint64_t* secondary_bitmap =
            bitmap_it == bitvecs.end() ? nullptr : bitmap_it->second.data();
        const auto support_complement_strategy =
            yfcc_support_complement::
                choose_base_only_support_complement_strategy(
                    n_tag, secondary_size, secondary_bitmap != nullptr);
        const auto strategy =
            support_complement_strategy ==
                    yfcc_support_complement::IntersectionStrategy::
                        SortedMerge
                ? yfcc_support_complement::choose_intersection_strategy(
                      n_tag, secondary_size, false)
                : support_complement_strategy;
        const size_t worker = parlay::worker_id();
        if (worker >= support_complement_scratch.size()) {
          throw std::runtime_error(
              "support-complement worker id is out of range");
        }
        auto& scratch = support_complement_scratch[worker];
        std::array<int32_t, 10> exact_ids;
        if (n_tag <=
            yfcc_support_complement::kMaximumPrimarySupport) {
          yfcc_support_complement::
              intersect_postings_to_local_offset16_scratch_prevalidated(
                  primary_ids, n_tag, secondary_ids, secondary_size,
                  secondary_bitmap, strategy, scratch.local_offsets);
          auto packed_it = packed_cold.find(T);
          if (packed_it != packed_cold.end()) {
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BOUNDED_EXACT
#if defined(BCI_ENABLE_SUPPORT_COMPLEMENT_BLOCK16_CASCADE)
            if (support_exact_mode == "block16") {
              exact_ids =
                  yfcc_pair_offset16_block16_cascade_v2::
                      scan_prevalidated<10, false>(
                          scratch.local_offsets.data(),
                          scratch.local_offsets.size(), primary_ids,
                          packed_it->second.data(), aligned_dim,
                          packed_q_data);
            } else {
              exact_ids =
                  yfcc_support_complement::
                      scan_exact_packed_primary_bounded_prevalidated<10>(
                          scratch.local_offsets.data(),
                          scratch.local_offsets.size(), primary_ids,
                          packed_it->second.data(), aligned_dim,
                          packed_q_data);
            }
#elif defined(BCI_ENABLE_SUPPORT_COMPLEMENT_PREFIX128_CASCADE)
            if (support_exact_mode == "prefix128") {
              exact_ids =
                  yfcc_pair_offset16_prefix128_cascade_v1::
                      scan_prevalidated<10>(
                          scratch.local_offsets.data(),
                          scratch.local_offsets.size(), primary_ids,
                          packed_it->second.data(), aligned_dim,
                          packed_q_data, scratch.prefix128_survivors.data(),
                          scratch.prefix128_survivors.size(), nullptr);
            } else {
              exact_ids =
                  yfcc_support_complement::
                      scan_exact_packed_primary_bounded_prevalidated<10>(
                          scratch.local_offsets.data(),
                          scratch.local_offsets.size(), primary_ids,
                          packed_it->second.data(), aligned_dim,
                          packed_q_data);
            }
#else
#ifdef BCI_ENABLE_SUPPORT_COMPLEMENT_BOUNDED32
            exact_ids =
                yfcc_support_complement::
                    scan_exact_packed_primary_bounded32_prevalidated<10>(
                        scratch.local_offsets.data(),
                        scratch.local_offsets.size(), primary_ids,
                        packed_it->second.data(), aligned_dim,
                        packed_q_data);
#else
            exact_ids =
                yfcc_support_complement::
                    scan_exact_packed_primary_bounded_prevalidated<10>(
                        scratch.local_offsets.data(),
                        scratch.local_offsets.size(), primary_ids,
                        packed_it->second.data(), aligned_dim,
                        packed_q_data);
#endif
#endif
#else
            exact_ids =
                yfcc_support_complement::
                    scan_exact_packed_primary_prevalidated<10>(
                        scratch.local_offsets.data(),
                        scratch.local_offsets.size(), primary_ids,
                        packed_it->second.data(), aligned_dim, q_data);
#endif
          } else {
            exact_ids =
                yfcc_support_complement::scan_exact_prevalidated<10>(
                    scratch.local_offsets.data(),
                    scratch.local_offsets.size(), primary_ids,
                    base[0].get(), aligned_dim, q_data);
          }
        } else {
          yfcc_support_complement::
              intersect_postings_to_global_id_scratch_prevalidated(
                  primary_ids, n_tag, secondary_ids, secondary_size,
                  secondary_bitmap, strategy, scratch.global_ids);
          exact_ids =
              yfcc_support_complement::
                  scan_global_id_scratch_exact_prevalidated<10>(
                      scratch.global_ids.data(), scratch.global_ids.size(),
                      base[0].get(), aligned_dim, q_data);
        }
        for (int32_t global_id : exact_ids) {
          if (global_id >= 0) top.push_back(global_id);
        }
        direct_support_complement_result = true;
      } else {
        auto packed_it = packed_cold.find(T);
        if (packed_it != packed_cold.end()) {
          const uint8_t* packed = packed_it->second.data();
          size_t dim = aligned_dim;
          // ADAPTIVE PACH-in-brute: activate only when secondary density × CHUNK_SIZE < 1
          // (the regime where skip ratio (1-ρ)^c > 1/e, i.e. cost-effective).
          // Per-query check: only if (sec_freq / N_total) × CHUNK_SIZE < 1.
          const uint64_t* tag_chunk_bvs = nullptr;
          if (use_pach && sec >= 0) {
            int64_t sec_freq = bmt.row_offsets[sec+1] - bmt.row_offsets[sec];
            double rho_sec = (double)sec_freq / (double)bm.n_points;
            if (rho_sec * CHUNK_SIZE < 1.0) {
              auto cb_it = chunk_bvs.find(T);
              if (cb_it != chunk_bvs.end()) tag_chunk_bvs = cb_it->second.data();
            }
          }
          if (tag_chunk_bvs) {
            // PACH-in-brute: chunk-level emptiness skip integrated INTO flat loop
            // (preserves SIMD/compiler optimization; only adds chunk-aligned check).
            int chunk_word_off = sec >> 6;
            uint64_t chunk_bit_mask = 1ULL << (sec & 63);
            int64_t local_skip = 0, local_total = 0;
            size_t k = 0;
            while (k < n_tag) {
              // Chunk-boundary check: skip whole chunk if secondary absent
              size_t chunk_id = k / CHUNK_SIZE;
              ++local_total;
              uint64_t bv_word = tag_chunk_bvs[chunk_id * pach_bv_words + chunk_word_off];
              if ((bv_word & chunk_bit_mask) == 0ULL) {
                ++local_skip;
                k += CHUNK_SIZE;  // jump to next chunk
                continue;
              }
              // Within chunk: tight inner loop (no extra branches)
              size_t chunk_end = std::min(k + CHUNK_SIZE, n_tag);
              do {
                int32_t g = bmt.row_indices[start + k];
                if (has_sec_brute(g)) {
                  float d = l2_sq_uint8_avx2(q_data, packed + k * dim, q_dim);
                  heap_insert_brute(d, g);
                }
                ++k;
              } while (k < chunk_end);
            }
            pach_total_clusters.fetch_add(local_total, std::memory_order_relaxed);
            pach_kept_clusters.fetch_add(local_total - local_skip, std::memory_order_relaxed);
          } else {
            // PACH off (use_pach=0) or no chunk bitvec — original flat scan
            for (size_t k = 0; k < n_tag; ++k) {
              int32_t g = bmt.row_indices[start + k];
              if (!has_sec_brute(g)) continue;
              float d = l2_sq_uint8_avx2(q_data, packed + k * dim, q_dim);
              heap_insert_brute(d, g);
            }
          }
        } else {
          // Fallback: random base[g] with prefetch + AVX2 SIMD distance
          const int PREFETCH_AHEAD = 16;
          for (size_t k = 0; k + PREFETCH_AHEAD < n_tag; ++k) {
            int32_t gp = bmt.row_indices[start + k + PREFETCH_AHEAD];
            base[gp].prefetch();
          }
          for (size_t k = 0; k < n_tag; ++k) {
            int32_t g = bmt.row_indices[start + k];
            if (!has_sec_brute(g)) continue;
            if (k + PREFETCH_AHEAD < n_tag) {
              int32_t gp = bmt.row_indices[start + k + PREFETCH_AHEAD];
              base[gp].prefetch();
            }
            const uint8_t* bp_data = base[g].get();
            float d = l2_sq_uint8_avx2(q_data, bp_data, q_dim);
            heap_insert_brute(d, g);
          }
        }
      }
      if (!direct_support_complement_result) {
        for (int k = 0; k < K; ++k) {
          if (frontier[k].second >= 0) top.push_back(frontier[k].second);
        }
      }
    }
    while ((int)top.size() < K) top.push_back(-1);
    results[i] = std::move(top);

    auto t_b = std::chrono::steady_clock::now();
    latencies[i] = std::chrono::duration<double>(t_b-t_a).count() * 1000.0;
  });
  auto t_q1 = std::chrono::steady_clock::now();
  const double phase_graph_seconds =
      std::chrono::duration<double>(t_phase_graph - t_q0).count();
  const double phase_exact_seconds =
      std::chrono::duration<double>(
          t_phase_exact - t_phase_graph).count();
  const double phase_pair_seconds =
      std::chrono::duration<double>(
          t_phase_pair - t_phase_exact).count();
  const double phase_brute_seconds =
      std::chrono::duration<double>(t_q1 - t_phase_pair).count();
  double qs_iter = std::chrono::duration<double>(t_q1-t_q0).count();
  run_walls.push_back(qs_iter);
  bool predictions_identical_to_first = true;
  if (defer_pass_reporting) {
    const size_t stride =
        static_cast<size_t>(n_active) * static_cast<size_t>(K);
    size_t offset = static_cast<size_t>(run_iter) * stride;
    for (int i = 0; i < n_q; ++i) {
      if (q_route[i] == -1) continue;
      if (results[i].size() != static_cast<size_t>(K)) {
        throw std::runtime_error("deferred result width differs");
      }
      std::copy(results[i].begin(), results[i].end(),
                deferred_measured_ids.begin() + offset);
      offset += static_cast<size_t>(K);
    }
    if (offset != (static_cast<size_t>(run_iter) + 1) * stride) {
      throw std::runtime_error("deferred result coverage differs");
    }
  } else if (run_iter == 0) {
      first_pass_results = results;
  } else {
    predictions_identical_to_first = results == first_pass_results;
    all_pass_predictions_identical =
        all_pass_predictions_identical &&
        predictions_identical_to_first;
  }
  if (n_runs > 1 || pass_csv) {
    std::vector<double> pass_lat;
    pass_lat.reserve(n_q);
    if (!defer_pass_reporting) {
      for (int i = 0; i < n_q; ++i) {
        if (q_route[i] != -1) pass_lat.push_back(latencies[i]);
      }
      std::sort(pass_lat.begin(), pass_lat.end());
    }
    auto pass_pct = [&](double p) {
      if (defer_pass_reporting) return 0.0;
      if (pass_lat.empty()) return 0.0;
      size_t idx = std::min(pass_lat.size() - 1,
                            (size_t)(p * (pass_lat.size() - 1)));
      return pass_lat[idx];
    };
    if (n_runs > 1 && !defer_pass_reporting) {
      printf("  [run %d/%d] wall=%.3fs QPS=%.1f p50=%.3fms "
             "p95=%.3fms p99=%.3fms graph=%.3fs exact=%.3fs "
             "pair=%.3fs brute=%.3fs identical=%d\n",
             run_iter+1, n_runs, qs_iter, n_active/qs_iter,
             pass_pct(0.50), pass_pct(0.95), pass_pct(0.99),
             phase_graph_seconds, phase_exact_seconds,
             phase_pair_seconds, phase_brute_seconds,
             (int)predictions_identical_to_first);
      if (reuse_beam_workspace) {
        size_t staged_candidates = 0;
        size_t early_abandoned = 0;
        for (const auto& workspace : reusable_beam_workspaces) {
          staged_candidates += workspace.staged_distance_candidates;
          early_abandoned += workspace.early_abandoned_candidates;
        }
        printf("  [staged graph L2] candidates=%zu early_abandoned=%zu "
               "fraction=%.6f\n",
               staged_candidates, early_abandoned,
               staged_candidates == 0
                   ? 0.0
                   : static_cast<double>(early_abandoned) /
                         static_cast<double>(staged_candidates));
      }
      if (batched_l2_beam || fixed_frontier_batched_l2) {
        size_t staged_candidates = 0;
        size_t prefix_rejected = 0;
        for (int i = 0; i < n_q; ++i) {
          staged_candidates += q_graph_staged_candidates[i];
          prefix_rejected += q_graph_prefix_rejected[i];
        }
        printf("  [batched graph L2] candidates=%zu "
               "prefix_rejected=%zu fraction=%.6f\n",
               staged_candidates, prefix_rejected,
               staged_candidates == 0
                   ? 0.0
                   : static_cast<double>(prefix_rejected) /
                         static_cast<double>(staged_candidates));
#ifdef BCI_ENABLE_BITPLANE_LB_DIAGNOSTIC
        const double rejection_fraction =
            staged_candidates == 0
                ? 0.0
                : static_cast<double>(prefix_rejected) /
                      static_cast<double>(staged_candidates);
        const double high_bytes =
            192.0 * static_cast<double>(bitplane_high_bits) / 8.0;
        const double residual_bytes =
            192.0 - high_bytes;
        const double expected_payload_bytes =
            high_bytes +
            (1.0 - rejection_fraction) * residual_bytes;
        printf("  [bit-plane lower bound] retained_bits=%d "
               "high_bytes=%.3f residual_bytes=%.3f "
               "expected_navigation_payload_bytes=%.3f "
               "payload_upper_bound_speedup=%.6f "
               "timing_claim=0 exact_fallback=1\n",
               bitplane_high_bits, high_bytes, residual_bytes,
               expected_payload_bytes,
               192.0 / expected_payload_bytes);
#endif
      }
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
      {
        size_t triggered = 0;
        size_t exact_records = 0;
        size_t width_records = 0;
        size_t cutoff_records = 0;
        size_t prefix_records = 0;
        size_t metric_excluded = 0;
        size_t handoff_records = 0;
        size_t extra_comps = 0;
        size_t new_visits = 0;
        for (int i = 0; i < n_q; ++i) {
          triggered += q_eviction_spill_triggered[i];
          exact_records += q_eviction_spill_exact_records[i];
          width_records += q_eviction_spill_width_records[i];
          cutoff_records += q_eviction_spill_cutoff_records[i];
          prefix_records += q_eviction_spill_prefix_records[i];
          metric_excluded += q_eviction_spill_metric_excluded[i];
          handoff_records += q_eviction_spill_handoff_records[i];
          extra_comps += q_eviction_spill_extra_comps[i];
          new_visits += q_eviction_spill_new_visits[i];
        }
        printf("  [eviction spill] triggered=%zu exact_records=%zu "
               "width_records=%zu cutoff_records=%zu "
               "prefix_records=%zu metric_excluded=%zu "
               "handoff_records=%zu extra_comps=%zu "
               "new_visits=%zu cap=8 added_width=26\n",
               triggered, exact_records, width_records,
               cutoff_records, prefix_records, metric_excluded,
               handoff_records, extra_comps, new_visits);
      }
#endif
#ifdef BCI_ENABLE_Q4_GRAPH_DIAGNOSTIC
      {
        size_t rerank_candidates = 0;
        size_t graph_queries = 0;
        for (int i = 0; i < n_q; ++i) {
          if (q_route[i] == 0) {
            ++graph_queries;
            rerank_candidates +=
                q_q4_exact_rerank_candidates[i];
          }
        }
        printf("  [q4 graph] graph_queries=%zu "
               "exact_rerank_candidates=%zu mean=%.3f "
               "navigation_payload=unpacked_quality_only\n",
               graph_queries, rerank_candidates,
               graph_queries == 0
                   ? 0.0
                   : static_cast<double>(rerank_candidates) /
                         static_cast<double>(graph_queries));
      }
#endif
    }
    if (pass_csv.is_open()) {
      if (defer_pass_reporting) {
        deferred_pass_csv_rows.push_back(DeferredPassCsvRow{
            run_iter + 1, qs_iter, n_active / qs_iter,
            pass_pct(0.50), pass_pct(0.95), pass_pct(0.99),
            (int)predictions_identical_to_first});
      } else {
        pass_csv << run_iter + 1 << ',' << n_active << ','
                 << parlay::num_workers() << ',' << std::setprecision(12)
                 << qs_iter << ',' << n_active / qs_iter << ','
                 << pass_pct(0.50) << ',' << pass_pct(0.95) << ','
                 << pass_pct(0.99) << ','
                 << (int)predictions_identical_to_first << '\n';
        pass_csv.flush();
        if (!pass_csv) {
          throw std::runtime_error("failed to flush BCI_PASS_CSV");
        }
        if (has_pass_barrier) {
          fsync_file_path(pass_csv_path);
        }
      }
    }
    if (has_pass_barrier) {
      std::ostringstream stem_builder;
      stem_builder << "pass_" << std::setw(2) << std::setfill('0')
                   << (run_iter + 1);
      const std::string stem = stem_builder.str();
      const std::filesystem::path raw_partial =
          pass_artifact_dir / (stem + ".raw.csv.partial");
      const std::filesystem::path raw_path =
          pass_artifact_dir / (stem + ".raw.csv");
      const std::filesystem::path ready_partial =
          pass_artifact_dir / (stem + ".ready.partial");
      const std::filesystem::path ready_path =
          pass_artifact_dir / (stem + ".ready");
      const std::filesystem::path release_path =
          pass_artifact_dir / (stem + ".release");
      for (const auto& path :
           {raw_partial, raw_path, ready_partial, ready_path, release_path}) {
        if (std::filesystem::exists(path)) {
          throw std::runtime_error("pass-barrier artifact already exists: " +
                                   path.string());
        }
      }
      {
        std::ofstream raw(raw_partial, std::ios::out | std::ios::trunc);
        if (!raw) {
          throw std::runtime_error("cannot create pass raw artifact");
        }
        raw << "qid,latency_ms";
        for (int k = 0; k < K; ++k) raw << ",result_" << k;
        raw << '\n' << std::setprecision(12);
        for (int i = 0; i < n_q; ++i) {
          if (q_route[i] == -1) continue;
          raw << (qid_lo + i) << ',' << latencies[i];
          if ((int)results[i].size() != K) {
            throw std::runtime_error("pass result row has wrong width");
          }
          for (int32_t id : results[i]) raw << ',' << id;
          raw << '\n';
        }
        raw.flush();
        if (!raw) {
          throw std::runtime_error("failed to flush pass raw artifact");
        }
      }
      durable_publish_no_replace(raw_partial, raw_path);
      {
        std::ofstream ready(ready_partial, std::ios::out | std::ios::trunc);
        if (!ready) {
          throw std::runtime_error("cannot create pass ready artifact");
        }
        ready << "schema=bci-pass-ready-v1\n"
              << "run_token=" << run_token << '\n'
              << "pass=" << (run_iter + 1) << '\n'
              << "qid_lo=" << qid_lo << '\n'
              << "qid_hi=" << qid_hi << '\n'
              << "k=" << K << '\n'
              << "rows=" << n_active << '\n'
              << "threads=" << parlay::num_workers() << '\n'
              << "raw_file=" << raw_path.filename().string() << '\n'
              << std::setprecision(12)
              << "wall_seconds=" << qs_iter << '\n'
              << "qps=" << (n_active / qs_iter) << '\n'
              << "p50_ms=" << pass_pct(0.50) << '\n'
              << "p95_ms=" << pass_pct(0.95) << '\n'
              << "p99_ms=" << pass_pct(0.99) << '\n';
        ready.flush();
        if (!ready) {
          throw std::runtime_error("failed to flush pass ready artifact");
        }
      }
      durable_publish_no_replace(ready_partial, ready_path);
      printf("[pass barrier] pass=%d ready=%s waiting_release=1\n",
             run_iter + 1, ready_path.c_str());
      std::fflush(stdout);

      const auto wait_start = std::chrono::steady_clock::now();
      while (!std::filesystem::exists(release_path)) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - wait_start);
        if (elapsed.count() >= pass_barrier_timeout_seconds) {
          throw std::runtime_error("timed out waiting for pass release");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      auto release = read_key_value_file(release_path);
      const std::string expected_ack = stem + ".ack";
      if (release.size() != 5 ||
          release["schema"] != "bci-pass-release-v1" ||
          release["run_token"] != run_token ||
          release["pass"] != std::to_string(run_iter + 1) ||
          release["ack_file"] != expected_ack) {
        throw std::runtime_error("pass release identity mismatch");
      }
      const std::filesystem::path ack_path =
          pass_artifact_dir / release["ack_file"];
      if (!std::filesystem::is_regular_file(ack_path)) {
        throw std::runtime_error("pass release references missing acknowledgement");
      }
      auto acknowledgement = read_key_value_file(ack_path);
      const std::string expected_raw = stem + ".raw.csv";
      size_t expected_acknowledgement_fields = 17;
#ifdef BCI_REQUIRE_YFCC_PAIR_DENSE_CALIBRATION
      expected_acknowledgement_fields = 18;
#endif
      if (acknowledgement.size() != expected_acknowledgement_fields ||
          acknowledgement["schema"] != "bci-pass-ack-v1" ||
          acknowledgement["run_token"] != run_token ||
          acknowledgement["pass"] != std::to_string(run_iter + 1) ||
          acknowledgement["raw_file"] != expected_raw ||
          acknowledgement["ready_file"] != ready_path.filename().string() ||
          acknowledgement["query_mode"] != query_mode ||
          acknowledgement["gt_sha256"] != gt_sha256_binding ||
          acknowledgement["active_qids_sha256"] !=
              active_qids_sha256_binding ||
          acknowledgement["audited_rows"] != std::to_string(n_active) ||
          acknowledgement["strict_recall_min"] != "1.000000" ||
          acknowledgement["invalid_returned_ids"] != "0" ||
          acknowledgement["duplicate_returned_ids"] != "0" ||
          acknowledgement["status"] != "pass") {
        throw std::runtime_error("pass acknowledgement audit contract mismatch");
      }
#ifdef BCI_REQUIRE_YFCC_PAIR_DENSE_CALIBRATION
      if (acknowledgement["tie_aware_recall_min"] != "1.000000") {
        throw std::runtime_error(
            "dense pass acknowledgement tie-aware audit mismatch");
      }
#endif
      std::ostringstream observed_raw_hash;
      observed_raw_hash << std::hex << std::setw(16) << std::setfill('0')
                        << fnv1a64_file(raw_path);
      if (acknowledgement["raw_fnv1a64"] != observed_raw_hash.str()) {
        throw std::runtime_error("pass acknowledgement raw hash mismatch");
      }
      std::ostringstream observed_ready_hash;
      observed_ready_hash << std::hex << std::setw(16) << std::setfill('0')
                          << fnv1a64_file(ready_path);
      if (acknowledgement["ready_fnv1a64"] != observed_ready_hash.str() ||
          !std::isfinite(std::stod(acknowledgement["wall_seconds"])) ||
          !std::isfinite(std::stod(acknowledgement["qps"])) ||
          std::abs(std::stod(acknowledgement["wall_seconds"]) - qs_iter) >
              std::max(1e-9, std::abs(qs_iter) * 1e-9) ||
          std::abs(std::stod(acknowledgement["qps"]) -
                   n_active / qs_iter) >
              std::max(1e-9, std::abs(n_active / qs_iter) * 1e-9)) {
        throw std::runtime_error("pass acknowledgement summary/ready mismatch");
      }
      std::ostringstream observed_hash;
      observed_hash << std::hex << std::setw(16) << std::setfill('0')
                    << fnv1a64_file(ack_path);
      if (release["ack_fnv1a64"] != observed_hash.str()) {
        throw std::runtime_error("pass release acknowledgement hash mismatch");
      }
      printf("[pass barrier] pass=%d released=1 ack=%s\n",
             run_iter + 1, ack_path.c_str());
      std::fflush(stdout);
    }
  }
  }  // end outer warmup loop
  if (defer_pass_reporting) {
    const size_t stride =
        static_cast<size_t>(n_active) * static_cast<size_t>(K);
    for (size_t pass = 1; pass < static_cast<size_t>(n_runs); ++pass) {
      all_pass_predictions_identical =
          all_pass_predictions_identical && std::equal(
              deferred_measured_ids.begin(),
              deferred_measured_ids.begin() + stride,
              deferred_measured_ids.begin() + pass * stride);
    }
  }
  if (defer_pass_reporting && pass_csv.is_open()) {
    for (auto& row : deferred_pass_csv_rows) {
      row.predictions_identical = (int)all_pass_predictions_identical;
      pass_csv << row.pass << ',' << n_active << ',' << parlay::num_workers()
               << ',' << std::setprecision(12) << row.wall_seconds << ','
               << row.qps << ',' << row.p50_ms << ',' << row.p95_ms << ','
               << row.p99_ms << ',' << row.predictions_identical << '\n';
    }
    pass_csv.flush();
    if (!pass_csv) {
      throw std::runtime_error("failed to flush deferred BCI_PASS_CSV");
    }
    printf("[deferred pass reporting] passes=%zu measurement_block_io=0\n",
           deferred_pass_csv_rows.size());
  }
  printf("[repeat output audit] all_pass_predictions_identical=%d "
         "passes=%d\n",
         (int)all_pass_predictions_identical, n_runs);
  // Use the LAST run's timing as the steady-state warm measurement
  double qs = run_walls.back();
  double qps = n_active / qs;

  // -- Recall@K --------------------------------------------------------------
  // BCI_TIE_AWARE=1 → NeurIPS'23 BigANN official protocol (distance-threshold):
  //   result v counts as match iff dist(q, v) <= dist(q, GT[K-1]).
  // We report this secondary metric beside strict recall.  The prospective GT
  // has deterministic (distance, ascending-ID) ties, which exact routes match.
  const bool TIE_AWARE = std::getenv("BCI_TIE_AWARE") != nullptr &&
                         std::atoi(std::getenv("BCI_TIE_AWARE")) != 0;
  double total = 0.0; int counted = 0;
  unsigned d_dim = (unsigned)base.dimension();
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == -1) continue;
    int hit = 0;
    if (TIE_AWARE) {
      const auto gt_begin = gt.distances.begin() + (size_t)i * gt.K;
      float tau = *std::max_element(gt_begin, gt_begin + K);
      const uint8_t* q_data = query[qid_lo + i].get();
      for (auto v : results[i]) {
        if (v < 0) continue;
        float d_v = l2_sq_uint8_avx2(q_data, base[v].get(), d_dim);
        if (d_v <= tau) ++hit;
      }
    } else {
      std::vector<uint32_t> gt_set(gt.indices.begin() + i*gt.K,
                                    gt.indices.begin() + i*gt.K + K);
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    total += double(hit) / K;
    ++counted;
  }
  double recall = counted > 0 ? total / counted : 0.0;

  // Predicate-validity and duplicate audit is deliberately outside the timed
  // region.  A benchmark run is invalid if any returned ID violates its filter
  // or if a result row contains the same valid ID more than once.
  int64_t invalid_returned_ids = 0;
  int64_t duplicate_returned_ids = 0;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] == -1) continue;
    std::unordered_set<int32_t> seen;
    int qid = qid_lo + i;
    int64_t q_start = qm.row_offsets[qid];
    int64_t q_end = qm.row_offsets[qid + 1];
    for (int32_t v : results[i]) {
      if (v < 0) continue;
      if (v >= bm.n_points) {
        ++invalid_returned_ids;
        continue;
      }
      duplicate_returned_ids += !seen.insert(v).second;
      for (int64_t j = q_start; j < q_end; ++j) {
        if (!bm.match(v, qm.row_indices[j])) {
          ++invalid_returned_ids;
          break;
        }
      }
    }
  }
  printf("[output audit] invalid_returned_ids=%ld duplicate_returned_ids=%ld\n",
         (long)invalid_returned_ids, (long)duplicate_returned_ids);
  if (invalid_returned_ids != 0 || duplicate_returned_ids != 0) {
    throw std::runtime_error("invalid or duplicate returned IDs");
  }
  int graph_retry_queries = 0;
  int graph_exact_retry_queries = 0;
  int graph_alt_retry_queries = 0;
  uint64_t graph_retry_dist_comps = 0;
  uint64_t graph_retry_cache_hits = 0;
  uint64_t graph_retry_uncached_candidates = 0;
  uint64_t graph_retry_exact_completions = 0;
  uint64_t graph_retry_uncached_prefix_rejections = 0;
  uint64_t graph_alt_retry_dist_comps = 0;
  for (int i = 0; i < n_q; ++i) {
    graph_retry_queries += q_graph_retried[i];
    graph_exact_retry_queries += q_graph_exact_retried[i];
    graph_alt_retry_queries += q_graph_alt_retried[i];
    graph_retry_dist_comps += q_graph_retry_dist_comps[i];
    graph_retry_cache_hits += q_graph_retry_cache_hits[i];
    graph_retry_uncached_candidates +=
        q_graph_retry_uncached_candidates[i];
    graph_retry_exact_completions +=
        q_graph_retry_exact_completions[i];
    graph_retry_uncached_prefix_rejections +=
        q_graph_retry_uncached_prefix_rejections[i];
    graph_alt_retry_dist_comps += q_graph_alt_retry_dist_comps[i];
  }
  printf("[single retry outcome] queries=%d fraction_all=%.6f "
         "retry_dist_comps=%lu\n",
         graph_retry_queries,
         n_active ? (double)graph_retry_queries / (double)n_active : 0.0,
         (unsigned long)graph_retry_dist_comps);
  if (single_retry_distance_cache &&
      (graph_retry_cache_hits + graph_retry_uncached_candidates !=
           graph_retry_dist_comps ||
       graph_retry_exact_completions +
               graph_retry_uncached_prefix_rejections !=
           graph_retry_uncached_candidates)) {
    throw std::runtime_error(
        "single-retry aggregate distance-cache accounting differs");
  }
  printf("[single retry distance-cache outcome] enabled=%d "
         "logical=%lu hits=%lu uncached_candidates=%lu "
         "exact_completions=%lu uncached_prefix_rejects=%lu "
         "hit_rate=%.6f\n",
         (int)single_retry_distance_cache,
         (unsigned long)graph_retry_dist_comps,
         (unsigned long)graph_retry_cache_hits,
         (unsigned long)graph_retry_uncached_candidates,
         (unsigned long)graph_retry_exact_completions,
         (unsigned long)graph_retry_uncached_prefix_rejections,
         graph_retry_dist_comps == 0
             ? 0.0
             : (double)graph_retry_cache_hits /
                   (double)graph_retry_dist_comps);
  printf("[single exact-retry outcome] queries=%d fraction_all=%.6f\n",
         graph_exact_retry_queries,
         n_active ? (double)graph_exact_retry_queries /
                        (double)n_active
                  : 0.0);
  printf("[single alternate-retry outcome] queries=%d fraction_all=%.6f "
         "retry_dist_comps=%lu\n",
         graph_alt_retry_queries,
         n_active ? (double)graph_alt_retry_queries /
                        (double)n_active
                  : 0.0,
         (unsigned long)graph_alt_retry_dist_comps);
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
  size_t cached_graph_queries = 0;
  size_t cached_probe_churn_queries = 0;
  size_t cached_repair_queries = 0;
  uint64_t cached_base_logical = 0;
  uint64_t cached_base_hits = 0;
  uint64_t cached_base_misses = 0;
  uint64_t cached_base_completions = 0;
  uint64_t cached_base_prefix_rejections = 0;
  uint64_t cached_probe_logical = 0;
  uint64_t cached_probe_hits = 0;
  uint64_t cached_probe_misses = 0;
  uint64_t cached_probe_completions = 0;
  uint64_t cached_probe_prefix_rejections = 0;
  uint64_t cached_repair_logical = 0;
  uint64_t cached_repair_hits = 0;
  uint64_t cached_repair_misses = 0;
  uint64_t cached_repair_completions = 0;
  uint64_t cached_repair_prefix_rejections = 0;
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] != 0) continue;
    ++cached_graph_queries;
    cached_probe_churn_queries +=
        q_cr_probe_top10_entered[i] != 0 ||
        q_cr_probe_top10_dropped[i] != 0;
    cached_repair_queries += q_cr_repair_ran[i];
    cached_base_logical += q_graph_dist_comps[i];
    cached_base_hits += q_cr_base_cache_hits[i];
    cached_base_misses += q_cr_base_exact_misses[i];
    cached_base_completions += q_cr_base_exact_completions[i];
    cached_base_prefix_rejections +=
        q_cr_base_uncached_prefix_rejections[i];
    cached_probe_logical += q_cr_probe_logical_comps[i];
    cached_probe_hits += q_cr_probe_cache_hits[i];
    cached_probe_misses += q_cr_probe_exact_misses[i];
    cached_probe_completions += q_cr_probe_exact_completions[i];
    cached_probe_prefix_rejections +=
        q_cr_probe_uncached_prefix_rejections[i];
    cached_repair_logical += q_cr_repair_logical_comps[i];
    cached_repair_hits += q_cr_repair_cache_hits[i];
    cached_repair_misses += q_cr_repair_exact_misses[i];
    cached_repair_completions +=
        q_cr_repair_exact_completions[i];
    cached_repair_prefix_rejections +=
        q_cr_repair_uncached_prefix_rejections[i];
  }
  if (cached_base_hits + cached_base_misses !=
          cached_base_logical ||
      cached_probe_hits + cached_probe_misses !=
          cached_probe_logical ||
      cached_repair_hits + cached_repair_misses !=
          cached_repair_logical ||
      cached_base_completions + cached_base_prefix_rejections !=
          cached_base_misses ||
      cached_probe_completions + cached_probe_prefix_rejections !=
          cached_probe_misses ||
      cached_repair_completions +
              cached_repair_prefix_rejections !=
          cached_repair_misses ||
      cached_probe_churn_queries != cached_repair_queries) {
    throw std::runtime_error(
        "cached replay logical/hit/miss or churn/repair audit failed");
  }
  printf("[cached replay outcome] graph_queries=%zu "
         "probe_churn_queries=%zu fraction_graph=%.6f "
         "base_logical=%lu base_hits=%lu base_physical_misses=%lu "
         "base_exact_completions=%lu base_prefix_rejects=%lu "
         "probe_logical=%lu probe_hits=%lu "
         "probe_physical_misses=%lu probe_exact_completions=%lu "
         "probe_prefix_rejects=%lu repair_queries=%zu "
         "repair_logical=%lu repair_hits=%lu "
         "repair_physical_misses=%lu repair_exact_completions=%lu "
         "repair_prefix_rejects=%lu "
         "total_extra_physical_misses=%lu\n",
         cached_graph_queries, cached_probe_churn_queries,
         cached_graph_queries == 0
             ? 0.0
             : static_cast<double>(cached_probe_churn_queries) /
                   static_cast<double>(cached_graph_queries),
         static_cast<unsigned long>(cached_base_logical),
         static_cast<unsigned long>(cached_base_hits),
         static_cast<unsigned long>(cached_base_misses),
         static_cast<unsigned long>(cached_base_completions),
         static_cast<unsigned long>(cached_base_prefix_rejections),
         static_cast<unsigned long>(cached_probe_logical),
         static_cast<unsigned long>(cached_probe_hits),
         static_cast<unsigned long>(cached_probe_misses),
         static_cast<unsigned long>(cached_probe_completions),
         static_cast<unsigned long>(cached_probe_prefix_rejections),
         cached_repair_queries,
         static_cast<unsigned long>(cached_repair_logical),
         static_cast<unsigned long>(cached_repair_hits),
         static_cast<unsigned long>(cached_repair_misses),
         static_cast<unsigned long>(cached_repair_completions),
         static_cast<unsigned long>(cached_repair_prefix_rejections),
         static_cast<unsigned long>(
             cached_probe_misses + cached_repair_misses));
#endif

  // -- Per-route statistics -----------
  // Isolate HAMCG_single (route 0), HAMCG_conj (route 1), brute (route 2) timing+recall.
  double sum_lat[3] = {0.0, 0.0, 0.0};
  int count_route[3] = {0, 0, 0};
  double sum_recall_route[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] < 0 || q_route[i] > 2) continue;
    sum_lat[q_route[i]] += latencies[i];
    count_route[q_route[i]]++;
    int hit = 0;
    if (TIE_AWARE) {
      const auto gt_begin = gt.distances.begin() + (size_t)i * gt.K;
      float tau = *std::max_element(gt_begin, gt_begin + K);
      const uint8_t* q_data = query[qid_lo + i].get();
      for (auto v : results[i]) {
        if (v < 0) continue;
        float d_v = l2_sq_uint8_avx2(q_data, base[v].get(), d_dim);
        if (d_v <= tau) ++hit;
      }
    } else {
      std::vector<uint32_t> gt_set(gt.indices.begin() + i*gt.K,
                                    gt.indices.begin() + i*gt.K + K);
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    sum_recall_route[q_route[i]] += double(hit) / K;
  }
  printf("\n[per-route stats]\n");
  const char* route_name[] = {"HAMCG_single", "HAMCG_conj  ", "brute_cold  "};
  for (int r = 0; r < 3; ++r) {
    double avg_ms = count_route[r] > 0 ? sum_lat[r] / count_route[r] : 0;
    double inverse_mean_latency_rate =
        count_route[r] > 0 ? 1000.0 * count_route[r] / sum_lat[r] : 0;
    double route_recall = count_route[r] > 0 ? sum_recall_route[r] / count_route[r] : 0;
    printf("  %s: %4d queries, avg %7.3f ms/q, "
           "%7.1f inverse_mean_latency_rate, recall@%d=%.6f\n",
           route_name[r], count_route[r], avg_ms,
           inverse_mean_latency_rate, K, route_recall);
  }

  double sum_lat_card[2] = {0.0, 0.0};
  int count_card[2] = {0, 0};
  double sum_recall_card[2] = {0.0, 0.0};
  for (int i = 0; i < n_q; ++i) {
    if (q_route[i] < 0 || q_route[i] > 2) continue;
    int c = (q_secondary[i] >= 0) ? 1 : 0;  // 0=single-tag, 1=two-tag
    sum_lat_card[c] += latencies[i];
    count_card[c]++;
    int hit = 0;
    if (TIE_AWARE) {
      const auto gt_begin = gt.distances.begin() + (size_t)i * gt.K;
      float tau = *std::max_element(gt_begin, gt_begin + K);
      const uint8_t* q_data = query[qid_lo + i].get();
      for (auto v : results[i]) {
        if (v < 0) continue;
        float d_v = l2_sq_uint8_avx2(q_data, base[v].get(), d_dim);
        if (d_v <= tau) ++hit;
      }
    } else {
      std::vector<uint32_t> gt_set(gt.indices.begin() + i*gt.K,
                                    gt.indices.begin() + i*gt.K + K);
      std::sort(gt_set.begin(), gt_set.end());
      for (auto v : results[i]) {
        if (v < 0) continue;
        if (std::binary_search(gt_set.begin(), gt_set.end(), (uint32_t)v)) ++hit;
      }
    }
    sum_recall_card[c] += double(hit) / K;
  }
  printf("\n[per-cardinality stats]\n");
  const char* card_name[] = {"single_tag", "two_tag   "};
  for (int c = 0; c < 2; ++c) {
    double avg_ms = count_card[c] > 0 ? sum_lat_card[c] / count_card[c] : 0;
    double inverse_mean_latency_rate =
        count_card[c] > 0 ? 1000.0 * count_card[c] / sum_lat_card[c] : 0;
    double card_recall = count_card[c] > 0 ? sum_recall_card[c] / count_card[c] : 0;
    printf("  %s: %4d queries, avg %7.3f ms/q, "
           "%7.1f inverse_mean_latency_rate, recall@%d=%.6f\n",
           card_name[c], count_card[c], avg_ms,
           inverse_mean_latency_rate, K, card_recall);
  }

  // -- Latency stats ---------------------------------------------------------
  // Per-query dump for cardinality-stratified analysis (set env BCI_PERQUERY_CSV to a path)
  {
    const char* pq_csv = std::getenv("BCI_PERQUERY_CSV");
    if (pq_csv) {
      FILE* fpq = std::fopen(pq_csv, "w");
      if (fpq) {
        std::fprintf(fpq,
                     "qid,n_tags,tag_a,tag_b,route,primary_tag,secondary_tag,"
                     "graph_primary,packed_primary,pair_materialized,"
                     "effective_beam,effective_cut,effective_starts,"
                     "graph_dist_comps,graph_visited,graph_frontier_size,"
                     "graph_boundary_ratio,graph_retried,"
                     "graph_retry_dist_comps,graph_retry_cache_hits,"
                     "graph_retry_uncached_candidates,"
                     "graph_retry_exact_completions,"
                     "graph_retry_uncached_prefix_rejections,"
                     "graph_exact_retried,"
                     "graph_alt_retried,graph_alt_retry_dist_comps,"
                     "latency_ms,recall");
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
        std::fprintf(
            fpq,
            ",residual_landmark_scores,residual_landmark_selected");
#endif
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
        std::fprintf(
            fpq,
            ",spill_triggered,spill_exact_records,"
            "spill_width_records,spill_cutoff_records,"
            "spill_prefix_records,spill_metric_excluded,"
            "spill_handoff_records,spill_extra_comps,"
            "spill_new_visits");
#endif
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
        std::fprintf(
            fpq,
            ",graph_top10_last_change_visit,"
            "graph_top10_last_change_dist_comps,"
            "graph_top10_stability_age,"
            "graph_top10_changed_steps_last16,"
            "graph_top10_entries_last16,"
            "graph_top10_total_changed_steps,"
            "graph_top10_total_entries,"
            "graph_top10_final_size,"
            "graph_top10_margin_abs");
#endif
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
        std::fprintf(
            fpq,
            ",cr_base_cache_hits,cr_base_physical_misses,"
            "cr_base_exact_completions,"
            "cr_base_uncached_prefix_rejections,"
            "cr_probe_beam,cr_probe_logical_comps,"
            "cr_probe_cache_hits,cr_probe_physical_misses,"
            "cr_probe_exact_completions,"
            "cr_probe_uncached_prefix_rejections,"
            "cr_probe_visited,cr_probe_top10_intersection,"
            "cr_probe_top10_entered,cr_probe_top10_dropped,"
            "cr_base_d10,cr_base_frontier_margin_abs,"
            "cr_probe_d10,cr_probe_frontier_margin_abs,"
            "cr_probe_min_entered_margin_abs,"
            "cr_probe_max_entered_margin_abs,"
            "cr_repair_beam,cr_repair_ran,"
            "cr_repair_logical_comps,cr_repair_cache_hits,"
            "cr_repair_physical_misses,"
            "cr_repair_exact_completions,"
            "cr_repair_uncached_prefix_rejections,"
            "cr_repair_visited,cr_repair_top10_intersection,"
            "cr_repair_top10_entered,cr_repair_top10_dropped,"
            "cr_repair_d10,cr_repair_frontier_margin_abs,"
            "cr_total_extra_physical_misses");
#endif
        std::fprintf(fpq, "\n");
        for (int i = 0; i < n_q; ++i) {
          if (q_route[i] < 0 || q_route[i] > 2) continue;
          // re-compute per-query recall to dump alongside
          std::vector<uint32_t> gt_set_pq(gt.indices.begin() + i*gt.K,
                                          gt.indices.begin() + i*gt.K + K);
          std::sort(gt_set_pq.begin(), gt_set_pq.end());
          int hit_pq = 0;
          for (auto v : results[i]) {
            if (v < 0) continue;
            if (std::binary_search(gt_set_pq.begin(), gt_set_pq.end(), (uint32_t)v)) ++hit_pq;
          }
          double per_q_recall = double(hit_pq) / K;
          int qid = qid_lo + i;
          int64_t q_start = qm.row_offsets[qid];
          int64_t q_end = qm.row_offsets[qid + 1];
          int n_tags = (int)(q_end - q_start);
          int32_t tag_a = n_tags >= 1 ? qm.row_indices[q_start] : -1;
          int32_t tag_b = n_tags >= 2 ? qm.row_indices[q_start + 1] : -1;
	          bool pair_materialized = q_secondary[i] >= 0 &&
	              pair_postings.contains(
	                  pair_key(q_primary[i], q_secondary[i]));
          std::fprintf(fpq,
                       "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.12g,%d,"
                       "%zu,%zu,%zu,%.12g,%d,%zu,%zu,%zu,%zu,%zu,%d,%d,%zu,"
                       "%.12g,%.12g",
                       qid, n_tags, tag_a, tag_b, q_route[i], q_primary[i],
                       q_secondary[i], shards.count(q_primary[i]) ? 1 : 0,
                       packed_cold.count(q_primary[i]) ? 1 : 0,
                       pair_materialized ? 1 : 0, q_effective_beam[i],
                       q_effective_cut[i], q_effective_starts[i],
                       q_graph_dist_comps[i], q_graph_visited[i],
                       q_graph_frontier_size[i], q_graph_boundary_ratio[i],
                       q_graph_retried[i], q_graph_retry_dist_comps[i],
                       q_graph_retry_cache_hits[i],
                       q_graph_retry_uncached_candidates[i],
                       q_graph_retry_exact_completions[i],
                       q_graph_retry_uncached_prefix_rejections[i],
                       q_graph_exact_retried[i],
                       q_graph_alt_retried[i],
                       q_graph_alt_retry_dist_comps[i], latencies[i],
                       per_q_recall);
#ifdef BCI_ENABLE_RESIDUAL_LANDMARK_DIAGNOSTIC
          std::fprintf(
              fpq, ",%zu,%d",
              q_residual_landmark_scores[i],
              q_residual_landmark_selected[i]);
#endif
#ifdef BCI_ENABLE_EVICTION_SPILL_DIAGNOSTIC
          std::fprintf(
              fpq, ",%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu",
              q_eviction_spill_triggered[i],
              q_eviction_spill_exact_records[i],
              q_eviction_spill_width_records[i],
              q_eviction_spill_cutoff_records[i],
              q_eviction_spill_prefix_records[i],
              q_eviction_spill_metric_excluded[i],
              q_eviction_spill_handoff_records[i],
              q_eviction_spill_extra_comps[i],
              q_eviction_spill_new_visits[i]);
#endif
#ifdef BCI_ENABLE_TOP10_STABILITY_DIAGNOSTIC
          std::fprintf(
              fpq, ",%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.12g",
              q_graph_top10_last_change_visit[i],
              q_graph_top10_last_change_dist_comps[i],
              q_graph_top10_stability_age[i],
              q_graph_top10_changed_steps_last16[i],
              q_graph_top10_entries_last16[i],
              q_graph_top10_total_changed_steps[i],
              q_graph_top10_total_entries[i],
              q_graph_top10_final_size[i],
              q_graph_top10_margin_abs[i]);
#endif
#ifdef BCI_ENABLE_CACHED_REPLAY_DIAGNOSTIC
          std::fprintf(
              fpq,
              ",%zu,%zu,%zu,%zu,%d,%zu,%zu,%zu,%zu,%zu,"
              "%zu,%zu,%zu,%zu,"
              "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
              "%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
              "%.12g,%.12g,%zu",
              q_cr_base_cache_hits[i],
              q_cr_base_exact_misses[i],
              q_cr_base_exact_completions[i],
              q_cr_base_uncached_prefix_rejections[i],
              cached_replay_probe_beam,
              q_cr_probe_logical_comps[i],
              q_cr_probe_cache_hits[i],
              q_cr_probe_exact_misses[i],
              q_cr_probe_exact_completions[i],
              q_cr_probe_uncached_prefix_rejections[i],
              q_cr_probe_visited[i],
              q_cr_probe_top10_intersection[i],
              q_cr_probe_top10_entered[i],
              q_cr_probe_top10_dropped[i],
              q_cr_base_d10[i],
              q_cr_base_frontier_margin_abs[i],
              q_cr_probe_d10[i],
              q_cr_probe_frontier_margin_abs[i],
              q_cr_probe_min_entered_margin_abs[i],
              q_cr_probe_max_entered_margin_abs[i],
              cached_replay_repair_beam,
              q_cr_repair_ran[i],
              q_cr_repair_logical_comps[i],
              q_cr_repair_cache_hits[i],
              q_cr_repair_exact_misses[i],
              q_cr_repair_exact_completions[i],
              q_cr_repair_uncached_prefix_rejections[i],
              q_cr_repair_visited[i],
              q_cr_repair_top10_intersection[i],
              q_cr_repair_top10_entered[i],
              q_cr_repair_top10_dropped[i],
              q_cr_repair_d10[i],
              q_cr_repair_frontier_margin_abs[i],
              q_cr_probe_exact_misses[i] +
                  q_cr_repair_exact_misses[i]);
#endif
          std::fprintf(fpq, "\n");
        }
        std::fclose(fpq);
        std::fprintf(stderr, "[per-query] wrote %s\n", pq_csv);
      }
    }
  }

  std::vector<double> lat_sorted;
  for (int i = 0; i < n_q; ++i) if (q_route[i] != -1) lat_sorted.push_back(latencies[i]);
  std::sort(lat_sorted.begin(), lat_sorted.end());
  auto pct = [&](double p){
    if (lat_sorted.empty()) return 0.0;
    size_t idx = std::min(lat_sorted.size()-1, (size_t)(p * lat_sorted.size()));
    return lat_sorted[idx];
  };

  printf("\n[RESULTS]\n");
  printf("  total queries  : %d (counted=%d, skip=%d)\n", n_q, counted, n_skip);
  printf("  wall time      : %.3fs\n", qs);
  printf("  QPS            : %.1f\n", qps);
  printf("  recall@%d      : %.6f\n", K, recall);
  printf("  latency p50    : %.3f ms\n", pct(0.50));
  printf("  latency p90    : %.3f ms\n", pct(0.90));
  printf("  latency p99    : %.3f ms\n", pct(0.99));
  printf("  latency p999   : %.3f ms\n", pct(0.999));

  // PACH skip-ratio report
  int64_t pt = pach_total_clusters.load();
  int64_t pk = pach_kept_clusters.load();
  if (pt > 0) {
    double skip_pct = 100.0 * (1.0 - (double)pk / (double)pt);
    printf("\n[PACH] total clusters considered=%ld, kept=%ld, SKIP=%.1f%%\n",
           (long)pt, (long)pk, skip_pct);
  }

  return 0;
}
