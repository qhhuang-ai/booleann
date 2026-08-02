#pragma once

// Project-owned packed exact kernel for the v25 matched-1T lane. All support
// storage is sized before dispatch. This header has no timers, files, output,
// ground truth, or dynamic growth in execute_query_timed().

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sieve_v25_packed {

constexpr uint32_t kNb = 100000;
constexpr uint32_t kDim = 128;
constexpr uint32_t kLabels = 16;
constexpr uint32_t kTopK = 10;
constexpr size_t kMaxSupportIds = 100000;

using Pair = std::array<uint16_t, 2>;

struct PairHash {
  size_t operator()(const Pair& value) const noexcept {
    return (static_cast<size_t>(value[0]) << 16) | value[1];
  }
};

struct PackedSupport {
  std::vector<uint32_t> ids;
  std::vector<uint8_t> vectors;
};

using AtomBitmaps = std::vector<std::vector<uint64_t>>;
using ScratchIds = std::array<uint32_t, kMaxSupportIds>;

struct Contract {
  std::vector<Pair> design;
  std::vector<Pair> eval;
  std::vector<Pair> admitted;
  std::vector<uint8_t> design_queries;
  std::vector<uint8_t> eval_queries;
};

struct QueryResult {
  std::array<uint32_t, kTopK> ids{};
};

inline std::vector<uint32_t> intersect_pair(
    const Pair& pair, const std::vector<std::vector<uint32_t>>& postings) {
  const auto& left = postings.at(pair[0]);
  const auto& right = postings.at(pair[1]);
  std::vector<uint32_t> result;
  result.reserve(std::min(left.size(), right.size()));
  size_t left_offset = 0;
  size_t right_offset = 0;
  while (left_offset < left.size() && right_offset < right.size()) {
    if (left[left_offset] < right[right_offset]) {
      ++left_offset;
    } else if (right[right_offset] < left[left_offset]) {
      ++right_offset;
    } else {
      result.push_back(left[left_offset]);
      ++left_offset;
      ++right_offset;
    }
  }
  return result;
}

inline AtomBitmaps make_atom_bitmaps(
    const std::vector<std::vector<uint32_t>>& postings) {
  const size_t words = (kNb + 63) / 64;
  AtomBitmaps result(kLabels, std::vector<uint64_t>(words, 0));
  if (postings.size() != kLabels) {
    throw std::runtime_error("packed posting label count differs");
  }
  for (uint32_t label = 0; label < kLabels; ++label) {
    for (uint32_t id : postings[label]) {
      if (id >= kNb) throw std::runtime_error("packed posting ID differs");
      result[label][id >> 6] |= uint64_t{1} << (id & 63);
    }
  }
  return result;
}

inline uint32_t l2_sq_uint8_avx2(const uint8_t* __restrict__ left,
                                 const uint8_t* __restrict__ right) {
  __m256i accumulator0 = _mm256_setzero_si256();
  __m256i accumulator1 = _mm256_setzero_si256();
  auto accumulate32 = [](__m256i& accumulator, const uint8_t* x,
                         const uint8_t* y) {
    const __m256i vx =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i vy =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y));
    const __m256i xlo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vx));
    const __m256i xhi =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vx, 1));
    const __m256i ylo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vy));
    const __m256i yhi =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vy, 1));
    const __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    const __m256i dhi = _mm256_sub_epi16(xhi, yhi);
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(dlo, dlo));
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(dhi, dhi));
  };
  accumulate32(accumulator0, left, right);
  accumulate32(accumulator1, left + 32, right + 32);
  accumulate32(accumulator0, left + 64, right + 64);
  accumulate32(accumulator1, left + 96, right + 96);
  const __m256i accumulator = _mm256_add_epi32(accumulator0, accumulator1);
  __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(accumulator),
                              _mm256_extracti128_si256(accumulator, 1));
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return static_cast<uint32_t>(_mm_cvtsi128_si32(sum));
}

inline void execute_query_timed(
    uint32_t logical_row, bool design, const Contract& contract,
    const std::vector<uint8_t>& base,
    const std::vector<std::vector<uint32_t>>& postings,
    const AtomBitmaps& bitmaps,
    const std::unordered_map<Pair, PackedSupport, PairHash>& packed,
    ScratchIds& scratch_ids, QueryResult& result) {
  const Pair pair = design ? contract.design.at(logical_row)
                           : contract.eval.at(logical_row);
  const uint8_t* query =
      (design ? contract.design_queries.data() : contract.eval_queries.data()) +
      static_cast<size_t>(logical_row) * kDim;
  const auto found = packed.find(pair);
  const uint32_t* ids = nullptr;
  const uint8_t* packed_vectors = nullptr;
  size_t support_size = 0;
  if (found != packed.end()) {
    ids = found->second.ids.data();
    support_size = found->second.ids.size();
    packed_vectors = found->second.vectors.data();
  } else {
    const auto& left = bitmaps.at(pair[0]);
    const auto& right = bitmaps.at(pair[1]);
    for (size_t word = 0; word < left.size(); ++word) {
      uint64_t matches = left[word] & right[word];
      while (matches != 0) {
        const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(matches));
        const uint32_t id = static_cast<uint32_t>(word * 64 + bit);
        if (id < kNb) {
          if (support_size >= scratch_ids.size()) {
            throw std::runtime_error("packed support exceeds fixed capacity");
          }
          scratch_ids[support_size++] = id;
        }
        matches &= matches - 1;
      }
    }
    ids = scratch_ids.data();
  }
  if (support_size < kTopK) {
    throw std::runtime_error("packed support below top-k");
  }
  std::array<std::pair<uint32_t, uint32_t>, kTopK> heap{};
  size_t heap_size = 0;
  for (size_t index = 0; index < support_size; ++index) {
    if (packed_vectors == nullptr && index + 16 < support_size) {
      const uint8_t* future =
          base.data() + static_cast<size_t>(ids[index + 16]) * kDim;
      _mm_prefetch(reinterpret_cast<const char*>(future), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(future + 64), _MM_HINT_T0);
    }
    const uint8_t* vector =
        packed_vectors != nullptr
            ? packed_vectors + index * kDim
            : base.data() + static_cast<size_t>(ids[index]) * kDim;
    const std::pair<uint32_t, uint32_t> scored{
        l2_sq_uint8_avx2(query, vector), ids[index]};
    if (heap_size < kTopK) {
      heap[heap_size++] = scored;
      if (heap_size == kTopK) std::make_heap(heap.begin(), heap.end());
    } else if (scored < heap.front()) {
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = scored;
      std::push_heap(heap.begin(), heap.end());
    }
  }
  if (heap_size != kTopK) {
    throw std::runtime_error("packed heap below top-k");
  }
  std::sort(heap.begin(), heap.end());
  for (uint32_t rank = 0; rank < kTopK; ++rank) {
    result.ids[rank] = heap[rank].second;
  }
}

}  // namespace sieve_v25_packed
