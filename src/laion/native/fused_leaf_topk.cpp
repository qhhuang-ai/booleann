// Project-owned one-pass exact range/DNF leaf for LAION1M float32[N,512].
//
// The input postings are immutable, strictly increasing int32 row IDs.  A
// range request scans the left posting; a DNF request merge-scans the two
// postings and coalesces an ID present in both clauses before scoring.  Each
// clause applies its own half-open numeric interval and domain predicate.  The
// query's base row must be admitted by the predicate and is removed exactly
// once.  No support/union array is produced.
//
// The retained scorer is included into this translation unit so every scored
// row uses exactly the same NumPy-2.0.2-compatible float32 reduction as the
// project-owned ABI-v7 scorer.  Eight independent rows are scored together;
// a 32-ID ring supplies the retained distance-16 prefetch without materializing
// the complete predicate result.

#ifndef LAION1M_FUSED_LEAF_TOPK_USE_EXISTING_RETAINED_SCORER
#include "exact_topk_avx2.cpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint32_t kFusedAbiVersion = 1;
constexpr std::uint64_t kFusedAuditSlots = 12;
constexpr std::int32_t kFusedModeRange = 2;
constexpr std::int32_t kFusedModeDnf2 = 3;
constexpr std::int32_t kFusedValueFloat64 = 1;
constexpr std::int32_t kFusedValueInt64 = 2;
constexpr std::size_t kPendingCapacity = 32;
constexpr std::size_t kPendingMask = kPendingCapacity - 1;
static_assert((kPendingCapacity & kPendingMask) == 0,
              "pending capacity must be a power of two");
static_assert(kPendingCapacity >=
                  kRetainedInterleaveRows + kPrefetchDistance,
              "pending ring must cover one batch and retained lookahead");

enum FusedStatus : std::int32_t {
  kFusedOk = 0,
  kFusedBadAbi = -1,
  kFusedBadArgument = -2,
  kFusedBadPosting = -3,
  kFusedBadInterval = -4,
  kFusedMissingSelf = -5,
  kFusedSupportBelowK = -6,
  kFusedNonfiniteDistance = -7,
};

enum FusedAudit : std::uint64_t {
  kFusedAuditAbi = 0,
  kFusedAuditMode = 1,
  kFusedAuditLeftVisited = 2,
  kFusedAuditRightVisited = 3,
  kFusedAuditPredicateChecks = 4,
  kFusedAuditClauseAdmissions = 5,
  kFusedAuditUniqueBeforeSelf = 6,
  kFusedAuditRemovedSelf = 7,
  kFusedAuditDistanceRows = 8,
  kFusedAuditBothClausesAdmitted = 9,
  kFusedAuditCutoffTie = 10,
  kFusedAuditInterleavedBatches = 11,
};

inline bool fused_valid_pointer(const void* pointer,
                                std::uint64_t rows) noexcept {
  return rows == 0 || pointer != nullptr;
}

inline bool fused_valid_interval(const void* values, std::int32_t kind,
                                 double lo, double hi,
                                 std::int32_t similarity) noexcept {
  return values != nullptr &&
         (kind == kFusedValueFloat64 || kind == kFusedValueInt64) &&
         (similarity == 0 || similarity == 1) && !std::isnan(lo) &&
         !std::isnan(hi) && lo < hi;
}

inline bool fused_value_in_interval(const void* values, std::int32_t kind,
                                    std::int32_t id, double lo, double hi,
                                    bool similarity) noexcept {
  double value;
  if (kind == kFusedValueFloat64) {
    value = static_cast<const double*>(values)[id];
  } else {
    value = static_cast<double>(static_cast<const std::int64_t*>(values)[id]);
  }
  if (!std::isfinite(value) || value < lo || !(value < hi)) {
    return false;
  }
  return similarity ? (value >= 0.0 && value <= 1.0) : (value > 0.0);
}

inline void fused_reset_audit(std::uint64_t* audit,
                              std::int32_t mode) noexcept {
  for (std::uint64_t index = 0; index < kFusedAuditSlots; ++index) {
    audit[index] = 0;
  }
  audit[kFusedAuditAbi] = kFusedAbiVersion;
  audit[kFusedAuditMode] = static_cast<std::uint64_t>(mode);
}

class StreamingRetainedTopK {
 public:
  StreamingRetainedTopK(const float* base, const float* query,
                        std::uint64_t* audit) noexcept
      : base_(base), query_(query), audit_(audit) {}

  void append(std::int32_t id) noexcept {
    pending_[(pending_begin_ + pending_count_) & kPendingMask] = id;
    ++pending_count_;
    if (pending_count_ == kPendingCapacity) {
      status_ = score_front_interleaved8();
    }
  }

  std::int32_t finish(std::int32_t* output_ids,
                      float* output_squared_l2) noexcept {
    if (status_ != kFusedOk) {
      return status_;
    }
    while (pending_count_ >= kRetainedInterleaveRows) {
      status_ = score_front_interleaved8();
      if (status_ != kFusedOk) {
        return status_;
      }
    }
    while (pending_count_ != 0) {
      const std::int32_t id = pending_[pending_begin_];
      pending_begin_ = (pending_begin_ + 1) & kPendingMask;
      --pending_count_;
      const float distance = squared_l2_f32_512_retained(
          query_, base_ + static_cast<std::size_t>(id) * kDimension);
      if (!std::isfinite(distance)) {
        return kFusedNonfiniteDistance;
      }
      retain(Candidate{distance, id});
      ++audit_[kFusedAuditDistanceRows];
    }
    if (best_size_ < kTopK) {
      return kFusedSupportBelowK;
    }
    std::sort(best_, best_ + best_size_, candidate_less);
    if (best_size_ > kTopK &&
        best_[kTopK - 1].distance == best_[kTopK].distance) {
      audit_[kFusedAuditCutoffTie] = 1;
    }
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      output_ids[rank] = best_[rank].id;
      output_squared_l2[rank] = best_[rank].distance;
    }
    return kFusedOk;
  }

  std::int32_t status() const noexcept { return status_; }

 private:
  void retain(const Candidate& scored) noexcept {
    if (best_size_ < kRetainedCutoff) {
      best_[best_size_] = scored;
      ++best_size_;
      std::push_heap(best_, best_ + best_size_, candidate_less);
    } else if (candidate_less(scored, best_[0])) {
      std::pop_heap(best_, best_ + best_size_, candidate_less);
      best_[best_size_ - 1] = scored;
      std::push_heap(best_, best_ + best_size_, candidate_less);
    }
  }

  std::int32_t score_front_interleaved8() noexcept {
    const float* vectors[kRetainedInterleaveRows];
    const float* future[kRetainedInterleaveRows];
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      const std::int32_t id =
          pending_[(pending_begin_ + lane) & kPendingMask];
      vectors[lane] =
          base_ + static_cast<std::size_t>(id) * kDimension;
      const std::size_t future_position = lane + kPrefetchDistance;
      if (future_position < pending_count_) {
        const std::int32_t future_id =
            pending_[(pending_begin_ + future_position) & kPendingMask];
        future[lane] =
            base_ + static_cast<std::size_t>(future_id) * kDimension;
        if constexpr (kRetainedPrefetchPolicy == 0) {
          prefetch_row(future[lane]);
        }
      } else {
        future[lane] = nullptr;
      }
    }
    float distances[kRetainedInterleaveRows];
    squared_l2_f32_512_retained_interleaved8(
        query_, vectors, future, distances);
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      if (!std::isfinite(distances[lane])) {
        return kFusedNonfiniteDistance;
      }
      const std::int32_t id =
          pending_[(pending_begin_ + lane) & kPendingMask];
      retain(Candidate{distances[lane], id});
    }
    pending_begin_ =
        (pending_begin_ + kRetainedInterleaveRows) & kPendingMask;
    pending_count_ -= kRetainedInterleaveRows;
    audit_[kFusedAuditDistanceRows] += kRetainedInterleaveRows;
    ++audit_[kFusedAuditInterleavedBatches];
    return kFusedOk;
  }

  const float* base_;
  const float* query_;
  std::uint64_t* audit_;
  Candidate best_[kRetainedCutoff];
  std::size_t best_size_ = 0;
  std::int32_t pending_[kPendingCapacity];
  std::size_t pending_begin_ = 0;
  std::size_t pending_count_ = 0;
  std::int32_t status_ = kFusedOk;
};

inline bool fused_valid_id(std::int32_t id,
                           std::uint64_t base_rows) noexcept {
  return id >= 0 && static_cast<std::uint64_t>(id) < base_rows;
}

}  // namespace

extern "C" std::uint32_t
laion1m_fused_leaf_topk_abi_version_v1() noexcept {
  return kFusedAbiVersion;
}

extern "C" std::uint32_t laion1m_fused_leaf_topk_dimension_v1() noexcept {
  return static_cast<std::uint32_t>(kDimension);
}

extern "C" std::uint32_t laion1m_fused_leaf_topk_k_v1() noexcept {
  return static_cast<std::uint32_t>(kTopK);
}

extern "C" std::uint64_t
laion1m_fused_leaf_topk_audit_slots_v1() noexcept {
  return kFusedAuditSlots;
}

extern "C" std::int32_t laion1m_fused_leaf_topk_validate_posting_v1(
    const std::int32_t* posting, std::uint64_t rows,
    std::uint64_t base_rows) noexcept {
  if (base_rows == 0 || base_rows > 0x7fffffffULL ||
      rows > base_rows || !fused_valid_pointer(posting, rows)) {
    return kFusedBadArgument;
  }
  std::int32_t previous = -1;
  for (std::uint64_t index = 0; index < rows; ++index) {
    const std::int32_t id = posting[index];
    if (!fused_valid_id(id, base_rows) ||
        (index != 0 && id <= previous)) {
      return kFusedBadPosting;
    }
    previous = id;
  }
  return kFusedOk;
}

// Status: 0=success, -1=ABI, -2=argument, -3=posting, -4=interval,
// -5=self not admitted exactly once, -6=post-self support below ten,
// -7=non-finite retained distance.
//
// Equal retained distances use the native scorer's total order (distance, ID),
// so a cutoff tie remains deterministic and needs no materialized fallback.
extern "C" std::int32_t laion1m_fused_leaf_top10_i32_f32_retained_v1(
    std::uint32_t abi_version, std::int32_t mode,
    const float* base, std::uint64_t base_rows, const float* query,
    const std::int32_t* left, std::uint64_t left_rows,
    const std::int32_t* right, std::uint64_t right_rows,
    const void* left_values, std::int32_t left_value_kind,
    double left_lo, double left_hi, std::int32_t left_similarity,
    const void* right_values, std::int32_t right_value_kind,
    double right_lo, double right_hi, std::int32_t right_similarity,
    std::int32_t self_id, std::int32_t* output_ids,
    float* output_squared_l2, std::uint64_t* audit,
    std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kFusedAuditSlots) {
    return kFusedBadArgument;
  }
  fused_reset_audit(audit, mode);
  if (abi_version != kFusedAbiVersion) {
    return kFusedBadAbi;
  }
  if ((mode != kFusedModeRange && mode != kFusedModeDnf2) ||
      base == nullptr || query == nullptr || output_ids == nullptr ||
      output_squared_l2 == nullptr || base_rows == 0 ||
      base_rows > 0x7fffffffULL || left_rows > base_rows ||
      right_rows > base_rows || !fused_valid_pointer(left, left_rows) ||
      !fused_valid_id(self_id, base_rows) ||
      (mode == kFusedModeRange && right_rows != 0) ||
      (mode == kFusedModeDnf2 &&
       !fused_valid_pointer(right, right_rows))) {
    return kFusedBadArgument;
  }
  if (!fused_valid_interval(left_values, left_value_kind, left_lo,
                            left_hi, left_similarity) ||
      (mode == kFusedModeDnf2 &&
       !fused_valid_interval(right_values, right_value_kind, right_lo,
                             right_hi, right_similarity))) {
    return kFusedBadInterval;
  }

  StreamingRetainedTopK topk(base, query, audit);
  std::uint64_t left_index = 0;
  std::uint64_t right_index = 0;
  std::int32_t left_previous = -1;
  std::int32_t right_previous = -1;
  while (left_index < left_rows ||
         (mode == kFusedModeDnf2 && right_index < right_rows)) {
    std::int32_t left_id = std::numeric_limits<std::int32_t>::max();
    std::int32_t right_id = std::numeric_limits<std::int32_t>::max();
    if (left_index < left_rows) {
      left_id = left[left_index];
      if (!fused_valid_id(left_id, base_rows) ||
          (left_index != 0 && left_id <= left_previous)) {
        return kFusedBadPosting;
      }
    }
    if (mode == kFusedModeDnf2 && right_index < right_rows) {
      right_id = right[right_index];
      if (!fused_valid_id(right_id, base_rows) ||
          (right_index != 0 && right_id <= right_previous)) {
        return kFusedBadPosting;
      }
    }
    const std::int32_t id = left_id < right_id ? left_id : right_id;
    const bool has_left = left_index < left_rows && left_id == id;
    const bool has_right = mode == kFusedModeDnf2 &&
                           right_index < right_rows && right_id == id;
    bool left_admitted = false;
    bool right_admitted = false;
    if (has_left) {
      left_previous = left_id;
      ++left_index;
      ++audit[kFusedAuditLeftVisited];
      ++audit[kFusedAuditPredicateChecks];
      left_admitted = fused_value_in_interval(
          left_values, left_value_kind, id, left_lo, left_hi,
          left_similarity != 0);
      audit[kFusedAuditClauseAdmissions] += left_admitted;
    }
    if (has_right) {
      right_previous = right_id;
      ++right_index;
      ++audit[kFusedAuditRightVisited];
      ++audit[kFusedAuditPredicateChecks];
      right_admitted = fused_value_in_interval(
          right_values, right_value_kind, id, right_lo, right_hi,
          right_similarity != 0);
      audit[kFusedAuditClauseAdmissions] += right_admitted;
    }
    if (left_admitted && right_admitted) {
      ++audit[kFusedAuditBothClausesAdmitted];
    }
    if (left_admitted || right_admitted) {
      ++audit[kFusedAuditUniqueBeforeSelf];
      if (id == self_id) {
        ++audit[kFusedAuditRemovedSelf];
      } else {
        topk.append(id);
        if (topk.status() != kFusedOk) {
          return topk.status();
        }
      }
    }
  }
  if (audit[kFusedAuditRemovedSelf] != 1) {
    return kFusedMissingSelf;
  }
  if (audit[kFusedAuditUniqueBeforeSelf] < kTopK + 1) {
    return kFusedSupportBelowK;
  }
  const std::int32_t status = topk.finish(output_ids, output_squared_l2);
  if (status != kFusedOk) {
    return status;
  }
  if (audit[kFusedAuditDistanceRows] + 1 !=
      audit[kFusedAuditUniqueBeforeSelf]) {
    return kFusedBadArgument;
  }
  return kFusedOk;
}
