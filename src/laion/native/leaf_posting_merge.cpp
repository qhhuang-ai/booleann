#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint32_t kAbiVersion = 1;
constexpr std::uint64_t kAuditSlots = 8;
constexpr std::int32_t kModeEquality = 0;
constexpr std::int32_t kModeConjunction = 1;
constexpr std::int32_t kModeRange = 2;
constexpr std::int32_t kModeDnf2 = 3;
constexpr std::int32_t kValueNone = 0;
constexpr std::int32_t kValueFloat64 = 1;
constexpr std::int32_t kValueInt64 = 2;

enum Status : std::int32_t {
  kOk = 0,
  kBadAbi = -1,
  kBadArgument = -2,
  kBadPosting = -3,
  kBadInterval = -4,
  kMissingSelf = -5,
  kOverflow = -6,
  kBadOutput = -7,
};

enum Audit : std::uint64_t {
  kAuditAbi = 0,
  kAuditOutputRows = 1,
  kAuditRemovedSelf = 2,
  kAuditLeftVisited = 3,
  kAuditRightVisited = 4,
  kAuditUniqueCandidates = 5,
  kAuditPredicateChecks = 6,
  kAuditOverflowRequiredMinimum = 7,
};

inline bool valid_pointer(const void* pointer, std::uint64_t rows) noexcept {
  return rows == 0 || pointer != nullptr;
}

inline bool valid_id(std::int32_t id, std::int32_t universe) noexcept {
  return id >= 0 && id < universe;
}

inline bool value_in_interval(const void* values, std::int32_t kind,
                              std::int32_t id, double lo, double hi,
                              bool similarity) noexcept {
  double value = std::numeric_limits<double>::quiet_NaN();
  if (kind == kValueFloat64) {
    value = static_cast<const double*>(values)[id];
  } else if (kind == kValueInt64) {
    value = static_cast<double>(static_cast<const std::int64_t*>(values)[id]);
  } else {
    return false;
  }
  if (!std::isfinite(value) || value < lo || !(value < hi)) {
    return false;
  }
  return similarity ? (value >= 0.0 && value <= 1.0) : (value > 0.0);
}

inline bool valid_interval(const void* values, std::int32_t kind, double lo,
                           double hi, std::int32_t similarity) noexcept {
  return values != nullptr && (kind == kValueFloat64 || kind == kValueInt64) &&
         (similarity == 0 || similarity == 1) && lo < hi;
}

inline void reset_audit(std::uint64_t* audit) noexcept {
  for (std::uint64_t index = 0; index < kAuditSlots; ++index) {
    audit[index] = 0;
  }
  audit[kAuditAbi] = kAbiVersion;
}

inline std::int32_t emit_or_remove(std::int32_t id, std::int32_t self_id,
                                   std::int32_t* output,
                                   std::uint64_t output_capacity,
                                   std::uint64_t* audit) noexcept {
  ++audit[kAuditUniqueCandidates];
  if (id == self_id) {
    ++audit[kAuditRemovedSelf];
    return kOk;
  }
  const std::uint64_t position = audit[kAuditOutputRows];
  if (position >= output_capacity) {
    audit[kAuditOverflowRequiredMinimum] = position + 1;
    return kOverflow;
  }
  if (position > 0 && output[position - 1] >= id) {
    return kBadOutput;
  }
  output[position] = id;
  audit[kAuditOutputRows] = position + 1;
  return kOk;
}

inline std::int32_t checked_id(const std::int32_t* posting,
                               std::uint64_t index, std::int32_t universe,
                               std::int32_t previous,
                               bool has_previous) noexcept {
  const std::int32_t id = posting[index];
  if (!valid_id(id, universe) || (has_previous && id <= previous)) {
    return -1;
  }
  return id;
}

}  // namespace

extern "C" std::uint32_t laion1m_leaf_merge_abi_version_v1() noexcept {
  return kAbiVersion;
}

extern "C" std::uint64_t laion1m_leaf_merge_audit_slots_v1() noexcept {
  return kAuditSlots;
}

extern "C" std::int32_t laion1m_leaf_validate_posting_v1(
    const std::int32_t* posting, std::uint64_t rows,
    std::int32_t universe) noexcept {
  if (universe <= 0 || !valid_pointer(posting, rows)) {
    return kBadArgument;
  }
  std::int32_t previous = -1;
  for (std::uint64_t index = 0; index < rows; ++index) {
    const std::int32_t id = posting[index];
    if (!valid_id(id, universe) || (index > 0 && id <= previous)) {
      return kBadPosting;
    }
    previous = id;
  }
  return kOk;
}

extern "C" std::int32_t laion1m_leaf_sorted_posting_merge_v1(
    std::uint32_t abi_version, std::int32_t mode,
    const std::int32_t* left, std::uint64_t left_rows,
    const std::int32_t* right, std::uint64_t right_rows,
    const void* left_values, std::int32_t left_value_kind,
    double left_lo, double left_hi, std::int32_t left_similarity,
    const void* right_values, std::int32_t right_value_kind,
    double right_lo, double right_hi, std::int32_t right_similarity,
    std::int32_t self_id, std::int32_t universe,
    std::int32_t* output, std::uint64_t output_capacity,
    std::uint64_t* audit, std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kAuditSlots) {
    return kBadArgument;
  }
  reset_audit(audit);
  if (abi_version != kAbiVersion) {
    return kBadAbi;
  }
  if (mode < kModeEquality || mode > kModeDnf2 || universe <= 0 ||
      !valid_id(self_id, universe) || !valid_pointer(left, left_rows) ||
      output == nullptr || output_capacity == 0) {
    return kBadArgument;
  }
  if ((mode == kModeConjunction || mode == kModeDnf2) &&
      !valid_pointer(right, right_rows)) {
    return kBadArgument;
  }
  if ((mode == kModeRange || mode == kModeDnf2) &&
      !valid_interval(left_values, left_value_kind, left_lo, left_hi,
                      left_similarity)) {
    return kBadInterval;
  }
  if (mode == kModeDnf2 &&
      !valid_interval(right_values, right_value_kind, right_lo, right_hi,
                      right_similarity)) {
    return kBadInterval;
  }

  std::int32_t status = kOk;
  if (mode == kModeEquality || mode == kModeRange) {
    std::int32_t previous = -1;
    for (std::uint64_t index = 0; index < left_rows; ++index) {
      const std::int32_t id = checked_id(left, index, universe, previous,
                                         index > 0);
      if (id < 0) {
        return kBadPosting;
      }
      previous = id;
      ++audit[kAuditLeftVisited];
      bool keep = true;
      if (mode == kModeRange) {
        ++audit[kAuditPredicateChecks];
        keep = value_in_interval(left_values, left_value_kind, id, left_lo,
                                 left_hi, left_similarity != 0);
      }
      if (keep) {
        status = emit_or_remove(id, self_id, output, output_capacity, audit);
        if (status != kOk) {
          return status;
        }
      }
    }
  } else if (mode == kModeConjunction) {
    std::uint64_t left_index = 0;
    std::uint64_t right_index = 0;
    std::int32_t left_previous = -1;
    std::int32_t right_previous = -1;
    while (left_index < left_rows && right_index < right_rows) {
      const std::int32_t left_id = checked_id(
          left, left_index, universe, left_previous, left_index > 0);
      const std::int32_t right_id = checked_id(
          right, right_index, universe, right_previous, right_index > 0);
      if (left_id < 0 || right_id < 0) {
        return kBadPosting;
      }
      if (left_id < right_id) {
        left_previous = left_id;
        ++left_index;
        ++audit[kAuditLeftVisited];
      } else if (right_id < left_id) {
        right_previous = right_id;
        ++right_index;
        ++audit[kAuditRightVisited];
      } else {
        left_previous = left_id;
        right_previous = right_id;
        ++left_index;
        ++right_index;
        ++audit[kAuditLeftVisited];
        ++audit[kAuditRightVisited];
        status = emit_or_remove(left_id, self_id, output, output_capacity,
                                audit);
        if (status != kOk) {
          return status;
        }
      }
    }
  } else {
    std::uint64_t left_index = 0;
    std::uint64_t right_index = 0;
    std::int32_t left_previous = -1;
    std::int32_t right_previous = -1;
    while (left_index < left_rows || right_index < right_rows) {
      std::int32_t left_id = std::numeric_limits<std::int32_t>::max();
      std::int32_t right_id = std::numeric_limits<std::int32_t>::max();
      if (left_index < left_rows) {
        left_id = checked_id(left, left_index, universe, left_previous,
                             left_index > 0);
        if (left_id < 0) {
          return kBadPosting;
        }
      }
      if (right_index < right_rows) {
        right_id = checked_id(right, right_index, universe, right_previous,
                              right_index > 0);
        if (right_id < 0) {
          return kBadPosting;
        }
      }
      const std::int32_t id = left_id < right_id ? left_id : right_id;
      const bool has_left = left_index < left_rows && left_id == id;
      const bool has_right = right_index < right_rows && right_id == id;
      bool keep = false;
      if (has_left) {
        left_previous = left_id;
        ++left_index;
        ++audit[kAuditLeftVisited];
        ++audit[kAuditPredicateChecks];
        keep = value_in_interval(left_values, left_value_kind, id, left_lo,
                                 left_hi, left_similarity != 0);
      }
      if (has_right) {
        right_previous = right_id;
        ++right_index;
        ++audit[kAuditRightVisited];
        ++audit[kAuditPredicateChecks];
        keep = value_in_interval(right_values, right_value_kind, id, right_lo,
                                 right_hi, right_similarity != 0) || keep;
      }
      if (keep) {
        status = emit_or_remove(id, self_id, output, output_capacity, audit);
        if (status != kOk) {
          return status;
        }
      }
    }
  }

  if (audit[kAuditRemovedSelf] != 1) {
    return kMissingSelf;
  }
  return kOk;
}
