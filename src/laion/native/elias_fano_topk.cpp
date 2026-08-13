// Project-owned checked Elias--Fano support codec and streaming retained
// top-k scanner for LAION1M.
//
// Each object represents one query-independent, strictly increasing predicate
// support over [0, 1,000,000).  Its low and high streams are independently
// rounded to 64-bit boundaries.  The scanner decodes IDs sequentially, skips
// the online query self ID if present, and feeds the canonical retained
// NumPy-2.0.2-compatible float32 scorer without constructing an int32 support
// array.  Nonzero scan statuses are explicit requests for the immutable LEAF
// fallback; result arrays are committed only after complete validation.

#include "exact_topk_avx2.cpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>

struct Laion1mEliasFanoDirectoryV1 {
  std::uint64_t low_offset;
  std::uint64_t high_offset;
  std::uint32_t rows;
  std::uint8_t lower_bits;
  std::uint8_t version;
  std::uint16_t reserved;
};

static_assert(sizeof(Laion1mEliasFanoDirectoryV1) == 24,
              "Elias--Fano directory ABI must remain 24 bytes");
static_assert(alignof(Laion1mEliasFanoDirectoryV1) == 8,
              "Elias--Fano directory alignment drifted");

namespace {

constexpr std::uint32_t kEfAbiVersion = 1;
constexpr std::uint8_t kEfObjectVersion = 1;
constexpr std::uint64_t kEfUniverse = 1000000;
constexpr std::uint64_t kEfAuditSlots = 16;
constexpr std::size_t kEfPendingCapacity = 32;
constexpr std::size_t kEfPendingMask = kEfPendingCapacity - 1U;

static_assert((kEfPendingCapacity & kEfPendingMask) == 0,
              "pending ring must be a power of two");
static_assert(kEfPendingCapacity >=
                  kRetainedInterleaveRows + kPrefetchDistance,
              "pending ring must cover retained scorer lookahead");

enum EfStatus : std::int32_t {
  kEfOk = 0,
  kEfBadAbi = -1,
  kEfBadArgument = -2,
  kEfBadInputOrder = -3,
  kEfCapacity = -4,
  kEfCorruptDirectory = -5,
  kEfCorruptPayload = -6,
  kEfSupportBelowK = -7,
  kEfNonfiniteDistance = -8,
  kEfFpEnvironment = -9,
  kEfUnsupportedCpu = -10,
};

enum EfAudit : std::uint64_t {
  kEfAuditAbi = 0,
  kEfAuditStoredRows = 1,
  kEfAuditLowerBits = 2,
  kEfAuditLowBytes = 3,
  kEfAuditHighBytes = 4,
  kEfAuditDecodedRows = 5,
  kEfAuditRowsAfterSelf = 6,
  kEfAuditSelfOccurrences = 7,
  kEfAuditScoredRows = 8,
  kEfAuditInterleavedBatches = 9,
  kEfAuditCutoffTie = 10,
  kEfAuditPayloadBytes = 11,
  kEfAuditHighOneBits = 12,
  kEfAuditFallbackStatus = 13,
  kEfAuditCodecInputRows = 14,
  kEfAuditCodecUniqueRows = 15,
};

struct EfAccount {
  std::uint8_t lower_bits;
  std::uint64_t low_bits;
  std::uint64_t high_bits;
  std::uint64_t low_bytes;
  std::uint64_t high_bytes;
  std::uint64_t payload_bytes;
};

inline bool checked_add(std::uint64_t left, std::uint64_t right,
                        std::uint64_t& output) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) return false;
  output = left + right;
  return true;
}

inline bool checked_mul(std::uint64_t left, std::uint64_t right,
                        std::uint64_t& output) noexcept {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) return false;
  output = left * right;
  return true;
}

inline bool align_bits64(std::uint64_t bits,
                         std::uint64_t& bytes) noexcept {
  std::uint64_t rounded = 0;
  if (!checked_add(bits, 63U, rounded)) return false;
  bytes = (rounded / 64U) * 8U;
  return true;
}

inline std::uint8_t lower_bits_for(std::uint64_t rows) noexcept {
  const std::uint64_t quotient = kEfUniverse / rows;
  std::uint8_t result = 0;
  std::uint64_t value = quotient;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

inline bool account_for(std::uint64_t rows, EfAccount& account) noexcept {
  if (rows == 0 || rows > kEfUniverse) return false;
  account.lower_bits = lower_bits_for(rows);
  if (!checked_mul(rows, account.lower_bits, account.low_bits)) return false;
  account.high_bits = (kEfUniverse >> account.lower_bits) + rows;
  if (!align_bits64(account.low_bits, account.low_bytes) ||
      !align_bits64(account.high_bits, account.high_bytes) ||
      !checked_add(account.low_bytes, account.high_bytes,
                   account.payload_bytes)) {
    return false;
  }
  return true;
}

inline bool valid_id(std::int32_t id) noexcept {
  return id >= 0 && static_cast<std::uint64_t>(id) < kEfUniverse;
}

inline bool ranges_overlap(const void* left, std::uint64_t left_bytes,
                           const void* right,
                           std::uint64_t right_bytes) noexcept {
  if (left_bytes == 0 || right_bytes == 0) return false;
  if (left == nullptr || right == nullptr ||
      left_bytes > std::numeric_limits<std::uintptr_t>::max() ||
      right_bytes > std::numeric_limits<std::uintptr_t>::max()) return true;
  const std::uintptr_t left_begin =
      reinterpret_cast<std::uintptr_t>(left);
  const std::uintptr_t right_begin =
      reinterpret_cast<std::uintptr_t>(right);
  if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
      right_begin >
          std::numeric_limits<std::uintptr_t>::max() - right_bytes) return true;
  const std::uintptr_t left_end =
      left_begin + static_cast<std::uintptr_t>(left_bytes);
  const std::uintptr_t right_end =
      right_begin + static_cast<std::uintptr_t>(right_bytes);
  return left_begin < right_end && right_begin < left_end;
}

inline void reset_audit(std::uint64_t* audit) noexcept {
  for (std::uint64_t index = 0; index < kEfAuditSlots; ++index) {
    audit[index] = 0;
  }
  audit[kEfAuditAbi] = kEfAbiVersion;
}

inline std::int32_t fail(std::uint64_t* audit,
                         std::int32_t status) noexcept {
  if (audit != nullptr) {
    audit[kEfAuditFallbackStatus] =
        static_cast<std::uint64_t>(-static_cast<std::int64_t>(status));
  }
  return status;
}

inline std::uint32_t fp_environment_status() noexcept {
  constexpr std::uint32_t kFtz = 1U << 15U;
  constexpr std::uint32_t kDaz = 1U << 6U;
  constexpr std::uint32_t kRounding = 3U << 13U;
  const std::uint32_t mxcsr = _mm_getcsr();
  if (std::fegetround() != FE_TONEAREST) return 1;
  if ((mxcsr & kRounding) != 0U) return 2;
  if ((mxcsr & (kFtz | kDaz)) != 0U) return 3;
  return 0;
}

inline std::uint64_t load_word(const std::uint8_t* owner,
                               std::uint64_t offset) noexcept {
  std::uint64_t value = 0;
  std::memcpy(&value, owner + static_cast<std::size_t>(offset),
              sizeof(value));
  return value;
}

inline void store_word(std::uint8_t* owner, std::uint64_t offset,
                       std::uint64_t value) noexcept {
  std::memcpy(owner + static_cast<std::size_t>(offset), &value,
              sizeof(value));
}

inline std::int32_t validate_directory(
    const Laion1mEliasFanoDirectoryV1& directory,
    std::uint64_t owner_bytes, EfAccount& account) noexcept {
  if (directory.version != kEfObjectVersion || directory.reserved != 0 ||
      directory.rows == 0 || directory.rows > kEfUniverse ||
      (directory.low_offset & 7U) != 0 ||
      (directory.high_offset & 7U) != 0 ||
      owner_bytes > std::numeric_limits<std::size_t>::max() ||
      !account_for(directory.rows, account) ||
      directory.lower_bits != account.lower_bits) {
    return kEfCorruptDirectory;
  }
  std::uint64_t expected_high = 0;
  std::uint64_t terminal = 0;
  if (!checked_add(directory.low_offset, account.low_bytes, expected_high) ||
      directory.high_offset != expected_high ||
      !checked_add(directory.high_offset, account.high_bytes, terminal) ||
      terminal > owner_bytes) {
    return kEfCorruptDirectory;
  }
  return kEfOk;
}

class EfDecoder {
 public:
  EfDecoder(const std::uint8_t* owner,
            const Laion1mEliasFanoDirectoryV1& directory,
            const EfAccount& account) noexcept
      : owner_(owner), directory_(directory), account_(account) {}

  std::int32_t next(std::int32_t& output) noexcept {
    if (index_ >= directory_.rows) return kEfCorruptPayload;
    std::uint64_t position = 0;
    if (!next_high_one(position) || position < index_) {
      return kEfCorruptPayload;
    }
    const std::uint64_t high = position - index_;
    if (high > (std::numeric_limits<std::uint64_t>::max() >>
                account_.lower_bits)) {
      return kEfCorruptPayload;
    }
    const std::uint64_t low = read_low(index_);
    const std::uint64_t decoded =
        (high << account_.lower_bits) | low;
    if (decoded >= kEfUniverse ||
        (index_ != 0 && decoded <= previous_)) {
      return kEfCorruptPayload;
    }
    previous_ = decoded;
    output = static_cast<std::int32_t>(decoded);
    ++index_;
    return kEfOk;
  }

  std::int32_t finish() noexcept {
    if (index_ != directory_.rows) return kEfCorruptPayload;
    if (high_remaining_ != 0) return kEfCorruptPayload;
    while (high_word_index_ < account_.high_bytes / 8U) {
      if (load_word(owner_, directory_.high_offset +
                                high_word_index_ * 8U) != 0) {
        return kEfCorruptPayload;
      }
      ++high_word_index_;
    }
    if (account_.low_bits != 0 && (account_.low_bits & 63U) != 0) {
      const std::uint64_t used = account_.low_bits & 63U;
      const std::uint64_t last = load_word(
          owner_, directory_.low_offset + account_.low_bytes - 8U);
      const std::uint64_t padding_mask = ~((std::uint64_t{1} << used) - 1U);
      if ((last & padding_mask) != 0) return kEfCorruptPayload;
    }
    return kEfOk;
  }

  std::uint64_t high_one_bits() const noexcept { return high_one_bits_; }

 private:
  bool next_high_one(std::uint64_t& position) noexcept {
    const std::uint64_t words = account_.high_bytes / 8U;
    while (high_remaining_ == 0) {
      if (high_word_index_ >= words) return false;
      high_remaining_ = load_word(
          owner_, directory_.high_offset + high_word_index_ * 8U);
      high_base_ = high_word_index_ * 64U;
      ++high_word_index_;
    }
#if defined(__GNUC__) || defined(__clang__)
    const unsigned bit =
        static_cast<unsigned>(__builtin_ctzll(high_remaining_));
#else
    unsigned bit = 0;
    while (((high_remaining_ >> bit) & 1U) == 0) ++bit;
#endif
    position = high_base_ + bit;
    high_remaining_ &= high_remaining_ - 1U;
    ++high_one_bits_;
    return position < account_.high_bits;
  }

  std::uint64_t read_low(std::uint64_t index) const noexcept {
    if (account_.lower_bits == 0) return 0;
    const std::uint64_t bit = index * account_.lower_bits;
    const std::uint64_t word_index = bit / 64U;
    const unsigned shift = static_cast<unsigned>(bit & 63U);
    std::uint64_t value = load_word(
        owner_, directory_.low_offset + word_index * 8U) >> shift;
    if (shift + account_.lower_bits > 64U) {
      value |= load_word(owner_, directory_.low_offset +
                                     (word_index + 1U) * 8U)
               << (64U - shift);
    }
    return value & ((std::uint64_t{1} << account_.lower_bits) - 1U);
  }

  const std::uint8_t* owner_;
  const Laion1mEliasFanoDirectoryV1& directory_;
  const EfAccount& account_;
  std::uint64_t index_ = 0;
  std::uint64_t previous_ = 0;
  std::uint64_t high_word_index_ = 0;
  std::uint64_t high_remaining_ = 0;
  std::uint64_t high_base_ = 0;
  std::uint64_t high_one_bits_ = 0;
};

template <typename Next>
std::int32_t encode_validated(
    Next&& next, std::uint64_t rows, std::uint8_t* owner,
    std::uint64_t owner_bytes, std::uint64_t payload_offset,
    Laion1mEliasFanoDirectoryV1* output_directory,
    std::uint64_t* audit) noexcept {
  EfAccount account{};
  if (!account_for(rows, account) || owner == nullptr ||
      output_directory == nullptr || (payload_offset & 7U) != 0 ||
      owner_bytes > std::numeric_limits<std::size_t>::max()) {
    return fail(audit, kEfBadArgument);
  }
  std::uint64_t terminal = 0;
  if (!checked_add(payload_offset, account.payload_bytes, terminal) ||
      terminal > owner_bytes ||
      ranges_overlap(output_directory, sizeof(*output_directory),
                     owner + static_cast<std::size_t>(payload_offset),
                     account.payload_bytes)) {
    return fail(audit, kEfCapacity);
  }
  std::memset(owner + static_cast<std::size_t>(payload_offset), 0,
              static_cast<std::size_t>(account.payload_bytes));
  const std::uint64_t low_offset = payload_offset;
  const std::uint64_t high_offset = payload_offset + account.low_bytes;
  const std::uint64_t low_mask = account.lower_bits == 0
      ? 0
      : (std::uint64_t{1} << account.lower_bits) - 1U;
  for (std::uint64_t index = 0; index < rows; ++index) {
    const std::uint64_t id = static_cast<std::uint64_t>(next());
    if (account.lower_bits != 0) {
      const std::uint64_t bit = index * account.lower_bits;
      const std::uint64_t word_index = bit / 64U;
      const unsigned shift = static_cast<unsigned>(bit & 63U);
      std::uint64_t word = load_word(owner, low_offset + word_index * 8U);
      word |= (id & low_mask) << shift;
      store_word(owner, low_offset + word_index * 8U, word);
      if (shift + account.lower_bits > 64U) {
        std::uint64_t next_word = load_word(
            owner, low_offset + (word_index + 1U) * 8U);
        next_word |= (id & low_mask) >> (64U - shift);
        store_word(owner, low_offset + (word_index + 1U) * 8U,
                   next_word);
      }
    }
    const std::uint64_t high_position =
        (id >> account.lower_bits) + index;
    const std::uint64_t word_index = high_position / 64U;
    const unsigned shift = static_cast<unsigned>(high_position & 63U);
    std::uint64_t word = load_word(owner, high_offset + word_index * 8U);
    word |= std::uint64_t{1} << shift;
    store_word(owner, high_offset + word_index * 8U, word);
  }
  const Laion1mEliasFanoDirectoryV1 directory{
      low_offset, high_offset, static_cast<std::uint32_t>(rows),
      account.lower_bits, kEfObjectVersion, 0};
  *output_directory = directory;
  if (audit != nullptr) {
    audit[kEfAuditStoredRows] = rows;
    audit[kEfAuditLowerBits] = account.lower_bits;
    audit[kEfAuditLowBytes] = account.low_bytes;
    audit[kEfAuditHighBytes] = account.high_bytes;
    audit[kEfAuditPayloadBytes] = account.payload_bytes;
    audit[kEfAuditCodecUniqueRows] = rows;
  }
  return kEfOk;
}

class StreamingEfTopK {
 public:
  StreamingEfTopK(const float* base, const float* query,
                  std::uint64_t* audit) noexcept
      : base_(base), query_(query), audit_(audit) {}

  std::int32_t append(std::int32_t id) noexcept {
    pending_[(pending_begin_ + pending_count_) & kEfPendingMask] = id;
    ++pending_count_;
    if (pending_count_ == kEfPendingCapacity) return score_front8();
    return kEfOk;
  }

  std::int32_t finish(std::int32_t output_ids[kTopK],
                      float output_distances[kTopK]) noexcept {
    while (pending_count_ >= kRetainedInterleaveRows) {
      const std::int32_t status = score_front8();
      if (status != kEfOk) return status;
    }
    while (pending_count_ != 0) {
      const std::int32_t id = pending_[pending_begin_];
      pending_begin_ = (pending_begin_ + 1U) & kEfPendingMask;
      --pending_count_;
      const float distance = squared_l2_f32_512_retained(
          query_, base_ + static_cast<std::size_t>(id) * kDimension);
      if (!std::isfinite(distance)) return kEfNonfiniteDistance;
      retain(Candidate{distance, id});
      ++audit_[kEfAuditScoredRows];
    }
    if (best_size_ < kTopK) return kEfSupportBelowK;
    std::sort(best_, best_ + best_size_, candidate_less);
    if (best_size_ > kTopK &&
        best_[kTopK - 1U].distance == best_[kTopK].distance) {
      audit_[kEfAuditCutoffTie] = 1;
    }
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      output_ids[rank] = best_[rank].id;
      output_distances[rank] = best_[rank].distance;
    }
    return kEfOk;
  }

 private:
  void retain(const Candidate& candidate) noexcept {
    if (best_size_ < kRetainedCutoff) {
      best_[best_size_++] = candidate;
      std::push_heap(best_, best_ + best_size_, candidate_less);
    } else if (candidate_less(candidate, best_[0])) {
      std::pop_heap(best_, best_ + best_size_, candidate_less);
      best_[best_size_ - 1U] = candidate;
      std::push_heap(best_, best_ + best_size_, candidate_less);
    }
  }

  std::int32_t score_front8() noexcept {
    const float* vectors[kRetainedInterleaveRows];
    const float* future[kRetainedInterleaveRows];
    for (std::size_t lane = 0; lane < kRetainedInterleaveRows; ++lane) {
      const std::int32_t id =
          pending_[(pending_begin_ + lane) & kEfPendingMask];
      vectors[lane] = base_ + static_cast<std::size_t>(id) * kDimension;
      const std::size_t future_position = lane + kPrefetchDistance;
      if (future_position < pending_count_) {
        const std::int32_t future_id = pending_[
            (pending_begin_ + future_position) & kEfPendingMask];
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
      if (!std::isfinite(distances[lane])) return kEfNonfiniteDistance;
      const std::int32_t id =
          pending_[(pending_begin_ + lane) & kEfPendingMask];
      retain(Candidate{distances[lane], id});
    }
    pending_begin_ =
        (pending_begin_ + kRetainedInterleaveRows) & kEfPendingMask;
    pending_count_ -= kRetainedInterleaveRows;
    audit_[kEfAuditScoredRows] += kRetainedInterleaveRows;
    ++audit_[kEfAuditInterleavedBatches];
    return kEfOk;
  }

  const float* base_;
  const float* query_;
  std::uint64_t* audit_;
  Candidate best_[kRetainedCutoff];
  std::size_t best_size_ = 0;
  std::int32_t pending_[kEfPendingCapacity];
  std::size_t pending_begin_ = 0;
  std::size_t pending_count_ = 0;
};

}  // namespace

extern "C" {

std::uint32_t laion1m_elias_fano_v1_abi_version() noexcept {
  return kEfAbiVersion;
}

std::uint64_t laion1m_elias_fano_v1_universe() noexcept {
  return kEfUniverse;
}

std::uint32_t laion1m_elias_fano_v1_directory_bytes() noexcept {
  return sizeof(Laion1mEliasFanoDirectoryV1);
}

std::uint32_t laion1m_elias_fano_v1_dimension() noexcept {
  return static_cast<std::uint32_t>(kDimension);
}

std::uint32_t laion1m_elias_fano_v1_k() noexcept {
  return static_cast<std::uint32_t>(kTopK);
}

std::uint64_t laion1m_elias_fano_v1_audit_slots() noexcept {
  return kEfAuditSlots;
}

std::uint32_t laion1m_elias_fano_v1_fp_environment_status() noexcept {
  return fp_environment_status();
}

std::uint32_t laion1m_elias_fano_v1_get_mxcsr() noexcept {
  return _mm_getcsr();
}

void laion1m_elias_fano_v1_set_mxcsr_for_selftest(
    std::uint32_t value) noexcept {
  _mm_setcsr(value);
}

int laion1m_elias_fano_v1_get_rounding_mode() noexcept {
  return std::fegetround();
}

int laion1m_elias_fano_v1_set_rounding_mode_for_selftest(
    int mode) noexcept {
  return std::fesetround(mode);
}

int laion1m_elias_fano_v1_downward_rounding_mode() noexcept {
  return FE_DOWNWARD;
}

std::int32_t laion1m_elias_fano_v1_account(
    std::uint32_t abi_version, std::uint64_t rows,
    std::uint8_t* output_lower_bits, std::uint64_t* output_low_bytes,
    std::uint64_t* output_high_bytes,
    std::uint64_t* output_payload_bytes) noexcept {
  if (abi_version != kEfAbiVersion) return kEfBadAbi;
  if (output_lower_bits == nullptr || output_low_bytes == nullptr ||
      output_high_bytes == nullptr || output_payload_bytes == nullptr) {
    return kEfBadArgument;
  }
  EfAccount account{};
  if (!account_for(rows, account)) return kEfBadArgument;
  *output_lower_bits = account.lower_bits;
  *output_low_bytes = account.low_bytes;
  *output_high_bytes = account.high_bytes;
  *output_payload_bytes = account.payload_bytes;
  return kEfOk;
}

std::int32_t laion1m_elias_fano_v1_encode_sorted(
    std::uint32_t abi_version, const std::int32_t* ids,
    std::uint64_t rows, std::uint8_t* owner, std::uint64_t owner_bytes,
    std::uint64_t payload_offset,
    Laion1mEliasFanoDirectoryV1* output_directory,
    std::uint64_t* audit, std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kEfAuditSlots) return kEfBadArgument;
  reset_audit(audit);
  if (abi_version != kEfAbiVersion) return fail(audit, kEfBadAbi);
  if (ids == nullptr || rows == 0 || rows > kEfUniverse) {
    return fail(audit, kEfBadArgument);
  }
  std::int32_t previous = -1;
  for (std::uint64_t index = 0; index < rows; ++index) {
    const std::int32_t id = ids[index];
    if (!valid_id(id) || (index != 0 && id <= previous)) {
      return fail(audit, kEfBadInputOrder);
    }
    previous = id;
  }
  EfAccount account{};
  if (!account_for(rows, account)) return fail(audit, kEfBadArgument);
  std::uint64_t terminal = 0;
  if (owner == nullptr || !checked_add(payload_offset, account.payload_bytes,
                                        terminal) ||
      terminal > owner_bytes ||
      ranges_overlap(ids, rows * sizeof(std::int32_t),
                     owner + static_cast<std::size_t>(payload_offset),
                     account.payload_bytes)) {
    return fail(audit, kEfCapacity);
  }
  audit[kEfAuditCodecInputRows] = rows;
  std::uint64_t index = 0;
  return encode_validated(
      [&]() noexcept { return ids[index++]; }, rows, owner, owner_bytes,
      payload_offset, output_directory, audit);
}

std::int32_t laion1m_elias_fano_v1_encode_union(
    std::uint32_t abi_version, const std::int32_t* left,
    std::uint64_t left_rows, const std::int32_t* right,
    std::uint64_t right_rows, std::uint8_t* owner,
    std::uint64_t owner_bytes, std::uint64_t payload_offset,
    Laion1mEliasFanoDirectoryV1* output_directory,
    std::uint64_t* audit, std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kEfAuditSlots) return kEfBadArgument;
  reset_audit(audit);
  if (abi_version != kEfAbiVersion) return fail(audit, kEfBadAbi);
  if (left == nullptr || right == nullptr || left_rows == 0 ||
      right_rows == 0 || left_rows > kEfUniverse ||
      right_rows > kEfUniverse) {
    return fail(audit, kEfBadArgument);
  }
  auto validate = [](const std::int32_t* values,
                     std::uint64_t rows) noexcept {
    std::int32_t previous = -1;
    for (std::uint64_t index = 0; index < rows; ++index) {
      if (!valid_id(values[index]) ||
          (index != 0 && values[index] <= previous)) return false;
      previous = values[index];
    }
    return true;
  };
  if (!validate(left, left_rows) || !validate(right, right_rows)) {
    return fail(audit, kEfBadInputOrder);
  }
  std::uint64_t left_index = 0;
  std::uint64_t right_index = 0;
  std::uint64_t unique = 0;
  while (left_index < left_rows || right_index < right_rows) {
    const std::int32_t left_id = left_index < left_rows
        ? left[left_index]
        : std::numeric_limits<std::int32_t>::max();
    const std::int32_t right_id = right_index < right_rows
        ? right[right_index]
        : std::numeric_limits<std::int32_t>::max();
    const std::int32_t id = std::min(left_id, right_id);
    if (left_id == id) ++left_index;
    if (right_id == id) ++right_index;
    ++unique;
  }
  if (unique == 0 || unique > kEfUniverse) {
    return fail(audit, kEfBadArgument);
  }
  EfAccount account{};
  if (!account_for(unique, account)) return fail(audit, kEfBadArgument);
  std::uint64_t terminal = 0;
  if (owner == nullptr || !checked_add(payload_offset, account.payload_bytes,
                                        terminal) ||
      terminal > owner_bytes ||
      ranges_overlap(left, left_rows * sizeof(std::int32_t),
                     owner + static_cast<std::size_t>(payload_offset),
                     account.payload_bytes) ||
      ranges_overlap(right, right_rows * sizeof(std::int32_t),
                     owner + static_cast<std::size_t>(payload_offset),
                     account.payload_bytes)) {
    return fail(audit, kEfCapacity);
  }
  audit[kEfAuditCodecInputRows] = left_rows + right_rows;
  left_index = 0;
  right_index = 0;
  return encode_validated(
      [&]() noexcept {
        const std::int32_t left_id = left_index < left_rows
            ? left[left_index]
            : std::numeric_limits<std::int32_t>::max();
        const std::int32_t right_id = right_index < right_rows
            ? right[right_index]
            : std::numeric_limits<std::int32_t>::max();
        const std::int32_t id = std::min(left_id, right_id);
        if (left_id == id) ++left_index;
        if (right_id == id) ++right_index;
        return id;
      },
      unique, owner, owner_bytes, payload_offset, output_directory, audit);
}

std::int32_t laion1m_elias_fano_v1_decode(
    std::uint32_t abi_version,
    const Laion1mEliasFanoDirectoryV1* directory,
    const std::uint8_t* owner, std::uint64_t owner_bytes,
    std::int32_t* output_ids, std::uint64_t output_capacity,
    std::uint64_t* audit, std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kEfAuditSlots) return kEfBadArgument;
  reset_audit(audit);
  if (abi_version != kEfAbiVersion) return fail(audit, kEfBadAbi);
  if (directory == nullptr || owner == nullptr || output_ids == nullptr) {
    return fail(audit, kEfBadArgument);
  }
  EfAccount account{};
  std::int32_t status = validate_directory(*directory, owner_bytes, account);
  if (status != kEfOk) return fail(audit, status);
  audit[kEfAuditStoredRows] = directory->rows;
  audit[kEfAuditLowerBits] = account.lower_bits;
  audit[kEfAuditLowBytes] = account.low_bytes;
  audit[kEfAuditHighBytes] = account.high_bytes;
  audit[kEfAuditPayloadBytes] = account.payload_bytes;
  if (output_capacity < directory->rows) return fail(audit, kEfCapacity);

  // First pass validates the complete object so a late corruption cannot
  // partially overwrite the caller's decode arena.
  EfDecoder validator(owner, *directory, account);
  for (std::uint64_t index = 0; index < directory->rows; ++index) {
    std::int32_t ignored = -1;
    status = validator.next(ignored);
    if (status != kEfOk) return fail(audit, status);
  }
  status = validator.finish();
  if (status != kEfOk) return fail(audit, status);

  EfDecoder decoder(owner, *directory, account);
  for (std::uint64_t index = 0; index < directory->rows; ++index) {
    status = decoder.next(output_ids[index]);
    if (status != kEfOk) return fail(audit, status);
  }
  status = decoder.finish();
  if (status != kEfOk) return fail(audit, status);
  audit[kEfAuditDecodedRows] = directory->rows;
  audit[kEfAuditHighOneBits] = decoder.high_one_bits();
  return kEfOk;
}

std::int32_t laion1m_elias_fano_v1_scan_checksum(
    std::uint32_t abi_version,
    const Laion1mEliasFanoDirectoryV1* directory,
    const std::uint8_t* owner, std::uint64_t owner_bytes,
    std::uint64_t* output_checksum, std::uint64_t* audit,
    std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kEfAuditSlots) return kEfBadArgument;
  reset_audit(audit);
  if (abi_version != kEfAbiVersion) return fail(audit, kEfBadAbi);
  if (directory == nullptr || owner == nullptr || output_checksum == nullptr) {
    return fail(audit, kEfBadArgument);
  }
  EfAccount account{};
  std::int32_t status = validate_directory(*directory, owner_bytes, account);
  if (status != kEfOk) return fail(audit, status);
  audit[kEfAuditStoredRows] = directory->rows;
  audit[kEfAuditLowerBits] = account.lower_bits;
  audit[kEfAuditLowBytes] = account.low_bytes;
  audit[kEfAuditHighBytes] = account.high_bytes;
  audit[kEfAuditPayloadBytes] = account.payload_bytes;
  EfDecoder decoder(owner, *directory, account);
  std::uint64_t checksum = 1469598103934665603ULL;
  for (std::uint64_t index = 0; index < directory->rows; ++index) {
    std::int32_t id = -1;
    status = decoder.next(id);
    if (status != kEfOk) return fail(audit, status);
    checksum ^= static_cast<std::uint32_t>(id);
    checksum *= 1099511628211ULL;
    ++audit[kEfAuditDecodedRows];
  }
  status = decoder.finish();
  if (status != kEfOk) return fail(audit, status);
  audit[kEfAuditHighOneBits] = decoder.high_one_bits();
  *output_checksum = checksum;
  return kEfOk;
}

std::int32_t laion1m_elias_fano_v1_exact_top10(
    std::uint32_t abi_version,
    const Laion1mEliasFanoDirectoryV1* directory,
    const std::uint8_t* owner, std::uint64_t owner_bytes,
    const float* base, std::uint64_t base_rows, const float* query,
    std::int32_t self_id, std::int32_t* output_ids,
    float* output_squared_l2, std::uint64_t* audit,
    std::uint64_t audit_slots) noexcept {
  if (audit == nullptr || audit_slots < kEfAuditSlots) return kEfBadArgument;
  reset_audit(audit);
  if (abi_version != kEfAbiVersion) return fail(audit, kEfBadAbi);
  if (directory == nullptr || owner == nullptr || base == nullptr ||
      query == nullptr || output_ids == nullptr ||
      output_squared_l2 == nullptr || base_rows == 0 ||
      base_rows > kEfUniverse || self_id < 0 ||
      static_cast<std::uint64_t>(self_id) >= base_rows) {
    return fail(audit, kEfBadArgument);
  }
  if (!laion1m_exact_topk_avx2_v1_cpu_supported()) {
    return fail(audit, kEfUnsupportedCpu);
  }
  if (fp_environment_status() != 0) {
    return fail(audit, kEfFpEnvironment);
  }
  EfAccount account{};
  std::int32_t status = validate_directory(*directory, owner_bytes, account);
  if (status != kEfOk) return fail(audit, status);
  audit[kEfAuditStoredRows] = directory->rows;
  audit[kEfAuditLowerBits] = account.lower_bits;
  audit[kEfAuditLowBytes] = account.low_bytes;
  audit[kEfAuditHighBytes] = account.high_bytes;
  audit[kEfAuditPayloadBytes] = account.payload_bytes;

  EfDecoder decoder(owner, *directory, account);
  StreamingEfTopK topk(base, query, audit);
  for (std::uint64_t index = 0; index < directory->rows; ++index) {
    std::int32_t id = -1;
    status = decoder.next(id);
    if (status != kEfOk) return fail(audit, status);
    ++audit[kEfAuditDecodedRows];
    if (static_cast<std::uint64_t>(id) >= base_rows) {
      return fail(audit, kEfCorruptPayload);
    }
    if (id == self_id) {
      ++audit[kEfAuditSelfOccurrences];
    } else {
      ++audit[kEfAuditRowsAfterSelf];
      status = topk.append(id);
      if (status != kEfOk) return fail(audit, status);
    }
  }
  status = decoder.finish();
  if (status != kEfOk) return fail(audit, status);
  audit[kEfAuditHighOneBits] = decoder.high_one_bits();
  if (audit[kEfAuditRowsAfterSelf] < kTopK) {
    return fail(audit, kEfSupportBelowK);
  }
  std::int32_t local_ids[kTopK];
  float local_distances[kTopK];
  status = topk.finish(local_ids, local_distances);
  if (status != kEfOk) return fail(audit, status);
  std::memcpy(output_ids, local_ids, sizeof(local_ids));
  std::memcpy(output_squared_l2, local_distances,
              sizeof(local_distances));
  return kEfOk;
}

}  // extern "C"
