#pragma once

// Query-independent predicate-cube layout for the SIFT100K attempt-04
// diagnostic.  Construction accepts only the canonical vectors and the three
// schema columns.  Physical order is exactly (bin, A, B, global_id).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace sift100k_predicate_cube {

// Exact four-candidate AVX2 kernel used by the cube's sequential segment
// scanner.  It hoists each query-vector expansion across four independent
// candidates; arithmetic and reduction are identical to four scalar AVX2
// distance calls.  Callers audit that equivalence outside timing.
inline void l2_sq_uint8_avx2_four(const uint8_t* query,
                                  const uint8_t* const vectors[4],
                                  uint32_t dim, uint32_t distances[4]) {
  if (query == nullptr || vectors == nullptr || distances == nullptr ||
      dim == 0 || dim % 32 != 0)
    throw std::runtime_error("four-way AVX2 distance arguments differ");
  for (uint32_t lane = 0; lane < 4; ++lane)
    if (vectors[lane] == nullptr)
      throw std::runtime_error("four-way AVX2 candidate is null");
  __m256i accumulators[4] = {
      _mm256_setzero_si256(), _mm256_setzero_si256(),
      _mm256_setzero_si256(), _mm256_setzero_si256()};
  for (uint32_t offset = 0; offset < dim; offset += 32) {
    const __m256i query_bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(query + offset));
    const __m256i query_lo = _mm256_cvtepu8_epi16(
        _mm256_castsi256_si128(query_bytes));
    const __m256i query_hi = _mm256_cvtepu8_epi16(
        _mm256_extracti128_si256(query_bytes, 1));
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const __m256i candidate_bytes = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(vectors[lane] + offset));
      const __m256i candidate_lo = _mm256_cvtepu8_epi16(
          _mm256_castsi256_si128(candidate_bytes));
      const __m256i candidate_hi = _mm256_cvtepu8_epi16(
          _mm256_extracti128_si256(candidate_bytes, 1));
      const __m256i delta_lo = _mm256_sub_epi16(candidate_lo, query_lo);
      const __m256i delta_hi = _mm256_sub_epi16(candidate_hi, query_hi);
      accumulators[lane] = _mm256_add_epi32(
          accumulators[lane],
          _mm256_add_epi32(_mm256_madd_epi16(delta_lo, delta_lo),
                           _mm256_madd_epi16(delta_hi, delta_hi)));
    }
  }
  for (uint32_t lane = 0; lane < 4; ++lane) {
    __m128i reduced = _mm_add_epi32(
        _mm256_castsi256_si128(accumulators[lane]),
        _mm256_extracti128_si256(accumulators[lane], 1));
    reduced = _mm_hadd_epi32(reduced, reduced);
    reduced = _mm_hadd_epi32(reduced, reduced);
    distances[lane] = uint32_t(_mm_cvtsi128_si32(reduced));
  }
}

struct Slice {
  uint32_t begin = 0;
  uint32_t end = 0;

  uint32_t rows() const { return end - begin; }
};

struct LogicalCharge {
  uint64_t serving_vector_bytes = 0;
  uint64_t offsets_bytes = 0;
  uint64_t pos_to_global_bytes = 0;
  uint64_t global_to_pos_bytes = 0;

  uint64_t maps_bytes() const {
    return pos_to_global_bytes + global_to_pos_bytes;
  }
  uint64_t incremental_bytes() const {
    return offsets_bytes + maps_bytes();
  }
  uint64_t absolute_serving_bytes() const {
    return serving_vector_bytes + incremental_bytes();
  }
};

struct Audit {
  uint64_t rows_expected = 0;
  uint64_t rows_audited = 0;
  uint64_t cells_expected = 0;
  uint64_t cells_audited = 0;
  uint64_t unique_global_ids = 0;
  uint64_t extent_failures = 0;
  uint64_t offset_failures = 0;
  uint64_t invalid_global_ids = 0;
  uint64_t duplicate_global_ids = 0;
  uint64_t missing_global_ids = 0;
  uint64_t inverse_map_failures = 0;
  uint64_t label_domain_failures = 0;
  uint64_t label_bucket_failures = 0;
  uint64_t global_order_failures = 0;
  uint64_t vector_identity_failures = 0;

  uint64_t failure_count() const {
    return extent_failures + offset_failures + invalid_global_ids +
           duplicate_global_ids + missing_global_ids + inverse_map_failures +
           label_domain_failures + label_bucket_failures +
           global_order_failures + vector_identity_failures;
  }

  bool passed() const {
    return failure_count() == 0 && rows_audited == rows_expected &&
           unique_global_ids == rows_expected &&
           cells_audited == cells_expected;
  }
};

class PredicateCube {
 public:
  PredicateCube(const std::vector<uint8_t>& canonical_vectors,
                const std::vector<uint8_t>& bin_labels,
                const std::vector<uint8_t>& a_labels,
                const std::vector<uint8_t>& b_labels, uint32_t dim,
                uint32_t bins, uint32_t a_count, uint32_t b_count)
      : rows_(checked_rows(canonical_vectors, bin_labels, a_labels, b_labels,
                           dim)),
        dim_(dim),
        bins_(bins),
        a_count_(a_count),
        b_count_(b_count) {
    require(bins_ > 0 && a_count_ > 0 && b_count_ > 0,
            "predicate-cube label domains must be positive");
    const uint64_t cells_64 = uint64_t(bins_) * a_count_ * b_count_;
    require(cells_64 < std::numeric_limits<uint32_t>::max(),
            "predicate-cube cell count exceeds uint32 offsets");
    cells_ = uint32_t(cells_64);

    offsets_.assign(size_t(cells_) + 1, 0);
    for (uint32_t global = 0; global < rows_; ++global) {
      require(bin_labels[global] < bins_ && a_labels[global] < a_count_ &&
                  b_labels[global] < b_count_,
              "predicate-cube label is outside its declared domain");
      ++offsets_[flat_cell(bin_labels[global], a_labels[global],
                           b_labels[global]) + 1];
    }
    for (uint32_t cell = 0; cell < cells_; ++cell)
      offsets_[cell + 1] += offsets_[cell];
    require(offsets_.front() == 0 && offsets_.back() == rows_,
            "predicate-cube offsets do not span all rows");

    pos_to_global_.resize(rows_);
    global_to_pos_.resize(rows_);
    vectors_by_pos_.resize(uint64_t(rows_) * dim_);
    std::vector<uint32_t> cursor(offsets_.begin(), offsets_.end() - 1);
    // Iterating global IDs in ascending order makes the scatter stable and
    // establishes the final component of (bin,A,B,global_id).
    for (uint32_t global = 0; global < rows_; ++global) {
      const uint32_t cell = flat_cell(bin_labels[global], a_labels[global],
                                      b_labels[global]);
      const uint32_t position = cursor[cell]++;
      pos_to_global_[position] = global;
      global_to_pos_[global] = position;
      std::memcpy(vectors_by_pos_.data() + uint64_t(position) * dim_,
                  canonical_vectors.data() + uint64_t(global) * dim_, dim_);
    }
    for (uint32_t cell = 0; cell < cells_; ++cell)
      require(cursor[cell] == offsets_[cell + 1],
              "predicate-cube stable scatter ended outside its cell");

    audit_ = audit_against(canonical_vectors, bin_labels, a_labels, b_labels);
    require(audit_.passed(), "predicate-cube construction audit failed");
  }

  static LogicalCharge logical_charge(uint32_t rows, uint32_t dim,
                                      uint32_t bins, uint32_t a_count,
                                      uint32_t b_count) {
    require(dim > 0 && bins > 0 && a_count > 0 && b_count > 0,
            "predicate-cube charge dimensions must be positive");
    const uint64_t cells = uint64_t(bins) * a_count * b_count;
    require(cells < std::numeric_limits<uint32_t>::max(),
            "predicate-cube charge cell count exceeds uint32 offsets");
    LogicalCharge charge;
    charge.serving_vector_bytes = uint64_t(rows) * dim;
    charge.offsets_bytes = (cells + 1) * sizeof(uint32_t);
    charge.pos_to_global_bytes = uint64_t(rows) * sizeof(uint32_t);
    charge.global_to_pos_bytes = uint64_t(rows) * sizeof(uint32_t);
    return charge;
  }

  LogicalCharge logical_charge() const {
    return logical_charge(rows_, dim_, bins_, a_count_, b_count_);
  }

  uint32_t rows() const { return rows_; }
  uint32_t dim() const { return dim_; }
  uint32_t bins() const { return bins_; }
  uint32_t a_count() const { return a_count_; }
  uint32_t b_count() const { return b_count_; }
  uint32_t cells() const { return cells_; }
  uint32_t offset_count() const { return uint32_t(offsets_.size()); }

  const Audit& audit() const { return audit_; }
  const std::vector<uint32_t>& offsets() const { return offsets_; }
  const std::vector<uint32_t>& pos_to_global() const {
    return pos_to_global_;
  }
  const std::vector<uint32_t>& global_to_pos() const {
    return global_to_pos_;
  }
  const std::vector<uint8_t>& vectors_by_pos() const {
    return vectors_by_pos_;
  }

  Slice range_slice(uint32_t lo, uint32_t hi) const {
    require(lo <= hi && hi < bins_,
            "predicate-cube range is outside the bin domain");
    const uint32_t cells_per_bin = a_count_ * b_count_;
    return Slice{offsets_[lo * cells_per_bin],
                 offsets_[(hi + 1) * cells_per_bin]};
  }

  Slice a_slice(uint32_t bin, uint32_t a) const {
    require(bin < bins_ && a < a_count_,
            "predicate-cube A slice is outside the label domain");
    const uint32_t first = flat_cell(bin, a, 0);
    return Slice{offsets_[first], offsets_[first + b_count_]};
  }

  Slice cell_slice(uint32_t bin, uint32_t a, uint32_t b) const {
    const uint32_t cell = flat_cell(bin, a, b);
    return Slice{offsets_[cell], offsets_[cell + 1]};
  }

  Audit audit_against(const std::vector<uint8_t>& canonical_vectors,
                      const std::vector<uint8_t>& bin_labels,
                      const std::vector<uint8_t>& a_labels,
                      const std::vector<uint8_t>& b_labels) const {
    Audit out;
    out.rows_expected = rows_;
    out.cells_expected = cells_;
    const bool extents_ok =
        canonical_vectors.size() == uint64_t(rows_) * dim_ &&
        bin_labels.size() == rows_ && a_labels.size() == rows_ &&
        b_labels.size() == rows_ && offsets_.size() == size_t(cells_) + 1 &&
        pos_to_global_.size() == rows_ && global_to_pos_.size() == rows_ &&
        vectors_by_pos_.size() == uint64_t(rows_) * dim_;
    if (!extents_ok) {
      out.extent_failures = 1;
      return out;
    }

    if (offsets_.front() != 0 || offsets_.back() != rows_)
      ++out.offset_failures;
    std::vector<uint8_t> seen(rows_, 0);
    for (uint32_t cell = 0; cell < cells_; ++cell) {
      ++out.cells_audited;
      const uint32_t begin = offsets_[cell];
      const uint32_t end = offsets_[cell + 1];
      if (begin > end || end > rows_) {
        ++out.offset_failures;
        continue;
      }
      uint32_t previous_global = 0;
      bool have_previous = false;
      for (uint32_t position = begin; position < end; ++position) {
        ++out.rows_audited;
        const uint32_t global = pos_to_global_[position];
        if (global >= rows_) {
          ++out.invalid_global_ids;
          continue;
        }
        if (seen[global]) ++out.duplicate_global_ids;
        else {
          seen[global] = 1;
          ++out.unique_global_ids;
        }
        if (global_to_pos_[global] != position)
          ++out.inverse_map_failures;
        const bool labels_valid = bin_labels[global] < bins_ &&
            a_labels[global] < a_count_ && b_labels[global] < b_count_;
        if (!labels_valid) ++out.label_domain_failures;
        else if (flat_cell_unchecked(bin_labels[global], a_labels[global],
                                     b_labels[global]) != cell)
          ++out.label_bucket_failures;
        if (have_previous && global <= previous_global)
          ++out.global_order_failures;
        previous_global = global;
        have_previous = true;
        if (std::memcmp(
                vectors_by_pos_.data() + uint64_t(position) * dim_,
                canonical_vectors.data() + uint64_t(global) * dim_, dim_) != 0)
          ++out.vector_identity_failures;
      }
    }
    for (uint32_t global = 0; global < rows_; ++global) {
      if (!seen[global]) ++out.missing_global_ids;
      const uint32_t position = global_to_pos_[global];
      if (position >= rows_ || pos_to_global_[position] != global)
        ++out.inverse_map_failures;
    }
    return out;
  }

 private:
  static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
  }

  static uint32_t checked_rows(const std::vector<uint8_t>& vectors,
                               const std::vector<uint8_t>& bin_labels,
                               const std::vector<uint8_t>& a_labels,
                               const std::vector<uint8_t>& b_labels,
                               uint32_t dim) {
    require(dim > 0 && vectors.size() % dim == 0,
            "canonical vectors are not an integral matrix");
    const uint64_t rows = vectors.size() / dim;
    require(rows > 0 && rows <= std::numeric_limits<uint32_t>::max(),
            "predicate-cube row count is outside uint32 range");
    require(bin_labels.size() == rows && a_labels.size() == rows &&
                b_labels.size() == rows,
            "predicate-cube labels do not align with canonical rows");
    return uint32_t(rows);
  }

  uint32_t flat_cell_unchecked(uint32_t bin, uint32_t a,
                               uint32_t b) const {
    return (bin * a_count_ + a) * b_count_ + b;
  }

  uint32_t flat_cell(uint32_t bin, uint32_t a, uint32_t b) const {
    require(bin < bins_ && a < a_count_ && b < b_count_,
            "predicate-cube cell is outside the label domain");
    return flat_cell_unchecked(bin, a, b);
  }

  uint32_t rows_ = 0;
  uint32_t dim_ = 0;
  uint32_t bins_ = 0;
  uint32_t a_count_ = 0;
  uint32_t b_count_ = 0;
  uint32_t cells_ = 0;
  std::vector<uint8_t> vectors_by_pos_;
  std::vector<uint32_t> offsets_;
  std::vector<uint32_t> pos_to_global_;
  std::vector<uint32_t> global_to_pos_;
  Audit audit_;
};

}  // namespace sift100k_predicate_cube
