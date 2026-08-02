#pragma once

// Project-owned PRIMARY_PAIR_MAJOR base layout for the SIFT100K attempt-04
// diagnostic.  This file has no dependency on a SIEVE baseline and accepts no
// query/workload/result inputs during construction.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sift100k_pair_major {

enum class IdSpace : uint8_t { GlobalId = 0, PhysicalPosition = 1 };

inline const char* id_space_name(IdSpace space) {
  return space == IdSpace::GlobalId ? "GLOBAL_ID" : "PHYSICAL_POSITION";
}

struct PhysicalSlice {
  uint32_t begin = 0;
  uint32_t end = 0;
  IdSpace id_space = IdSpace::PhysicalPosition;

  uint32_t rows() const { return end - begin; }
};

struct TaggedPostingView {
  const uint32_t* values = nullptr;
  size_t rows = 0;
  IdSpace id_space = IdSpace::GlobalId;
};

struct RankedHit {
  uint32_t distance = 0;
  uint32_t global_id = 0;

  bool operator<(const RankedHit& other) const {
    return std::tie(distance, global_id) <
           std::tie(other.distance, other.global_id);
  }
  bool operator==(const RankedHit& other) const {
    return distance == other.distance && global_id == other.global_id;
  }
};

struct PrimaryPairMajorCharge {
  uint64_t base_vector_payload_bytes = 0;
  uint64_t pair_offsets_bytes = 0;
  uint64_t pos_to_global_bytes = 0;
  uint64_t global_to_pos_bytes = 0;

  uint64_t incremental_logical_bytes() const {
    return pair_offsets_bytes + pos_to_global_bytes + global_to_pos_bytes;
  }
  uint64_t absolute_logical_bytes() const {
    return base_vector_payload_bytes + incremental_logical_bytes();
  }
};

struct PrimaryPairMajorAudit {
  uint64_t rows_expected = 0;
  uint64_t rows_audited = 0;
  uint64_t pair_cells_expected = 0;
  uint64_t pair_cells_audited = 0;
  uint64_t empty_pair_cells = 0;
  uint64_t sub_k_pair_cells = 0;
  uint64_t unique_global_ids = 0;
  uint64_t extent_failures = 0;
  uint64_t offset_endpoint_failures = 0;
  uint64_t offset_order_failures = 0;
  uint64_t offset_range_failures = 0;
  uint64_t invalid_global_ids = 0;
  uint64_t duplicate_global_ids = 0;
  uint64_t missing_global_ids = 0;
  uint64_t pos_to_global_inverse_failures = 0;
  uint64_t global_to_pos_inverse_failures = 0;
  uint64_t misbucketed_global_ids = 0;
  uint64_t unstable_bucket_order_failures = 0;
  uint64_t vector_payload_mismatches = 0;

  uint64_t failure_count() const {
    return extent_failures + offset_endpoint_failures +
           offset_order_failures + offset_range_failures +
           invalid_global_ids + duplicate_global_ids + missing_global_ids +
           pos_to_global_inverse_failures +
           global_to_pos_inverse_failures + misbucketed_global_ids +
           unstable_bucket_order_failures + vector_payload_mismatches;
  }
  bool passed() const {
    return failure_count() == 0 && rows_audited == rows_expected &&
           unique_global_ids == rows_expected &&
           pair_cells_audited == pair_cells_expected;
  }
};

struct PostingCompilationReceipt {
  IdSpace input_id_space = IdSpace::GlobalId;
  IdSpace output_id_space = IdSpace::PhysicalPosition;
  uint64_t rows = 0;
  uint64_t logical_bytes_before = 0;
  uint64_t logical_bytes_after = 0;
  uint64_t additional_payload_bytes = 0;
  uint64_t transient_audit_bytes = 0;
  bool all_ids_valid = false;
  bool output_sorted = false;
  bool output_unique = false;
  bool exact_round_trip = false;

  bool passed() const {
    return input_id_space == IdSpace::GlobalId &&
           output_id_space == IdSpace::PhysicalPosition &&
           logical_bytes_before == logical_bytes_after &&
           additional_payload_bytes == 0 && all_ids_valid && output_sorted &&
           output_unique && exact_round_trip;
  }
};

struct PrimaryPairMajorSelfTestReceipt {
  uint64_t checks = 0;
  uint64_t empty_support_rows = 0;
  uint64_t sub_k_support_rows = 0;
  uint64_t corruption_classes_rejected = 0;
  bool passed = false;
};

class PrimaryPairMajorBase {
 public:
  PrimaryPairMajorBase(const std::vector<uint8_t>& canonical_base,
                       const std::vector<uint8_t>& cat_a_labels,
                       const std::vector<uint8_t>& cat_b_labels,
                       uint32_t dim, uint32_t cat_a, uint32_t cat_b)
      : rows_(checked_rows(canonical_base, cat_a_labels, cat_b_labels, dim)),
        dim_(dim), cat_a_(cat_a), cat_b_(cat_b) {
    require(cat_a_ > 0 && cat_b_ > 0,
            "pair-major label domains must be positive");
    const uint64_t pair_count_64 = uint64_t(cat_a_) * cat_b_;
    require(pair_count_64 < std::numeric_limits<uint32_t>::max(),
            "pair-major label domain exceeds uint32 offsets");
    const uint32_t pairs = uint32_t(pair_count_64);

    pair_offset_.assign(size_t(pairs) + 1, 0);
    for (uint32_t global = 0; global < rows_; ++global) {
      require(cat_a_labels[global] < cat_a_ &&
                  cat_b_labels[global] < cat_b_,
              "pair-major label is outside the declared domain");
      ++pair_offset_[pair_index(cat_a_labels[global],
                                cat_b_labels[global]) + 1];
    }
    for (uint32_t pair = 0; pair < pairs; ++pair)
      pair_offset_[pair + 1] += pair_offset_[pair];
    require(pair_offset_.front() == 0 && pair_offset_.back() == rows_,
            "pair-major offsets do not cover every row");

    pos_to_global_.resize(rows_);
    global_to_pos_.resize(rows_);
    vectors_by_pos_.resize(uint64_t(rows_) * dim_);
    std::vector<uint32_t> cursor(pair_offset_.begin(),
                                 pair_offset_.end() - 1);
    // Ascending-global stable scatter freezes order within every pair cell.
    for (uint32_t global = 0; global < rows_; ++global) {
      const uint32_t pair = pair_index(cat_a_labels[global],
                                       cat_b_labels[global]);
      const uint32_t position = cursor[pair]++;
      pos_to_global_[position] = global;
      global_to_pos_[global] = position;
      std::memcpy(vectors_by_pos_.data() + uint64_t(position) * dim_,
                  canonical_base.data() + uint64_t(global) * dim_, dim_);
    }
    for (uint32_t pair = 0; pair < pairs; ++pair)
      require(cursor[pair] == pair_offset_[pair + 1],
              "pair-major stable scatter ended outside its cell");

    const PrimaryPairMajorAudit receipt =
        audit_against(canonical_base, cat_a_labels, cat_b_labels, 10);
    require(receipt.passed(), "pair-major construction audit failed");
  }

  static PrimaryPairMajorCharge logical_charge(uint32_t rows, uint32_t dim,
                                                uint32_t cat_a,
                                                uint32_t cat_b) {
    require(dim > 0 && cat_a > 0 && cat_b > 0,
            "pair-major charge dimensions must be positive");
    const uint64_t pairs = uint64_t(cat_a) * cat_b;
    require(pairs < std::numeric_limits<uint32_t>::max(),
            "pair-major charge domain exceeds uint32 offsets");
    PrimaryPairMajorCharge out;
    out.base_vector_payload_bytes = uint64_t(rows) * dim;
    out.pair_offsets_bytes = (pairs + 1) * sizeof(uint32_t);
    out.pos_to_global_bytes = uint64_t(rows) * sizeof(uint32_t);
    out.global_to_pos_bytes = uint64_t(rows) * sizeof(uint32_t);
    return out;
  }

  PrimaryPairMajorCharge logical_charge() const {
    return logical_charge(rows_, dim_, cat_a_, cat_b_);
  }

  PrimaryPairMajorCharge allocated_charge() const {
    PrimaryPairMajorCharge out;
    out.base_vector_payload_bytes =
        uint64_t(vectors_by_pos_.capacity()) * sizeof(uint8_t);
    out.pair_offsets_bytes =
        uint64_t(pair_offset_.capacity()) * sizeof(uint32_t);
    out.pos_to_global_bytes =
        uint64_t(pos_to_global_.capacity()) * sizeof(uint32_t);
    out.global_to_pos_bytes =
        uint64_t(global_to_pos_.capacity()) * sizeof(uint32_t);
    return out;
  }

  uint32_t rows() const { return rows_; }
  uint32_t dim() const { return dim_; }
  uint32_t cat_a_count() const { return cat_a_; }
  uint32_t cat_b_count() const { return cat_b_; }
  uint32_t pair_count() const { return cat_a_ * cat_b_; }

  const uint8_t* vector_by_position(uint32_t position) const {
    require(position < rows_, "pair-major physical position is invalid");
    return vectors_by_pos_.data() + uint64_t(position) * dim_;
  }

  const uint8_t* vector_by_global(uint32_t global) const {
    return vector_by_position(position_of_global(global));
  }

  uint32_t global_at_position(uint32_t position) const {
    require(position < rows_, "pair-major physical position is invalid");
    return pos_to_global_[position];
  }

  uint32_t position_of_global(uint32_t global) const {
    require(global < rows_, "pair-major global ID is invalid");
    return global_to_pos_[global];
  }

  const std::vector<uint32_t>& pair_offsets() const { return pair_offset_; }
  const std::vector<uint32_t>& pos_to_global() const {
    return pos_to_global_;
  }
  const std::vector<uint32_t>& global_to_pos() const {
    return global_to_pos_;
  }
  const std::vector<uint8_t>& vectors_by_pos() const {
    return vectors_by_pos_;
  }

  PhysicalSlice pair_slice(uint32_t a, uint32_t b) const {
    require(a < cat_a_ && b < cat_b_,
            "pair-major pair slice is outside the label domain");
    const uint32_t pair = pair_index(a, b);
    return PhysicalSlice{pair_offset_[pair], pair_offset_[pair + 1],
                         IdSpace::PhysicalPosition};
  }

  PhysicalSlice a_slice(uint32_t a) const {
    require(a < cat_a_,
            "pair-major A slice is outside the label domain");
    return PhysicalSlice{pair_offset_[a * cat_b_],
                         pair_offset_[(a + 1) * cat_b_],
                         IdSpace::PhysicalPosition};
  }

  PostingCompilationReceipt compile_global_posting_to_positions_in_place(
      std::vector<uint32_t>& values) const {
    PostingCompilationReceipt receipt;
    receipt.rows = values.size();
    receipt.logical_bytes_before = uint64_t(values.size()) * sizeof(uint32_t);
    const std::vector<uint32_t> original = values;
    receipt.transient_audit_bytes =
        uint64_t(original.capacity()) * sizeof(uint32_t);
    receipt.all_ids_valid = true;
    for (uint32_t& global : values) {
      if (global >= rows_) {
        receipt.all_ids_valid = false;
        continue;
      }
      global = global_to_pos_[global];
    }
    if (receipt.all_ids_valid) std::sort(values.begin(), values.end());
    receipt.logical_bytes_after = uint64_t(values.size()) * sizeof(uint32_t);
    receipt.additional_payload_bytes = 0;
    receipt.output_sorted = receipt.all_ids_valid &&
        std::is_sorted(values.begin(), values.end());
    receipt.output_unique = receipt.output_sorted &&
        std::adjacent_find(values.begin(), values.end()) == values.end();
    if (receipt.all_ids_valid) {
      std::vector<uint32_t> round_trip;
      round_trip.reserve(values.size());
      for (uint32_t position : values)
        round_trip.push_back(pos_to_global_[position]);
      std::sort(round_trip.begin(), round_trip.end());
      std::vector<uint32_t> sorted_original = original;
      std::sort(sorted_original.begin(), sorted_original.end());
      receipt.transient_audit_bytes +=
          uint64_t(round_trip.capacity() + sorted_original.capacity()) *
          sizeof(uint32_t);
      receipt.exact_round_trip = round_trip == sorted_original;
    }
    return receipt;
  }

  PrimaryPairMajorAudit audit_against(
      const std::vector<uint8_t>& canonical_base,
      const std::vector<uint8_t>& cat_a_labels,
      const std::vector<uint8_t>& cat_b_labels,
      uint32_t top_k) const {
    PrimaryPairMajorAudit out;
    out.rows_expected = rows_;
    out.pair_cells_expected = uint64_t(cat_a_) * cat_b_;
    const bool extents_ok =
        canonical_base.size() == uint64_t(rows_) * dim_ &&
        cat_a_labels.size() == rows_ && cat_b_labels.size() == rows_ &&
        pair_offset_.size() == out.pair_cells_expected + 1 &&
        pos_to_global_.size() == rows_ && global_to_pos_.size() == rows_ &&
        vectors_by_pos_.size() == uint64_t(rows_) * dim_;
    if (!extents_ok) {
      out.extent_failures = 1;
      return out;
    }

    if (pair_offset_.front() != 0) ++out.offset_endpoint_failures;
    if (pair_offset_.back() != rows_) ++out.offset_endpoint_failures;
    std::vector<uint8_t> seen(rows_, 0);
    for (uint32_t pair = 0; pair < pair_count(); ++pair) {
      ++out.pair_cells_audited;
      const uint32_t begin = pair_offset_[pair];
      const uint32_t end = pair_offset_[pair + 1];
      if (begin > end) {
        ++out.offset_order_failures;
        continue;
      }
      if (begin > rows_ || end > rows_) {
        ++out.offset_range_failures;
        continue;
      }
      const uint32_t cell_rows = end - begin;
      if (cell_rows == 0) ++out.empty_pair_cells;
      if (cell_rows < top_k) ++out.sub_k_pair_cells;
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
          ++out.pos_to_global_inverse_failures;
        const uint32_t expected_pair =
            pair_index(cat_a_labels[global], cat_b_labels[global]);
        if (expected_pair != pair) ++out.misbucketed_global_ids;
        if (have_previous && global <= previous_global)
          ++out.unstable_bucket_order_failures;
        previous_global = global;
        have_previous = true;
        if (std::memcmp(vectors_by_pos_.data() + uint64_t(position) * dim_,
                        canonical_base.data() + uint64_t(global) * dim_,
                        dim_) != 0)
          ++out.vector_payload_mismatches;
      }
    }
    for (uint32_t global = 0; global < rows_; ++global) {
      if (!seen[global]) ++out.missing_global_ids;
      const uint32_t position = global_to_pos_[global];
      if (position >= rows_ || pos_to_global_[position] != global)
        ++out.global_to_pos_inverse_failures;
    }
    return out;
  }

  template <class DistanceFn>
  std::vector<RankedHit> exact_top_k_segments(
      std::vector<PhysicalSlice> segments, size_t k,
      DistanceFn&& distance) const {
    if (k == 0) return {};
    for (const PhysicalSlice& segment : segments) {
      require(segment.id_space == IdSpace::PhysicalPosition,
              "pair-major segment has the wrong ID space");
      require(segment.begin <= segment.end && segment.end <= rows_,
              "pair-major segment extent is invalid");
    }
    std::sort(segments.begin(), segments.end(),
              [](const PhysicalSlice& left, const PhysicalSlice& right) {
                return std::tie(left.begin, left.end) <
                       std::tie(right.begin, right.end);
              });
    segments.erase(std::unique(
        segments.begin(), segments.end(),
        [](const PhysicalSlice& left, const PhysicalSlice& right) {
          return left.begin == right.begin && left.end == right.end;
        }), segments.end());
    for (size_t i = 1; i < segments.size(); ++i)
      require(segments[i - 1].end <= segments[i].begin,
              "pair-major exact segments overlap without being identical");

    std::priority_queue<RankedHit> heap;
    for (const PhysicalSlice& segment : segments) {
      for (uint32_t position = segment.begin; position < segment.end;
           ++position) {
        const RankedHit hit{
            uint32_t(distance(vector_by_position(position), dim_)),
            global_at_position(position)};
        consider_hit(hit, k, heap);
      }
    }
    return sorted_hits(std::move(heap));
  }

  template <class DistanceFn>
  std::vector<RankedHit> exact_top_k_postings(
      const std::vector<TaggedPostingView>& postings, size_t k,
      DistanceFn&& distance) const {
    if (k == 0) return {};
    std::priority_queue<RankedHit> heap;
    std::vector<uint8_t> seen(rows_, 0);
    for (const TaggedPostingView& posting : postings) {
      require(posting.rows == 0 || posting.values != nullptr,
              "non-empty posting has a null payload");
      for (size_t index = 0; index < posting.rows; ++index) {
        uint32_t position = posting.values[index];
        uint32_t global = 0;
        if (posting.id_space == IdSpace::GlobalId) {
          global = position;
          require(global < rows_, "posting contains an invalid global ID");
          position = global_to_pos_[global];
        } else {
          require(position < rows_,
                  "posting contains an invalid physical position");
          global = pos_to_global_[position];
        }
        if (seen[global]) continue;
        seen[global] = 1;
        const RankedHit hit{
            uint32_t(distance(vector_by_position(position), dim_)), global};
        consider_hit(hit, k, heap);
      }
    }
    return sorted_hits(std::move(heap));
  }

  static PrimaryPairMajorSelfTestReceipt deterministic_self_test() {
    PrimaryPairMajorSelfTestReceipt out;
    const uint32_t dim = 2, cat_a = 3, cat_b = 3;
    const std::vector<uint8_t> canonical = {
        1, 0,  0, 1,  1, 0,  3, 3,  2, 2,  4, 4,  3, 2};
    const std::vector<uint8_t> a = {0, 0, 0, 0, 1, 2, 0};
    const std::vector<uint8_t> b = {0, 0, 0, 1, 0, 2, 1};
    PrimaryPairMajorBase base(canonical, a, b, dim, cat_a, cat_b);
    check(base.audit_against(canonical, a, b, 10).passed(), out);
    check(base.pos_to_global_[base.global_to_pos_[5]] == 5, out);
    check(std::memcmp(base.vector_by_global(4), canonical.data() + 8, 2) == 0,
          out);

    const PhysicalSlice empty = base.pair_slice(1, 1);
    const PhysicalSlice sub_k = base.pair_slice(0, 1);
    const uint8_t query[2] = {0, 0};
    auto distance = [&](const uint8_t* vector, uint32_t vector_dim) {
      uint32_t sum = 0;
      for (uint32_t column = 0; column < vector_dim; ++column) {
        const int32_t delta = int32_t(query[column]) - vector[column];
        sum += uint32_t(delta * delta);
      }
      return sum;
    };
    const std::vector<RankedHit> empty_hits =
        base.exact_top_k_segments({empty}, 10, distance);
    const std::vector<RankedHit> sub_k_hits =
        base.exact_top_k_segments({sub_k}, 10, distance);
    out.empty_support_rows = empty_hits.size();
    out.sub_k_support_rows = sub_k_hits.size();
    check(empty_hits.empty(), out);
    check(sub_k_hits.size() == 2 && sub_k_hits[0].global_id == 6 &&
              sub_k_hits[1].global_id == 3,
          out);

    const std::vector<RankedHit> tied = base.exact_top_k_segments(
        {base.pair_slice(0, 0)}, 3, distance);
    check(tied.size() == 3 && tied[0].global_id == 0 &&
              tied[1].global_id == 1 && tied[2].global_id == 2,
          out);
    const std::vector<RankedHit> same_dnf = base.exact_top_k_segments(
        {base.pair_slice(0, 0), base.pair_slice(0, 0)}, 10, distance);
    check(same_dnf.size() == 3, out);
    const std::vector<RankedHit> a_hits = base.exact_top_k_segments(
        {base.a_slice(0)}, 10, distance);
    const std::vector<RankedHit> distinct_dnf = base.exact_top_k_segments(
        {base.pair_slice(0, 0), base.pair_slice(0, 1)}, 10, distance);
    check(a_hits.size() == 5 && a_hits == distinct_dnf, out);

    std::vector<uint32_t> posting = {0, 3, 6};
    const PostingCompilationReceipt compiled =
        base.compile_global_posting_to_positions_in_place(posting);
    check(compiled.passed(), out);
    const TaggedPostingView physical_posting{
        posting.data(), posting.size(), IdSpace::PhysicalPosition};
    const uint32_t globals[] = {0, 3, 6};
    const TaggedPostingView global_posting{
        globals, 3, IdSpace::GlobalId};
    check(base.exact_top_k_postings({physical_posting}, 10, distance) ==
              base.exact_top_k_postings({global_posting}, 10, distance),
          out);

    PrimaryPairMajorBase bad_pos = base;
    bad_pos.pos_to_global_[0] = bad_pos.pos_to_global_[1];
    reject_corruption(bad_pos.audit_against(canonical, a, b, 10), out);
    PrimaryPairMajorBase bad_global = base;
    bad_global.global_to_pos_[0] = bad_global.global_to_pos_[1];
    reject_corruption(bad_global.audit_against(canonical, a, b, 10), out);
    PrimaryPairMajorBase bad_offset = base;
    bad_offset.pair_offset_[0] = 1;
    reject_corruption(bad_offset.audit_against(canonical, a, b, 10), out);
    std::vector<uint8_t> bad_labels = a;
    bad_labels[0] = 1;
    reject_corruption(base.audit_against(canonical, bad_labels, b, 10), out);
    PrimaryPairMajorBase bad_vector = base;
    bad_vector.vectors_by_pos_[0] ^= 1;
    reject_corruption(bad_vector.audit_against(canonical, a, b, 10), out);

    out.passed = out.checks >= 12 && out.empty_support_rows == 0 &&
                 out.sub_k_support_rows == 2 &&
                 out.corruption_classes_rejected == 5;
    require(out.passed, "pair-major deterministic self-test failed");
    return out;
  }

 private:
  static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
  }

  static uint32_t checked_rows(const std::vector<uint8_t>& canonical_base,
                               const std::vector<uint8_t>& cat_a_labels,
                               const std::vector<uint8_t>& cat_b_labels,
                               uint32_t dim) {
    require(dim > 0 && canonical_base.size() % dim == 0,
            "canonical base is not an integral vector matrix");
    const uint64_t rows = canonical_base.size() / dim;
    require(rows > 0 && rows <= std::numeric_limits<uint32_t>::max(),
            "pair-major row count is outside uint32 range");
    require(cat_a_labels.size() == rows && cat_b_labels.size() == rows,
            "pair-major labels do not align with canonical rows");
    return uint32_t(rows);
  }

  uint32_t pair_index(uint32_t a, uint32_t b) const {
    return a * cat_b_ + b;
  }

  static void consider_hit(const RankedHit& hit, size_t k,
                           std::priority_queue<RankedHit>& heap) {
    if (heap.size() < k) heap.push(hit);
    else if (hit < heap.top()) {
      heap.pop();
      heap.push(hit);
    }
  }

  static std::vector<RankedHit> sorted_hits(
      std::priority_queue<RankedHit> heap) {
    std::vector<RankedHit> out(heap.size());
    for (size_t at = out.size(); at > 0; --at) {
      out[at - 1] = heap.top();
      heap.pop();
    }
    return out;
  }

  static void check(bool condition, PrimaryPairMajorSelfTestReceipt& out) {
    require(condition, "pair-major deterministic self-test check failed");
    ++out.checks;
  }

  static void reject_corruption(const PrimaryPairMajorAudit& audit,
                                PrimaryPairMajorSelfTestReceipt& out) {
    require(!audit.passed(),
            "pair-major deterministic corruption was not rejected");
    ++out.corruption_classes_rejected;
    ++out.checks;
  }

  uint32_t rows_ = 0;
  uint32_t dim_ = 0;
  uint32_t cat_a_ = 0;
  uint32_t cat_b_ = 0;
  std::vector<uint8_t> vectors_by_pos_;
  std::vector<uint32_t> pos_to_global_;
  std::vector<uint32_t> global_to_pos_;
  std::vector<uint32_t> pair_offset_;
};

}  // namespace sift100k_pair_major
