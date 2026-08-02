// Frozen C/P/X physical-layout runner for SIFT100K layout-selector v1.
// This translation unit is project adapter code.  It includes the pristine
// official SIEVE adapter source but never edits the official baseline tree.
#define main sift100k_layout_selector_embedded_v1_main_do_not_call
#include "mixed_runner.cpp"
#undef main

#include "predicate_cube.hpp"
#include "pair_major_layout.hpp"

#include <cerrno>
#include <cstdlib>
#include <functional>
#include <sys/resource.h>

namespace {

using sift100k_pair_major::PhysicalSlice;
using sift100k_pair_major::PrimaryPairMajorBase;
using sift100k_predicate_cube::PredicateCube;
using sift100k_predicate_cube::Slice;
using sift100k_predicate_cube::l2_sq_uint8_avx2_four;

constexpr uint64_t ABSOLUTE_BUDGET = 24ULL * 1024 * 1024;
constexpr uint64_t VECTOR_BYTES = uint64_t(N) * D;
constexpr uint64_t P_BASE_BYTES =
    VECTOR_BYTES + uint64_t(CAT_A * CAT_B + 1) * 4 + uint64_t(N) * 8;
constexpr uint64_t X_BASE_BYTES =
    VECTOR_BYTES + uint64_t(BINS * CAT_A * CAT_B + 1) * 4 +
    uint64_t(N) * 8;

class FourWayTop10 {
 public:
  void add(const uint8_t* vector, uint32_t global_id) {
    vectors_[pending_] = vector;
    globals_[pending_] = global_id;
    if (++pending_ == 4) flush_four();
  }

  Result finish(const uint8_t* query) {
    require(query != nullptr, "four-way scanner query is null");
    query_ = query;
    for (uint32_t lane = 0; lane < pending_; ++lane)
      consider(Hit{sieve_v25_packed::l2_sq_uint8_avx2(query_, vectors_[lane]),
                   globals_[lane]});
    pending_ = 0;
    require(size_ == K, "predicate support is below top-10");
    std::array<Hit, K> ordered = heap_;
    std::sort(ordered.begin(), ordered.end());
    Result result;
    for (uint32_t rank = 0; rank < K; ++rank)
      result.ids[rank] = ordered[rank].second;
    return result;
  }

  void set_query(const uint8_t* query) {
    require(query != nullptr, "four-way scanner query is null");
    query_ = query;
  }

 private:
  void consider(const Hit& hit) {
    if (size_ < K) {
      heap_[size_++] = hit;
      if (size_ == K) std::make_heap(heap_.begin(), heap_.end());
    } else if (hit < heap_.front()) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = hit;
      std::push_heap(heap_.begin(), heap_.end());
    }
  }
  void flush_four() {
    uint32_t distances[4]{};
    l2_sq_uint8_avx2_four(query_, vectors_.data(), D, distances);
    for (uint32_t lane = 0; lane < 4; ++lane)
      consider(Hit{distances[lane], globals_[lane]});
    pending_ = 0;
  }
  const uint8_t* query_ = nullptr;
  std::array<const uint8_t*, 4> vectors_{};
  std::array<uint32_t, 4> globals_{};
  uint32_t pending_ = 0;
  std::array<Hit, K> heap_{};
  uint32_t size_ = 0;
};

struct Options {
  std::string phase;
  std::string system;
  std::string base, metadata, query, attribute;
  std::string development_workload_tsv;
  std::string evaluation_workload_tsv;
  std::string family_order = "E,C,R,D";
  uint32_t cycles = 16;
  uint64_t budget_bytes = ABSOLUTE_BUDGET;
  uint32_t sieve_ef = 0;
};

struct Workload {
  std::vector<Query> queries;
  std::set<uint32_t> query_rows;
  std::set<uint32_t> conjunction_pairs;
  std::set<uint64_t> dnf_identities;
  std::array<uint32_t, 4> family_counts{};
  uint32_t min_row = std::numeric_limits<uint32_t>::max();
  uint32_t max_row = 0;
};

uint32_t parse_u32(const std::string& token, const std::string& context) {
  size_t used = 0;
  uint64_t value = 0;
  try {
    value = std::stoull(token, &used, 10);
  } catch (const std::exception&) {
    fail("invalid uint32 at " + context);
  }
  require(used == token.size() &&
              value <= std::numeric_limits<uint32_t>::max(),
          "invalid uint32 at " + context);
  return uint32_t(value);
}

Family parse_family(const std::string& token, const std::string& context) {
  if (token == "E" || token == "equality" || token == "0")
    return Family::Equality;
  if (token == "C" || token == "conjunction" || token == "1")
    return Family::Conjunction;
  if (token == "R" || token == "range" || token == "2")
    return Family::Range;
  if (token == "D" || token == "dnf" || token == "3")
    return Family::Dnf;
  fail("unknown family at " + context);
}

Workload load_workload(const Data& data, const std::string& path,
                       bool evaluation) {
  std::ifstream input(path);
  require(bool(input), "cannot open workload TSV: " + path);
  Workload out;
  std::string line;
  uint64_t line_number = 0;
  bool header_seen = false;
  while (std::getline(input, line)) {
    ++line_number;
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream stream(line);
    std::array<std::string, 8> fields{};
    for (std::string& field : fields)
      require(bool(stream >> field), "short workload row at " + path + ":" +
                                         std::to_string(line_number));
    std::string extra;
    require(!(stream >> extra), "long workload row at " + path + ":" +
                                    std::to_string(line_number));
    if (fields[0] == "query_row") {
      require(!header_seen && out.queries.empty() && fields[1] == "family",
              "malformed workload header");
      header_seen = true;
      continue;
    }
    const std::string context = path + ":" + std::to_string(line_number);
    const uint32_t row = parse_u32(fields[0], context);
    const Family family = parse_family(fields[1], context);
    const uint32_t a = parse_u32(fields[2], context);
    const uint32_t b = parse_u32(fields[3], context);
    const uint32_t c = parse_u32(fields[4], context);
    const uint32_t d = parse_u32(fields[5], context);
    const uint32_t lo = parse_u32(fields[6], context);
    const uint32_t hi = parse_u32(fields[7], context);
    require(row < 10000, "query row exceeds source at " + context);
    require(evaluation ? row >= 800 : row < 800,
            evaluation ? "evaluation row is below 800"
                       : "development row is not below 800");
    require(out.query_rows.insert(row).second,
            "duplicate query row at " + context);
    Spec spec;
    spec.family = family;
    if (family == Family::Equality) {
      require(a < CAT_A && data.a_post[a].size() >= K,
              "invalid equality predicate at " + context);
      spec.a = a;
    } else if (family == Family::Conjunction) {
      require(a < CAT_A && b < CAT_B &&
                  data.pair_post[pair_id(a, b)].size() >= K,
              "invalid conjunction predicate at " + context);
      require(out.conjunction_pairs.insert(pair_id(a, b)).second,
              "duplicate conjunction predicate at " + context);
      spec.a = a;
      spec.b = b;
    } else if (family == Family::Range) {
      require(lo <= hi && hi < BINS, "invalid range at " + context);
      const uint32_t width = hi - lo + 1;
      require(width == 1 || width == 2 || width == 4 || width == 8,
              "range width is outside frozen grid at " + context);
      uint64_t rows = 0;
      for (uint32_t bin = lo; bin <= hi; ++bin)
        rows += data.bin_post[bin].size();
      require(rows >= K, "range support below top-k at " + context);
      spec.lo = lo;
      spec.hi = hi;
    } else {
      require(a < CAT_A && b < CAT_B && c < CAT_A && d < CAT_B,
              "invalid DNF clause at " + context);
      const uint32_t first_pair = pair_id(a, b);
      const uint32_t second_pair = pair_id(c, d);
      require(first_pair != second_pair &&
                  data.pair_post[first_pair].size() >= K &&
                  data.pair_post[second_pair].size() >= K,
              "invalid DNF support at " + context);
      const uint32_t low = std::min(first_pair, second_pair);
      const uint32_t high = std::max(first_pair, second_pair);
      const uint64_t identity = uint64_t(low) * CAT_A * CAT_B + high;
      require(out.dnf_identities.insert(identity).second,
              "duplicate DNF identity at " + context);
      spec.a = a;
      spec.b = b;
      spec.c = c;
      spec.d = d;
    }
    Query query;
    query.spec = spec;
    std::memcpy(query.vector.data(),
                data.query_rows.data() + uint64_t(row) * D, D);
    out.queries.push_back(query);
    ++out.family_counts[uint32_t(family)];
    out.min_row = std::min(out.min_row, row);
    out.max_row = std::max(out.max_row, row);
  }
  require(input.eof(), "workload read failed: " + path);
  require(out.queries.size() == 400 && out.query_rows.size() == 400,
          "workload must contain 400 unique rows");
  for (uint32_t family = 0; family < 4; ++family)
    require(out.family_counts[family] == 100,
            "workload must contain 100 rows per family");
  return out;
}

void require_split_disjoint(const Workload& development,
                            const Workload& evaluation) {
  for (uint32_t row : development.query_rows)
    require(!evaluation.query_rows.count(row), "split query rows overlap");
  for (uint32_t pair : development.conjunction_pairs)
    require(!evaluation.conjunction_pairs.count(pair),
            "split conjunction predicates overlap");
  for (uint64_t dnf : development.dnf_identities)
    require(!evaluation.dnf_identities.count(dnf),
            "split DNF predicates overlap");
}

std::array<Family, 4> parse_family_order(const std::string& text) {
  std::array<Family, 4> order{};
  std::istringstream stream(text);
  std::string token;
  std::set<uint32_t> seen;
  uint32_t at = 0;
  while (std::getline(stream, token, ',')) {
    require(at < 4, "family order has too many entries");
    order[at] = parse_family(token, "family order");
    require(seen.insert(uint32_t(order[at])).second,
            "family order repeats a family");
    ++at;
  }
  require(at == 4 && seen.size() == 4,
          "family order must be a permutation of E,C,R,D");
  return order;
}

Options parse_options(int argc, char** argv) {
  Options out;
  require(argc >= 3, "options are required");
  for (int index = 1; index < argc; index += 2) {
    const std::string key = argv[index];
    require(index + 1 < argc, "missing option value");
    if (key == "--phase") {
      require(out.phase.empty(), "phase specified twice");
      out.phase = argv[index + 1];
    }
  }
  require(out.phase == "select" || out.phase == "execute",
          "phase must be select or execute");
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    require(index + 1 < argc, "missing option value");
    if (out.phase == "select" && key == "--evaluation-workload-tsv")
      fail("selection phase rejects the evaluation-workload option");
    const std::string value = argv[++index];
    if (key == "--phase") continue;
    if (key == "--system") out.system = value;
    else if (key == "--base") out.base = value;
    else if (key == "--metadata") out.metadata = value;
    else if (key == "--query") out.query = value;
    else if (key == "--attribute") out.attribute = value;
    else if (key == "--development-workload-tsv")
      out.development_workload_tsv = value;
    else if (key == "--evaluation-workload-tsv")
      out.evaluation_workload_tsv = value;
    else if (key == "--family-order") out.family_order = value;
    else if (key == "--cycles") out.cycles = std::stoul(value);
    else if (key == "--budget-bytes") out.budget_bytes = std::stoull(value);
    else if (key == "--sieve-ef") out.sieve_ef = std::stoul(value);
    else fail("unknown layout-selector option " + key);
  }
  require(out.system == "C" || out.system == "P" || out.system == "X" ||
              out.system == "SIEVE",
          "system must be C, P, X, or SIEVE");
  require(!out.base.empty() && !out.metadata.empty() && !out.query.empty() &&
              !out.attribute.empty() && !out.development_workload_tsv.empty(),
          "data and development workload options are required");
  require(out.phase == "select" ? out.evaluation_workload_tsv.empty()
                                 : !out.evaluation_workload_tsv.empty(),
          "evaluation workload is required only for execute");
  require(out.cycles > 0, "cycles must be positive");
  require(out.budget_bytes == ABSOLUTE_BUDGET,
          "absolute logical budget must equal 24 MiB");
  require(out.system == "SIEVE" ? out.sieve_ef > 0 : out.sieve_ef == 0,
          "sieve-ef is required only for SIEVE");
  if (out.system == "SIEVE")
    require(out.sieve_ef >= 100 && out.sieve_ef <= 200 &&
                out.sieve_ef % 20 == 0,
            "sieve-ef is outside the frozen grid");
  (void)parse_family_order(out.family_order);
  return out;
}

uint64_t support_catalog_bytes(const Data& data, uint64_t* posting_rows_out) {
  uint64_t rows = 0;
  auto add = [&](const auto& postings) {
    for (const auto& posting : postings) rows += posting.size();
  };
  add(data.a_post);
  add(data.b_post);
  add(data.pair_post);
  add(data.bin_post);
  for (const auto& entry : data.dyadic_post) rows += entry.second.size();
  if (posting_rows_out) *posting_rows_out = rows;
  return data.cat_a.size() + data.cat_b.size() + data.bin.size() + rows * 4;
}

std::string join_keys(const std::set<std::string>& keys) {
  std::ostringstream out;
  bool first = true;
  for (const std::string& key : keys) {
    if (!first) out << ',';
    first = false;
    out << key;
  }
  return out.str();
}

std::vector<Query> range_development(const std::vector<Query>& queries) {
  std::vector<Query> out;
  for (const Query& query : queries)
    if (query.spec.family == Family::Range) out.push_back(query);
  require(out.size() == 100, "range development must contain 100 rows");
  return out;
}

Engine::Selection active_selection(const Engine& engine) {
  Engine::Selection active;
  for (const auto& entry : engine.materialized) active.ids.insert(entry.first);
  for (const auto& entry : engine.packed) active.packed.insert(entry.first);
  return active;
}

class CEngine {
 public:
  CEngine(const Data& data, uint64_t residual_budget,
          const std::vector<Query>& development)
      : data_(data), engine_(data, Arm::Joint, residual_budget, development) {}

  Result query(const Query& query) const {
    FourWayTop10 scan;
    scan.set_query(query.vector.data());
    const Spec& spec = query.spec;
    if (spec.family == Family::Equality) {
      feed_key("A" + std::to_string(spec.a), scan);
    } else if (spec.family == Family::Conjunction) {
      feed_pair(spec.a, spec.b, scan);
    } else if (spec.family == Family::Range) {
      const auto cover = engine_.cover(spec.lo, spec.hi,
                                        active_selection(engine_));
      require(!cover.keys.empty(), "C range cover is empty");
      for (const std::string& key : cover.keys) feed_key(key, scan);
    } else {
      feed_pair(spec.a, spec.b, scan);
      if (spec.a != spec.c || spec.b != spec.d)
        feed_pair(spec.c, spec.d, scan);
    }
    return scan.finish(query.vector.data());
  }

  uint64_t residual_bytes() const { return engine_.bytes; }
  const Engine& engine() const { return engine_; }

 private:
  const std::vector<uint32_t>& ids_for(const std::string& key) const {
    const auto materialized = engine_.materialized.find(key);
    return materialized == engine_.materialized.end()
               ? *engine_.catalog.at(key).ids
               : materialized->second;
  }
  void feed_key(const std::string& key, FourWayTop10& scan) const {
    const auto& ids = ids_for(key);
    const auto packed = engine_.packed.find(key);
    if (packed != engine_.packed.end()) {
      require(packed->second.vectors.size() == uint64_t(ids.size()) * D,
              "C packed extent differs");
      for (size_t at = 0; at < ids.size(); ++at)
        scan.add(packed->second.vectors.data() + uint64_t(at) * D, ids[at]);
    } else {
      for (uint32_t id : ids)
        scan.add(data_.base.data() + uint64_t(id) * D, id);
    }
  }
  void feed_pair(uint32_t a, uint32_t b, FourWayTop10& scan) const {
    const std::string key = "P" + std::to_string(pair_id(a, b));
    if (engine_.materialized.count(key) || engine_.packed.count(key)) {
      feed_key(key, scan);
      return;
    }
    const auto& left = data_.a_post[a];
    const auto& right = data_.b_post[b];
    size_t i = 0, j = 0;
    while (i < left.size() && j < right.size()) {
      if (left[i] < right[j]) ++i;
      else if (right[j] < left[i]) ++j;
      else {
        const uint32_t id = left[i];
        scan.add(data_.base.data() + uint64_t(id) * D, id);
        ++i;
        ++j;
      }
    }
  }
  const Data& data_;
  Engine engine_;
};

class PEngine {
 public:
  PEngine(const Data& data, uint64_t residual_budget,
          const std::vector<Query>& development)
      : data_(data), base_(data.base, data.cat_a, data.cat_b, D, CAT_A, CAT_B),
        residual_(data, Arm::Joint, residual_budget,
                  range_development(development)) {
    for (const std::string& key : residual_.admitted)
      require(residual_.is_range(key), "P admitted a non-range residual");
    for (const auto& entry : residual_.packed)
      require(residual_.is_range(entry.first),
              "P packed a non-range residual");
    for (auto it = residual_.catalog.begin(); it != residual_.catalog.end();) {
      if (!residual_.is_range(it->first)) it = residual_.catalog.erase(it);
      else ++it;
    }
    for (auto it = residual_.universe.begin(); it != residual_.universe.end();) {
      if (it->empty() || (*it)[0] != 'R') it = residual_.universe.erase(it);
      else ++it;
    }
    require(residual_.catalog.size() == 30 && residual_.universe.size() == 30,
            "P residual catalog is not the frozen dyadic catalog");
  }

  Result query(const Query& query) const {
    FourWayTop10 scan;
    scan.set_query(query.vector.data());
    const Spec& spec = query.spec;
    if (spec.family == Family::Equality) {
      feed_slice(base_.a_slice(spec.a), scan);
    } else if (spec.family == Family::Conjunction) {
      feed_slice(base_.pair_slice(spec.a, spec.b), scan);
    } else if (spec.family == Family::Range) {
      const auto cover = residual_.cover(spec.lo, spec.hi,
                                          active_selection(residual_));
      require(!cover.keys.empty(), "P range cover is empty");
      for (const std::string& key : cover.keys) feed_range_key(key, scan);
    } else {
      feed_slice(base_.pair_slice(spec.a, spec.b), scan);
      if (spec.a != spec.c || spec.b != spec.d)
        feed_slice(base_.pair_slice(spec.c, spec.d), scan);
    }
    return scan.finish(query.vector.data());
  }

  uint64_t residual_bytes() const { return residual_.bytes; }
  const Engine& engine() const { return residual_; }
  const PrimaryPairMajorBase& base() const { return base_; }

 private:
  void feed_slice(const PhysicalSlice& slice, FourWayTop10& scan) const {
    const auto& vectors = base_.vectors_by_pos();
    const auto& globals = base_.pos_to_global();
    require(slice.begin <= slice.end && slice.end <= globals.size() &&
                vectors.size() == uint64_t(globals.size()) * D,
            "P segment extent differs");
    for (uint32_t position = slice.begin; position < slice.end; ++position)
      scan.add(vectors.data() + uint64_t(position) * D, globals[position]);
  }
  void feed_range_key(const std::string& key, FourWayTop10& scan) const {
    const auto materialized = residual_.materialized.find(key);
    const auto& ids = materialized == residual_.materialized.end()
                          ? *residual_.catalog.at(key).ids
                          : materialized->second;
    const auto packed = residual_.packed.find(key);
    if (packed != residual_.packed.end()) {
      require(packed->second.vectors.size() == uint64_t(ids.size()) * D,
              "P packed range extent differs");
      for (size_t at = 0; at < ids.size(); ++at)
        scan.add(packed->second.vectors.data() + uint64_t(at) * D, ids[at]);
    } else {
      const auto& vectors = base_.vectors_by_pos();
      const auto& positions = base_.global_to_pos();
      require(vectors.size() == uint64_t(positions.size()) * D &&
                  (ids.empty() || ids.back() < positions.size()),
              "P range base extent differs");
      for (uint32_t id : ids) {
        const uint32_t position = positions[id];
        scan.add(vectors.data() + uint64_t(position) * D, id);
      }
    }
  }
  const Data& data_;
  PrimaryPairMajorBase base_;
  Engine residual_;
};

class XEngine {
 public:
  explicit XEngine(const PredicateCube& cube) : cube_(cube) {}
  Result query(const Query& query) const {
    std::array<Slice, 2 * BINS> segments{};
    uint32_t count = 0;
    const Spec& spec = query.spec;
    if (spec.family == Family::Range) {
      segments[count++] = cube_.range_slice(spec.lo, spec.hi);
    } else if (spec.family == Family::Equality) {
      for (uint32_t bin = 0; bin < BINS; ++bin)
        segments[count++] = cube_.a_slice(bin, spec.a);
    } else if (spec.family == Family::Conjunction) {
      for (uint32_t bin = 0; bin < BINS; ++bin)
        segments[count++] = cube_.cell_slice(bin, spec.a, spec.b);
    } else {
      const uint32_t first = pair_id(spec.a, spec.b);
      const uint32_t second = pair_id(spec.c, spec.d);
      for (uint32_t bin = 0; bin < BINS; ++bin) {
        if (first < second) {
          segments[count++] = cube_.cell_slice(bin, spec.a, spec.b);
          segments[count++] = cube_.cell_slice(bin, spec.c, spec.d);
        } else {
          segments[count++] = cube_.cell_slice(bin, spec.c, spec.d);
          segments[count++] = cube_.cell_slice(bin, spec.a, spec.b);
        }
      }
    }
    FourWayTop10 scan;
    scan.set_query(query.vector.data());
    const auto& vectors = cube_.vectors_by_pos();
    const auto& globals = cube_.pos_to_global();
    for (uint32_t part = 0; part < count; ++part) {
      for (uint32_t position = segments[part].begin;
           position < segments[part].end; ++position) {
        if (position + 16 < segments[part].end) {
          const uint8_t* future =
              vectors.data() + uint64_t(position + 16) * D;
          _mm_prefetch(reinterpret_cast<const char*>(future), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(future + 64),
                       _MM_HINT_T0);
        }
        scan.add(vectors.data() + uint64_t(position) * D,
                 globals[position]);
      }
    }
    return scan.finish(query.vector.data());
  }
 private:
  const PredicateCube& cube_;
};

#ifdef SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE

// X3 changes only the exact range scan.  The cube layout and the equality,
// conjunction, and DNF routes remain the frozen X implementation above.
constexpr uint32_t X3_SEED_ROWS = 256;
constexpr uint32_t X3_BATCH_ROWS = 128;
constexpr uint32_t X3_PREFETCH_ROWS = 16;
constexpr uint64_t X3_TRANSIENT_SCRATCH_UPPER_BOUND_BYTES = 4096;

struct alignas(32) X3ExpandedQuery {
  __m256i lo[4];
  __m256i hi[4];
};

inline X3ExpandedQuery x3_expand_query(const uint8_t* query) {
  X3ExpandedQuery out;
  for (uint32_t block = 0; block < 4; ++block) {
    const __m256i bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(query + uint64_t(block) * 32));
    out.lo[block] =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bytes));
    out.hi[block] =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bytes, 1));
  }
  return out;
}

inline __m256i x3_squared_block(const X3ExpandedQuery& query,
                                uint32_t block,
                                const uint8_t* candidate) {
  const __m256i bytes = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(candidate + uint64_t(block) * 32));
  const __m256i lo =
      _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bytes));
  const __m256i hi =
      _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bytes, 1));
  const __m256i dlo = _mm256_sub_epi16(lo, query.lo[block]);
  const __m256i dhi = _mm256_sub_epi16(hi, query.hi[block]);
  return _mm256_add_epi32(_mm256_madd_epi16(dlo, dlo),
                          _mm256_madd_epi16(dhi, dhi));
}

inline void x3_reduce_four(__m256i a, __m256i b, __m256i c, __m256i d,
                           uint32_t out[4]) {
  const __m256i ab = _mm256_hadd_epi32(a, b);
  const __m256i cd = _mm256_hadd_epi32(c, d);
  const __m256i halves = _mm256_hadd_epi32(ab, cd);
  const __m128i totals = _mm_add_epi32(
      _mm256_castsi256_si128(halves),
      _mm256_extracti128_si256(halves, 1));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), totals);
}

inline void x3_half_four(const X3ExpandedQuery& query,
                         uint32_t first_block,
                         const uint8_t* const candidates[4],
                         uint32_t out[4]) {
  __m256i sums[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                     _mm256_setzero_si256(), _mm256_setzero_si256()};
  for (uint32_t block = first_block; block < first_block + 2; ++block)
    for (uint32_t lane = 0; lane < 4; ++lane)
      sums[lane] = _mm256_add_epi32(
          sums[lane], x3_squared_block(query, block, candidates[lane]));
  x3_reduce_four(sums[0], sums[1], sums[2], sums[3], out);
}

inline uint32_t x3_half_one(const X3ExpandedQuery& query,
                            uint32_t first_block,
                            const uint8_t* candidate) {
  __m256i sum = _mm256_add_epi32(
      x3_squared_block(query, first_block, candidate),
      x3_squared_block(query, first_block + 1, candidate));
  __m128i reduced = _mm_add_epi32(
      _mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  reduced = _mm_hadd_epi32(reduced, reduced);
  reduced = _mm_hadd_epi32(reduced, reduced);
  return uint32_t(_mm_cvtsi128_si32(reduced));
}

class X3Top10 {
 public:
  inline void consider(uint32_t distance, uint32_t global_id) {
    const Hit hit{distance, global_id};
    if (size_ < K) {
      heap_[size_++] = hit;
      if (size_ == K) std::make_heap(heap_.begin(), heap_.end());
    } else if (hit < heap_.front()) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = hit;
      std::push_heap(heap_.begin(), heap_.end());
    }
  }
  bool full() const { return size_ == K; }
  uint32_t threshold() const {
    require(full(), "X3 threshold read before top-10 initialization");
    return heap_.front().first;
  }
  Result finish() const {
    require(full(), "X3 exact top-10 is incomplete");
    std::array<Hit, K> ordered = heap_;
    std::sort(ordered.begin(), ordered.end());
    Result result;
    for (uint32_t rank = 0; rank < K; ++rank)
      result.ids[rank] = ordered[rank].second;
    return result;
  }
 private:
  std::array<Hit, K> heap_{};
  uint32_t size_ = 0;
};

struct X3Survivor {
  uint32_t position;
  uint32_t partial;
};

Result x3_compact_range_scan(const uint8_t* query, const uint8_t* vectors,
                             const uint32_t* ids, uint32_t rows) {
  require(query && vectors && ids && rows >= K,
          "X3 compact range extent differs");
  const X3ExpandedQuery expanded = x3_expand_query(query);
  X3Top10 top10;
  const uint32_t seed_rows = std::min(X3_SEED_ROWS, rows);
  uint32_t position = 0;
  for (; position + 4 <= seed_rows; position += 4) {
    const uint8_t* candidates[4] = {
        vectors + uint64_t(position) * D,
        vectors + uint64_t(position + 1) * D,
        vectors + uint64_t(position + 2) * D,
        vectors + uint64_t(position + 3) * D};
    uint32_t first[4]{}, second[4]{};
    x3_half_four(expanded, 0, candidates, first);
    x3_half_four(expanded, 2, candidates, second);
    for (uint32_t lane = 0; lane < 4; ++lane)
      top10.consider(first[lane] + second[lane], ids[position + lane]);
  }
  for (; position < seed_rows; ++position)
    top10.consider(
        sieve_v25_packed::l2_sq_uint8_avx2(
            query, vectors + uint64_t(position) * D), ids[position]);
  require(top10.full(), "X3 seed did not initialize exact top-10");

  std::array<X3Survivor, X3_BATCH_ROWS> survivors{};
  while (position < rows) {
    const uint32_t batch_end =
        std::min<uint32_t>(rows, position + X3_BATCH_ROWS);
    const uint32_t threshold = top10.threshold();
    uint32_t survivor_count = 0;
    for (; position + 4 <= batch_end; position += 4) {
      if (position + X3_PREFETCH_ROWS < batch_end) {
        const uint8_t* future =
            vectors + uint64_t(position + X3_PREFETCH_ROWS) * D;
        _mm_prefetch(reinterpret_cast<const char*>(future), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(future + 64), _MM_HINT_T0);
      }
      const uint8_t* candidates[4] = {
          vectors + uint64_t(position) * D,
          vectors + uint64_t(position + 1) * D,
          vectors + uint64_t(position + 2) * D,
          vectors + uint64_t(position + 3) * D};
      uint32_t partial[4]{};
      x3_half_four(expanded, 0, candidates, partial);
      for (uint32_t lane = 0; lane < 4; ++lane)
        if (partial[lane] <= threshold)
          survivors[survivor_count++] =
              X3Survivor{position + lane, partial[lane]};
    }
    for (; position < batch_end; ++position) {
      const uint32_t partial = x3_half_one(
          expanded, 0, vectors + uint64_t(position) * D);
      if (partial <= threshold)
        survivors[survivor_count++] = X3Survivor{position, partial};
    }
    require(survivor_count <= X3_BATCH_ROWS,
            "X3 survivor compaction overflow");
    uint32_t survivor = 0;
    for (; survivor + 4 <= survivor_count; survivor += 4) {
      const uint8_t* candidates[4] = {
          vectors + uint64_t(survivors[survivor].position) * D,
          vectors + uint64_t(survivors[survivor + 1].position) * D,
          vectors + uint64_t(survivors[survivor + 2].position) * D,
          vectors + uint64_t(survivors[survivor + 3].position) * D};
      uint32_t suffix[4]{};
      x3_half_four(expanded, 2, candidates, suffix);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const X3Survivor& kept = survivors[survivor + lane];
        top10.consider(kept.partial + suffix[lane], ids[kept.position]);
      }
    }
    for (; survivor < survivor_count; ++survivor) {
      const X3Survivor& kept = survivors[survivor];
      top10.consider(
          kept.partial + x3_half_one(
              expanded, 2, vectors + uint64_t(kept.position) * D),
          ids[kept.position]);
    }
  }
  return top10.finish();
}

class X3Engine {
 public:
  explicit X3Engine(const PredicateCube& cube) : cube_(cube), legacy_(cube) {}
  Result query(const Query& query) const {
    if (query.spec.family != Family::Range) return legacy_.query(query);
    const Slice slice = cube_.range_slice(query.spec.lo, query.spec.hi);
    return x3_compact_range_scan(
        query.vector.data(),
        cube_.vectors_by_pos().data() + uint64_t(slice.begin) * D,
        cube_.pos_to_global().data() + slice.begin, slice.rows());
  }
 private:
  const PredicateCube& cube_;
  XEngine legacy_;
};

#endif  // SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE

class IndependentOracle {
 public:
  explicit IndependentOracle(const Data& data) : data_(data) {}
  Result query(const Query& query) const {
    return scan(data_, query.vector.data(), support(data_, query.spec));
  }
 private:
  const Data& data_;
};

Result ordered_sieve_result(const Result& raw) {
  Result result = raw;
  std::reverse(result.ids.begin(), result.ids.end());
  return result;
}

struct SelectorMetrics {
  uint64_t elapsed_ns = 0;
  uint64_t queries = 0;
  uint64_t recall_hits = 0;
  uint64_t recall_denominator = 0;
  uint64_t invalid = 0;
  uint64_t predicate_fail = 0;
  uint64_t duplicate = 0;
  uint64_t nondeterministic = 0;
  uint64_t exact_mismatch_queries = 0;
  uint64_t exact_mismatch_ranks = 0;
  double qps() const {
    return elapsed_ns == 0 ? std::numeric_limits<double>::infinity()
                           : double(queries) * 1e9 / elapsed_ns;
  }
};

struct TimedFamily {
  Family family{};
  std::vector<size_t> rows;
  std::vector<Result> results;
  SelectorMetrics metrics;
};

void add_selector_metrics(SelectorMetrics& total,
                          const SelectorMetrics& part) {
  total.elapsed_ns += part.elapsed_ns;
  total.queries += part.queries;
  total.recall_hits += part.recall_hits;
  total.recall_denominator += part.recall_denominator;
  total.invalid += part.invalid;
  total.predicate_fail += part.predicate_fail;
  total.duplicate += part.duplicate;
  total.nondeterministic += part.nondeterministic;
  total.exact_mismatch_queries += part.exact_mismatch_queries;
  total.exact_mismatch_ranks += part.exact_mismatch_ranks;
}

void emit_block(const char* record, const std::string& system,
                const std::string& family_order,
                const SelectorMetrics& metrics,
                const char* family = nullptr) {
  std::cout << record << " system=" << system;
  if (family) std::cout << " family=" << family;
  std::cout << " queries=" << metrics.queries
            << " elapsed_ns=" << metrics.elapsed_ns
            << " qps=" << std::setprecision(17) << metrics.qps();
  if (metrics.recall_denominator) {
    std::cout << " recall=" << std::setprecision(17)
              << double(metrics.recall_hits) / metrics.recall_denominator
              << " invalid=" << metrics.invalid
              << " predicate_fail=" << metrics.predicate_fail
              << " duplicate=" << metrics.duplicate
              << " nondeterministic=" << metrics.nondeterministic
              << " exact_mismatch_queries="
              << metrics.exact_mismatch_queries
              << " exact_mismatch_ranks=" << metrics.exact_mismatch_ranks;
  }
  std::cout << " family_order=" << family_order
            << " validation_after_all_timing=true" << std::endl;
}

template <class QueryFn>
SelectorMetrics execute_workload(const Data& data,
                                 const std::vector<Query>& workload,
                                 const Options& options, bool exact_required,
                                 QueryFn& query_fn) {
  const auto order = parse_family_order(options.family_order);
  std::array<std::vector<size_t>, 4> family_rows;
  for (size_t row = 0; row < workload.size(); ++row)
    family_rows[uint32_t(workload[row].spec.family)].push_back(row);
  for (uint32_t family = 0; family < 4; ++family)
    require(family_rows[family].size() == 100,
            "timed workload family count differs from 100");

  // Exactly one complete untimed warm pass precedes every timed block.
  for (Family family : order)
    for (size_t row : family_rows[uint32_t(family)])
      (void)query_fn(workload[row]);

  std::array<TimedFamily, 4> timed;
  for (Family family : order) {
    TimedFamily& one = timed[uint32_t(family)];
    one.family = family;
    one.rows = family_rows[uint32_t(family)];
    one.results.resize(one.rows.size());
    const auto begin = Clock::now();
    for (uint32_t cycle = 0; cycle < options.cycles; ++cycle)
      for (size_t at = 0; at < one.rows.size(); ++at)
        one.results[at] = query_fn(workload[one.rows[at]]);
    const auto end = Clock::now();
    one.metrics.elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count();
    one.metrics.queries = uint64_t(one.rows.size()) * options.cycles;
    emit_block("TIMED_BLOCK", options.system, options.family_order,
               one.metrics, family_name(family));
  }

  // The independent oracle and every recall/integrity operation begin only
  // after all four timed family blocks have completed.
  const IndependentOracle oracle(data);
  SelectorMetrics total;
  for (Family family : order) {
    TimedFamily& one = timed[uint32_t(family)];
    for (size_t at = 0; at < one.rows.size(); ++at) {
      const Query& query = workload[one.rows[at]];
      const Result expected = oracle.query(query);
      const Result repeated_one = query_fn(query);
      const Result repeated_two = query_fn(query);
      if (repeated_one.ids != one.results[at].ids ||
          repeated_two.ids != one.results[at].ids)
        ++one.metrics.nondeterministic;
      if (expected.ids != one.results[at].ids)
        ++one.metrics.exact_mismatch_queries;
      std::set<uint32_t> expected_set(expected.ids.begin(), expected.ids.end());
      std::set<uint32_t> unique;
      for (uint32_t rank = 0; rank < K; ++rank) {
        const uint32_t id = one.results[at].ids[rank];
        if (id != expected.ids[rank]) ++one.metrics.exact_mismatch_ranks;
        ++one.metrics.recall_denominator;
        if (!unique.insert(id).second) ++one.metrics.duplicate;
        if (id >= N) {
          ++one.metrics.invalid;
          continue;
        }
        if (!matches(data, query.spec, id)) ++one.metrics.predicate_fail;
        if (expected_set.count(id)) ++one.metrics.recall_hits;
      }
    }
    require(one.metrics.invalid == 0 && one.metrics.predicate_fail == 0 &&
                one.metrics.duplicate == 0 &&
                one.metrics.nondeterministic == 0,
            "result integrity validation failed");
    if (exact_required)
      require(one.metrics.exact_mismatch_queries == 0 &&
                  one.metrics.exact_mismatch_ranks == 0 &&
                  one.metrics.recall_hits == one.metrics.recall_denominator,
              "exact package differs from independent oracle");
    else if (options.phase == "execute")
      require(one.metrics.recall_denominator > 0 &&
                  double(one.metrics.recall_hits) /
                          one.metrics.recall_denominator >=
                      .995,
              "formal SIEVE family recall is below 0.995");
    emit_block("BLOCK", options.system, options.family_order, one.metrics,
               family_name(family));
    add_selector_metrics(total, one.metrics);
  }
  emit_block("SUMMARY", options.system, options.family_order, total);
  return total;
}

std::string manifest_states(const Engine& engine) {
  std::ostringstream out;
  bool first = true;
  for (const std::string& key : engine.universe) {
    if (!first) out << ',';
    first = false;
    const bool ids = engine.materialized.count(key);
    const bool packed = engine.packed.count(key);
    out << key << ':' << (packed ? "PACKED" : (ids ? "IDS" : "ABSENT"));
  }
  return out.str();
}

std::set<std::string> packed_keys(const Engine& engine) {
  std::set<std::string> out;
  for (const auto& entry : engine.packed) out.insert(entry.first);
  return out;
}

std::set<std::string> admitted_union(const Engine& engine) {
  std::set<std::string> out = engine.admitted;
  for (const auto& entry : engine.packed) out.insert(entry.first);
  return out;
}

void emit_manifest(const std::string& package, const Engine& engine) {
  std::cout << "CATALOG_RECEIPT package=" << package
            << " candidate_count=" << engine.universe.size()
            << " candidate_keys=" << join_keys(engine.universe)
            << " objective_id=unified_mixed_v1_total_cost"
            << " tie_rule_id=density_then_cost_then_key"
            << " states=ABSENT,IDS,PACKED" << std::endl;
  std::cout << "MANIFEST_RECEIPT package=" << package
            << " candidate_keys=" << join_keys(engine.universe)
            << " admitted_keys=" << join_keys(admitted_union(engine))
            << " admitted_ids=" << join_keys(engine.admitted)
            << " admitted_packed=" << join_keys(packed_keys(engine))
            << " state_by_key=" << manifest_states(engine)
            << " residual_bytes=" << engine.bytes
            << " objective_id=unified_mixed_v1_total_cost"
            << " tie_rule_id=density_then_cost_then_key"
            << " selection_source=current-development-tsv"
            << " copied_from_older_manifest=false" << std::endl;
}

uint64_t peak_rss_bytes() {
  struct rusage usage {};
  require(getrusage(RUSAGE_SELF, &usage) == 0,
          "getrusage failed");
#if defined(__APPLE__)
  return uint64_t(usage.ru_maxrss);
#else
  return uint64_t(usage.ru_maxrss) * 1024;
#endif
}

void emit_sieve_resource_ledger(const SieveEngine& engine) {
  require(engine.filters && engine.space && engine.index,
          "SIEVE resource ledger encountered an incomplete engine");
  const auto& filters = *engine.filters;
  const auto& index = *engine.index;

  uint64_t single_filter_bitmap_serialized_bytes = 0;
  for (const auto* bitmap : filters.filter_bvs) {
    require(bitmap != nullptr, "SIEVE single-filter bitmap is null");
    single_filter_bitmap_serialized_bytes += bitmap->getSizeInBytes();
  }
  require(filters.base_bv != nullptr, "SIEVE base bitmap is null");
  const uint64_t base_bitmap_serialized_bytes =
      filters.base_bv->getSizeInBytes();
  uint64_t multifilter_bitmap_serialized_bytes = 0;
  for (const auto& item : filters.multifilters_bvs) {
    require(item.second != nullptr, "SIEVE multifilter bitmap is null");
    multifilter_bitmap_serialized_bytes += item.second->getSizeInBytes();
  }

  uint64_t hnsw_instances = 0;
  uint64_t hnsw_elements = 0;
  uint64_t hnsw_level0_used_bytes = 0;
  uint64_t hnsw_level0_backing_bytes = 0;
  uint64_t hnsw_vector_pointer_used_bytes = 0;
  uint64_t hnsw_vector_pointer_backing_bytes = 0;
  uint64_t hnsw_upper_link_backing_bytes = 0;
  uint64_t hnsw_upper_link_pointer_used_bytes = 0;
  uint64_t hnsw_upper_link_pointer_backing_bytes = 0;
  uint64_t hnsw_element_level_logical_bytes = 0;
  uint64_t hnsw_element_level_capacity_bytes = 0;
  uint64_t hnsw_label_lookup_entries = 0;
  uint64_t hnsw_deleted_entries = 0;
  uint64_t child_pointer_logical_bytes = 0;
  uint64_t child_pointer_capacity_bytes = 0;
  std::set<const hnswlib::PartitionedHNSWNode<int, uint8_t>*> seen_nodes;
  auto inventory_node = [&](const auto* node) {
    require(node != nullptr && node->_hnsw != nullptr,
            "SIEVE graph node is incomplete");
    require(seen_nodes.insert(node).second,
            "SIEVE graph node is reachable more than once");
    const auto& hnsw = *node->_hnsw;
    const uint64_t elements = hnsw.cur_element_count.load();
    require(elements <= hnsw.max_elements_ &&
                elements <= hnsw.element_levels_.size(),
            "SIEVE HNSW element extent is inconsistent");
    ++hnsw_instances;
    hnsw_elements += elements;
    hnsw_level0_used_bytes += elements * hnsw.size_data_per_element_;
    hnsw_level0_backing_bytes +=
        hnsw.max_elements_ * hnsw.size_data_per_element_;
    hnsw_vector_pointer_used_bytes += elements * sizeof(void*);
    hnsw_vector_pointer_backing_bytes += hnsw.max_elements_ * sizeof(void*);
    hnsw_upper_link_pointer_used_bytes += elements * sizeof(char*);
    hnsw_upper_link_pointer_backing_bytes +=
        hnsw.max_elements_ * sizeof(char*);
    for (uint64_t row = 0; row < elements; ++row) {
      const int level = hnsw.element_levels_[row];
      require(level >= 0, "SIEVE HNSW element level is negative");
      if (level > 0)
        hnsw_upper_link_backing_bytes +=
            hnsw.size_links_per_element_ * uint64_t(level) + 1;
    }
    hnsw_element_level_logical_bytes +=
        hnsw.element_levels_.size() * sizeof(int);
    hnsw_element_level_capacity_bytes +=
        hnsw.element_levels_.capacity() * sizeof(int);
    hnsw_label_lookup_entries += hnsw.label_lookup_.size();
    hnsw_deleted_entries += hnsw.deleted_elements.size();
    child_pointer_logical_bytes +=
        node->_children.size() * sizeof(decltype(node));
    child_pointer_capacity_bytes +=
        node->_children.capacity() * sizeof(decltype(node));
  };
  inventory_node(index._root);
  for (const auto* node : index._nodes) inventory_node(node);

  const uint64_t filter_csr_offset_bytes =
      (filters.n_points + 1) * sizeof(uint64_t);
  const uint64_t filter_csr_index_bytes =
      filters.n_nonzero * sizeof(uint32_t);
  std::cout
      << "SIEVE_LOGICAL_BYTE_LEDGER"
      << " status=partial_unknown_not_claimed"
      << " total_logical_bytes=unknown_not_claimed"
      << " serving_vector_payload_bytes=" << engine.d.base.size()
      << " serving_vector_payload_count=1"
      << " retained_metadata_adapter_bytes=" << engine.metadata.size()
      << " retained_metadata_adapter_online_reachable=false"
      << " filter_csr_offset_bytes=" << filter_csr_offset_bytes
      << " filter_csr_index_bytes=" << filter_csr_index_bytes
      << " filter_single_bitmap_portable_serialized_bytes="
      << single_filter_bitmap_serialized_bytes
      << " filter_base_bitmap_portable_serialized_bytes="
      << base_bitmap_serialized_bytes
      << " filter_multifilter_bitmap_portable_serialized_bytes="
      << multifilter_bitmap_serialized_bytes
      << " filter_single_bitmap_count=" << filters.filter_bvs.size()
      << " filter_multifilter_bitmap_count=" << filters.multifilters_bvs.size()
      << " hnsw_instance_count=" << hnsw_instances
      << " hnsw_element_count=" << hnsw_elements
      << " hnsw_level0_used_bytes=" << hnsw_level0_used_bytes
      << " hnsw_vector_pointer_used_bytes="
      << hnsw_vector_pointer_used_bytes
      << " hnsw_upper_link_backing_bytes=" << hnsw_upper_link_backing_bytes
      << " hnsw_upper_link_pointer_used_bytes="
      << hnsw_upper_link_pointer_used_bytes
      << " hnsw_element_level_logical_bytes="
      << hnsw_element_level_logical_bytes
      << " graph_child_pointer_logical_bytes=" << child_pointer_logical_bytes
      << " graph_node_count=" << seen_nodes.size()
      << " graph_node_map_entries=" << index._node_map.size()
      << " hnsw_label_lookup_entries=" << hnsw_label_lookup_entries
      << " hnsw_deleted_entries=" << hnsw_deleted_entries
      << " dynamic_roaring_heap_bytes=unknown_not_claimed"
      << " hash_node_and_bucket_bytes=unknown_not_claimed"
      << " visited_list_pool_bytes=unknown_not_claimed"
      << std::endl;
  std::cout
      << "SIEVE_CONTAINER_BYTE_RECEIPT"
      << " total_container_bytes=unknown_not_claimed"
      << " metadata_vector_size_bytes=" << engine.metadata.size()
      << " metadata_vector_capacity_bytes=" << engine.metadata.capacity()
      << " base_vector_size_bytes=" << engine.d.base.size()
      << " base_vector_capacity_bytes=" << engine.d.base.capacity()
      << " filter_pointer_vector_size_bytes="
      << filters.filter_bvs.size() * sizeof(roaring::Roaring*)
      << " filter_pointer_vector_capacity_bytes="
      << filters.filter_bvs.capacity() * sizeof(roaring::Roaring*)
      << " index_node_pointer_vector_size_bytes="
      << index._nodes.size() * sizeof(index._nodes[0])
      << " index_node_pointer_vector_capacity_bytes="
      << index._nodes.capacity() * sizeof(index._nodes[0])
      << " hnsw_level0_backing_bytes=" << hnsw_level0_backing_bytes
      << " hnsw_vector_pointer_backing_bytes="
      << hnsw_vector_pointer_backing_bytes
      << " hnsw_upper_link_pointer_backing_bytes="
      << hnsw_upper_link_pointer_backing_bytes
      << " hnsw_element_level_capacity_bytes="
      << hnsw_element_level_capacity_bytes
      << " graph_child_pointer_capacity_bytes="
      << child_pointer_capacity_bytes
      << " unordered_container_backing_bytes=unknown_not_claimed"
      << " roaring_in_memory_container_bytes=unknown_not_claimed"
      << " allocator_padding_bytes=unknown_not_claimed"
      << std::endl;
}

void emit_precalibration(const Options& options) {
#ifdef __AVX2__
  constexpr bool avx2 = true;
#else
  constexpr bool avx2 = false;
#endif
#ifdef __OPTIMIZE__
  constexpr bool optimized = true;
#else
  constexpr bool optimized = false;
#endif
  require(__cplusplus >= 201703L && avx2 && optimized,
          "runner requires C++17, optimization, and AVX2");
  std::cout << "PRECALIBRATION_RECEIPT phase=" << options.phase
            << " system=" << options.system
#ifdef SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE
            << " source_files=layout_selector_x3.cpp,layout_selector.cpp,mixed_runner.cpp,predicate_cube.hpp,pair_major_layout.hpp"
#else
            << " source_files=layout_selector.cpp,mixed_runner.cpp,predicate_cube.hpp,pair_major_layout.hpp"
#endif
            << " compiler=" << __VERSION__
            << " cplusplus=" << __cplusplus
            << " required_compile_flags=-std=c++17,-O3,-march=native,-mavx2"
            << " avx2=true optimization_enabled=true"
            << " binary_path=/proc/self/exe"
            << " source_frozen_before_calibration=true"
            << " evaluation_option_accepted="
            << (options.phase == "execute" ? "true" : "false")
            << std::endl;
}

void emit_layout(const std::string& package, uint64_t common,
                 uint64_t base, uint64_t residual, uint64_t budget,
                 bool canonical_serving, bool reordered_serving,
                 const std::string& ledger, const std::string& reads) {
  const uint64_t total = common + base + residual;
  require(total <= budget, "package exceeds the absolute logical budget");
  std::cout << "LAYOUT_RECEIPT package=" << package
            << " common_catalog_bytes=" << common
            << " base_layout_bytes=" << base
            << " residual_bytes=" << residual
            << " total_logical_bytes=" << total
            << " budget_bytes=" << budget
            << " vector_payload_count=1"
            << " canonical_payload_retained="
            << (canonical_serving ? "true" : "false")
            << " reordered_payload_retained="
            << (reordered_serving ? "true" : "false")
            << " validation_canonical_payload_retained=true"
            << " budget_pass=true" << std::endl;
  std::cout << "ONLINE_STATE_LEDGER package=" << package
            << " ledger=" << ledger << " route_reads=" << reads
            << " all_route_state_accounted=true"
            << " construction_validation_state_online_unreachable=true"
            << std::endl;
}

int run_main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  emit_precalibration(options);
  Data data = load_data(options.base, options.metadata, options.query,
                        options.attribute);
  uint64_t posting_rows = 0;
  const uint64_t common = support_catalog_bytes(data, &posting_rows);
  require(common == 3500000 && posting_rows == 800000,
          "common support catalog charge differs from the frozen contract");
  std::cout << "SUPPORT_CATALOG_RECEIPT common_catalog_bytes=" << common
            << " label_bytes=300000 posting_rows=" << posting_rows
            << " posting_bytes=" << posting_rows * 4
            << " accounting=size_times_element_width"
            << " allocator_overhead_in_rss_only=true" << std::endl;

  const Workload development =
      load_workload(data, options.development_workload_tsv, false);
  Workload evaluation;
  if (options.phase == "execute") {
    evaluation = load_workload(data, options.evaluation_workload_tsv, true);
    require_split_disjoint(development, evaluation);
  }
  const std::vector<Query>& timed = options.phase == "select"
                                          ? development.queries
                                          : evaluation.queries;
  SelectorMetrics metrics;
  if (options.system == "C") {
    require(VECTOR_BYTES + common < options.budget_bytes,
            "C has no residual budget");
    CEngine engine(data, options.budget_bytes - VECTOR_BYTES - common,
                   development.queries);
    emit_manifest("C", engine.engine());
    emit_layout("C", common, VECTOR_BYTES, engine.residual_bytes(),
                options.budget_bytes, true, false,
                "canonical_vectors,labels,A_postings,B_postings,pair_postings,bin_postings,dyadic_postings,materialized_ids,packed_vectors,fragment_manifest",
                "canonical_vectors,A_postings,B_postings,dyadic_postings,materialized_ids,packed_vectors,fragment_manifest");
    auto query = [&](const Query& q) { return engine.query(q); };
    metrics = execute_workload(data, timed, options, true, query);
  } else if (options.system == "P") {
    require(P_BASE_BYTES + common < options.budget_bytes,
            "P has no residual budget");
    PEngine engine(data, options.budget_bytes - P_BASE_BYTES - common,
                   development.queries);
    emit_manifest("P", engine.engine());
    emit_layout("P", common, P_BASE_BYTES, engine.residual_bytes(),
                options.budget_bytes, false, true,
                "pair_major_vectors,pair_offsets,pos_to_global,global_to_pos,labels,A_postings,B_postings,pair_postings,bin_postings,dyadic_postings,range_materialized_ids,range_packed_vectors,range_manifest",
                "pair_major_vectors,pair_offsets,pos_to_global,global_to_pos,dyadic_postings,range_materialized_ids,range_packed_vectors,range_manifest");
    auto query = [&](const Query& q) { return engine.query(q); };
    metrics = execute_workload(data, timed, options, true, query);
  } else if (options.system == "X") {
    PredicateCube cube(data.base, data.bin, data.cat_a, data.cat_b, D, BINS,
                       CAT_A, CAT_B);
    require(cube.logical_charge().absolute_serving_bytes() == X_BASE_BYTES,
            "X base charge differs from frozen contract");
    emit_layout("X", common, X_BASE_BYTES, 0, options.budget_bytes, false,
                true,
#ifdef SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE
                "cube_vectors,cube_offsets,pos_to_global,global_to_pos,labels,A_postings,B_postings,pair_postings,bin_postings,dyadic_postings,compact_prefix64_stack_scratch",
                "cube_vectors,cube_offsets,pos_to_global,compact_prefix64_stack_scratch");
#else
                "cube_vectors,cube_offsets,pos_to_global,global_to_pos,labels,A_postings,B_postings,pair_postings,bin_postings,dyadic_postings",
                "cube_vectors,cube_offsets,pos_to_global");
#endif
    std::cout << "LAYOUT_INVARIANT package=X order=bin,A,B,global_id"
              << " cells=" << cube.cells()
              << " audit_pass=" << (cube.audit().passed() ? "true" : "false")
              << " range_segments=1 equality_segments=16"
              << " conjunction_segments=16 dnf_segments=32" << std::endl;
#ifdef SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE
    static_assert(X_BASE_BYTES + 3500000 +
                          X3_TRANSIENT_SCRATCH_UPPER_BOUND_BYTES <=
                      ABSOLUTE_BUDGET,
                  "X3 layout plus transient scratch exceeds 24 MiB");
    std::cout
        << "X3_KERNEL_RECEIPT range_plan=exact_prefix64_compact_suffix"
        << " seed_rows=" << X3_SEED_ROWS
        << " batch_rows=" << X3_BATCH_ROWS
        << " transient_scratch_upper_bound_bytes="
        << X3_TRANSIENT_SCRATCH_UPPER_BOUND_BYTES
        << " total_including_transient_upper_bound_bytes="
        << X_BASE_BYTES + common + X3_TRANSIENT_SCRATCH_UPPER_BOUND_BYTES
        << " budget_bytes=" << options.budget_bytes
        << " range_exact_fallback=complete_cube_support"
        << " equality_conjunction_dnf_route=frozen_X"
        << " official_baseline_core_modified=false" << std::endl;
    X3Engine engine(cube);
#else
    XEngine engine(cube);
#endif
    auto query = [&](const Query& q) { return engine.query(q); };
    metrics = execute_workload(data, timed, options, true, query);
  } else {
    SieveEngine engine(data, development.queries);
    engine.index->setEf(options.sieve_ef);
    std::cout << "SIEVE_PARAMETER_RECEIPT ef_search=" << options.sieve_ef
              << " ef_grid_member="
              << ((options.sieve_ef >= 100 && options.sieve_ef <= 200 &&
                   options.sieve_ef % 20 == 0)
                      ? "true" : "false")
              << " official_core_modified=false"
              << " metadata_bytes=" << engine.metadata.size()
              << " vector_payload_bytes=" << VECTOR_BYTES
              << " common_budget_comparator=false" << std::endl;
    auto query = [&](const Query& q) {
      return ordered_sieve_result(engine.query(q));
    };
    metrics = execute_workload(data, timed, options, false, query);
    const double recall =
        double(metrics.recall_hits) / metrics.recall_denominator;
    if (options.phase == "execute")
      require(recall >= .995, "formal SIEVE pooled recall is below 0.995");
    emit_sieve_resource_ledger(engine);
  }
  const double recall =
      double(metrics.recall_hits) / metrics.recall_denominator;
  std::cout << "PROCESS_RSS_RECEIPT system=" << options.system
            << " peak_rss_bytes=" << peak_rss_bytes()
            << " logical_bytes_separate_from_rss=true" << std::endl;
  std::cout << "RUN_RECEIPT phase=" << options.phase
            << " system=" << options.system
            << " cycles=" << options.cycles
            << " family_rows=100"
            << " queries=" << metrics.queries
            << " elapsed_ns=" << metrics.elapsed_ns
            << " qps=" << std::setprecision(17) << metrics.qps()
            << " recall=" << recall
            << " exact_required="
            << (options.system == "SIEVE" ? "false" : "true")
            << " validation_after_all_timing=true"
            << " warm_passes=1"
            << " family_order=" << options.family_order
            << " evaluation_path_recorded="
            << (options.phase == "execute" ? "true" : "false")
            << std::endl;
  return 0;
}

}  // namespace

#ifndef SIFT100K_LAYOUT_SELECTOR_V1_NO_MAIN
int main(int argc, char** argv) {
  try {
    return run_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
#endif
