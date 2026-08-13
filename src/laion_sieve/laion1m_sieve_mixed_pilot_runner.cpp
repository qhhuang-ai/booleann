#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include "partitioned_hnsw.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t N = 1000000;
constexpr uint32_t D = 512;
constexpr uint32_t K = 10;
constexpr uint32_t TOKENS = 200;
constexpr uint32_t ATTRIBUTES = 3;
constexpr uint32_t TOTAL_BINS = 564;
constexpr uint32_t LABELS = TOKENS + TOKENS * TOTAL_BINS;
constexpr uint64_t EXPECTED_MEMBERSHIPS = 5015846;

enum class Family : uint32_t { Equality, Conjunction, Range, Dnf2 };

struct Interval {
  uint16_t attribute = 0;
  uint16_t lo = 0;
  uint16_t hi = 0;
};

struct Spec {
  Family family{};
  uint32_t query_base_id = 0;
  uint16_t primary = 0;
  uint16_t secondary = 0;
  Interval first{};
  Interval second{};
  uint32_t expected_support = 0;
};

struct Result {
  std::array<uint32_t, K> ids{};
  uint32_t size = 0;
  uint32_t removed_self = 0;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

class Mapping {
 public:
  explicit Mapping(const std::string& path) {
    descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(descriptor_ >= 0, "open failed: " + path);
    struct stat state {};
    require(fstat(descriptor_, &state) == 0 && state.st_size > 0,
            "stat failed: " + path);
    size_ = static_cast<size_t>(state.st_size);
    address_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
    require(address_ != MAP_FAILED, "mmap failed: " + path);
  }
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  ~Mapping() {
    if (address_ != MAP_FAILED) munmap(address_, size_);
    if (descriptor_ >= 0) close(descriptor_);
  }
  const uint8_t* bytes() const { return static_cast<const uint8_t*>(address_); }
  size_t size() const { return size_; }

 private:
  int descriptor_ = -1;
  void* address_ = MAP_FAILED;
  size_t size_ = 0;
};

template <class T>
T scalar(const uint8_t* source) {
  T value{};
  std::memcpy(&value, source, sizeof(T));
  return value;
}

size_t value_position(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const size_t found = text.find(needle);
  require(found != std::string::npos, "missing JSON field: " + key);
  size_t value = found + needle.size();
  while (value < text.size() &&
         std::isspace(static_cast<unsigned char>(text[value]))) ++value;
  return value;
}

int64_t json_integer(const std::string& text, const std::string& key) {
  size_t at = value_position(text, key);
  size_t end = at;
  if (text[end] == '-') ++end;
  while (end < text.size() && text[end] >= '0' && text[end] <= '9') ++end;
  require(end > at, "noninteger JSON field: " + key);
  return std::stoll(text.substr(at, end - at));
}

std::string json_string(const std::string& text, const std::string& key) {
  size_t at = value_position(text, key);
  require(at < text.size() && text[at] == '"', "nonstring JSON field: " + key);
  const size_t end = text.find('"', at + 1);
  require(end != std::string::npos, "unterminated JSON string: " + key);
  return text.substr(at + 1, end - at - 1);
}

std::string json_object(const std::string& text, const std::string& key) {
  size_t at = value_position(text, key);
  require(at < text.size() && text[at] == '{', "nonobject JSON field: " + key);
  size_t depth = 0;
  bool quoted = false;
  for (size_t end = at; end < text.size(); ++end) {
    if (text[end] == '"' && (end == 0 || text[end - 1] != '\\')) quoted = !quoted;
    if (quoted) continue;
    if (text[end] == '{') ++depth;
    if (text[end] == '}' && --depth == 0) return text.substr(at, end - at + 1);
  }
  fail("unterminated JSON object: " + key);
}

std::vector<double> json_number_array(const std::string& object,
                                      const std::string& key) {
  size_t at = value_position(object, key);
  require(at < object.size() && object[at] == '[', "nonarray JSON field: " + key);
  const size_t end = object.find(']', at + 1);
  require(end != std::string::npos, "unterminated JSON array: " + key);
  std::vector<double> values;
  std::string token;
  std::istringstream input(object.substr(at + 1, end - at - 1));
  while (std::getline(input, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) { return std::isspace(c); }),
                token.end());
    if (token == "\"-inf\"") values.push_back(-std::numeric_limits<double>::infinity());
    else if (token == "\"+inf\"") values.push_back(std::numeric_limits<double>::infinity());
    else values.push_back(std::stod(token));
  }
  return values;
}

std::string read_text(const std::string& path) {
  std::ifstream input(path);
  require(bool(input), "open failed: " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

uint16_t attribute_id(const std::string& name) {
  if (name == "similarity") return 0;
  if (name == "original_width") return 1;
  if (name == "original_height") return 2;
  fail("unknown numeric attribute: " + name);
}

const char* family_name(Family family) {
  static constexpr const char* names[] = {
      "equality", "conjunction", "range", "dnf2"};
  return names[static_cast<uint32_t>(family)];
}

Family parse_family(const std::string& name) {
  if (name == "equality") return Family::Equality;
  if (name == "conjunction") return Family::Conjunction;
  if (name == "range") return Family::Range;
  if (name == "dnf2") return Family::Dnf2;
  fail("unknown family: " + name);
}

Interval parse_interval(const std::string& line, const std::string& key) {
  const std::string object = json_object(line, key);
  Interval interval;
  interval.attribute = attribute_id(json_string(object, "attribute"));
  interval.lo = static_cast<uint16_t>(json_integer(object, "lo_endpoint_bin"));
  interval.hi = static_cast<uint16_t>(json_integer(object, "hi_endpoint_bin"));
  require(interval.lo < interval.hi, "empty interval in workload");
  return interval;
}

Spec parse_spec(const std::string& line) {
  Spec spec;
  spec.family = parse_family(json_string(line, "family"));
  spec.query_base_id = static_cast<uint32_t>(json_integer(line, "query_base_id"));
  spec.primary = static_cast<uint16_t>(json_integer(line, "primary_token_id"));
  spec.expected_support =
      static_cast<uint32_t>(json_integer(line, "support_after_new_leave_one_out"));
  require(spec.query_base_id < N && spec.primary < TOKENS,
          "workload ID outside LAION1M domain");
  if (spec.family == Family::Conjunction || spec.family == Family::Dnf2) {
    spec.secondary = static_cast<uint16_t>(json_integer(line, "secondary_token_id"));
    require(spec.secondary < TOKENS && spec.secondary != spec.primary,
            "invalid secondary token");
  }
  if (spec.family == Family::Range || spec.family == Family::Dnf2)
    spec.first = parse_interval(line, "interval_1");
  if (spec.family == Family::Dnf2)
    spec.second = parse_interval(line, "interval_2");
  return spec;
}

std::vector<Spec> read_workload(const std::string& path, uint32_t per_family) {
  std::ifstream input(path);
  require(bool(input), "open failed: " + path);
  std::array<uint32_t, 4> counts{};
  std::vector<Spec> result;
  std::string line;
  while (std::getline(input, line)) {
    require(!line.empty(), "blank workload row");
    Spec spec = parse_spec(line);
    const uint32_t family = static_cast<uint32_t>(spec.family);
    if (per_family == 0 || counts[family] < per_family) {
      result.push_back(spec);
      ++counts[family];
    }
  }
  if (per_family == 0) {
    require(!result.empty(), "empty workload");
  } else {
    for (uint32_t family = 0; family < 4; ++family)
      require(counts[family] == per_family,
              "workload family coverage differs");
  }
  return result;
}

struct NumericColumn {
  std::vector<double> values;
  std::vector<uint8_t> valid;
  std::vector<double> edges;
  std::vector<int16_t> bins;
};

NumericColumn read_numeric(const std::string& path, bool floating,
                           std::vector<double> edges) {
  Mapping mapping(path);
  require(mapping.size() == 9000000, "numeric extent differs: " + path);
  NumericColumn column;
  column.values.resize(N);
  column.valid.resize(N);
  column.edges = std::move(edges);
  require(column.edges.size() >= 2 && column.edges.size() <= 257 &&
              std::isinf(column.edges.front()) && column.edges.front() < 0 &&
              std::isinf(column.edges.back()) && column.edges.back() > 0 &&
              std::is_sorted(column.edges.begin(), column.edges.end()),
          "endpoint edge contract differs");
  for (uint32_t row = 0; row < N; ++row) {
    column.values[row] = floating
        ? scalar<double>(mapping.bytes() + uint64_t(row) * 8)
        : static_cast<double>(scalar<int64_t>(mapping.bytes() + uint64_t(row) * 8));
    column.valid[row] = mapping.bytes()[8000000ULL + row];
    require(column.valid[row] <= 1, "numeric validity byte differs");
  }
  column.bins.assign(N, -1);
  for (uint32_t row = 0; row < N; ++row) {
    if (!column.valid[row]) continue;
    const double value = column.values[row];
    require(std::isfinite(value), "valid numeric value is nonfinite");
    const auto begin = column.edges.begin() + 1;
    const auto end = column.edges.end() - 1;
    column.bins[row] = static_cast<int16_t>(std::upper_bound(begin, end, value) - begin);
  }
  return column;
}

struct Data {
  std::unique_ptr<Mapping> base_mapping;
  const float* base = nullptr;
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> indices;
  std::array<std::vector<uint32_t>, TOKENS> postings;
  std::array<NumericColumn, ATTRIBUTES> numeric;
  std::array<uint32_t, ATTRIBUTES + 1> bin_offsets{};
};

std::array<std::vector<double>, ATTRIBUTES> read_endpoint_edges(
    const std::string& path) {
  const std::string root = read_text(path);
  const std::string attributes = json_object(root, "numeric_attributes");
  std::array<std::vector<double>, ATTRIBUTES> result;
  for (const std::string name : {"similarity", "original_width", "original_height"}) {
    const uint16_t id = attribute_id(name);
    result[id] = json_number_array(json_object(attributes, name), "endpoint_edges");
  }
  require(result[0].size() == 257 && result[1].size() == 146 &&
              result[2].size() == 164,
          "endpoint-bin census differs from 256/145/163");
  return result;
}

Data load_data(const std::string& base_path, const std::string& spmat_path,
               const std::array<std::string, ATTRIBUTES>& numeric_paths,
               const std::string& endpoint_path) {
  Data data;
  data.base_mapping = std::make_unique<Mapping>(base_path);
  require(data.base_mapping->size() == 8ULL + uint64_t(N) * D * sizeof(float),
          "base extent differs");
  require(scalar<uint32_t>(data.base_mapping->bytes()) == N &&
              scalar<uint32_t>(data.base_mapping->bytes() + 4) == D,
          "base f32bin header differs");
  data.base = reinterpret_cast<const float*>(data.base_mapping->bytes() + 8);

  Mapping spmat(spmat_path);
  const uint64_t rows = scalar<uint64_t>(spmat.bytes());
  const uint64_t columns = scalar<uint64_t>(spmat.bytes() + 8);
  const uint64_t nonzeros = scalar<uint64_t>(spmat.bytes() + 16);
  require(rows == N && columns >= TOKENS &&
              spmat.size() == 24 + 8 * (rows + 1) + 4 * nonzeros,
          "richer spmat container differs");
  data.offsets.resize(N + 1);
  data.indices.resize(nonzeros);
  std::memcpy(data.offsets.data(), spmat.bytes() + 24, 8 * (N + 1));
  std::memcpy(data.indices.data(), spmat.bytes() + 24 + 8 * (N + 1),
              4 * nonzeros);
  require(data.offsets.front() == 0 && data.offsets.back() == nonzeros,
          "spmat CSR endpoints differ");
  for (uint32_t row = 0; row < N; ++row) {
    require(data.offsets[row] <= data.offsets[row + 1],
            "spmat offsets are not monotone");
    uint32_t previous = 0;
    bool first = true;
    for (uint64_t at = data.offsets[row]; at < data.offsets[row + 1]; ++at) {
      const uint32_t token = data.indices[at];
      require(token < columns && (first || previous < token),
              "spmat row is not sorted unique");
      first = false;
      previous = token;
      if (token < TOKENS) data.postings[token].push_back(row);
    }
  }

  auto edges = read_endpoint_edges(endpoint_path);
  data.numeric[0] = read_numeric(numeric_paths[0], true, std::move(edges[0]));
  data.numeric[1] = read_numeric(numeric_paths[1], false, std::move(edges[1]));
  data.numeric[2] = read_numeric(numeric_paths[2], false, std::move(edges[2]));
  for (uint32_t attribute = 0; attribute < ATTRIBUTES; ++attribute)
    data.bin_offsets[attribute + 1] = data.bin_offsets[attribute] +
        static_cast<uint32_t>(data.numeric[attribute].edges.size() - 1);
  require(data.bin_offsets.back() == TOTAL_BINS, "endpoint-bin total differs");
  std::cout << "DATA_READY n=" << N << " dim=" << D
            << " token_atoms=" << TOKENS << " endpoint_bins=" << TOTAL_BINS
            << " labels=" << LABELS << std::endl;
  return data;
}

bool has_token(const Data& data, uint32_t row, uint16_t token) {
  const auto begin = data.indices.begin() + static_cast<ptrdiff_t>(data.offsets[row]);
  const auto end = data.indices.begin() + static_cast<ptrdiff_t>(data.offsets[row + 1]);
  return std::binary_search(begin, end, uint32_t(token));
}

bool interval_matches(const Data& data, uint32_t row, const Interval& interval) {
  const int16_t bin = data.numeric[interval.attribute].bins[row];
  return bin >= 0 && uint16_t(bin) >= interval.lo && uint16_t(bin) < interval.hi;
}

bool matches(const Data& data, const Spec& spec, uint32_t row) {
  if (row >= N) return false;
  if (spec.family == Family::Equality) return has_token(data, row, spec.primary);
  if (spec.family == Family::Conjunction)
    return has_token(data, row, spec.primary) && has_token(data, row, spec.secondary);
  if (spec.family == Family::Range)
    return has_token(data, row, spec.primary) && interval_matches(data, row, spec.first);
  return (has_token(data, row, spec.primary) && interval_matches(data, row, spec.first)) ||
         (has_token(data, row, spec.secondary) && interval_matches(data, row, spec.second));
}

uint32_t cross_label(const Data& data, uint16_t token, const Interval& interval,
                     uint16_t bin) {
  require(bin < data.numeric[interval.attribute].edges.size() - 1,
          "cross-label bin outside endpoint library");
  return TOKENS + uint32_t(token) * TOTAL_BINS +
         data.bin_offsets[interval.attribute] + bin;
}

hnswlib::QueryFilter query_filter(const Data& data, const Spec& spec) {
  std::unordered_set<int32_t> labels;
  if (spec.family == Family::Equality) {
    labels.insert(spec.primary);
    return hnswlib::QueryFilter(std::move(labels), true);
  }
  if (spec.family == Family::Conjunction) {
    labels.insert(spec.primary);
    labels.insert(spec.secondary);
    return hnswlib::QueryFilter(std::move(labels), true);
  }
  auto add = [&](uint16_t token, const Interval& interval) {
    require(interval.hi <= data.numeric[interval.attribute].edges.size() - 1,
            "query interval outside endpoint library");
    for (uint16_t bin = interval.lo; bin < interval.hi; ++bin)
      labels.insert(static_cast<int32_t>(cross_label(data, token, interval, bin)));
  };
  add(spec.primary, spec.first);
  if (spec.family == Family::Dnf2) add(spec.secondary, spec.second);
  return hnswlib::QueryFilter(std::move(labels), false);
}

std::vector<uint32_t> exact_support(const Data& data, const Spec& spec) {
  std::vector<uint32_t> support;
  if (spec.family == Family::Equality) {
    support = data.postings[spec.primary];
  } else if (spec.family == Family::Conjunction) {
    const auto& left = data.postings[spec.primary];
    const auto& right = data.postings[spec.secondary];
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::back_inserter(support));
  } else {
    auto branch = [&](uint16_t token, const Interval& interval) {
      std::vector<uint32_t> out;
      for (uint32_t row : data.postings[token])
        if (interval_matches(data, row, interval)) out.push_back(row);
      return out;
    };
    support = branch(spec.primary, spec.first);
    if (spec.family == Family::Dnf2) {
      const auto second = branch(spec.secondary, spec.second);
      std::vector<uint32_t> merged;
      std::set_union(support.begin(), support.end(), second.begin(), second.end(),
                     std::back_inserter(merged));
      support = std::move(merged);
    }
  }
  const auto self = std::lower_bound(support.begin(), support.end(), spec.query_base_id);
  require(self != support.end() && *self == spec.query_base_id,
          "leave-one-out identity is absent from exact support");
  support.erase(self);
  require(support.size() == spec.expected_support && support.size() >= K,
          "post-LOO support cardinality differs from workload");
  return support;
}

uint64_t current_rss_kib(bool peak) {
  std::ifstream input("/proc/self/status");
  const std::string prefix = peak ? "VmHWM:" : "VmRSS:";
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind(prefix, 0) == 0) {
      std::istringstream fields(line.substr(prefix.size()));
      uint64_t value = 0;
      fields >> value;
      return value;
    }
  }
  return 0;
}

struct MetadataFile {
  FILE* file = nullptr;
  uint64_t memberships = 0;
  uint64_t bytes = 0;
};

MetadataFile build_metadata(const Data& data) {
  std::vector<uint64_t> offsets(N + 1);
  for (uint32_t row = 0; row < N; ++row) {
    uint64_t tokens = 0;
    for (uint64_t at = data.offsets[row]; at < data.offsets[row + 1]; ++at)
      tokens += data.indices[at] < TOKENS;
    uint64_t valid_attributes = 0;
    for (const auto& column : data.numeric) valid_attributes += column.bins[row] >= 0;
    offsets[row + 1] = offsets[row] + tokens * (1 + valid_attributes);
  }
  require(offsets.back() == EXPECTED_MEMBERSHIPS,
          "expanded metadata membership census differs");
  std::vector<uint32_t> indices(offsets.back());
  for (uint32_t row = 0; row < N; ++row) {
    uint64_t out = offsets[row];
    for (uint64_t at = data.offsets[row]; at < data.offsets[row + 1]; ++at)
      if (data.indices[at] < TOKENS) indices[out++] = data.indices[at];
    for (uint64_t at = data.offsets[row]; at < data.offsets[row + 1]; ++at) {
      const uint32_t token = data.indices[at];
      if (token >= TOKENS) break;
      for (uint16_t attribute = 0; attribute < ATTRIBUTES; ++attribute) {
        const int16_t bin = data.numeric[attribute].bins[row];
        if (bin < 0) continue;
        Interval interval{attribute, 0, 1};
        indices[out++] = cross_label(data, static_cast<uint16_t>(token), interval,
                                     static_cast<uint16_t>(bin));
      }
    }
    require(out == offsets[row + 1] &&
                std::is_sorted(indices.begin() + static_cast<ptrdiff_t>(offsets[row]),
                               indices.begin() + static_cast<ptrdiff_t>(offsets[row + 1])),
            "expanded metadata row differs");
  }
  FILE* file = std::tmpfile();
  require(file != nullptr, "tmpfile failed for SIEVE metadata");
  const uint64_t header[3] = {N, LABELS, offsets.back()};
  require(std::fwrite(header, sizeof(uint64_t), 3, file) == 3 &&
              std::fwrite(offsets.data(), sizeof(uint64_t), offsets.size(), file) == offsets.size() &&
              std::fwrite(indices.data(), sizeof(uint32_t), indices.size(), file) == indices.size() &&
              std::fflush(file) == 0 && std::fseek(file, 0, SEEK_SET) == 0,
          "writing SIEVE metadata failed");
  return {file, offsets.back(), 24 + 8 * offsets.size() + 4 * indices.size()};
}

class SieveEngine {
 public:
  SieveEngine(const Data& data, const std::vector<Spec>& history, size_t m,
              size_t ef_construction, size_t ef_search, size_t vector_budget,
              size_t cutoff, size_t threads)
      : data_(data) {
    MetadataFile metadata = build_metadata(data);
    metadata_bytes = metadata.bytes;
    memberships = metadata.memberships;
    filters = std::make_unique<hnswlib::DatasetFilters>(metadata.file, threads, false);
    filters->transpose_inplace();
    filters->make_bvs();
    space = std::make_unique<hnswlib::L2Space>(D);
    std::vector<hnswlib::QueryFilter> workload;
    workload.reserve(history.size());
    for (const Spec& spec : history) workload.push_back(query_filter(data, spec));
    hnswlib::PartitionedIndexParams parameters{};
    parameters.dataset_size = N;
    parameters.dim = D;
    parameters.M = m;
    parameters.ef_construction = ef_construction;
    parameters.index_vector_budget = vector_budget;
    parameters.bitvector_cutoff = cutoff;
    parameters.historical_workload_window_size = workload.size();
    parameters.enable_heterogeneous_indexing = true;
    parameters.enable_heterogeneous_search = true;
    parameters.query_correlation_constant = 0.5f;
    parameters.num_threads = threads;
    parameters.ef_search_scaling_constant = 3.0f;
    parameters.enable_multipartition_search = true;
    std::cout << "SIEVE_BUILD_START history=" << workload.size()
              << " labels=" << LABELS << " memberships=" << memberships
              << " M=" << m << " ef_construction=" << ef_construction
              << " ef_search=" << ef_search << " index_vector_budget="
              << vector_budget << " bitvector_cutoff=" << cutoff
              << " threads=" << threads << " baseline_core_modified=false"
              << std::endl;
    index = std::make_unique<hnswlib::PartitionedHNSW<float, float>>(
        const_cast<float*>(data.base), space.get(), filters.get(), parameters,
        workload);
    index->setEf(ef_search);
    std::cout << "SIEVE_BUILD_DONE current_rss_kib=" << current_rss_kib(false)
              << " peak_rss_kib=" << current_rss_kib(true) << std::endl;
  }

  void prewarm(const Spec& spec) {
    hnswlib::Predicate predicate(filters.get(), query_filter(data_, spec));
    require(predicate.cardinality() == uint64_t(spec.expected_support) + 1,
            "SIEVE filter cardinality differs from independent support");
    // The local-graph and SIEVE arms use the same steady-state protocol:
    // execute every timed request once before the forked measurement.  This
    // prevents route-specific vector/index page warmth from biasing the batch.
    (void)query(spec);
  }

  Result query(const Spec& spec) {
    hnswlib::Predicate predicate(filters.get(), query_filter(data_, spec));
    auto heap = index->searchKnn(data_.base + uint64_t(spec.query_base_id) * D,
                                 K + 1, predicate);
    std::vector<std::pair<float, uint32_t>> hits;
    Result result;
    while (!heap.empty()) {
      const auto hit = heap.top();
      heap.pop();
      if (hit.second == spec.query_base_id) {
        ++result.removed_self;
      } else {
        hits.emplace_back(hit.first, static_cast<uint32_t>(hit.second));
      }
    }
    std::sort(hits.begin(), hits.end());
    result.size = std::min<uint32_t>(K, hits.size());
    result.ids.fill(UINT32_MAX);
    for (uint32_t i = 0; i < result.size; ++i) result.ids[i] = hits[i].second;
    return result;
  }

  const Data& data_;
  uint64_t metadata_bytes = 0;
  uint64_t memberships = 0;
  std::unique_ptr<hnswlib::DatasetFilters> filters;
  std::unique_ptr<hnswlib::L2Space> space;
  std::unique_ptr<hnswlib::PartitionedHNSW<float, float>> index;
};

Result exact_top10(const Data& data, hnswlib::L2Space& space, const Spec& spec,
                   const std::vector<uint32_t>& support) {
  using Hit = std::pair<float, uint32_t>;
  std::priority_queue<Hit> heap;
  const auto distance = space.get_dist_func();
  void* parameter = space.get_dist_func_param();
  const float* query = data.base + uint64_t(spec.query_base_id) * D;
  for (uint32_t row : support) {
    const Hit hit{distance(query, data.base + uint64_t(row) * D, parameter), row};
    if (heap.size() < K) heap.push(hit);
    else if (hit < heap.top()) {
      heap.pop();
      heap.push(hit);
    }
  }
  Result result;
  result.size = K;
  for (int32_t i = K - 1; i >= 0; --i) {
    result.ids[static_cast<size_t>(i)] = heap.top().second;
    heap.pop();
  }
  return result;
}

struct Metrics {
  uint64_t elapsed_ns = 0;
  uint64_t queries = 0;
  uint64_t hits = 0;
  uint64_t denominator = 0;
  uint64_t invalid = 0;
  uint64_t predicate_fail = 0;
  uint64_t duplicate = 0;
  uint64_t forbidden_self = 0;
  uint64_t removed_self = 0;
  uint64_t underfull = 0;
  uint64_t nondeterministic = 0;
  double qps() const { return queries * 1e9 / elapsed_ns; }
};

Metrics run_family(const Data& data, SieveEngine& engine,
                   const std::vector<Spec>& queries, Family family,
                   uint32_t cycles) {
  std::vector<const Spec*> selected;
  for (const Spec& spec : queries) if (spec.family == family) selected.push_back(&spec);
  require(!selected.empty(), "empty timed family");
  std::vector<Result> results(selected.size());
  for (const Spec* spec : selected) (void)engine.query(*spec);
  const auto begin = Clock::now();
  for (uint32_t cycle = 0; cycle < cycles; ++cycle)
    for (size_t row = 0; row < selected.size(); ++row)
      results[row] = engine.query(*selected[row]);
  const auto end = Clock::now();
  Metrics metrics;
  metrics.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  metrics.queries = uint64_t(selected.size()) * cycles;
  for (size_t row = 0; row < selected.size(); ++row) {
    const Spec& spec = *selected[row];
    const Result repeat = engine.query(spec);
    if (repeat.ids != results[row].ids || repeat.size != results[row].size)
      ++metrics.nondeterministic;
    const std::vector<uint32_t> support = exact_support(data, spec);
    const Result truth = exact_top10(data, *engine.space, spec, support);
    std::set<uint32_t> truth_ids(truth.ids.begin(), truth.ids.end());
    std::set<uint32_t> observed;
    metrics.denominator += K;
    metrics.removed_self += results[row].removed_self;
    metrics.underfull += results[row].size < K;
    for (uint32_t i = 0; i < results[row].size; ++i) {
      const uint32_t id = results[row].ids[i];
      if (!observed.insert(id).second) ++metrics.duplicate;
      if (id >= N) ++metrics.invalid;
      else {
        if (id == spec.query_base_id) ++metrics.forbidden_self;
        if (!matches(data, spec, id)) ++metrics.predicate_fail;
        if (truth_ids.count(id)) ++metrics.hits;
      }
    }
  }
  return metrics;
}

struct ForkShared {
  std::atomic<uint32_t> next{0};
  std::atomic<uint32_t> ready{0};
  std::atomic<uint32_t> go{0};
  std::atomic<uint32_t> done{0};
  std::array<Result, 800> results{};
  std::array<uint64_t, 800> service_ns{};
  std::array<uint64_t, 8> child_rss_kib{};
  std::array<int32_t, 8> child_status{};
};

int run_forked(const Data& data, SieveEngine& engine,
               const std::vector<Spec>& queries, uint32_t cycles,
               uint32_t lanes) {
  require(queries.size() <= 800 && cycles == 1 && lanes >= 2 && lanes <= 8,
          "forked pilot requires at most 800 queries, one cycle, and 2--8 lanes");
  void* address = mmap(nullptr, sizeof(ForkShared), PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  require(address != MAP_FAILED, "shared result mmap failed");
  auto* shared = new(address) ForkShared();
  static constexpr std::array<int, 8> cpus = {60, 61, 63, 64, 66, 67, 69, 70};
  std::vector<pid_t> children;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const pid_t pid = fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
      int32_t status = 0;
      try {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[lane], &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) status = 3;
        shared->ready.fetch_add(1, std::memory_order_release);
        while (shared->go.load(std::memory_order_acquire) == 0) sched_yield();
        if (status == 0) {
          for (;;) {
            const uint32_t row = shared->next.fetch_add(1, std::memory_order_relaxed);
            if (row >= queries.size()) break;
            const auto begin = Clock::now();
            shared->results[row] = engine.query(queries[row]);
            const auto end = Clock::now();
            shared->service_ns[row] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
          }
        }
      } catch (...) { status = 2; }
      shared->child_rss_kib[lane] = current_rss_kib(false);
      shared->child_status[lane] = status;
      shared->done.fetch_add(1, std::memory_order_release);
      _exit(status);
    }
    children.push_back(pid);
  }
  while (shared->ready.load(std::memory_order_acquire) != lanes) usleep(1000);
  const auto begin = Clock::now();
  shared->go.store(1, std::memory_order_release);
  while (shared->done.load(std::memory_order_acquire) != lanes) usleep(100);
  const auto end = Clock::now();
  const uint64_t batch_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  for (pid_t pid : children) { int state = 0; require(waitpid(pid, &state, 0) == pid,
      "waitpid failed"); require(WIFEXITED(state) && WEXITSTATUS(state) == 0,
      "forked SIEVE worker failed"); }

  std::array<Metrics, 4> families{};
  for (size_t row = 0; row < queries.size(); ++row) {
    const Spec& spec = queries[row]; const Result& result = shared->results[row];
    Metrics& metrics = families[static_cast<uint32_t>(spec.family)];
    ++metrics.queries; metrics.elapsed_ns += shared->service_ns[row];
    const Result repeat = engine.query(spec);
    if (repeat.ids != result.ids || repeat.size != result.size) ++metrics.nondeterministic;
    const std::vector<uint32_t> support = exact_support(data, spec);
    const Result truth = exact_top10(data, *engine.space, spec, support);
    std::set<uint32_t> truth_ids(truth.ids.begin(), truth.ids.end()), observed;
    metrics.denominator += K; metrics.removed_self += result.removed_self;
    metrics.underfull += result.size < K;
    for (uint32_t i = 0; i < result.size; ++i) {
      const uint32_t id = result.ids[i];
      if (!observed.insert(id).second) ++metrics.duplicate;
      if (id >= N) ++metrics.invalid;
      else { if (id == spec.query_base_id) ++metrics.forbidden_self;
        if (!matches(data, spec, id)) ++metrics.predicate_fail;
        if (truth_ids.count(id)) ++metrics.hits; }
    }
  }
  Metrics total;
  for (uint32_t family = 0; family < 4; ++family) {
    const Metrics& value = families[family];
    std::cout << "FORK_BLOCK system=sieve family=" << family_name(Family(family))
              << " queries=" << value.queries
              << " core_service_sum_ns=" << value.elapsed_ns
              << " core_service_qps=" << value.qps()
              << " strict_recall=" << double(value.hits) / value.denominator
              << " invalid=" << value.invalid << " predicate_fail=" << value.predicate_fail
              << " duplicate=" << value.duplicate << " forbidden_self=" << value.forbidden_self
              << " removed_self=" << value.removed_self << " underfull=" << value.underfull
              << " nondeterministic=" << value.nondeterministic << std::endl;
    total.queries += value.queries; total.hits += value.hits;
    total.denominator += value.denominator; total.invalid += value.invalid;
    total.predicate_fail += value.predicate_fail; total.duplicate += value.duplicate;
    total.forbidden_self += value.forbidden_self; total.removed_self += value.removed_self;
    total.underfull += value.underfull; total.nondeterministic += value.nondeterministic;
  }
  uint64_t max_child_rss = 0; for (uint32_t lane=0; lane<lanes; ++lane)
    max_child_rss = std::max(max_child_rss, shared->child_rss_kib[lane]);
  std::cout << "FORK_SUMMARY system=sieve assignment=persistent_shared_pull"
            << " request_order=file_order lanes=" << lanes << " queries=" << total.queries
            << " batch_wall_ns=" << batch_ns
            << " complete_batch_qps=" << double(total.queries) * 1e9 / batch_ns
            << " strict_recall=" << double(total.hits) / total.denominator
            << " invalid=" << total.invalid << " predicate_fail=" << total.predicate_fail
            << " duplicate=" << total.duplicate << " forbidden_self=" << total.forbidden_self
            << " removed_self=" << total.removed_self << " underfull=" << total.underfull
            << " nondeterministic=" << total.nondeterministic
            << " max_child_rss_kib=" << max_child_rss
            << " parent_rss_kib=" << current_rss_kib(false)
            << " metadata_bytes=" << engine.metadata_bytes
            << " memberships=" << engine.memberships << " labels=" << LABELS << std::endl;
  const bool bad = total.invalid || total.predicate_fail || total.duplicate ||
      total.forbidden_self || total.underfull || total.nondeterministic;
  shared->~ForkShared(); munmap(address, sizeof(ForkShared)); return bad ? 2 : 0;
}

struct Options {
  std::string base, spmat, endpoints, history, queries;
  std::array<std::string, ATTRIBUTES> numeric;
  uint32_t history_per_family = 0;
  uint32_t timed_per_family = 0;
  uint32_t cycles = 1;
  size_t m = 16;
  size_t ef_construction = 40;
  size_t ef_search = 200;
  size_t vector_budget = 500000;
  size_t cutoff = 512;
  size_t threads = 8;
  size_t lanes = 1;
};

Options options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    require(i + 1 < argc, "missing option value: " + key);
    const std::string value = argv[++i];
    if (key == "--base") result.base = value;
    else if (key == "--spmat") result.spmat = value;
    else if (key == "--endpoints") result.endpoints = value;
    else if (key == "--history") result.history = value;
    else if (key == "--queries") result.queries = value;
    else if (key == "--numeric-similarity") result.numeric[0] = value;
    else if (key == "--numeric-original-width") result.numeric[1] = value;
    else if (key == "--numeric-original-height") result.numeric[2] = value;
    else if (key == "--history-per-family") result.history_per_family = std::stoul(value);
    else if (key == "--timed-per-family") result.timed_per_family = std::stoul(value);
    else if (key == "--cycles") result.cycles = std::stoul(value);
    else if (key == "--M") result.m = std::stoull(value);
    else if (key == "--ef-construction") result.ef_construction = std::stoull(value);
    else if (key == "--ef-search") result.ef_search = std::stoull(value);
    else if (key == "--index-vector-budget") result.vector_budget = std::stoull(value);
    else if (key == "--bitvector-cutoff") result.cutoff = std::stoull(value);
    else if (key == "--threads") result.threads = std::stoull(value);
    else if (key == "--lanes") result.lanes = std::stoull(value);
    else fail("unknown option: " + key);
  }
  require(!result.base.empty() && !result.spmat.empty() && !result.endpoints.empty() &&
              !result.history.empty() && !result.queries.empty() &&
              std::all_of(result.numeric.begin(), result.numeric.end(),
                          [](const std::string& path) { return !path.empty(); }) &&
              result.cycles > 0 && result.threads > 0 && result.lanes > 0 &&
              result.lanes <= 8 && result.vector_budget >= K,
          "required input/config option missing");
  return result;
}

int main_impl(int argc, char** argv) {
  const Options config = options(argc, argv);
  Data data = load_data(config.base, config.spmat, config.numeric, config.endpoints);
  std::vector<Spec> history = read_workload(config.history, config.history_per_family);
  std::vector<Spec> queries = read_workload(config.queries, config.timed_per_family);
  for (const Spec& spec : history) (void)exact_support(data, spec);
  for (const Spec& spec : queries) (void)exact_support(data, spec);
  std::cout << "WORKLOAD_READY history=" << history.size() << " timed=" << queries.size()
            << " current_rss_kib=" << current_rss_kib(false)
            << " peak_rss_kib=" << current_rss_kib(true) << std::endl;
  SieveEngine engine(data, history, config.m, config.ef_construction,
                     config.ef_search, config.vector_budget, config.cutoff,
                     config.threads);
  for (const Spec& spec : queries) engine.prewarm(spec);
  if (config.lanes > 1)
    return run_forked(data, engine, queries, config.cycles,
                      static_cast<uint32_t>(config.lanes));
  std::array<Metrics, 4> families;
  for (uint32_t family = 0; family < 4; ++family) {
    families[family] = run_family(data, engine, queries, Family(family), config.cycles);
    const Metrics& value = families[family];
    std::cout << "BLOCK system=sieve family=" << family_name(Family(family))
              << " queries=" << value.queries << " elapsed_ns=" << value.elapsed_ns
              << " qps=" << std::setprecision(12) << value.qps()
              << " strict_recall=" << double(value.hits) / value.denominator
              << " invalid=" << value.invalid
              << " predicate_fail=" << value.predicate_fail
              << " duplicate=" << value.duplicate
              << " forbidden_self=" << value.forbidden_self
              << " removed_self=" << value.removed_self
              << " underfull=" << value.underfull
              << " nondeterministic=" << value.nondeterministic
              << " current_rss_kib=" << current_rss_kib(false)
              << " peak_rss_kib=" << current_rss_kib(true) << std::endl;
  }
  Metrics total;
  for (const Metrics& value : families) {
    total.elapsed_ns += value.elapsed_ns;
    total.queries += value.queries;
    total.hits += value.hits;
    total.denominator += value.denominator;
    total.invalid += value.invalid;
    total.predicate_fail += value.predicate_fail;
    total.duplicate += value.duplicate;
    total.forbidden_self += value.forbidden_self;
    total.removed_self += value.removed_self;
    total.underfull += value.underfull;
    total.nondeterministic += value.nondeterministic;
  }
  std::cout << "SUMMARY system=sieve queries=" << total.queries
            << " elapsed_ns=" << total.elapsed_ns << " qps=" << total.qps()
            << " strict_recall=" << double(total.hits) / total.denominator
            << " invalid=" << total.invalid
            << " predicate_fail=" << total.predicate_fail
            << " duplicate=" << total.duplicate
            << " forbidden_self=" << total.forbidden_self
            << " removed_self=" << total.removed_self
            << " underfull=" << total.underfull
            << " nondeterministic=" << total.nondeterministic
            << " metadata_bytes=" << engine.metadata_bytes
            << " memberships=" << engine.memberships
            << " labels=" << LABELS
            << " current_rss_kib=" << current_rss_kib(false)
            << " peak_rss_kib=" << current_rss_kib(true) << std::endl;
  return 0;
}

}  // namespace

#ifndef BOOLEANN_LAION_MIXED_RUNNER_NO_MAIN
int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
#endif
