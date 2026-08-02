// Hierarchical fixed-block executor for SIFT100K range search.
//
// The retained graph/search implementation is adapted only at its data and
// protocol boundary.  Timed regions contain search calls only: no GT, recall,
// telemetry, stdout, or filesystem operations occur between begin/end.

#define main v52h_retained_fixed_block_main_unused
#include "fixed_block_100k_adapter.cpp"
#undef main

#include <openssl/sha.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "block_local_point_range.h"

using bci_block_local::BlockContiguousVectors;
using bci_block_local::BlockLocalPointRange;
using bci_block_local::load_vector_row_offsets;

namespace {

constexpr std::array<int, 8> kBeams{64, 80, 96, 112, 128, 160, 192, 256};
constexpr uint64_t kExpectedGraphBytes = 80'386'488;
constexpr uint64_t kPersistentLimit = 192ULL * 1024ULL * 1024ULL;
constexpr const char* kImplementationVersion = "submission-2026-08-02";

struct SegmentNodeV52H {
  int first_block = 0;
  int end_block = 0;
  uint64_t begin_row = 0;
  size_t rows = 0;
  std::unique_ptr<GraphI> graph;
  long max_degree = 0;
};

struct Options {
  std::string repo_root;
  std::string point_root;
  std::string workload_root;
  std::string dual_gt_root;
  std::string graph_root;
  int fresh_process = 0;
  int paired_block = 0;
  std::string order;
  int position_in_pair = 0;
  int process_ordinal = 0;
  bool synthetic = false;
  std::string synthetic_workload_sha256;
  std::string synthetic_dual_gt_sha256;
};

[[noreturn]] void fail(const std::string& detail) {
  throw std::runtime_error(detail);
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

std::string read_all(const std::string& path, size_t maximum = 128 << 20) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) fail("cannot open regular file " + path);
  struct stat info {};
  if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
      static_cast<uint64_t>(info.st_size) > maximum) {
    ::close(fd);
    fail("invalid regular file " + path);
  }
  std::string payload(static_cast<size_t>(info.st_size), '\0');
  size_t offset = 0;
  while (offset < payload.size()) {
    const ssize_t got = ::read(fd, payload.data() + offset, payload.size() - offset);
    if (got <= 0) {
      ::close(fd);
      fail("short read " + path);
    }
    offset += static_cast<size_t>(got);
  }
  char trailing = 0;
  if (::read(fd, &trailing, 1) != 0) {
    ::close(fd);
    fail("trailing read " + path);
  }
  ::close(fd);
  return payload;
}

std::string sha256_bytes(const void* data, size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(data), size, digest);
  std::ostringstream out;
  for (unsigned char byte : digest)
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(byte);
  return out.str();
}

std::string sha256_string(const std::string& payload) {
  return sha256_bytes(payload.data(), payload.size());
}

std::string sha256_file(const std::string& path, size_t maximum = 128 << 20) {
  return sha256_string(read_all(path, maximum));
}

void fsync_dir(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0 || ::fsync(fd) != 0) {
    if (fd >= 0) ::close(fd);
    fail("directory fsync failed " + path);
  }
  ::close(fd);
}

template <class T>
std::string publish_raw(
    const std::string& root, const std::string& leaf,
    const std::vector<T>& values) {
  const std::string path = root + "/" + leaf;
  const int fd = ::open(
      path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
      0444);
  if (fd < 0) fail("refusing raw replacement " + path);
  const unsigned char* cursor =
      reinterpret_cast<const unsigned char*>(values.data());
  size_t remaining = values.size() * sizeof(T);
  while (remaining) {
    const ssize_t written = ::write(fd, cursor, remaining);
    if (written <= 0) {
      ::close(fd);
      fail("raw write failed " + path);
    }
    cursor += written;
    remaining -= static_cast<size_t>(written);
  }
  if (::fsync(fd) != 0) {
    ::close(fd);
    fail("raw fsync failed " + path);
  }
  ::close(fd);
  const std::string digest = sha256_file(path, values.size() * sizeof(T) + 1);
  fsync_dir(root);
  return digest;
}

std::string publish_text(
    const std::string& root, const std::string& leaf,
    const std::string& payload) {
  const std::string path = root + "/" + leaf;
  const int fd = ::open(
      path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
      0444);
  if (fd < 0) fail("refusing text replacement " + path);
  size_t offset = 0;
  while (offset < payload.size()) {
    const ssize_t written = ::write(fd, payload.data() + offset, payload.size() - offset);
    if (written <= 0) {
      ::close(fd);
      fail("text write failed " + path);
    }
    offset += static_cast<size_t>(written);
  }
  if (::fsync(fd) != 0) {
    ::close(fd);
    fail("text fsync failed " + path);
  }
  ::close(fd);
  if (read_all(path, payload.size() + 1) != payload)
    fail("text reopen differs " + path);
  fsync_dir(root);
  return sha256_string(payload);
}

std::string json_string(const std::string& payload, const std::string& key) {
  const std::string prefix = "\"" + key + "\":\"";
  const size_t begin = payload.find(prefix);
  if (begin == std::string::npos) fail("JSON string key missing " + key);
  const size_t value_begin = begin + prefix.size();
  const size_t end = payload.find('"', value_begin);
  if (end == std::string::npos) fail("JSON string unterminated " + key);
  return payload.substr(value_begin, end - value_begin);
}

long long json_integer(const std::string& payload, const std::string& key) {
  const std::string prefix = "\"" + key + "\":";
  const size_t begin = payload.find(prefix);
  if (begin == std::string::npos) fail("JSON integer key missing " + key);
  const char* start = payload.c_str() + begin + prefix.size();
  char* end = nullptr;
  errno = 0;
  const long long value = std::strtoll(start, &end, 10);
  if (errno || end == start) fail("JSON integer invalid " + key);
  return value;
}

bool has_exact_fragment(const std::string& payload, const std::string& fragment) {
  return payload.find(fragment) != std::string::npos;
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::map<std::string, std::string*> strings{
      {"--repo-root", &options.repo_root},
      {"--point-root", &options.point_root},
      {"--workload-root", &options.workload_root},
      {"--dual-gt-root", &options.dual_gt_root},
      {"--graph-root", &options.graph_root},
      {"--order", &options.order},
      {"--synthetic-workload-sha256", &options.synthetic_workload_sha256},
      {"--synthetic-dual-gt-sha256", &options.synthetic_dual_gt_sha256},
  };
  std::map<std::string, int*> integers{
      {"--fresh-process", &options.fresh_process},
      {"--paired-block", &options.paired_block},
      {"--position-in-pair", &options.position_in_pair},
      {"--process-ordinal", &options.process_ordinal},
  };
  for (int index = 1; index < argc; ++index) {
    const std::string token = argv[index];
    if (token == "--query-free-synthetic") {
      options.synthetic = true;
      continue;
    }
    if (index + 1 >= argc) fail("missing option value " + token);
    if (strings.count(token)) {
      *strings[token] = argv[++index];
    } else if (integers.count(token)) {
      *integers[token] = std::stoi(argv[++index]);
    } else {
      fail("unknown option " + token);
    }
  }
  if (options.point_root.empty() || options.fresh_process < 1 ||
      options.paired_block < 1 || options.position_in_pair < 1 ||
      options.process_ordinal < 1 ||
      (options.order != "AB" && options.order != "BA"))
    fail("incomplete provider coordinate/options");
  if (options.synthetic) {
    const char* allowed = std::getenv("V52H_QUERY_FREE_SYNTHETIC");
    if (!allowed || std::string(allowed) != "1")
      fail("synthetic mode is not enabled");
    if (options.synthetic_workload_sha256.size() != 64 ||
        options.synthetic_dual_gt_sha256.size() != 64)
      fail("Boole synthetic digests missing");
  } else if (options.workload_root.empty() || options.dual_gt_root.empty() ||
             options.graph_root.empty()) {
    fail("required input paths are missing");
  }
  return options;
}

uint64_t segment_key(int first, int end) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(first)) << 32) |
         static_cast<uint32_t>(end);
}

std::vector<std::pair<int, int>> all_nodes() {
  std::vector<std::pair<int, int>> nodes;
  for (int width = 1; width <= 128; width *= 2)
    for (int first = 0; first + width <= kNumBlocks; first += width)
      nodes.emplace_back(first, first + width);
  if (nodes.size() != 397) fail("unexpected hierarchy node count");
  return nodes;
}

std::vector<std::pair<int, int>> cover(int first, int end) {
  std::vector<std::pair<int, int>> result;
  while (first < end) {
    int width = 1;
    while (first % (2 * width) == 0 && first + 2 * width <= end) width *= 2;
    result.emplace_back(first, first + width);
    first += width;
  }
  return result;
}

void finish(
    std::vector<std::pair<float, int32_t>>& merged,
    std::vector<int32_t>& ids,
    std::vector<int64_t>& distances) {
  const int take = std::min<int>(kK, merged.size());
  if (take)
    std::partial_sort(
        merged.begin(), merged.begin() + take, merged.end(),
        [](const auto& left, const auto& right) {
          return left.first < right.first ||
                 (left.first == right.first && left.second < right.second);
        });
  ids.assign(kK, -1);
  distances.assign(kK, std::numeric_limits<int64_t>::max());
  for (int index = 0; index < take; ++index) {
    const double rounded = std::nearbyint(static_cast<double>(merged[index].first));
    if (!std::isfinite(merged[index].first) ||
        std::abs(static_cast<double>(merged[index].first) - rounded) > 0.01)
      fail("Boole nonintegral squared distance");
    ids[index] = merged[index].second;
    distances[index] = static_cast<int64_t>(rounded);
  }
}

class RealBackend {
 public:
  explicit RealBackend(const Options& options)
      : packed_(options.workload_root + "/base_100k.block_contiguous.f32raw", kN, kDim),
        queries_(options.workload_root + "/query_512.f32raw", 512, kDim),
        packed_ids_(load_i32(options.workload_root + "/packed_to_original.i32", kN)),
        offsets_(load_vector_row_offsets(
            options.workload_root + "/block_offsets.u64", kNumBlocks, kN)),
        attrs_(load_i32(options.workload_root + "/attrs_100k.i32", kN)),
        ranges_flat_(load_i32(options.workload_root + "/ranges_512x2.i32", 512 * 2)) {
    ranges_.resize(512);
    for (int row = 0; row < 512; ++row) {
      ranges_[row] = {ranges_flat_[2 * row], ranges_flat_[2 * row + 1]};
      if (ranges_[row].lo < 0 || ranges_[row].hi >= kCardinality ||
          ranges_[row].lo > ranges_[row].hi)
        fail("Boole invalid closed range");
    }
    uint64_t graph_bytes = 0;
    for (const auto [first, end] : all_nodes()) {
      auto node = std::make_unique<SegmentNodeV52H>();
      node->first_block = first;
      node->end_block = end;
      node->begin_row = offsets_[first];
      node->rows = static_cast<size_t>(offsets_[end] - offsets_[first]);
      const std::string path = options.graph_root + "/segment_" +
          std::to_string(first) + "_" + std::to_string(end) + ".bin";
      if (!std::filesystem::is_regular_file(path)) fail("Boole graph missing " + path);
      graph_bytes += std::filesystem::file_size(path);
      node->graph = std::make_unique<GraphI>(const_cast<char*>(path.c_str()));
      node->max_degree = node->graph->max_degree();
      lookup_.emplace(segment_key(first, end), nodes_.size());
      nodes_.push_back(std::move(node));
    }
    if (nodes_.size() != 397 || graph_bytes != kExpectedGraphBytes)
      fail("Boole frozen graph inventory mismatch");
    persistent_bytes_ = graph_bytes;
  }

  uint64_t persistent_bytes() const { return persistent_bytes_; }

  void run(
      int row_count,
      std::vector<int32_t>& flat_ids,
      std::vector<int64_t>& flat_distances,
      std::vector<int64_t>& walls_ns,
      int64_t& latest_end_ns) {
    flat_ids.clear();
    flat_distances.clear();
    walls_ns.clear();
    flat_ids.reserve(kBeams.size() * row_count * kK);
    flat_distances.reserve(kBeams.size() * row_count * kK);
    for (int beam : kBeams) {
      std::vector<std::vector<int32_t>> ids(row_count);
      std::vector<std::vector<int64_t>> distances(row_count);
      const auto begin = std::chrono::steady_clock::now();
      parlay::parallel_for(0, row_count, [&](size_t row) {
        query(static_cast<int>(row), beam, ids[row], distances[row]);
      });
      const auto end = std::chrono::steady_clock::now();
      const int64_t wall = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
      const int64_t end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          end.time_since_epoch()).count();
      if (wall <= 0) fail("Boole nonpositive batch wall");
      walls_ns.push_back(wall);
      latest_end_ns = std::max(latest_end_ns, end_ns);
      for (int row = 0; row < row_count; ++row) {
        flat_ids.insert(flat_ids.end(), ids[row].begin(), ids[row].end());
        flat_distances.insert(
            flat_distances.end(), distances[row].begin(), distances[row].end());
      }
    }
  }

 private:
  SegmentNodeV52H& node(int first, int end) {
    const auto found = lookup_.find(segment_key(first, end));
    if (found == lookup_.end()) fail("Boole hierarchy node absent");
    return *nodes_[found->second];
  }

  void search(
      int qid, int beam, SegmentNodeV52H& active, int overlap_values,
      bool full_support, const Range& range,
      std::vector<std::pair<float, int32_t>>& output) {
    int pool = full_support ? kK : static_cast<int>(std::ceil(
        kBoundaryFactor * kK * kBlockWidth / static_cast<double>(overlap_values)));
    pool = std::min<int>(pool, active.rows);
    const int task_beam = std::max(beam, pool);
    const long limit = std::min<long>(
        active.graph->size(), std::max<long>(100000L, 100L * task_beam));
    QueryParams params(pool, task_beam, 1.35, limit, active.max_degree);
    BlockLocalPointRange points(packed_, active.begin_row, active.rows);
    auto result = beam_search<PointT, BlockLocalPointRange, Indx>(
        queries_[qid], *active.graph, points, 0, params);
    output.clear();
    output.reserve(result.first.first.size());
    for (const auto& item : result.first.first) {
      const int32_t global_id = packed_ids_[active.begin_row + item.first];
      if (full_support ||
          (attrs_[global_id] >= range.lo && attrs_[global_id] <= range.hi))
        output.emplace_back(item.second, global_id);
    }
  }

  void query(
      int qid, int beam, std::vector<int32_t>& ids,
      std::vector<int64_t>& distances) {
    const Range range = ranges_[qid];
    const int first = range.lo / kBlockWidth;
    const int last = range.hi / kBlockWidth;
    std::vector<std::pair<float, int32_t>> merged;
    std::vector<std::pair<float, int32_t>> output;
    if (first == last) {
      search(qid, beam, node(first, first + 1), range.hi - range.lo + 1,
             false, range, output);
      merged.insert(merged.end(), output.begin(), output.end());
    } else {
      search(qid, beam, node(first, first + 1),
             (first + 1) * kBlockWidth - range.lo, false, range, output);
      merged.insert(merged.end(), output.begin(), output.end());
      for (const auto [begin, end] : cover(first + 1, last)) {
        search(qid, beam, node(begin, end), kBlockWidth, true, range, output);
        merged.insert(merged.end(), output.begin(), output.end());
      }
      search(qid, beam, node(last, last + 1),
             range.hi - last * kBlockWidth + 1, false, range, output);
      merged.insert(merged.end(), output.begin(), output.end());
    }
    finish(merged, ids, distances);
  }

  BlockContiguousVectors packed_;
  RawPointRange queries_;
  std::vector<int32_t> packed_ids_;
  std::vector<uint64_t> offsets_;
  std::vector<int32_t> attrs_;
  std::vector<int32_t> ranges_flat_;
  std::vector<Range> ranges_;
  std::vector<std::unique_ptr<SegmentNodeV52H>> nodes_;
  std::unordered_map<uint64_t, size_t> lookup_;
  uint64_t persistent_bytes_ = 0;
};

void synthetic_stage(
    int row_count,
    std::vector<int32_t>& ids,
    std::vector<int64_t>& distances,
    std::vector<int64_t>& walls,
    int64_t& latest_end) {
  ids.resize(kBeams.size() * row_count * kK);
  distances.resize(ids.size());
  for (size_t cell = 0; cell < kBeams.size(); ++cell)
    for (int row = 0; row < row_count; ++row)
      for (int rank = 0; rank < kK; ++rank) {
        const size_t index = (cell * row_count + row) * kK + rank;
        ids[index] = rank;
        distances[index] = static_cast<int64_t>(rank) * rank;
      }
  walls.clear();
  for (size_t cell = 0; cell < kBeams.size(); ++cell)
    walls.push_back(1'000'000 + static_cast<int64_t>(cell));
  latest_end = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string stage_summary(
    const Options& options, const std::string& stage, int row_count,
    const std::vector<int64_t>& walls, int64_t latest_end,
    const std::string& ids_leaf, size_t ids_bytes, const std::string& ids_sha,
    const std::string& distances_leaf, size_t distances_bytes,
    const std::string& distances_sha, const std::string& workload_sha,
    const std::string& dual_sha, uint64_t persistent_bytes) {
  const int pass_index = stage == "Q16" ? 0 : std::stoi(stage.substr(5));
  std::ostringstream cells;
  cells << '[';
  for (size_t index = 0; index < kBeams.size(); ++index) {
    if (index) cells << ',';
    cells << "{\"batch_wall_ns\":" << walls[index]
          << ",\"cell_id\":\"BOOLE_BEAM_" << kBeams[index]
          << "\",\"complete_query_count\":" << row_count
          << ",\"physical_state\":\"BOOLE_HIERARCHY\""
          << ",\"result_offset\":" << index * row_count * kK << '}';
  }
  cells << ']';
  std::ostringstream out;
  out << "{\"cells\":" << cells.str()
      << ",\"coordinate\":{\"fresh_process\":" << options.fresh_process
      << ",\"order\":\"" << options.order
      << "\",\"ordinal\":" << options.process_ordinal
      << ",\"paired_block\":" << options.paired_block
      << ",\"position_in_pair\":" << options.position_in_pair
      << ",\"system\":\"BOOLE_ANN\"}"
      << ",\"distances_file\":\"" << distances_leaf
      << "\",\"distances_sha256\":\"" << distances_sha
      << "\",\"distances_size_bytes\":" << distances_bytes
      << ",\"dual_gt_sha256\":\"" << dual_sha
      << "\",\"ids_file\":\"" << ids_leaf
      << "\",\"ids_sha256\":\"" << ids_sha
      << "\",\"ids_size_bytes\":" << ids_bytes
      << ",\"interleave_nodes\":[4,5,6],\"k\":10"
      << ",\"latest_timed_worker_end_ns\":" << latest_end
      << ",\"implementation_version\":\"" << kImplementationVersion
      << "\",\"parent_directory_fsynced\":true"
      << ",\"pass_index\":" << pass_index
      << ",\"physical_state_bytes\":{\"BOOLE_HIERARCHY\":"
      << persistent_bytes << "}"
      << ",\"raw_files_fsynced\":true"
      << ",\"ready_rss_bytes\":" << peak_rss_kib() * 1024LL
      << ",\"row_count\":" << row_count
      << ",\"schema\":\"SIFT100K_RANGE_V52H_GATE_B_CHILD_STAGE/v1\""
      << ",\"stage\":\"" << stage
      << "\",\"summary_published_after_latest_worker_end\":true"
      << ",\"supervisor_cpu\":20"
      << ",\"timed_worker_forbidden_counts\":{"
      << "\"CPUFREQ_SAMPLING\":0,\"GROUND_TRUTH_LOOKUP\":0,"
      << "\"RECALL_EVALUATION\":0,\"RESOURCE_SAMPLING\":0,"
      << "\"SET_INTERSECTION\":0,\"STDOUT_WRITE\":0,"
      << "\"TELEMETRY_IO\":0,\"THERMAL_SAMPLING\":0}"
      << ",\"worker_cpus\":[12,13,14,15,16,17,18,19]"
      << ",\"workload_manifest_sha256\":\"" << workload_sha << "\"}\n";
  return out.str();
}

void validate_release(
    const Options& options, const std::string& stage,
    const std::string& summary_leaf, const std::string& summary_sha,
    const std::string& ids_leaf, const std::string& ids_sha,
    const std::string& distances_leaf, const std::string& distances_sha) {
  std::string message;
  if (!std::getline(std::cin, message)) fail("Boole release protocol EOF");
  message.push_back('\n');
  if (json_string(message, "event") != "RELEASE" ||
      json_string(message, "stage") != stage)
    fail("Boole release message stage");
  const std::string release_leaf = json_string(message, "release_leaf");
  const std::string expected_release = stage;
  std::string lower = expected_release;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (release_leaf != lower + ".release.json") fail("Boole release leaf");
  const std::string release_path = options.point_root + "/" + release_leaf;
  const std::string release_payload = read_all(release_path, 4 << 20);
  if (sha256_string(release_payload) != json_string(message, "release_sha256"))
    fail("Boole release digest");
  if (json_string(release_payload, "stem") != lower ||
      json_string(release_payload, "action") != "START_NEXT_STAGE" ||
      !has_exact_fragment(release_payload, "\"ack_directory_durable\":true"))
    fail("Boole release semantics");
  const std::string ack_leaf = json_string(release_payload, "ack_leaf");
  if (ack_leaf != lower + ".ack.json") fail("Boole ACK leaf");
  const std::string ack_payload = read_all(options.point_root + "/" + ack_leaf, 4 << 20);
  if (sha256_string(ack_payload) != json_string(release_payload, "ack_payload_sha256"))
    fail("Boole ACK digest");
  if (json_string(ack_payload, "stem") != lower ||
      !has_exact_fragment(ack_payload, "\"valid\":true"))
    fail("Boole ACK semantics");
  const std::map<std::string, std::string> raw{
      {distances_leaf, distances_sha}, {ids_leaf, ids_sha},
      {summary_leaf, summary_sha}};
  std::ostringstream binding;
  binding << "\"raw_bindings\":{";
  bool first = true;
  for (const auto& [leaf, digest] : raw) {
    if (!first) binding << ',';
    first = false;
    binding << '\"' << leaf << "\":\"" << digest << '\"';
  }
  binding << '}';
  if (!has_exact_fragment(ack_payload, binding.str())) fail("Boole ACK raw bindings");
  const std::string audit_leaf = json_string(ack_payload, "audit_leaf");
  const std::string audit_payload = read_all(options.point_root + "/" + audit_leaf, 16 << 20);
  if (sha256_string(audit_payload) != json_string(ack_payload, "audit_sha256"))
    fail("Boole audit digest");
  std::ostringstream coordinate;
  coordinate << "\"coordinate\":{\"fresh_process\":" << options.fresh_process
             << ",\"order\":\"" << options.order
             << "\",\"ordinal\":" << options.process_ordinal
             << ",\"paired_block\":" << options.paired_block
             << ",\"position_in_pair\":" << options.position_in_pair
             << ",\"system\":\"BOOLE_ANN\"}";
  if (!has_exact_fragment(audit_payload, coordinate.str()) ||
      json_string(audit_payload, "stage") != stage ||
      !has_exact_fragment(audit_payload, "\"valid\":true") ||
      json_string(audit_payload, "raw_summary_leaf") != summary_leaf ||
      json_string(audit_payload, "raw_summary_sha256") != summary_sha ||
      json_string(audit_payload, "raw_ids_leaf") != ids_leaf ||
      json_string(audit_payload, "raw_ids_sha256") != ids_sha ||
      json_string(audit_payload, "raw_distances_leaf") != distances_leaf ||
      json_string(audit_payload, "raw_distances_sha256") != distances_sha)
    fail("Boole audit binding");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::unique_ptr<RealBackend> real;
    uint64_t persistent_bytes = kExpectedGraphBytes;
    std::string workload_sha = options.synthetic_workload_sha256;
    std::string dual_sha = options.synthetic_dual_gt_sha256;
    if (!options.synthetic) {
      std::filesystem::canonical(options.workload_root);
      std::filesystem::canonical(options.dual_gt_root);
      std::filesystem::canonical(options.graph_root);
      workload_sha = sha256_file(options.workload_root + "/WORKLOAD_COMPLETE.json", 4 << 20);
      dual_sha = sha256_file(options.dual_gt_root + "/dual_exact_gt.json", 64 << 20);
      real = std::make_unique<RealBackend>(options);
      persistent_bytes = real->persistent_bytes();
    }
    if (persistent_bytes > kPersistentLimit)
      fail("Boole persistent state exceeds frozen budget");
    const std::array<std::string, 7> stages{
        "Q16", "PASS_1", "PASS_2", "PASS_3", "PASS_4", "PASS_5", "PASS_6"};
    for (const std::string& stage : stages) {
      const int rows = stage == "Q16" ? 16 : 512;
      std::vector<int32_t> ids;
      std::vector<int64_t> distances;
      std::vector<int64_t> walls;
      int64_t latest_end = 0;
      if (options.synthetic)
        synthetic_stage(rows, ids, distances, walls, latest_end);
      else
        real->run(rows, ids, distances, walls, latest_end);
      std::string lower = stage;
      std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
      const std::string ids_leaf = lower + ".ids.i32";
      const std::string distances_leaf = lower + ".distances.i64";
      const std::string ids_sha = publish_raw(options.point_root, ids_leaf, ids);
      const std::string distances_sha = publish_raw(
          options.point_root, distances_leaf, distances);
      const std::string summary_leaf = lower + ".summary.json";
      const std::string summary = stage_summary(
          options, stage, rows, walls, latest_end,
          ids_leaf, ids.size() * sizeof(int32_t), ids_sha,
          distances_leaf, distances.size() * sizeof(int64_t), distances_sha,
          workload_sha, dual_sha, persistent_bytes);
      const std::string summary_sha = publish_text(
          options.point_root, summary_leaf, summary);
      std::cout << "{\"event\":\"STAGE_READY\",\"stage\":\"" << stage
                << "\",\"summary_leaf\":\"" << summary_leaf << "\"}\n"
                << std::flush;
      validate_release(
          options, stage, summary_leaf, summary_sha,
          ids_leaf, ids_sha, distances_leaf, distances_sha);
      std::cout << "{\"event\":\"RELEASE_ACCEPTED\",\"stage\":\""
                << stage << "\"}\n" << std::flush;
    }
    std::cout << "{\"event\":\"CHILD_COMPLETE\",\"passes\":6}\n"
              << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"detail\":\"" << json_escape(error.what())
              << "\",\"verdict\":\"FAIL_CLOSED_V52H_BOOLE_PROVIDER\"}\n"
              << std::flush;
    return 3;
  }
}
