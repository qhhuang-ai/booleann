// Resource-clean builder for the fixed-block SIFT10M range index.
//
// Unlike the earlier generic SIFT100M shard builder, this target mmaps only
// the 10M-vector raw array.  It reads the already-frozen, query-independent
// block memberships and writes one Vamana graph per block.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#include "parlay/sequence.h"
#include "utils/euclidian_point.h"
#include "utils/graph.h"
#include "utils/stats.h"
#include "utils/types.h"
#include "vamana/index.h"

using PointT = Euclidian_Point<float>;
using Indx = int32_t;
using GraphI = Graph<Indx>;

class RawPointRange {
 public:
  RawPointRange(const std::string& path, size_t n, unsigned int dim)
      : n_(n), dim_(dim), bytes_(n * static_cast<size_t>(dim) * sizeof(float)) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open raw vectors " + path);
    off_t size = ::lseek(fd_, 0, SEEK_END);
    if (size != static_cast<off_t>(bytes_)) {
      ::close(fd_);
      throw std::runtime_error("unexpected raw-vector size " + path);
    }
    void* mapped = ::mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("mmap failed for " + path);
    }
    data_ = static_cast<const float*>(mapped);
  }

  ~RawPointRange() {
    if (data_) ::munmap(const_cast<float*>(data_), bytes_);
    if (fd_ >= 0) ::close(fd_);
  }

  RawPointRange(const RawPointRange&) = delete;
  RawPointRange& operator=(const RawPointRange&) = delete;
  size_t size() const { return n_; }
  long dimension() const { return dim_; }
  long aligned_dimension() const { return dim_; }
  PointT operator[](long i) {
    return PointT(const_cast<float*>(data_ + static_cast<size_t>(i) * dim_),
                  dim_, dim_, i);
  }

 private:
  int fd_ = -1;
  const float* data_ = nullptr;
  size_t n_ = 0;
  unsigned int dim_ = 0;
  size_t bytes_ = 0;
};

struct ThinSubPR {
  RawPointRange& points;
  const parlay::sequence<int32_t>& subset;
  ThinSubPR(RawPointRange& p, const parlay::sequence<int32_t>& s)
      : points(p), subset(s) {}
  size_t size() const { return subset.size(); }
  long dimension() const { return points.dimension(); }
  long aligned_dimension() const { return points.aligned_dimension(); }
  PointT operator[](long i) { return points[subset[i]]; }
};

using KNNIdx = knn_index<PointT, ThinSubPR, Indx>;

struct BlockEntry {
  int32_t block;
  int64_t count;
};

static void fsync_regular_file(const std::string& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    throw std::system_error(
        errno, std::generic_category(), "open for file fsync " + path);
  if (::fsync(descriptor) != 0) {
    const int saved_errno = errno;
    ::close(descriptor);
    throw std::system_error(
        saved_errno, std::generic_category(), "file fsync " + path);
  }
  if (::close(descriptor) != 0)
    throw std::system_error(
        errno, std::generic_category(), "close after file fsync " + path);
}

static void fsync_parent_directory(const std::string& path) {
  const std::string parent =
      std::filesystem::path(path).parent_path().string();
  const int descriptor =
      ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (descriptor < 0)
    throw std::system_error(
        errno, std::generic_category(), "open for directory fsync " + parent);
  if (::fsync(descriptor) != 0) {
    const int saved_errno = errno;
    ::close(descriptor);
    throw std::system_error(
        saved_errno, std::generic_category(), "directory fsync " + parent);
  }
  if (::close(descriptor) != 0)
    throw std::system_error(
        errno, std::generic_category(), "close after directory fsync " + parent);
}

static uintmax_t expected_graph_bytes(GraphI& graph) {
  uintmax_t bytes =
      2 * sizeof(Indx) + graph.size() * static_cast<uintmax_t>(sizeof(Indx));
  for (size_t vertex = 0; vertex < graph.size(); ++vertex)
    bytes += graph[static_cast<Indx>(vertex)].size() *
             static_cast<uintmax_t>(sizeof(Indx));
  return bytes;
}

static parlay::sequence<int32_t> load_subset(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("cannot open subset " + path);
  int32_t n = 0;
  if (std::fread(&n, sizeof(n), 1, file) != 1 || n < 0)
    throw std::runtime_error("bad subset header " + path);
  parlay::sequence<int32_t> ids(n);
  if (std::fread(ids.data(), sizeof(int32_t), n, file) != static_cast<size_t>(n))
    throw std::runtime_error("short subset " + path);
  std::fclose(file);
  return ids;
}

static std::vector<BlockEntry> load_blocks(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("cannot open block manifest " + path);
  int32_t n = 0;
  if (std::fread(&n, sizeof(n), 1, file) != 1 || n < 0)
    throw std::runtime_error("bad block manifest " + path);
  std::vector<BlockEntry> blocks(n);
  for (int32_t i = 0; i < n; ++i) {
    if (std::fread(&blocks[i].block, sizeof(int32_t), 1, file) != 1 ||
        std::fread(&blocks[i].count, sizeof(int64_t), 1, file) != 1)
      throw std::runtime_error("short block manifest " + path);
  }
  std::fclose(file);
  return blocks;
}

int main(int argc, char** argv) {
  const std::string data = argc > 1 ? argv[1] : "./data/sift10m/range_blocks";
  const std::string output = argc > 2 ? argv[2] : data + "/shards_clean";
  const int cap = argc > 3 ? std::max(0, std::atoi(argv[3])) : 0;
  const int R = argc > 4 ? std::max(1, std::atoi(argv[4])) : 32;
  const int L = argc > 5 ? std::max(R, std::atoi(argv[5])) : 200;
  const double alpha = argc > 6 ? std::atof(argv[6]) : 1.175;
  const std::string base_path = argc > 7 ? argv[7] : "./data/sift10m/sift_base.fvecs";

  std::filesystem::create_directories(output);
  auto blocks = load_blocks(data + "/sweet_spot_tags.bin");
  std::sort(blocks.begin(), blocks.end(), [](const BlockEntry& a, const BlockEntry& b) {
    return a.count > b.count;
  });
  if (cap > 0 && static_cast<int>(blocks.size()) > cap) blocks.resize(cap);

  std::printf("=== resource-clean SIFT10M fixed-block builder ===\n");
  std::printf("data=%s output=%s blocks=%zu R=%d L=%d alpha=%.3f workers=%ld\n",
              data.c_str(), output.c_str(), blocks.size(), R, L, alpha,
              parlay::num_workers());
  std::fflush(stdout);
  RawPointRange base(base_path, 10'000'000, 128);

  const std::string build_log_path = output + "/build_log.csv";
  std::ofstream log(build_log_path, std::ios::app);
  if (!log.is_open())
    throw std::runtime_error("cannot open build log " + build_log_path);
  if (log.tellp() == 0)
    log << "block,n_points,R,L,alpha,build_seconds,saved_bytes\n";

  size_t built = 0;
  size_t memberships = 0;
  double build_seconds = 0.0;
  for (const auto& entry : blocks) {
    const std::string graph_path =
        output + "/vamana_tag_" + std::to_string(entry.block) + ".bin";
    if (std::filesystem::exists(graph_path)) continue;
    auto subset = load_subset(data + "/subset_idx/subset_idx_" +
                              std::to_string(entry.block) + ".bin");
    if (static_cast<int64_t>(subset.size()) != entry.count)
      throw std::runtime_error("manifest/subset count mismatch");
    memberships += subset.size();

    ThinSubPR points(base, subset);
    GraphI graph(static_cast<long>(R), subset.size());
    stats<Indx> stats_for_graph(graph.size());
    BuildParams params(R, L, alpha);
    KNNIdx index(params);
    const auto begin = std::chrono::steady_clock::now();
    index.build_index(graph, points, stats_for_graph);
    const uintmax_t expected_bytes = expected_graph_bytes(graph);
    graph.save(const_cast<char*>(graph_path.c_str()));
    const auto bytes = std::filesystem::file_size(graph_path);
    if (bytes != expected_bytes)
      throw std::runtime_error(
          "graph serialization byte count mismatch " + graph_path);
    fsync_regular_file(graph_path);
    fsync_parent_directory(graph_path);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    build_seconds += elapsed;
    ++built;
    log << entry.block << ',' << subset.size() << ',' << R << ',' << L << ','
        << alpha << ',' << elapsed << ',' << bytes << '\n';
    log.flush();
    if (!log)
      throw std::runtime_error("build log flush failed " + build_log_path);
    fsync_regular_file(build_log_path);
    std::printf("[%zu/%zu] block=%d n=%zu build=%.2fs cumulative=%.1fs\n",
                built, blocks.size(), entry.block, subset.size(), elapsed,
                build_seconds);
    std::fflush(stdout);
  }
  std::printf("[done] built=%zu memberships=%zu build_seconds=%.1f\n",
              built, memberships, build_seconds);
  std::fflush(stdout);
  return 0;
}
