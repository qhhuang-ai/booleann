#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "utils/euclidian_point.h"

namespace bci_block_local {

namespace fs = std::filesystem;
using Point = Euclidian_Point<float>;

inline bool host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

inline std::vector<uint64_t> load_vector_row_offsets(
    const fs::path& path, size_t expected_blocks, uint64_t expected_rows) {
  if (!host_is_little_endian()) {
    throw std::runtime_error("block offsets require a little-endian host");
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open block offsets " + path.string());
  const size_t expected_count = expected_blocks + 1;
  const uint64_t expected_bytes = expected_count * sizeof(uint64_t);
  const auto end = input.tellg();
  if (end < 0 || static_cast<uint64_t>(end) != expected_bytes) {
    throw std::runtime_error("block offset file has the wrong byte size");
  }
  input.seekg(0);
  std::vector<uint64_t> offsets(expected_count);
  input.read(reinterpret_cast<char*>(offsets.data()),
             static_cast<std::streamsize>(expected_bytes));
  if (!input || offsets.front() != 0 || offsets.back() != expected_rows) {
    throw std::runtime_error("block offsets have invalid endpoints");
  }
  for (size_t block = 0; block < expected_blocks; ++block) {
    if (offsets[block + 1] < offsets[block]) {
      throw std::runtime_error("block offsets are not monotone");
    }
  }
  return offsets;
}

class BlockContiguousVectors {
 public:
  BlockContiguousVectors(const fs::path& path, size_t rows,
                         unsigned int dimension)
      : rows_(rows), dimension_(dimension),
        bytes_(rows * static_cast<size_t>(dimension) * sizeof(float)) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("cannot open block-contiguous vectors " +
                               path.string());
    }
    const off_t size = ::lseek(fd_, 0, SEEK_END);
    if (size != static_cast<off_t>(bytes_)) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("unexpected block-contiguous vector byte size");
    }
    void* mapped = ::mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap failed for block-contiguous vectors");
    }
    data_ = static_cast<const float*>(mapped);
  }

  ~BlockContiguousVectors() {
    if (data_) ::munmap(const_cast<float*>(data_), bytes_);
    if (fd_ >= 0) ::close(fd_);
  }

  BlockContiguousVectors(const BlockContiguousVectors&) = delete;
  BlockContiguousVectors& operator=(const BlockContiguousVectors&) = delete;

  size_t size() const { return rows_; }
  long dimension() const { return dimension_; }
  long aligned_dimension() const { return dimension_; }

  Point point(uint64_t absolute_row, long exposed_local_id) const {
    return Point(data_ + absolute_row * static_cast<uint64_t>(dimension_),
                 dimension_, dimension_, exposed_local_id);
  }

 private:
  int fd_{-1};
  const float* data_{nullptr};
  size_t rows_{0};
  unsigned int dimension_{0};
  size_t bytes_{0};
};

class BlockLocalPointRange {
 public:
  BlockLocalPointRange(BlockContiguousVectors& vectors, uint64_t begin_row,
                       size_t rows)
      : vectors_(vectors), begin_row_(begin_row), rows_(rows) {
    if (begin_row_ > vectors_.size() ||
        rows_ > vectors_.size() - static_cast<size_t>(begin_row_)) {
      throw std::runtime_error("block-local point range exceeds mapped vectors");
    }
  }

  size_t size() const { return rows_; }
  long dimension() const { return vectors_.dimension(); }
  long aligned_dimension() const { return vectors_.aligned_dimension(); }

  Point operator[](long local_id) {
    return vectors_.point(begin_row_ + static_cast<uint64_t>(local_id), local_id);
  }

 private:
  BlockContiguousVectors& vectors_;
  uint64_t begin_row_{0};
  size_t rows_{0};
};

}  // namespace bci_block_local
