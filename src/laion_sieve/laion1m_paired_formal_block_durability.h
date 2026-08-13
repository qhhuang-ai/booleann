#pragma once

// Query-free durability boundary for a LAION paired formal runner.  The
// dataset runner owns raw result/summary publication; the parent adapter owns
// ACK then release publication.  This helper only validates that permission
// to start the next block is bound to the exact two raw byte counts.

#include "pass_release_contract.h"
#include "bci_cpp_port/include/pass_boundary_contract.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace laion_paired_formal_block_durability {

namespace fs = std::filesystem;

inline constexpr durable_pass_release::ContractSchemas kSchemas{
    "laion1m-paired-formal-block-adapter-ack/v1",
    "laion1m-paired-formal-block-adapter-release/v1"};

struct FileSnapshot {
  dev_t device;
  ino_t inode;
  off_t bytes;
  timespec modified;
};

inline void validate_state_id(const std::string& state_id) {
  if (state_id.empty() || state_id.size() > 128) {
    throw std::runtime_error("state-id must contain 1..128 safe characters");
  }
  for (const unsigned char value : state_id) {
    const bool safe = (value >= 'a' && value <= 'z') ||
                      (value >= 'A' && value <= 'Z') ||
                      (value >= '0' && value <= '9') || value == '.' ||
                      value == '_' || value == '-';
    if (!safe) {
      throw std::runtime_error(
          "state-id may contain only ASCII letters, digits, dot, underscore, or dash");
    }
  }
}

inline std::string block_stem(size_t block) {
  if (block > 9999) throw std::runtime_error("block is outside 0..9999");
  std::ostringstream output;
  output << "block_" << std::setw(2) << std::setfill('0') << block;
  return output.str();
}

inline void validate_ack_directory(
    const fs::path& path, const std::function<void()>& note_io) {
  note_io();
  if (!path.is_absolute() || !fs::is_directory(fs::symlink_status(path))) {
    throw std::runtime_error(
        "ack directory must be an existing absolute real directory");
  }
}

inline void publish_block_artifact_atomic_no_replace(
    const fs::path& final_path, std::string_view payload,
    const std::function<void()>& require_not_stopped) {
  // Reuse the established BCI pass-boundary publisher itself: exclusive temp
  // inode, file fsync, no-replace rename, then parent-directory fsync.
  require_not_stopped();
  bci_pass_boundary::publish_atomic_no_replace(final_path, payload);
  require_not_stopped();
}

inline FileSnapshot snapshot_nonempty_regular_file(
    const fs::path& path, const std::string& label,
    const std::function<void()>& note_io) {
  note_io();
  const int descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) {
    throw std::runtime_error("cannot open " + label + " " + path.string() +
                             ": " + std::string(std::strerror(errno)));
  }
  struct stat status {};
  const int stat_result = ::fstat(descriptor, &status);
  const int close_result = ::close(descriptor);
  if (stat_result != 0 || close_result != 0 || !S_ISREG(status.st_mode) ||
      status.st_size <= 0) {
    throw std::runtime_error(label + " must be a nonempty regular file");
  }
  return FileSnapshot{
      status.st_dev, status.st_ino, status.st_size, status.st_mtim};
}

inline bool same_snapshot(const FileSnapshot& left,
                          const FileSnapshot& right) {
  return left.device == right.device && left.inode == right.inode &&
         left.bytes == right.bytes &&
         left.modified.tv_sec == right.modified.tv_sec &&
         left.modified.tv_nsec == right.modified.tv_nsec;
}

inline durable_pass_release::AckRecord wait_for_next_block_release(
    const fs::path& ack_directory, const std::string& state_id, size_t block,
    size_t expected_summary_bytes, size_t expected_results_bytes,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval,
    const std::function<void()>& require_not_stopped,
    const std::function<void()>& note_io) {
  validate_state_id(state_id);
  validate_ack_directory(ack_directory, note_io);
  if (expected_summary_bytes == 0 || expected_results_bytes == 0) {
    throw std::runtime_error(
        "formal block result and summary byte counts must be nonzero");
  }
  durable_pass_release::AckRecord acknowledgement =
      durable_pass_release::wait_for_release_then_ack(
          ack_directory, block_stem(block), state_id, block, timeout,
          poll_interval, require_not_stopped, note_io, kSchemas);
  // The shared SIFT field names remain stable for compatibility.  For this
  // block contract passes_jsonl_bytes binds the one block summary, while
  // results_jsonl_bytes binds the one block result payload.
  if (acknowledgement.passes_jsonl_bytes != expected_summary_bytes ||
      acknowledgement.results_jsonl_bytes != expected_results_bytes) {
    throw std::runtime_error(
        "adapter acknowledgement does not bind exact block summary/results bytes");
  }
  return acknowledgement;
}

inline durable_pass_release::AckRecord wait_for_next_block_release_for_files(
    const fs::path& ack_directory, const std::string& state_id, size_t block,
    const fs::path& summary_path, const fs::path& results_path,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval,
    const std::function<void()>& require_not_stopped,
    const std::function<void()>& note_io) {
  const FileSnapshot summary_before = snapshot_nonempty_regular_file(
      summary_path, "formal block summary", note_io);
  const FileSnapshot results_before = snapshot_nonempty_regular_file(
      results_path, "formal block results", note_io);
  if (summary_before.device == results_before.device &&
      summary_before.inode == results_before.inode) {
    throw std::runtime_error(
        "formal block summary and results must be distinct files");
  }
  const auto acknowledgement = wait_for_next_block_release(
      ack_directory, state_id, block,
      static_cast<size_t>(summary_before.bytes),
      static_cast<size_t>(results_before.bytes), timeout, poll_interval,
      require_not_stopped, note_io);
  const FileSnapshot summary_after = snapshot_nonempty_regular_file(
      summary_path, "formal block summary", note_io);
  const FileSnapshot results_after = snapshot_nonempty_regular_file(
      results_path, "formal block results", note_io);
  if (!same_snapshot(summary_before, summary_after) ||
      !same_snapshot(results_before, results_after)) {
    throw std::runtime_error(
        "formal block summary/results changed across ACK-release wait");
  }
  return acknowledgement;
}

}  // namespace laion_paired_formal_block_durability
