#pragma once

#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

#include "../../pass_release_contract.h"

namespace bci_pass_boundary {

namespace fs = std::filesystem;

inline volatile std::sig_atomic_t termination_signal = 0;
inline thread_local bool worker_timed_region = false;
inline std::atomic<uint64_t> explicit_worker_timed_io_calls{0};

inline void record_explicit_io_call() {
  if (worker_timed_region) {
    explicit_worker_timed_io_calls.fetch_add(1, std::memory_order_relaxed);
  }
}

inline void signal_handler(int signal_number) {
  termination_signal = signal_number;
}

inline void install_signal_handlers() {
  struct sigaction action {};
  action.sa_handler = signal_handler;
  ::sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  if (::sigaction(SIGINT, &action, nullptr) != 0 ||
      ::sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error("cannot install SIGINT/SIGTERM handlers");
  }
}

inline bool stop_requested() { return termination_signal != 0; }

inline void require_not_stopped() {
  if (stop_requested()) {
    throw std::runtime_error(
        "termination requested by signal " +
        std::to_string(static_cast<int>(termination_signal)));
  }
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) ::close(value_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : value_(other.value_) {
    other.value_ = -1;
  }
  int get() const { return value_; }
  void close_checked() {
    if (value_ >= 0) {
      const int saved = value_;
      value_ = -1;
      if (::close(saved) != 0) throw std::runtime_error("close failed");
    }
  }

 private:
  int value_;
};

inline void write_all(int fd, std::string_view content) {
  record_explicit_io_call();
  size_t offset = 0;
  while (offset < content.size()) {
    const ssize_t written = ::write(
        fd, content.data() + offset, content.size() - offset);
    if (written < 0) {
      if (errno == EINTR && !stop_requested()) continue;
      throw std::runtime_error("write failed: " +
                               std::string(std::strerror(errno)));
    }
    if (written == 0) throw std::runtime_error("zero-byte write");
    offset += static_cast<size_t>(written);
  }
}

inline void fsync_checked(int fd, const std::string& description) {
  record_explicit_io_call();
  if (::fsync(fd) != 0) {
    throw std::runtime_error("fsync failed for " + description + ": " +
                             std::string(std::strerror(errno)));
  }
}

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

inline void validate_empty_contract_directory(
    const fs::path& path, const std::string& label) {
  record_explicit_io_call();
  if (!path.is_absolute()) {
    throw std::runtime_error(label + " must be an absolute path");
  }
  const fs::file_status status = fs::symlink_status(path);
  if (!fs::is_directory(status)) {
    throw std::runtime_error(label + " must already exist as a real directory");
  }
  if (fs::directory_iterator(path) != fs::directory_iterator()) {
    throw std::runtime_error(label + " must be empty at C++ startup");
  }
}

inline std::string pass_stem(size_t pass) {
  if (pass < 1 || pass > 99) throw std::runtime_error("pass out of range");
  return pass < 10 ? "pass_0" + std::to_string(pass)
                   : "pass_" + std::to_string(pass);
}

inline void fsync_directory(const fs::path& directory);

class PassEvents {
 public:
  explicit PassEvents(const fs::path& raw_pass_dir)
      : path_(raw_pass_dir / "pass_events.csv"),
        file_(open_exclusive(path_)) {
    write_all(file_.get(), "event,pass\n");
    fsync_checked(file_.get(), path_.string());
    fsync_directory(raw_pass_dir);
  }

  void append_start(size_t pass) {
    require_not_stopped();
    write_all(file_.get(), "pass_start," + std::to_string(pass) + "\n");
    fsync_checked(file_.get(), path_.string());
  }

 private:
  static int open_exclusive(const fs::path& path) {
    record_explicit_io_call();
    const int fd = ::open(
        path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) {
      throw std::runtime_error("cannot exclusively create " + path.string() +
                               ": " + std::string(std::strerror(errno)));
    }
    return fd;
  }

  fs::path path_;
  FileDescriptor file_;
};

inline void fsync_directory(const fs::path& directory) {
  record_explicit_io_call();
  FileDescriptor fd(::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (fd.get() < 0) {
    throw std::runtime_error("cannot open directory for fsync " +
                             directory.string());
  }
  fsync_checked(fd.get(), directory.string());
}

inline void rename_no_replace(const fs::path& source, const fs::path& target) {
  record_explicit_io_call();
#ifdef SYS_renameat2
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1U << 0)
#endif
  if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                target.c_str(), RENAME_NOREPLACE) != 0) {
    throw std::runtime_error("no-replace rename failed from " + source.string() +
                             " to " + target.string() + ": " +
                             std::string(std::strerror(errno)));
  }
#else
  (void)source;
  (void)target;
  throw std::runtime_error("renameat2(RENAME_NOREPLACE) is unavailable");
#endif
}

inline void publish_atomic_no_replace(
    const fs::path& final_path, std::string_view content) {
  require_not_stopped();
  const fs::path directory = final_path.parent_path();
  const fs::path temporary = directory /
      ("." + final_path.filename().string() + ".tmp." +
       std::to_string(static_cast<long long>(::getpid())));
  record_explicit_io_call();
  FileDescriptor file(::open(
      temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644));
  if (file.get() < 0) {
    throw std::runtime_error("cannot exclusively create temporary file " +
                             temporary.string() + ": " +
                             std::string(std::strerror(errno)));
  }
  write_all(file.get(), content);
  fsync_checked(file.get(), temporary.string());
  require_not_stopped();
  file.close_checked();
  rename_no_replace(temporary, final_path);
  fsync_directory(directory);
}

inline std::string read_small_regular_file(
    const fs::path& path, size_t maximum_bytes) {
  record_explicit_io_call();
  const fs::file_status status = fs::symlink_status(path);
  if (!fs::is_regular_file(status)) {
    throw std::runtime_error("acknowledgement is not a regular file: " +
                             path.string());
  }
  const uintmax_t bytes = fs::file_size(path);
  if (bytes == 0 || bytes > maximum_bytes) {
    throw std::runtime_error("acknowledgement has an invalid byte size");
  }
  FileDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (file.get() < 0) {
    throw std::runtime_error("cannot open acknowledgement " + path.string());
  }
  std::string content(static_cast<size_t>(bytes), '\0');
  size_t offset = 0;
  while (offset < content.size()) {
    const ssize_t count = ::read(
        file.get(), content.data() + offset, content.size() - offset);
    if (count < 0) {
      if (errno == EINTR && !stop_requested()) continue;
      throw std::runtime_error("acknowledgement read failed");
    }
    if (count == 0) throw std::runtime_error("short acknowledgement read");
    offset += static_cast<size_t>(count);
  }
  return content;
}

inline size_t unique_json_field_position(
    const std::string& json, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  const size_t position = json.find(marker);
  if (position == std::string::npos ||
      json.find(marker, position + marker.size()) != std::string::npos) {
    throw std::runtime_error("acknowledgement must contain exactly one " + key);
  }
  size_t cursor = position + marker.size();
  while (cursor < json.size() &&
         (json[cursor] == ' ' || json[cursor] == '\t' ||
          json[cursor] == '\r' || json[cursor] == '\n')) {
    ++cursor;
  }
  if (cursor >= json.size() || json[cursor] != ':') {
    throw std::runtime_error("malformed acknowledgement field " + key);
  }
  ++cursor;
  while (cursor < json.size() &&
         (json[cursor] == ' ' || json[cursor] == '\t' ||
          json[cursor] == '\r' || json[cursor] == '\n')) {
    ++cursor;
  }
  return cursor;
}

inline std::string json_string_field(
    const std::string& json, const std::string& key) {
  size_t cursor = unique_json_field_position(json, key);
  if (cursor >= json.size() || json[cursor] != '"') {
    throw std::runtime_error("acknowledgement field is not a string: " + key);
  }
  ++cursor;
  std::string value;
  while (cursor < json.size()) {
    const char current = json[cursor++];
    if (current == '"') {
      while (cursor < json.size() &&
             (json[cursor] == ' ' || json[cursor] == '\t' ||
              json[cursor] == '\r' || json[cursor] == '\n')) {
        ++cursor;
      }
      if (cursor >= json.size() ||
          (json[cursor] != ',' && json[cursor] != '}')) {
        throw std::runtime_error("malformed acknowledgement string field " + key);
      }
      return value;
    }
    if (current == '\\') {
      if (cursor >= json.size()) break;
      const char escaped = json[cursor++];
      if (escaped == '"' || escaped == '\\' || escaped == '/') {
        value.push_back(escaped);
      } else {
        throw std::runtime_error("unsupported escape in acknowledgement string");
      }
    } else {
      value.push_back(current);
    }
  }
  throw std::runtime_error("unterminated acknowledgement string field " + key);
}

inline int64_t json_integer_field(
    const std::string& json, const std::string& key) {
  const size_t begin = unique_json_field_position(json, key);
  size_t end = begin;
  if (end < json.size() && json[end] == '-') ++end;
  while (end < json.size() && json[end] >= '0' && json[end] <= '9') ++end;
  if (end == begin || (end == begin + 1 && json[begin] == '-')) {
    throw std::runtime_error("acknowledgement field is not an integer: " + key);
  }
  const size_t digit_begin = json[begin] == '-' ? begin + 1 : begin;
  if (end - digit_begin > 1 && json[digit_begin] == '0') {
    throw std::runtime_error("acknowledgement integer has a leading zero: " + key);
  }
  int64_t value = 0;
  const auto result = std::from_chars(
      json.data() + begin, json.data() + end, value);
  if (result.ec != std::errc() || result.ptr != json.data() + end) {
    throw std::runtime_error("invalid acknowledgement integer: " + key);
  }
  size_t delimiter = end;
  while (delimiter < json.size() &&
         (json[delimiter] == ' ' || json[delimiter] == '\t' ||
          json[delimiter] == '\r' || json[delimiter] == '\n')) {
    ++delimiter;
  }
  if (delimiter >= json.size() ||
      (json[delimiter] != ',' && json[delimiter] != '}')) {
    throw std::runtime_error("non-integral acknowledgement field: " + key);
  }
  return value;
}

inline void validate_acknowledgement(
    const std::string& json, const std::string& state_id, size_t pass) {
  (void)durable_pass_release::validate_canonical_ack(json, state_id, pass);
}

inline void wait_for_acknowledgement(
    const fs::path& ack_directory, const std::string& state_id, size_t pass,
    std::chrono::milliseconds poll_interval = std::chrono::milliseconds(50)) {
  durable_pass_release::wait_for_release_then_ack(
      ack_directory, pass_stem(pass), state_id, pass,
      std::chrono::duration_cast<std::chrono::milliseconds>(
          durable_pass_release::kFrozenWaitTimeout),
      poll_interval, require_not_stopped, record_explicit_io_call);
}

}  // namespace bci_pass_boundary
