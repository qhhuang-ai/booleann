#pragma once

// Small shared two-phase adapter release contract.  This header deliberately
// has no dataset, index, ground-truth, or recall dependency.

#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace durable_pass_release {

namespace fs = std::filesystem;

inline constexpr const char* kAckSchema =
    "sift10m-range-v2-adapter-ack/v1";
inline constexpr const char* kReleaseSchema =
    "sift10m-range-v2-adapter-release/v1";

// Dataset-specific users may supply different schema identifiers while the
// existing SIFT callers continue to use these defaults byte-for-byte.
struct ContractSchemas {
  std::string_view ack;
  std::string_view release;
};

inline constexpr ContractSchemas kDefaultSchemas{kAckSchema, kReleaseSchema};
inline constexpr size_t kMaximumContractBytes = 64U * 1024U;
inline constexpr auto kFrozenWaitTimeout = std::chrono::minutes(10);

struct AckRecord {
  size_t pass;
  size_t passes_jsonl_bytes;
  size_t results_jsonl_bytes;
  std::string state_id;
};

struct ReleaseRecord {
  std::string ack_sha256;
  size_t pass;
  std::string state_id;
};

inline size_t unique_value_offset(const std::string& payload,
                                  const std::string& key,
                                  const std::string& label) {
  const std::string marker = "\"" + key + "\":";
  const size_t found = payload.find(marker);
  if (found == std::string::npos ||
      payload.find(marker, found + marker.size()) != std::string::npos) {
    throw std::runtime_error(label + " must contain exactly one " + key);
  }
  return found + marker.size();
}

inline std::string canonical_string_value(const std::string& payload,
                                          const std::string& key,
                                          const std::string& label) {
  size_t cursor = unique_value_offset(payload, key, label);
  if (cursor >= payload.size() || payload[cursor++] != '"') {
    throw std::runtime_error(label + " field is not a canonical string: " + key);
  }
  const size_t begin = cursor;
  while (cursor < payload.size() && payload[cursor] != '"') {
    const unsigned char value = static_cast<unsigned char>(payload[cursor]);
    if (value < 0x20 || payload[cursor] == '\\') {
      throw std::runtime_error(label + " string is not canonical ASCII: " + key);
    }
    ++cursor;
  }
  if (cursor >= payload.size()) {
    throw std::runtime_error(label + " string is unterminated: " + key);
  }
  return payload.substr(begin, cursor - begin);
}

inline size_t canonical_size_value(const std::string& payload,
                                   const std::string& key,
                                   const std::string& label) {
  const size_t begin = unique_value_offset(payload, key, label);
  size_t end = begin;
  while (end < payload.size() && payload[end] >= '0' && payload[end] <= '9') {
    ++end;
  }
  if (end == begin || (end - begin > 1 && payload[begin] == '0')) {
    throw std::runtime_error(label + " field is not a canonical integer: " + key);
  }
  size_t value = 0;
  const auto parsed = std::from_chars(
      payload.data() + begin, payload.data() + end, value);
  if (parsed.ec != std::errc() || parsed.ptr != payload.data() + end) {
    throw std::runtime_error(label + " integer is invalid: " + key);
  }
  return value;
}

inline std::string canonical_ack(
    const AckRecord& record,
    const ContractSchemas& schemas = kDefaultSchemas) {
  return "{\"pass\":" + std::to_string(record.pass) +
         ",\"passes_jsonl_bytes\":" +
         std::to_string(record.passes_jsonl_bytes) +
         ",\"results_jsonl_bytes\":" +
         std::to_string(record.results_jsonl_bytes) +
         ",\"schema\":\"" + std::string(schemas.ack) +
         "\",\"state_id\":\"" +
         record.state_id + "\"}\n";
}

inline std::string canonical_release(
    const ReleaseRecord& record,
    const ContractSchemas& schemas = kDefaultSchemas) {
  return "{\"ack_sha256\":\"" + record.ack_sha256 +
         "\",\"pass\":" + std::to_string(record.pass) +
         ",\"schema\":\"" + std::string(schemas.release) +
         "\",\"state_id\":\"" +
         record.state_id + "\"}\n";
}

inline AckRecord validate_canonical_ack(const std::string& payload,
                                        const std::string& state_id,
                                        size_t pass,
                                        const ContractSchemas& schemas =
                                            kDefaultSchemas) {
  AckRecord record{
      canonical_size_value(payload, "pass", "acknowledgement"),
      canonical_size_value(payload, "passes_jsonl_bytes", "acknowledgement"),
      canonical_size_value(payload, "results_jsonl_bytes", "acknowledgement"),
      canonical_string_value(payload, "state_id", "acknowledgement")};
  if (canonical_string_value(payload, "schema", "acknowledgement") !=
          schemas.ack ||
      record.state_id != state_id || record.pass != pass) {
    throw std::runtime_error(
        "canonical acknowledgement identity does not match frozen state/pass");
  }
  if (payload != canonical_ack(record, schemas)) {
    throw std::runtime_error(
        "acknowledgement is not exact canonical five-field JSON");
  }
  return record;
}

inline ReleaseRecord validate_canonical_release(const std::string& payload,
                                                const std::string& state_id,
                                                size_t pass,
                                                const ContractSchemas& schemas =
                                                    kDefaultSchemas) {
  ReleaseRecord record{
      canonical_string_value(payload, "ack_sha256", "release"),
      canonical_size_value(payload, "pass", "release"),
      canonical_string_value(payload, "state_id", "release")};
  if (canonical_string_value(payload, "schema", "release") !=
          schemas.release ||
      record.state_id != state_id || record.pass != pass) {
    throw std::runtime_error(
        "release identity does not match frozen state/pass");
  }
  if (record.ack_sha256.size() != 64) {
    throw std::runtime_error("release ack_sha256 must have 64 lowercase hex digits");
  }
  for (const char value : record.ack_sha256) {
    if (!((value >= '0' && value <= '9') ||
          (value >= 'a' && value <= 'f'))) {
      throw std::runtime_error(
          "release ack_sha256 must have 64 lowercase hex digits");
    }
  }
  if (payload != canonical_release(record, schemas)) {
    throw std::runtime_error("release is not exact canonical four-field JSON");
  }
  return record;
}

inline uint32_t rotate_right(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

inline std::string sha256_hex(std::string_view input) {
  static constexpr std::array<uint32_t, 64> constants{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::array<uint32_t, 8> hash{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::string bytes(input);
  const uint64_t bit_count = static_cast<uint64_t>(bytes.size()) * 8U;
  bytes.push_back(static_cast<char>(0x80));
  while (bytes.size() % 64 != 56) bytes.push_back('\0');
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<char>((bit_count >> shift) & 0xffU));
  }
  for (size_t block = 0; block < bytes.size(); block += 64) {
    std::array<uint32_t, 64> words{};
    for (size_t index = 0; index < 16; ++index) {
      const size_t offset = block + index * 4;
      words[index] =
          (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) << 24U) |
          (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 16U) |
          (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 8U) |
          static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3]));
    }
    for (size_t index = 16; index < 64; ++index) {
      const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                          rotate_right(words[index - 15], 18) ^
                          (words[index - 15] >> 3U);
      const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                          rotate_right(words[index - 2], 19) ^
                          (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    uint32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    uint32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];
    for (size_t index = 0; index < 64; ++index) {
      const uint32_t upper = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                             rotate_right(e, 25);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t first = h + upper + choose + constants[index] + words[index];
      const uint32_t lower = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                             rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t second = lower + majority;
      h = g; g = f; f = e; e = d + first;
      d = c; c = b; b = a; a = first + second;
    }
    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const uint32_t value : hash) output << std::setw(8) << value;
  return output.str();
}

inline std::optional<std::string> read_visible_contract_file(
    const fs::path& path, const std::string& label,
    const std::function<void()>& note_io) {
  note_io();
  const int descriptor =
      // O_NONBLOCK is essential before fstat: opening a FIFO for read without
      // a writer otherwise blocks in open(2), bypassing both the frozen wait
      // deadline and the parent stop path.  It has no effect on regular-file
      // reads; fstat below immediately rejects every non-regular inode.
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) {
    if (errno == ENOENT) return std::nullopt;
    throw std::runtime_error("cannot open " + label + " " + path.string() +
                             ": " + std::string(std::strerror(errno)));
  }
  struct stat before {};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<uintmax_t>(before.st_size) > kMaximumContractBytes) {
    ::close(descriptor);
    throw std::runtime_error(label + " is not a nonempty bounded regular file");
  }
  std::string payload(static_cast<size_t>(before.st_size), '\0');
  size_t offset = 0;
  while (offset < payload.size()) {
    note_io();
    const ssize_t count = ::read(
        descriptor, payload.data() + offset, payload.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      ::close(descriptor);
      throw std::runtime_error("short read from " + label);
    }
    offset += static_cast<size_t>(count);
  }
  char trailing = 0;
  note_io();
  const ssize_t extra = ::read(descriptor, &trailing, 1);
  struct stat after {};
  const bool stable = extra == 0 && ::fstat(descriptor, &after) == 0 &&
      before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
      before.st_size == after.st_size &&
      before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
      before.st_mtim.tv_nsec == after.st_mtim.tv_nsec;
  const int close_result = ::close(descriptor);
  if (!stable || close_result != 0) {
    throw std::runtime_error(label + " changed while being validated");
  }
  return payload;
}

inline AckRecord wait_for_release_then_ack(
    const fs::path& ack_directory, const std::string& pass_stem,
    const std::string& state_id, size_t pass,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval,
    const std::function<void()>& require_not_stopped,
    const std::function<void()>& note_io,
    const ContractSchemas& schemas = kDefaultSchemas) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  const fs::path release_path = ack_directory / (pass_stem + ".release.json");
  const fs::path ack_path = ack_directory / (pass_stem + ".json");
  while (true) {
    require_not_stopped();
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("adapter release/ack wait timed out before release");
    }
    const auto payload = read_visible_contract_file(
        release_path, "adapter release", note_io);
    if (payload.has_value()) {
      const ReleaseRecord release =
          validate_canonical_release(*payload, state_id, pass, schemas);
      // A release pathname is the permission edge.  Its publisher must have
      // already made the canonical ACK durable.  Waiting for an ACK after
      // observing release would turn a forbidden release-before-ACK ordering
      // into success, so absence is terminal and fail closed.
      const auto ack_payload = read_visible_contract_file(
          ack_path, "adapter acknowledgement", note_io);
      if (!ack_payload.has_value()) {
        throw std::runtime_error(
            "adapter release became visible before acknowledgement");
      }
      AckRecord acknowledgement = validate_canonical_ack(
          *ack_payload, state_id, pass, schemas);
      if (sha256_hex(*ack_payload) != release.ack_sha256) {
        throw std::runtime_error(
            "adapter release ack_sha256 does not bind canonical acknowledgement");
      }
      return acknowledgement;
    }
    std::this_thread::sleep_for(poll_interval);
  }
}

}  // namespace durable_pass_release
