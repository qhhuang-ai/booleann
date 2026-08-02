#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace yfcc_shortcut_residual_mask_v1 {

inline constexpr std::array<char, 8> kMagic{
    'Y', 'F', 'R', 'M', 'A', 'S', 'K', '1'};
inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kHeaderBytes = 64;
inline constexpr uint32_t kFlagLsbFirst = 1;

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;
  uint64_t rows;
  uint64_t edge_count;
  uint64_t payload_bytes;
  uint64_t current_graph_bytes;
  uint64_t control_graph_bytes;
  uint32_t max_degree;
  uint32_t flags;
};

static_assert(sizeof(Header) == kHeaderBytes);
static_assert(offsetof(Header, version) == 8);
static_assert(offsetof(Header, rows) == 16);
static_assert(offsetof(Header, max_degree) == 56);

inline constexpr uint64_t payload_bytes_for(uint64_t edge_count) {
  return edge_count / 8 + static_cast<uint64_t>(edge_count % 8 != 0);
}

inline bool bit_is_set(
    const uint8_t* payload, uint64_t edge_position) {
  return
      (payload[edge_position >> 3] &
       static_cast<uint8_t>(uint8_t{1} << (edge_position & 7))) != 0;
}

}  // namespace yfcc_shortcut_residual_mask_v1
