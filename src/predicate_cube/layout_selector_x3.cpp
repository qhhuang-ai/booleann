// Layout selector with the exact compact-prefix64 range route.
// The categorical and SIEVE code paths are unchanged.
#define SIFT100K_LAYOUT_SELECTOR_V1_NO_MAIN
#define SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE
#include "layout_selector.cpp"
#undef SIFT100K_LAYOUT_SELECTOR_V3_COMPACT_RANGE
#undef SIFT100K_LAYOUT_SELECTOR_V1_NO_MAIN

namespace {

Result x3_scalar_top10(const uint8_t* query,
                       const std::vector<uint8_t>& vectors,
                       const std::vector<uint32_t>& ids) {
  require(vectors.size() == uint64_t(ids.size()) * D && ids.size() >= K,
          "X3 scalar test extent differs");
  std::vector<Hit> hits;
  hits.reserve(ids.size());
  for (size_t row = 0; row < ids.size(); ++row) {
    uint32_t distance = 0;
    for (uint32_t dimension = 0; dimension < D; ++dimension) {
      const int32_t delta =
          int32_t(vectors[uint64_t(row) * D + dimension]) - query[dimension];
      distance += uint32_t(delta * delta);
    }
    hits.emplace_back(distance, ids[row]);
  }
  std::sort(hits.begin(), hits.end());
  Result out;
  for (uint32_t rank = 0; rank < K; ++rank) out.ids[rank] = hits[rank].second;
  return out;
}

void run_x3_counterexamples() {
  std::array<uint8_t, D> query{};
  for (uint32_t rows : {10U, 255U, 256U, 257U, 383U, 384U, 385U}) {
    std::vector<uint8_t> vectors(uint64_t(rows) * D);
    std::vector<uint32_t> ids(rows);
    for (uint32_t row = 0; row < rows; ++row) {
      ids[row] = 1000 + row;
      for (uint32_t dimension = 0; dimension < D; ++dimension)
        vectors[uint64_t(row) * D + dimension] =
            uint8_t((row * 37 + dimension * 11 + 5) & 255);
    }
    require(x3_compact_range_scan(query.data(), vectors.data(), ids.data(),
                                  rows).ids ==
                x3_scalar_top10(query.data(), vectors, ids).ids,
            "X3 seed/batch boundary counterexample failed");
  }
  for (uint32_t mask = 0; mask < 16; ++mask) {
    constexpr uint32_t rows = X3_SEED_ROWS + 4;
    std::vector<uint8_t> vectors(uint64_t(rows) * D, 1);
    std::vector<uint32_t> ids(rows);
    std::iota(ids.begin(), ids.end(), 1000);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      ids[X3_SEED_ROWS + lane] = lane;
      uint8_t* value = vectors.data() + uint64_t(X3_SEED_ROWS + lane) * D;
      std::memset(value, 0, D);
      if (((mask >> lane) & 1U) == 0) std::memset(value, 2, 64);
    }
    require(x3_compact_range_scan(query.data(), vectors.data(), ids.data(),
                                  rows).ids ==
                x3_scalar_top10(query.data(), vectors, ids).ids,
            "X3 masks 0..15 counterexample failed");
  }
  {
    constexpr uint32_t rows = X3_SEED_ROWS + 4;
    std::vector<uint8_t> vectors(uint64_t(rows) * D, 2);
    std::vector<uint32_t> ids(rows);
    std::iota(ids.begin(), ids.end(), 1000);
    for (uint32_t row = 0; row < 10; ++row) {
      std::memset(vectors.data() + uint64_t(row) * D, 1, D);
      ids[row] = 100 + row;
    }
    uint8_t* tie = vectors.data() + uint64_t(X3_SEED_ROWS) * D;
    std::memset(tie, 0, D);
    tie[0] = 8;
    tie[1] = 8;
    ids[X3_SEED_ROWS] = 0;
    const Result actual = x3_compact_range_scan(
        query.data(), vectors.data(), ids.data(), rows);
    require(actual.ids == x3_scalar_top10(query.data(), vectors, ids).ids &&
                actual.ids[0] == 0,
            "X3 partial-equals-threshold tie counterexample failed");
  }
  std::cout
      << "X3_SELF_TEST status=PASS tests=masks_0_15,stable_compaction,"
         "strict_tie,seed_batch_boundaries,scalar_exactness"
      << " real_data_opened=false performance_timed=false" << std::endl;
}

}  // namespace

#ifndef SIFT100K_LAYOUT_SELECTOR_V3_NO_MAIN
int main(int argc, char** argv) {
  try {
    run_x3_counterexamples();
    if (argc == 2 && std::string(argv[1]) == "--self-test") return 0;
    return run_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
#endif
