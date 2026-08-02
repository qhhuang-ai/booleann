// Exact, one-layer causal controls for the frozen SIFT100K X3 design.
//
// This is a project adapter.  It includes the frozen X3 translation unit and
// does not edit the official SIEVE tree or the frozen four-family workload.
#define SIFT100K_LAYOUT_SELECTOR_V3_NO_MAIN
#include "layout_selector_x3.cpp"
#undef SIFT100K_LAYOUT_SELECTOR_V3_NO_MAIN

namespace {

enum class CausalArm { Full, SupportOff, LayoutOff, PlanOff };

const char* causal_arm_name(CausalArm arm) {
  switch (arm) {
    case CausalArm::Full: return "FULL";
    case CausalArm::SupportOff: return "SUPPORT_OFF";
    case CausalArm::LayoutOff: return "LAYOUT_OFF";
    case CausalArm::PlanOff: return "PLAN_OFF";
  }
  fail("unknown causal arm");
}

CausalArm parse_causal_arm(const std::string& value) {
  if (value == "FULL") return CausalArm::Full;
  if (value == "SUPPORT_OFF") return CausalArm::SupportOff;
  if (value == "LAYOUT_OFF") return CausalArm::LayoutOff;
  if (value == "PLAN_OFF") return CausalArm::PlanOff;
  fail("causal arm must be FULL, SUPPORT_OFF, LAYOUT_OFF, or PLAN_OFF");
}

struct PrefixStats {
  uint64_t candidate_rows = 0;
  uint64_t prefix_rows = 0;
  uint64_t suffix_rows = 0;
};

// The frozen X3 arithmetic and batching, generalized only over physical-row
// and global-ID accessors.  Contiguous FULL uses the original frozen function;
// this accessor form is used by the posting/gather and post-timing audit paths.
template <class PositionAt, class IdAt>
Result causal_x3_indexed_scan(const uint8_t* query,
                              const std::vector<uint8_t>& vectors,
                              uint32_t rows, PositionAt position_at,
                              IdAt id_at, PrefixStats* stats = nullptr) {
  require(query != nullptr && rows >= K,
          "causal indexed X3 support is below top-k");
  const X3ExpandedQuery expanded = x3_expand_query(query);
  X3Top10 top10;
  const uint32_t seed_rows = std::min(X3_SEED_ROWS, rows);
  uint32_t index = 0;
  for (; index + 4 <= seed_rows; index += 4) {
    const uint8_t* candidates[4] = {
        vectors.data() + uint64_t(position_at(index)) * D,
        vectors.data() + uint64_t(position_at(index + 1)) * D,
        vectors.data() + uint64_t(position_at(index + 2)) * D,
        vectors.data() + uint64_t(position_at(index + 3)) * D};
    uint32_t first[4]{}, second[4]{};
    x3_half_four(expanded, 0, candidates, first);
    x3_half_four(expanded, 2, candidates, second);
    for (uint32_t lane = 0; lane < 4; ++lane)
      top10.consider(first[lane] + second[lane], id_at(index + lane));
  }
  for (; index < seed_rows; ++index) {
    const uint32_t position = position_at(index);
    top10.consider(sieve_v25_packed::l2_sq_uint8_avx2(
                       query, vectors.data() + uint64_t(position) * D),
                   id_at(index));
  }
  require(top10.full(), "causal indexed X3 seed is incomplete");

  std::array<X3Survivor, X3_BATCH_ROWS> survivors{};
  uint64_t suffix_rows = seed_rows;
  while (index < rows) {
    const uint32_t batch_end = std::min<uint32_t>(rows, index + X3_BATCH_ROWS);
    const uint32_t threshold = top10.threshold();
    uint32_t survivor_count = 0;
    for (; index + 4 <= batch_end; index += 4) {
      const uint8_t* candidates[4] = {
          vectors.data() + uint64_t(position_at(index)) * D,
          vectors.data() + uint64_t(position_at(index + 1)) * D,
          vectors.data() + uint64_t(position_at(index + 2)) * D,
          vectors.data() + uint64_t(position_at(index + 3)) * D};
      uint32_t partial[4]{};
      x3_half_four(expanded, 0, candidates, partial);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        if (partial[lane] <= threshold)
          survivors[survivor_count++] =
              X3Survivor{index + lane, partial[lane]};
      }
    }
    for (; index < batch_end; ++index) {
      const uint32_t position = position_at(index);
      const uint32_t partial = x3_half_one(
          expanded, 0, vectors.data() + uint64_t(position) * D);
      if (partial <= threshold)
        survivors[survivor_count++] = X3Survivor{index, partial};
    }
    require(survivor_count <= X3_BATCH_ROWS,
            "causal indexed X3 survivor overflow");
    suffix_rows += survivor_count;
    uint32_t survivor = 0;
    for (; survivor + 4 <= survivor_count; survivor += 4) {
      const uint8_t* candidates[4] = {
          vectors.data() +
              uint64_t(position_at(survivors[survivor].position)) * D,
          vectors.data() +
              uint64_t(position_at(survivors[survivor + 1].position)) * D,
          vectors.data() +
              uint64_t(position_at(survivors[survivor + 2].position)) * D,
          vectors.data() +
              uint64_t(position_at(survivors[survivor + 3].position)) * D};
      uint32_t suffix[4]{};
      x3_half_four(expanded, 2, candidates, suffix);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const X3Survivor& kept = survivors[survivor + lane];
        top10.consider(kept.partial + suffix[lane], id_at(kept.position));
      }
    }
    for (; survivor < survivor_count; ++survivor) {
      const X3Survivor& kept = survivors[survivor];
      const uint32_t position = position_at(kept.position);
      top10.consider(
          kept.partial + x3_half_one(
                             expanded, 2,
                             vectors.data() + uint64_t(position) * D),
          id_at(kept.position));
    }
  }
  if (stats) {
    stats->candidate_rows += rows;
    stats->prefix_rows += rows - seed_rows;
    stats->suffix_rows += suffix_rows;
  }
  return top10.finish();
}

class SupportOffEngine {
 public:
  SupportOffEngine(const Data& data, const PredicateCube& cube)
      : data_(data), cube_(cube) {}

  Result query(const Query& query) const {
    // No offset or support directory is consulted.  Filtering the full cube
    // retains exactly the physical candidate order of every frozen X route.
    if (query.spec.family == Family::Range) {
      std::vector<uint32_t> positions;
      positions.reserve(N);
      const auto& globals = cube_.pos_to_global();
      for (uint32_t position = 0; position < globals.size(); ++position)
        if (matches(data_, query.spec, globals[position]))
          positions.push_back(position);
      return causal_x3_indexed_scan(
          query.vector.data(), cube_.vectors_by_pos(), positions.size(),
          [&](uint32_t index) { return positions[index]; },
          [&](uint32_t index) {
            return cube_.pos_to_global()[positions[index]];
          });
    }
    FourWayTop10 scan;
    scan.set_query(query.vector.data());
    const auto& vectors = cube_.vectors_by_pos();
    const auto& globals = cube_.pos_to_global();
    for (uint32_t position = 0; position < globals.size(); ++position) {
      const uint32_t global = globals[position];
      if (matches(data_, query.spec, global))
        scan.add(vectors.data() + uint64_t(position) * D, global);
    }
    return scan.finish(query.vector.data());
  }

 private:
  const Data& data_;
  const PredicateCube& cube_;
};

class LayoutOffEngine {
 public:
  LayoutOffEngine(const Data& data, const PredicateCube& cube)
      : data_(data), cube_(cube) {}

  Result query(const Query& query) const {
    const std::vector<uint32_t> ids = support(data_, query.spec);
    const auto& positions = cube_.global_to_pos();
    if (query.spec.family == Family::Range) {
      return causal_x3_indexed_scan(
          query.vector.data(), cube_.vectors_by_pos(), ids.size(),
          [&](uint32_t index) { return positions[ids[index]]; },
          [&](uint32_t index) { return ids[index]; });
    }
    FourWayTop10 scan;
    scan.set_query(query.vector.data());
    const auto& vectors = cube_.vectors_by_pos();
    for (uint32_t id : ids) {
      const uint32_t position = positions[id];
      scan.add(vectors.data() + uint64_t(position) * D, id);
    }
    return scan.finish(query.vector.data());
  }

 private:
  const Data& data_;
  const PredicateCube& cube_;
};

void run_causal_counterexamples() {
  std::array<uint8_t, D> query{};
  std::vector<uint8_t> vectors(uint64_t(20) * D, 9);
  std::vector<uint32_t> globals(20);
  std::iota(globals.begin(), globals.end(), 0);
  // Ten predicate members are deliberately farther than ten nonmembers.
  for (uint32_t row = 0; row < 10; ++row)
    std::memset(vectors.data() + uint64_t(row) * D, 0, D);
  for (uint32_t row = 10; row < 20; ++row)
    std::memset(vectors.data() + uint64_t(row) * D, uint8_t(row - 9), D);
  std::vector<uint32_t> localized(globals.begin() + 10, globals.end());
  FourWayTop10 filtered;
  filtered.set_query(query.data());
  for (uint32_t row = 0; row < 20; ++row)
    if (row >= 10)
      filtered.add(vectors.data() + uint64_t(row) * D, globals[row]);
  const Result filtered_result = filtered.finish(query.data());
  require(filtered_result.ids ==
              x3_scalar_top10(query.data(),
                              std::vector<uint8_t>(vectors.begin() + 10 * D,
                                                   vectors.end()),
                              localized).ids,
          "support-off predicate filtering counterexample failed");
  require(filtered_result.ids != x3_scalar_top10(query.data(), vectors,
                                                  globals).ids,
          "support-off negative control did not distinguish nonmembers");

  // A nonidentity physical permutation must be addressed through its inverse.
  std::array<uint32_t, 12> permutation{5, 1, 9, 0, 11, 3,
                                       8, 2, 10, 4, 7, 6};
  std::vector<uint8_t> logical(uint64_t(12) * D);
  std::vector<uint8_t> physical(uint64_t(12) * D);
  std::vector<uint32_t> inverse(12), ids(12);
  std::iota(ids.begin(), ids.end(), 0);
  for (uint32_t id = 0; id < 12; ++id) {
    std::memset(logical.data() + uint64_t(id) * D, uint8_t(id + 1), D);
    const uint32_t position = permutation[id];
    inverse[id] = position;
    std::memcpy(physical.data() + uint64_t(position) * D,
                logical.data() + uint64_t(id) * D, D);
  }
  const Result gathered = causal_x3_indexed_scan(
      query.data(), physical, ids.size(),
      [&](uint32_t index) { return inverse[ids[index]]; },
      [&](uint32_t index) { return ids[index]; });
  require(gathered.ids == x3_scalar_top10(query.data(), logical, ids).ids,
          "layout-off inverse-map counterexample failed");
  const Result wrong_offset = causal_x3_indexed_scan(
      query.data(), physical, ids.size(),
      [&](uint32_t index) { return ids[index]; },
      [&](uint32_t index) { return ids[index]; });
  require(wrong_offset.ids != gathered.ids,
          "layout-off negative offset control did not distinguish permutation");

  std::cout << "CAUSAL_SELF_TEST status=PASS"
            << " tests=support_filter,layout_inverse_map,x3_prefix_ties"
            << " real_data_opened=false performance_timed=false" << std::endl;
}

void emit_causal_precalibration(const Options& options, CausalArm arm) {
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
          "causal runner requires C++17, optimization, and AVX2");
  std::cout
      << "PRECALIBRATION_RECEIPT phase=" << options.phase
      << " system=" << causal_arm_name(arm)
      << " source_files=mechanism_ablation.cpp,layout_selector_x3.cpp,layout_selector.cpp,mixed_runner.cpp,predicate_cube.hpp,pair_major_layout.hpp"
      << " compiler=" << __VERSION__
      << " cplusplus=" << __cplusplus
      << " required_compile_flags=-std=c++17,-O3,-march=native,-mavx2"
      << " avx2=true optimization_enabled=true"
      << " binary_path=/proc/self/exe"
      << " source_frozen_before_calibration=true"
      << " evaluation_option_accepted=true" << std::endl;
}

uint64_t causal_transient_upper_bound(CausalArm arm) {
  if (arm == CausalArm::SupportOff || arm == CausalArm::LayoutOff)
    return uint64_t(N) * sizeof(uint32_t) + 8192;
  if (arm == CausalArm::Full) return X3_TRANSIENT_SCRATCH_UPPER_BOUND_BYTES;
  return 4096;
}

void emit_mechanism_receipt(const Data& data, const PredicateCube& cube,
                            const std::vector<Query>& workload,
                            CausalArm arm) {
  uint64_t support_rows = 0;
  uint64_t transitions = 0;
  uint64_t nonadjacent = 0;
  PrefixStats range_stats;
  for (const Query& query : workload) {
    const std::vector<uint32_t> ids = support(data, query.spec);
    support_rows += ids.size();
    for (size_t index = 1; index < ids.size(); ++index) {
      ++transitions;
      const uint32_t previous = cube.global_to_pos()[ids[index - 1]];
      const uint32_t current = cube.global_to_pos()[ids[index]];
      if (current != previous + 1) ++nonadjacent;
    }
    if (query.spec.family == Family::Range) {
      const Slice slice = cube.range_slice(query.spec.lo, query.spec.hi);
      PrefixStats one;
      const Result measured = causal_x3_indexed_scan(
          query.vector.data(), cube.vectors_by_pos(), slice.rows(),
          [&](uint32_t index) { return slice.begin + index; },
          [&](uint32_t index) {
            return cube.pos_to_global()[slice.begin + index];
          }, &one);
      require(measured.ids == x3_compact_range_scan(
                                  query.vector.data(),
                                  cube.vectors_by_pos().data() +
                                      uint64_t(slice.begin) * D,
                                  cube.pos_to_global().data() + slice.begin,
                                  slice.rows()).ids,
              "post-timing prefix mechanism audit differs from frozen X3");
      range_stats.candidate_rows += one.candidate_rows;
      range_stats.prefix_rows += one.prefix_rows;
      range_stats.suffix_rows += one.suffix_rows;
    }
  }
  const uint64_t checks =
      arm == CausalArm::SupportOff ? uint64_t(workload.size()) * N : 0;
  const uint64_t gathered =
      arm == CausalArm::LayoutOff ? support_rows : 0;
  const uint64_t suffix_rows =
      arm == CausalArm::PlanOff ? range_stats.candidate_rows
                                : range_stats.suffix_rows;
  std::cout
      << "MECHANISM_RECEIPT arm=" << causal_arm_name(arm)
      << " workload_queries=" << workload.size()
      << " exact_support_rows=" << support_rows
      << " predicate_rows_checked=" << checks
      << " offset_directory_used="
      << ((arm == CausalArm::Full || arm == CausalArm::PlanOff) ? "true"
                                                                : "false")
      << " posting_directory_used="
      << (arm == CausalArm::LayoutOff ? "true" : "false")
      << " gathered_vector_rows=" << gathered
      << " gather_transitions="
      << (arm == CausalArm::LayoutOff ? transitions : 0)
      << " gather_nonadjacent_transitions="
      << (arm == CausalArm::LayoutOff ? nonadjacent : 0)
      << " range_candidate_rows=" << range_stats.candidate_rows
      << " range_prefix64_rows="
      << (arm == CausalArm::PlanOff ? 0 : range_stats.prefix_rows)
      << " range_suffix_rows=" << suffix_rows
      << " family_routing_preserved_for_single_layer=true"
      << " measured_after_all_timing=true" << std::endl;
}

int causal_main(int argc, char** argv) {
  run_x3_counterexamples();
  run_causal_counterexamples();
  if (argc == 2 && std::string(argv[1]) == "--self-test") return 0;

  std::string arm_text;
  std::vector<char*> forwarded;
  forwarded.reserve(size_t(argc));
  forwarded.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    require(index + 1 < argc, "missing causal runner option value");
    if (key == "--causal-arm") {
      require(arm_text.empty(), "causal arm specified twice");
      arm_text = argv[++index];
    } else {
      forwarded.push_back(argv[index]);
      forwarded.push_back(argv[++index]);
    }
  }
  const CausalArm arm = parse_causal_arm(arm_text);
  Options options = parse_options(int(forwarded.size()), forwarded.data());
  require(options.phase == "execute" && options.system == "X",
          "causal closure accepts only execute/X");
  options.system = causal_arm_name(arm);
  emit_causal_precalibration(options, arm);

  Data data = load_data(options.base, options.metadata, options.query,
                        options.attribute);
  uint64_t posting_rows = 0;
  const uint64_t common = support_catalog_bytes(data, &posting_rows);
  require(common == 3500000 && posting_rows == 800000,
          "causal common support catalog differs");
  std::cout << "SUPPORT_CATALOG_RECEIPT common_catalog_bytes=" << common
            << " label_bytes=300000 posting_rows=" << posting_rows
            << " posting_bytes=" << posting_rows * 4
            << " accounting=size_times_element_width"
            << " allocator_overhead_in_rss_only=true" << std::endl;

  const Workload development =
      load_workload(data, options.development_workload_tsv, false);
  const Workload evaluation =
      load_workload(data, options.evaluation_workload_tsv, true);
  require_split_disjoint(development, evaluation);

  PredicateCube cube(data.base, data.bin, data.cat_a, data.cat_b, D, BINS,
                     CAT_A, CAT_B);
  require(cube.logical_charge().absolute_serving_bytes() == X_BASE_BYTES &&
              cube.audit().passed() && cube.rows() == N &&
              cube.cells() == BINS * CAT_A * CAT_B &&
              cube.offset_count() == BINS * CAT_A * CAT_B + 1,
          "causal cube partition or charge differs");
  const uint64_t transient = causal_transient_upper_bound(arm);
  require(X_BASE_BYTES + common + transient < options.budget_bytes,
          "causal arm including transient scratch exceeds 24 MiB");
  emit_layout(causal_arm_name(arm), common, X_BASE_BYTES, 0,
              options.budget_bytes, false, true,
              "cube_vectors,cube_offsets,pos_to_global,global_to_pos,labels,A_postings,B_postings,pair_postings,bin_postings,dyadic_postings,bounded_transient_scratch",
              arm == CausalArm::SupportOff
                  ? "cube_vectors,pos_to_global,labels,bounded_transient_scratch"
              : arm == CausalArm::LayoutOff
                  ? "cube_vectors,global_to_pos,A_postings,pair_postings,bin_postings,bounded_transient_scratch"
                  : "cube_vectors,cube_offsets,pos_to_global,bounded_transient_scratch");
  std::cout << "LAYOUT_INVARIANT package=" << causal_arm_name(arm)
            << " order=bin,A,B,global_id cells=" << cube.cells()
            << " audit_pass=true vector_payload_count=1" << std::endl;
  std::cout << "CAUSAL_ARM_RECEIPT arm=" << causal_arm_name(arm)
            << " persistent_logical_bytes=" << X_BASE_BYTES + common
            << " transient_scratch_upper_bound_bytes=" << transient
            << " total_including_transient_upper_bound_bytes="
            << X_BASE_BYTES + common + transient
            << " budget_bytes=" << options.budget_bytes
            << " exact_required=true"
            << " support_atoms_complete_partition=true"
            << " official_baseline_core_modified=false"
            << " frozen_workload_required=true" << std::endl;

  SelectorMetrics metrics;
  if (arm == CausalArm::Full) {
    X3Engine engine(cube);
    auto query_fn = [&](const Query& query) { return engine.query(query); };
    metrics = execute_workload(data, evaluation.queries, options, true,
                               query_fn);
  } else if (arm == CausalArm::SupportOff) {
    SupportOffEngine engine(data, cube);
    auto query_fn = [&](const Query& query) { return engine.query(query); };
    metrics = execute_workload(data, evaluation.queries, options, true,
                               query_fn);
  } else if (arm == CausalArm::LayoutOff) {
    LayoutOffEngine engine(data, cube);
    auto query_fn = [&](const Query& query) { return engine.query(query); };
    metrics = execute_workload(data, evaluation.queries, options, true,
                               query_fn);
  } else {
    XEngine engine(cube);
    auto query_fn = [&](const Query& query) { return engine.query(query); };
    metrics = execute_workload(data, evaluation.queries, options, true,
                               query_fn);
  }
  emit_mechanism_receipt(data, cube, evaluation.queries, arm);

  const double recall =
      double(metrics.recall_hits) / metrics.recall_denominator;
  std::cout << "PROCESS_RSS_RECEIPT system=" << causal_arm_name(arm)
            << " peak_rss_bytes=" << peak_rss_bytes()
            << " logical_bytes_separate_from_rss=true" << std::endl;
  std::cout << "RUN_RECEIPT phase=execute system=" << causal_arm_name(arm)
            << " cycles=" << options.cycles
            << " family_rows=100 queries=" << metrics.queries
            << " elapsed_ns=" << metrics.elapsed_ns
            << " qps=" << std::setprecision(17) << metrics.qps()
            << " recall=" << recall
            << " exact_required=true"
            << " validation_after_all_timing=true warm_passes=1"
            << " family_order=" << options.family_order
            << " evaluation_path_recorded=true" << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return causal_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "FATAL " << error.what() << std::endl;
    return 2;
  }
}
