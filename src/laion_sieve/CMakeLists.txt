find_package(Threads REQUIRED)

set(SIEVE_HNSW_INCLUDE ${SIEVE_ROOT}/hnswtest/hnswlib)
if(NOT EXISTS ${SIEVE_HNSW_INCLUDE}/partitioned_hnsw.h)
  message(FATAL_ERROR "SIEVE_ROOT does not contain hnswtest/hnswlib/partitioned_hnsw.h")
endif()
if(NOT EXISTS ${PARLAYANN_ROOT}/algorithms/utils/beamSearch.h)
  message(FATAL_ERROR "PARLAYANN_ROOT does not contain algorithms/utils/beamSearch.h")
endif()

add_executable(
  laion_sieve_paired_formal
  laion1m_v20b_exact_equality_same_resident_paired_formal_runner.cpp)
set_property(TARGET laion_sieve_paired_formal PROPERTY CXX_STANDARD 20)
target_include_directories(
  laion_sieve_paired_formal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SIEVE_HNSW_INCLUDE}
  ${PARLAYANN_ROOT}/algorithms
  ${PARLAYANN_ROOT}/parlaylib/include)
target_compile_options(
  laion_sieve_paired_formal PRIVATE -O3 -march=native -DNDEBUG)
target_link_libraries(laion_sieve_paired_formal PRIVATE Threads::Threads)
