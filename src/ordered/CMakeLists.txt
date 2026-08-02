find_package(Threads REQUIRED)
find_package(OpenMP)
find_package(OpenSSL QUIET)

set(PARLAY_INCLUDE_DIRS
  ${PARLAYANN_ROOT}/parlaylib/include
  ${PARLAYANN_ROOT}/algorithms
  ${PARLAYANN_ROOT}/algorithms/utils
  ${PARLAYANN_ROOT}/algorithms/bench)

function(add_boole_ordered_target target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PARLAY_INCLUDE_DIRS})
  target_compile_options(${target} PRIVATE -O3 -march=native -mavx2 -mfma)
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_boole_ordered_target(build_range_blocks build_range_blocks.cpp)
add_boole_ordered_target(boole_range range_bench.cpp)
add_boole_ordered_target(boole_range_100k fixed_block_100k_adapter.cpp)
if(OpenSSL_FOUND)
  add_boole_ordered_target(boole_range_hierarchy hierarchical_range_bench.cpp)
  target_link_libraries(boole_range_hierarchy PRIVATE OpenSSL::Crypto)
else()
  message(STATUS "OpenSSL development files are unavailable; hierarchy executor is skipped")
endif()
