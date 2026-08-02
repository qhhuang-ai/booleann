find_package(Threads REQUIRED)
find_package(OpenMP)

set(SIEVE_HNSW_INCLUDE ${SIEVE_ROOT}/hnswtest/hnswlib)
if(NOT EXISTS ${SIEVE_HNSW_INCLUDE}/partitioned_hnsw.h)
  message(FATAL_ERROR "SIEVE_ROOT does not contain hnswtest/hnswlib/partitioned_hnsw.h")
endif()

function(add_predicate_cube_target target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SIEVE_HNSW_INCLUDE})
  target_compile_options(${target} PRIVATE -O3 -march=native -mavx2 -mfma)
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_predicate_cube_target(predicate_cube layout_selector_x3.cpp)
add_predicate_cube_target(predicate_cube_ablation mechanism_ablation.cpp)

add_test(NAME predicate_cube_self_test COMMAND predicate_cube --self-test)
