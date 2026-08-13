find_package(Threads REQUIRED)
find_package(OpenMP)

set(PARLAY_INCLUDE_DIRS
  ${PARLAYANN_ROOT}/parlaylib/include
  ${PARLAYANN_ROOT}/algorithms
  ${PARLAYANN_ROOT}/algorithms/utils
  ${PARLAYANN_ROOT}/algorithms/bench)

function(add_boole_categorical_target target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PARLAY_INCLUDE_DIRS})
  target_compile_options(${target} PRIVATE -O3 -march=native -mavx2 -mfma)
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_boole_categorical_target(boole_yfcc categorical_bench.cpp)
add_boole_categorical_target(boole_sift100m sift100m_bench.cpp)
add_boole_categorical_target(build_sift100m_shards build_sift100m_shards.cpp)
