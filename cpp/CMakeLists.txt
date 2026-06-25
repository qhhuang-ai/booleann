cmake_minimum_required(VERSION 3.16)
project(bci_cpp_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma -funroll-loops -DNDEBUG")

set(PARLAYANN_ROOT "/tmp/parlayann_filter" CACHE PATH "ParlayANN clone root")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PARLAYANN_ROOT}/parlaylib/include
  ${PARLAYANN_ROOT}/algorithms
  ${PARLAYANN_ROOT}/algorithms/utils
  ${PARLAYANN_ROOT}/algorithms/bench
)

find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(bci_main src/bci_main.cpp)
target_link_libraries(bci_main PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bci_main PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(yfcc_probe src/yfcc_probe.cpp)
target_link_libraries(yfcc_probe PRIVATE Threads::Threads)

add_executable(extract_sweet_spot_tags src/extract_sweet_spot_tags.cpp)
target_link_libraries(extract_sweet_spot_tags PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(extract_sweet_spot_tags PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(build_hamcg_shards src/build_hamcg_shards.cpp)
target_link_libraries(build_hamcg_shards PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(build_hamcg_shards PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(query_shard_test src/query_shard_test.cpp)
target_link_libraries(query_shard_test PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(query_shard_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bci_bench src/bci_bench.cpp)
target_link_libraries(bci_bench PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bci_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(build_shard_clusters src/build_shard_clusters.cpp)
target_link_libraries(build_shard_clusters PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(build_shard_clusters PRIVATE OpenMP::OpenMP_CXX)
endif()

# LAION float32 variants
add_executable(extract_sweet_spot_tags_laion src/extract_sweet_spot_tags_laion.cpp)
target_link_libraries(extract_sweet_spot_tags_laion PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(extract_sweet_spot_tags_laion PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(build_hamcg_shards_laion src/build_hamcg_shards_laion.cpp)
target_link_libraries(build_hamcg_shards_laion PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(build_hamcg_shards_laion PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(build_shard_clusters_laion src/build_shard_clusters_laion.cpp)
target_link_libraries(build_shard_clusters_laion PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(build_shard_clusters_laion PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(laion_bci_bench src/laion_bci_bench.cpp)
target_link_libraries(laion_bci_bench PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(laion_bci_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# SIFT100M variant (float32, dim=128, 100M base for 2-tag conjunction breadth experiment)
add_executable(sift100m_bci_bench src/sift100m_bci_bench.cpp)
target_link_libraries(sift100m_bci_bench PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sift100m_bci_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
