add_library(u24_decode SHARED native/u24_decode.c)
set_target_properties(u24_decode PROPERTIES OUTPUT_NAME u24_decode)

add_library(delta16_decode SHARED native/delta16_decode.c)
set_target_properties(delta16_decode PROPERTIES OUTPUT_NAME delta16_decode)

add_library(fixed_block_compositor SHARED
  native/fixed_block_compositor.cpp
  native/u24_decode.c)
target_compile_options(fixed_block_compositor PRIVATE -O3 -march=native -mavx2 -mfma)
set_target_properties(fixed_block_compositor PROPERTIES
  OUTPUT_NAME fixed_block_compositor)

add_library(packed_fixed_block_compositor SHARED
  native/packed_fixed_block_compositor.cpp
  native/u24_decode.c)
target_compile_options(packed_fixed_block_compositor PRIVATE -O3 -march=native -mavx2 -mfma)
set_target_properties(packed_fixed_block_compositor PROPERTIES
  OUTPUT_NAME packed_fixed_block_compositor)

add_library(leaf_posting_merge SHARED native/leaf_posting_merge.cpp)
target_compile_options(leaf_posting_merge PRIVATE -O3 -march=native -mavx2 -mfma)
set_target_properties(leaf_posting_merge PROPERTIES
  OUTPUT_NAME leaf_posting_merge)

add_library(fused_leaf_topk SHARED native/fused_leaf_topk.cpp)
target_compile_options(fused_leaf_topk PRIVATE
  -O3 -march=native -mavx2 -mfma
  -fno-fast-math -fno-associative-math -ffp-contract=off)
set_target_properties(fused_leaf_topk PROPERTIES
  OUTPUT_NAME fused_leaf_topk)

add_library(elias_fano_topk SHARED native/elias_fano_topk.cpp)
target_compile_options(elias_fano_topk PRIVATE
  -O3 -march=native -mavx2
  -fno-fast-math -fno-associative-math -ffp-contract=off)
set_target_properties(elias_fano_topk PROPERTIES
  OUTPUT_NAME elias_fano_topk)
