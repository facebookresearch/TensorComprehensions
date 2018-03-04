add_subdirectory(polyhedral)

find_library(CUDA_NVRTC_LIBRARIES nvrtc
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

add_library(
  tc_core

  SHARED

  flags.cc
  compilation_cache.cc
  cuda.cc
  execution_engine.cc
  tc_executor.cc
  mapping_options.cc
  mapping_options_cpp_printer.cc
  rtc.cc
  islpp.cc

  tc2halide.cc

  halide2isl.cc
  halide_utils.cc

  utils/cuda_info.cc

  polyhedral/codegen_cuda.cc
  polyhedral/codegen_llvm.cc
  polyhedral/llvm_jit.cc
  polyhedral/mapped_scop.cc
  polyhedral/mapping_types.cc
  polyhedral/memory_promotion.cc
  polyhedral/memory_promotion_heuristic.cc
  polyhedral/reduction_matcher.cc
  polyhedral/schedule_isl_conversion.cc
  polyhedral/schedule_transforms.cc
  polyhedral/schedule_tree.cc
  polyhedral/schedule_tree_elem.cc
  polyhedral/schedule_print.cc
  polyhedral/scop.cc
  polyhedral/separation.cc
  polyhedral/tighten_launch_bounds.cc
  polyhedral/unroll.cc
  ${CMAKE_CURRENT_BINARY_DIR}/polyhedral/libpaths.cc
)

target_include_directories(tc_core PUBLIC ${PROJECT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS})

target_link_libraries(
  tc_core

  ${CUDA_CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_NVRTC_LIBRARIES}
  ${HALIDE_LIBRARIES}
  ${ISL_LIBRARIES}
  -lLLVM

  tc_lang
  tc_version
  tc_proto
)

install(
  TARGETS
  tc_core

  DESTINATION lib
)
