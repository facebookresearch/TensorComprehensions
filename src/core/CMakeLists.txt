add_library(
  tc_core

  SHARED

  flags.cc
  mapping_options.cc
  mapping_options_cpp_printer.cc
  tc_executor.cc
  islpp.cc

  tc2halide.cc

  halide2isl.cc
  halide_utils.cc

  polyhedral/codegen.cc
  polyhedral/mapping_types.cc
  polyhedral/memory_promotion.cc
  polyhedral/reduction_matcher.cc
  polyhedral/schedule_isl_conversion.cc
  polyhedral/schedule_transforms.cc
  polyhedral/schedule_tree.cc
  polyhedral/schedule_tree_elem.cc
  polyhedral/schedule_print.cc
  polyhedral/scop.cc
  polyhedral/separation.cc
  polyhedral/unroll.cc

  # Builds and runs without the CUDA SDK and used in cross-compilation tests
  cuda/cuda_mapping_options.cc
  cuda/cuda_mapping_options_cpp_printer.cc
  polyhedral/cuda/codegen.cc
  polyhedral/cuda/mapped_scop.cc
  polyhedral/cuda/mapping_types.cc
  polyhedral/cuda/memory_promotion_heuristic.cc
  polyhedral/cuda/tighten_launch_bounds.cc
)
target_include_directories(tc_core PUBLIC ${PROJECT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  tc_core

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

add_library(
  tc_core_cpu

  SHARED

  cpu/cpu_tc_executor.cc

  polyhedral/codegen_llvm.cc
  polyhedral/llvm_jit.cc
)
target_include_directories(tc_core_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  tc_core_cpu

  ${HALIDE_LIBRARIES}
  ${ISL_LIBRARIES}
  -lLLVM

  tc_lang
  tc_version
  tc_proto
)
install(
  TARGETS
  tc_core_cpu

  DESTINATION lib
)

if (WITH_CUDA)
  find_library(CUDA_NVRTC_LIBRARIES nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

  add_library(
    tc_core_cuda

    SHARED

    cuda/cuda.cc
    cuda/cuda_compilation_cache.cc
    cuda/cuda_rtc.cc
    cuda/cuda_tc_executor.cc
  )
  target_include_directories(tc_core_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS})
  target_link_libraries(
    tc_core_cuda

    ${CUDA_CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIBRARIES}
    ${ISL_LIBRARIES}

    tc_lang
    tc_version
    tc_proto
  )
  target_link_libraries(tc_core tc_core_cuda)
  install(
    TARGETS
    tc_core_cuda

    DESTINATION lib
  )
endif()
