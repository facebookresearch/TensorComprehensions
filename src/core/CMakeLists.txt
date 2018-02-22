if (WITH_CUDA)
  find_library(CUDA_NVRTC_LIBRARIES nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
endif()

if (WITH_CUDA)
  add_library(
    tc_core_cuda

    SHARED

    compilation_cache.cc
    cuda.cc
    cuda/cuda_execution_engine.cc
    cuda/cuda_tc_executor.cc
    rtc.cc
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
endif()

add_library(
  tc_core

  SHARED

  execution_engine.cc
  flags.cc
  mapping_options.cc
  mapping_options_cpp_printer.cc
  tc_executor.cc
  islpp.cc

  tc2halide.cc

  halide2isl.cc
  halide_utils.cc

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

if(WITH_CUDA)
  target_link_libraries(tc_core tc_core_cuda)
endif()

install(
  TARGETS
  tc_core

  DESTINATION lib
)

if (WITH_CUDA)
  install(
    TARGETS
    tc_core_cuda

    DESTINATION lib
  )
endif()
