################################################################################
# tc_core
#
# Core CPU library with cross-compilation capabilities linked from
# tc_core_cuda
################################################################################
add_library(
  tc_core

  SHARED

  flags.cc
  mapping_options.cc
  mapping_options_cpp_printer.cc
  islpp.cc
  compiler.cc
  tensor.cc

  tc2halide.cc

  halide2isl.cc
  halide_utils.cc

  polyhedral/body.cc
  polyhedral/codegen.cc
  polyhedral/memory_promotion.cc
  polyhedral/reduction_matcher.cc
  polyhedral/schedule_isl_conversion.cc
  polyhedral/schedule_transforms.cc
  polyhedral/schedule_tree.cc
  polyhedral/schedule_tree_elem.cc
  polyhedral/schedule_print.cc
  polyhedral/schedule_utils.cc
  polyhedral/scop.cc
  polyhedral/unroll.cc
  polyhedral/utils.cc
)
target_include_directories(tc_core PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  tc_core

  gflags
  glog

  ${HALIDE_LIBRARIES}
  -lLLVM

  tc_isl
  tc_lang
  tc_version
  tc_proto
)
add_dependencies(tc_core generate_isl_cpp_h)
install(
  TARGETS
  tc_core

  DESTINATION lib
)

################################################################################
# tc_core_cpu
#
# Core CPU library for compiling to LLVMIR and running on CPUs
################################################################################
add_library(
  tc_core_cpu

  SHARED

  cpu/cpu_tc_executor.cc
  cpu/cpu_mapping_options.cc
  cpu/cpu_mapping_options_cpp_printer.cc

  polyhedral/codegen_llvm.cc
  polyhedral/llvm_jit.cc
  polyhedral/cpu/mapped_scop.cc
)
target_include_directories(tc_core_cpu PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  tc_core_cpu

  ${HALIDE_LIBRARIES}
  -lLLVM

  tc_lang
  tc_version
  tc_proto
  tc_core
)
install(
  TARGETS
  tc_core_cpu

  DESTINATION lib
)

set(TC_CUDA_CROSS_COMPILE_FILES
    cuda/cuda_mapping_options.cc
    cuda/cuda_mapping_options_cpp_printer.cc
    polyhedral/cuda/codegen.cc
    polyhedral/cuda/mapped_scop.cc
    polyhedral/cuda/mapping_types.cc
    polyhedral/cuda/memory_promotion_heuristic.cc
    polyhedral/cuda/tighten_launch_bounds.cc
)

################################################################################
# tc_core_cuda_no_sdk
#
# Core CUDA cross-compilation library for compiling to CUDA and not running
# Builds **without** the CUDA SDK and used in cross-compilation tests
################################################################################
add_library(
  tc_core_cuda_no_sdk

  SHARED

  ${TC_CUDA_CROSS_COMPILE_FILES}
)
# Sets NO_CUDA_SDK to compile without SDK
set_target_properties(tc_core_cuda_no_sdk
                      PROPERTIES COMPILE_FLAGS
                      "${CMAKE_CXX_FLAGS} -DNO_CUDA_SDK")
target_include_directories(tc_core_cuda_no_sdk PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  tc_core_cuda_no_sdk

  ${HALIDE_LIBRARIES}
  -lLLVM

  tc_lang
  tc_version
  tc_proto
  tc_core
)
install(
  TARGETS
  tc_core_cuda_no_sdk

  DESTINATION lib
)

################################################################################
# tc_cuda
#
# Core CUDA library for compiling to PTX and running on actual GPUs
################################################################################
if (WITH_CUDA)
  find_library(CUDA_NVRTC_LIBRARIES nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

  add_library(
    tc_cuda

    SHARED

    ${TC_CUDA_CROSS_COMPILE_FILES}

    # Files needed for execution
    cuda/cuda.cc
    cuda/cuda_rtc.cc
    cuda/cuda_tc_executor.cc
  )
  target_include_directories(tc_cuda PUBLIC ${LLVM_INCLUDE_DIRS})
  target_link_libraries(
    tc_cuda

    ${CUDA_CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIBRARIES}

    tc_lang
    tc_version
    tc_proto
    tc_core
  )
  install(
    TARGETS
    tc_cuda

    DESTINATION lib
  )
endif()
