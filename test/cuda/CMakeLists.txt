include_directories(.)
include_directories(..)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Finally, add tests
################################################################################

################################################################################
# Basic tests of external functionalities we rely upon
################################################################################
add_executable(test_basic_gpu test_basic_gpu.cc)
add_test(test_basic_gpu test_basic_gpu)
target_link_libraries(test_basic_gpu ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARIES} ${CUDA_NVRTC_LIBRARIES} ${GOOGLE_LIBRARIES} pthread )

################################################################################
# Core GPU library tests
################################################################################
add_executable(test_autotuner_utility test_autotuner_utility.cc)
add_test(test_autotuner_utility test_autotuner_utility)
target_link_libraries(
    test_autotuner_utility
    ${GTEST_LIBRARIES}
    ${ATEN_LIBRARIES}
    tc_cuda
    tc_autotuner
    tc_aten
    pthread)

################################################################################
# ATen execution based test
################################################################################
set(ATEN_TEST_FILES
  test_autotuner
  test_compilation_cache
  test_compile_and_run
  test_tc_mapper
  test_tc_mapper_bugs
  test_corner_cases
)
foreach(i ${ATEN_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(
      ${i}
      ${GTEST_LIBRARIES}
      ${ATEN_LIBRARIES}
      tc_autotuner
      tc_aten
      pthread)
endforeach()
