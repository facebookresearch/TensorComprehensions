include_directories(.)
include_directories(../tc)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Finally, add tests
################################################################################

################################################################################
# Basic tests of external functionalities we rely upon
################################################################################
add_executable(test_basic test_basic.cc)
add_test(test_basic test_basic)
target_link_libraries(test_basic ${GOOGLE_LIBRARIES} ${ISL_LIBRARIES} ${ATEN_LIBRARIES} pthread)
add_dependencies(test_basic generate_isl_cpp_h)

################################################################################
# Core library only tests
################################################################################
set(CORE_TEST_FILES
  test_core
  test_inference
  test_tc2halide
)

foreach(i ${CORE_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GTEST_LIBRARIES} tc_core pthread)
endforeach()

add_executable(test_check test_check.cc)
target_link_libraries(test_check ${GTEST_LIBRARIES} tc_core)

################################################################################
# CPP cuda_mapper crosscompile tests
################################################################################
set(CORE_TEST_FILES
  test_cuda_mapper
  test_cuda_mapper_memory_promotion
)

foreach(i ${CORE_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GTEST_LIBRARIES} tc_core_cuda_no_sdk pthread)
endforeach()

################################################################################
# CPP mapper tests, execution should use ATen C++ API
################################################################################
add_executable(test_mapper_llvm test_mapper_llvm.cc)
add_test(test_mapper_llvm test_mapper_llvm)
target_link_libraries(
  test_mapper_llvm

  ${GTEST_LIBRARIES}
  ${ATEN_LIBRARIES}
  -lLLVM

  tc_core_cpu tc_lang tc_aten pthread)

################################################################################
# Lang tests
################################################################################
set(TEST_TC_FILES
  test_lang
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/test/lang)

foreach(i ${TEST_TC_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GTEST_LIBRARIES} tc_lang pthread)
endforeach()

if (WITH_CUDA)
  add_subdirectory(cuda)
endif()
