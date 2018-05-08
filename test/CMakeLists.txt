include_directories(.)
include_directories(../tc)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Don't ask ... TODO: cleanup gtest
################################################################################
include_directories(${PROJECT_SOURCE_DIR}/third-party/googletest/googletest/include)
set(GTEST_LIBS gtest gtest_main)

set(GOOGLE_LIBS ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} ${GTEST_LIBS})

################################################################################
# Finally, add tests
################################################################################

################################################################################
# Basic tests of external functionalities we rely upon
################################################################################
add_executable(test_basic test_basic.cc)
add_test(test_basic test_basic)
target_link_libraries(test_basic ${GOOGLE_LIBS} ${ISL_LIBRARIES} ${ATEN_LIBRARIES})

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
  target_link_libraries(${i} ${GOOGLE_LIBS} tc_core)
endforeach()

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
  target_link_libraries(${i} ${GOOGLE_LIBS} tc_core_cuda_no_sdk)
endforeach()

################################################################################
# CPP mapper tests, execution should use ATen C++ API
################################################################################
add_executable(test_mapper_llvm test_mapper_llvm.cc)
add_test(test_mapper_llvm test_mapper_llvm)
target_link_libraries(
  test_mapper_llvm

  ${GOOGLE_LIBS}
  ${ATEN_LIBRARIES}
  -lLLVM

  tc_core_cpu tc_lang tc_aten)

if (WITH_TAPIR)
  add_executable(test_mapper_tapir test_mapper_tapir.cc)
  add_test(test_mapper_tapir test_mapper_tapir)
  target_link_libraries(
    test_mapper_tapir

    ${GOOGLE_LIBS}
    ${ATEN_LIBRARIES}
    -lLLVM

    tc_core_cpu tc_lang tc_aten)
endif()

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
  target_link_libraries(${i} ${GOOGLE_LIBS} pthread tc_lang)
endforeach()

if (WITH_CUDA)
  add_subdirectory(cuda)
endif()

if (WITH_CAFFE2)
  add_subdirectory(caffe2)
endif()
