include_directories(.)
include_directories(../include)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Don't ask ... TODO: cleanup gtest
################################################################################
include_directories(${PROJECT_SOURCE_DIR}/third-party/googletest/googletest/include)
set(GTEST_LIBS gtest gtest_main)

set(GOOGLE_LIBS ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} ${GTEST_LIBS})

add_library(tc_test INTERFACE)
target_include_directories(tc_test INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tc_test INTERFACE ${GOOGLE_LIBS})


################################################################################
# Finally, add tests
################################################################################

################################################################################
# Basic tests of external functionalities we rely upon
################################################################################
add_executable(test_basic test_basic.cc)
add_test(test_basic test_basic)
target_link_libraries(test_basic ${GOOGLE_LIBS} ${ISL_LIBRARIES} ${ATEN_LIBRARIES})

if (WITH_CUDA)
  add_executable(test_basic_gpu test_basic_gpu.cc)
  set_target_properties(test_basic_gpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_CUDA_SDK}" )
  add_test(test_basic_gpu test_basic_gpu)
  target_link_libraries(test_basic_gpu ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARIES} ${CUDA_NVRTC_LIBRARIES} ${GOOGLE_LIBS} )
endif()

################################################################################
# Core library only tests
################################################################################
set(CORE_TEST_FILES
  test_core
  test_inference
  test_isl_functionality
  test_tc2halide
)

foreach(i ${CORE_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GOOGLE_LIBS} tc_core)
endforeach()

################################################################################
# Core GPU library tests
################################################################################
if (WITH_CUDA)
  add_executable(test_autotuner_utility test_autotuner_utility.cc)
  add_test(test_autotuner_utility test_autotuner_utility)
  target_link_libraries(test_autotuner_utility ${GOOGLE_LIBS} ${ATEN_LIBRARIES} tc_cuda tc_autotuner)
endif()


################################################################################
# ATen execution based test
################################################################################
if (WITH_CUDA)
  set(ATEN_TEST_FILES
    test_autotuner
    test_compilation_cache
    test_execution_engine
    test_execution_engine_cache
    test_execution_engine_db
    test_tc_mapper
    test_tc_mapper_bugs
    test_corner_cases
  )
  foreach(i ${ATEN_TEST_FILES})
    add_executable(${i} ${i}.cc)
    add_test(${i} ${i})
    target_link_libraries(
        ${i}

        ${GOOGLE_LIBS}
        ${ATEN_LIBRARIES}
        tc_autotuner)
  endforeach()
endif()

################################################################################
# CPP mapper tests, execution should use ATen C++ API
################################################################################
set(CORE_TEST_FILES
  test_mapper_memory_promotion
  test_mapper
)

foreach(i ${CORE_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GOOGLE_LIBS} tc_core_cuda_no_sdk)
endforeach()


add_executable(test_mapper_llvm test_mapper_llvm.cc)
add_test(test_mapper_llvm test_mapper_llvm)
target_link_libraries(
  test_mapper_llvm

  ${GOOGLE_LIBS}
  ${ATEN_LIBRARIES}
  -lLLVM

  tc_core_cpu tc_lang)

################################################################################
# TensorComprehensions tests
# No real need for NVCC if we only use NVRTC
# cuda_add_executable(test_matmul test_matmul.cu OPTIONS -std=c++11)
################################################################################

if (WITH_CAFFE2 AND WITH_CUDA)
  set(TEST_FILES
    test_caffe2
  )

  foreach(i ${TEST_FILES})
    add_executable(${i} ${i}.cc)
    add_test(${i} ${i})
    target_link_libraries(${i} ${GOOGLE_LIBS} pthread tc_c2)
  endforeach()
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
