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
set(BASIC_TEST_FILES
  test_basic
  test_basic_gpu
)

foreach(i ${BASIC_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
endforeach()

target_link_libraries(test_basic ${GOOGLE_LIBS} ${ISL_LIBRARIES} tc_aten)
target_link_libraries(test_basic_gpu ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARIES} ${CUDA_NVRTC_LIBRARIES} ${GOOGLE_LIBS} )

################################################################################
# Core library only tests
################################################################################
set(CORE_TEST_FILES
  test_autotuner_utility
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

target_link_libraries(test_autotuner_utility tc_autotuner tc_aten)

################################################################################
# ATen execution based test
################################################################################
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
      tc_aten
      tc_autotuner)
endforeach()

################################################################################
# CPP mapper tests, execution should use ATen C++ API
################################################################################
set(CORE_TEST_FILES
  test_mapper_memory_promotion
  test_mapper
  test_mapper_llvm
)

foreach(i ${CORE_TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GOOGLE_LIBS} tc_core tc_lang)
#  target_compile_options(${i}  PUBLIC -fsanitize=address)
#  set(CMAKE_EXE_LINKER_FLAGS  -fsanitize=address)
endforeach()
target_link_libraries(test_mapper_llvm tc_aten ${ATEN_LIBRARIES})

################################################################################
# TensorComprehensions tests
# No real need for NVCC if we only use NVRTC
# cuda_add_executable(test_matmul test_matmul.cu OPTIONS -std=c++11)
################################################################################

if (WITH_CAFFE2)
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
