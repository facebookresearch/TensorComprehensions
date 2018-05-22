include_directories(.)
include_directories(..)
include_directories(../..)

################################################################################
# Caffe2 + TC tests
################################################################################
add_executable(test_basic_caffe2_cuda test_basic_caffe2_cuda.cc)
add_test(test_basic_caffe2_cuda test_basic_caffe2_cuda)
target_link_libraries(test_basic_caffe2_cuda ${ATEN_LIBRARIES} ${GTEST_LIBRARIES} tc_c2 pthread)

set(TEST_FILES
  test_caffe2
)
foreach(i ${TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GTEST_LIBRARIES} pthread tc_caffe2_test_harness tc_c2
  pthread)
endforeach()
