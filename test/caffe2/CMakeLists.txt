if (WITH_CUDA)
  include_directories(BEFORE ${gtest_INCLUDE_DIR})

  add_library(tc_caffe2_test_harness SHARED test_harness.cc)
  target_link_libraries(tc_caffe2_test_harness tc_c2)
  install(TARGETS tc_caffe2_test_harness DESTINATION lib)

  add_subdirectory(cuda)
endif()
