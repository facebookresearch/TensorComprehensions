if (WITH_CUDA)
  add_library(tc_caffe2_test_harness SHARED test_harness.cc)
  target_link_libraries(tc_caffe2_test_harness tc_c2)
  install(TARGETS tc_caffe2_test_harness DESTINATION lib)

  add_subdirectory(cuda)
endif()
