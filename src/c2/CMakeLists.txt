
add_library(
  tc_c2

  SHARED

  operator_meta.cc
  register.cc
)
target_include_directories(tc_c2 PUBLIC ../include)

target_link_libraries(
  tc_c2

  ${PROTOBUF_LIBRARIES}
  ${CAFFE2_CPU_LIBRARIES}
  ${CAFFE2_GPU_LIBRARIES}

  tc_cuda
)

install(
  TARGETS
  tc_c2

  DESTINATION lib
)
