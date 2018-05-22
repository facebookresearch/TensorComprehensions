
add_library(
  tc_c2

  SHARED

  operator_meta.cc
  register.cc
)

target_link_libraries(
  tc_c2

  ${PROTOBUF_LIBRARIES}
  ${CAFFE2_LIBRARIES}

  tc_cuda
)

install(
  TARGETS
  tc_c2

  DESTINATION lib
)
