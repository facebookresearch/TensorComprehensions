add_library(
  tc_aten

  SHARED

  aten_compiler.cc
)

target_include_directories(tc_aten PUBLIC ${PROJECT_SOURCE_DIR}/include)

message(STATUS "Found ATen.so file: ${ATEN_LIBRARIES}")

target_link_libraries(
  tc_aten

  tc_core
  tc_lang
  tc_version
  tc_proto

  ${ATEN_LIBRARIES})

install(
  TARGETS
  tc_aten

  DESTINATION lib
)
