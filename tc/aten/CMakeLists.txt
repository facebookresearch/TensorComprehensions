################################################################################
# tc_aten
#
# Core CPU library with cross-compilation capabilities linked from
# tc_aten
################################################################################
add_library(
  tc_aten

  SHARED

  aten_compiler_new_api.cc
)
target_link_libraries(
  tc_aten

  ${HALIDE_LIBRARIES}

  tc_core
)
install(
  TARGETS
  tc_aten

  DESTINATION lib
)
