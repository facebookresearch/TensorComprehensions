
add_library(
  tc_lang

  SHARED

  error_report.cc
  parser.cc
  lexer.cc
  tc_format.cc
)

set_target_properties(tc_lang PROPERTIES LINK_INTERFACE_LIBRARIES gflags)

install(
  TARGETS
  tc_lang

  DESTINATION lib
)
