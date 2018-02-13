include_directories(../include)

add_library(
  tc_lang

  SHARED

  parser.cc
  lexer.cc
)

install(
  TARGETS
  tc_lang

  DESTINATION lib
)
