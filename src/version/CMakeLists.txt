get_git_head_revision(GIT_REFSPEC GIT_HASH)
execute_process(COMMAND ${GIT_EXECUTABLE} describe --all --long HEAD OUTPUT_VARIABLE GIT_TAG)
SET(GIT_DESCRIPTION "${GIT_REFSPEC}${GIT_HASH}")
configure_file(version.cc.in version.cc)

add_library(tc_version STATIC ${CMAKE_CURRENT_BINARY_DIR}/version.cc)
SET_TARGET_PROPERTIES(tc_version PROPERTIES LINKER_LANGUAGE CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tc_version INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
