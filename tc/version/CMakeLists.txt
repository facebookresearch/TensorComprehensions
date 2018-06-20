get_git_head_revision(GIT_REFSPEC GIT_HASH)
execute_process(COMMAND ${GIT_EXECUTABLE} describe --all --long HEAD OUTPUT_VARIABLE GIT_TAG)
SET(GIT_DESCRIPTION "${GIT_REFSPEC}${GIT_HASH}")
configure_file(version.cc.in version.cc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(tc_version STATIC ${CMAKE_CURRENT_BINARY_DIR}/version.cc)
SET_TARGET_PROPERTIES(tc_version PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(tc_version INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if (WITH_CUDA)
  find_library(CUDA_LIBDEVRT_PATH libcudadevrt.a
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
  configure_file(cuda_version.cc.in cuda_version.cc)
  add_library(tc_cuda_version STATIC ${CMAKE_CURRENT_BINARY_DIR}/cuda_version.cc)
  SET_TARGET_PROPERTIES(tc_cuda_version PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(tc_cuda_version INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
