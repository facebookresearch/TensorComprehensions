include_directories(.)
include_directories(..)
include_directories(../../include)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Don't ask ... TODO: cleanup gtest
################################################################################
include_directories(${PROJECT_SOURCE_DIR}/third-party/googletest/googletest/include)
set(GTEST_LIBS gtest gtest_main)

################################################################################
# Examples
################################################################################
set(EXAMPLES_FILES
  tensordot
  blockdiagperm
)
foreach(i ${EXAMPLES_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(
     ${i}

     tc_autotuner
     tc_core

     ${GTEST_LIBS}
     ${GFLAGS_LIBRARIES}
     ${GLOG_LIBRARIES}

     ${ATEN_LIBRARIES}
  )
endforeach()
