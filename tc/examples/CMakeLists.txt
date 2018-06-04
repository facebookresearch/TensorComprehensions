include_directories(.)
include_directories(..)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Examples
################################################################################
set(EXAMPLES_FILES
  blockdiagperm
  group_normalization
  tensordot
)
foreach(i ${EXAMPLES_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(
     ${i}

     ${GTEST_LIBRARIES}
     ${ATEN_LIBRARIES}

     tc_autotuner
     tc_core
     tc_aten
  )
endforeach()
