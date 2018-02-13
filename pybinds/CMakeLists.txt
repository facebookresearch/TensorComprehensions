add_library(mapping_options MODULE pybind_options.cc)
add_library(autotuner MODULE pybind_autotuner.cc)

# pybind11 headers
set(
  pybind11_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third-party/pybind11/include
)
include_directories(${pybind11_INCLUDE_DIRS})

# Python.h
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(
  mapping_options

  tc_core

  ${PYTHON_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
)
target_link_libraries(
  autotuner

  tc_core
  tc_aten
  tc_autotuner

  ${PYTHON_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
)

set_target_properties(mapping_options PROPERTIES PREFIX "")
set_target_properties(mapping_options PROPERTIES SUFFIX ".so")
set_target_properties(mapping_options PROPERTIES DEBUG_POSTFIX "")
install(TARGETS mapping_options DESTINATION ${PROJECT_SOURCE_DIR})

set_target_properties(autotuner PROPERTIES PREFIX "")
set_target_properties(autotuner PROPERTIES SUFFIX ".so")
set_target_properties(autotuner PROPERTIES DEBUG_POSTFIX "")
install(TARGETS autotuner DESTINATION ${PROJECT_SOURCE_DIR})
