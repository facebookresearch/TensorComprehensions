# pybind11 headers
set(
  pybind11_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third-party/pybind11/include
)
include_directories(${pybind11_INCLUDE_DIRS})
# Python.h
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(tc MODULE tc.cc)
target_include_directories(tc PUBLIC ${PROJECT_SOURCE_DIR}/tc)
target_link_libraries(
  tc

  tc_autotuner
  tc_aten

  ${PYTHON_LIBRARIES}
)

set_target_properties(tc PROPERTIES PREFIX "")
set_target_properties(tc PROPERTIES SUFFIX ".so")
set_target_properties(tc PROPERTIES DEBUG_POSTFIX "")

# "Install" to source_dir/tensor_comprehensions for easy inclusion in python
# package with setuptools. Without this, setuptools is hell.
install(TARGETS tc DESTINATION ${PROJECT_SOURCE_DIR}/tensor_comprehensions)
