# pybind11 headers
set(
  pybind11_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third-party/pybind11/include
)
include_directories(${pybind11_INCLUDE_DIRS})
# Python.h
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(tclib MODULE tclib.cc)
target_include_directories(tclib PUBLIC ${PROJECT_SOURCE_DIR}/tc)
target_link_libraries(
  tclib

  tc_autotuner
  tc_aten

  ${PYTHON_LIBRARIES}
)

set_target_properties(tclib PROPERTIES PREFIX "")
set_target_properties(tclib PROPERTIES SUFFIX ".so")
set_target_properties(tclib PROPERTIES DEBUG_POSTFIX "")

# "Install" to source_dir/tensor_comprehensions for easy inclusion in python
# package with setuptools. Without this, setuptools is hell.
install(TARGETS tclib DESTINATION ${PROJECT_SOURCE_DIR}/tensor_comprehensions)
