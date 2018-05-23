# pybind11 headers
set(
  pybind11_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/third-party/pybind11/include
)
include_directories(${pybind11_INCLUDE_DIRS})
# Python.h
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(tc MODULE pybind_engine.cc)
target_include_directories(tc PUBLIC ${PROJECT_SOURCE_DIR}/tc)
target_link_libraries(
  tc

  tc_autotuner
  tc_aten

  ${PYTHON_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
)
add_library(mapping_options MODULE pybind_options.cc)
target_include_directories(mapping_options PUBLIC ${PROJECT_SOURCE_DIR}/tc)
target_link_libraries(
  mapping_options

  tc_core_cuda_no_sdk

  ${PYTHON_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
)

set_target_properties(tc PROPERTIES PREFIX "")
set_target_properties(tc PROPERTIES SUFFIX ".so")
set_target_properties(tc PROPERTIES DEBUG_POSTFIX "")
install(TARGETS tc DESTINATION lib)

set_target_properties(mapping_options PROPERTIES PREFIX "")
set_target_properties(mapping_options PROPERTIES SUFFIX ".so")
set_target_properties(mapping_options PROPERTIES DEBUG_POSTFIX "")
install(TARGETS mapping_options DESTINATION lib)

if (WITH_CUDA)
  add_library(autotuner MODULE pybind_autotuner.cc)
  target_include_directories(mapping_options PUBLIC ${PROJECT_SOURCE_DIR}/tc)
  target_link_libraries(
    autotuner

    tc_autotuner
    tc_aten

    ${PYTHON_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
  )

  set_target_properties(autotuner PROPERTIES PREFIX "")
  set_target_properties(autotuner PROPERTIES SUFFIX ".so")
  set_target_properties(autotuner PROPERTIES DEBUG_POSTFIX "")
  install(TARGETS autotuner DESTINATION lib)
endif()
