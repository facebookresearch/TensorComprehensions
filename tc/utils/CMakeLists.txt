include_directories(..)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(cache_merger cache_merger.cc)
target_link_libraries(cache_merger tc_cuda tc_core ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})
