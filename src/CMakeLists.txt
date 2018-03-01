include_directories(.)
include_directories(../include)

add_subdirectory(autotuner)
add_subdirectory(core)
add_subdirectory(lang)
add_subdirectory(proto)
add_subdirectory(version)

if (WITH_CUDA)
  add_subdirectory(aten)
endif()

if (WITH_CAFFE2 AND WITH_CUDA)
  add_subdirectory(c2)
endif()
