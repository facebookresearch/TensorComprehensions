include_directories(.)
include_directories(../include)

add_subdirectory(lang)
add_subdirectory(proto)
add_subdirectory(version)
add_subdirectory(core)
add_subdirectory(aten)

if (WITH_CAFFE2)
  add_subdirectory(c2)
endif()

add_subdirectory(autotuner)
