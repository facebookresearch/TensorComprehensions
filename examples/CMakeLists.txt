include_directories(.)
include_directories(..)
include_directories(../../include)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Needed to ensure reproducibility (proper cublas version) via call to
# cublasGetVersion_v2.
find_library(CUDA_CUBLAS_LIBRARIES cublas
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
find_library(CUDA_CUDNN_LIBRARIES cudnn
  PATHS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64)

################################################################################
# Don't ask ... TODO: cleanup gtest
################################################################################
include_directories(${PROJECT_SOURCE_DIR}/third-party/googletest/googletest/include)
set(GTEST_LIBS gtest gtest_main)

################################################################################
# Simple Example
################################################################################
add_executable(example_simple example_simple.cc)
add_test(example_simple example_simple)
target_link_libraries(
   example_simple

   tc_autotuner
   tc_core

   ${GTEST_LIBS}
   ${GFLAGS_LIBRARIES}
   ${GLOG_LIBRARIES}

   ${ATEN_LIBRARIES}
)

################################################################################
# Examples
################################################################################
set(EXAMPLES_FILES
  example_batchmatmul
  example_group_convolution
  example_tmm
  example_MLP_model
)
foreach(i ${EXAMPLES_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(
     ${i}
     tc_autotuner
     tc_core
     tc_c2

     ${CUDA_CUBLAS_LIBRARIES}
     ${CUDA_CUDNN_LIBRARIES}
     ${GTEST_LIBS}
     ${GFLAGS_LIBRARIES}
     ${GLOG_LIBRARIES}

     ${ATEN_LIBRARIES}
  )
endforeach()
