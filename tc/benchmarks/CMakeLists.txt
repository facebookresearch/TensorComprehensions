include_directories(.)
include_directories(..)

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
# Benchmarks
################################################################################
set(BENCHMARKS
  batchmatmul
  group_convolution
  kronecker
  moments
  tmm
  MLP_model
  wavenet
)
foreach(i ${BENCHMARKS})
  add_executable(benchmark_${i} ${i}.cc)
  target_link_libraries(
     benchmark_${i}

     ${CUDA_CUBLAS_LIBRARIES}
     ${CUDA_CUDNN_LIBRARIES}
     ${GTEST_LIBRARIES}
     ${ATEN_LIBRARIES}

     tc_caffe2_test_harness
     tc_autotuner
     tc_core
     tc_c2
     tc_aten
  )
endforeach()
