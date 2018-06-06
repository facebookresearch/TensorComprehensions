cmake_minimum_required (VERSION 2.6)

# See https://stackoverflow.com/a/31010221/23845
if(CMAKE_VERSION VERSION_LESS "3.1")
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  if(NOT COMPILER_SUPPORTS_CXX11)
    MESSAGE(FATAL_ERROR "Your compiler does not support c++11")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

project (TensorComprehensions C CXX)

################################################################################
# RPATH business
################################################################################
# RPATH - how to properly handle rpath https://cmake.org/Wiki/CMake_RPATH_handling
# RPATH - a list of directories which is linked into the executable, supported on most UNIX systems.
# By default if you don't change any RPATH related settings, CMake will link the
# executables and shared libraries with full RPATH to all used libraries in
# the build tree. When installing, it will clear the RPATH of these targets so
# they are installed with an empty RPATH. The following settings are recommended

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# don't add the automatically determined parts of the RPATH which point to
# directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  # the RPATH to be used when installing
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_PREFIX}/lib64")
endif()

################################################################################
# Third-party libraries
################################################################################

################################################################################
# protobuf-3.5.2 (must use the same version as Caffe2 as long as we use Caffe2)
################################################################################
# TODO: turn this horror into a custom command and custom targets
find_program(test_PROTOBUF_PROTOC_EXECUTABLE protoc
  PATHS ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
  PATH_SUFFIXES src/
  NO_DEFAULT_PATH)
if (NOT test_PROTOBUF_PROTOC_EXECUTABLE)
  execute_process(
    COMMAND ./autogen.sh
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
  )
  execute_process(
    COMMAND ./configure "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" --disable-shared --enable-static
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
  )
endif()

# TODO: For some reason make -j"$(nproc)" fails here
execute_process(
  COMMAND make -j8
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
)
find_library(PROTOBUF_LIBRARIES libprotobuf.a
  REQUIRED
  PATHS ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
  PATH_SUFFIXES src/.libs
  NO_DEFAULT_PATH)
message("Found PROTOBUF_LIBRARIES:" ${PROTOBUF_LIBRARIES})
find_program(PROTOBUF_PROTOC_EXECUTABLE protoc
  REQUIRED
  PATHS ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
  PATH_SUFFIXES src/
  NO_DEFAULT_PATH)
message("Found PROTOBUF_PROTOC_EXECUTABLE:" ${PROTOBUF_PROTOC_EXECUTABLE})
find_path(PROTOBUF_INCLUDES google
  REQUIRED
  PATHS ${PROJECT_SOURCE_DIR}/third-party/googlelibraries/protobuf-3.5.2
  PATH_SUFFIXES src
  NO_DEFAULT_PATH)
message("Found PROTOBUF_INCLUDES:" ${PROTOBUF_INCLUDES})
include_directories(BEFORE ${PROTOBUF_INCLUDES})

################################################################################
# gflags
################################################################################
set(GFLAGS_BUILD_SHARED_LIBS OFF)
set(GFLAGS_BUILD_STATIC_LIBS ON)
set(GFLAGS_BUILD_gflags_LIB ON)
set(GFLAGS_BUILD_gflags_nothreads_LIB ON)
add_subdirectory(third-party/googlelibraries/gflags)
include_directories(BEFORE ${gflags_BINARY_DIR}/include)
# This is a TARGET, not library names so no .d gets appended ever
set(GFLAGS_LIBRARIES gflags)

#################################################################################
## glog
#################################################################################
# For our purposes we need to build glog with gflags support.
# For this, glog needs to know about the gflags TARGET.
# Since TC builds the gflags TARGET in tree, declare it found for use in glog.
set(gflags_FOUND ON)
add_subdirectory(third-party/googlelibraries/glog)
include_directories(BEFORE ${glog_BINARY_DIR})
# This is a TARGET, not library names so no .d gets appended ever
set(GLOG_LIBRARIES glog)

#################################################################################
## gtest
#################################################################################
set(BUILD_GMOCK OFF)
set(BUILD_GTEST ON)
add_subdirectory(third-party/googlelibraries/googletest)
# These are TARGET, not library names so no .d gets appended ever
set(GTEST_LIBRARIES gtest gtest_main)
set(gtest_INCLUDE_DIR ${gtest_SOURCE_DIR}/include)

#################################################################################
## Set GOOGLE_LIBRARIES
#################################################################################
set(GOOGLE_LIBRARIES ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} ${GTEST_LIBRARIES} ${PROTOBUF_LIBRARIES})

################################################################################
# cuda
################################################################################
set(CUB_INSTALL_DIR ${THIRD_PARTY_INSTALL_PREFIX}/include)
if(WITH_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(BEFORE ${CUDA_TOOLKIT_ROOT_DIR}/include)

  # modified CUB
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_CUDA -DCUDA_HOME=\"\\\"${CUDA_INCLUDE_DIRS}\\\"\" -DCUB_HOME=\"\\\"${CUB_INSTALL_DIR}\\\"\" ")

  # Inherited from Torch, see
  # https://github.com/torch/cutorch/blob/master/lib/THC/cmake/select_compute_arch.cmake
  INCLUDE(cmake/select_compute_arch.cmake)
  CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)

  # TODO: Investigate these
  set (CUDA_VERBOSE_BUILD ON)
  set (CUDA_PROPAGATE_HOST_FLAGS ON)

  ##############################################################################
  # FindCUDA doesn't find all the libraries we need, add the extra ones
  # Cribbed from /lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
  ##############################################################################
  find_library(CUDA_CUDA_LIBRARIES cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

  set(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARIES})

  # Needed to build Caffe2 and some of our tests
  INCLUDE(cmake/FindCuDNN.cmake)
  include_directories(${CUDNN_INCLUDE_DIR})
else()
  message(STATUS "Building TC without CUDA support")
endif()

################################################################################
# LLVM
################################################################################
set(LLVM_DIR ${CLANG_PREFIX}/lib/cmake/llvm)
message(STATUS "Looking for LLVM in ${LLVM_DIR}")
find_package(LLVM REQUIRED LLVM)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

################################################################################
# dlpack
################################################################################
include_directories(${PROJECT_SOURCE_DIR}/third-party/dlpack/include)

################################################################################
# cub (used by nvrtc, only needs to be installed)
################################################################################
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third-party/cub/cub
        DESTINATION ${CUB_INSTALL_DIR})

################################################################################
# pybind11
################################################################################
add_subdirectory(third-party/pybind11)
include_directories(third-party/pybind11)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
execute_process(COMMAND ${CLANG_PREFIX}/bin/llvm-config --libdir  OUTPUT_VARIABLE LLVM_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
link_directories(${LLVM_LIBDIR})

################################################################################
# ATen must come from a conda installation, building from source has become
# unreasonable
################################################################################
# first find python path
execute_process(COMMAND which python OUTPUT_VARIABLE __python OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "PYTHON output: \n${__python}")

# run the import torch command, if torch is installed it returns 0 otherwise error
execute_process(COMMAND "${__python}" "-c" "import torch;" RESULT_VARIABLE __torch_install OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "IMPORTING TORCH: \n${__torch_install}")

# also get the site-packages path where conda installs pytorch
execute_process(COMMAND "python" "-c"
    "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "PYTHON site packages: \n${PYTHON_SITE_PACKAGES}")

message(STATUS "TORCH INSTALLED, linking to ATen from PyTorch")
set(ATEN_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/torch/lib/include")
include_directories(AFTER ${ATEN_INCLUDE_DIR})
find_library(ATEN_LIBRARIES NAMES libATen.so PATHS ${PYTHON_SITE_PACKAGES}/torch/lib NO_DEFAULT_PATH)
link_directories("${PYTHON_SITE_PACKAGES}/../../")
message(STATUS "Found ATen.so file: ${ATEN_LIBRARIES}")

################################################################################
# isl
################################################################################
set(ISL_INT gmp)
include_directories(AFTER ${PROJECT_SOURCE_DIR}/third-party/islpp/include)
include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR}/third-party/islpp/include)
add_subdirectory(third-party/islpp)
set(ISL_LIBRARIES isl-static)

################################################################################
# Halide
################################################################################
message(STATUS "Finding Halide")
find_path(HALIDE_INCLUDE_DIR REQUIRED NAMES Halide PATHS ${HALIDE_PREFIX} PATH_SUFFIXES include NO_DEFAULT_PATH)
message(STATUS "HALIDE_INCLUDE_DIRS: ${HALIDE_INCLUDE_DIR}")
include_directories(AFTER ${HALIDE_INCLUDE_DIR})
include_directories(AFTER ${HALIDE_INCLUDE_DIR}/Halide)
find_library(HALIDE_LIBRARIES REQUIRED NAMES Halide PATHS ${HALIDE_PREFIX} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
message(STATUS "Found Halide.so file: ${HALIDE_LIBRARIES}")

################################################################################
# git revision
################################################################################
message("include(cmake/GetGitRevisionDescription.cmake)")
include(cmake/GetGitRevisionDescription.cmake)

################################################################################
# Uncomment for debugging all the CMake variables
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

################################################################################
# Finally, build
################################################################################

################################################################################
# Compile flags
################################################################################
# Display compiler flags for build modes: Release/Debug
message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION is ${CMAKE_CXX_COMPILER_VERSION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.4")
  # we officially support these compiler flags for compiler version 5.4 or less
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Werror -Wno-sign-compare")
endif()
message(STATUS "CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_DIR=\"\\\"${TC_DIR}\\\"\" ")

include_directories(BEFORE ${PROJECT_SOURCE_DIR})
add_subdirectory(tc)

# At the moment pybind is only supported in CUDA mode and compilation fails
# for non-CUDA mode (CUDA_HOME and CUB_HOME undefined error).
# Once the core CPU mapper is stabilized we can worry about pybind, deactivate
# conditionally for now
if (WITH_CUDA)
  add_subdirectory(tensor_comprehensions/pybinds)
endif()

enable_testing()
add_subdirectory(test)

if (WITH_CUDA)
  add_subdirectory(tc/examples)
else()
  message(STATUS "Not building examples, CUDA not available")
endif()

if (WITH_CAFFE2 AND WITH_CUDA)
   find_path(EIGEN_INCLUDE_DIR REQUIRED NAMES Eigen PATHS ${EIGEN_PREFIX} PATH_SUFFIXES include/eigen3 NO_DEFAULT_PATH)
   message(STATUS "Found, EIGEN_INCLUDE_DIR: " ${EIGEN_INCLUDE_DIR})
   find_library(CAFFE2_CPU_LIBRARY REQUIRED NAMES caffe2
                PATHS ${CAFFE2_PREFIX} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
   find_library(CAFFE2_GPU_LIBRARY REQUIRED NAMES caffe2_gpu
                PATHS ${CAFFE2_PREFIX} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
   set(CAFFE2_LIBRARIES ${CAFFE2_CPU_LIBRARY} ${CAFFE2_GPU_LIBRARY})
   find_path(CAFFE2_INCLUDE_DIR REQUIRED NAMES caffe2 PATHS ${CAFFE2_PREFIX} PATH_SUFFIXES include NO_DEFAULT_PATH)
   message(STATUS "Found, CAFFE2_INCLUDE_DIR: " ${CAFFE2_INCLUDE_DIR})

   add_definitions(-DCAFFE2_USE_GOOGLE_GLOG)
   include_directories(AFTER ${EIGEN_INCLUDE_DIR})
   include_directories(AFTER ${CAFFE2_INCLUDE_DIR})
   add_subdirectory(tc/c2)
   add_subdirectory(tc/benchmarks)
   add_subdirectory(test/caffe2)
else()
  message(STATUS "Not building benchmarks, caffe2 or CUDA not available")
endif()
