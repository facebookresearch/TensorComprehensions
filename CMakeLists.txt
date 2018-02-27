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

# Display compiler flags for build modes: Release/Debug
message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")

add_subdirectory(third-party/pybind11)

set(LLVM_DIR ${CLANG_PREFIX}/lib/cmake/llvm)
message(STATUS "Looking for LLVM in ${LLVM_DIR}")
find_package(LLVM REQUIRED LLVM)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if (NOT TARGET gtest)
add_subdirectory(third-party/googletest)
endif()

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

set(GIT_EXECUTABLE git)

# If we don't set this, manually specified include_directories in our
# cmake files will be placed after system headers, which means that
# we will pick headers from a conda environment before our own. That's
# very confusing!
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

# Make it possible to start cmake directly in an appropriately conda environment
if(NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# NB: If the finding logic here gets complicated, make a proper FindXXX.cmake
# module and use find_package

# Normal dependencies

# pybind11
include_directories(third-party/pybind11)

# isl
find_path(ISL_INCLUDE_DIR NAMES isl)
include_directories(${ISL_INCLUDE_DIR})
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
find_library(ISL_LIBRARIES isl PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found isl: ${ISL_LIBRARIES}")

# dlpack
find_path(ISL_INCLUDE_DIR NAMES dlpack)
include_directories(${DLPACK_INCLUDE_DIR})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
execute_process(COMMAND ${CLANG_PREFIX}/bin/llvm-config --libdir  OUTPUT_VARIABLE LLVM_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
link_directories(${LLVM_LIBDIR})

# protobuf
# TODO: check if this respects CMAKE_PREFIX_PATH
find_package(Protobuf REQUIRED)

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# modified CUB
find_path(CUB_INCLUDE_DIR NAMES cub)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_HOME=\"\\\"${CUDA_INCLUDE_DIRS}\\\"\" -DCUB_HOME=\"\\\"${CUB_INCLUDE_DIR}\\\"\" ")

# Inherited from Torch, see
# https://github.com/torch/cutorch/blob/master/lib/THC/cmake/select_compute_arch.cmake
INCLUDE(cmake/select_compute_arch.cmake)
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)

# TODO: Investigate these
set (CUDA_VERBOSE_BUILD ON)
set (CUDA_PROPAGATE_HOST_FLAGS ON)

################################################################################
# FindCUDA doesn't find all the libraries we need, add the extra ones
# Cribbed from /lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
################################################################################
find_library(CUDA_CUDA_LIBRARIES cuda
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

set(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARIES})

# Needed to build C2 and some of our tests
INCLUDE(cmake/FindCuDNN.cmake)
include_directories(${CUDNN_INCLUDE_DIR})

################################################################################

# Halide
if(NOT HALIDE_PREFIX)
  message(FATAL_ERROR "-DHALIDE_PREFIX=... required")
endif()
message(STATUS "HALIDE_PREFIX: ${HALIDE_PREFIX}")
message(STATUS "Finding Halide")
find_path(HALIDE_INCLUDE_DIR NAMES Halide)
message(STATUS "HALIDE_INCLUDE_DIRS: ${HALIDE_INCLUDE_DIR}")
include_directories(${HALIDE_INCLUDE_DIR})
find_library(HALIDE_LIBRARIES NAMES Halide PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found Halide.so file: ${HALIDE_LIBRARIES}")

################################################################################

# Caffe2
if (WITH_CAFFE2)
  # We suppress sign-compare error messages because caffe2 headers have these
  # errors.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")

  # eigen - needed inside Caffe2 only
  # Caffe2 probably used its bundled Eigen (which it didn't install) in this case.
  message(WARNING "Couldn't find Eigen headers, assuming caffe2 used in-tree headers")
  include_directories(third-party/caffe2/third_party/eigen)

  find_path(CAFFE2_INCLUDE_DIR NAMES caffe2)
  include_directories(${CAFFE2_INCLUDE_DIR})
  find_library(CAFFE2_CPU_LIBRARIES NAMES caffe2 PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
  message(STATUS "Found Caffe2_CPU: ${CAFFE2_CPU_LIBRARIES}")
  find_library(CAFFE2_GPU_LIBRARIES NAMES caffe2_gpu PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
  message(STATUS "Found Caffe2_GPU: ${CAFFE2_GPU_LIBRARIES}")
else()
  message(STATUS "Caffe2 installation is turned off")
endif()

################################################################################
# ATen
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

# if PyTorch is installed, we get libATen.so.1 from there, otherwise build it
if (__torch_install EQUAL 0)
  message(STATUS "TORCH INSTALLED, linking to ATen from PyTorch")
  set(ATEN_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/torch/lib/include")
  include_directories(${ATEN_INCLUDE_DIR})
  find_library(ATEN_LIBRARIES NAMES libATen.so.1 PATHS ${PYTHON_SITE_PACKAGES}/torch/lib)
  link_directories("${PYTHON_SITE_PACKAGES}/../../")
  message(STATUS "Found ATen.so file: ${ATEN_LIBRARIES}")
else()
  message(STATUS "TORCH NOT INSTALLED, linking to ATen separately")
  message(STATUS "Finding ATen")
  find_path(ATEN_INCLUDE_DIR NAMES ATen)
  include_directories(${ATEN_INCLUDE_DIR})
  find_library(ATEN_LIBRARIES NAMES ATen PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
  message(STATUS "Found ATen.so file: ${ATEN_LIBRARIES}")
endif()

################################################################################

# git revision
message("include(cmake/GetGitRevisionDescription.cmake)")
include(cmake/GetGitRevisionDescription.cmake)

################################################################################

# gflags
message("Looking for gflags...")
find_library(GFLAGS_LIBRARIES gflags HINTS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found gflags: ${GFLAGS_LIBRARIES}")

# glog
# For some reason, the world wants to redefine the glog macros with the same
# name. For now, we use glog everywhere, also force DMLC/CAFFE2  to use glog
add_definitions(-DDMLC_USE_GLOG)
if (WITH_CAFFE2)
  add_definitions(-DCAFFE2_USE_GOOGLE_GLOG)
endif()

find_library(GLOG_LIBRARIES glog HINTS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found glog: ${GLOG_LIBRARIES}")

# Uncomment for debugging all the CMake variables
# get_cmake_property(_variableNames VARIABLES)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

add_subdirectory(src)
enable_testing()
add_subdirectory(tensor_comprehensions/pybinds)
add_subdirectory(test)

if (WITH_CAFFE2)
  add_subdirectory(examples)
else()
  message(STATUS "Not building examples, caffe2 not available")
endif()
