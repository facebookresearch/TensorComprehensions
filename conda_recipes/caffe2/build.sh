#!/usr/bin/env bash

set -e

# code gets checked out at below:
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/caffe2_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/caffe2_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the caffe2 package gets checked out
# NOTE: once the build finished, the packaging and unpackaging step takes long
# and might seem like it's stuck but it's not.

echo "Installing caffe2 to ${PREFIX}"
CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
PYTHON=${PYTHON:="`which python3`"}
PROTOC=${PROTOC:="`which protoc`"}

PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"
CMAKE_ARGS=()

# Build with minimal required libraries
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
CMAKE_ARGS+=("-DUSE_MPI=OFF")

CMAKE_ARGS+=("-DBUILD_BINARY=OFF")
CMAKE_ARGS+=("-DUSE_GLOG=ON")
CMAKE_ARGS+=("-DUSE_GFLAGS=ON")
CMAKE_ARGS+=("-DUSE_NNPACK=OFF")
CMAKE_ARGS+=("-DUSE_GLOO=OFF")
CMAKE_ARGS+=("-DUSE_LMDB=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_OPENMP=OFF")
CMAKE_ARGS+=("-DCMAKE_INSTALL_MESSAGE=NEVER")
CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS='-fno-var-tracking-assignments'")
CMAKE_ARGS+=("-DBUILD_TEST=OFF")
CMAKE_ARGS+=("-DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC}")
CMAKE_ARGS+=("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=${PYTHON}")


# Build with CUDA
CMAKE_ARGS+=("-DUSE_CUDA=ON")
CMAKE_ARGS+=("-DUSE_NCCL=OFF")

# Install under specified prefix
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

mkdir -p build
cd build
${CMAKE_VERSION} "${CMAKE_ARGS[@]}" $CONDA_CMAKE_ARGS $PYTHON_ARGS .. || exit 1

make "-j$(nproc)"

make install/fast
