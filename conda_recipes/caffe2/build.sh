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
THIRD_PARTY_INSTALL_PREFIX=${CONDA_PREFIX}

# The following hardcoding is very ugly but I don't know of many solutions to build
# conda packages with GPU support.
# Ohai, turns out we're not the only ones doing this..:
# https://github.com/pytorch/pytorch/blob/master/conda/caffe2/full/build.sh#L38
CUDA_TOOLKIT_ROOT_DIR=/public/apps/cuda/9.0/
CUDNN_ROOT_DIR=/public/apps/cudnn/v7.0/cuda/

PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"
CMAKE_ARGS=()

# Build with minimal required libraries
CMAKE_ARGS=("-DBUILD_BINARY=OFF")
CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS='-fno-var-tracking-assignments'")
CMAKE_ARGS+=("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
CMAKE_ARGS+=("-DUSE_GLOG=OFF")
CMAKE_ARGS+=("-DUSE_GFLAGS=OFF")
CMAKE_ARGS+=("-DUSE_GLOO=OFF")
CMAKE_ARGS+=("-DUSE_NCCL=OFF")
CMAKE_ARGS+=("-DUSE_LMDB=OFF")
CMAKE_ARGS+=("-DUSE_LEVELDB=OFF")
CMAKE_ARGS+=("-DBUILD_TEST=OFF")
CMAKE_ARGS+=("-DUSE_OPENCV=OFF")
CMAKE_ARGS+=("-DUSE_OPENMP=OFF")
CMAKE_ARGS+=("-DCMAKE_INSTALL_MESSAGE=NEVER")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DBUILD_PYTHON=OFF")
CMAKE_ARGS+=("-DUSE_NNPACK=OFF")
CMAKE_ARGS+=("-DPROTOBUF_PROTOC_EXECUTABLE=${THIRD_PARTY_INSTALL_PREFIX}/bin/protoc")
CMAKE_ARGS+=("-DUSE_CUDA=ON")
CMAKE_ARGS+=("-DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR}")
CMAKE_ARGS+=("-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")

# Install under specified prefix
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

mkdir -p build
cd build
${CMAKE_VERSION} "${CMAKE_ARGS[@]}" $CONDA_CMAKE_ARGS $PYTHON_ARGS .. || exit 1

make "-j$(nproc)"

make install/fast
