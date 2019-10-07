# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
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

PYTHON_ARGS="$(python ./scripts/get_python_cmake_flags.py)"
CMAKE_ARGS=()

CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0
CUDNN_ROOT_DIR=/usr/lib/x86_64-linux-gnu
CUB_INCLUDE_DIR=/opt/cuda/cub/

if ! test -e ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so; then
    echo 'CUDA_TOOLKIT_ROOT_DIR is needed to build'
    exit 1
fi

if ! test -e ${CUDNN_ROOT_DIR}/libcudnn.so; then
    echo 'CUDNN_ROOT_DIR is needed to build'
    exit 1
fi

if ! test -e ${CUB_INCLUDE_DIR}/cub/cub.cuh; then
    echo 'CUB_INCLUDE_DIR is needed to build'
    exit 1
fi

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
CMAKE_ARGS+=("-DCUB_INCLUDE_DIR=${CUB_INCLUDE_DIR}")
CMAKE_ARGS+=("-DCUDA_ARCH_NAME=All")

# Install under specified prefix
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$PREFIX")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX")

mkdir -p build
cd build
${CMAKE_VERSION} "${CMAKE_ARGS[@]}" $CONDA_CMAKE_ARGS $PYTHON_ARGS .. || exit 1

make "-j$(nproc)"

make install/fast
