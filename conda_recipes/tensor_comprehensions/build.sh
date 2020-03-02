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
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/tensor_comprehensions_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/tensor_comprehensions_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the tensor_comprehensions package gets checked out

CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
VERBOSE=${VERBOSE:=0}

CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

VERBOSE=${VERBOSE:=0}
PYTHON=${PYTHON:="`which python3`"}
PROTOC=${PROTOC:="`which protoc`"}
WITH_CAFFE2=OFF
BUILD_TYPE=Release

export TC_DIR=$(pwd)
export INSTALL_PREFIX=$PREFIX
export CLANG_PREFIX=$($PREFIX/bin/llvm-config --prefix)

echo "CLANG_PREFIX: ${CLANG_PREFIX}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "PREFIX: ${PREFIX}"
echo "PYTHON: ${PYTHON}"
echo "PROTOC: ${PROTOC}"
echo "PROTOC VERSION: $(${PROTOC} --version)"

echo "Clean up existing build packages if any"
rm -rf ${TC_DIR}/build || true
rm -rf ${TC_DIR}/third-party/*/build || true

# dlpack
echo "Installing DLPACK headers"
mkdir -p ${TC_DIR}/third-party/dlpack/build || exit 1
pushd ${TC_DIR}/third-party/dlpack/build || exit 1

VERBOSE=${VERBOSE} ${CMAKE_VERSION} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} .. || exit 1

VERBOSE=${VERBOSE} make -j"$(nproc)" -s || exit 1

popd
cp -R ${TC_DIR}/third-party/dlpack/include/dlpack ${INSTALL_PREFIX}/include/
echo "DLPACK headers installed"

# cub
echo "Installing CUB"
cp -R ${TC_DIR}/third-party/cub/cub ${INSTALL_PREFIX}/include/
echo "CUB installed"

# TC
echo "Installing Tensor Comprehensions"
mkdir -p ${TC_DIR}/build
pushd ${TC_DIR}/build

VERBOSE=${VERBOSE} ${CMAKE_VERSION} \
    -DWITH_CAFFE2=${WITH_CAFFE2} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DPYTHON_EXECUTABLE=${PYTHON} \
    -DHALIDE_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}/lib/cmake \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC} \
    -DCLANG_PREFIX=${CLANG_PREFIX} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR} .. || exit 1

VERBOSE=${VERBOSE} make -j"$(nproc)" -s || exit 1

VERBOSE=${VERBOSE} make install -j"$(nproc)" -s || exit 1

popd

##############################################################################
# setuptools install starts now
##############################################################################
echo "Packaging TC with setuptools"
${PYTHON} ${TC_DIR}/setup.py install
echo "Finished installing with setuptools"

##############################################################################
# Test installation
##############################################################################

echo "Running all tests now"
./test_cpu.sh || exit 1
./test.sh || exit 1

echo "Successfully built TC package"
