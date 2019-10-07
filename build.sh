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
#! /bin/bash
set -ex

export TC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! test ${CLANG_PREFIX}; then
    echo 'Environment variable CLANG_PREFIX is required, please run export CLANG_PREFIX=$(llvm-config --prefix)'
    exit 1
fi

if ! test ${CONDA_PREFIX}; then
    echo 'TC now requires conda to build, see BUILD.md'
    exit 1
fi

PYTHON=${PYTHON:="`which python3`"}
CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

WITH_CUDA=${WITH_CUDA:=ON}
WITH_CAFFE2=${WITH_CAFFE2:=OFF}
WITH_TAPIR=${WITH_TAPIR:=ON}
WITH_BINDINGS=${WITH_BINDINGS:=OFF}

CUDNN_ROOT_DIR=${CUDNN_ROOT_DIR:=${CONDA_PREFIX}}
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:=${CONDA_PREFIX}/lib/cmake}
HALIDE_PREFIX=${HALIDE_PREFIX:=${CONDA_PREFIX}}
EIGEN_PREFIX=${EIGEN_PREFIX:=${CONDA_PREFIX}}
CAFFE2_PREFIX=${CAFFE2_PREFIX:=${CONDA_PREFIX}}

THIRD_PARTY_INSTALL_PREFIX=${TC_DIR}/third-party-install

INSTALL_PREFIX=${INSTALL_PREFIX:=${TC_DIR}/install}
mkdir -p ${INSTALL_PREFIX}

echo "TC_DIR: " $TC_DIR
echo "GCC_VER: " $GCC_VER
BUILD_TYPE=${BUILD_TYPE:=Debug}
echo "Build Type: ${BUILD_TYPE}"

mkdir -p ${TC_DIR}/build || exit 1
cd       ${TC_DIR}/build || exit 1

echo "Installing TC"

rm -Rf ${INSTALL_PREFIX}

cmake -DWITH_CAFFE2=${WITH_CAFFE2} \
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
       -DWITH_TAPIR=${WITH_TAPIR} \
       -DPYTHON_EXECUTABLE=${PYTHON} \
       -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
       -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
       -DHALIDE_PREFIX=${HALIDE_PREFIX} \
       -DEIGEN_PREFIX=${EIGEN_PREFIX} \
       -DCAFFE2_PREFIX=${CAFFE2_PREFIX} \
       -DTHIRD_PARTY_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX} \
       -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
       -DCLANG_PREFIX=${CLANG_PREFIX} \
       -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
       -DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR} \
       -DWITH_CUDA=${WITH_CUDA} \
       -DTC_DIR=${TC_DIR} \
       -DCMAKE_C_COMPILER=${CC} \
       -DCMAKE_CXX_COMPILER=${CXX} .. || exit 1

make -j"$(nproc)" -s || exit 1
make install -j"$(nproc)" -s || exit 1

echo "Successfully installed TC"
