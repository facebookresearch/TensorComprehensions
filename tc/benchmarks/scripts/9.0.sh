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

module avail
module load cuda/9.0 cudnn/v7.0-cuda.9.0  mkl/2018.0.128 cmake/3.8.2/gcc.4.8.4

export EDITOR=emacs
export CLANG_PREFIX=$(llvm-config --prefix)
export TC_PREFIX=$(git rev-parse --show-toplevel)
export CUB_INCLUDE_DIR=${TC_PREFIX}/third-party/caffe2/third_party/cub/
export CUDNN_ROOT_DIR=/public/apps/cudnn/v7.0/cuda/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/public/apps/cuda/9.0/lib64/:/public/apps/cudnn/v7.0/cuda/lib64

echo Compile with: 'PYTHON=$(which python) WITH_PYTHON=OFF CLANG_PREFIX=$(llvm-config --prefix) CMAKE_VERSION=cmake ./build.sh --all'
