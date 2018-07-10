#!/usr/bin/env bash

set -e

echo "Packaging TC"

ANACONDA_USER=nicolasvasilache

# set the anaconda upload to NO for now
conda config --set anaconda_upload no

echo "Packaging Caffe2"

###############################################################################
# Caffe2 settings
CAFFE2_BUILD_VERSION="1.0.0"
CAFFE2_BUILD_NUMBER=1
PYTORCH_GIT_HASH="8d91a602cc2beab090715bb6bd63ab108db5fa36"

echo "Packaging Caffe2 ==> CAFFE2_BUILD_VERSION: ${CAFFE2_BUILD_VERSION} CAFFE2_BUILD_NUMBER: ${CAFFE2_BUILD_NUMBER}"

export CAFFE2_BUILD_VERSION=$CAFFE2_BUILD_VERSION
export CAFFE2_BUILD_NUMBER=$CAFFE2_BUILD_NUMBER
export PYTORCH_GIT_HASH=$CAFFE2PYTORCH_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 caffe2

echo "Caffe2 packaged Successfully"

###############################################################################
# LLVM_TRUNK settings
LLVM_TRUNK_BUILD_VERSION="1.0.0"
LLVM_TRUNK_BUILD_NUMBER=1
LLVM_TRUNK_SOURCE_DIR=$(mktemp -d /tmp/d.XXXXXX)
trap 'rm -rf "${LLVM_TRUNK_SOURCE_DIR}"' EXIT

svn co http://llvm.org/svn/llvm-project/llvm/trunk ${LLVM_TRUNK_SOURCE_DIR}
svn co http://llvm.org/svn/llvm-project/cfe/trunk ${LLVM_TRUNK_SOURCE_DIR}/tools/clang

echo "Building llvm-trunk"
echo "LLVM_TRUNK_BUILD_VERSION: $LLVM_TRUNK_BUILD_VERSION LLVM_TRUNK_BUILD_NUMBER: ${LLVM_TRUNK_BUILD_NUMBER}"

export LLVM_TRUNK_BUILD_VERSION=$LLVM_TRUNK_BUILD_VERSION
export LLVM_TRUNK_BUILD_NUMBER=$LLVM_TRUNK_BUILD_NUMBER
export LLVM_TRUNK_SOURCE_DIR=$LLVM_TRUNK_SOURCE_DIR

time conda build -c $ANACONDA_USER --python 3.6 llvm-trunk

echo "llvm-trunk packaged Successfully"

##############################################################################
# Halide settings
HALIDE_BUILD_VERSION="1.0.0"
HALIDE_BUILD_NUMBER=1
HALIDE_GIT_HASH="0b29cacf636852933892bbaa61dd2050c8dcaff2"

echo "Packaging HALIDE ==> HALIDE_BUILD_VERSION: ${HALIDE_BUILD_VERSION} HALIDE_BUILD_NUMBER: ${HALIDE_BUILD_NUMBER}"

export HALIDE_BUILD_VERSION=$HALIDE_BUILD_VERSION
export HALIDE_BUILD_NUMBER=$HALIDE_BUILD_NUMBER
export HALIDE_GIT_HASH=$HALIDE_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 halide

echo "HALIDE packaged Successfully"

################################################################################
## Tensor Comprehensions settings
#TC_BUILD_VERSION="0.2.0"
#TC_BUILD_NUMBER=1
#TC_GIT_HASH="5db46c68bd481b8a0fa1cbb6492fc26ab2f83de2"
#
#echo "Packaging TC ==> TC_BUILD_VERSION: ${TC_BUILD_VERSION} TC_BUILD_NUMBER: ${TC_BUILD_NUMBER}"
#
#export TC_BUILD_VERSION=$TC_BUILD_VERSION
#export TC_BUILD_NUMBER=$TC_BUILD_NUMBER
#export TC_GIT_HASH=$TC_GIT_HASH
#
#time conda build -c $ANACONDA_USER --python 3.6 tensor_comprehensions
#
#echo "Tensor Comprehensions packaged Successfully"
