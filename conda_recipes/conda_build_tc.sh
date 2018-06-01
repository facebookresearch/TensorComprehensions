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
# LLVM_TAPIR settings
LLVM_TAPIR_BUILD_VERSION="1.0.0"
LLVM_TAPIR_BUILD_NUMBER=1
LLVM_TAPIR_GIT_HASH="1482504e234a65bffc8c54de8de9fc877822345d"

echo "Building llvm-tapir50"
echo "LLVM_TAPIR_BUILD_VERSION: $LLVM_TAPIR_BUILD_VERSION LLVM_TAPIR_BUILD_NUMBER: ${LLVM_TAPIR_BUILD_NUMBER}"

export LLVM_TAPIR_BUILD_VERSION=$LLVM_TAPIR_BUILD_VERSION
export LLVM_TAPIR_BUILD_NUMBER=$LLVM_TAPIR_BUILD_NUMBER
export LLVM_TAPIR_GIT_HASH=$LLVM_TAPIR_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 llvm-tapir50

echo "llvm-tapir50 packaged Successfully"

##############################################################################
# Halide settings
HALIDE_BUILD_VERSION="1.0.0"
HALIDE_BUILD_NUMBER=0
HALIDE_GIT_HASH="35be67b3a3e4c4461f79949109ff35c54cf307de"

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
#
