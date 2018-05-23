#!/usr/bin/env bash

set -e

echo "Packaging TC"

ANACONDA_USER=nicolasvasilache

# set the anaconda upload to NO for now
conda config --set anaconda_upload no

###############################################################################
# CLANG+LLVM settings
CLANG_LLVM_BUILD_VERSION="0.1.0"
CLANG_LLVM_BUILD_NUMBER=3
CLANG_LLVM_GIT_HASH="1482504e234a65bffc8c54de8de9fc877822345d"

echo "Building clang+llvm-tapir5.0"
echo "CLANG_LLVM_BUILD_VERSION: $CLANG_LLVM_BUILD_VERSION CLANG_LLVM_BUILD_NUMBER: ${CLANG_LLVM_BUILD_NUMBER}"

export CLANG_LLVM_BUILD_VERSION=$CLANG_LLVM_BUILD_VERSION
export CLANG_LLVM_BUILD_NUMBER=$CLANG_LLVM_BUILD_NUMBER
export CLANG_LLVM_GIT_HASH=$CLANG_LLVM_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 clang+llvm-tapir5.0 --keep-old-work --no-remove-work-dir

echo "clang+llvm-tapir5.0 packaged Successfully"

##############################################################################
# Halide settings
HALIDE_BUILD_VERSION="0.1.0"
HALIDE_BUILD_NUMBER=3
HALIDE_GIT_HASH="35be67b3a3e4c4461f79949109ff35c54cf307de"

echo "Packaging HALIDE ==> HALIDE_BUILD_VERSION: ${HALIDE_BUILD_VERSION} HALIDE_BUILD_NUMBER: ${HALIDE_BUILD_NUMBER}"

export HALIDE_BUILD_VERSION=$HALIDE_BUILD_VERSION
export HALIDE_BUILD_NUMBER=$HALIDE_BUILD_NUMBER
export HALIDE_GIT_HASH=$HALIDE_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 halide --keep-old-work --no-remove-work-dir

echo "HALIDE packaged Successfully"

###############################################################################
# Tensor Comprehensions settings
#TC_BUILD_VERSION="0.1.1"
#TC_BUILD_NUMBER=3
#TC_GIT_HASH="8e112e9dccda62c30ef29208a827e783b9a7f156"
#
#echo "Packaging TC ==> TC_BUILD_VERSION: ${TC_BUILD_VERSION} TC_BUILD_NUMBER: ${TC_BUILD_NUMBER}"
#
#export TC_BUILD_VERSION=$TC_BUILD_VERSION
#export TC_BUILD_NUMBER=$TC_BUILD_NUMBER
#export TC_GIT_HASH=$TC_GIT_HASH
#
#time conda build -c pytorch --python 3.6 tensor_comprehensions --keep-old-work --no-remove-work-dir
#
#echo "Tensor Comprehensions packaged Successfully"
