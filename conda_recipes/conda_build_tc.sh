#!/usr/bin/env bash

set -e

echo "Packaging TC"

ANACONDA_USER=prigoyal

# set the anaconda upload to NO for now
conda config --set anaconda_upload no

##############################################################################
# ISL settings
ISL_TC_BUILD_VERSION="0.1.0"
ISL_TC_BUILD_NUMBER=2
ISL_TC_GIT_HASH="68e36add28a5e2018c24ff0f04d54d96359fba95"

echo "Packaging ISL-TC first"
echo "ISL_TC_BUILD_VERSION: ${ISL_TC_BUILD_VERSION} ISL_TC_BUILD_NUMBER: ${ISL_TC_BUILD_NUMBER}"

export ISL_TC_BUILD_VERSION=$ISL_TC_BUILD_VERSION
export ISL_TC_BUILD_NUMBER=$ISL_TC_BUILD_NUMBER
export ISL_TC_GIT_HASH=$ISL_TC_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 isl-tc --keep-old-work --no-remove-work-dir

echo "ISL-TC packaged Successfully"

###############################################################################
# CLANG+LLVM settings
CLANG_LLVM_BUILD_VERSION="0.1.0"
CLANG_LLVM_BUILD_NUMBER=2
CLANG_LLVM_GIT_HASH="ec3ad2b8d3810dde9c0aaccf3f3f971144d90bc2"

echo "Building clang+llvm-tapir5.0"
echo "CLANG_LLVM_BUILD_VERSION: $CLANG_LLVM_BUILD_VERSION CLANG_LLVM_BUILD_NUMBER: ${CLANG_LLVM_BUILD_NUMBER}"

export CLANG_LLVM_BUILD_VERSION=$CLANG_LLVM_BUILD_VERSION
export CLANG_LLVM_BUILD_NUMBER=$CLANG_LLVM_BUILD_NUMBER
export CLANG_LLVM_GIT_HASH=$CLANG_LLVM_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 clang+llvm-tapir5.0 --keep-old-work --no-remove-work-dir

echo "clang+llvm-tapir5.0 packaged Successfully, Hooray!!"

###############################################################################
# Gflags settings
GFLAGS_BUILD_VERSION="2.4.4"
GFLAGS_BUILD_NUMBER=2
GFLAGS_GIT_HASH="4663c80d3ab19fc7d9408fe8fb22b07b87c76e5a"

echo "Packaging GFLAGS ==> GFLAGS_BUILD_VERSION: ${GFLAGS_BUILD_VERSION} GFLAGS_BUILD_NUMBER: ${GFLAGS_BUILD_NUMBER}"

export GFLAGS_BUILD_VERSION=$GFLAGS_BUILD_VERSION
export GFLAGS_BUILD_NUMBER=$GFLAGS_BUILD_NUMBER
export GFLAGS_GIT_HASH=$GFLAGS_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 gflags --keep-old-work --no-remove-work-dir

echo "GFLAGS packaged Successfully"

###############################################################################
# Gflags settings
GLOG_BUILD_VERSION="0.3.9"
GLOG_BUILD_NUMBER=2
GLOG_GIT_HASH="d0531421fd5437ae3e5249106c6fc4247996e526"

echo "Packaging GLOG ==> GLOG_BUILD_VERSION: ${GLOG_BUILD_VERSION} GLOG_BUILD_NUMBER: ${GLOG_BUILD_NUMBER}"

export GLOG_BUILD_VERSION=$GLOG_BUILD_VERSION
export GLOG_BUILD_NUMBER=$GLOG_BUILD_NUMBER
export GLOG_GIT_HASH=$GLOG_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 glog --keep-old-work --no-remove-work-dir

echo "GLOG packaged Successfully"

##############################################################################
# Protobuf settings
PROTO_BUILD_VERSION="3.4.1"
PROTO_BUILD_NUMBER=2

echo "Packaging Protobuf ==> PROTO_BUILD_VERSION: ${PROTO_BUILD_VERSION} PROTO_BUILD_NUMBER: ${PROTO_BUILD_NUMBER}"

export PROTO_BUILD_VERSION=$PROTO_BUILD_VERSION
export PROTO_BUILD_NUMBER=$PROTO_BUILD_NUMBER

time conda build -c $ANACONDA_USER --python 3.6 protobuf --keep-old-work --no-remove-work-dir

echo "Protobuf packaged Successfully"

##############################################################################
# Halide settings
HALIDE_BUILD_VERSION="0.1.0"
HALIDE_BUILD_NUMBER=2
HALIDE_GIT_HASH="fe85f6a70e9d2f16d5a1591b90c8e598dbcf4c7f"

echo "Packaging HALIDE ==> HALIDE_BUILD_VERSION: ${HALIDE_BUILD_VERSION} HALIDE_BUILD_NUMBER: ${HALIDE_BUILD_NUMBER}"

export HALIDE_BUILD_VERSION=$HALIDE_BUILD_VERSION
export HALIDE_BUILD_NUMBER=$HALIDE_BUILD_NUMBER
export HALIDE_GIT_HASH=$HALIDE_GIT_HASH

time conda build -c $ANACONDA_USER --python 3.6 halide --keep-old-work --no-remove-work-dir

echo "HALIDE packaged Successfully"

###############################################################################
# Tensor Comprehensions settings
TC_BUILD_VERSION="0.1.1"
TC_BUILD_NUMBER=3
TC_GIT_HASH="02fe7370832d8e93839cfcd7c6798783bd4f31cc"

echo "Packaging TC ==> TC_BUILD_VERSION: ${TC_BUILD_VERSION} TC_BUILD_NUMBER: ${TC_BUILD_NUMBER}"

export TC_BUILD_VERSION=$TC_BUILD_VERSION
export TC_BUILD_NUMBER=$TC_BUILD_NUMBER
export TC_GIT_HASH=$TC_GIT_HASH

time conda build -c pytorch --python 3.6 tensor_comprehensions --keep-old-work --no-remove-work-dir

echo "Tensor Comprehensions packaged Successfully"
