#!/usr/bin/env bash

set -e

# Purpose: The purpose of this caffe2 conda package is to remove the caffe2 submodule
# from inside TC completely. But we still want to be able to run tests with caffe2,
# so we install caffe2 via conda package built using the recipe here, and get the
# caffe2 libs directly.
#
# How: You will need glog/gflags/protobuf conda packages that TC is compatible with.
# For that, run conda_build_tc.sh script to build those packages and then run
# this script to build caffe2 conda package.
#
# Use: The caffe2 conda package is to be used to install TC in conda environment
# but TC should build from source using the conda package dependencies - isl-tc,
# glog, gflags, protobuf, llvm-tapir5, caffe2.
#
# Don't: Please don't make caffe2 a dependency in the TC conda package. This makes
# the package size big and we don't want that. Rather build TC using conda deps.

echo "Packaging Caffe2"

ANACONDA_USER=prigoyal

# set the anaconda upload to NO for now
conda config --set anaconda_upload no

###############################################################################
# Caffe2 settings
CAFFE2_BUILD_VERSION="0.2.2"
CAFFE2_BUILD_NUMBER=2
CAFFE2_GIT_HASH="5651c4ce0279c895cbb1d1eeecc6594cc8791103"

echo "Packaging Caffe2 ==> CAFFE2_BUILD_VERSION: ${CAFFE2_BUILD_VERSION} CAFFE2_BUILD_NUMBER: ${CAFFE2_BUILD_NUMBER}"

export CAFFE2_BUILD_VERSION=$CAFFE2_BUILD_VERSION
export CAFFE2_BUILD_NUMBER=$CAFFE2_BUILD_NUMBER
export CAFFE2_GIT_HASH=$CAFFE2_GIT_HASH

time conda build -c prigoyal --python 3.6 caffe2 --keep-old-work --no-remove-work-dir

echo "Caffe2 packaged Successfully"
