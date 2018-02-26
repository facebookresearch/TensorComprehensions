#!/usr/bin/env bash

set -e

# code gets checked out at below:
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/halide_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/halide_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the halide package gets checked out

CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
VERBOSE=${VERBOSE:=0}

CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

export INSTALL_PREFIX=$PREFIX
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "PREFIX: ${PREFIX}"

export CLANG_PREFIX=$($PREFIX/bin/llvm-config --prefix)
echo "CLANG_PREFIX: ${CLANG_PREFIX}"

echo "Building Halide conda package"

echo "Clean up existing build packages if any"
rm -rf build || true

mkdir -p build
cd build

echo "Configuring Halide"

LLVM_CONFIG_FROM_PREFIX=${CLANG_PREFIX}/bin/llvm-config
LLVM_CONFIG=$( which $LLVM_CONFIG_FROM_PREFIX || which llvm-config-4.0 || which llvm-config )
CLANG_FROM_PREFIX=${CLANG_PREFIX}/bin/clang
CLANG=$( which $CLANG_FROM_PREFIX || which clang-4.0 || which clang )

CLANG=${CLANG} \
LLVM_CONFIG=${LLVM_CONFIG} \
VERBOSE=${VERBOSE} \
PREFIX=${INSTALL_PREFIX} \
WITH_LLVM_INSIDE_SHARED_LIBHALIDE= \
WITH_OPENCL= \
WITH_OPENGL= \
WITH_METAL= \
WITH_EXCEPTIONS=1 \
make -f ../Makefile -j"$(nproc)" install || exit 1
mkdir -p ${INSTALL_PREFIX}/include/Halide
mv ${INSTALL_PREFIX}/include/Halide*.h  ${INSTALL_PREFIX}/include/Halide/ || exit 1

echo "Successfully built Halide conda package"
