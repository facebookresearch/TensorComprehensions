#!/usr/bin/env bash

set -e

# code gets checked out at below:
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/clang+llvm-tapir5.0_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/clang+llvm-tapir5.0_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the clang+llvm-tapir5.0 package gets checked out
# NOTE: once the build finished, the packaging and unpackaging step takes long
# and might seem like it's stuck but it's not.

CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
VERBOSE=${VERBOSE:=0}

export INSTALL_PREFIX=$PREFIX
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "PREFIX: ${PREFIX}"

echo "Building clang+llvm-tapir5.0 conda package"

echo "Clean up existing build packages if any"
rm -rf llvm_build || true

mkdir -p llvm_build
cd llvm_build

echo "Configuring clang+llvm-tapir5.0"
VERBOSE=${VERBOSE}  ${CMAKE_VERSION} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCOMPILER_RT_BUILD_CILKTOOLS=OFF \
    -DLLVM_ENABLE_CXX1Y=ON \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_LLVM_DYLIB=ON  \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_INSTALL_OCAMLDOC_HTML_DIR=/tmp \
    -DLLVM_OCAML_INSTALL_PATH=/tmp \
    -DLLVM_ENABLE_RTTI=ON .. || exit 1

echo "Installing clang+llvm-tapir5.0"
VERBOSE=${VERBOSE}  make -j"$(nproc)" -s || exit 1

VERBOSE=${VERBOSE}  make install -j"$(nproc)" -s || exit 1

echo SUCCESS || echo FAILURE

echo "Successfully built clang+llvm-tapir5.0 conda package"
