#!/usr/bin/env bash

set -e

# code gets checked out at below:
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/isl-tc_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/isl-tc_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the packaged isl-tc package gets checked out

CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
VERBOSE=${VERBOSE:=0}

CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

export INSTALL_PREFIX=$PREFIX
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "PREFIX: ${PREFIX}"

echo "Building ISL-TC conda package"
mkdir -p build
cd build

echo "Clean up existing build packages if any"
rm -rf build || true

echo "Configuring ISL-TC"
VERBOSE=${VERBOSE} ${CMAKE_VERSION} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DISL_INT=gmp \
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ..

VERBOSE=${VERBOSE} make -j"$(nproc)" -s || exit 1

VERBOSE=${VERBOSE} make install -j"$(nproc)" -s || exit 1

echo "Successfully built isl-tc conda package"
