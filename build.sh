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

WITH_CUDA=${WITH_CUDA:=ON}
if test ${WITH_CUDA} == ON; then
    if ! test ${CUDA_TOOLKIT_ROOT_DIR}; then
        echo 'When CUDA is activated, TC requires CUDA_TOOLKIT_ROOT_DIR to build see BUILD.md'
        exit 1
    fi
fi

THIRD_PARTY_INSTALL_PREFIX=${CONDA_PREFIX}
mkdir -p ${THIRD_PARTY_INSTALL_PREFIX}

INSTALL_PREFIX=${INSTALL_PREFIX:=${TC_DIR}/install}
mkdir -p ${INSTALL_PREFIX}

WITH_CAFFE2=${WITH_CAFFE2:=OFF}
WITH_TAPIR=${WITH_TAPIR:=ON}
PYTHON=${PYTHON:="`which python3`"}
CORES=${CORES:=32}
VERBOSE=${VERBOSE:=0}
CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

echo $TC_DIR $GCC_VER
BUILD_TYPE=${BUILD_TYPE:=Debug}
echo "Build Type: ${BUILD_TYPE}"

clean=""
if [[ $* == *--clean* ]]
then
    echo "Forcing clean"
    clean="1"
fi

tc=""
if [[ $* == *--tc* ]]
then
    echo "Building TC"
    tc="1"
fi

all=""
if [[ $* == *--all* ]]
then
    echo "Building ALL"
    all="1"
fi

if [[ "${VERBOSE}" = "1" ]]; then
  set -x
fi

orig_make=$(which make)

function make() {
    # Workaround for https://cmake.org/Bug/view.php?id=3378
    if [[ "${VERBOSE}" = "1" ]]; then
        VERBOSE=${VERBOSE} ${orig_make} $@
    else
        ${orig_make} $@
    fi
}

function set_cache() {
    stat --format="%n %Y %Z %s" `find $1 -name CMakeLists.txt -o -name autogen.sh -o -name configure -o -name Makefile -exec realpath {} \;` > $2
}

function should_reconfigure() {
    if [ "$clean" = "1" ]
    then
        true
    else
        if [ -e $2 ]
        then
            OLD_STAT=`cat $2`
            NEW_STAT=$(stat --format="%n %Y %Z %s" `find $1 -name CMakeLists.txt -o -name autogen.sh -o -name configure -o -name Makefile -exec realpath {} \;`)
            if [ "$OLD_STAT" = "$NEW_STAT" ]
            then
                false
            else
                true
            fi
        else
            true
        fi
    fi
}

function set_bcache() {
    stat --format="%n %Y %Z %s" $1 > $2
}

function should_rebuild() {
    if [ "$clean" = "1" ]
    then
        true
    else
        if [ -e $2 ]
        then
            OLD_STAT=`cat $2`
            NEW_STAT=$(stat --format="%n %Y %Z %s" $1)
            if [ "$OLD_STAT" = "$NEW_STAT" ]
            then
                false
            else
                true
            fi
        else
            true
        fi
    fi
}

function install_tc() {
  mkdir -p ${TC_DIR}/build || exit 1
  cd       ${TC_DIR}/build || exit 1

  echo "Installing TC"

  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring TC"
    rm -rf *
    VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DWITH_CAFFE2=${WITH_CAFFE2} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_TAPIR=${WITH_TAPIR} \
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PREFIX}/lib/cmake \
        -DTHIRD_PARTY_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DPROTOBUF_PROTOC_EXECUTABLE=${THIRD_PARTY_INSTALL_PREFIX}/bin/protoc \
        -DCLANG_PREFIX=${CLANG_PREFIX} \
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
        -DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR} \
        -DWITH_CUDA=${WITH_CUDA} \
        -DTC_DIR=${TC_DIR} \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} .. || exit 1
  fi

  set_cache .. .build_cache
  make -j $CORES -s || exit 1
  make install -j $CORES -s || exit 1

  echo "Successfully installed TC"
}

if ! test -z $tc || ! test -z $all; then
    install_tc
fi
