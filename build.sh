#! /bin/bash
set -ex

export TC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! test ${CLANG_PREFIX}; then
    echo 'Environment variable CLANG_PREFIX is required, please run export CLANG_PREFIX=$(llvm-config --prefix)'
    exit 1
fi

ATEN_NO_CUDA=${ATEN_NO_CUDA:=0}
WITH_CAFFE2=${WITH_CAFFE2:=ON}
WITH_PYTHON_C2=${WITH_PYTHON_C2:=OFF}
WITH_NNPACK=${WITH_NNPACK:=OFF}
PYTHON=${PYTHON:="`which python3`"}
PROTOC=${PROTOC:="`which protoc`"}
CORES=${CORES:=32}
VERBOSE=${VERBOSE:=0}
CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
CAFFE2_BUILD_CACHE=${CAFFE2_BUILD_CACHE:=${TC_DIR}/third-party/.caffe2_build_cache}
HALIDE_BUILD_CACHE=${HALIDE_BUILD_CACHE:=${TC_DIR}/third-party/.halide_build_cache}
INSTALL_PREFIX=${INSTALL_PREFIX:=${TC_DIR}/third-party-install/}
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

gflags=""
if [[ $* == *--gflags* ]]
then
    echo "Building GFlags"
    gflags="1"
fi

glog=""
if [[ $* == *--glog* ]]
then
    echo "Building Glog"
    glog="1"
fi

aten=""
if [[ $* == *--aten* ]]
then
    echo "Building ATen"
    aten="1"
fi

caffe2=""
if [[ $* == *--caffe2* ]]
then
    echo "Building Caffe2"
    caffe2="1"
fi

isl=""
if [[ $* == *--isl* ]]
then
    echo "Building ISL"
    isl="1"
fi

dlpack=""
if [[ $* == *--dlpack* ]]
then
    echo "Building DLPACK"
    dlpack="1"
fi

tc=""
if [[ $* == *--tc* ]]
then
    echo "Building TC"
    tc="1"
fi

halide=""
if [[ $* == *--halide* ]]
then
    echo "Building Halide"
    halide="1"
fi

all=""
if [[ $* == *--all* ]]
then
    echo "Building ALL"
    all="1"
fi

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

function install_gflags() {
  mkdir -p ${TC_DIR}/third-party/gflags/build || exit 1
  cd       ${TC_DIR}/third-party/gflags/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/gflags" ]; then

      if should_rebuild ${TC_DIR}/third-party/gflags ${TC_DIR}/third-party/.gflags_build_cache; then
          echo "Installing Gflags"

          if should_reconfigure .. .build_cache; then
              echo "Reconfiguring GFlags"
              VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DGFLAGS_BUILD_SHARED_LIBS=ON -DGFLAGS_BUILD_STATIC_LIBS=OFF -DGFLAGS_BUILD_TESTING=ON -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..  || exit 1
          fi
          VERBOSE=${VERBOSE} make -j $CORES -s || exit 1

          set_cache .. .build_cache
          set_bcache ${TC_DIR}/third-party/gflags ${TC_DIR}/third-party/.gflags_build_cache
      fi

      VERBOSE=${VERBOSE} make install -j $CORES -s || exit 1
      echo "Successfully installed GFlags"

  fi
}

function install_glog() {
  mkdir -p ${TC_DIR}/third-party/glog/build || exit 1
  cd       ${TC_DIR}/third-party/glog/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/glog" ]; then

      if should_rebuild ${TC_DIR}/third-party/glog ${TC_DIR}/third-party/.glog_build_cache; then
          echo "Installing Glog"

          if should_reconfigure .. .build_cache; then
              echo "Reconfiguring Glog"
              VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RELWITHDEBINFO -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_DEBUG_POSTFIX="" ..  || exit 1
          fi
          VERBOSE=${VERBOSE} make -j $CORES -s || exit 1

          set_cache .. .build_cache
          set_bcache ${TC_DIR}/third-party/glog ${TC_DIR}/third-party/.glog_build_cache
      fi

      CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} VERBOSE=${VERBOSE} make install -j $CORES -s || exit 1
      echo "Successfully installed Glog"

  fi
}

function install_aten() {
  mkdir -p ${TC_DIR}/third-party/ATen/build || exit 1
  cd       ${TC_DIR}/third-party/ATen/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/ATen" ]; then

    if test ${USE_CONTBUILD_CACHE}; then
        echo "Setting compute-arch to 5.0 for ATen build"
        export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
        export TORCH_CUDA_ARCH_LIST="5.0"
        echo ${TORCH_NVCC_FLAGS}
        echo ${TORCH_CUDA_ARCH_LIST}
    fi

    # ATen errors out when using many threads for building - use 1 core for now
    if should_rebuild ${TC_DIR}/third-party/ATen ${TC_DIR}/third-party/.aten_build_cache; then
      echo "Installing ATen"

      if should_reconfigure .. .build_cache; then
        echo "Reconfiguring ATen"
        export PYTORCH_PYTHON=${PYTHON}
        ${CMAKE_VERSION} .. -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DHAS_C11_ATOMICS=OFF -DNO_CUDA=${ATEN_NO_CUDA}
      fi
      VERBOSE=${VERBOSE} make -j $CORES -s || exit 1

      set_cache .. .build_cache
      set_bcache ${TC_DIR}/third-party/ATen ${TC_DIR}/third-party/.aten_build_cache
    fi

    VERBOSE=${VERBOSE} make install -j $CORES -s || exit 1
    echo "Successfully installed ATen"

  fi
}

function install_caffe2() {
  mkdir -p ${TC_DIR}/third-party/caffe2/build || exit 1
  cd       ${TC_DIR}/third-party/caffe2/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/caffe2" ]; then

  if should_rebuild ${TC_DIR}/third-party/caffe2 ${CAFFE2_BUILD_CACHE}; then
  echo "Installing Caffe2"
  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring Caffe2"
    rm -rf * || exit 1

    if ! test ${USE_CONTBUILD_CACHE}; then
      ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPYTHON_EXECUTABLE=${PYTHON} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_PYTHON=${WITH_PYTHON_C2} -DUSE_GLOG=OFF -DUSE_GFLAGS=OFF -DUSE_NNPACK=${WITH_NNPACK} -DUSE_GLOO=OFF -DUSE_NCCL=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF -DBUILD_TEST=OFF -DUSE_OPENCV=OFF -DUSE_OPENMP=OFF -DCMAKE_INSTALL_MESSAGE=NEVER -DCMAKE_CXX_FLAGS="-fno-var-tracking-assignments" -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC} -DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR} -DCUB_INCLUDE_DIR=${CUB_INCLUDE_DIR} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} .. || exit
    else
      ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPYTHON_EXECUTABLE=${PYTHON} -DCUDA_ARCH_NAME="Maxwell" -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_PYTHON=${WITH_PYTHON_C2} -DUSE_GLOG=OFF -DUSE_GLOO=OFF -DUSE_NNPACK=${WITH_NNPACK} -DUSE_NCCL=OFF -DUSE_GFLAGS=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF -DBUILD_TEST=OFF -DUSE_OPENCV=OFF -DUSE_OPENMP=OFF -DCMAKE_INSTALL_MESSAGE=NEVER -DCMAKE_CXX_FLAGS="-fno-var-tracking-assignments" -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} .. || exit
    fi
  fi
  VERBOSE=${VERBOSE} make -j $CORES install -s || exit 1

  set_cache .. .build_cache
  set_bcache ${TC_DIR}/third-party/caffe2 ${CAFFE2_BUILD_CACHE}
  fi

  echo "Successfully installed caffe2"

  fi
}

function install_isl() {
  mkdir -p ${TC_DIR}/third-party/islpp/build || exit 1
  cd       ${TC_DIR}/third-party/islpp/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/isl" ]; then

  if should_rebuild ${TC_DIR}/third-party/islpp ${TC_DIR}/third-party/.islpp_build_cache; then
  echo "Installing ISL"

  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring ISL"
    rm -rf * || exit 1
    VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DISL_INT=gmp -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
  fi
  VERBOSE=${VERBOSE} make -j $CORES -s || exit 1

  set_cache .. .build_cache
  set_bcache ${TC_DIR}/third-party/islpp ${TC_DIR}/third-party/.islpp_build_cache
  fi

  make install -j $CORES -s || exit 1
  echo "Successfully installed isl"

  fi
}

function install_dlpack() {
  mkdir -p ${TC_DIR}/third-party/dlpack/build || exit 1
  cd       ${TC_DIR}/third-party/dlpack/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/dlpack" ]; then

  if should_rebuild ${TC_DIR}/third-party/dlpack ${TC_DIR}/third-party/.dlpack_build_cache; then
  echo "Installing DLPACK"

  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring DLPACK"
    rm -rf * || exit 1
    VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
  fi
  VERBOSE=${VERBOSE} make -j $CORES -s || exit 1

  set_cache .. .build_cache
  set_bcache ${TC_DIR}/third-party/dlpack ${TC_DIR}/third-party/.dlpack_build_cache
  fi

  # make install -j $CORES -s || exit 1
  cp -R ${TC_DIR}/third-party/dlpack/include/dlpack ${INSTALL_PREFIX}/include/
  echo "Successfully installed DLPACK"

  fi
}

function install_cub() {
    cp -R ${TC_DIR}/third-party/cub/cub ${INSTALL_PREFIX}/include/
}

function install_tc_python() {
    echo "Setting up python now"
    export PYTHONPATH=${TC_DIR}/build/pybinds:${PYTHONPATH}
    echo "PYTHONPATH: ${PYTHONPATH}"
}

function install_tc() {
  install_cub

  mkdir -p ${TC_DIR}/build || exit 1
  cd       ${TC_DIR}/build || exit 1

  echo "Installing TC"

  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring TC"
    rm -rf *
    VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DWITH_CAFFE2=${WITH_CAFFE2} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DHALIDE_PREFIX=${INSTALL_PREFIX} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}/lib/cmake \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC} \
        -DCLANG_PREFIX=${CLANG_PREFIX} \
        -DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR} \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} .. || exit 1
  fi

  set_cache .. .build_cache
  VERBOSE=${VERBOSE} make -j $CORES -s || exit 1
  VERBOSE=${VERBOSE} make install -j $CORES -s || exit 1

  install_tc_python

  echo "Successfully installed TC"
}

function install_halide() {
  mkdir -p ${TC_DIR}/third-party/halide/build || exit 1
  cd       ${TC_DIR}/third-party/halide/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/Halide" ]; then

    if should_rebuild ${TC_DIR}/third-party/halide ${HALIDE_BUILD_CACHE}; then
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
      make -f ../Makefile -j $CORES install || exit 1
      mkdir -p ${INSTALL_PREFIX}/include/Halide
      mv ${INSTALL_PREFIX}/include/Halide*.h  ${INSTALL_PREFIX}/include/Halide/
      set_bcache ${TC_DIR}/third-party/halide ${HALIDE_BUILD_CACHE}
    fi

    echo "Successfully installed Halide"

  fi
}

if ! test -z $gflags || ! test -z $all; then
  if [[ $(find $CONDA_PREFIX -name libgflags.so) ]]; then
      echo "gflags found"
  else
      echo "no files found"
      install_gflags
  fi
fi

if ! test -z $glog || ! test -z $all; then
    if [[ $(find $CONDA_PREFIX -name libglog.so) ]]; then
        echo "glog found"
    else
        echo "no files found"
        install_glog
    fi
fi

if ! test -z $aten || ! test -z $all; then
    if python -c "import torch" &> /dev/null; then
        echo 'PyTorch is installed, libATen.so will be used from there to avoid two copies'
    else
        install_aten
    fi
fi

if ! test -z $caffe2 || ! test -z $all ; then
    if [ "$WITH_CAFFE2" == "ON" ]; then
        if [[ $(find $CONDA_PREFIX -name libcaffe2_gpu.so) ]]; then
            echo "caffe2 found"
        else
            echo "no files found"
            install_caffe2
        fi
    fi
fi

if ! test -z $isl || ! test -z $all; then
    if [[ $(find $CONDA_PREFIX -name libisl.so) ]]; then
        echo "isl found"
    else
        echo "no files found"
        install_isl
    fi
fi

if ! test -z $dlpack || ! test -z $all; then
    install_dlpack
fi

if ! test -z $halide || ! test -z $all; then
    if [[ $(find $CONDA_PREFIX -name libHalide.so) ]]; then
        echo "Halide found"
    else
        echo "no files found"
        install_halide
    fi
fi

if ! test -z $tc || ! test -z $all; then
    install_tc
fi
