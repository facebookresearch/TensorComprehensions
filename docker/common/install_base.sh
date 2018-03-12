#!/bin/bash

set -ex

# Install common dependencies
apt-get update
apt-get install -y --no-install-recommends \
  curl \
  make \
  git \
  ssh \
  realpath \
  wget \
  unzip \
  vim \
  automake \
  libtool \
  valgrind \
  subversion \
  libgtest-dev \
  libz-dev \
  libgmp3-dev \
  libyaml-dev \
  libgoogle-glog-dev \
  ca-certificates \
  software-properties-common \
  build-essential

# setup gcc
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y --no-install-recommends libcilkrts5 gcc-$GCC_VERSION g++-$GCC_VERSION
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GCC_VERSION 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GCC_VERSION 50

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# install cmake - this unifies trusty/xenial cmake version > 3.4.3
[ -n "$CMAKE_VERSION" ]

# Turn 3.6.3 into v3.6
path=$(echo "${CMAKE_VERSION}" | sed -e 's/\([0-9].[0-9]\+\).*/v\1/')
file="cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz"
# Download and install specific CMake version in /usr/local
pushd /tmp
curl -Os "https://cmake.org/files/${path}/${file}"
tar -C /usr/local --strip-components 1 --no-same-owner -zxf cmake-*.tar.gz
rm -f cmake-*.tar.gz
popd


# LLVM+Clang-Tapir5.0
export LLVM_SOURCES=/tmp/llvm_sources-tapir5.0
mkdir -p $LLVM_SOURCES
pushd $LLVM_SOURCES

export CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0
export CMAKE_VERSION=cmake

git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm && \
mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build && \
${CMAKE_VERSION} -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_INSTALL_OCAMLDOC_HTML_DIR=/tmp \
  -DLLVM_OCAML_INSTALL_PATH=/tmp \
  -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DCOMPILER_RT_BUILD_CILKTOOLS=OFF \
  -DLLVM_ENABLE_CXX1Y=ON \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_LLVM_DYLIB=ON  \
  -DLLVM_ENABLE_RTTI=ON ../llvm/

make -j"$(nproc)" -s && make install -j"$(nproc)" -s

popd
rm -Rf ${LLVM_SOURCES}

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
