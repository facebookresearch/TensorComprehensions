#!/bin/bash

set -ex

# Install common dependencies
# cleanup again to avoid any sha mismatch
apt-get clean
rm -rf /var/lib/apt/lists/*
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
  autoconf \
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

# cleanup again to avoid any sha mismatch
apt-get clean
rm -rf /var/lib/apt/lists/*
# setup gcc
if [[ "$GCC_VERSION" == 4.9 ]]; then
  add-apt-repository ppa:ubuntu-toolchain-r/test
  apt-get update
  apt-get install -y --no-install-recommends libcilkrts5 gcc-$GCC_VERSION g++-$GCC_VERSION
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GCC_VERSION 50
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GCC_VERSION 50
else
  apt-get install -y --no-install-recommends libcilkrts5 gcc g++
fi

# Install ccache from source. Needs 3.4 or later for ccbin support
# Needs specific branch to work with nvcc (ccache/ccache#145)
# Also pulls in a commit that disables documentation generation,
# as this requires asciidoc to be installed (which pulls in a LOT of deps).
echo "Installing ccache"
pushd /tmp
git clone https://github.com/pietern/ccache -b ccbin
pushd ccache
./autogen.sh
./configure --prefix=/usr/local
make "-j$(nproc)" install
popd
popd

# Install ccache symlink wrappers
# A good read on ccache: https://software.intel.com/en-us/articles/accelerating-compilation-part-1-ccache
echo "Setting up ccache wrappers"
pushd /usr/local/bin
ln -sf "$(which ccache)" cc
ln -sf "$(which ccache)" c++
ln -sf "$(which ccache)" gcc
ln -sf "$(which ccache)" g++
ln -sf "$(which ccache)" x86_64-linux-gnu-gcc
# Install ccache wrapper for nvcc. We are using NVIDIA image so nvcc is there.
ln -sf "$(which ccache)" nvcc
# set the cache limit
ccache -M 25Gi

export CCACHE_WRAPPER_DIR="$PWD/ccache"
export PATH="$CCACHE_WRAPPER_DIR:$PATH"
# CMake should use ccache symlink for nvcc
export CUDA_NVCC_EXECUTABLE="$PWD/nvcc"
popd

echo "Setting CC and CXX env variables"
export CC=$(which gcc)
export CXX=$(which g++)
echo "CC: ${CC}"
echo "CXX: ${CXX}"

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
