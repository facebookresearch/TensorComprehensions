#!/bin/bash

set -ex

source /etc/lsb-release

# condition: if 14.04 and conda, conda install pytorch and build
# condition: if 16.04 and conda, conda install pytorch and build
# condition: if any and non-conda, simply build TC from scratch

if [[ "$DISTRIB_RELEASE" == 14.04 ]]; then
  if [[ $(conda --version | wc -c) -ne 0 ]]; then
    echo "Building TC in conda env"
    source activate
    conda install -y -c pytorch pytorch
    conda install -y pyyaml
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  else
    echo "Building TC in non-conda env"
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  fi
fi

if [[ "$DISTRIB_RELEASE" == 16.04 ]]; then
  if [[ $(conda --version | wc -c) -ne 0 ]]; then
    echo "Building TC in conda env"
    source activate
    conda install -y pytorch cuda90 -c pytorch
    conda install -y pyyaml
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  else
    echo "Building TC in non-conda env"
    WITH_PYTHON_C2=OFF CORES=$(nproc) CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
  fi
fi
