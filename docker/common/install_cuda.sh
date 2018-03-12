#!/bin/bash

set -ex

source /etc/lsb-release

if [[ "$DISTRIB_RELEASE" == 14.04 ]]; then
  export CUDA_VERSION=8.0
  export CUDNN_VERSION=7

  apt-get update
  apt-get install -y --no-install-recommends wget curl

  pushd /tmp
  export ML_REPO_PKG="nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb"
  wget --no-check-certificate https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/${ML_REPO_PKG}
  dpkg -i ${ML_REPO_PKG}
  rm -f ${ML_REPO_PKG}
  popd

  CUDNN_PKG_VERSION="7.0.5.15-1+cuda${CUDA_VERSION}"
  apt-get update
  # force-yes install since there is already a libcudnn shipped by nvidia docker
  # image but with a much higher libcudnn version
  apt-get install -y --force-yes --no-install-recommends \
    "libcudnn${CUDNN_VERSION}=${CUDNN_PKG_VERSION}" \
    "libcudnn${CUDNN_VERSION}-dev=${CUDNN_PKG_VERSION}"

  ls /usr/local/cuda/targets/x86_64-linux/lib

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
fi
