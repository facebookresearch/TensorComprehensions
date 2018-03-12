#!/bin/bash

set -ex

# NOTE: This script is called only when we are in conda environment

apt-get update
apt-get install -y --no-install-recommends wget curl

source /etc/lsb-release

if [[ "$DISTRIB_RELEASE" == 14.04 ]]; then
  export CUDA_REPO_PATH="ubuntu1404"
  export CUDA_VERSION=8.0
  export ML_REPO_PKG="nvidia-machine-learning-repo-${CUDA_REPO_PATH}_4.0-2_amd64.deb"
else
  export CUDA_REPO_PATH="ubuntu1604"
  export CUDA_VERSION=9.0
  export ML_REPO_PKG="nvidia-machine-learning-repo-${CUDA_REPO_PATH}_1.0.0-1_amd64.deb"
fi

export CUDNN_VERSION=7

pushd /tmp
wget --no-check-certificate https://developer.download.nvidia.com/compute/machine-learning/repos/${CUDA_REPO_PATH}/x86_64/${ML_REPO_PKG}
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
