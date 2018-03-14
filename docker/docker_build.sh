#!/bin/bash

set -ex

image=${image}
if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

OS="ubuntu"
# ubuntu version
if [[ "$image" == *-trusty* ]]; then
  UBUNTU_VERSION=14.04
elif [[ "$image" == *-xenial* ]]; then
  UBUNTU_VERSION=16.04
fi

CUDA_VERSION="$(echo "${image}" | perl -n -e'/-cuda(\d+(?:\.\d+)?)/ && print $1')"
CUDNN_VERSION="$(echo "${image}" | perl -n -e'/-cudnn(\d+)/ && print $1')"
GCC_VERSION="$(echo "${image}" | perl -n -e'/-gcc([^-]+)/ && print $1')"
CMAKE_VERSION=3.6.3   # hardcode it, we need cmake > 3.4.3 for Taper
DOCKERFILE="${OS}-cuda/Dockerfile"

# cuda versions
if [[ "$CUDA_VERSION" == "8" ]]; then
  CUDA_VERSION=8.0
elif [[ "$CUDA_VERSION" == "9" ]]; then
  CUDA_VERSION=9.0
fi

# Get python settings
if [[ "$image" == *-conda* ]]; then
  DOCKERFILE="${OS}-cuda-conda/Dockerfile"
fi

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

echo "============Printing summary============="
echo "image: ${image}"
echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "CUDNN_VERSION: ${CUDNN_VERSION}"
echo "GCC_VERSION: ${GCC_VERSION}"
echo "DOCKERFILE: ${DOCKERFILE}"
echo "============Summary Ended================"


# Build image
docker build \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "CMAKE_VERSION=${CMAKE_VERSION}" \
       --build-arg "JENKINS=${JENKINS:-}" \
       --build-arg "JENKINS_UID=${JENKINS_UID:-}" \
       --build-arg "JENKINS_GID=${JENKINS_GID:-}" \
       --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       -f $(dirname ${DOCKERFILE})/Dockerfile \
       "$@" \
       .
