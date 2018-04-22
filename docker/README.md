# How to build Tensor Comprehensions docker images

Tensor Comprehensions supports gcc4.9, gcc5, cuda8-cudnn6, cuda9-cudnn7, Xenial, Trusty
and building in Conda/ Non-conda environment. Below are the steps you can follow
to build the images for various combinations. This directory contains everything needed to build the Docker images that are used in our CI as well. The `Dockerfile`s are parameterized to build the image based on the build environment. The different configurations are identified by a freeform string that we call a build environment. This string is persisted in each image as the BUILD_ENVIRONMENT environment variable.

## Supported Build environments

Non-conda env and no python installs:

`linux-trusty-gcc4.9-cuda8-cudnn7-py3`

`linux-xenial-gcc5-cuda9-cudnn7-py3`

Conda env with PyTorch conda install:

`linux-trusty-gcc4.9-cuda8-cudnn7-py3-conda` [Run both C++ and Python tests]

`linux-xenial-gcc5-cuda9-cudnn7-py3-conda` [Run only Python tests for now]

See `docker_build.sh` for a full list of terms that are extracted from the build environment into parameters for the image build.

## Contents

`docker_build.sh` -- dispatch script to launch all builds

`common` -- scripts used to execute individual Docker build stages

`ubuntu-cuda` -- Dockerfile for Ubuntu image with CUDA support in non-conda env

`ubuntu-cuda-conda` -- Dockerfile for Ubuntu image with CUDA support in non-conda env

## Steps to build docker image

**NOTE**: You need to have docker installed on your system. Follow the instructions
on docker website to install it, if you don't have docker already.

You can verify your docker installation is fine by running a docker test:

```Shell
docker run hello-world
```

1. Clone TensorComprehensions repo

```Shell
git clone --recursive git@github.com:facebookresearch/TensorComprehensions.git
```

2. Build the docker image

Go to the TensorComprehensions repo

```Shell
cd TensorComprehensions/docker
```

Now, we will build a docker image for TC. We will build docker image for TC on
linux trusty with gcc4.9, cuda8, cudnn7 and conda environment settings. If you
need to build images for some other combination, the steps are still valid, just
make sure to change the paths below to fit your case.

```Shell
image=linux-trusty-gcc4.9-cuda8-cudnn7-py3-conda ./docker_build.sh
```

This will build the image for the above permutation and then we can test this image

6. Test the image

run `docker images` and get the `IMAGE_ID` for the image we just built.

```Shell
docker run --runtime=nvidia -i -t ${IMAGE_ID}
# now clone the TC repo
cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive && cd TensorComprehensions
git submodule update --init --recursive
# build TC
conda install -y mkl-include pyyaml
conda install -yc conda-forge pytest
conda install -y pytorch-nightly=2018.04.17 -c pytorch # OR  conda install -y pytorch-nightly=2018.04.17 cuda90 -c pytorch
CORES=$(nproc) WITH_CAFFE2=ON CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 BUILD_TYPE=Release ./build.sh --all
# Test the TC build is fine
./test.sh
./test_python/run_test.sh
```
