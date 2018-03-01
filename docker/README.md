# How to build Tensor Comprehensions docker images

Tensor Comprehensions supports gcc4.8, gcc5, cuda8-cudnn6, cuda9-cudnn7, Xenial, Trusty
and building in Conda vs no conda environment. Below are the steps you can follow
to build the images for various combinations. The docker image is built in a ladder
manner so as to reuse the images. Steps:

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
linux trusty with gcc4.8, cuda8, cudnn6 and conda environment settings. If you
need to build images for some other combination, the steps are still valid, just
make sure to change the paths below to fit your case.

```Shell
docker build -t tensorcomprehensions/linux-trusty-gcc4.8-cuda8-cudnn6-py3-conda .
```

This will build the image for the above permutation and then we can test this image

6. Test the image

```Shell
docker run --runtime=nvidia -i -t tensorcomprehensions/linux-trusty-gcc4.8-cuda8-cudnn6-py3-conda:2
# now clone the TC repo
cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive && cd TensorComprehensions
# build TC
WITH_PYTHON_C2=OFF CLANG_PREFIX=/usr/local/clang+llvm-tapir5.0 ./build.sh --all
# Test the TC build is fine
./test_cpu.sh
./test.sh
```
