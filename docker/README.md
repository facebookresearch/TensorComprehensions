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

1. First, you need to clone TensorComprehensions repo

```Shell
git clone --recursive git@github.com:facebookresearch/TensorComprehensions.git
```

and go to the TensorComprehensions repo

```Shell
cd TensorComprehensions/docker
```
Now, we are going to start building docker image for TC in various steps. We will
build docker image for TC on linux trusty with gcc4.8, cuda8, cudnn6 and conda environment settings. If you need to build images for some other combination,
the steps are still valid, just make sure to change the paths below to fit your case.

2. We will first build the docker image of trusty and all the dependencies of TC. Since the dependencies don't change, we can build this image once and keep reusing it.

```Shell
cd linux-trusty                    # go to the relevant directory
docker build -t tensorcomprehensions/linux-trusty:2 .   # build image with tag 2 (this can be any number)
```

3. Now, we will build the image using the base image built in step 1 and we will add gcc4.8 and clang+llvm-tapir5.0 to the image

```Shell
cd TensorComprehensions/docker/linux-trusty-gcc4.8-tapir5.0
docker build --build-arg BUILD_ID=2 -t tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0:2 .
```

4. Now, let's use the above image and add cuda8-cudnn6 to the image

```Shell
cd TensorComprehensions/docker
docker build --build-arg BUILD_ID=2 --build-arg BUILD=gcc48-cuda8-cudnn6 -f linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6/Dockerfile -t tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6:2 .
```

5. Now, using above image, let's build the final image with python3 + conda

```Shell
cd TensorComprehensions/docker/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6-py3-conda/
docker build --build-arg BUILD_ID=2 -t tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6-py3-conda:2 .
```

6. Finally, let's run the above image, and install TC to verify the image built
is correct.

```Shell
docker run -i -t linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6-py3-conda:2
# now clone the TC repo and run the build, at the end run test_cpu.sh to verify
```
