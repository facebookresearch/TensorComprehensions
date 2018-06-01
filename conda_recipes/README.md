# Important notice
***In order to uniformize and simplify the build system we had to make choices. TC is currently only officially supported on Ubuntu 16.04 with gcc 5.4.0, cuda 9.0 and cudnn 7.1.***
Other configurations may work too but are not yet officially supported.
For more information about setting up the config that we use to build the conda dependencies see the following [Dockerfile](conda_recipes/Dockerfile).

Our main goal with this decision is to make the build procedure extremely simple, reproducible both internally and extensible to new targets in the future.
In particular the gcc-4 / gcc-5 ABI switch is not something we want to concern ourselves with at this point, we go for gcc-5.4.0.

# How to build conda package for Tensor Comprehensions and its dependencies

```
cd conda_recipes
docker build -t tc-cuda9.0-cudnn7.1-ubuntu16.04-devel docker-images/tc-cuda9.0-cudnn7.1-ubuntu16.04-devel
docker image ls
nvidia-docker run --rm -i -t tc-cuda9.0-cudnn7.1-ubuntu16.04-devel
```

We are ready to build conda package for TC.
To simplify the build process we ship TC dependencies as conda packages.
We need to build packages for `llvm-tapir50`, `Halide`, `Caffe2` (optional) and finally `Tensor Comprehensions`.

For building each package, we need to specify a `build version`, `build number` and
`git hash`. This information is used to build each package.

Now, we will go ahead and build the conda package of TC and all of its dependencies. For that, run the command below:

```Shell
cd /
git clone http://www.github.com/facebookresearch/TensorComprehensions --recursive
cd TensorComprehensions/conda_recipes
. /opt/conda/anaconda/bin/activate
source activate tc_build
conda build purge
./conda_build_tc.sh
```

Conda packages are built in `<anaconda_root>/conda-bld/linux-64`

We can now ship the packages to our conda channel:
```
anaconda login
anaconda upload /opt/conda/anaconda/envs/tc_build/conda-bld/linux-64/...
```
