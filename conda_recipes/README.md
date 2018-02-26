# How to build conda package for Tensor Comprehensions and its dependencies

## Building Conda Packages inside TC nvidia-docker image

We have a runtime nvidia docker image for TC to build conda packages and run all
tests including gpu ones. To build the packages, follow the below commands:

```Shell
# Build a new nvidia docker image
cd TensorComprehensions/conda_recipes
docker build -t trusty-gcc4.8-py3-conda-cuda:2 .
```

Now, we have the docker image built, we will run it by following command:

```Shell
nvidia-docker run --rm -i -t trusty-gcc4.8-py3-conda-cuda:2
```

Once you are inside the docker image, run `nvidia-smi` and verify that you have
gpus access. Now, you are good to go and build conda packages. For that run the
following command:

```Shell
cd TensorComprehensions/conda_recipes
./conda_build_tc.sh
```

Sit back, relax and let the packages build. If you see any errors in packages, fix them.

## Building conda packages outside docker image

Steps:
1. You need to have conda installed. Follow the instructions in [dockerfile](https://github.com/facebookresearch/TensorComprehensions/tree/master/docker/linux-trusty-gcc4.8-cuda8-cudnn6-py3-conda) for how to install conda

*NOTE*: Make sure that conda get download things from web and you might need some
proxy-setup. You can add following lines to your `~/.condarc` file

```
proxy_servers:
  http: <http_proxy_to_use>
  https: <https_proxy_to_use>
```

2. Next install conda build

```Shell
conda install conda-build
```

This installs a bunch of dependencies like patchelf, glob, pkginfo, conda-verify.
When prompted, say yes to install these dependencies.

3. While building conda packaged, we might have some third-party dependencies for which we want to install the conda package available on [conda-forge](https://github.com/conda-forge/feedstocks). In order to install packages from conda-forge, we need to add conda-forge to the conda channels by running the following command

We will add `soumith` channel since we will get pytorch from there. Channels are prioritized in the order they were added.

```Shell
conda config --add channels conda-forge
conda config --add channels pytorch
```

For example: this is required for the libgmp dependency of ISL and we can get
conda package of libgmp from conda-forge

4. Now create a Python 3.6 conda environment for packaging TC by following below
instructions:

```Shell
conda create --name tc_build python=3.6
source activate tc_build
```

`tc_build` is the name of environment. You may chose any name you want.

**NOTE**: If you want to exit the conda environment, you can run

```Shell
source deactivate
```

5. Now, we have conda setup, we are ready to build conda package for TC. For this,
since TC has its dependencies that are linked dynamically, we will build conda
packages for them separately. In short, we need to build packages for `clang+llvm-tapir5.0`, `gflags`, `glog`, `Halide`, TC version of `isl`, TC version of `protobuf`, and finally `Tensor Comprehensions`.

For building each package, we need to specify a `build version`, `build number` and
`git hash`. This information is used to build each package.

If you ever want to build new packages corresponding to new git hashes for example
when submodules are bumped, you will need to build conda package for submodule
with the new git hash. For that, all you need to do is change the build version,
number and git hash.

Now, we will go ahead and build the conda package of TC and all of its dependencies. For that, run the command below:

```Shell
cd $TC_DIR/conda_recipes
./conda_build_tc.sh
```

You will see that you have conda packages in `<anaconda_root>/conda-bld/linux-64`

**NOTE**: If some of the conda packaging fails, you can clean them up by following
command:

```Shell
conda build purge
```

Now, let's install the tensor Comprehensions package. By default, the package
will get installed to `<anaconda_root>/lib/python3.6/site-packages`

```Shell
conda install --use-local tensor_comprehensions
```

**NOTE**: `--use-local` means that we are going to install locally built packages.

6. In order to uninstall a conda package, you can run

```Shell
conda uninstall <package_name>
```

If you want to uninstall a python version of package like `tensor_comprehensions`,
run the following command **twice**

```Shell
pip uninstall tensor_comprehensions
```

# A few helpful things
1. When building a private repo, git clone might fail with authentication errors
even when you pass the https_proxy (especially for private repo). In that case,
generate an access token from github https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/
and use that as your password when prompted for authentication.

2. Once the build finished, the packaging and un-packaging step takes long
and might seem like it's stuck but it's not.

3. Look at the description on top in `build.sh` script of conda-recipes for how
conda builds packages and tests the builds.

4. Last, [here](https://conda.io/docs/user-guide/tasks/index.html) is a very useful reference on conda packaging.
