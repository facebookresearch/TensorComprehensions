# How to build conda package for Tensor Comprehensions and its dependencies

First be sure to use the heavier anaconda distribution to build pacakges
(and not the miniconda distribution).



## Building conda packages

1. Install conda build

```Shell
conda install conda-build
```

This installs a bunch of dependencies like patchelf, glob, pkginfo, conda-verify.
When prompted, say yes to install these dependencies.

2. While building conda packaged, we might have some third-party dependencies for which we want to install the conda package available on [conda-forge](https://github.com/conda-forge/feedstocks). In order to install packages from conda-forge, we need to add conda-forge to the conda channels by running the following command

We will add the `pytorch` channel since we will get pytorch from there. Channels are prioritized in the order they were added.

```Shell
conda config --add channels conda-forge
conda config --add channels pytorch
```

For example: this is required for the libgmp dependency of ISL and we can get
conda package of libgmp from conda-forge

3. Now create a Python 3.6 conda environment for packaging TC by following below
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

4. Now, we have conda setup, we are ready to build conda package for TC. For this,
since TC has its dependencies that are linked dynamically, we will build conda
packages for them separately. In short, we need to build packages for `clang+llvm-tapir5.0`, `Halide` and finally `Tensor Comprehensions`.

For building each package, we need to specify a `build version`, `build number` and
`git hash`. This information is used to build each package.

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
