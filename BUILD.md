# Important notice
***In order to uniformize and simplify the build system we had to make choices. TC is currently only officially supported on Ubuntu 16.04 with gcc 5.4.0.***
Other configurations may work too but are not yet officially supported.
For more information about setting up the config that we use to build the conda dependencies see the following [Dockerfile](https://github.com/facebookresearch/TensorComprehensions/blob/master/conda_recipes/docker-images/tc-cuda9.0-cudnn7.1-ubuntu16.04-devel/Dockerfile).

Our main goal with this decision is to make the build procedure extremely simple, both reproducible internally and extensible to new targets in the future.
In particular the gcc-4 / gcc-5 ABI switch is not something we want to concern ourselves with at this point, we go for gcc-5.4.0.

# Conda from scratch (first time configuration)
Choose and set an INSTALLATION_PATH then run the following:

```
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O anaconda.sh && \
    chmod +x anaconda.sh && \
    ./anaconda.sh -b -p ${INSTALLATION_PATH} && \
    rm anaconda.sh

. ${INSTALLATION_PATH}/bin/activate
conda update -y -n base conda
```

Create a new environment in which TC will be built and install core dependencies:
```
conda create -y --name tc_build python=3.6
conda activate tc_build
conda install -y pyyaml mkl-include pytest
conda install -y -c nicolasvasilache llvm-tapir50 halide
```

Then install the PyTorch version that corresponds to your system binaries (e.g. for PyTorch with cuda 9.0):
```
conda install -y -c pytorch pytorch torchvision cuda90
conda remove -y cudatoolkit --force
```

***Note*** As of PyTorch 0.4, PyTorch links cuda libraries dynamically and it
pulls cudatoolkit. However cudatoolkit can never replace a system installation
because it cannot package libcuda.so (which comes with the driver, not the toolkit).
As a consequence cudatoolkit only contains redundant libraries and we remove it
explicitly. In a near future, the unified PyTorch + Caffe2 build system will link
everything statically and stop pulling the cudatoolkit dependency.

# Activate preinstalled conda in your current terminal

Once the first time configuration above has been completed, one should activate conda in
each new terminal window explicitly (it is discouraged to add this to your `.bashrc` or
equivalent)
```
. ${INSTALLATION_PATH}/bin/activate
conda activate tc_build
```

# Build TC with dependencies supplied by conda
```
CLANG_PREFIX=$(${CONDA_PREFIX}/bin/llvm-config --prefix) ./build.sh
```
You may need to pass the environment variable `CUDA_TOOLKIT_ROOT_DIR` pointing
to your cuda installation (this is required for `FindCUDA.cmake` to find your cuda installation
and can be omitted on most systems). When required, passing `CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda`
is generally sufficient.

# Test locally
Run C++ tests:
```
./test.sh
```

Install the TC Python package locally to `/tmp` for smoke checking:
```
python setup.py install --prefix=/tmp
export PYTHONPATH=${PYTHONPATH}:$(find /tmp/lib -name site-packages)
```

Run Python smoke checks:
```
python -c 'import torch'
python -c 'import tensor_comprehensions'
```

Run Python tests:
```
./test_python/run_test.sh
```

At this point, if things work as expected you can venture installing as follows
(always a good idea to record installed files for easy removal):
```
python setup.py install --record tc_files.txt
```

# Advanced / development mode installation

## Optional dependencies
Optionally if you want to use Caffe2 (this is necessary for building the C++ benchmarks
since Caffe2 is our baseline):
```
conda install -y -c conda-forge eigen
conda install -y -c nicolasvasilache caffe2
```

## Cudnn version 7.1 in Caffe2 / dev mode
***Note*** As of PyTorch 0.4, we need to package our own Caffe2. The curent PyTorch + Caffe2
build system links cudnn dynamically. The version of cudnn that is linked dynamically
is imposed on us by the docker image supported by NVIDIA
[Dockerfile](conda_recipes/docker-images/tc-cuda9.0-cudnn7.1-ubuntu16.04-devel/Dockerfile).
For now this cudnn version is cudnn 7.1.

If for some reason, one cannot install cudnn 7.1 system-wide, one may resort to the
following:
```
conda install -c anaconda cudnn
conda remove -y cudatoolkit --force
```

***Note*** cudnn pulls a cudatoolkit dependencey but this can never replace a system
installation because it cannot package libcuda.so (which comes with the driver,
not the toolkit).
As a consequence cudatoolkit only contains redundant libraries and we remove it
explicitly. In a near future, the unified PyTorch + Caffe2 build system will link
everything statically and we will not need to worry about cudnn anymore.
