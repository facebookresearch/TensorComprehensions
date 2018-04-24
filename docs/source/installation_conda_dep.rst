Building with conda packaged dependencies in Conda Environment
==============================================================

If you want to build TC in a conda environment using the conda packages of TC dependencies,
you are on the right place. Follow the following installation instructions **step-wise**. If you have already done some steps (in-order), you can skip them.

On :code:`Ubuntu 16.04`, for installation from source, follow the following instructions **step-wise**. If you wish to install
on :code:`Ubuntu 14.04`, make the changes where you see 1604.

Step 1: Install system dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ apt-get update
    $ apt-get install -y wget cmake git build-essential unzip curl automake build-essential realpath libtool software-properties-common

.. note::

    Please check that your system doesn't have any :code:`protobuf` installed via apt-get or other means. This can easily
    conflict with the protobuf version TC uses and can cause build failures. For example, check :code:`/usr/lib/x86_64-linux-gnu` for any protobuf installations
    that might look like :code:`libprotobuf.*`, :code:`libprotobuf-lite.*`, :code:`libprotoc.*`. Please uninstall them via
    :code:`apt-get remove` command.

Step 2: Setup gcc / g++
^^^^^^^^^^^^^^^^^^^^^^^

First, check your gcc/g++ and make sure they are in system path, somewhere like :code:`/usr/bin`. Also, check your gcc/g++ version. Currently, TC officially supports :code:`gcc 4.8` when installing using conda packages.

.. code-block:: bash

    $ which gcc
    $ which g++

If you don't have correct gcc, g++, follow the instructions below, otherwise skip:

.. code-block:: bash

    $ add-apt-repository ppa:ubuntu-toolchain-r/test
    $ apt-get update
    $ apt-get install -y --no-install-recommends libcilkrts5 gcc-4.8 g++-4.8

*Optionally*, you can configure the gcc versions available on your machine and set up higher priority for the version you want.

.. code-block:: bash

    $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
    $ update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

.. _install_anaconda:

Step 3: Install Anaconda3
^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to contribute to TC python/C++ API, you need to install TC from source. For this,
:code:`anaconda3` is required. Install :code:`anaconda3` by following the instructions below:

.. code-block:: bash

    $ cd $HOME
    $ wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh
    $ chmod +x anaconda3.sh
    $ ./anaconda3.sh -b -p $HOME/anaconda3 && rm anaconda3.sh

Now add :code:`anaconda3` to your :code:`PATH` so that you can use it. For that run the following command:

.. code-block:: bash

    $ export PATH=$HOME/anaconda3/bin:$PATH

Now, verify your conda installation and check the version:

.. code-block:: bash

      $ which conda

This command should print the path of your conda bin. If it doesn't, make sure conda is in your :code:`PATH`.

Step 4: Get CUDA and CUDNN
^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to build TC, you also need to have :code:`CUDA` and :code:`CUDNN`. If you already have it
you can just export the :code:`PATH`, :code:`LD_LIBRARY_PATH` (see the end of this step). If you don't have CUDA/CUDNN, then follow the instructions below:

First, install :code:`CUDA` Toolkit v8.0 (skip if you have it):

.. code-block:: bash

    $ CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
    $ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
    $ dpkg -i ${CUDA_REPO_PKG}
    $ rm -f ${CUDA_REPO_PKG}
    $ apt-get update
    $ apt-get -y install cuda

Now, Install cuDNN v6.0 (skip if you have it already):

.. code-block:: bash

    $ CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
    $ wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
    $ tar -xzvf ${CUDNN_TAR_FILE}
    $ cp -P cuda/include/cudnn.h /usr/local/cuda/include
    $ cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    $ chmod a+r /usr/local/cuda/lib64/libcudnn*

.. note::

    Please use :code:`sudo` to run the command that might fail with permission issues. Otherwise, run
    the commands as is.

Set environment variables:

.. code-block:: bash

    $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
    $ export PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH

.. _conda_dep_install_tc:

Step 5: Install TC
^^^^^^^^^^^^^^^^^^

We ship conda packages for most of TC dependencies like :code:`clang+llvm`, :code:`glog`,
:code:`gflags`, :code:`protobuf3`, :code:`halide`. We will directly install the
conda packages of TC dependencies and then build TC.

.. code-block:: bash

    $ conda create -y --name tc-build-conda python=3.6 && source activate tc-build-conda
    $ conda install -y -c tensorcomp llvm-tapir50 gflags glog protobuf
    $ conda install -y -c pytorch pytorch
    $ cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive
    $ cd TensorComprehensions && git submodule update --init --recursive
    $ BUILD_TYPE=Release INSTALL_PREFIX=$CONDA_PREFIX WITH_CAFFE2=OFF CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all

.. note::
    Please also make sure that you don't have :code:`gflags` or :code:`glog` in your system path. Those might conflict with the TC gflags/glog.


Optional: Building TC with PyTorch master
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested in using PyTorch master with TC, there are two options:

- Install PyTorch nightly conda package (these packages are built for PyTorch master every day) OR
- Install PyTorch from source

For the above two options, the :code:`Install TC` instructions would be as below:

**1. Using PyTorch nightly**:

.. code-block:: bash

    $ conda create -y --name tc-build-conda python=3.6 && source activate tc-build-conda
    $ conda install -y -c tensorcomp llvm-tapir50 gflags glog protobuf
    $ conda install -y pyyaml mkl-include
    $ conda install -yc conda-forge pytest
    $ conda install -y pytorch-nightly -c pytorch
    $ cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive
    $ cd TensorComprehensions && git submodule update --init --recursive
    $ BUILD_TYPE=Release INSTALL_PREFIX=$CONDA_PREFIX WITH_CAFFE2=OFF CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all

**1. With PyTorch install from source**:

.. code-block:: bash

    $ conda create -y --name tc-build-conda python=3.6 && source activate tc-build-conda
    $ conda install -y -c tensorcomp llvm-tapir50 gflags glog protobuf
    $ cd $HOME && git clone --recursive https://github.com/pytorch/pytorch && cd pytorch && git submodule update --init --recursive
    $ conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
    $ python setup.py install
    $ cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive
    $ cd TensorComprehensions && git submodule update --init --recursive
    $ BUILD_TYPE=Release INSTALL_PREFIX=$CONDA_PREFIX WITH_CAFFE2=OFF CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all


Step 6: Verify TC installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ cd $HOME/TensorComprehensions
    $ ./test.sh                   # if you have GPU
    $ ./test_cpu.sh               # if you have only CPU


Build with Basic Caffe2 Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install TC with Caffe2 as well, run the following:

.. code-block:: bash

    $ conda install -y -c tensorcomp caffe2
    $ BUILD_TYPE=Release INSTALL_PREFIX=$CONDA_PREFIX CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all

Now, you have the TC bindings with Caffe2 built as well and and you write python examples for TC in caffe2.
