Building from Source in Conda Environment
=========================================

If you want to build TC in a conda environment from source including all of its
dependencies, you are at the right installation page. But if you want to build TC
in conda environment using some pre-built conda packages that we ship for some
dependencies, then you need to checkout the next guide.

For native and python TC development, follow the following installation instructions **step-wise**. If you have already done some steps (in-order), you can skip them.

On :code:`Ubuntu 16.04`, for installation from source, follow the following instructions **step-wise**. If you wish to install
on :code:`Ubuntu 14.04`, make the changes where you see 1604.

Step 1: Install some build dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ apt-get update
    $ apt-get install -y libgoogle-glog-dev curl build-essential cmake git automake libgmp3-dev libtool ssh libyaml-dev realpath wget valgrind software-properties-common unzip

.. note::

    You might additionally need to install :code:`cmake3` and :code:`libz-dev`, if these packages are not found by apt-get, you can proceed otherwise install them by running :code:`apt-get install libz-dev`

Step 2: Setup gcc / g++
^^^^^^^^^^^^^^^^^^^^^^^
For building TC, you also need to install a custom clang+llvm. For that, follow the instructions below:

First, check your gcc/g++ and make sure they are in system path, somewhere like :code:`/usr/bin`. Also, check your gcc/g++ version. Currently, TC officially supports :code:`gcc 4.8`, :code:`gcc 4.9` and :code:`gcc 5.*`

.. code-block:: bash

    $ which gcc
    $ which g++

If you don't have correct gcc, g++, follow the instructions below, otherwise skip:

.. code-block:: bash

    $ add-apt-repository ppa:ubuntu-toolchain-r/test
    $ apt-get update
    $ apt-get install -y --no-install-recommends gcc-5 g++-5

*Optionally*, you can configure the gcc versions available on your machine and set up higher priority for the version you want.

.. code-block:: bash

    $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
    $ update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50

Step 3: Install Clang+LLVM
^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, you have the correct gcc, g++, we will move on and install CLANG+LLVM.

.. note::

    Please note that we use a special configuration of LLVM and also build the Tapir project. Therefore, system or separately built LLVM is unlikely to be suitable.

.. code-block:: bash

    $ export CC=$(which gcc)
    $ export CXX=$(which g++)
    $ export CORES=$(nproc)
    $ export LLVM_SOURCES=$HOME/llvm_sources-tapir5.0
    $ export CLANG_PREFIX=$HOME/clang+llvm-tapir5.0  # change this to whatever path you want
    $ export CMAKE_VERSION=cmake
    $ mkdir -p $LLVM_SOURCES && cd $LLVM_SOURCES

Now, clone the repo, build and install LLVM+Tapir:

.. code-block:: bash

    $ git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm
    $ mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build
    $ ${CMAKE_VERSION} -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_TARGETS_TO_BUILD=X86 -DCOMPILER_RT_BUILD_CILKTOOLS=OFF -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/
    $ make -j $CORES -s && make install -j $CORES -s
    $ cd $HOME && rm -rf $LLVM_SOURCES

Step 4: Install Anaconda3
^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to contribute to TC python/C++ API, you need to install TC from source. For this,
:code:`anaconda3` is required. Install :code:`anaconda3` by following the instructions below:

.. code-block:: bash

    $ cd $HOME
    $ wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh
    $ chmod +x anaconda3.sh
    $ ./anaconda3.sh -b -p $HOME/anaconda3
    $ rm anaconda3.sh

Now add :code:`anaconda3` to your :code:`PATH` so that you can use it. For that run the following command:

.. code-block:: bash

    $ export PATH=$HOME/anaconda3/bin:$PATH
    $ conda update conda

Now, verify your conda installation and check the version:

.. code-block:: bash

      $ which conda

This command should print the path of your conda bin. If it doesn't, make sure conda is in your :code:`PATH`.

Now, let's create a conda environment which we will work in and activate that environment:

.. code-block:: bash

    $ conda create -y --name tc-build python=3.6
    $ source activate tc-build

Step 5: Get CUDA and CUDNN
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

Set environment variables:

.. code-block:: bash

    $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
    $ export PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH

Step 6: Get Protobuf3.4
^^^^^^^^^^^^^^^^^^^^^^^

TC officially support protobuf3.4 at the moment. Please follow the below instructions
to install the protobuf.

.. note::

    Anaconda3 also has a protobuf3 available but that might not be compatible with TC. So we recommend following the below instructions to install Protobuf3.4

.. code-block:: bash

    $ mkdir -p /tmp/proto-install && cd /tmp/proto-install
    $ wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d .
    $ cd protobuf-3.4.0 && ./autogen.sh && ./configure && make -j 8 && make install && ldconfig

Now check your proto version by running:

.. code-block:: bash

    $ protoc --version

.. _conda_install_tc:

Step 7: Installing TC
^^^^^^^^^^^^^^^^^^^^^

Now, you need to install TC from source. For installing TC from source, checkout the TensorComprehensions repo and run the following commands:

.. code-block:: bash

    $ cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive
    $ cd TensorComprehensions
    $ git submodule update --init --recursive
    $ conda install -y pyyaml
    $ export TC_DIR=$(pwd)
    $ BUILD_TYPE=Release PYTHON=$(which python3) WITH_CAFFE2=OFF CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all


.. note::
    Please also make sure that you don't have gflags or glog in your system path. Those might conflict with the TC gflags/glog.


Step 8: Verify TC installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ cd $HOME/TensorComprehensions
    $ ./test.sh                   # if you have GPU
    $ ./test_cpu.sh               # if you have only CPU

Build with Basic Caffe2 Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. If you want to install TC with Caffe2 as well, run the following:

.. code-block:: bash

    $ BUILD_TYPE=Release PYTHON=$(which python3) WITH_PYTHON_C2=OFF CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all


.. note::

    This turns off the Caffe2 python build. If you want to turn on the Caffe2 python build, see next step:

2. For installing python binaries as well of Caffe2 with TC:

.. code-block:: bash

    $ BUILD_TYPE=Release PYTHON=$(which python3) WITH_PYTHON_C2=ON CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all

.. note::

    Caffe2 doesn't provide support for pip/conda at the moment and this means in order to use the caffe2 python, you might need to set $PYTHONPATH. Normally, it could be :code:`${TC_DIR}/third-party-install/`

However, please check caffe2 official instructions `here <https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile#test-the-caffe2-installation>`_ . TC doesn't yet provide support for caffe2 python usage.
