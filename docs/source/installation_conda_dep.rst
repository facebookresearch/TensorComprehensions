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
    $ apt-get install -y wget cmake git build-essential unzip curl automake build-essential realpath libtool

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
    $ apt-get install -y --no-install-recommends gcc-4.8 g++-4.8

*Optionally*, you can configure the gcc versions available on your machine and set up higher priority for the version you want.

.. code-block:: bash

    $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
    $ update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

Step 3: Get Protobuf3.4
^^^^^^^^^^^^^^^^^^^^^^^

TC officially support protobuf3.4 at the moment. Please follow the below instructions
to install the protobuf.

.. note::

    Anaconda3 also has a protobuf3 available but that might not be compatible with TC. So we recommend following the below instructions to install Protobuf3.4

.. code-block:: bash

    $ export CC=$(which gcc)
    $ export CXX=$(which g++)
    $ mkdir -p /tmp/proto-install && cd /tmp/proto-install
    $ wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d .
    $ cd protobuf-3.4.0 && ./autogen.sh && ./configure && make -j 8 && make install && ldconfig

Now check your proto version by running:

.. code-block:: bash

    $ protoc --version

Step 4: Install Anaconda3
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

Step 6: Install TC
^^^^^^^^^^^^^^^^^^

We ship conda packages for most of TC dependencies like :code:`clang+llvm`, :code:`glog`,
:code:`gflags`, :code:`protobuf3`, :code:`halide`. We will directly install the
conda packages of TC dependencies and then build TC.

.. code-block:: bash

    $ conda create -y --name tc-build-conda python=3.6
    $ source activate tc-build-conda
    $ conda install -y -c prigoyal tapir50 llvm isl-tc gflags glog
    $ conda install -y -c pytorch pytorch
    $ cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive
    $ cd TensorComprehensions
    $ git submodule update --init --recursive
    $ BUILD_TYPE=Release INSTALL_PREFIX=$CONDA_PREFIX WITH_CAFFE2=OFF CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all

.. note::
    Please also make sure that you don't have gflags or glog in your system path. Those might conflict with the TC gflags/glog.

Step 7: Verify TC installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ cd $HOME/TensorComprehensions
    $ ./test.sh                   # if you have GPU
    $ ./test_cpu.sh               # if you have only CPU
