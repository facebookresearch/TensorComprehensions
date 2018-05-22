# Prerequisite
Adapt CLANG_PREFIX to your need in the following:
```
export CLANG_PREFIX=...;
export LLVM_SOURCES=/tmp/llvm_sources-tapir5.0;
cd ${LLVM_SOURCES};
git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm && \
    mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build && \
    ${CMAKE_VERSION} -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_INSTALL_OCAMLDOC_HTML_DIR=/tmp -DLLVM_OCAML_INSTALL_PATH=/tmp -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_TARGETS_TO_BUILD=X86 -DCOMPILER_RT_BUILD_CILKTOOLS=OFF -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/ && \
    make -j"$(nproc)" -s && make install -j"$(nproc)" -s
```

# Conda from scratch (one time config)
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ~/conda/miniconda && \
    rm miniconda.sh

conda create -y --name tc_build python=3.6
conda activate tc_build
conda install -y gflags glog protobuf gtest
conda install -y pyyaml mkl-include pytest
conda install -y -c nicolasvasilache llvm-tapir50 halide
conda install pytorch torchvision cuda90 -c pytorch
```

# Activate preinstalled conda in your current terminal
Generally this [conda cheatsheet](https://conda.io/docs/_downloads/conda-cheatsheet.pdf) is useful.
```
. ~/conda/miniconda/bin/activate
conda activate tc_build
```

# Build TC
The following assumes a predefined location for `CUDA_TOOLKIT_ROOT_DIR`, `CUDNN_ROOT_DIR` and `CLANG_PREFIX`, adapt to your needs.
```
CUDA_TOOLKIT_ROOT_DIR=/public/apps/cuda/9.0/ CUDNN_ROOT_DIR=/public/apps/cudnn/v7.0/cuda/ CLANG_PREFIX=$(${CONDA_PREFIX}/bin/llvm-config --prefix) ./build.sh --all
```

# Test locally
```
PYTHONPATH=$(pwd)/install/lib:$(pwd)/install/tc:${PYTHONPATH} python -c 'import torch'
PYTHONPATH=$(pwd)/install/lib:$(pwd)/install/tc:${PYTHONPATH} python -c 'import tensor_comprehensions'
./test_python/.sh
./test.sh
```

# Install
The following commands will install TC into ${CONDA_PREFIX}.
This is a bit final and creates circular dependencies when subsequently building TC.
To test locally without installing, better to `export PYTHONPATH` as above.

If you build with `BUILD_TYPE=Release`:
```
$(which python3) setup.py install
```

Otherwise:
```
$(which python3) setup.py develop
```