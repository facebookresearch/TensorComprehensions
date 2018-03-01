FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

# System dependencies
RUN apt-get install -y --no-install-recommends curl git ssh realpath wget unzip cmake vim
RUN apt-get install -y --no-install-recommends libyaml-dev libgoogle-glog-dev
RUN apt-get install -y --no-install-recommends libgtest-dev libz-dev libgmp3-dev
RUN apt-get install -y --no-install-recommends automake libtool valgrind subversion
RUN apt-get install -y --no-install-recommends build-essential ca-certificates software-properties-common
RUN cmake --version

# GCC 4.8.*
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc-5 g++-5
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50

ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++

# LLVM+Clang-Tapir5.0
ENV LLVM_SOURCES /tmp/llvm_sources-tapir5.0
WORKDIR $LLVM_SOURCES

ENV CLANG_PREFIX /usr/local/clang+llvm-tapir5.0
ENV CMAKE_VERSION cmake

RUN git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm && \
    mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build && \
    ${CMAKE_VERSION} -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_INSTALL_OCAMLDOC_HTML_DIR=/tmp -DLLVM_OCAML_INSTALL_PATH=/tmp -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_TARGETS_TO_BUILD=X86 -DCOMPILER_RT_BUILD_CILKTOOLS=OFF -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/ && \
    make -j"$(nproc)" -s && make install -j"$(nproc)" -s

RUN rm -Rf ${LLVM_SOURCES}

# CUDA paths
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV PATH /usr/local/bin:/usr/local/cuda/bin:$PATH

# Python3
RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-setuptools
RUN pip3 install --upgrade pip
RUN pip3 install numpy decorator six future setuptools pyyaml

# Protobuf 3.4*
WORKDIR /proto-install
RUN wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d /

RUN cd /protobuf-3.4.0 && ./autogen.sh && ./configure && make -j 8
RUN cd /protobuf-3.4.0 && make install && ldconfig

RUN which python3
RUN python3 --version
RUN python3 -c 'import yaml'
RUN which protoc
RUN protoc --version

# Cleanup the package manager
RUN apt-get autoclean && apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD ["bash"]
