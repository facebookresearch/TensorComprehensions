ARG BUILD_ID
FROM tensorcomprehensions/linux-trusty:${BUILD_ID}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends vim

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get install -y --no-install-recommends libcilkrts5
RUN apt-get install -y --no-install-recommends gcc-4.8 g++-4.8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++

ENV LLVM_SOURCES /tmp/llvm_sources-tapir5.0
WORKDIR $LLVM_SOURCES

ENV CLANG_PREFIX /usr/local/clang+llvm-tapir5.0
ENV CMAKE_VERSION cmake

RUN git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm && \
    mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build && \
    ${CMAKE_VERSION} -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_INSTALL_OCAMLDOC_HTML_DIR=/tmp -DLLVM_OCAML_INSTALL_PATH=/tmp -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_TARGETS_TO_BUILD=X86 -DCOMPILER_RT_BUILD_CILKTOOLS=OFF -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/ && \
    make -j"$(nproc)" -s && make install -j"$(nproc)" -s

RUN rm -rf ${LLVM_SOURCES}

CMD ["bash"]
