FROM nicolasvasilache/tc-1404

RUN export LLVM_SOURCES=/tmp/llvm_sources-4.0; \
      mkdir -p ${LLVM_SOURCES} &&      \
      cd ${LLVM_SOURCES}/ &&     \
      svn co --quiet http://llvm.org/svn/llvm-project/llvm/tags/RELEASE_400/final/ llvm &&  \
      cd ${LLVM_SOURCES}/llvm/tools &&     \
      svn co --quiet http://llvm.org/svn/llvm-project/cfe/tags/RELEASE_400/final/ clang &&    \
      cd ${LLVM_SOURCES}/llvm/projects &&     \
      svn co --quiet http://llvm.org/svn/llvm-project/libcxx/tags/RELEASE_400/final/ libcxx &&    \
      cd ${LLVM_SOURCES}/llvm/projects &&     \
      svn co --quiet http://llvm.org/svn/llvm-project/libcxxabi/tags/RELEASE_400/final/ libcxxabi

RUN export CORES=8; \
    export CLANG_PREFIX=/clang+llvm-4.0; \
    export CMAKE_VERSION=cmake; \
    export LLVM_SOURCES=/tmp/llvm_sources-4.0; \
      mkdir -p ${LLVM_SOURCES}/llvm_build &&      \
      cd ${LLVM_SOURCES}/llvm_build &&    \
      ${CMAKE_VERSION} -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_TARGETS_TO_BUILD=all -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/ &&   \
       make -j $CORES -s &&   \
       make install -j $CORES -s&&   \
       echo SUCCESS || echo FAILURE

RUN rm -Rf ${LLVM_SOURCES}
