FROM nicolasvasilache/tc-1404

RUN wget -q -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN echo "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-4.0 main" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y --no-install-recommends llvm-4.0-dev libclang-4.0-dev clang-format-4.0 clang-4.0
