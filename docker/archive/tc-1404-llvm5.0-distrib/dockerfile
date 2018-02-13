FROM nicolasvasilache/tc-1404

RUN wget -q -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN echo "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-5.0 main" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y libclang-common-5.0-dev llvm-5.0-dev libclang-5.0-dev clang-format-5.0 libclang1-5.0 clang-5.0
