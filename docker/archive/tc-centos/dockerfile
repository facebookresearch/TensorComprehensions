FROM nvidia/cuda:8.0-cudnn7-devel-centos7
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1

RUN yum update
RUN yum install -q -y centos-release-scl
RUN yum install -q -y centos-release-scl-rh
RUN yum install -q -y scl-utils-build

RUN yum install -q -y make
RUN yum install -q -y cmake
RUN yum install -q -y git
RUN yum install -q -y automake
RUN yum install -q -y autoconf
RUN yum install -q -y libtool
RUN yum install -q -y openssh-clients
RUN yum install -q -y which
RUN yum install -q -y wget
RUN yum install -q -y gmp-devel
RUN yum install -q -y ncurses-devel
RUN yum install -q -y libyaml-devel
RUN yum install -q -y unzip
RUN yum install -q -y zlib-devel
RUN yum install -q -y bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    mercurial subversion
RUN yum install -q -y libmpc-devel mpfr-devel gmp-devel
RUN yum install -q -y texinfo
RUN yum install -q -y file

RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-llvm
RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-llvm-libs
RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-llvm-devel
RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-llvm-static

RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-clang
RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-clang-libs
RUN yum install -q -y --enablerepo=centos-sclo-rh-testing llvm-toolset-7-clang-devel

#ENV TMP_LIBRARY_PATH $LIBRARY_PATH
#ENV LIBRARY_PATH=
#RUN env
#RUN curl ftp://ftp.gnu.org/pub/gnu/gcc/gcc-4.9.4/gcc-4.9.4.tar.bz2 -O
#RUN tar xfj gcc-4.9.4.tar.bz2 && mv gcc-4.9.4 /tmp/gcc
#RUN source /opt/rh/llvm-toolset-7/enable && cd /tmp/gcc && CC=clang CXX=clang++ ./configure --with-system-zlib --disable-multilib --enable-bootstrap --enable-languages=c,c++ && make -j && make -j install
#ENV LIBRARY_PATH $TMP_LIBRARY_PATH

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib64:$LIBRARY_PATH
ENV PATH /usr/local/bin:$PATH

RUN yum install -q -y openssl-devel
RUN git clone http://github.com/python/cpython -b 3.6
RUN cd cpython && ./configure --enable-shared && make -j && make -j install
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python

RUN pip3 install\
         numpy\
         decorator\
         six\
         future\
         cmake

# Protobuf
RUN wget --quiet https://github.com/google/protobuf/archive/v3.4.1.zip --no-check-certificate -O /proto.zip &&  unzip -qq /proto.zip -d /tmp/ && cd /tmp/protobuf-3.4.1 && ./autogen.sh && ./configure && make -j && make install -j && ldconfig && rm /proto.zip && cd / && rm -rf /tmp/protobuf-3.4.1
RUN pip3 install protobuf


ENV PATH /opt/rh/llvm-toolset-7/root/bin:$PATH

ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
