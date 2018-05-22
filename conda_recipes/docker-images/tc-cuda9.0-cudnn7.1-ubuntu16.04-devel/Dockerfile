FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN rm /bin/sh && ln -sf /bin/bash /bin/sh

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends make git ssh realpath wget unzip cmake vim libncurses5-dev
RUN apt-get install -y --no-install-recommends libz-dev libgmp3-dev
RUN apt-get install -y --no-install-recommends automake libtool valgrind subversion
RUN apt-get install -y --no-install-recommends ca-certificates software-properties-common

##################################################################################
# Anaconda3
##################################################################################
WORKDIR /conda-install
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh &&\
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda.sh && \
    chmod +x anaconda.sh && \
    ./anaconda.sh -b -p /opt/conda/anaconda && \
    rm anaconda.sh

RUN . /opt/conda/anaconda/bin/activate && \
    conda create -y --name tc_build python=3.6

RUN . /opt/conda/anaconda/bin/activate && \
    source activate tc_build && \
    conda update -y -n root conda && \
    conda install -y conda-build && \
    conda install -y pyyaml mkl-include pytest && \
    conda install -y -c pytorch pytorch torchvision cuda90 && \
    conda config --add channels nicolasvasilache && \
    conda config --add channels anaconda && \
    conda config --add channels pytorch && \
    conda config --add channels conda-forge

##################################################################################
# CUB
##################################################################################
RUN mkdir -p /opt/cuda/
RUN git clone --recursive https://github.com/NVlabs/cub.git /opt/cuda/cub

##################################################################################
# Sanity checks
##################################################################################
RUN test "$(/opt/conda/anaconda/bin/conda --version | grep 'conda 4.5')" != "" && echo Found conda 4.5.x as expected
RUN test "$(gcc --version | grep 'Ubuntu 5.4.0')" != "" && echo Found gcc-5.4.0 as expected
RUN nvcc --version
RUN test "$(nvcc --version | grep '9.0')" != "" && echo Found nvcc-9.0 as expected

##################################################################################
# Environment
##################################################################################
ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
ENV PATH /usr/local/bin:/usr/local/cuda/bin:$PATH

##################################################################################
# Add Jenkins user for Jenkins CI
#   Note the userid 1014 is hardcoded inside Jenkins atm so we inherit that
##################################################################################
RUN useradd -m -d /var/lib/jenkins -u 1014 jenkins
RUN chown -R  jenkins /opt/conda
RUN chown -R  jenkins /usr/local
