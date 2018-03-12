ARG UBUNTU_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

# CUDA paths - we install cudnn ourselves (only Trusty) so we can match the version with
# PyTorch conda install
ARG CUDA_VERSION
ARG CUDNN_VERSION
ADD ./common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh && rm install_cuda.sh
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV PATH /usr/local/cuda/bin:/usr/local/bin:$PATH

# Install common dependencies
ARG GCC_VERSION
ARG CMAKE_VERSION
ADD ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Anaconda3
ADD ./common/install_conda.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Protobuf 3.4*
ADD ./common/install_protobuf.sh install_protobuf.sh
RUN bash ./install_protobuf.sh && rm install_protobuf.sh

# (optional) Add Jenkins user
ARG JENKINS
ARG JENKINS_UID
ARG JENKINS_GID
ADD ./common/add_jenkins_user.sh add_jenkins_user.sh
RUN if [ -n "${JENKINS}" ]; then bash ./add_jenkins_user.sh ${JENKINS_UID} ${JENKINS_GID}; fi
RUN rm add_jenkins_user.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

# Set all the environment variables now
ENV CC /usr/bin/gcc
ENV CXX /usr/bin/g++
ENV PATH /opt/conda/bin:$PATH
ENV CMAKE_VERSION cmake

# Verify the environment settings
RUN which conda
RUN conda --version
RUN which protoc
RUN protoc --version
RUN which python
RUN python --version
RUN which cmake
RUN cmake --version

CMD ["bash"]
