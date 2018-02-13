ARG BUILD_ID
FROM tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6:${BUILD_ID}

ENV LD_LIBRARY_PATH /usr/lib/:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV PATH /usr/local/bin:/usr/local/cuda/bin:$PATH

WORKDIR /conda-install
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh &&\
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda.sh && \
    chmod +x anaconda.sh && \
    ./anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install \
         numpy\
         decorator\
         six\
         future\
         cmake

WORKDIR /proto-install
RUN wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d /

RUN cd /protobuf-3.4.0 && ./autogen.sh && ./configure && make -j 8
RUN cd /protobuf-3.4.0 && make install && ldconfig

RUN which conda
RUN conda --version
RUN which protoc
RUN protoc --version
RUN which python
RUN python --version
