ARG BUILD_ID
FROM tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6:${BUILD_ID}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev
RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-setuptools

ENV LD_LIBRARY_PATH /usr/lib/:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV PATH /usr/local/bin:/usr/local/cuda/bin:$PATH

RUN pip3 install --upgrade pip
RUN pip3 install \
         numpy\
         decorator\
         six\
         future\
         protobuf\
         setuptools\
         pyyaml

WORKDIR /proto-install
RUN wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d /

RUN cd /protobuf-3.4.0 && ./autogen.sh && ./configure && make -j 8
RUN cd /protobuf-3.4.0 && make install && ldconfig

RUN which python3
RUN python3 --version
RUN python3 -c 'import yaml'
RUN which protoc
RUN protoc --version
