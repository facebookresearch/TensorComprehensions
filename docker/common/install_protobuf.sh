#!/bin/bash

set -ex

export TEMP_INSTALL=/tmp/proto-install
mkdir -p $TEMP_INSTALL
pushd $TEMP_INSTALL
# Protobuf 3.4*
wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d /
cd /protobuf-3.4.0 && ./autogen.sh && ./configure && make -j 8
cd /protobuf-3.4.0 && make install && ldconfig
popd
rm -rf $TEMP_INSTALL

which protoc
protoc --version

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
