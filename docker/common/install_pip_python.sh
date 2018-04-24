#!/bin/bash

set -ex

# Install common dependencies
apt-get update
apt-get install -y --no-install-recommends python3-dev python3-setuptools

# Install pip from source. The python-pip package on Ubuntu Trusty is old
# and upon install numpy doesn't use the binary distribution, and fails to compile it from source.
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
rm get-pip.py

pip install numpy decorator six future pyyaml
# install the devel needed for aten
pip install mkl mkl-devel typing

which python3
python3 --version
python3 -c 'import yaml'

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
