#!/bin/bash

set -ex

export TEMP_INSTALL=/tmp/conda-install
mkdir -p $TEMP_INSTALL

pushd $TEMP_INSTALL

echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda.sh
chmod +x anaconda.sh
/bin/bash ./anaconda.sh -b -p /opt/conda

rm anaconda.sh

popd

export PATH=/opt/conda/bin:$PATH

/opt/conda/bin/conda install numpy decorator six future cmake pyyaml

which conda
conda --version
which python
python --version

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
