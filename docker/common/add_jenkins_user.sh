#!/bin/bash

set -ex

apt-get update
apt-get install sudo

# Mirror jenkins user in container
echo "jenkins:x:$JENKINS_UID:$JENKINS_GID::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:$JENKINS_GID:" >> /etc/group

# Create $HOME
mkdir -p /var/lib/jenkins
chown jenkins:jenkins /var/lib/jenkins
mkdir -p /var/lib/jenkins/.ccache
chown jenkins:jenkins /var/lib/jenkins/.ccache

# Allow writing to /usr/local (for make install)
chown jenkins:jenkins /usr/local

# Allow writing to conda root env
if [[ "$BUILD_ENVIRONMENT" == *-conda* ]]; then
  echo "Chowning Conda"
  chown -R jenkins:jenkins /opt/conda
fi

# Allow sudo
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins
