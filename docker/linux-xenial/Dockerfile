FROM ubuntu:xenial

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends cmake
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y --no-install-recommends libgoogle-glog-dev
RUN apt-get install -y --no-install-recommends libgtest-dev
RUN apt-get install -y --no-install-recommends automake
RUN apt-get install -y --no-install-recommends libgmp3-dev
RUN apt-get install -y --no-install-recommends libtool
RUN apt-get install -y --no-install-recommends ssh
RUN apt-get install -y --no-install-recommends libyaml-dev
RUN apt-get install -y --no-install-recommends realpath
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends valgrind
RUN apt-get install -y --no-install-recommends software-properties-common
RUN apt-get install -y --no-install-recommends subversion
RUN apt-get install -y --no-install-recommends unzip

CMD ["bash"]
