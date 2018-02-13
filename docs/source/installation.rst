Installation Guide
==================

The following instructions are provided for developers who would like to
experiment with the library.

At the moment, only :code:`Ubuntu 14.04` and :code:`Ubuntu 16.04` configurations are
officially supported. Additionally, we routinely run on a custom CentOS7
installation. If you are interested in running on non-Ubuntu configurations
please reach out and we will do our best to assist you. Contributing back new
docker configurations to provide a stable environment to build from source on
new systems would be highly appreciated. Please read the :code:`docker/README.md` for how
to build new docker images.

Some users might prefer building TC in :code:`non-conda` enviroment and some might prefer building in :code:`conda` enviroment. We provide installation instructions for both environments.

Further, we also provide runtime :code:`docker` images for both :code:`conda` and :code:`non-conda` environment and also an :code:`nvidia-docker` runtime image for TC to have access to GPUs.

You can chose whatever build settings suit your requirements best and follow the instructions to build.
