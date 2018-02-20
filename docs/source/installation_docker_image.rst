Installing TC from Docker image
===============================

We provide docker runtime images for :code:`conda` and :code:`non-conda` environments both. TC officially supports
running gcc 4.*, CUDA 8, CUDNN 6 and ubuntu14.04 and gcc 5.*, CUDA9, CUDNN6 on ubuntu16.04. You can find all available images
for Tensor Comprehensions at the `dockerhub <https://hub.docker.com/u/tensorcomprehensions/>`_

The conda and non-conda images for each setup are below:

.. note::

    Docker images below require 12Gb to 16Gb of RAM to run. If you need to adjust the
    memory requirements, then please add :code:`--memory=16g` to the docker run command.
    If your memory is not enough, TC will not build and you will see the errors like
    `g++: internal compiler error: Killed (program cc1plus)`


* :code:`Ubuntu 14.04 conda environment`

.. code-block:: bash

    $ docker run -i -t tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6-py3-conda:x86_1

Now, follow the instructions below to build TC:

* **Option1:** If you want to build everything from scratch including TC dependencies, follow :ref:`conda_install_tc`.
* **Option2:** If you want to build TC but using pre-built conda packages, follow :ref:`conda_dep_install_tc`.


* :code:`Ubuntu 14.04 non-conda environment`

.. code-block:: bash

    $ docker run -i -t tensorcomprehensions/linux-trusty-gcc4.8-tapir5.0-cuda8-cudnn6-py3:x86_1

Now, to install TC, follow :ref:`non_conda_install_tc`


* :code:`Ubuntu 16.04 conda environment`

.. code-block:: bash

    $ docker run -i -t tensorcomprehensions/linux-xenial-gcc5-tapir5.0-cuda9-cudnn7-py3-conda:x86_1

We don't ship the conda packages for TC dependencies that are compatible with gcc5 yet, so you have to
build them from source. To install TC, see instructions here: :ref:`conda_install_tc`


* :code:`Ubuntu 16.04 non-conda environment`

.. code-block:: bash

    $ docker run -i -t tensorcomprehensions/linux-xenial-gcc5-tapir5.0-cuda9-cudnn7-py3:x86_1

Now, to install TC, follow :ref:`non_conda_install_tc`


TC runtime image with nvidia-docker
-----------------------------------

We also provide a runtime nvidia-docker image for :code:`Ubuntu 14.04`, :code:`gcc 4.8`, :code:`CUDA 8` and :code:`CUDNN 6`.
Using this image, you can also run gpu tests. To run the image, make sure you
have :code:`nvidia-docker` installed. Then run the image using following command:

* :code:`NVIDIA-Docker Ubuntu 14.04 conda environment`

.. code-block:: bash

    $ nvidia-docker run --rm -i -t tensorcomprehensions/trusty-gcc4.8-py3-conda-cuda:x86_1

Now, follow the instructions below to build TC:

* **Option1:** If you want to build everything from scratch including TC dependencies, follow :ref:`conda_install_tc`.
* **Option2:** If you want to build TC but using pre-built conda packages, follow :ref:`conda_dep_install_tc`.
