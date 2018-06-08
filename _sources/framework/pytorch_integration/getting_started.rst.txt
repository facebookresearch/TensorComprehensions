.. _tc_with_pytorch:

Getting Started
===============

**Author**: `Priya Goyal <https://github.com/prigoyal>`_

We provide integration of Tensor Comprehensions (TC) with PyTorch for both
**training** and **inference** purposes. Using TC with PyTorch, you can express an
operator using Einstein notation and get the fast CUDA code for that layer with
just a few lines of code (examples below).

A **few cases** where TC can be useful:

* if you want to specialize your layer for input tensor sizes like (27, 23, 5, 3) unlike some specific sizes/architectures that have been heavily optimized *or*

* you are interested in fusing layers like group convolution, ReLU, FC *or*

* if you have a different new layer, let's call it :code:`hconv` (a variant of convolution), for which you wish you had an efficient kernel available *or*

* if you have standard operation on different data layouts that you didn't want to use because you couldn't get good kernels for them

TC makes its very trivial to get CUDA code for such cases and many more. By providing
TC integration with PyTorch, we hope to make it further easy for PyTorch users
to express their operations and bridge the gap between research and engineering.


Installation
------------

We provide :code:`conda` package for Tensor Comprehensions (only :code:`linux-64` package)
to quickly get started with using TC. Follow the steps below to install TC :code:`conda` package:

**Step 1:** Setup Anaconda
If you don't have Anaconda setup already, please follow the step :ref:`install_anaconda`.
If you have already installed anaconda3, make sure :code:`conda` bin is in your
:code:`$PATH`. For that run the following command:

.. code-block:: bash

    $ export PATH=$HOME/anaconda3/bin:$PATH

To verify, run the following command:

.. code-block:: bash

      $ which conda

This command should print the path of your :code:`conda` bin. If it doesn't,
please add :code:`conda` in your :code:`$PATH`.

**Step 2:** Conda Install Tensor Comprehensions

Now, go ahead and install Tensor Comprehensions by running following command.

.. code-block:: bash

      $ conda install -y -c pytorch -c tensorcomp tensor_comprehensions

Now, you are ready to start using Tensor Comprehensions with PyTorch. As an example,
let's see a simple example of writing :code:`matmul` layer with TC in PyTorch.

Example
-------

For demonstration purpose, we will pick a simple example for :code:`matmul` layer.

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)

As you can see, with just 3-4 lines of code, you can get a reasonably fast CUDA
code for an operation you want. Read the documentation for finding out more.
