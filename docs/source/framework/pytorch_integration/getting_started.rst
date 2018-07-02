.. _tc_with_pytorch:

Getting Started
===============

We provide integration of Tensor Comprehensions (TC) with PyTorch for both
**training** and **inference** purposes. Using TC with PyTorch, you can express an
operator using Einstein notation and get a fast CUDA implementation for that
layer with just a few lines of code (examples below).

Here are a few cases where TC can be useful:

* specialize your layer for uncommon tensor sizes and get better performance
  than libraries, *or*

* experiment with layer fusion like group convolution, ReLU, FC, *or*

* synthesize new layers and get an efficient kernel automatically, *or*

* synthesize layers for tensors with unconventional memory layouts

TC makes it easy to synthesize CUDA kernels for such cases and more. By providing
TC integration with PyTorch, we hope to make it easy for PyTorch users
to express their operations and bridge the gap between research and engineering.

Installation
------------

See instructions here: :ref:`installation_guide`.

Example
-------

For demonstration purposes, we illustrate a simple :code:`matmul` operation
backed by TC.

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    mm = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    TC = tc.define(mm, tc.make_naive_options_factory())
    A, B = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    C = TC.matmul(A, B)

With a few lines of code, you can get a functional CUDA implementation for an
operation expressed in TC. Note, however, that this simplest example is not
expected to be fast. Read the documentation to find out more.
