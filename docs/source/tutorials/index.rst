Tensor Comprehensions Tutorials
===============================

**Author**: `Priya Goyal <https://github.com/prigoyal>`_

Tensor Comprehensions (TC) is a framework agnostic library to **automatically**
synthesize high-performance machine learning kernels. TC relies on
`Halide <https://github.com/halide/Halide>`_ IR to express computation and analysis
tools to reason about it. TC uses :code:`polyhedral` compilation techniques to
(semi-)automatically decide how to perform this computation efficiently and produce
fast code. We also provide TC integration with PyTorch and Caffe2.

To automatically tune the performance of the kernel, we provide a genetic algorithms
based **Autotuner** details of which are available at :ref:`pytorch_autotune_layers`.

To read more about Tensor Comprehensions, see our documentation available
at https://facebookresearch.github.io/TensorComprehensions/ and C++ API documentation is
available at https://facebookresearch.github.io/TensorComprehensions/api.

We provide many **python examples** for expressing and running various different ML layers
with TC. The examples can be found `here <https://github.com/facebookresearch/TensorComprehensions/tree/master/test_python/layers>`_.

To read more about Framework integrations, checkout our documentation on `PyTorch integration <https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/getting_started.html>`_
and `Caffe2 integration <https://facebookresearch.github.io/TensorComprehensions/framework/caffe2_integration/integration_with_example.html>`_.

If you want to **integrate your framework** with TC, it's easy and the instructions are
available at https://facebookresearch.github.io/TensorComprehensions/integrating_any_ml_framework.html


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial_tensordot_with_tc
