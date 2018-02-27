Using TC to get fast CUDA code for Tensor Contraction
=====================================================

In this tutorial, we will see how we can start from a random math operation,
express it in TC language and easily get the fast CUDA code for it. We will also
see how to tune the CUDA code to a better performance. All of this is possible with
only 3-4 lines of code. Let's get started.

For this tutorial, you will need to install Tensor Comprehensions binary. You can
get binary builds of Tensor Comprehensions with ``conda install -y -c pytorch -c prigoyal tensor_comprehensions``.

About TensorDot operation
-------------------------

First, we find an operation that we want to generate fast CUDA code for. A lot of
operations like convolution, pooling are standard and have CUDA code easily available, so
rather we are going to pick a new and different operation. How do we find a new operation?

**Sources**: Maybe there is a research paper idea you have like KRU or there is a
numpy operation that is interesting to you and is needed in Machine Learning model.
As per Numpy docs on linear algebra, tensordot seems like an interesting operation
`TensorDot <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html#numpy.tensordot>`_.

**The TensorDot operation**

Assume that we have two tensors, one with dimension :code:`(N, C1, C2, H, W)` and one with dimension
:code:`(N, C2, C3, H, W)`, and we want to do a gemm-type computation on the :code:`C`
dimensions to get an output of shape :code:`(N, C1, C3, H, W)`. Basically, for each
:code:`(N, H, W)` combination, we want to do a reduction from :code:`(C1, C2) * (C2, C3) = (C1, C3)`.

So basically, this operation can be represented as `N x H x W` independent gemms and one could try to
write batched gemm kernel for it. But does that guarantee good performance? What if the
tensor sizes are like this: :code:`N=32, C1=512, C2=8, C3=2, H=28, W=28` i.e.
the value of :code:`C1` is pretty large compared to :code:`C2` / :code:`C3`.

Let's see how we can get the CUDA kernel for such operation and then tune the kernel.

Step 1: Write TC for TensorDot Operation
----------------------------------------

First step is to express the Tensordot operation in TC language. For more information on how to do
so, you can refer to our `Documentation <https://facebookresearch.github.io/TensorComprehensions/index.html>`_
and also find various TC examples `here <https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/layers_database.html>`_.

.. code-block:: python

    # import tc and torch both
    import tensor_comprehensions as tc
    import torch
    # define the operation as TC language
    lang = """
    def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
        O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
    }
    """

Step 2: Register operation with TC
----------------------------------

Now, we will use the TC string and register it with the TC backend by calling :code:`tc.define`.

.. code-block:: python

    # register the lang with TC backend
    tensordot = tc.define(lang, name="tensordot")

.. note::

    The :code:`name` variable should match the name of the def in the :code:`lang`.

Step 3: Create input tensors and run TC
---------------------------------------

Now that TC is registered, we will create the input tensors and run it.

.. code-block:: python

    # create input cuda tensors
    N, C1, C2, C3, H, W = 32, 512, 8, 2, 28, 28
    I0, I1 = torch.randn(N, C1, C2, H, W).cuda(), torch.randn(N, C2, C3, H, W).cuda()
    # choose the options that resemble the operation and run
    out = tensordot(I0, I1, options=tc.Options("conv"))

.. note::

    The :code:`options` can be obtained by autotuning the kernel using Autotuner
    (next step) or you can chose defaults provided. We strongly recommend to run
    the autotuner instead of manual options for better performance. See :ref:`must_pass_options`
    for more information about options.

Step 4: Autotune and get better performing kernel
-------------------------------------------------

So, it was very quick and easy to define the TensorDot operation with TC and get it running.

But how about a better performing kernel?

TC provides a genetic algorithm based autotuner to tune the kernel performance. Let's
autotune the kernel and get a better performance kernel. We will also cache the better
kernel options by setting :code:`cache={filepath}` so that we can use these options
later.

.. code-block:: python

    # autotune the kernel
    best_options = tensordot.autotune(I0, I1, cache="tensordot_32_512_8_2_28.tc")
    # run the kernel with the autotuned options
    out = tensordot(I0, I1, options=best_options)

You can control the amount of autotuning by changing the autotuner parameters. See
:ref:`autotune_parameters` for how to change the settings.

For the setting ``settings={"generations": 25, "pop_size": 100, "number_elites": 10}``, we
get a decent kernel performance as shown in the screenshot below:

.. figure:: ../_static/img/autotuning-py.jpg
    :alt: python-autotuning-tensordot
    :align: center

Early stopping
^^^^^^^^^^^^^^

If your kernel performance is good enough while the autotuning continues, you
can stop autotuning by pressing :code:`Ctrl+C` and the autotuning cache will be saved
and then the autotuning will stop.
