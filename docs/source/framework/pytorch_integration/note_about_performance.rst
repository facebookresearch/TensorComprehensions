Note about Performance/Autotuning
=================================

Reuse Outputs
-------------

TC depends on a tensor library to do the allocations for temporary variables or output tensors.
So everytime TC is run on given input sizes, the output tensor shapes inferred by
TC backend is passed back to the tensor library and the output variables are allocated
by making a :code:`malloc` call. However, this can be expensive and effect performance
significantly. Rather, if your input tensor sizes do not change every time TC is run,
you can keep reusing the output tensor already allocated in previous call. This helps
with better performance. In order to reuse the outputs, you can pass :code:`outputs`
argument when you run the TC. For a concrete example:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,N) A, float(N,K) B) -> (output) {
      output(i, j) +=! A(i, kk) * B(kk, j)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)
    mat3, mat4 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    matmul(mat3, mat4, outputs=out)     # outputs re-used


Static sizes for Autotuning
---------------------------

Tensor Comprehensions have an autotuner that uses evolutionary search to find
faster kernels. TC tries to specialize the kernels to the given input sizes.
If the sizes are parametric, then the search space will become bigger and the performance
is not as good static input sizes. Hence, for now, TC takes static input sizes. More
concretely,

1. you can not tune a kernel for parametric size ranges like batchsize between 16 and 32.

2. you can tune a kernel let's say :code:`avgpool` for input shape :code:`(16, 32, 24, 23)`
by simply calling:

.. code::

    avgpool.autotune((16, 32, 24, 23), **tc.small_size_autotuner_options, cache="16x32x24x23.tc")

In the first release, we have made the sizes as static but we are looking into lifting
this constraint for our future release.
