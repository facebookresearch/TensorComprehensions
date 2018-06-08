What is Tensor Comprehensions?
==============================

Tensor Comprehensions(TC) is a notation based on generalized Einstein notation
for computing on multi-dimensional arrays. TC greatly simplifies ML framework
implementations by providing a concise and powerful syntax which can be efficiently
translated to high-performance computation kernels, automatically.

Example of using TC with framework
----------------------------------

TC is supported both in Python and C++ and we also provide lightweight integration
with PyTorch/Caffe2 frameworks.

An example of how using TC in PyTorch looks like:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)


For more details on how to use TC with PyTorch, see :ref:`tc_with_pytorch`.

More generally the only requirement to integrate TC into a workflow is to use a
simple tensor library with a few basic functionalities. For more details, see
:ref:`integrating_ml_frameworks`.

.. _tc_einstein_notation:

Tensor Comprehension Notation
-----------------------------
TC borrow three ideas from Einstein notation that make expressions concise:

1. loop index variables are defined implicitly by using them in an expression and their range is aggressively inferred based on what they index,
2. indices that appear on the right of an expression but not on the left are assumed to be reduction dimensions,
3. the evaluation order of points in the iteration space does not affect the output.

Let's start with a simple example is a matrix vector product:

.. code::

    def mv(float(R,C) A, float(C) x) -> (o) {
        o(r) +=! A(r,r_c) * x(r_c)
    }

:code:`A` and :code:`x` are input tensors. :code:`o` is an output tensor.
The statement :code:`o(r) += A(r,r_c) * x(r_c)` introduces two index variables :code:`r` and :code:`r_`.
Their range is inferred by their use indexing :code:`A` and :code:`x`. :code:`r = [0,R)`, :code:`r_c = [0,C)`.
Because :code:`r_c` only appears on the right side,
stores into :code:`o` will reduce over :code:`r_c` with the reduction specified for the loop.
Reductions can occur across multiple variables, but they all share the same kind of associative reduction (e.g. :code:`+=`)
to maintain invariant (3). Note that we prefix reduction indices names with
:code:`r_` for improved readability. :code:`mv` computes the same thing as this C++ loop:

.. code::

    for(int i = 0; i < R; i++) {
      o(i) = 0.0f;
      for(int j = 0; j < C; j++) {
        o(i) += A(i,j) * x(j);
      }
    }

The loop order :code:`[i,j]` here is arbitrarily chosen because the computed value of a TC is always independent of the loop order.

Examples of TC
--------------

We provide a few basic examples.

Simple matrix-vector
^^^^^^^^^^^^^^^^^^^^

.. code::

    def mv(float(R,C) A, float(C) x) -> (o) {
        o(r) +=! A(r,r_c) * x(r_c)
    }

Simple 2-D convolution (no stride, no padding)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    def conv(float(B,IP,H,W) input, float(OP,IP,KH,KW) weight) -> (output) {
        output(b, op, h, w) +=! input(b, r_ip, h + r_kh, w + r_kw) * weight(op, r_ip, r_kh, r_kw)
    }

Simple 2D max pooling
^^^^^^^^^^^^^^^^^^^^^^

Note the similarity with a convolution with a "select"-style kernel:

.. code::

    def maxpool2x2(float(B,C,H,W) input) -> (output) {
        output(b,c,h,w) max=! input(b,c,2*h + r_kw, 2*w + r_kh)
            where r_kw in 0:2, r_kh in 0..2
    }
