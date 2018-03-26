ML Layers database
==================

We provide a database of about 30 machine learning layers that are used across
various types of neural networks. We hope that by using this database, users will
be able to write their operators with TC even more easily.

If you want to use one of the layers in the database, you can query this database
in your code easily. The database can be accessed by calling :code:`tc.database`.
This database is a dictionary of TC name to the TC definition. Each entry in the
dictionary looks like: :code:`{tc_name: {"lang": language, "grad": grad_language}}`
where :code:`tc_name` is the name of the operation, :code:`lang` is the tc language
describing that operation, :code:`grad` is the TC language describing the gradient
of that operation. The :code:`grad` is optional entry.

An example to do so:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    matmul = tc.define(tc.database['matmul']['lang'], name='matmul')
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)


Pooling Layers
--------------

Average pooling
^^^^^^^^^^^^^^^

.. code::

    def avgpool(float(B, C, H, W) input) -> (output) {{
        output(b, c, h, w) +=! input(b, c, h * {sH} + r_kh, w * {sW} + r_kw) / ({kH} * {kW})
            where r_kh in 0:{kH}, r_kw in 0:{kW}
    }}


Max pooling
^^^^^^^^^^^

.. code::

    def maxpool(float(B, C, H, W) input) -> (output) {{
        output(b, c, h, w) max=! input(b, c, h * {sH} + r_kh, w * {sW} + r_kw)
            where r_kh in 0:{kH}, r_kw in 0:{kW}
    }}

Convolution layers
------------------

Simple Convolution
^^^^^^^^^^^^^^^^^^

.. code::

    def convolution(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(M) B) -> (O) {
        O(n, m, h, w) +=! I(n, r_c, h + r_kh, w + r_kw) * W1(m, r_c, r_kh, r_kw)
        O(n, m, h, w)  =  O(n, m, h, w) + B(m)
    }

Strided Convolution
^^^^^^^^^^^^^^^^^^^

.. code::

    def convolution_strided(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(M) B) -> (O) {{
        O(n, m, h, w) +=! I(n, r_c, {sh} * h + r_kh, {sw} * w + r_kw) * W1(m, r_c, r_kh, r_kw)
        O(n, m, h, w)  = O(n, m, h, w) + B(m)
    }}

Strided Convolution Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    def convolution_grad(float(N, C, H, W) I, float(M, C, KH, KW) W1, float(N, M, H, W) g_O) -> (g_I, g_W1) {{
         g_I(n, c, h, w)   +=! g_O(n, r_m, {sh} *   h - r_kh, {sw} *   w - r_kw) * W1(r_m, c, r_kh, r_kw)
        g_W1(m, c, kh, kw) +=! g_O(n,   m, {sh} * r_h -   kh, {sw} * r_w -   kw) *  I(r_n, c,  r_h,  r_w)
    }}

Simple Group Convolution
^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    def group_convolution(float(N, G, C, H, W) I, float(G, F, C, KH, KW) W1, float(G, F) B) -> (O) {
        O(n, g, f, h, w) +=! I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
        O(n, g, f, h, w)  =  O(n, g, f, h, w) + B(g, f)
    }

Group Convolution Strided
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    def group_convolution_strided(float(N, G, C, H, W) I, float(G, F, C, KH, KW) W1, float(G, F) B) -> (O) {{
        O(n, g, f, h, w) +=! I(n, g, r_c, {sh} * h + r_kh, {sw} * w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
        O(n, g, f, h, w) = O(n, g, f, h, w) + B(g, f)
    }}

Linear layers
-------------

Fully Connected layer
^^^^^^^^^^^^^^^^^^^^^

.. code::

    def fully_connected(float(B, M) I, float(N, M) W1, float(N) B1) -> (O1) {
        O1(b, n) +=! I(b, r_m) * W1(n, r_m)
        O1(b, n) = O1(b, n) + B1(n)
    }

Non-Linear layers
-----------------

ReLU
^^^^

.. code::

    def relu(float(B, M) I) -> (O1){
        O1(b, m) = fmax(I(b, m), 0)
    }

Sigmoid
^^^^^^^

.. code::

    def sigmoid(float(N, C, H, W) I) -> (O) {
        O(n, c, h, w) = 1 / (1 + exp(-I(n, c, h, w)))
    }

Softmax
^^^^^^^

.. code::

    def softmax(float(N, D) I) -> (O, maxVal, expDistance, expSum) {
        maxVal(n) max=! I(n, d)
        expDistance(n, d) = exp(I(n, d) - maxVal(n))
        expSum(n) +=! expDistance(n, d)
        O(n, d) = expDistance(n, d) / expSum(n)
    }

Tanh
^^^^

.. code::

    def Tanh(float(M) I) -> (O) {
        O(m) = tanh(I(m))
    }

Cosine
^^^^^^

.. code::

    def cosine(float(M) I) -> (O) {
        O(i) = cos(I(i))
    }

Math Operations
---------------

TensorDot
^^^^^^^^^

.. code::

    def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
        O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
    }

Matmul
^^^^^^

.. code::

    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }

Matmul Gradient
^^^^^^^^^^^^^^^

.. code::

    def matmul_bw(float(M,K) A, float(K,N) B, float(M,N) g_C) -> (g_A, g_B){
        g_A(m, k) +=! g_C(  m, r_n) * B(  k, r_n)
        g_B(k, n) +=! g_C(r_m,   n) * A(r_m,   k)
    }

Batch Matmul
^^^^^^^^^^^^

.. code::

    def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
        Z(b, n, k) +=! X(b, n, r_m) * Y(b, r_m, k)
    }

Absolute
^^^^^^^^

.. code::

    def abs(float(M, N) A) -> (O1) {
        O1(m, n) = fabs(A(m, n))
    }

Add
^^^

.. code::

    def add(float(N) A, float(N) B) -> (output) {
        output(n) = A(n) + B(n)
    }

Tensor Operations
-----------------

Indexing
^^^^^^^^

.. code::

    def indexing(float(H, W) input, int32(L) index) -> (output) {{
        output(l, w) = input(index(l), w)
    }}

Lookup Table
^^^^^^^^^^^^

.. code::

    def lut(float(B, R) LUT, int32(B, N) I) -> (O) {
        O(b, n) +=! LUT(I(b, n), r_r)
    }

Transpose
^^^^^^^^^

.. code::

    def transpose(float(N, C, H, W) I) -> (O) {
        O(c, n, w, h) = I(n, c, h, w)
    }

Concat
^^^^^^

.. code::

    def concat(float(M, N) A, float(M, N) B) -> (O1) {
        O1(n, i, m) = i == 0 ? A(m, n) : B(m, n) where i in 0:2
    }

Cast
^^^^

.. code::

    def cast(float(M,N) A) -> (int32(M,N) O1) {{
        O1(m, n) = int32(A(m, n) + {constant})
    }}

Copy
^^^^

.. code::

    def copy(float(M, N) I) -> (O) {
        O(m, n) = I(m, n)
    }

Scale
^^^^^

.. code::

    def scale(float(M, N) I) -> (O) {{
        O(m, n) = I(m, n) * {s}
    }}

Fused layers
------------

FCRelu
^^^^^^

.. code::

    def fcrelu(float(B,M) I, float(N,M) W1, float(N) B1) -> (O1){
        O1(b, n) +=! I(b, r_m) * W1(n, r_m)
        O1(b, n)  = O1(b,   n) + B1(n)
        O1(b, n)  = fmax(O1(b, n), 0)
    }

Small MobileNet
^^^^^^^^^^^^^^^

.. code::

    def small_mobilenet(float(C1, H, W) I, float(C1, KH1, KW1) W1, float(C1) B1, float(C2, C1) W2, float(C2) B2)
    -> (O1, O2) {
        O1(c1, h, w) +=! I(c1, h + r_kh, w + r_kw) * W1(c1, r_kh, r_kw)
        O1(c1, h, w)  = O1(c1,        h,        w) + B1(c1)
        O1(c1, h, w)  = fmax(O1(c1, h, w), 0)

        O2(c2, h, w) +=! O1(r_c1, h, w) * W2(c2, r_c1)
        O2(c2, h, w)  =  O2(  c2, h, w) + B2(c2)
        O2(c2, h, w)  = fmax(O2(c2, h, w), 0)
    }

Normalization layers
--------------------

Batch Normalization
^^^^^^^^^^^^^^^^^^^

.. code::

    def batchnorm(float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn)
    -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut)
    {{
        mean(c) +=! I(nn, c, hh, ww)
        mean(c)  = mean(c) / (N * H * W)
        rMeanOut(c) = (1 - {momentum}) * rMeanIn(c) + {momentum} * mean(c)
        centered(n, c, h, w) =        I(n, c, h, w) - rMeanOut(c)
        variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
        expectedVariance(c) +=! (variance(n, c, h, w) + {eps}) / (N * H * W)
        rVarOut(c) = rsqrt((1 - {momentum}) * rVarIn(c) + {momentum} * expectedVariance(c))
        O(n, c, h, w) = centered(n, c, h, w) * rVarOut(c)
        normalizedOut(n, c, h, w) = O(n, c, h, w)
    }}

Layer Normalization
^^^^^^^^^^^^^^^^^^^

.. code::

    def layernorm(float(T, B, C) I) -> (O, mean, centered, var) {{
              mean(t, b) +=! I(t, b, c) / C
        centered(t, b, c) =  I(t, b, c) - mean(t, b)
        var(t, b) +=! centered(t, b, c) * centered(t, b, c)
        var(t, b)  =  (var(t, b) + {eps}) / C
        O(t, b, c) =  centered(t, b, c) / rsqrt(var(t, b))
    }}

Distance Functions
------------------

Cosine Similarity
^^^^^^^^^^^^^^^^^

.. code::

    def cosine_similarity(float(M, N) I1, float(M, N) I2) -> (O, sumI1, sumI2) {{
        sumI1(m) +=!  I1(m, n) * I1(m, n)
        sumI2(m) +=!  I2(m, n) * I2(m, n)
            O(m) +=! (I1(m, n) * I2(m, n)) / fmax(rsqrt(sumI1(m)) * sqrt(sumI2(m)), {eps})
    }}

What operations can not be expressed
------------------------------------
* **Reshape**: Reshaping tensors inside the language.
* **Dropout**: RNGs are not supported inside TC language, because TC doesn't do internal allocations.
* **Strided tensors**: Input tensors have to be contiguous. If they are not contiguous, they are made contiguous before passing to the TC backend.
* **RNNs**: TC language doesn't have loops yet. You can write them unrolled if you want.
