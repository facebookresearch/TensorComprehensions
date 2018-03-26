Writing PyTorch layers with TC
==============================

In order to write a new layer with TC, you need to follow the steps below:

1. Define your TC language and pass it to :code:`tc.define`
2. Create input torch tensors
3. Run the layer and get output

In the third step, when the TC is run on give set of inputs, TC backend will first
compile the language on given tensor sizes, runs the layer and returns the output.
If the layer has already been run at least once, in the next runs, TC backend
will skip the compilation and will run the layer directly.

Example
-------

An example demonstrating each step above is:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    MATMUL_LANG = """
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    """
    # the `name` should match the definition name in the `lang`
    matmul = tc.define(MATMUL_LANG, name="matmul")
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)

Below is a complete documentation of each API call:

.. automodule:: tensor_comprehensions

tc.define
---------

.. autofunction:: define

.. autoclass:: TcUnit
   :members: __call__

.. _must_pass_options:

Specifying Mapping Options
--------------------------

TC is transformed into :code:`CUDA` kernel by using the :code:`Options` which
is used to run the layer and hence also determines the performance of the kernel
generated. Therefore, it is important to use good :code:`Options` for running a
kernel. You can read more about mapping options here - :ref:`tc_mapping_options`.

There are two ways to set the :code:`Options`:

* **Autotuning**: You can autotune the kernel the kernel on certain input tensor sizes, cache the options and use them to run the layer. See :ref:`pytorch_autotune_layers` for how to autotune kernels.

* **Default Mapping**: We provide various default options that can be chosen to closely represent the kernel. The defaults provided are:

  * :code:`pointwise`: if kernel resembles a pointwise operation
  * :code:`mlp`: if kernel resembles an Linear layer operation
  * :code:`conv`: if kernel resembles a convolution operation
  * :code:`group_conv`: if kernel resembles a group convolution operation
  * :code:`naive`: if none of the above, then chose naive default

An example for how to pass options:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
    out = matmul(mat1, mat2, options=tc.Options("mlp"))

.. note::

    If the mapping options are not passed by user, the :code:`naive` mapping
    options will be chosen as default and the kernel performance might be very bad.
    Hence, we strongly recommend user to use either of two ways above for specifying
    kernel mapping options.

Reduction Operators
-------------------

Reduction operators may be suffixed with :code:`!` (for example :code:`+=!`) to
indicate that the tensor to which values are accumulated should first be initialized
with the identity of the reduction operator (e.g., :code:`0` for :code:`+`).
Otherwise, values are accumulated directly to the output or temporary tensor passed to the kernel.


Different input sizes for same TC
---------------------------------

If you have a TC definition that would like to use to run on different combinations
of input sizes, you need to define TC once. An example:

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
    out1 = matmul(mat1, mat2)

    # different input sizes
    mat3, mat4 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
    out2 = matmul(mat3, mat4)

Whenever the input tensor sizes change, TC backend will re-compile the definition
with input sizes again. If the input tensor sizes do not change, the compilation
happens only once and then you can keep running the layer.

Multiple TC definitions in language
-----------------------------------

Let's say you want to define all of your TCs in one string and later use that string
for running different operations defined in the string. You an do so easily. You
can define a :code:`lang` variable that holds the TC definition for all your operations.
Every time you want to run a different operation, you can make a :code:`tc.define` call
on the :code:`lang` variable, specify the :code:`name` corresponding to the operation
definition and get the TC layer for it. Below is an example for how to do this:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    def abs(float(M, N) A) -> (O1) {
        O1(m, n) = fabs(A(m, n))
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)

    abs = tc.define(lang, name="abs")
    A = torch.randn(3, 4).cuda()
    out = abs(A)

.. note::

    We are working on better ways to leverage using multiple TC in one language
    nicely. This current behavior will likely change in near future.


Writing layers with scalars
---------------------------

If you have an operation that requires a constant scalar value for bounds inference,
for example, kernel or stride in case of convolution operation, we need to pass
the TC with the substituted scalar value because right now, we don't support using
scalars for bound inference. The substitution can be done in two ways and users can
adopt whatever feels more convenient.

* **Option 1**: Pass a constants dictionary to the :code:`tc.define` call. An example for how to do this easily is below:

.. warning::

    This particular way of using scalar is a stop-gap solution while we work on
    finding better way of handling scalars for bounds inference. This solution
    will likely be changed in ~1 month timespan.

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def avgpool(float(B, C, H, W) input) -> (output) {{
        output(b, c, h, w) +=! input(b, c, h * {sH} + r_kh, w * {sW} + r_kw) / ({kH} * {kW})
            where r_kh in 0:{kH}, r_kw in 0:{kW}
    }}
    """
    avgpool = tc.define(lang, name="avgpool", constants={"sH":1, "sW":1, "kH":2, "kW":2})
    inp = torch.ones(32, 3, 10, 10).cuda()
    out = avgpool(inp)

.. note::

    In python, the formatting of strings requires usage of :code:`{{...}}`. Hence
    the above example uses these brackets. You only need to do this if your TC
    consists of scalars.


* **Option 2**: Format the string using python regex. An example below:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    import re
    LANG="""
    def avgpool(float(B, C, H, W) input) -> (output) {
        output(b, c, h, w) +=! input(b, c, h * <sH> + r_kh, w * <sW> + r_kw) / (<kH> * <kW>)
            where r_kh in 0:<kH>, r_kw in 0:<kW>
    }
    """
    sH, sW, kH, kW = 1, 1, 2, 2
    LANG = re.sub('<sh>', str(sH), LANG)
    LANG = re.sub('<sw>', str(sW), LANG)
    LANG = re.sub('<kH>', str(kH), LANG)
    LANG = re.sub('<kW>', str(kW), LANG)
    avgpool = tc.define(LANG, name="avgpool")
    inp = torch.ones(1, 1, 4, 4).cuda()
    out = avgpool(inp)


Manually injecting external CUDA code
-------------------------------------

If you have an external efficient CUDA code that you want to use rather than
the CUDA code that TC generates, you can inject your code easily. For this,
you need to create a string which has the CUDA code you want to inject and you
need to pass the name of the kernel and the CUDA code string to the :code:`tc.define`
call. For example:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def add(float(N) A, float(N) B) -> (output) {
        output(n) = A(n) + B(n)
    }
    """

    cuda_code = """
    extern "C"{
    __global__ void my_add(float* __restrict__ output, const float* __restrict__ A, const float* __restrict B)
    {
        int t = threadIdx.x;
        output[t] = A[t] + B[t];
    }
    }
    """

    add = tc.define(lang, name="add", inject_kernel="my_add", cuda_code=cuda_code)
    a, b = torch.randn(100).cuda(), torch.randn(100).cuda()
    out = add(a, b, grid=[1, 1, 1], block=[100, 1, 1])

.. note::

    In such cases, please note that TC doesn't modify the injected CUDA kernel. It will
    simply run the kernel injected as is and TC will also not guarantee the performance
    of the kernel. User needs to specify the :code:`grid` and :code:`block` values
    when running the layer and TC will simply use those settings.


Built-in Functions
------------------

TC allows using some CUDA built-in functions as well when defining the TC language.
During the execution, CUDA API will be called for those built-in functions. For example,
let's say we want to use :code:`fmax` CUDA function in our TC language. An example
for how this would be done is below:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    LANG = """
    def relu(float(B,M) I) -> (O1){
      O1(b, m) = fmax(I(b, m), 0)
    }
    """
    relu = tc.define(LANG, name="relu")
    inp = torch.randn(100, 128).cuda()
    out = relu(inp)

TC only supports a subset of built-in CUDA functions. You can find the documentation
for these functions at the official CUDA documentation `here <http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE>`_.
The functions supported in TC are:

:code:`acos`, :code:`acosh`, :code:`asin`, :code:`asinh`, :code:`atan2`, :code:`atan`,
:code:`atanh`, :code:`cbrt`, :code:`ceil`, :code:`copysign`, :code:`cos`, :code:`cosh`,
:code:`cospi`, :code:`cyl_bessel_i0`, :code:`cyl_bessel_i1`, :code:`erfc`, :code:`erfcinv`,
:code:`erfcx`, :code:`erf`, :code:`erfinv`, :code:`exp10`, :code:`exp2`, :code:`exp`,
:code:`expm1`, :code:`fabs`, :code:`fdim`, :code:`fdivide`, :code:`floor`, :code:`fma`,
:code:`fmax`, :code:`fmin`, :code:`fmod`, :code:`hypot`, :code:`j0`, :code:`j1`,
:code:`lgamma`, :code:`log10`, :code:`log1p`, :code:`log2`, :code:`logb`, :code:`log`,
:code:`nextafter`, :code:`normf`, :code:`norm3d`, :code:`norm4d`, :code:`normcdf`,
:code:`normcdfinv`, :code:`pow`, :code:`rcbrt`, :code:`remainder`, :code:`rhypot`,
:code:`rnorm3d`, :code:`rnorm4d`, :code:`round`, :code:`rsqrt`, :code:`sin`,
:code:`sinh`, :code:`sinpi`, :code:`sqrt`, :code:`tan`, :code:`tanh`, :code:`tgamma`,
:code:`trunc`, :code:`y0`, :code:`y1`
