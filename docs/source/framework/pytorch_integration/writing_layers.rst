Writing TC operations
=====================

To create a CUDA kernel implementing an operation backed by TC, one can:

1. Create a TC object by calling :code:`tc.define`
2. Create input torch tensors
3. Optionally tune the TC and use the best mapping options

When running such a TC on the inputs created in step 2, the backend ensures
the TC is compiled and memoized for the given input tensor sizes.
Calling the :code:`TC` object returned by :code:`tc.define` executes the
corresponding operation and returns a list of outputs.
If the operation has already been compiled, in the following runs, the TC
backend will reuse the memoized compilation result and run the operation
directly.

Example
-------

The following example demonstrates the steps above.
Note that an explicit fallback :code:`MappingOptions` object is passed when
defining a TC; the only user-facing :code:`MappingOptions` object that can be
constructed is a :code:`naive` object by calling :code:`tc.MappingOptions('naive')`.
At this time there is no notion of a default :code:`MappingOptions` object.
Instead one should use the autotuner to perform an evolutionary search
starting from an initial :code:`MappingOptions` object and return a better
:code:`MappingOptions` object for a given TC function and sizes (more on this
below).

    .. note::

       The fallback parameter is optional, however a TC constructed without a
       fallback must be explicitly tuned or compiled beforehand. Trying to
       call a TC that hasn't been compiled or tuned and that was constructed
       without a fallback will result in an error.

    .. code-block:: python

        import tensor_comprehensions as tc
        import torch
        mm = """
        def matmul(float(M, K) A, float(K, N) B) -> (C) {
            C(m, n) +=! A(m, r_k) * B(r_k, n)
        }
        """
        # the `entry_point` should match the definition name in `mm`
        matmul = tc.define(mm, entry_point="matmul", fallback=tc.MappingOptions('naive'))
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        # the following call will trigger compilation and memoization and return a
        # list of output tensors
        out, = matmul(mat1, mat2)
        # a subsequent call to the same TC with the same sizes will not re-trigger
        # compilation and memoization
        out, = matmul(mat1, mat2)
        # optionally, a list of properly-sized output tensors can be passed and
        # the kernel will use them as outputs
        out, = matmul(mat1, mat2, outputs=[out])

Specifying MappingOptions
-----------------------------

There are three ways to construct :code:`MappingOptions` when defining a TC:

* **Naive MappingOptions**:

  * :code:`naive`: this is provided to create a basic mapping strategy with
    3-D tiling by 32x32x32, mapping to 256x256 blocks and 8x32 threads. This
    is should by no means be considered a good baseline but just a point to
    get started using TC. Once a correct TC is written, we recommend either
    using options loaded from a :code:`MappingOptionsCache` or resulting from
    a tuning run.

* **Loading from MappingOptionsCache**: a :code:`MappingOptionsCache` provides
  a simple interface to load the best options from a previous tuning run.

* **Autotuning**: A kernel can be autotuned for fixed input tensor sizes.
  Optionally the best performing options can be cached to a file and reused to
  compile and run a TC operation.


Loading from cache
------------------

To load the best options from a previously saved :code:`MappingOptionsCache`
object, one can reconstruct it explicitly from a filename and load the best
(top-1) options given a TC string, an entry_point and a tuple of input
tensors as such:

    .. code-block:: python

        # Setup code for mm and tensors as above
        cache = tc.MappingOptionsCache(cache_filename)
        best_options, = cache.load(mm, entry_point, (A, B), 1)
        matmul = tc.define(mm, entry_point="matmul", fallback=best_options)
        C, = matmul(A, B)

One may also create a TC without fallback options and call compile explicitly

    .. code-block:: python

        # Setup code for mm and tensors as above
        matmul = tc.define(mm, entry_point="matmul")
        cache = tc.MappingOptionsCache(cache_filename)
        best_options, = cache.load(mm, entry_point, (A, B), 1)
        matmul.compile(best_options, A, B)
        C, = matmul(A, B)

Autotuning
----------

Tuning can be achieved by constructing a TC and calling :code:`tune` on it.
If the optional parameter :code:`cache_filename` is provided, the best options
will be loaded from file via a :code:`MappingOptionsCache`
object and will be used as a starting point. If additionally, the optional
parameter :code:`store_to_cache` is set to True, tuning will append the best
options to the cache. In the absence of a cache filename, tuning will start
from :code:`tc.MappingOptions('naive')`.

    .. code-block:: python

        # Setup code for mm and tensors as above
        matmul = tc.define(mm, entry_point="matmul")
        best_options = matmul.tune(A, B, cache_filename="some_file_name", store_to_cache=True)

    .. note::

       A tuning run can be aborted by sending the SIGINT signal (Ctrl+C). In
       that case, the compilation and evaluation jobs currently in flight will
       be flushed, but no new compilation job will be created. Once the jobs in
       flight are flushed, saving to cache occurs (if requested) and the best
       :code:`tc.MappingOptions` found so far will be returned.

Tuning behavior can be modified by passing an optional
:code:`tuner_config` parameter constructed as such:
:code:`tuner_config = tc.TunerConfig(threads=5, generations=3, pop_size=5)`.
For the list of configurable parameters and their defaults, one can
query :code:`help(tc.TunerConfig)`.

    .. note::

       By providing a fixed filename and calling short tuning runs over
       multiple executions, one can effectively reinforce the tuning process
       over time without paying a longer startup cost.

Fixed TC, varying input sizes
-----------------------------

Given a TC definition that one like to use to run on different combinations
of input sizes, one can define the TC once. Generally, options

.. code-block:: python

    # Setup code for mm and tensors as above
    matmul = tc.define(mm, name="matmul", fallback=best_options)
    mat1, mat2 = torch.randn(300, 400).cuda(), torch.randn(400, 500).cuda()
    out1, = matmul(mat1, mat2)

    # different input sizes
    mat3, mat4 = torch.randn(320, 450).cuda(), torch.randn(450, 300).cuda()
    out2, = matmul(mat3, mat4)

Whenever the TC backend encounters a combination of TC entry point and input
tensor sizes for which no compilation occured previously, compilation will be
triggered and memoized. The same remarks mentioned previously regarding
:code:`fallback`, explicitly calling :code:`matmul.compile` and tuning still
apply.

    .. note::

        While we recommend tuning independently for each TC and input size
        variation, the best options found for a particular TC and input size
        combination may transfer well to another input size (especially if
        sizes are close and the kernels exhibit the same type of bottlenecs;
        i.e. memory-bound, latency-bound, instruction-issue-bound,
        compute-bound).

Multiple TC definitions
-----------------------

If one wants to define all of TCs in one string and later use that string
for running different operations, one can define a :code:`lang` variable that
holds the TC definition for all operations.
Each time one wants to run a different operation, one can make a new TC object
by calling :code:`tc.define` on the :code:`lang` variable, specify the
:code:`entry_point` corresponding to the operation definition and obtain the
kernel implementing that operation:

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
    matmul = tc.define(lang, entry_point="matmul", fallback=best_options)
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out, = matmul(mat1, mat2)

    abs = tc.define(lang, entry_point="abs", fallback=best_options)
    A = torch.randn(3, 4).cuda()
    out, = abs(A)

.. note::


Writing layers with scalars
---------------------------

The TC mapper performs significantly better when provided with statically
affine tensor indexing functions. Without getting into deeper details, the
dependence analysis process is significantly simplified and can be represented
exactly. As a consequence, tensor subscripts should avoid multiplications
between an unknown parametric quantity and an index variable.
In practice this may require writing different TC versions for different stride
and kernel sizes. A simple workaround woud be for TC to provide a templating
mechanism.
A simple way to achieve the same effect is to dynamically perform string
substitutions based on runtime values by formatting the TC string with python
regular expressions:

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
        avgpool = tc.define(LANG, entry_point="avgpool", fallback=...)
        inp = torch.ones(1, 1, 4, 4).cuda()
        out = avgpool(inp)

Built-in Functions
------------------

TC allows using CUDA built-in functions as well when defining the TC operations.
During execution, the CUDA API will be called for those built-in functions. For example,
asusme one wants to use :code:`fmaxf` CUDA function in TC:

    .. code-block:: python

        import tensor_comprehensions as tc
        import torch
        LANG = """
        def relu(float(B,M) I) -> (O1){
            O1(b, m) = fmaxf(I(b, m), 0)
        }
        """
        relu = tc.define(LANG, entry_point="relu", fallback=tc.MappingOptions('naive'))
        inp = torch.randn(100, 128).cuda()
        out = relu(inp)

TC only supports a subset of built-in CUDA functions. Documentation
for these functions is available as part of the official CUDA documentation `here <http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE>`_.

Built-in functions supported in TC comprise:

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
