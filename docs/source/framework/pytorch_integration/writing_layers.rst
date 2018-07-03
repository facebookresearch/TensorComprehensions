Writing TC operations
=====================

.. automodule:: tensor_comprehensions

This document focuses on writing TC operations using the high-level API.
For examples of using the low-level API, see the Python API documentation.

To create a CUDA kernel implementing an operation backed by TC, one should:

1. Create a callable TC object by calling :func:`define`
2. Create input PyTorch Tensors
3. Call the TC object with the input PyTorch Tensors

When running, the backend ensures the TC is compiled and memoized for the
given input tensor sizes (see the documentation for :func:`define` for more details).
Calling the object returned by :func:`define` executes the
corresponding operation and returns a list of outputs.
If the operation has already been compiled, in the following runs, the TC
backend will reuse the memoized compilation result and run the operation
directly.

Example
-------

The following example demonstrates the steps above.
We use the :func:`make_naive_options_factory` builder function to provide
naive :class:`~tclib.MappingOptions`.  Naive options result in poor performance.
At this time, there is no notion of a default :class:`~tclib.MappingOptions`.
Instead one should use the autotuner to perform an evolutionary search
starting from an initial :class:`~tclib.MappingOptions` object and return a better
:class:`~tclib.MappingOptions` object for a given TC function and sizes (more on this
below).

    .. code-block:: python

        import torch
        import tensor_comprehensions as tc
        T = tc.define(
            """
            def add(float(N) A, float(N) B) -> (C) { C(i) = A(i) + B(i) }
            def sub(float(N) A, float(N) B) -> (C) { C(i) = A(i) - B(i) }
            """,
            tc.make_naive_options_factory())
        A, B = torch.randn(100, device='cuda'), torch.randn(100, device='cuda')
        C = T.add(A, B)
        tc.assert_almost_equal(C, torch.add(A, B), A, B)
        D = T.sub(A, B)
        tc.assert_almost_equal(D, (A - B), A, B)


Specifying MappingOptions
-------------------------

There are three ways to construct :class:`~tclib.MappingOptions` when defining a TC:

* **Naive MappingOptions**:

  * :code:`naive`: this is provided to create a basic GPU mapping strategy with
    3-D tiling by 32x32x32, mapping to 256x256 blocks 32x8 threads. This
    should by no means be considered a good baseline but just a point to
    get started using TC. Once a correct TC is written, we recommend either
    using options loaded from a :class:`~tclib.MappingOptionsCache` or resulting from
    a tuning run. One can also modify a :class:`~tclib.MappingOptions` object
    programmatically (see the API documentation).

* **Loading from MappingOptionsCache**: a :class:`~tclib.MappingOptionsCache` provides
  a simple interface to load the best options from a previous tuning run.

* **Autotuning**: A kernel can be autotuned for fixed input tensor sizes.
  Optionally the best performing options can be cached to a file and reused to
  compile and run a TC operation.


Loading from cache
------------------

Loading the best options from a previously serialized :class:`~tclib.MappingOptionsCache`
can be achieved by making a factory function with
:func:`make_load_from_cache_options_factory` and passing it as an argument to the
:func:`define` function:

    .. code-block:: python

        group_normalization="""..."""
        N, G, D, H, W = 32, 32, 4, 56, 56
        T = tc.define(
            group_normalization,
            tc.make_load_from_cache_options_factory('some_file_path'))
        I, gamma, beta = (
            torch.randn(N, G, D, H, W, device='cuda'),
            torch.randn(G, D, device='cuda'),
            torch.randn(G, D, device='cuda'))
        Sum, SumSq, O = T.group_normalization(I, gamma, beta)

One can also use the low-level :class:`~tclib.MappingOptionsCache`.

Autotuning
----------

Tuning can be achieved by making a factory function with
:func:`make_autotuned_options_factory` and passing it as an argument to the
:func:`define` function.

    .. code-block:: python

        group_normalization="""..."""
        N, G, D, H, W = 32, 32, 4, 56, 56
        T = tc.define(
            group_normalization,
            tc.make_autotuned_options_factory(
                starting_options='naive',
                tuner_config=tuner_config))
        I, gamma, beta = (
            torch.randn(N, G, D, H, W, device='cuda'),
            torch.randn(G, D, device='cuda'),
            torch.randn(G, D, device='cuda'))
        Sum, SumSq, O = T.group_normalization(I, gamma, beta)

    .. note::

       A tuning run can be aborted by sending the SIGINT signal (Ctrl+C). In
       that case, the compilation and evaluation jobs currently in flight will
       be flushed, but no new compilation job will be created. Once the jobs in
       flight are flushed, saving to cache occurs (if requested) and the best
       :class:`~tclib.MappingOptions` found so far will be returned.

Tuning behavior can be modified by defining the TC with an optional
:class:`~tclib.TunerConfig` parameter constructed as such:
:code:`tuner_config=tc.TunerConfig().threads(5).generations(3).pop_size(5)`.

    .. note::

       By providing a fixed filename and calling short tuning runs over
       multiple executions with load_from_cache=True and store_to_cache=True,
       one can effectively reinforce the tuning process over time without
       paying a longer startup cost.

Fixed TC, varying input sizes
-----------------------------

A TC definition can be reused but will trigger recompilation for different size
combinations. While we recommend tuning independently for each TC and input size
variation, the best options found for a particular TC and input size
combination may transfer well to another input size (especially if
sizes are close and the kernels exhibit the same type of bottlenecs;
i.e. memory-bound, latency-bound, instruction-issue-bound,
compute-bound).

Pseudo-templating
-----------------

The TC mapper requires statically affine tensor indexing functions.
Without getting into deeper details, the dependence analysis process is
significantly simplified and can be represented exactly.
As a consequence, tensor subscripts should avoid multiplications
between an unknown parametric quantity and an index variable.
In practice this may require writing different TC versions for different stride
and kernel sizes. A simple workaround would be for TC language to provide a
templating mechanism.
A much simpler way to achieve the same effect is to dynamically perform string
substitutions based on runtime values by formatting the TC string with python
regular expressions:

    .. code-block:: python

        import re
        import torch
        import tensor_comprehensions as tc
        tc_str="""
        def avgpool(float(B, C, H, W) input) -> (output) {
            output(b, c, h, w) +=! input(b, c, h * <sH> + r_kh, w * <sW> + r_kw) / (<kH> * <kW>)
                where r_kh in 0:<kH>, r_kw in 0:<kW>
        }
        """
        tc_str = re.sub('<sh>', '1', tc_str)
        tc_str = re.sub('<sw>', '1', tc_str)
        tc_str = re.sub('<kH>', '2', tc_str)
        tc_str = re.sub('<kW>', '3', tc_str)
        T = tc.define(tc_str, tc.make_naive_options_factory())
        out = T.avgpool(torch.ones(1, 1, 4, 4, device='cuda')

Built-in Functions
------------------

TC allows using CUDA built-in functions as well when defining the TC operations.
During execution, the CUDA API will be called for those built-in
functions. For example, assume one wants to use :code:`fmax` CUDA function in TC:

    .. code-block:: python

        import torch
        import tensor_comprehensions as tc
        tc_str = """
        def relu(float(B,M) I) -> (O) {
            O(b, m) = fmax(I(b, m), 0)
        }
        """
        T = tc.define(tc_str, tc.make_naive_options_factory())
        O = T.relu(torch.randn(100, 128, device='cuda'))

TC only supports a subset of built-in CUDA functions.
Built-in functions supported in TC are listed in `this file <https://github.com/facebookresearch/TensorComprehensions/blob/master/tc/core/libraries.h#L67>`_.
Documentation
for these functions is available as part of the official `CUDA documentation <http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE>`_.


More examples
-------------
You can find more examples in our `unit tests <https://github.com/facebookresearch/TensorComprehensions/blob/master/python/tests/test_tc.py>`_.
We also provide more elaborate examples on how to `compute argmin <https://github.com/facebookresearch/TensorComprehensions/blob/master/python/examples/min_distance.py#L151>`_ as well as a simple TC + PyTorch `python overhead benchmark <https://github.com/facebookresearch/TensorComprehensions/blob/master/python/benchmarks/python_overhead.py>`_.
