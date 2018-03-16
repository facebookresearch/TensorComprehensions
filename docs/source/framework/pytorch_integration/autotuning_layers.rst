.. _pytorch_autotune_layers:

Autotuning layers
=================

TC provides a genetic search based autotuner that can be used to optimize a TC on
given input tensor sizes.

To autotune a new layer with TC, you need to follow the steps below:

1. Define your TC language and pass it to :code:`tc.define`
2. Create input torch tensors or tuples denoting tensor sizes
3. Run autotuning by calling :code:`my_layer.autotune` and get (or cache) the tuned options.

Autotuner has various parameters that we can adjust to control how much user wants to
autotune. We will go into details of those but let's start a simple example of autotuning.

Example
-------
An example demonstrating each step above is:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
    matmul.autotune(mat1, mat2, **tc.autotuner_settings)
    out = matmul(mat1, mat2)

The documentation of the API call is given below:

.. automodule:: tensor_comprehensions

.. _autotune_api:

my_layer.autotune
-----------------

.. autoclass:: TcUnit
   :members: autotune

.. _autotune_parameters:

Autotuning parameters
---------------------

Autotuner exposes various parameters that can be adjusted to control amount of tuning.
You can read about all the parameters here - :ref:`autotuner_parameters`.

**A brief summary**:

- :code:`threads` - set this to number of CPU cores available.
- :code:`generations` - 5 to 10 generations is a good number.
- :code:`pop_size` - 10 is usually reasonable. You can try 10 to 20.
- :code:`min_launch_total_threads` - If you have really input small sizes, set this to `1`.
- :code:`gpus`: Number of gpus to use for autotuning. Default value is "0". Set this to "0,1" if you wish to use two gpus (for example).

As you autotune, you will see the :code:`best`, :code:`median` and :code:`worst`
kernel timing. You can adopt the following parameter settings as starters for autotuning:

* The **default**, :code:`tc.autotuner_settings` are:

.. code::

     settings = {
         "threads": 32, "generations": 2, "pop_size": 10
     }

* The good defaults that run for a bit longer (in exchange for better performance):

.. code::

     settings = {
         "threads": 32, "generations": 5, "pop_size": 10
     }


* The good defaults that runs for a **LOT** longer:

.. code::

     settings = {
         "threads": 32, "generations": 25, "pop_size": 100
     }


Initial Mapping Options
-----------------------

At the beginning of autotuning, the kernel is mapped to whatever :code:`mapping options`
user passes. If no mapping options are passed by user, then the default :code:`naive`
options will be used. However, since the autotuning evolves from the previous
set of options, it is strongly recommended that user passes the better matching options
to start autotuning. This also ensures higher chances of better performant kernel.
See :ref:`autotune_api` for how to pass options.

An example for how to pass options:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(100, 400).cuda(), torch.randn(400, 500).cuda()
    options = Options("mlp")
    matmul.autotune(mat1, mat2, options=options, **tc.autotuner_settings)
    out = matmul(mat1, mat2)

.. _autotuner_cache_choices:

Caching autotuned options
-------------------------

As user autotunes kernels on given input tensor sizes, user can also cache the options
for later use. In order to cache the options, user needs to pass :code:`cache`
argument to the autotuning call. There are two ways of caching the tuned options:

* :code:`cache=True`: the cache file will look like :code:`/tmp/kernel_name_input_sizes_uuid` string. Example:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(72, 26).cuda(), torch.randn(26, 72).cuda()
    matmul.autotune(mat1, mat2, cache=True)
    out = matmul(mat1, mat2)


* :code:`cache={filepath}`: The options will be cached to the filepath that is passed by the user. Example:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    mat1, mat2 = torch.randn(72, 26).cuda(), torch.randn(26, 72).cuda()
    matmul.autotune(mat1, mat2, cache="matmul_72_26_72.tc")
    out = matmul(mat1, mat2)


Using Cached kernel options
---------------------------

If you have autotuned some kernel on some tensor sizes and you want to use those options
for running the kernel, you can pass the cache to the layer run call.

.. note::

    If you want to run the same kernel many times using the same options, you need
    to pass the cached file only once and the options are loaded the first time
    kernel is run. Once the kernel has run first time, for subsequent runs, TC
    doesn't need to compile the kernel and hence the cache file is not needed for
    subsequent runs.

For example:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    cache_file = "matmul_72_26_72.tc"
    mat1, mat2 = torch.randn(72, 26).cuda(), torch.randn(26, 72).cuda()
    out1 = matmul(mat1, mat2, cache=cache_file)
    # the second time we run the kernel, we skip the compilation since it was
    # already compiled earlier
    out2 = matmul(mat1, mat2)


Using tuple sizes to autotune
-----------------------------

If you want to autotune a kernel on variety of sizes and store the cache for later
use, you don't need to create the input tensor for each sizes you want to tune
kernel for. Rather you can pass the tuples containing the sizes you want to tune.
For example:

.. code-block:: python

    import tensor_comprehensions as tc
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    matmul.autotune((3, 4), (4, 5), cache=True, **tc.small_sizes_autotuner_settings)
    matmul.autotune((100, 400), (400, 500), cache=True, **tc.autotuner_settings)


tc.decode
---------

When you save the autotuner cache, two files are created ending in :code:`.cuda/.options`.
The :code:`.options` file contains the encoded kernel options. If you are curious
about what those options look like, you can decode the options by calling :code:`tc.decode`

The API description is given below:

.. autofunction:: decode

Decoding example
^^^^^^^^^^^^^^^^

Below is example describing the above usage:

.. code-block:: python

    import tensor_comprehensions as tc
    cache = "{}/matmul_3_4_5".format(PATH_PREFIX)
    lang = """
    def matmul(float(M,K) A, float(N,K) B) -> (output) {
        output(m, n) +=! A(m, r_k) * B(n, r_k)
    }
    """
    matmul = tc.define(lang, name="matmul")
    matmul.autotune((3, 4), (4, 5), cache=cache, **tc.small_sizes_autotuner_settings)
    tc.decode(cache + ".options")


This will create a file :code:`cache + ".decoded"` which contains the decoded options.
