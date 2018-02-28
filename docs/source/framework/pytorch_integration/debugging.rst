Debugging
=========

We provide various flags in order to debug various types of issues that might happen.
Also, if you are curious about what happens when a TC is compiled and run, you
can use these flags to enable logging. Various types of flags exposed are:

* :code:`dump_cuda`: print the generated cuda code

* :code:`debug_tc_mapper`: print debug spew for the tc_mapper like cuda code, mapping options etc

* :code:`debug_lang`: dump TC lang information.

* :code:`debug_halide`: dump Halide information.

* :code:`debug_tuner`: print debug spew for the tuner multithreading behavior.


In order to use enable these flags, you need to call :code:`tc.GlobalDebugInit`
and set the proper flags to :code:`true`. All of these flags are :code:`boolean`
flags that take values :code:`true` or :code:`false`.

Example usage
-------------

.. code-block:: python

    import tensor_comprehensions as tc
    import torch

    tc.GlobalDebugInit(["--debug_tc_mapper=true", "--debug_lang=false"])

    matmul = tc.define(tc.database['matmul']['lang'], name='matmul')
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)

In above example, when the TC executes, we will see the TC mapper information.
You can chose to set any number of flags but the :code:`tc.GlobalDebugInit` should
only be called once.

Printing TC generated CUDA code
-------------------------------

Using the flags above, you can also see the CUDA code that TC generates for a
kernel. This flag can be used during autotuning or simply running the kernel
and the generated CUDA code will be printed on command line.

.. code-block:: python

    import tensor_comprehensions as tc
    import torch

    tc.GlobalDebugInit(["--dump_cuda=true"])

    matmul = tc.define(tc.database['matmul']['lang'], name='matmul')
    mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    out = matmul(mat1, mat2)
