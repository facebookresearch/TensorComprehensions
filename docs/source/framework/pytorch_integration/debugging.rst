Debugging
=========

We provide functions to enable the output of debugging information, including
the kernels
that TC generates. If you are curious about what happens when a TC is compiled
and run, you can use these functions to enable logging:

* :code:`dump_cuda`: print the generated cuda code

* :code:`debug_lang`: print the frontend IR and other information

* :code:`debug_halide`: print the Halide IR and other information.

* :code:`debug_tc_mapper`: print polyhedral IR and other information.

* :code:`debug_tuner`: print information logged by the autotuner.

The logging functionality is backed by Google's glog library.
To activate logging to screen, call :code:`tc.logtostderr(True)`.
Otherwise, the logs are printed into uniquely-named files in the default
temporary directory.

Example usage
-------------

.. code-block:: python

    import tensor_comprehensions as tc

    tc.logtostderr(True)
    tc.debug_tc_mapper(True)

    # polyhedral IR will now be printed to stderr

Printing TC generated CUDA code
-------------------------------

Using the functions above, you can also see the CUDA code that the
TC polyhedral mapper generates for a kernel.

.. code-block:: python

    import tensor_comprehensions as tc

    tc.logtostderr(True)
    tc.dump_cuda(True)

    # The generated CUDA code will now be printed to stderr
