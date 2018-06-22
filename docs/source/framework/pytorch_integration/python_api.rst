Python API
==========

.. automodule:: tensor_comprehensions

High-level API
--------------

We provide a high-level API which allows to easily experiment with Tensor
Comprehensions.

.. autofunction:: define

.. autofunction:: make_autograd


Low-level API
-------------

We also provide a low-overhead API which avoids implicit behavior and is
generally useful for benchmarking.

.. autoclass:: Executor
   :members:

      .. automethod:: __call__

.. autofunction:: compile

.. autofunction:: autotune

.. autofunction:: autotune_and_compile

Additionally the :code:`assert_almost_equal` helper function is useful in
performing numerical checks.

.. autofunction:: assert_almost_equal

Python bindings
---------------

Finally we also document a subset of the Python bindings that are commonly
used.

.. automodule:: tensor_comprehensions.tclib

.. autoclass:: MappingOptionsCache
    :members:

.. autoclass:: MappingOptions
    :members:

.. autoclass:: TunerConfig
    :members:
