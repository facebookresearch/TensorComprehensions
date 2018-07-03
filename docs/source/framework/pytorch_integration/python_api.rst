Python API
==========

.. automodule:: tensor_comprehensions

High-level API
--------------

We provide a high-level API which allows one to easily experiment with Tensor
Comprehensions.

.. autofunction:: define

.. autofunction:: make_autograd

The :func:`define` function provides an implicit compilation caching
functionality which alleviates the need to implement a caching mechanism at
the user-facing level. The question still remains which :class:`~tclib.MappingOptions`
to use to compile. Since this is still an open problem, we provide support
for user-defined functions to specify this behavior. We require a user
of the :func:`define` function to provide a :class:`~tclib.MappingOptions` generator
function whose sole purpose is to determine the options with which to compile
a particular TC def for particular input sizes.

To facilitate usage we provide the following generators:
		  
.. autofunction:: make_naive_options_factory
		  
.. autofunction:: make_load_from_cache_options_factory
		  
.. autofunction:: make_autotuned_options_factory	 

Custom behavior to select :class:`~tclib.MappingOptions` may be implemented
in addition to the provided defaults. The signature of custom generators must
match:

.. code-block:: python
		
   def some_generator(tc: str, entry_point: str, *inputs: torch.Tensor)
       -> MappingOptions:
           ...

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

Additionally the :func:`assert_almost_equal` helper function is useful in
performing numerical checks.

.. autofunction:: assert_almost_equal

Caching and Configuration
-------------------------

Finally we also document a subset of the helper types for caching and
configuration that are commonly used.

.. automodule:: tensor_comprehensions.tclib

.. autoclass:: MappingOptionsCache
    :members:

.. autoclass:: MappingOptions
    :members:

.. autoclass:: TunerConfig
    :members:
