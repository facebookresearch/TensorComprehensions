.. _tc_mapping_options:

Mapping Options
===============

Tensor Comprehensions (:code:`TC`) can be transformed, or *mapped*, into :code:`CUDA` kernels almost automatically. Because there is more than one possible way to execute tensor operations in parallel on modern GPUs, for example, use different :code:`CUDA` :code:`grids` or different relative execution order, :code:`TC` engine requires the user to make a set of choices regarding the mapping process and provide them through the *mapping options*. Given the specific options, the translation process becomes fully automatic.

The mapping options provide a relatively high-level declarative interface to the GPU mapping process. They are not expressed in terms of loops or other control flow constructs, or individual tensors. Instead, they enable or parameterize certain classes of transformations similarly to regular compiler options. In particular, they control the resources allocated to the GPU kernel, the number of threads, the amount of shared memory to use, the amount of computation per thread, etc. These resources affect occupancy and ultimately the performance of the generated kernel. Mapping Options are mostly intended for programmatic use, they can be configured through API calls, saved and loaded from a Protocol Buffer.


How to choose starting mapping options?
---------------------------------------

*Don't.*

We recommend to not set up the mapping options manually unless you understand how TCs map to :code:`CUDA` code and how the latter can be optimized. Use the Autotuner or the operation- and GPU-specific options provided with :code:`TC`, see `Defaults Provided`_.

Options API
-----------

Options can be set up programmatically using the C++ or Python API. Both implement a `fluent interface <https://en.wikipedia.org/wiki/Fluent_interface>`_ through method chaining. Mapping options construction always starts from the *naïve* options that enable some kernel code to be generated but oftentimes provide poor performance. :code:`TC` provide more efficient mapping options for some common deep learning operations, see `Defaults Provided`_. Individual mapping parameters can be modified by calling option-specific functions, for example:

C++

.. code-block:: c++

  #include <tc/core/cuda/cuda_mapping_options.h>

  auto options = MappingOptions::makeNaiveMappingOptions()
      .mapToBlocks(100, 20)
      .mapToThreads(32, 4, 4);

Python

.. code-block:: python

  from tensor_comprehensions.mapping_options import Options

  options = Options("naive")
  options.mapToBlocks([100, 20])
  options.mapToThreads([32, 4, 4])

When an option allows for multiple arguments, Python API accepts a list while C++ API provides variadic-argument overloads along with ``vector``- and ``initializer_list``-based versions.  See `Available options`_ for the full list.

Defaults provided
------------------

:code:`TC` comes with a list of pre-tuned mapping options for some common classes of deep learning operations.  Although these options were tested on recent production GPUs, performance remains *sensitive* both to the available GPU resources (number of SMs, shared memory size) and to the input sizes. We *highly recommend* using the autotuner for cases that require competitive performance.

The mapping options for the following classes of operations are provided as static methods of the ``MappingOptions`` class.

* :code:`makePointwiseMappingOptions()`: Mapping options for point-wise arithmetic operations (e.g. bias).

* :code:`makeMlpMappingOptions()`: Mapping options for multilayer perceptrons (sequences of fully connected layers followed by non-linearity).

* :code:`makeConvolutionMappingOptions()`: Mapping options for convolutional layers.


Available options
-----------------

The following options are currently available:

* :code:`.mapToBlocks(<list of 1..3 positive integers>)`: The configuration of :code:`CUDA` :code:`grid`, i.e. the number of :code:`CUDA` blocks along three dimensions. Must be within the range allowed by :code:`CUDA` (maximum 2^31-1 for the first value and 65535 for the second and third).  Note that :code:`TC` mapper eliminates empty blocks and the actual launch size may be smaller than requested.

* :code:`.mapToThreads(<list of 1..3 positive integers>)`: The configuration of :code:`CUDA` :code:`block`, i.e. the number of :code:`CUDA` threads in each :code:`block` along three dimensions. Must be within the range allowed by :code:`CUDA` (maximum 1024 for the first and second value, 32 for the third, product below 1024). Note that :code:`TC` mapper eliminates empty threads and the actual launch size may be smaller than requested.

* :code:`.tile(<list of positive integers>)`: Perform `loop tiling <https://en.wikipedia.org/wiki/Loop_nest_optimization>`_ on the generated code with the given sizes. Independent of mapping to a :code:`grid` of thread blocks.

* :code:`.useSharedMemory(<boolean>)`: Create :code:`block`-local copies of data in shared memory when this can leverage data reuse or global memory access coalescing.

* :code:`.maxSharedMemory(<positive integer>)`: The amount of shared memory to use, in bytes. If not provided, :code:`TC` will query the active GPU and use all available shared memory.

* :code:`.unroll(<positive integer>)`: Perform `loop unrolling <https://en.wikipedia.org/wiki/Loop_unrolling>`_ on the generated code and produce *at most* the given number of statements.

* :code:`.unrollCopyShared(<boolean>)`: Also unroll the copies to and from shared memory introduced by the :code:`TC` mapper. If :code:`unroll` value is not provided, has no effect.

* :code:`.matchLibraryCalls(<boolean>)`: Replace computation patterns with calls to highly optimized libraries (such as CUB, CUTLASS) when possible.

* :code:`.fixParametersBeforeScheduling(<boolean>)`: Perform automatic loop scheduling taking into account specific tensor sizes. May produce faster kernels but significantly increases compilation time. Note that the *mapping* will be performed for specific tensor sizes anyway.

* :code:`.outerScheduleFusionStrategy(<choice of Max, Preserve3Coincident, Min>)`: Require :code:`TC` to try and execute different :code:`TC` expressions interleaved (:code:`Max`), separately (:code:`Min`) or interleaved as long as sufficient parallelism is exploited (:code:`Preserve3Coincident`) by performing `loop fusion and fission <https://en.wikipedia.org/wiki/Loop_fission_and_fusion>`_. Applies before tiling.

* :code:`.intraTileFusionStrategy(<choice of Max, Preserve3Coincident, Min>)`: Require :code:`TC` to try and execute different :code:`TC` expressions interleaved (:code:`Max`), separately (:code:`Min`) or interleaved as long as sufficient parallelism is exploited (:code:`Preserve3Coincident`) by performing `loop fusion and fission <https://en.wikipedia.org/wiki/Loop_fission_and_fusion>`_. Applies to inner loops created by tiling.

* :code:`.scheduleFusionStrategy(<choice of Max, Preserve3Coincident, Min>)`: Set up :code:`outerScheduleFusionStrategy` and :code:`intraTileFusionStrategy` to the given value.

.. note::

    Other, *experimental* options may be exposed in the API. Unless explained in the documentation, their behavior is *undefined*. They may or may not affect the kernel, and change the outputs. Use them at your own risk.

Impact on Performance
---------------------

There is no general approach to choosing the best mapping options. We provide several recommendations that have proven successful several times in the past.

* First and foremost, explore the mapping options together with a profiling tool that indicates what are the bottlenecks of your kernel. Since :code:`CUDA` kernel performance is mostly affected by the GPU *occupancy*, identify the occupancy limiting factor and change the options that may affect it.

* While dimensions of the :code:`LHS` tensor are typically transformed into loops, some of which may be mapped to :code:`CUDA` blocks and threads, you should not assume any correspondence between these dimensions, generated loops or positions of the mapping options arguments. To get more comfortable with mapping options, analyze how the generated :code:`CUDA` code changes along with an option change.

* The amount of parallelism and computation per thread is controlled by a combination of :code:`grid` and :code:`block` sizes. If the total number of threads (number of blocks times number of threads per :code:`block`) equals the number of :code:`LHS` tensor elements, then each thread computes a single element of that tensor. As different loops are generated for iterating over different tensor dimensions, and these loops end up mapped to GPU threads, consider :code:`grid`/:code:`block` size pairs that correspond to tensor sizes along different dimensions. Using a *factor* of the tensor size as the total number of threads will make each thread compute multiple elements of the tensor. Number of threads that do not evenly divide the tensor size will lead to thread divergence: some threads will do the computation while others will not. While divergence is generally detrimental for performance, you may want to consider multipliers of the warp size (32) as number of threads. Also keep in mind the limitation of the number of threads per :code:`block` (typically 1024). Note that :code:`TC` mapping engine will eliminate any blocks and threads that do not compute anything, e.g., if the total number of threads is greater than the number of :code:`LHS` tensor elements that can be computed independently.

* Different pairs of :code:`grid` and :code:`block` sizes result in the same total number of threads. If there is data reuse, i.e. the *same* elements of the :code:`RHS` tensors are necessary to compute *different* elements of the :code:`LHS` tensor, larger blocks allow the mapper to place more of the reused data into faster shared memory. However, the larger is the :code:`block`, the more shared memory it requires, which may end up limiting the occupancy. You may want to set up the shared memory size to a value smaller than the physically available shared memory size in this case. Eventually, the data reused inside the :code:`block` may stop fitting the shared memory.

* :code:`Tiling` may leverage the caches by making reuse more localized. Elements of the :code:`LHS` tensor in :code:`TC` can be computed independently yet, when not computed in parallel, they are computed in some order. While this order is optimized for maximal parallelism and reuse by an automatic procedure, it only changes the order in which tensor dimensions are processed. One can think of it as an extension to tensors of per-row or per-column matrix traversals. In any case, the entire slice (row, plane, hyper-place) of the :code:`LHS` tensor is computed before the next slice starts. If some :code:`RHS` tensor element is reused for computing :code:`LHS` values in the same column, but the order was chosen to be per rows, this element is likely to be evicted from cache before it is needed again. :code:`Tiling` changes the order in which :code:`LHS` elements are computed by creating smaller *blocks* inside each slice. :code:`Tile` sizes define the number of elements along each dimension in this :code:`block`. This transformation reminds of how iterations are mapped to the :code:`CUDA` :code:`grid` of thread blocks. In fact, mapping to blocks implicitly performs tiling. Contrary to the thread :code:`block` mapping, tiling does not require all elements to be computed independently from each other as long as other validity conditions hold. Note that :code:`TC` engine performs tiling independently of mapping to the :code:`CUDA` :code:`grid`, i.e., the tiled dimensions may or may not be mapped to blocks or threads. Similarly to :code:`block` and :code:`grid` sizes, :code:`tile` sizes that are divisors of the input tensor size are a reasonable choice. Keep them relatively small to benefit from caches.

* Using :code:`shared memory` is profitable in many cases. Even if when there is no reuse, data may be preloaded into a shared memory cache in a more efficient way than it is accessed during computation, in particular using memory coalescing. However, it may limit the amount of parallelism. Copying to shared memory also uses barrier synchronization inside blocks, which may be undesirable for short kernels. Promotion to shared memory may be disabled for cases where global memory access is not the principal bottleneck of the kernel.

* :code:`Unrolling` eliminates control flow by introducing copies of statements. This reduces the number of integer instructions but may *significantly* increase the compilation time.

* :code:`Fusion strategy` controls how different :code:`TC` expressions will be interleaved with each other. Maximal fusion will attempt to "pipeline" the computation of tensor elements whenever it is possible while minimal fusion will try and ensure that all elements of one :code:`LHS` tensor are computed before starting the next one. Fusion often makes reuse more local, but increases requirements to memory resources and, more importantly, may lead to a loss of parallelism. Maximal fusion is sometimes required at the outer level to produce kernels mappable to more than one :code:`block` (or requiring a global synchronization), minimal fusion at the inner level can decrease the resources requirements at the const of additional synchronizations inside the loop.

Possible compiler issues
------------------------

* :code:`Mapping failures`: Some combinations of mapping options are forbidden, for example using more than 1024 threads per :code:`block` or more shared memory than physically available on the device. In these cases, :code:`TC` mapper will throw an exception. In some extreme cases of catastrophic failure, :code:`TC` may abort completely. Please report such cases to us.

* :code:`Long compilation times`: :code:`TC` internally relies on a mathematical optimization problem that may be hard to solve. Mapping options related to scheduling, fusion and unrolling are known to affect compilation time significantly. Large unroll values and some cases of :code:`fixParametersBeforeScheduling` may lead to *minutes* of compilation time for simple kernels. We recommend disabling these options if compilation takes too long or using the autotuner that prunes options resulting in long compilation times.
