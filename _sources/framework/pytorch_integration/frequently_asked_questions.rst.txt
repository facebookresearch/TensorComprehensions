Frequently Asked Questions
==========================

Below are some frequently asked questions in TC language and Autotuner.

TC language
-----------

How are temporary variables handled in TC?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since TC doesn't do any allocations itself, every variable has to be either an input
or output in the TC language. For example:

**Invalid TC:**

The following TC is Invalid because the variable :code:`expSum` is neither marked
as input not output.

.. code::

    def softmax(float(N, D) I) -> (O, maxVal, expDistance) {
      maxVal(n) max= I(n, d)
      expDistance(n, d) = exp(I(n, d) - maxVal(n))
      expSum(n) +=! expDistance(n, d)
      O(n, d) = expDistance(n, d) / expSum(n)
    }

**Valid TC**

The correct TC would be:

.. code::

    def softmax(float(N, D) I) -> (O, maxVal, expDistance, expSum) {
      maxVal(n) max= I(n, d)
      expDistance(n, d) = exp(I(n, d) - maxVal(n))
      expSum(n) +=! expDistance(n, d)
      O(n, d) = expDistance(n, d) / expSum(n)
    }

Can I re-use a temporary variable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can as long as the tensor dependencies are strictly DAG. For example:

**Invalid**

.. code::

    def softmax(float(N, D) I) -> (O, tmp) {
        tmp(n) max=! I(n, d)
        O(n, d) = exp(I(n, d) - tmp(n))
        tmp(n) +=! O(n, d)
        O(n, d) = O(n, d) / tmp(n)
    }

This TC is invalid because :code:`tmp` and :code:`O(n, d)` have cyclic dependency.

**Valid**

.. code::

    def softmax(float(N, D) I) -> (O, expsum, maxVal) {
        maxVal(n) max= I(n, d)
        expsum(n) +=! exp(I(n, d) - maxVal(n))
        O(n, d) = exp(I(n, d) - maxVal(n)) / expsum(n)
    }


Autotuner
---------

At the start of new generation, I see high kernel runtime, Why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is not a bug. When the new generation starts and you suddenly see 600us
instead of 168us from last generation, this is completely okay. The reason is that
autotuning is multithreaded and each thread has various kernel configurations/mutations to
evaluate. We don't enforce strict evaluation order i.e. the best configurations
from the previous generation might not be evaluated first in next generation. Further,
the other mutations in generation might be bad, hence the initial jump in time
at generation (i+1) is expected.

I seeded my autotuning but the worse kernel time is still higher. Why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We don't guarantee anything on the worst time but the best time should be
better than what you seeded with. In autotuning, we generate a lot of candidates
and some of the candidates might be very bad leading to the higher worst time.
Also, we don't seed all the candidates, for example we generate 10 candidates
(pop_size) but seed only 5. If you want to start exactly where you left off,
set the pop_size = seeds.

I sometimes see fluctuations in the best kernel time, why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The best time reported is the median of the best candidate runtime and the GPU
runtime is noisy because of synchronization, kernel launch overheads.
So 10-20% variation is expected and normal.

I see some CUDA errors during autotuning, should I worry?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
No you shouldn't worry, autotuning should continue. These errors can happen when
bad kernel options are generated in the candidate pool.

How do I stop autotuning early and save cache?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can do :code:`Ctrl+C` to stop the autotuning early. For how to save cache,
refer to the autotuning documentation.
