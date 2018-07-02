Frequently Asked Questions
==========================

Below are some frequently asked questions.

TC language
-----------

How are temporary variables handled in TC?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since TC doesn't perform any allocations internally, every variable has to be either an input
or output in the TC language. For example:

**Invalid TC:**

The following TC is Invalid because the variable :code:`expSum` is not marked
as either input or output:

.. code::

    def softmax(float(N, D) I) -> (O, maxVal, expDistance) {
        maxVal(n) max=! I(n, d)
        expDistance(n, d) = exp(I(n, d) - maxVal(n))
        expSum(n) +=! expDistance(n, d)
        O(n, d) = expDistance(n, d) / expSum(n)
    }

**Valid TC**

The correct TC would be:

.. code::

    def softmax(float(N, D) I) -> (O, maxVal, expDistance, expSum) {
        maxVal(n) max=! I(n, d)
        expDistance(n, d) = exp(I(n, d) - maxVal(n))
        expSum(n) +=! expDistance(n, d)
        O(n, d) = expDistance(n, d) / expSum(n)
    }

Can I re-use a temporary variable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can as long as the tensor dependencies form a DAG. For example:

**Invalid**

.. code::

    def softmax(float(N, D) I) -> (O, tmp) {
        tmp(n) max=! I(n, d)
        O(n, d) = exp(I(n, d) - tmp(n))
        tmp(n) +=! O(n, d)
        O(n, d) = O(n, d) / tmp(n)
    }

This TC is invalid because :code:`tmp` and :code:`O(n, d)` have a cyclic dependency.

**Valid**

.. code::

    def softmax(float(N, D) I) -> (O, expsum, maxVal) {
        maxVal(n) max=! I(n, d)
        expsum(n) +=! exp(I(n, d) - maxVal(n))
        O(n, d) = exp(I(n, d) - maxVal(n)) / expsum(n)
    }

Autotuner
---------

At the start of a new generation, I see higher kernel runtimes, Why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is expected behavior. When a new generation starts, the best runtime may
bump to e.g. 600us when e.g. 168us was found as the best time from the
previous generation.
This is expected because the autotuner is multithreaded and we don't
enforce a strict order of evaluation: the best configurations
from the previous generation may not be evaluated first in next generation.
Furthermore, the other mutations in the current generation may perform worse
than the last best known configuration. Therefore the initial jump in best runtime at
generation (i+1) is likely to appear temporarily.

I sometimes see fluctuations in the best kernel time, why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The best time reported is the median of the best candidate runtime and the GPU
runtime may be noisy. So a 10-20% variation is expected and normal.

How do I stop autotuning early and save cache?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can send a SIGINT signal (i.e. hit :code:`Ctrl+C`) to stop the autotuning
early. All compilations and evaluations in progress will be completed, but no
new compilation or evaluation will be started.  Therefore, stopping the
autotuner may take some time.
