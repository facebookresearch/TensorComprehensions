.. _inference:

Range Inference
===============

In Tensor Comprehensions, loops are implicit and output tensors sizes
are inferred. Concretely, when a user write something like the
following stencil:

.. code::

    A(i) = B(i) + C(i-2)

Tensor Comprehensions must deduce the range of values that :code:`i` iterates
over, and from this we derive the size of the tensor :code:`A`. We also must
infer the size of loops that reduce, for example, the implicit loop
over :code:`k` in a matrix multiply:

.. code::

    def matmul(float(I, K) B, float(K, J) C) -> A {
      A(i, j) +=! B(i, k) * C(k, j)
    }


If this range inference procedure fails to match the user's intent,
then in the first case the output will not be the size they expect,
and in the second case the output *values* will be incorrect, as
either too few or too many terms were included in the summation.

To program productively, one must be able to mentally emulate the
written code on the abstract machine defined by the language
semantics. Regardless of how well-defined it is, if your source code
doesn't do what you think it does, you have a bug. Thus it's critical
that users build a mental model of how we infer ranges, and are able
to do range inference in their heads as they write code. If this
requires more thought than writing explicit loops would, we have
failed.

With this in mind, we eschew heavy-duty mathematical tools, and take a
more straightforward approach for the sake of usability. We infer
ranges only in cases where we feel they are obvious, and require
explicit annotation elsewhere. We intend to fine-tune this boundary in
the future depending on what users find surprising.

The Range Inference Algorithm
-----------------------------

To a good first approximation, we infer rectangular ranges that are as
large as possible without reading out of bounds on the inputs. If
there is more than one way to do this, we throw an error and require
explicit annotation using a :code:`where` clause.

This rule is sufficient to understand the matrix multiply case
above. Maximizing the range of :code:`i` gives it the range of the number of
rows of B. Similarly maximizing the range of :code:`j` gives it the range of
the columns of C. :code:`k` is used twice, so making :code:`k` as large as
possible gives it the lesser of the number of columns of B and the
number of rows of C. These in turn are constrained to be equal by the
type signature of the function (they are both :code:`K`).

Now consider a stencil:

.. code::

    A(i) += B(i + k) * K(k)

There are multiple ways in which we could maximize the ranges of :code:`i`
and :code:`k`. If we first maximize :code:`i`, we might say that it ranges over
the entirety of B. This forces :code:`k` to take on a single value only,
which will not result in the output one expects from a convolution (it
ignores most of the kernel!). If we first maximize :code:`k`, so that it
ranges over the entirety of :code:`K`, then in order to not read out of bounds
the range of :code:`i` must be smaller, and we get an output that is
slightly smaller than the input. This is the behavior we prefer.

In order to make this unambiguous without requiring explicit
annotation in this simple case, range inference proceeds in rounds. We
maintain a set of unresolved variables. Initially it contains all
variables not constrained with an explicit :code:`where` clause. In each
round, we consider all the tensor argument expressions that contain a
single unresolved variable, and construct a boolean expression that
states the access is not out-of-bounds. We then use tools from
Halide (:code:`solve_for_inner_interval`) to find the maximal range for
the variable that satisfies this condition, given the ranges of
variables already resolved in previous rounds. If the variable was
already constrained this round by some other tensor access, we take
the intersection of the inferred ranges.

For the stencil above, in the first round we ignore the expression
:code:`B(i + k)` because it contains multiple unresolved variables. We use
the expression :code:`K(k)` to deduce a range for :code:`k`. In the second
round, :code:`B(i + k)` now contains a single unresolved variable, and we
use the already-inferred range of :code:`k` to deduce a maximal range for
:code:`i`.

Preconditions
-------------

While this procedure produces easy-to-justify ranges for each
variable, it is not sufficient to ensure that no out-of-bounds reads
occur. For example consider:

.. code::

    A(i, j) = B(i) * C(i + j) * D(j)

In the first round, we can resolve both :code:`i` and :code:`j`, using :code:`B(i)` and
:code:`D(j)` respectively. This guarantees that there are no out-of-bounds reads
on :code:`B` and :code:`D`, and defines the size of :code:`A`. The size of :code:`C` could be used to further
restrain the range of :code:`i` or :code:`j` to avoid out-of-bounds reads on :code:`C`. But there
is no second round since the ranges for all variables have been resolved; and
even if there would be, the expression :code:`i + j` would not translate into a
unique rectangular shape for :code:`i` and :code:`j`. So we are left with the requirement
that :code:`C` is large enough to cover all reads. In some cases we can statically
prove this condition (for example if the sizes of :code:`B`, :code:`C`, and :code:`D` are known
constants). In general we emit a compile-time warning.

We intend to add runtime checking of these conditions in the future. However, for some preconditions, it is never desirable to
check them at runtime. Consider a lookup table:

.. code::

    def lut(float(J) B, float(I) C) -> A {
      A(i) = B(C(i))
    }

The range of :code:`i` is constrained by its use in :code:`C`, but we are left with
the additional precondition that the *values* in :code:`C` over that range
never exceed the size of :code:`B`. Checking this at runtime would require an
expensive bounds check in the inner loop. As with the previous case,
we currently just emit a compile-time notification that this unchecked
precondition exists. The user can suppress it and make this code
unconditionally safe by explicitly clamping the expression :code:`C(i)` to be
within the bounds of :code:`B` like so:

.. code::

    def lut(float(J) B, float(I) C) -> A {
      A(i) = B(max(min(C(i), J-1), 0))
    }

Though of course this also has a performance impact.

Worked Examples
---------------

We now describe how range inference reasons about several more complex
examples. If you find a confusing case, feel free to request that we
add it to this section.

Inverted indexing
~~~~~~~~~~~~~~~~~

.. code::

    def reverted(float(I) B) -> A {
      A(i) = B(10 - i)
    }

From the use in :code:`B`, range inference constructs the condition:

.. code::

  0 <= 10 - i < I

This is rearranged by Halide's solver to give the following range:

.. code::

  9 - I <= i < 11

Strided indexing with constant stride
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    def subsample_2(float(I) B) -> A {
      A(i) = B(2*i)
    }

From the use, range inference constructs the condition:

.. code::

  0 <= 2*i < I

This is rearranged into:

.. code::

  0 <= i < (I+1)/2

Note that the division is integer division, which rounds towards
negative infinity in Tensor Comprehensions and Halide.

Strided indexing with offsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    def average_pool_2(float(I) B) -> A {
      A(i) = B(2*i) + B(2*i + 1)
    }

From the uses, range inference constructs the conditions:

.. code::

  0 <= 2*i < I
  0 <= 2*i + 1 < I

These are rearranged into:

.. code::

  0 <= i < (I+1)/2
  0 <= i < I/2

The intersection of these two ranges is:

.. code::

  0 <= i < I/2

One could write the equivalent code:

.. code::

    def average_pool_2(float(I) B) -> A {
      A(i) = B(2*i + k) where k in 0:2
    }

The syntax :code:`where k in lb:ub` is inclusive of the lower bound
and exclusive of the upper bound: it constrains the range of :code:`k` to
the integers between :code:`lb` and :code:`ub-1`, here :code:`k` may only take the values
:code:`0` and :code:`1`.

Since the variable :code:`k` is already resolved by the where clause. From the
use of :code:`i`, range inference constructs the condition:

.. code::

  0 <= 2*i + k < I

We eliminate k by taking the conjunction of the expression over all
values of k, using Halide's ``and_condition_over_domain``. For the
lower bound, :code:`k == 0` dominates. For the upper bound, :code:`k == 1`
dominates.

.. code::

  0 <= 2*i && 2*i + 1 < I

This is equivalent to the intersection of the conditions in the
unrolled case, and so we get the same result:

.. code::

  0 <= i < I/2

Strided indexing with dynamic stride
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    def subsample_2(float(I) B, int(1) S) -> A {
      A(i) = B(S(0)*i)
    }

The value of :code:`S(0)` is not fixed until runtime, so we can't resolve the
size of :code:`A` or the range of the loop. This case throws a compile-time
error. A :code:`where` clause that defines the range of :code:`i` is required.
