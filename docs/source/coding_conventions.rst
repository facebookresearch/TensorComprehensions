Coding Conventions
==================

In order to increase readability across Tensor Comprehensions written by
multiple authors and to reduce the amount of surprising behavior, the
following conventions should be adopted when writing TC. Generally in TC, one
should increment nesting by 4 whitespaces at each level and align tensor names
and indices where appropriate to make memory access patterns emerge. Since
these two goals can easily be conflicting, use your best judgement to tradeoff
between the two goals. Such examples are provided below.

Use indices named after parameters
----------------------------------

Use upper-case names for parameters and capital-case names for input/output tensors.
Use lower-case names for indices to match the name of the parameter
corresponding to the dimension upon which they iterate.
In other words, prefer:

.. code::

    def copy2d(float(M, N) I) -> (O) {
        O(m, n) = I(m, n)
    }

to:

.. code::

    def copy2d(float(M, N) I) -> (O) {
        O(i, j) = I(i, j)
    }

Prefix reduction index names with :code:`r_`
--------------------------------------------

By definition, reduction indices are the ones that appear on the RHS of a TC
expression but not on the LHS. On larger expressions it can get challenging to easily
detect the reduction variables by mentally parsing the set of indices on the
RHS and subtracting the set of indices on the LHS from it. To alleviate such
issues, name the reduction variables with a :code:`r_` prefix.
In other words, prefer:

.. code::

    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }

to:

.. code::

    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, k) * B(k, n)
    }

Filter non-rectangular regions with data-dependencies
-----------------------------------------------------

TC semantics are restricted to (hyper-)rectangular iteration spaces.
This is a hard requirement to ensure range inference is non-ambiguous (see inference_).
To simulate non-rectangular iteration spaces, one can use the following:

.. code::

    def matmul(float(M, K) L, float(K, M) U) -> (LU) {
        LU(m1, m2) +=! (r_k >= m1 and r_k =< m2) ? L(m1, r_k) * U(r_k, m2) : 0
    }

However, non-(hyper)-rectangular iteration spaces (e.g. triangular) are
incompatible with range inference and will fail the semantic checks in the TC
compiler:

.. code::

    def matmul(float(M, K) L, float(K, M) U) -> (LU) {
        LU(m1, m2) +=! L(m1, r_k) * U(r_k, m2) where r_k in m1:M, r_k in 0:m2+1
    }

The reader may remark that this is an inefficient way of writing
matrix-multiplication of triangular matrices.
Lowering such operations efficiently from TC is the subject of future work.

Prefix gradient tensors names with :code:`d_`
---------------------------------------------

When implementing backward operations, pass the inputs to the backwards pass
in the same order as the outputs of the forward pass and use the same tensor
name prefixed by :code:`d_`. For instance:

.. code::

     def conv(float(N,C,H,W) I, float(M,C,KH,KW) Wt) -> (O) {
         ...
     }

     def conv_bw(float(N,C,H,W) I, float(M,C,KH,KW) Wt, float(N,M,HO,WO) d_O) -> (d_I) {
         ...
     }

A more complex example
----------------------

The following shows a possible implementation for a more complex forward and
backward example. Notice the proper alignment of indices in the backward pass
and the emergence of an antidiagonal pattern in the reduction accesses:

.. code::

    def matmul(float(M,K) A, float(K,N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    def matmul_bw(float(M,K) A, float(K,N) B, float(M,N) d_C) -> (d_A, d_B){
        d_A(m, k) +=! d_C(  m, r_n) * B(  k, r_n)
        d_B(k, n) +=! d_C(r_m,   n) * A(r_m,   k)
    }

Reasoning on such reduction patterns at the level of TC has already proven
valuable in other circumstances.
