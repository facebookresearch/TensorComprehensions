Positioning of TC in ML Software stacks
=======================================

TC is a concise notation which can easily be used to write ML layers.
The positioning of TC in the ML ecosystem aims at achieving the following:

1. Easy to integrate with any ML framework and any tensor library.
2. Carry the minimal amount of information to write layers concisely.
3. Be automatically translated to HPC implementations on various runtimes / HW.
4. Carry the minimal amount of information to support automatic transformation to HPC implementations.
5. Be usable by our intended target audience.
6. Be non-ambiguous and non-surprising.

For now, we detail the first two points below:

Implications of ML Framework Integration
----------------------------------------

The simple fact that TC wants to be ML-framework agnostic has deeper
implications that we will look into lifting in the future.

One TC function one kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^

A single TC function must correspond to exactly one synthesized HPC
kernel. If TC doesn't guarantee that invariant, it will have to handle the
integration of kernel launches and their proper synchronization in each
framework it targets. This is off the table for now because it is
counter-productive; the moment one operates at this level of control, one is
effectively competing with frameworks and wants to write one's own.

No Variable Allocations
^^^^^^^^^^^^^^^^^^^^^^^

TC cannot allocate memory and handle :code:`host <-> accelerator` copies. This is
for the same reason stated above. As a consequence there is never a notion of TC variable definition, local
scope in a TC, allocations etc (i.e. all the basic stuff you expect from a
Programming Language). As a consequence TC is not a Programming Language but a concise
notation. For now, it should not try to be a Programming Language.

As a result of this, everything in TC should be either an input or output. For example:
consider the TC definition below:

.. code::

    def softmax(float(N, D) I) -> (O, expsum) {
      expsum(n) +=! exp(I(n, d))
      O(n, d) = exp(I(n, d)) / expsum(n)
    }

In this TC, :code:`expsum` is a temporary variable that needs to be computed but
since TC doesn't do allocations itself, we set it as another output. User can chose
to ignore this output. We will work on enhancing this and deal with temporary
allocations better in future.

Graph Level
^^^^^^^^^^^

For allocations, variable definitions, unique names, SSA etc. one probably wants
to work at the :code:`NNVM / XLA` level which is not what TC wants to work at for now.
Probably if one wants function calls (i.e. TC :code:`def` functions calling other
TC def functions) then the TC notation is probably not where it should
happen. TC calling built-in functions with side effects is fine though.

Minimal information to write ML layers concisely
------------------------------------------------

This should be as simple as possible but no simpler. Let's discuss TC in the context of alternative solutions:

C-style loops
^^^^^^^^^^^^^

:code:`C` loops over multi-dimensional arrays are non-ambiguous and general.
For the purpose of writing programs operating on dense tensors, they are
quite verbose and generally quite tedious to write and maintain code in.
Still :code:`C` loops are very informative because they are understandable by anyone
who wants to program a layer. So :code:`C` loops are a good fit for emitting a
baseline implementation and debugging. TC must be able to emit simple :code:`C` loops
that compiles and run.

Halide
^^^^^^

Halide could be viewed as a specialization of :code:`C` loops for image
processing. This specialization is a trade-off, you can only express box
regions within boxes (no triangular loops). This specialization fits our
application domain and we are using it heavily. Halide's backwards shape
inference allows removing much of the boilerplate specification for
intermediate tensors. The tradeoff is you need to specify input tensor shapes
(also in TC), output tensor shapes (omitted in TC) and the ranges of all
reduction indices (omitted in TC).

TC
^^

The current TC implementation sits somewhere here; less verbose than Halide,
more verbose than matrix algebra. The inference procedure has been one subtle
tradeoff in TC. It has been designed to follow an intuitive enough mental model,
but may still evolve in the future towards greater expressiveness, see :ref:`inference`.

Matrix Languages
^^^^^^^^^^^^^^^^
Matrix languages such as Matlab are very concise and make sense mathematically
but don't naturally extend to tensors (what does the operator :code:`*` mean on 3-D
tensors?). As a consequence loops need to be introduced prematurely; TC avoids
this pit-hole.
