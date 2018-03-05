Relation to Halide
==================

Tensor Comprehensions uses tools from `Halide <http://halide-lang.org>`_ whenever possible. We
do this both to avoid reinventing the wheel, and also so that we can
exploit more of Halide's features in the future.

But don't be misled by our use of Halide. Tensor Comprehensions is a
different language, with different semantics, a different scope, and a
radically different way of scheduling code. For a user, two major
differences are:

1. In Halide, pipeline stages are functions over a potentially infinite
   domain. In Tensor Comprehensions, pipeline stages are tensors,
   which have a semantically meaningful size. Two practical benefits
   are:

  * In reduction-heavy code (such as deep neural net pipelines) this
    lets us infer the ranges of reductions, which makes for terser
    code.

  * In Halide one must specify the size of the output tensors. In
    Tensor Comprehensions the output size is inferred. It is
    effectively part of the specification of the algorithm.

2. Halide requires manual scheduling of code onto hardware resources
   using the scheduling sub-language. This requires architectural
   expertise. There is an automatic scheduler, but it currently only
   works on the CPU, and only works well on stencil pipelines. Tensor
   Comprehensions automatically schedules code for CPU and GPU using
   polyhedral techniques. It is a more suitable tool for the machine
   learning practitioner without architectural expertise. In the
   future we hope to help improve Halide's autoscheduler to the point
   where this distinction no longer exists, but for now it creates a
   significant difference in the userbase of the two languages.

Use of Halide in TC
-------------------

In our compiler implementation, we use Halide's intermediate
representation (IR), some analysis tools that operate on it, portions
of Halide's lowering pipeline, and some of Halide's X86 backend. A
Halide-centric description of our compilation flow, using Halide's
terminology, is as follows:

1. Tensor Comprehensions source code is parsed, semantically checked,
   then translated into an equivalent Halide pipeline using Halide's
   front-end IR (:code:`Funcs`, :code:`Vars`, and :code:`RDoms`).

2. Bounds on :code:`Funcs` and :code:`RDoms` are inferred in the forwards direction
   during this translation according to our semantics, and explicitly
   set on the :code:`Funcs` using the :code:`Func::bound` scheduling
   directive. Halide's equation solving tools are employed to
   implement our range inference semantics. Every :code:`Func` is scheduled
   compute_root.

3. Using a heavily abbreviated version of Halide's lowering pipeline,
   this is converted to Halide's imperative IR (a :code:`Stmt`). The lowering
   stages from Halide used are:

   * :code:`schedule_functions:code:`, which creates the initial loop nest
   * :code:`remove_undef`, which handles in-place mutation of outputs
   * :code:`uniquify_variable_names`
   * :code:`simplify`

   Notably absent are any sort of bounds inference (bounds were already determined during step 2), or any sort of storage
   flattening, vectorization, unrolling, or cross-device copies. This
   abbreviated lowering stops at a much higher level of abstraction
   than Halide proper.

4. This :code:`Stmt` is then converted to a polyhedral representation. The
   individual Provide nodes are preserved, but the containing loop
   structure is discarded. In the polyhedral representation, the
   Provide nodes are represented as abstract statement instances with
   known data dependencies.

5. The loop structure is optimized using polyhedral techniques.

6. If compiling to CUDA: The polyhedral loop structure is compiled to
   CUDA source code, with the Halide expressions (:code:`Exprs`) in each
   individual Provide node emitted as strings using a class derived
   from Halide's :code:`IRPrinter`.

7. If compiling to X86: The polyhedral loop structure is compiled to
   llvm bitcode, with the Halide expressions emitted using a class
   derived from Halide's :code:`CodeGen_X86`.
