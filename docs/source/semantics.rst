Semantics
=========

Tensor Comprehensions follows the follow semantics.

Types
-----

Values between statements are always tensors of primitive types (e.g. :code:`float(A,B)`, a tensor of rank 2).
They can be 0-rank and omit the dimension list (e.g :code:`float`).
Size variables (e.g. :code:`A` and :code:`B` in :code:`float(A,B)`) are used to represent the sizes of the dimensions.
If a size variable is repeated it means that tensors of that type must share the same size in that dimension.
Size variables evaluate to the size of the dimension when used in expressions.

The type of output values is omitted and is inferred based on how it is defined as described below.

Data Layout
-----------
The memory layout implied by TC is row-major (C-like).

Variable Scoping
----------------

There are three different kinds of variables, which all share the same namespace:

1. size variables, introduced by tensor types in the type signature, which evaluate to the size of the dimension;
2. tensor variables, introduced by tensor types in the type signature, with ranges either prescribed (input tensors) or inferred (output tensors);
3. loop index variables, are implicitly defined when used in a statement.

When an identifier is used in a statement but is otherwise not in scope, it is defined to be an index variable for that statement.
Each index variable has an associated range :code:`[b,e)` over which it operates.
That range is inferred by its use, as described below.
Index variables go out of scope after the statement, allowing the reuse of short variable names like :code:`i`.

Implied Reductions and operators
--------------------------------

If an index variable appears on the right but not on the left of a statement,
it is a reduction index for the statement.
If a statement has one or more reduction variables then it must specify a :code:`reduction`
operator such as :code:`+` or :code:`max`.
There is only one reduction operator for the entire statement because
combinations like :code:`max/+` on different dimensions have different mathematical meanings depending on loop order.
All reduction operators are considered to be associative and commutative to allow for arbitrary order of evaluation.

Reduction operators may be suffixed with :code:`!` (for example :code:`+=!`) to indicate that the
tensor to which values are accumulated should first be initialized with the identity of the reduction
operator (e.g., :code:`0` for :code:`+`). Otherwise, values are accumulated directly to the output or
temporary tensor passed to the kernel.

Size Expressions
----------------

Size expressions are a subset of normal expressions that can be used in explicit range constraints and in pattern matching.
They are any expression over integral scalars that do not include tensor reads :code:`T(...)` or any loop index variables.
They may include size variables, or dimension specifiers :code:`T.1`, for tensors that have already been defined in previous statements.
These values can be computed without performing any tensor-wide loops.

Statements
----------

A statement specifies a new operation to define, an optional reduction, and a right hand side:

.. code::

    v(index_variables) reduction=! rhs_expression

:code:`index_variables` must be a list of index variables defined in the :code:`rhs_expressions`
:code:`reduction` is optional if all index variables appear on the left hand side.
The value computed for tensor :code:`v` is equivalent to first assigning all
elements of :code:`v` to the identity value of :code:`reduction`, then
evaluating :code:`rhs_expression` at all points in the iteration space defined
by the ranges of the loop index variables and reducing into the entry of the
tensor specified on the left-hand side. The order in which these expressions
are evaluated should not change the result because the reduction is
associative and commutative. If :code:`!` is not present, :code:`v` is not
re-initialized, and the reduction takes into account the existing values in :code:`v`.

Expressions
-----------

Mathematical expressions behave as expected, including built-in functions like :code:`log(...)`.

:code:`tensor_variable(exp_list)` represents a read of a tensor at the indices defined by evaluating :code:`exp_list`. :code:`exp_list` can include arbitrary expressions (pattern matching of indices is limited to linear expressions, but actual computation is not). The effect of reading outside of the valid range of the tensor results in undefined behavior.

Grammar
-------

The EBNF for the TC comprehension language is::

    num ::= <number literal with C syntax>
    id ::= [_a-zA-Z0-9]*[_a-zA-Z][_a-zA-Z0-9]*
    exp ::= num
          | ( '-' | '!' | ... ) exp
          | exp ( [+-*/%] | '==' | '!=' | '<=' | ... ) exp
          | exp '?' exp ':' exp
          | id '.' num # range of num-th dimension of id
          | id '(' exp_list ')' # builtin call or tensor access

    reduction ::= <associative reduction operator>
                | '+='  | '*='  | 'min='  | 'max='
                | '+=!' | '*=!' | 'min=!' | 'max=!'

    range_constraint ::= id 'in' exp ':' exp

    stmt ::= id '(' id_list ')' [ '=' | reduction ] exp
               [ 'where' range_constraint_list ]
           | id_list = id '('id_list ')' # TC function call

    arg ::= type id
    return ::= id # inferred return type and range

    scalar_type ::= 'double' | 'float' | 'half'
                  | 'int' | 'byte' | 'uint32' | ...

    type ::= scalar_type [ '(' id_list ')' ]

    func ::= # TC function definition
      'def' id '(' arg_list ')' '->' '(' return_list ')' '{'
        stmt_list
      '}'

    id_list ::= <comma separated id list>
    exp_list ::= <comma separated exp list>
    arg_list ::= <comma separated arg list>
    stmt_list ::= <whitespace separated stmt list>
    return_list ::= <comma separated return list>
    range_constraint_list ::= <non-empty comma separated
                               range_constraint list>
