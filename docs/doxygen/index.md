What is Tensor Comprehensions?      {#mainpage}
==============================

Tensor Comprehensions(TC) is a notation based on generalized Einstein notation for computing on
multi-dimensional arrays. TC greatly simplifies ML framework implementations by
providing a concise and powerful syntax which can be efficiently translated to
high-performance computation kernels, automatically.

TC are supported both in Python and C++, we also provide
lightweight integration with Caffe2 and PyTorch. More generally the only
requirement to integrate TC into a workflow is to use a simple tensor library
with a few basic functionalities.

Tensor Comprehension Notation
-----------------------------
TC borrow three ideas from Einstein notation that make expressions concise:

1. Loop index variables are defined implicitly by using them in an expression and their range is aggressively inferred based on what they index.
2. Indices that appear on the right of an expression but not on the left are assumed to be reduction dimensions.
3. The evaluation order of points in the iteration space does not affect the output.

Let's start with a simple example is a matrix vector product:

    def mv(float(R,C) A, float(C) B) -> (o) {
      o(i) +=! A(i,j) * B(j)
    }

`A` and `x` are input tensors. `o` is an output tensor.
The statement `o(i) += A(i,j) * b(j)` introduces two index variables `i` and `j`.
Their range is inferred by their use indexing `A` and `B`. `i = [0,R)`, `j = [0,C)`.
Because `j` only appears on the right side,
stores into `o` will reduce over `j` with the reduction specified for the loop.
Reductions can occur across multiple variables, but they all share the same kind of associative reduction (e.g. +=)
to maintain invariant (3). `mv` computes the same thing as this C++ loop:

    for(int i = 0; i < R; i++) {
      o(i) = 0.0f;
      for(int j = 0; j < C; j++) {
        o(i) += A(i,j) * B(j);
      }
    }

The loop order `[i,j]` here is arbitrarily chosen because the computed value of a TC is always independent of the loop order.

Examples of TC
--------------

We provide a few basic examples.

**Simple matrix-vector**:

    def mv(float(R,C) A, float(C) B) -> (o) {
      o(i) += A(i,j) * B(j)
    }

**Simple matrix-multiply:**

Note the layout for B is transposed and matches the
traditional layout of the weight matrix in a linear layer):

    def mm(float(X,Y) A, float(Y,Z) B) -> (R) {
      R(i,j) += A(i,j) * B(j,k)
    }

**Simple 2-D convolution (no stride, no padding):**

    def conv(float(B,IP,H,W) input, float(OP,IP,KH,KW) weight) -> (output) {
      output(b, op, h, w) += input(b, ip, h + kh, w + kw) * weight(op, ip, kh, kw)
    }

**Simple 2D max pooling:**

Note the similarity with a convolution with a
"select"-style kernel):

    def maxpool2x2(float(B,C,H,W) input) -> (output) {
      output(b,c,i,j) max= input(b,c,2*i + kw, 2*j + kh)
        where kw = [0, 2[, kh = [0, 2[
    }
