# ![Tensor Comprehensions](docs/source/_static/img/tc-logo-full-color-with-text-2.png)

# Using Tensor Comprehensions with PyTorch

A **blogpost** on Tensor Comprehensions can be read [here](https://research.fb.com/announcing-tensor-comprehensions/).

Tensor Comprehensions (TC) is a framework agnostic library to *automatically* synthesize high-performance machine learning kernels. TC relies on `Halide's` intermediate representation to express computation and analysis tools to reason about it. TC uses `polyhedral` compilation techniques to (semi-)automatically decide how to perform this computation efficiently on and produce fast code.

**Disclaimer:** Polyhedral compilation is an active research area, new results may affect TC performance and/or behavior

We provide integration of Tensor Comprehensions (TC) with PyTorch for both training
and inference purposes. Using TC, you can express an operator using Einstein notation and get the fast CUDA code for that layer with a few lines of code. By providing TC integration with PyTorch, we hope to make it further easy to write new operations with TC. Here is what the PyTorch-TC package provides:

- inputs and outputs to functions are `torch.*Tensor`s
- Integration with PyTorch `autograd`: if you specify forward and backward functions, you get an autograd function that takes `Variable` as input and returns `Variable` as output.
- autotuner results can be cached to a file (for reuse)


## Installation

To make it easy to use TC, we provide conda packages for it. Follow the instructions at our documentation [here](https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/getting_started.html#installation) for installing the PyTorch TC package.

## Examples and Documentation

In order to explore Tensor Comprehensions (TC), there are few helpful resources to get started:

1. We provide **examples** of TC definitions covering wide range of Deep Learning layers. Please look at `test_python/layers/` for various layers definitions.

2. [TC Documentation](https://facebookresearch.github.io/TensorComprehensions/index.html)
is a very helpful resource to understand how Tensor Comprehensions are expressed. The sections on
[introduction](https://facebookresearch.github.io/TensorComprehensions/introduction.html),
[range inference](https://facebookresearch.github.io/TensorComprehensions/inference.html),
[semantics](https://facebookresearch.github.io/TensorComprehensions/semantics.html), [mapping_options](https://facebookresearch.github.io/TensorComprehensions/mapping_options.html) are particularly helpful to get insights into writing Tensor Comprehensions.

3. **Autotuner**: TC provides an evolutionary search based algorithm to automatically tune the kernel.
You can read briefly about autotuner [here](https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/autotuning_layers.html) and look at various tests at `test_python/layers/test_autotuner.py`.

4. To construct a TC autograd function, `test_python/layers/test_convolution_train.py` is one self-descriptive example.

### A simple Example

```python
import tensor_comprehensions as tc
import torch
lang = """
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(i, j) +=! A(i, kk) * B(kk, j)
}
"""
# The name should match the name of the "def" in "lang"
matmul = tc.define(lang, name="matmul")
mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
out = matmul(mat1, mat2)
```

**NOTE:** Performance of above kernel will be bad in this case because the mapping options were not chosen and the autotuning was not applied. See the documentation for how to supply mapping options properly.
