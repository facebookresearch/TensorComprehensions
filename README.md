# ![Tensor Comprehensions](docs/source/_static/img/tc-logo-full-color-with-text-2.png)

Test

Tensor Comprehensions (TC) is a fully-functional C++ library to *automatically* synthesize high-performance machine learning kernels using [Halide](https://github.com/halide/Halide), [ISL](http://isl.gforge.inria.fr/) and NVRTC or LLVM. TC additionally provides basic integration with Caffe2 and PyTorch. We provide more details in our paper on [arXiv](https://arxiv.org/abs/1802.04730).

This library is designed to be highly portable, machine-learning-framework agnostic and only requires a simple tensor library with memory allocation, offloading and synchronization capabilities.

For now, we have integrated TC with the [Caffe2](https://github.com/caffe2/caffe2) and [PyTorch](https://github.com/pytorch/pytorch/).

# A simple example

The following illustrates a short but powerful feature of the library: the capacity to JIT-compile high-performance machine learning kernels on demand, for specific sizes.

```python
import tensor_comprehensions as tc
import torch
lang = """
def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
    O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
}
"""
N, C1, C2, C3, H, W = 32, 512, 8, 2, 28, 28
tensordot = tc.define(lang, name="tensordot")
I0, I1 = torch.randn(N, C1, C2, H, W).cuda(), torch.randn(N, C2, C3, H, W).cuda()
best_options = tensordot.autotune(I0, I1, cache=True)
out = tensordot(I0, I1, options=best_options)
```

After a few generations of `autotuning` on a 2-GPU P100 system, we see results resembling:

![Autotuning Sample](docs/source/_static/img/autotuning.png)

We have not yet characterized the precise fraction of peak performance we obtain but it is not uncommon to obtain 80%+ of peak shared memory bandwidth after autotuning. Solid register-level optimizations are still in the work but TC in its current form already addresses the productivity gap between the needs of research and the needs of production. Which is why we are excited to share it with the entire community and bring this collaborative effort in the open.

# Documentation

**General**: You can find detailed information about Tensor Comprehensions [here](https://facebookresearch.github.io/TensorComprehensions/).

**C++ API**: We also provide documentation for our C++ API which can can be found [here](https://facebookresearch.github.io/TensorComprehensions/api/)

# Installation

## Binaries

We provide conda package for making it easy to install and use TC binary. Please refer to our documentation
[here](https://facebookresearch.github.io/TensorComprehensions/framework/pytorch_integration/getting_started.html) for instructions.

## From Source

You can find documentation [here](https://facebookresearch.github.io/TensorComprehensions/) which contains instructions for building TC via docker, conda packages or in non-conda environment.

# Communication

* **Email**: tensorcomp@fb.com
* **GitHub issues**: bug reports, feature requests, install issues, RFCs, thoughts, etc.
* **Slack**: For discussion around framework integration, build support, collaboration, etc. join our slack channel https://tensorcomprehensions.herokuapp.com/.

# Code of Conduct
See the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) file for more details.

# License
Tensor Comprehensions is distributed under a permissive Apache v2.0 license, see the [LICENSE](LICENSE) file for more details.

# Contributing
See the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.
