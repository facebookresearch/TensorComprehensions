# ![Tensor Comprehensions](docs/source/_static/img/tc-logo-full-color-with-text-2.png)

Tensor Comprehensions (TC) is a fully-functional C++ library to *automatically* synthesize high-performance machine learning kernels using [Halide](https://github.com/halide/Halide), [ISL](http://isl.gforge.inria.fr/) and NVRTC or LLVM. TC additionally provides basic integration with Caffe2 and pybind11 bindings for use with python.

This library is designed to be highly portable, machine-learning-framework agnostic and only requires a simple tensor library with memory allocation, offloading and synchronization capabilities.

For now, we have integrated TC with the [Caffe2](https://github.com/caffe2/caffe2) and [ATen](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen) tensor libraries.

# A simple example

The following illustrates a short but powerful feature of the library: the capacity to JIT-compile high-performance machine learning kernels on demand, for specific sizes.

```cpp
  #include <ATen/ATen.h>

  #include "tc/aten/aten_compiler.h"
  #include "tc/core/mapping_options.h"

  // 1. Define and setup the TC compilation unit with CUDA memory
  // management backed by ATen tensors.
  std::string tc = R"TC(
    def channel_contraction(float(N, C1, C2, H, W) I0,
                            float(N, C2, C3, H, W) I1)
    -> (O)
    {
      O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
    }
  )TC";

  tc::ATenCompilationUnit atCompl;
  atCompl.define(tc);

  // 2. Allocate tensors with random data
  std::vector<at::Tensor> outputs;
  at::Tensor I0 = at::CUDA(at::kFloat).rand({32, 512, 8, 28, 28});
  at::Tensor I1 = at::CUDA(at::kFloat).rand({32,   8, 2, 28, 28});;

  // 3. Run autotuning with evolutionary search starting from a naive option
  auto options = tc::MappingOptions::makeNaiveMappingOptions();
  auto bestOption =
    autotune(cacheFilename, TC, "channel_contraction", {I0, I1}, options, {options});

  // 4. Compile and run the TC with the best option.
  // Outputs get allocated; could also be pre-allocated and passed
  auto handle = atCompl.compile("channel_contraction", {I0, I1}, bestOption);
  atCompl.run("channel_contraction", {I0, I1}, outputs, handle);

  // 5. Perform precision checks against an ATen reference implementation
  check({I0, I1}, outputs, [&I0, &I1](){ return ...; });
```

After a few generations of autotuning on a 2-GPU P100 system, we see results resembling:

![Autotuning Sample](docs/source/_static/img/autotuning.png)

We have not yet characterized the precise fraction of peak performance we obtain but it is not uncommon to obtain 80%+ of peak shared memory bandwidth after autotuning. Solid register-level optimizations are still in the work but TC in its current form already addresses the productivity gap between the needs of research and the needs of production. Which is why we are excited to share it with the entire community and bring this collaborative effort in the open.

# Documentation, Environment and Prerequisites
We provide pre-built docker images in the docker subdirectory, they can be downloaded from [dockerhub](https://hub.docker.com/u/tensorcomprehensions/). We use and support those images as part of our continuous integration. Note that we can cross-compile CUDA (but not execute) even if the machine has no physical GPUs. In any case the CUDA toolkit and libraries should always be installed, for now.

To get started, see the [docs](master/docs) directory.

# Preparing the source

Once the environment is set up properly you can:
``` shell
git clone --recursive git@github.com:facebookresearch/TensorComprehensions.git
cd TensorComprehensions
```

# Build and test

```shell
BUILD_TYPE=Release CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all && ./test_cpu.sh
BUILD_TYPE=Release CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all && ./test.sh
```

# Build and test with Caffe2

```shell
BUILD_TYPE=Release WITH_CAFFE2=ON CLANG_PREFIX=$(llvm-config --prefix) ./build.sh --all && ./build/test/test_caffe2
```

# License
Tensor Comprehensions is distributed under a permissive Apache v2.0 license, see the [LICENSE](LICENSE) file for more details.


# Contributing
See the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.
