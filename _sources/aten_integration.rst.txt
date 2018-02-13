Examples of TC integration
==========================

TC greatly simplifies ML framework implementations by providing a concise and
powerful syntax which can be efficiently translated to high-performance computation kernels, automatically.

We provide lightweight integration with Caffe2 and a C++ Tensor library called ATen.
The integration API looks as simple as below:

.. code-block:: cpp

    #include <ATen/ATen.h>
    #include "tc/aten/aten_compiler.h"
    #include "tc/core/mapping_options.h"

    // 1. Define and setup the TC compilation unit with CUDA memory management backed by ATen.
    std::string tc = R"TC(
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
      C(i, j) +=! A(i, k) * B(k, j)
    })TC";

    // 2. Allocate tensors with random data
    at::Tensor A = at::CUDA(at::kFloat).rand({12, 34});
    at::Tensor B = at::CUDA(at::kFloat).rand({34, 56});
    std::vector<at::Tensor> outputs;

    // 3. Chose mapping options
    auto mappingOptions = tc::MappingOptions::makeNaiveMappingOptions();

    // 4. Compile and run the TC
    tc::ATenCompilationUnit atCompl;
    atCompl.define(tc);
    auto handle = atCompl.compile("matmul", {A, B}, mappingOptions);
    atCompl.run("matmul", {A, B}, outputs, handle);
