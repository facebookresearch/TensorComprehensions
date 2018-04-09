/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/lang/canonicalize.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

using OutputsAndCode = std::pair<std::vector<at::Tensor>, std::string>;

///////////////////////////////////////////////////////////////////////////////
// Base unit test class
///////////////////////////////////////////////////////////////////////////////
template <typename Backend>
struct TcMapperTest : public ::testing::Test {
  uint32_t M = 165, N = 197, K = 227;
  int B = 100, D = 1000;
  int C1 = 512, C2 = 8, C3 = 2, H = 28, W = 28;

  template <typename CheckFunction>
  OutputsAndCode Check(
      const std::string& tc,
      const std::string& name,
      const typename Backend::MappingOptionsType& mappingOptions,
      const std::vector<at::Tensor> inputs,
      CheckFunction checkFun) {
    auto pExecutor =
        tc::aten::compile<Backend>(tc, name, inputs, mappingOptions);
    std::vector<at::Tensor> outputs =
        tc::aten::prepareOutputs(tc, name, inputs);
    tc::aten::run(*pExecutor, inputs, outputs);
    checkFun(inputs, outputs);
    auto outputDLTensors = tc::aten::makeDLTensors(outputs);
    return std::make_pair(
        std::move(outputs), std::move(pExecutor->compiledSource));
  }
};

///////////////////////////////////////////////////////////////////////////////
// 1-D reduction
//   C +=! A(r_m)
///////////////////////////////////////////////////////////////////////////////
template <typename Backend>
struct TcMapper1DReductionTest : public TcMapperTest<Backend> {
  using TcMapperTest<Backend>::Check;
  using TcMapperTest<Backend>::M;

  OutputsAndCode Check(
      at::Tensor A,
      const typename Backend::MappingOptionsType& mappingOptions,
      uint32_t version = 0) {
    auto reduction1DTCs = {
        R"TC(
def sum1D(float(M) A) -> (C) {
    C(0) +=! A(r_m) where i in 0:1
}
)TC",
        R"TC(
def sum1D(float(M) A) -> (C) {
    C() +=! A(r_m)
}
)TC",
        R"TC(
def sum1D(float(M) A) -> (C) {
    C +=! A(r_m)
}
)TC",
        R"TC(
def sum1D(float(M) A) -> (C) {
    C(i) +=! A(r_m) where i in 0:1
}
)TC"};

    CHECK_GE(3, version) << "Versions [0-3] supported, asked for: " << version;
    auto refOutput = A.sum();
    auto checkFun = [&, refOutput](
                        const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      at::Tensor diff = outputs[0].sub(refOutput);
      return checkRtol(diff, inputs, M, 5e-7);
    };
    return Check(
        *(reduction1DTCs.begin() + version),
        "sum1D",
        mappingOptions,
        {A},
        checkFun);
  }
};

///////////////////////////////////////////////////////////////////////////////
// 2-D reduction
//   C(m) +=! A(m, r_n)
///////////////////////////////////////////////////////////////////////////////
template <typename Backend>
struct TcMapper2DReductionTest : public TcMapperTest<Backend> {
  using TcMapperTest<Backend>::Check;
  using TcMapperTest<Backend>::M;
  using TcMapperTest<Backend>::N;

  OutputsAndCode Check(
      at::Tensor A,
      const typename Backend::MappingOptionsType& mappingOptions,
      bool skipCheck = false) {
    auto tc = R"TC(
def sum2D(float(M, N) A) -> (C) {
    C(m) +=! A(m, r_n)
}
)TC";
    auto refOutput = A.sum(1);
    auto checkFun = [&, refOutput](
                        const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      at::Tensor diff = outputs[0].sub(refOutput);
      return checkRtol(diff, inputs, N, 5e-7);
    };
    auto noCheckFun = [](const std::vector<at::Tensor>& inputs,
                         std::vector<at::Tensor>& outputs) { return true; };
    return skipCheck ? Check(tc, "sum2D", mappingOptions, {A}, noCheckFun)
                     : Check(tc, "sum2D", mappingOptions, {A}, checkFun);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Matmul tests
//   C(m, n) +=! A(m, r_k) * B(r_k, n)
///////////////////////////////////////////////////////////////////////////////
template <typename Backend>
struct TcMapperMatmulTest : public TcMapperTest<Backend> {
  using TcMapperTest<Backend>::Check;
  using TcMapperTest<Backend>::K;
  using TcMapperTest<Backend>::M;
  using TcMapperTest<Backend>::N;

  OutputsAndCode Check(
      at::Tensor A,
      at::Tensor B,
      const typename Backend::MappingOptionsType& mappingOptions) {
    constexpr auto tc = R"TC(
def matmul(float(M, K) A, float(K, N) B) -> (C) {
    C(m, n) +=! A(m, r_k) * B(r_k, n)
}
)TC";
    auto refOutput = A.mm(B);
    auto checkFun = [&, refOutput](
                        const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      at::Tensor diff = outputs[0].sub(refOutput);
      return checkRtol(diff, inputs, K, 5e-7);
    };
    return Check(tc, "matmul", mappingOptions, {A, B}, checkFun);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Batch Matmul tests
//   Z(b, n, k) +=! X(b, n, r_m) * Y(b, r_m, k)
///////////////////////////////////////////////////////////////////////////////
template <typename Backend>
struct TcMapperBatchMatmulTest : public TcMapperTest<Backend> {
  using TcMapperTest<Backend>::Check;
  using TcMapperTest<Backend>::K;
  using TcMapperTest<Backend>::M;
  using TcMapperTest<Backend>::N;

  OutputsAndCode Check(
      at::Tensor A,
      at::Tensor B,
      const typename Backend::MappingOptionsType& mappingOptions) {
    constexpr auto tc = R"TC(
def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
    Z(b, n, k) +=! X(b, n, r_m) * Y(b, r_m, k)
}
)TC";
    auto refOutput = A.bmm(B);
    auto checkFun = [&, refOutput](
                        const std::vector<at::Tensor>& inputs,
                        const std::vector<at::Tensor>& outputs) {
      TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
      at::Tensor diff = outputs[0].sub(refOutput);
      return checkRtol(diff, inputs, K, 5e-7);
    };
    return Check(tc, "batch_matmul", mappingOptions, {A, B}, checkFun);
  }
};
