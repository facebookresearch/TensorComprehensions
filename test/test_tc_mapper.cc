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

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/canonicalize.h"
#include "tc/lang/sema.h"
#include "tc/lang/tree.h"

#include "test_harness_aten_cuda.h"

using namespace std;

using OutputsAndCuda = std::pair<std::vector<at::Tensor>, std::string>;

struct TcMapperTest : public ::testing::Test {
  uint32_t M = 165, N = 197, K = 227;
  int B = 100, D = 1000;
  int C1 = 512, C2 = 8, C3 = 2, H = 28, W = 28;

  template <typename CheckFunction>
  OutputsAndCuda Check(
      const std::string& tc,
      const std::string& name,
      const tc::CudaMappingOptions& mappingOptions,
      const std::vector<at::Tensor> inputs,
      CheckFunction checkFun) {
    tc::CudaCache::enableCache();

    std::vector<at::Tensor> outputs;
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(tc);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    atCompl.run(name, inputs, outputs, handle);
    checkFun(inputs, outputs);

    auto inputDLTensorsPair = tc::toConstDlpackTensors(inputs);
    auto outputDLTensorsPair = tc::toConstDlpackTensors(outputs);
    tc::ScopeGuard sg([&]() {
      tc::deleteDlmTensors(inputDLTensorsPair.second);
      tc::deleteDlmTensors(outputDLTensorsPair.second);
    });
    auto cached = tc::CudaCache::getCache()->retrieveKernel(
        [&]() {
          std::stringstream ss;
          ss << lang::canonicalize(
              lang::Sema().checkFunction(lang::Parser(tc).parseFunction()));
          return ss.str();
        }(),
        mappingOptions,
        inputDLTensorsPair.first,
        outputDLTensorsPair.first);
    EXPECT_FALSE(cached == nullptr);

    return std::make_pair(std::move(outputs), std::move(cached->source));
  }
};

///////////////////////////////////////////////////////////////////////////////
// 1-D reduction
//   C +=! A(j)
///////////////////////////////////////////////////////////////////////////////
constexpr auto reduction1DTCs = {
    R"TC(
def sum1D(float(M) A) -> (C) {
  C(0) +=! A(j) where i in 0:2
}
)TC",
    R"TC(
def sum1D(float(M) A) -> (C) {
  C() +=! A(j)
}
)TC",
    R"TC(
def sum1D(float(M) A) -> (C) {
  C +=! A(j)
}
)TC",
    R"TC(
def sum1D(float(M) A) -> (C) {
  C(i) +=! A(j) where i in 0:1
}
)TC"};

struct TcMapper1DReductionTest : public TcMapperTest {
  using TcMapperTest::Check;
  using TcMapperTest::M;

  OutputsAndCuda Check(
      at::Tensor A,
      const tc::CudaMappingOptions& mappingOptions,
      uint32_t version = 0) {
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

TEST_F(TcMapper1DReductionTest, DISABLED_Reduction1Dv0) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .tile(0)
                            .mapToBlocks({})
                            .mapToThreads({16});
  LOG(INFO) << mappingOptions << endl;
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 0);
}

TEST_F(TcMapper1DReductionTest, DISABLED_Reduction1Dv1) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .tile(0)
                            .mapToBlocks({})
                            .mapToThreads({16});
  LOG(INFO) << mappingOptions << endl;
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 1);
}

TEST_F(TcMapper1DReductionTest, DISABLED_Reduction1Dv2) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .tile(0)
                            .mapToBlocks({})
                            .mapToThreads({16});
  LOG(INFO) << mappingOptions << endl;
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 2);
}

TEST_F(TcMapper1DReductionTest, DISABLED_Reduction1Dv3) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .tile(0)
                            .mapToBlocks({})
                            .mapToThreads({16});
  LOG(INFO) << mappingOptions << endl;
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 3);
}

///////////////////////////////////////////////////////////////////////////////
// 2-D reduction
//   C(i) +=! A(i, j)
///////////////////////////////////////////////////////////////////////////////
struct TcMapper2DReductionTest : public TcMapperTest {
  using TcMapperTest::Check;
  using TcMapperTest::M;
  using TcMapperTest::N;

  OutputsAndCuda Check(
      at::Tensor A,
      const tc::CudaMappingOptions& mappingOptions,
      bool skipCheck = false) {
    string tc = R"TC(
def sum2D(float(M, N) A) -> (C) {
  C(i) +=! A(i, j)
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

TEST_F(TcMapper2DReductionTest, Reduction2D1) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .tile(32, 32)
                            .mapToBlocks({1, 1})
                            .mapToThreads({32})
                            .matchLibraryCalls(true);
  LOG(INFO) << mappingOptions << endl;
  at::Tensor A = at::CUDA(at::kFloat).rand({M, N});
  Check(A, mappingOptions);
}

struct TcMapper2DReductionStressTest : public TcMapper2DReductionTest {
  using TcMapper2DReductionTest::M;
  using TcMapper2DReductionTest::N;

  OutputsAndCuda
  Check(size_t tix, size_t tiy, bool skipCheck = false, bool ones = false) {
    M = tiy;
    N = tix;
    auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                              .tile(tiy, tix)
                              .mapToBlocks({1})
                              .mapToThreads({tix, tiy})
                              .matchLibraryCalls(true);
    LOG(INFO) << mappingOptions << endl;
    at::Tensor A = ones ? at::CUDA(at::kFloat).ones({M, N})
                        : at::CUDA(at::kFloat).rand({M, N});
    return TcMapper2DReductionTest::Check(A, mappingOptions, skipCheck);
  }
};

TEST_F(TcMapper2DReductionStressTest, ThreadIdy1) {
  for (int i : {1, 2, 4, 7, 8, 11, 15, 17, 24, 32, 35, 42, 64, 128, 130}) {
    auto res = Check(i, 1);
    if (i > 1) {
      std::string expected = std::string("__tc::CubReduceAlongX<") +
          std::to_string(i) + std::string(",1,1,__tc::ReductionOp::Sum>");
      ASSERT_NE(std::string::npos, res.second.find(expected))
          << "In resulting code:\n"
          << res.second << "\ncould not find: " << expected;
    } else {
      std::string expected = "__tc::CubReduceAlongX<";
      ASSERT_EQ(std::string::npos, res.second.find(expected))
          << "In resulting code:\n"
          << res.second << "\nfound unexpected: " << expected;
    }
  }
}

TEST_F(TcMapper2DReductionStressTest, 4x7) {
  Check(4, 7);
}

TEST_F(TcMapper2DReductionStressTest, 11x5) {
  Check(11, 5);
}

TEST_F(TcMapper2DReductionStressTest, 16x9) {
  Check(16, 9);
}

TEST_F(TcMapper2DReductionStressTest, 8x11) {
  Check(8, 11);
}

TEST_F(TcMapper2DReductionStressTest, 11x8) {
  Check(11, 8);
}

TEST_F(TcMapper2DReductionStressTest, 111x7) {
  Check(111, 7);
}

TEST_F(TcMapper2DReductionStressTest, 128x7) {
  Check(128, 7);
}

TEST_F(TcMapper2DReductionStressTest, 7x128) {
  Check(7, 128);
}

// Run this iterative example to find new cases
TEST_F(TcMapper2DReductionStressTest, Iterate) {
  for (auto tix : {1, 2, 5, 8, 11}) {
    for (auto tiy : {3, 5, 11}) {
      Check(tix, tiy);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matmul tests
//   C(i, j) += A(i, k) * B(k, j)
///////////////////////////////////////////////////////////////////////////////
struct TcMapperMatmulTest : public TcMapperTest {
  using TcMapperTest::Check;
  using TcMapperTest::K;
  using TcMapperTest::M;
  using TcMapperTest::N;

  OutputsAndCuda Check(
      at::Tensor A,
      at::Tensor B,
      const tc::CudaMappingOptions& mappingOptions) {
    string tc = R"TC(
def matmul(float(M, K) A, float(K, N) B) -> (C) {
  C(i, j) +=! A(i, k) * B(k, j)
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

TEST_F(TcMapperMatmulTest, Matmul1DSchedule) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .fixParametersBeforeScheduling(true)
                            .tile(1, 1, K)
                            .mapToBlocks({M, N})
                            .mapToThreads({std::min(32u, K)})
                            .matchLibraryCalls(true);
  LOG(INFO) << mappingOptions << endl;

  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

TEST_F(TcMapperMatmulTest, Matmul1DScheduleMultipleOccurrence) {
  // Without full specialization, AST generator will duplicate the first
  // statement C[i][j] = 0.0f (for K > 0 and K < 0).  The codegen must be able
  // to handle the same statement appearing in different contexts.
  auto mappingOptions = tc::CudaMappingOptions::makeMlpCudaMappingOptions()
                            .fixParametersBeforeScheduling(false)
                            .tile(32, 32, 32)
                            .mapToBlocks({8})
                            .mapToThreads({16})
                            .matchLibraryCalls(true);
  LOG(INFO) << mappingOptions << endl;

  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

TEST_F(TcMapperMatmulTest, Matmul3DSchedule) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .fixParametersBeforeScheduling(true)
                            .mapToBlocks({1, 1, 1})
                            .mapToThreads({4, 1, 1});
  mappingOptions.matchLibraryCalls(true);
  LOG(INFO) << mappingOptions << endl;

  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

TEST_F(TcMapperMatmulTest, Matmul3DScheduleMultipleOccurrence) {
  auto mappingOptions = tc::CudaMappingOptions::makeMlpCudaMappingOptions()
                            .tile(32, 32, 32)
                            .mapToBlocks({8})
                            .mapToThreads({16})
                            .matchLibraryCalls(true);
  LOG(INFO) << mappingOptions << endl;

  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

///////////////////////////////////////////////////////////////////////////////
// Batch Matmul tests
//   Z(b, n, k) += X(b, n, mm) * Y(b, mm, k)
///////////////////////////////////////////////////////////////////////////////
struct TcMapperBatchMatmulTest : public TcMapperTest {
  using TcMapperTest::Check;
  using TcMapperTest::K;
  using TcMapperTest::M;
  using TcMapperTest::N;

  OutputsAndCuda Check(
      at::Tensor A,
      at::Tensor B,
      const tc::CudaMappingOptions& mappingOptions) {
    string tc = R"TC(
  def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
    Z(b, n, k) +=! X(b, n, mm) * Y(b, mm, k)
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

TEST_F(TcMapperBatchMatmulTest, BatchMatmul) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                            .tile(1)
                            .mapToThreads({123})
                            .mapToBlocks({50})
                            .usePrivateMemory(true)
                            .useSharedMemory(true);
  mappingOptions.matchLibraryCalls(true);
  LOG(INFO) << mappingOptions << endl;

  at::Tensor A = at::CUDA(at::kFloat).rand({50, 26, 72});
  at::Tensor B = at::CUDA(at::kFloat).rand({50, 72, 26});
  Check(A, B, mappingOptions);
}

///////////////////////////////////////////////////////////////////////////////
// Hadamard tests
//       Z(b, d) = U(b, d) * V(b, d) * W(b, d)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcMapperTest, BatchTripleHadamard) {
  at::Tensor U = at::CUDA(at::kFloat).rand({B, D});
  at::Tensor V = at::CUDA(at::kFloat).rand({B, D});
  at::Tensor W = at::CUDA(at::kFloat).rand({B, D});
  std::vector<at::Tensor> inputs = {U, V, W};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
    def batch_triple_hadamard(float(B, D) U,
                              float(B, D) V,
                              float(B, D) W)
    -> (Z)
    {
       Z(b, d) = U(b, d) * V(b, d) * W(b, d)
    }
  )TC";

  auto checkFun = [=](const std::vector<at::Tensor>& inputs,
                      std::vector<at::Tensor>& outputs) {
    at::Tensor diff = outputs[0].sub(inputs[0] * inputs[1] * inputs[2]);
    checkRtol(diff, inputs, D);
  };
  Check(
      TC,
      "batch_triple_hadamard",
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions(),
      inputs,
      checkFun);
}

TEST_F(TcMapperTest, TensorDot) {
  N = 32;
  at::Tensor I0 = at::CUDA(at::kFloat).rand({N, C1, C2, H, W});
  at::Tensor I1 = at::CUDA(at::kFloat).rand({N, C2, C3, H, W});
  std::vector<at::Tensor> inputs = {I0, I1};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
    def tensordot(float(N, C1, C2, H, W) I0,
                            float(N, C2, C3, H, W) I1)
    -> (O)
    {
      O(n, c1, c3, h, w) +=! I0(n, c1, c2, h, w) * I1(n, c2, c3, h, w)
    }
  )TC";
  // No defaults for this case
  auto checkFun = [](const std::vector<at::Tensor>& inputs,
                     std::vector<at::Tensor>& outputs) { return true; };
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto name = "tensordot";
  Check(TC, name, options, inputs, checkFun);
  ::benchmarkKernelOptions(TC, name, inputs, options);
}

TEST_F(TcMapperTest, LUT) {
  const int B = 17, N = 82, R = 22;
  at::Tensor LUT = at::CUDA(at::kFloat).rand({B, R});
  at::Tensor I =
      at::CUDA(at::kFloat).rand({B, N}).mul_(B).floor_().toType(at::kInt);
  std::vector<at::Tensor> inputs = {LUT, I};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def fun(float(B, R) LUT, int32(B, N) I) -> (O) {
  O(b, n) +=! LUT(I(b, n), r)
}
)TC";

  auto checkFun = [=](const std::vector<at::Tensor>& inputs,
                      std::vector<at::Tensor>& outputs) {
    at::Tensor LUT = inputs[0].toBackend(at::kCPU);
    at::Tensor I = inputs[1].toBackend(at::kCPU);
    at::Tensor O = outputs[0].toBackend(at::kCPU);
    auto LUTAccessor = LUT.accessor<float, 2>();
    auto IAccessor = I.accessor<int, 2>();
    auto OAccessor = O.accessor<float, 2>();
    for (int b = 0; b < B; b++) {
      for (int n = 0; n < N; n++) {
        float correct = 0;
        for (int r = 0; r < R; r++) {
          int idx = IAccessor[b][n];
          CHECK(idx >= 0 && idx < B);
          correct += LUTAccessor[idx][r];
        }
        OAccessor[b][n] -= correct;
      }
    }

    checkRtol(O, inputs, 5e-7);
  };
  Check(
      TC,
      "fun",
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions(),
      inputs,
      checkFun);
}

TEST_F(TcMapperTest, DISABLED_SpatialBatchNormalization) {
  N = 32;
  at::Tensor eps = at::CUDA(at::kFloat).rand({});
  eps[0] = 1.0f;
  at::Tensor momentum = at::CUDA(at::kFloat).rand({});
  momentum[0] = 1.0;
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C2, H, W});
  at::Tensor rMeanIn = at::CUDA(at::kFloat).rand({C2});
  at::Tensor rVarIn = at::CUDA(at::kFloat).rand({C2});
  std::vector<at::Tensor> inputs = {momentum, eps, I, rMeanIn, rVarIn};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
  def spatial_batch_norm(
    float momentum, float eps,
    float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn)
         -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut)
  {
     mean(c) +=! I(nn, c, hh, ww)
     mean(c)  = mean(c) / (N * H * W)
     rMeanOut(c) = (1 - momentum) * rMeanIn(c) + momentum * mean(c)
     centered(n, c, h, w) = I(n, c, h, w) - rMeanOut(c)
     variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
     expectedVariance(c) +=! (variance(n, c, h, w) + eps) / (N * H * W)
     rVarOut(c) = rsqrt(
       (1 - momentum) * rVarIn(c) + momentum * expectedVariance(c))
     O(n, c, h, w) = centered(n, c, h, w) * rVarOut(c)
     normalizedOut(n, c, h, w) = O(n, c, h, w)
  })TC";

  auto checkFun = [=](const std::vector<at::Tensor>& inputs,
                      std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    bool training = true;
    at::Tensor weight = at::CUDA(at::kFloat).ones({C1});
    at::Tensor bias = at::CUDA(at::kFloat).zeros({C1});
    auto save_mean = outputs[1].clone().zero_();
    auto save_std = outputs[2].clone().zero_();
    auto O = at::batch_norm_forward(
        I,
        weight,
        bias,
        rMeanIn,
        rVarIn,
        training,
        at::Scalar(momentum).toFloat(),
        at::Scalar(eps).toFloat(),
        save_mean,
        save_std);
    auto diff = O.sub(outputs[0]);
    checkRtol(diff, inputs, N * H * W, prec);
  };

  auto name = "spatial_batch_norm";
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .tile(0)
                     .mapToBlocks({1})
                     .mapToThreads({32, 4});
  Check(TC, name, options, inputs, checkFun);
  ::benchmarkKernelOptions(TC, name, inputs, options);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
