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
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/canonicalize.h"
#include "tc/lang/sema.h"
#include "tc/lang/tree.h"

#include "test_harness_aten_cuda.h"
#include "test_tc_mapper_harness-inl.h"

using namespace std;

using TcCudaMapperTest = TcMapperTest<tc::CudaTcExecutor>;
using TcCudaMapper1DReductionTest = TcMapper1DReductionTest<tc::CudaTcExecutor>;
using TcCudaMapper2DReductionTest = TcMapper2DReductionTest<tc::CudaTcExecutor>;
using TcCudaMapperMatmulTest = TcMapperMatmulTest<tc::CudaTcExecutor>;
using TcCudaMapperBatchMatmulTest = TcMapperBatchMatmulTest<tc::CudaTcExecutor>;

///////////////////////////////////////////////////////////////////////////////
// 1-D reduction
//   C +=! A(r_m)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapper1DReductionTest, DISABLED_Reduction1Dv0) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(0)
                            .mapToBlocks({})
                            .mapToThreads({16});
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 0);
}

TEST_F(TcCudaMapper1DReductionTest, Reduction1Dv1) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(0)
                            .mapToBlocks({1})
                            .mapToThreads({16});
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 1);
}

TEST_F(TcCudaMapper1DReductionTest, Reduction1Dv2) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(0)
                            .mapToBlocks({1})
                            .mapToThreads({16});
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 2);
}

TEST_F(TcCudaMapper1DReductionTest, Reduction1Dv3) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(0)
                            .mapToBlocks({1})
                            .mapToThreads({16});
  at::Tensor A = at::CUDA(at::kFloat).rand({M});
  Check(A, mappingOptions, 3);
}

///////////////////////////////////////////////////////////////////////////////
// 2-D reduction
//   C(m) +=! A(m, r_n)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapper2DReductionTest, Reduction2D1) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(32, 32)
                            .mapToBlocks({1, 1})
                            .mapToThreads({32})
                            .matchLibraryCalls(true);
  at::Tensor A = at::CUDA(at::kFloat).rand({M, N});
  Check(A, mappingOptions);
}

///////////////////////////////////////////////////////////////////////////////
// 2-D reduction stress test (CUDA-only)
//   C(m) +=! A(m, r_n)
///////////////////////////////////////////////////////////////////////////////
struct TcCudaMapper2DReductionStressTest : public TcCudaMapper2DReductionTest {
  using TcCudaMapper2DReductionTest::M;
  using TcCudaMapper2DReductionTest::N;

  std::vector<at::Tensor>
  Check(size_t tix, size_t tiy, bool skipCheck = false, bool ones = false) {
    M = tiy;
    N = tix;
    auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                              .tile(tiy, tix)
                              .mapToBlocks({1})
                              .mapToThreads({tix, tiy})
                              .matchLibraryCalls(true);
    at::Tensor A = ones ? at::CUDA(at::kFloat).ones({M, N})
                        : at::CUDA(at::kFloat).rand({M, N});
    return TcCudaMapper2DReductionTest::Check(A, mappingOptions, skipCheck);
  }
};

TEST_F(TcCudaMapper2DReductionStressTest, ThreadIdy1) {
  for (int i : {1, 2, 4, 7, 8, 11, 15, 17, 24, 32, 35, 42, 64, 128, 130}) {
    Check(i, 1);
  }
}

TEST_F(TcCudaMapper2DReductionStressTest, 4x7) {
  Check(4, 7);
}

TEST_F(TcCudaMapper2DReductionStressTest, 11x5) {
  Check(11, 5);
}

TEST_F(TcCudaMapper2DReductionStressTest, 16x9) {
  Check(16, 9);
}

TEST_F(TcCudaMapper2DReductionStressTest, 8x11) {
  Check(8, 11);
}

TEST_F(TcCudaMapper2DReductionStressTest, 11x8) {
  Check(11, 8);
}

TEST_F(TcCudaMapper2DReductionStressTest, 111x7) {
  Check(111, 7);
}

TEST_F(TcCudaMapper2DReductionStressTest, 128x7) {
  Check(128, 7);
}

TEST_F(TcCudaMapper2DReductionStressTest, 7x128) {
  Check(7, 128);
}

// Run this iterative example to find new cases
TEST_F(TcCudaMapper2DReductionStressTest, Iterate) {
  for (auto tix : {1, 2, 5, 8, 11}) {
    for (auto tiy : {3, 5, 11}) {
      Check(tix, tiy);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matmul tests
//   C(m, n) +=! A(m, r_k) * B(r_k, n)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapperMatmulTest, Matmul1DSchedule) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .fixParametersBeforeScheduling(true)
                            .tile(1, 1, K)
                            .mapToBlocks({M, N})
                            .mapToThreads({std::min(32u, K)})
                            .matchLibraryCalls(true);
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

TEST_F(TcCudaMapperMatmulTest, Matmul1DScheduleMultipleOccurrence) {
  // Without full specialization, AST generator will duplicate the first
  // statement C[i][j] = 0.0f (for K > 0 and K < 0).  The codegen must be able
  // to handle the same statement appearing in different contexts.
  auto mappingOptions = tc::CudaMappingOptions::makeMlpMappingOptions()
                            .fixParametersBeforeScheduling(false)
                            .tile(32, 32, 32)
                            .mapToBlocks({8})
                            .mapToThreads({16})
                            .matchLibraryCalls(true);
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

TEST_F(TcCudaMapperMatmulTest, Matmul3DSchedule) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .fixParametersBeforeScheduling(true)
                            .mapToBlocks({1, 1, 1})
                            .mapToThreads({4, 1, 1});
  mappingOptions.matchLibraryCalls(true);
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

TEST_F(TcCudaMapperMatmulTest, Matmul3DScheduleMultipleOccurrence) {
  auto mappingOptions = tc::CudaMappingOptions::makeMlpMappingOptions()
                            .tile(32, 32, 32)
                            .mapToBlocks({8})
                            .mapToThreads({16})
                            .matchLibraryCalls(true);
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({K, N});
  Check(A, B, mappingOptions);
}

///////////////////////////////////////////////////////////////////////////////
// Batch Matmul tests
//   Z(b, n, k) +=! X(b, n, r_m) * Y(b, r_m, k)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapperBatchMatmulTest, BatchMatmul) {
  auto mappingOptions = tc::CudaMappingOptions::makeNaiveMappingOptions()
                            .tile(1)
                            .mapToThreads({123})
                            .mapToBlocks({50})
                            .usePrivateMemory(true)
                            .useSharedMemory(true);
  mappingOptions.matchLibraryCalls(true);
  at::Tensor A = at::CUDA(at::kFloat).rand({50, 26, 72});
  at::Tensor B = at::CUDA(at::kFloat).rand({50, 72, 26});
  Check(A, B, mappingOptions);
}

///////////////////////////////////////////////////////////////////////////////
// Hadamard
//       Z(b, d) = U(b, d) * V(b, d) * W(b, d)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapperTest, BatchTripleHadamard) {
  at::Tensor U = at::CUDA(at::kFloat).rand({B, D});
  at::Tensor V = at::CUDA(at::kFloat).rand({B, D});
  at::Tensor W = at::CUDA(at::kFloat).rand({B, D});
  std::vector<at::Tensor> inputs = {U, V, W};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def batch_triple_hadamard(float(B, D) U, float(B, D) V, float(B, D) W) -> (Z) {
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
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      checkFun);
}

///////////////////////////////////////////////////////////////////////////////
// TensorDot
//   O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapperTest, TensorDot) {
  N = 32;
  at::Tensor I0 = at::CUDA(at::kFloat).rand({N, C1, C2, H, W});
  at::Tensor I1 = at::CUDA(at::kFloat).rand({N, C2, C3, H, W});
  std::vector<at::Tensor> inputs = {I0, I1};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
    O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC";
  // No defaults for this case
  auto checkFun = [](const std::vector<at::Tensor>& inputs,
                     std::vector<at::Tensor>& outputs) { return true; };
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto name = "tensordot";
  Check(TC, name, options, inputs, checkFun);
  ::benchmarkKernelOptions(TC, name, inputs, options);
}

///////////////////////////////////////////////////////////////////////////////
// Lookup Table
//   O(b, n) +=! LUT(I(b, n), r_r)
///////////////////////////////////////////////////////////////////////////////
TEST_F(TcCudaMapperTest, LUT) {
  const int B = 17, N = 82, R = 22;
  at::Tensor LUT = at::CUDA(at::kFloat).rand({B, R});
  at::Tensor I =
      at::CUDA(at::kFloat).rand({B, N}).mul_(B).floor_().toType(at::kInt);
  std::vector<at::Tensor> inputs = {LUT, I};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def fun(float(B, R) LUT, int32(B, N) I) -> (O) {
  O(b, n) +=! LUT(I(b, n), r_r)
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
      tc::CudaMappingOptions::makeNaiveMappingOptions(),
      inputs,
      checkFun);
}

///////////////////////////////////////////////////////////////////////////////
// SpatialBatchNormalization
///////////////////////////////////////////////////////////////////////////////
// TODO: https://github.com/facebookresearch/TensorComprehensions/issues/319
TEST_F(TcCudaMapperTest, DISABLED_SpatialBatchNormalization) {
  N = 32;
  at::Tensor eps = at::CUDA(at::kFloat).rand({1});
  eps[0] = 1.0f;
  at::Tensor momentum = at::CUDA(at::kFloat).rand({1});
  momentum[0] = 1.0;
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C2, H, W});
  at::Tensor rMeanIn = at::CUDA(at::kFloat).rand({C2});
  at::Tensor rVarIn = at::CUDA(at::kFloat).rand({C2});
  std::vector<at::Tensor> inputs = {momentum, eps, I, rMeanIn, rVarIn};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def spatial_batch_norm(
    float(1) momentum, float(1) eps,
    float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn)
-> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut)
{
    mean(c)    +=!    I(r_n, c, r_h, r_w)
    mean(c)     =  mean(c) / (N * H * W)
    rMeanOut(c) = (1 - momentum(0)) * rMeanIn(c) + momentum(0) * mean(c)
    centered(n, c, h, w) =          I(  n, c,   h,   w) - rMeanOut(c)
    variance(n, c, h, w) =   centered(  n, c,   h,   w) * centered(n, c, h, w)
    expectedVariance(c) +=! (variance(r_n, c, r_h, r_w) + eps(0)) / (N * H * W)
    rVarOut(c) = rsqrt(
        (1 - momentum(0)) * rVarIn(c) +
             momentum(0)  * expectedVariance(c))
    O(n, c, h, w)             = centered(n, c, h, w) * rVarOut(c)
    normalizedOut(n, c, h, w) =        O(n, c, h, w)
})TC";

  auto checkFun = [=](const std::vector<at::Tensor>& inputs,
                      std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    bool training = true;
    at::Tensor weight = at::CUDA(at::kFloat).ones({C2});
    at::Tensor bias = at::CUDA(at::kFloat).zeros({C2});
    auto O = at::batch_norm(
        I,
        weight,
        bias,
        rMeanIn,
        rVarIn,
        training,
        at::Scalar(momentum[0]).toFloat(),
        at::Scalar(eps[0]).toFloat(),
        true);
    auto diff = O.sub(outputs[0]);
    checkRtol(diff, inputs, N * H * W, prec);
  };

  auto name = "spatial_batch_norm";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
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
