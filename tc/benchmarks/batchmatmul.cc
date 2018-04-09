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
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/cuda/test_harness_aten_cuda.h"
#include "../test/test_harness.h"
#include "benchmark_fixture.h"

#include "tc/c2/context.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/flags.h"

using namespace caffe2;

DEFINE_uint32(B, 500, "Batch size in Z(b, n, m) += X(b, n, kk) * Y(b, kk, m)");
DEFINE_uint32(N, 26, "N dimension in Z(b, n, m) += X(b, n, kk) * Y(b, kk, m)");
DEFINE_uint32(M, 72, "M dimension in Z(b, n, m) += X(b, n, kk) * Y(b, kk, m)");
DEFINE_uint32(K, 26, "K dimension in Z(b, n, m) += X(b, n, kk) * Y(b, kk, m)");

class BatchMatMul : public Benchmark {
 public:
  void runBatchMatMul(
      uint32_t B,
      uint32_t N,
      uint32_t M,
      uint32_t K,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
};

void BatchMatMul::runBatchMatMul(
    uint32_t B,
    uint32_t N,
    uint32_t M,
    uint32_t K,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  at::Tensor X = at::CUDA(at::kFloat).rand({B, N, M});
  at::Tensor Y = at::CUDA(at::kFloat).rand({B, M, K});

  auto refOutput = X.bmm(Y);
  auto checkFun = [&, refOutput](
                      const std::vector<at::Tensor>& inputs,
                      const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    at::Tensor diff = outputs[0].sub(refOutput);
    checkRtol(diff, inputs, M, prec);
    return true;
  };

  std::vector<at::Tensor> inputs = {X, Y};
  std::string tc = R"(
def batch_matmul(float(B, N, M) X, float(B, M, K) Y) -> (Z) {
    Z(b, n, k) +=! X(b, n, r_m) * Y(b, r_m, k)
}
)";

  std::string suffix = std::string("_B_") + std::to_string(FLAGS_B) +
      std::string("_K_") + std::to_string(FLAGS_K) + std::string("_M_") +
      std::to_string(FLAGS_M) + std::string("_N_") + std::to_string(FLAGS_N);
  if (useFlags && FLAGS_validate_proto) {
    validateProto(
        FLAGS_save_tuner_proto_prefix + std::string("/batchmatmul_cache") +
            suffix,
        tc,
        "batch_matmul",
        inputs,
        checkFun);
  } else {
    std::vector<at::Tensor> outputs;
    Check(tc, "batch_matmul", options, inputs, outputs, checkFun);
    if (useFlags) {
      autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/batchmatmul_cache") +
              suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/batchmatmul_best") +
              suffix,
          tc,
          "batch_matmul",
          inputs,
          options,
          checkFun);
    }
  }
}

TEST_F(BatchMatMul, TransposedBatchMatMul) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  auto K = FLAGS_K;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1)
                     .mapToThreads({128})
                     .mapToBlocks({B})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(256);
  runBatchMatMul(B, N, M, K, options, true);
}

TEST_F(BatchMatMul, TransposedBatchMatMul_P100_autotuned_B_500_K_26_M_72_N_26) {
  uint32_t B = 500;
  uint32_t K = 26;
  uint32_t M = 72;
  uint32_t N = 26;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
                     .outerScheduleAllowSkewing(false)
                     .outerSchedulePositiveOrthant(true)
                     .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
                     .intraTileScheduleAllowSkewing(false)
                     .intraTileSchedulePositiveOrthant(true)
                     .tile(3)
                     .mapToThreads(4, 36, 3)
                     .mapToBlocks(512)
                     .unroll(64)
                     .tileImperfectlyNested(false)
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unrollCopyShared(true)
                     .matchLibraryCalls(true);
  runBatchMatMul(B, N, M, K, options);
}

TEST_F(BatchMatMul, ATenTransposedBatchMatMulReference) {
  auto B = FLAGS_B;
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  auto K = FLAGS_K;
  at::Tensor X = at::CUDA(at::kFloat).rand({B, N, M});
  at::Tensor Y = at::CUDA(at::kFloat).rand({B, M, K});
  Reference(
      [&]() { return bmm(X, Y); },
      [&](at::Tensor& res) { bmm_out(res, X, Y); });
}

TEST_F(BatchMatMul, C2TransposedBatchMatMulReference) {
  int B = FLAGS_B;
  int N = FLAGS_N;
  int M = FLAGS_M;
  int K = FLAGS_K;

  Workspace w_ref;
  auto AddInput =
      TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
  AddInput(w_ref, {B, N, M}, "X");
  AddInput(w_ref, {B, M, K}, "Y");
  OperatorDef ref_def =
      TestHarness::ConfigureCUDA("BatchMatMul", {"X", "Y"}, {"Z"});
  std::unique_ptr<OperatorBase> net(CreateOperator(ref_def, &w_ref));
  Reference([&]() { return true; }, [&](bool flag) { net->Run(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
