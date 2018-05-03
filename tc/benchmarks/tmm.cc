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

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/test_harness.h"
#include "../test/test_harness_aten_cuda.h"
#include "benchmark_fixture.h"

#include "tc/c2/context.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/flags.h"

using namespace caffe2;

DEFINE_uint32(N, 128, "N dimension in C(m, n) += A(m, kk) * B(n, kk)");
DEFINE_uint32(M, 32, "M dimension in C(m, n) += A(m, kk) * B(n, kk)");
DEFINE_uint32(K, 256, "K dimension in C(m, n) += A(m, kk) * B(n, kk)");

class TransposedMatMul : public Benchmark {
 public:
  void runTransposedMatMul(
      uint32_t N,
      uint32_t M,
      uint32_t K,
      const tc::CudaMappingOptions& options,
      bool useFlags = false);
};

void TransposedMatMul::runTransposedMatMul(
    uint32_t N,
    uint32_t M,
    uint32_t K,
    const tc::CudaMappingOptions& options,
    bool useFlags) {
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({N, K});

  auto refOutput = A.mm(B.transpose(0, 1));
  auto checkFun = [&, refOutput](
                      const std::vector<at::Tensor>& inputs,
                      const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    at::Tensor diff = outputs[0].sub(refOutput);
    checkRtol(diff, inputs, M * N, prec);
    return true;
  };

  std::vector<at::Tensor> inputs = {A, B};
  std::string tc = R"TC(
def tmm(float(M,K) A, float(N,K) B) -> (C) {
    C(m, n) +=! A(m, r_k) * B(n, r_k)
}
)TC";

  std::string suffix = std::string("_M_") + std::to_string(FLAGS_M) +
      std::string("_N_") + std::to_string(FLAGS_N) + std::string("_K_") +
      std::to_string(FLAGS_K);
  if (useFlags && FLAGS_validate_proto) {
    validateProto(
        FLAGS_save_tuner_proto_prefix + std::string("/tmm_cache") + suffix,
        tc,
        "tmm",
        inputs,
        checkFun);
  } else {
    std::vector<at::Tensor> outputs;
    Check(tc, "tmm", options, inputs, outputs, checkFun);
    if (useFlags) {
      autotune(
          FLAGS_save_tuner_proto_prefix + std::string("/tmm_cache") + suffix,
          FLAGS_save_tuner_proto_prefix + std::string("/tmm_best") + suffix,
          tc,
          "tmm",
          inputs,
          options,
          {options},
          checkFun);
    }
  }
}

TEST_F(TransposedMatMul, TransposedMatMul) {
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  auto K = FLAGS_K;
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .fixParametersBeforeScheduling(true)
                     .tile(32, 32, 32)
                     .mapToThreads({32, 32})
                     .mapToBlocks({M / 32, N / 32})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(256);
  runTransposedMatMul(N, M, K, options, true);
}

TEST_F(TransposedMatMul, TransposedMatMul_P100_autotuned_M_128_N_1024_K_1024) {
  uint32_t M = 128;
  uint32_t N = 1024;
  uint32_t K = 1024;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(
              tc::FusionStrategy::Preserve3Coincident)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(1, 32)
          .mapToThreads(64, 4)
          .mapToBlocks(256, 32)
          .unroll(256)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(false)
          .unrollCopyShared(true)
          .matchLibraryCalls(true);
  runTransposedMatMul(N, M, K, options);
}

TEST_F(TransposedMatMul, TransposedMatMul_P100_autotuned_M_128_N_256_K_32) {
  uint32_t M = 128;
  uint32_t N = 256;
  uint32_t K = 32;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(
              tc::FusionStrategy::Preserve3Coincident)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(8, 32)
          .mapToThreads(64)
          .mapToBlocks(64, 32, 64)
          .unroll(64)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(false)
          .matchLibraryCalls(false);
  runTransposedMatMul(N, M, K, options);
}

TEST_F(TransposedMatMul, TransposedMatMul_P100_autotuned_M_128_N_16384_K_4096) {
  uint32_t M = 128;
  uint32_t N = 16384;
  uint32_t K = 4096;
  auto options =
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .outerScheduleFusionStrategy(tc::FusionStrategy::Preserve3Coincident)
          .outerScheduleAllowSkewing(false)
          .outerSchedulePositiveOrthant(true)
          .intraTileScheduleFusionStrategy(
              tc::FusionStrategy::Preserve3Coincident)
          .intraTileScheduleAllowSkewing(false)
          .intraTileSchedulePositiveOrthant(true)
          .tile(32, 32, 2)
          .mapToThreads(32)
          .mapToBlocks(4, 128)
          .unroll(8)
          .tileImperfectlyNested(false)
          .useSharedMemory(true)
          .usePrivateMemory(true)
          .unrollCopyShared(false)
          .matchLibraryCalls(false);
  runTransposedMatMul(N, M, K, options);
}

TEST_F(TransposedMatMul, ATenTransposedMatMulReference) {
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  auto K = FLAGS_K;
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({N, K});
  Reference(
      [&]() { return at::mm(A, B.t()); },
      [&](at::Tensor& res) { at::mm_out(res, A, B.t()); });
}

TEST_F(TransposedMatMul, C2TransposedMatMulReference) {
  auto N = FLAGS_N;
  auto M = FLAGS_M;
  auto K = FLAGS_K;

  auto ws_init_func = [&](Workspace& w) {
    auto AddInput =
        TestHarness::AddDeterministicallyRandomInput<float, CUDAContext>;
    AddInput(w, {M, K}, "I");
    AddInput(w, {N, K}, "W");
  };
  OperatorDef op_def =
      TestHarness::ConfigureCUDA("TcMatMulOp", {"I", "W"}, {"O"});
  float precision = 0.0;
  std::unique_ptr<TestHarness::OpTester> reference(
      new TestHarness::OpTester(op_def, precision));
  reference->InitializeReference(ws_init_func, {{"trans_b", 1}});

  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
