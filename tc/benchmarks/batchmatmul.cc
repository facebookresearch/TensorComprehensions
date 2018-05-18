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
#include "batchmatmul.h"

#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"

#include "../test/caffe2/cuda/test_harness.h"
#include "../test/caffe2/test_harness.h"
#include "../test/test_harness_aten_cuda.h"
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
 protected:
  uint32_t B, N, M, K;

 public:
  void Init(uint32_t b, uint32_t n, uint32_t m, uint32_t k) {
    B = b;
    N = n;
    M = m;
    K = k;
  }
  void runBatchMatMul(const tc::CudaMappingOptions& options);
};

void BatchMatMul::runBatchMatMul(const tc::CudaMappingOptions& options) {
  at::Tensor X = at::CUDA(at::kFloat).rand({B, N, M});
  at::Tensor Y = at::CUDA(at::kFloat).rand({B, M, K});

  auto ref_output = X.bmm(Y);
  auto check_fun = [&, ref_output](
                       const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    at::Tensor diff = outputs[0].sub(ref_output);
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
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/batchmatmul_cache") +
            suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/batchmatmul_best") +
            suffix,
        tc,
        "batch_matmul",
        inputs,
        options,
        check_fun);
  }
  Check(tc, "batch_matmul", bestOptions[0], inputs, check_fun);
}

TEST_F(BatchMatMul, TransposedBatchMatMul) {
  Init(FLAGS_B, FLAGS_N, FLAGS_M, FLAGS_K);
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1)
                     .mapToThreads({128})
                     .mapToBlocks({B})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(256);
  runBatchMatMul(options);
}

TEST_F(BatchMatMul, TransposedBatchMatMul_P100_autotuned_B_500_K_26_M_72_N_26) {
  Init(500, 26, 72, 26);
  runBatchMatMul(
      tc::options_TransposedBatchMatMul_P100_autotuned_B_500_K_26_M_72_N_26);
}

TEST_F(BatchMatMul, ATenTransposedBatchMatMulReference) {
  Init(FLAGS_B, FLAGS_N, FLAGS_M, FLAGS_K);
  at::Tensor X = at::CUDA(at::kFloat).rand({B, N, M});
  at::Tensor Y = at::CUDA(at::kFloat).rand({B, M, K});
  Reference(
      [&]() { return bmm(X, Y); },
      [&](at::Tensor& res) { bmm_out(res, X, Y); });
}

TEST_F(BatchMatMul, C2TransposedBatchMatMulReference) {
  Init(FLAGS_B, FLAGS_N, FLAGS_M, FLAGS_K);
  Workspace w_ref;
  auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
  AddInput(w_ref, {B, N, M}, "X");
  AddInput(w_ref, {B, M, K}, "Y");
  OperatorDef ref_def =
      MakeOperatorDef<caffe2::CUDABackend>("BatchMatMul", {"X", "Y"}, {"Z"});
  std::unique_ptr<OperatorBase> net(CreateOperator(ref_def, &w_ref));
  Reference([&]() { return true; }, [&](bool flag) { net->Run(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
