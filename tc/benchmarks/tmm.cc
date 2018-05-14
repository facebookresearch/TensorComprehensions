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
#include "tmm.h"

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

DEFINE_uint32(N, 128, "N dimension in C(m, n) += A(m, kk) * B(n, kk)");
DEFINE_uint32(M, 32, "M dimension in C(m, n) += A(m, kk) * B(n, kk)");
DEFINE_uint32(K, 256, "K dimension in C(m, n) += A(m, kk) * B(n, kk)");

class TransposedMatMul : public Benchmark {
 protected:
  uint32_t M, N, K;

 public:
  void Init(uint32_t m, uint32_t n, uint32_t k) {
    M = m;
    N = n;
    K = k;
  }
  void runTransposedMatMul(const tc::CudaMappingOptions& options);
};

void TransposedMatMul::runTransposedMatMul(
    const tc::CudaMappingOptions& options) {
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({N, K});

  auto ref_output = A.mm(B.transpose(0, 1));
  auto check_fun = [&, ref_output](
                       const std::vector<at::Tensor>& inputs,
                       const std::vector<at::Tensor>& outputs) {
    TC_CUDA_RUNTIMEAPI_ENFORCE(cudaDeviceSynchronize());
    double prec = 3e-7;
    std::cout << "Checking expected output relative precision @" << prec;
    at::Tensor diff = outputs[0].sub(ref_output);
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
  std::vector<tc::CudaMappingOptions> bestOptions{options};
  if (FLAGS_autotune) {
    bestOptions = autotune(
        FLAGS_save_tuner_proto_prefix + std::string("/tmm_cache") + suffix,
        FLAGS_save_tuner_proto_prefix + std::string("/tmm_best") + suffix,
        tc,
        "tmm",
        inputs,
        options,
        check_fun);
    CHECK_GE(bestOptions.size(), 1u);
  }
  Check(tc, "tmm", bestOptions[0], inputs, check_fun);
}

TEST_F(TransposedMatMul, TransposedMatMul) {
  Init(FLAGS_M, FLAGS_N, FLAGS_K);
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .fixParametersBeforeScheduling(true)
                     .tile(32, 32, 32)
                     .mapToThreads({32, 32})
                     .mapToBlocks({M / 32, N / 32})
                     .useSharedMemory(true)
                     .usePrivateMemory(true)
                     .unroll(256);
  runTransposedMatMul(options);
}

TEST_F(TransposedMatMul, TransposedMatMul_P100_autotuned_M_128_N_1024_K_1024) {
  Init(128, 1024, 1024);
  runTransposedMatMul(
      tc::options_TransposedMatMul_P100_autotuned_M_128_N_1024_K_1024);
}

TEST_F(TransposedMatMul, TransposedMatMul_P100_autotuned_M_128_N_256_K_32) {
  Init(128, 256, 32);
  runTransposedMatMul(
      tc::options_TransposedMatMul_P100_autotuned_M_128_N_256_K_32);
}

TEST_F(TransposedMatMul, TransposedMatMul_P100_autotuned_M_128_N_16384_K_4096) {
  Init(128, 16384, 4096);
  runTransposedMatMul(
      tc::options_TransposedMatMul_P100_autotuned_M_128_N_16384_K_4096);
}

TEST_F(TransposedMatMul, ATenTransposedMatMulReference) {
  Init(FLAGS_M, FLAGS_N, FLAGS_K);
  at::Tensor A = at::CUDA(at::kFloat).rand({M, K});
  at::Tensor B = at::CUDA(at::kFloat).rand({N, K});
  Reference(
      [&]() { return at::mm(A, B.t()); },
      [&](at::Tensor& res) { at::mm_out(res, A, B.t()); });
}

TEST_F(TransposedMatMul, C2TransposedMatMulReference) {
  Init(FLAGS_M, FLAGS_N, FLAGS_K);
  auto ws_init_func = [&](Workspace& w) {
    auto AddInput = AddDeterministicallyRandomInput<caffe2::CUDABackend, float>;
    AddInput(w, {M, K}, "I");
    AddInput(w, {N, K}, "W");
  };
  OperatorDef op_def =
      MakeOperatorDef<caffe2::CUDABackend>("TcMatMulOp", {"I", "W"}, {"O"});
  float precision = 0.0;
  std::unique_ptr<OpTester> reference(new OpTester(op_def, precision));
  reference->InitializeReference(ws_init_func, {{"trans_b", 1}});
  Reference(
      [&]() { return true; }, [&](bool flag) { reference->RunReference(); });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
