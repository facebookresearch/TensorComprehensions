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
#include "tc/aten/aten_autotuner.h"
#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

#include "test_harness_aten_cuda.h"

DEFINE_bool(
    smoke_check,
    true,
    "launches a mini autotune (true) or a full run (false)");
DEFINE_bool(no_memory_promotion, false, "disable memory promotion");

struct ATenCompilationUnitTest : public ::testing::Test {
  static constexpr uint32_t N = 32, C1 = 512, C2 = 8, C3 = 2, H = 28, W = 28;

  ATenCompilationUnitTest() {
    if (FLAGS_smoke_check) {
      // Override some default flags
      tc::FLAGS_tuner_gen_pop_size = 8;
      tc::FLAGS_tuner_gen_generations = 5;
      tc::FLAGS_tuner_threads = std::min(8u, tc::FLAGS_tuner_gen_pop_size);
      tc::FLAGS_tuner_gen_number_elites = tc::FLAGS_tuner_gen_pop_size / 4;
    }
  }

  void Check(
      const std::string& tc,
      const std::string& name,
      const tc::CudaMappingOptions& mappingOptions,
      const std::vector<at::Tensor> inputs) {
    auto pExecutor =
        tc::aten::compile<tc::CudaBackend>(tc, name, inputs, mappingOptions);
    auto outputs = tc::aten::prepareOutputs(tc, name, inputs);
    tc::aten::run(*pExecutor, inputs, outputs);
  }

  tc::CudaMappingOptions autotune(
      const std::string& tc,
      const std::string& name,
      const std::vector<at::Tensor> inputs,
      tc::CudaMappingOptions baseMapping) {
    tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>
        geneticAutotuneATen(tc);
    tc::autotune::TuningParameterFixer fix;
    if (FLAGS_no_memory_promotion) {
      fix.fixUseSharedMemory(false).fixUsePrivateMemory(false);
    }
    auto options = geneticAutotuneATen.tune(name, inputs, {baseMapping}, fix);
    if (options.size() > 0) {
      return options[0];
    }
    LOG(WARNING) << "Autotuner returned no options, returning the baseMapping"
                 << std::endl;
    return baseMapping;
  }
};

TEST_F(ATenCompilationUnitTest, LayerNorm) {
  at::Tensor mat1 = at::CUDA(at::kFloat).rand({7, 32, 64});
  std::vector<at::Tensor> inputs = {mat1};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def layernorm(float(T, B, C) I) -> (O, mean, centered, var) {
        mean(t, b)    +=! I(t, b, c) / C
    centered(t, b, c)  = I(t, b, c) - mean(t, b)

    var(t, b)   +=! centered(t, b, c) * centered(t, b, c)
    var(t, b)    =       var(t, b) / C
      O(t, b, c) =  centered(t, b, c) / rsqrt(var(t, b))
}
  )TC";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto name = "layernorm";
  auto bestOptions = autotune(TC, name, inputs, options);
}

TEST_F(ATenCompilationUnitTest, MatmulA) {
  at::Tensor mat1 = at::CUDA(at::kFloat).rand({3, 4});
  at::Tensor mat2 = at::CUDA(at::kFloat).rand({4, 5});
  std::vector<at::Tensor> inputs = {mat1, mat2};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(m, k) +=! A(m, r_n) * B(r_n, k)
}
  )TC";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto name = "matmul";

  auto bestOptions = autotune(TC, name, inputs, options);
}

TEST_F(ATenCompilationUnitTest, MatmulB) {
  at::Tensor mat1 = at::CUDA(at::kFloat).rand({72, 26});
  at::Tensor mat2 = at::CUDA(at::kFloat).rand({26, 72});
  std::vector<at::Tensor> inputs = {mat1, mat2};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(m, k) +=! A(m, r_n) * B(r_n, k)
}
  )TC";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto name = "matmul";

  auto bestOptions = autotune(TC, name, inputs, options);
}

TEST_F(ATenCompilationUnitTest, MatmulC) {
  at::Tensor mat1 = at::CUDA(at::kFloat).rand({100, 400});
  at::Tensor mat2 = at::CUDA(at::kFloat).rand({400, 500});
  std::vector<at::Tensor> inputs = {mat1, mat2};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
  output(m, k) +=! A(m, r_n) * B(r_n, k)
}
  )TC";
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto name = "matmul";

  auto bestOptions = autotune(TC, name, inputs, options);
}

TEST_F(ATenCompilationUnitTest, TensorDot) {
  at::Tensor I0 = at::CUDA(at::kFloat).rand({N, C1, C2, H, W});
  at::Tensor I1 = at::CUDA(at::kFloat).rand({N, C2, C3, H, W});
  std::vector<at::Tensor> inputs = {I0, I1};

  static constexpr auto TC = R"TC(
def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
  O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC";
  auto options = tc::CudaMappingOptions::makeConvolutionMappingOptions();
  auto name = "tensordot";
  Check(TC, name, options, inputs);
  benchmarkKernelOptions(TC, name, inputs, options);

  std::string suffix = std::string("_N_") + std::to_string(N) +
      std::string("_C1_") + std::to_string(C1) + std::string("_C2_") +
      std::to_string(C2) + std::string("_C3_") + std::to_string(C3) +
      std::string("_H_") + std::to_string(H) + std::string("_W_") +
      std::to_string(W);

  auto bestOptions = autotune(TC, name, inputs, options);
  benchmarkKernelOptions(TC, name, inputs, bestOptions);
}

TEST_F(ATenCompilationUnitTest, BatchNorm) {
  static constexpr uint32_t N = 32, C = 4, H = 56, W = 56;
  at::Tensor I = at::CUDA(at::kFloat).rand({N, C, H, W});
  at::Tensor rMeanIn = at::CUDA(at::kFloat).rand({C});
  at::Tensor rVarIn = at::CUDA(at::kFloat).rand({C});
  std::vector<at::Tensor> inputs = {I, rMeanIn, rVarIn};

  // This may not be correct
  static constexpr auto TC = R"TC(
  def batchnorm(float(N,C,H,W) I, float(C) rMeanIn, float(C) rVarIn) -> (O, rMeanOut, rVarOut, mean, centered, variance, expectedVariance, normalizedOut) {
    mean(c) +=! I(nn, c, hh, ww)
    mean(c)  = mean(c) / (N * H * W)
    rMeanOut(c) = (1 - 0.2) * rMeanIn(c) + 0.2 * mean(c)
    centered(n, c, h, w) =        I(n, c, h, w) - rMeanOut(c)
    variance(n, c, h, w) = centered(n, c, h, w) * centered(n, c, h, w)
    expectedVariance(c) +=! (variance(n, c, h, w) + 0.001) / (N * H * W)
    rVarOut(c) = rsqrt((1 - 0.2) * rVarIn(c) + 0.2 * expectedVariance(c))
    O(n, c, h, w) = centered(n, c, h, w) / rVarOut(c)
    normalizedOut(n, c, h, w) = O(n, c, h, w)
  }
)TC";

  auto name = "batchnorm";
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();

  std::string cacheFilename = "";
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});
}

TEST_F(ATenCompilationUnitTest, GroupConvolution) {
  uint32_t N = 32, G = 32, C = 16, F = 16;
  uint32_t W = 14, H = 14, KW = 3, KH = 3;
  at::Tensor I = at::CUDA(at::kFloat).rand({N, G, C, H, W});
  at::Tensor W1 = at::CUDA(at::kFloat).rand({G, F, C, KH, KW});
  at::Tensor B = at::CUDA(at::kFloat).rand({G, F});
  std::vector<at::Tensor> inputs = {I, W1, B};

  static constexpr auto TC = R"(
def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)";

  auto name = "group_convolution";
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();

  std::string cacheFilename = "";
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
