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
#include "tc/autotuner/genetic_autotuner_aten.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

#include "test_harness_aten_cuda.h"

DEFINE_bool(
    smoke_check,
    true,
    "launches a mini autotune (true) or a full run (false)");
DEFINE_string(save_tuner_proto_prefix, "/tmp", "Enable autotuning");
DEFINE_bool(
    load_or_store_cache,
    false,
    "load options from previously stored cache (or store them)");
DEFINE_bool(no_memory_promotion, false, "disable memory promotion");

struct ATenCompilationUnitTest : public ::testing::Test {
  static constexpr uint32_t N = 32, C1 = 512, C2 = 8, C3 = 2, H = 28, W = 28;

  ATenCompilationUnitTest() {
    if (FLAGS_smoke_check) {
      // Override some default flags
      tc::FLAGS_tuner_gen_pop_size = 8;
      tc::FLAGS_tuner_gen_generations = 5;
      tc::FLAGS_tuner_threads = std::min(8u, tc::FLAGS_tuner_gen_pop_size);
      tc::FLAGS_tuner_gen_mating_pool_size = tc::FLAGS_tuner_gen_pop_size * 3;
      tc::FLAGS_tuner_gen_selection_pool_size =
          tc::FLAGS_tuner_gen_pop_size * 3;
    }
  }

  void Check(
      const std::string& tc,
      const std::string& name,
      const tc::CudaMappingOptions& mappingOptions,
      const std::vector<at::Tensor> inputs,
      std::vector<at::Tensor>& outputs) {
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(tc);
    auto handle = atCompl.compile(name, inputs, mappingOptions);
    atCompl.run(name, inputs, outputs, handle);
  }

  tc::CudaMappingOptions autotune(
      const std::string& cacheFilename,
      const std::string& tc,
      const std::string& name,
      const std::vector<at::Tensor> inputs,
      tc::CudaMappingOptions baseMapping,
      std::vector<tc::CudaMappingOptions> startingPoints) {
    tc::autotune::GeneticAutotunerATen geneticAutotuneATen(tc);
    tc::autotune::TuningParameterFixer fix;
    if (FLAGS_no_memory_promotion) {
      fix.fixUseSharedMemory(false).fixUsePrivateMemory(false);
    }
    auto options = geneticAutotuneATen.tune(
        cacheFilename, name, inputs, baseMapping, startingPoints, fix);
    if (options) {
      return *options;
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
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto name = "layernorm";

  std::string cacheFilename = "";
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});
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
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto name = "matmul";

  std::string cacheFilename = "";
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});
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
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto name = "matmul";

  std::string cacheFilename = "";
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});
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
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto name = "matmul";

  std::string cacheFilename = "";
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});
}

TEST_F(ATenCompilationUnitTest, TensorDot) {
  at::Tensor I0 = at::CUDA(at::kFloat).rand({N, C1, C2, H, W});
  at::Tensor I1 = at::CUDA(at::kFloat).rand({N, C2, C3, H, W});
  std::vector<at::Tensor> inputs = {I0, I1};
  std::vector<at::Tensor> outputs;

  static constexpr auto TC = R"TC(
def tensordot(float(N, C1, C2, H, W) I0, float(N, C2, C3, H, W) I1) -> (O) {
  O(n, c1, c3, h, w) +=! I0(n, c1, r_c2, h, w) * I1(n, r_c2, c3, h, w)
}
  )TC";
  auto options = tc::CudaMappingOptions::makeConvolutionCudaMappingOptions();
  auto name = "tensordot";
  Check(TC, name, options, inputs, outputs);
  benchmarkKernelOptions(TC, name, inputs, options);

  std::string suffix = std::string("_N_") + std::to_string(N) +
      std::string("_C1_") + std::to_string(C1) + std::string("_C2_") +
      std::to_string(C2) + std::string("_C3_") + std::to_string(C3) +
      std::string("_H_") + std::to_string(H) + std::string("_W_") +
      std::to_string(W);

  std::string cacheFilename = "";
  if (FLAGS_load_or_store_cache) {
    cacheFilename = FLAGS_save_tuner_proto_prefix +
        std::string("/tensordot_cache") + suffix;
  }
  auto bestOptions =
      autotune(cacheFilename, TC, name, inputs, options, {options});

  benchmarkKernelOptions(TC, name, inputs, bestOptions);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
