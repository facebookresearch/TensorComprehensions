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

#include "tc/aten/aten_compiler.h"
#include "tc/aten/utils.h"
#include "tc/autotuner/genetic_autotuner.h"
#include "tc/autotuner/utils.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/canonicalize.h"

using namespace tc;
using namespace autotune;

TEST(DivisorsAndPowers, Default) {
  auto dp = powers2andCeilDivisors(10);
  std::vector<size_t> expected{1, 2, 3, 4, 5, 8, 10, 16};
  ASSERT_EQ(dp, expected);

  dp = powers2andCeilDivisors(72);
  expected = {1, 2, 3, 4, 5, 8, 9, 16, 18, 32, 36, 64, 72, 128};
  ASSERT_EQ(dp, expected);

  dp = powers2andCeilDivisors(35);
  expected = {1, 2, 3, 4, 5, 8, 9, 16, 18, 32, 35, 64};
  ASSERT_EQ(dp, expected);

  dp = powers2andCeilDivisors(130);
  expected = {1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 130, 256};
  ASSERT_EQ(dp, expected);
}

std::vector<CudaMappingOptions> restoreCandidates(
    const std::string& tc,
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs) {
  auto inputsPair = toConstDlpackTensors(inputs);
  auto outputsPair = toConstDlpackTensors(outputs);
  ScopeGuard g([&]() {
    deleteDlmTensors(inputsPair.second);
    deleteDlmTensors(outputsPair.second);
  });

  return tc::autotune::restoreCandidates(
      lang::canonicalTc(tc), inputsPair.first, outputsPair.first);
}

TEST(RestoreCandidates, NoCache) {
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};
  static constexpr auto tc = R"(
      def tc2(float(M,N) A, float(N,K) B) -> (output) {
        output(m, k) +=! A(m, nn) * B(nn, k) + 1
      })";
  ASSERT_THROW(restoreCandidates(tc, inputs, inputs), std::runtime_error);
}

TEST(RestoreCandidates, NotATCid) {
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};
  ASSERT_THROW(restoreCandidates("bla", inputs, inputs), lang::ErrorReport);
}

static constexpr auto tc_ = R"(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
    output(m, k) +=! A(m, r_n) * B(r_n, k)
})";

void EnableCaches() {
  tc::OptionsCache::enableCache();
  tc::OptionsCache::getCache()->clear();
}

TEST(RestoreCandidates, NoRuntimeRecorded) {
  EnableCaches();
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};

  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc_);
  tc::CudaMappingOptions options =
      tc::CudaMappingOptions::makeMlpMappingOptions();
  std::vector<at::Tensor> outputs_;
  auto handle = atCompl.compile("matmul", inputs, options);
  atCompl.run("matmul", inputs, outputs_, handle);

  FLAGS_tuner_gen_restore_number = 1;
  ASSERT_EQ(restoreCandidates(tc_, inputs, outputs_).size(), 0u);
}

TEST(RestoreCandidates, Hit) {
  EnableCaches();
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};

  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc_);
  tc::CudaMappingOptions options =
      tc::CudaMappingOptions::makeMlpMappingOptions();
  std::vector<at::Tensor> outputs_;
  auto handle = atCompl.compile("matmul", inputs, options);
  atCompl.run("matmul", inputs, outputs_, handle, true);

  options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  handle = atCompl.compile("matmul", inputs, options);
  atCompl.run("matmul", inputs, outputs_, handle, true);

  FLAGS_tuner_gen_restore_number = 2;
  auto restored = restoreCandidates(tc_, inputs, outputs_);
  ASSERT_EQ(restored.size(), 2u);

  FLAGS_tuner_gen_restore_number = 1;
  restored = restoreCandidates(tc_, inputs, outputs_);
  ASSERT_EQ(restored.size(), 1u);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
