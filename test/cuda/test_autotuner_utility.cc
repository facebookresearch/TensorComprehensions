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
#include "tc/autotuner/autotuner.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/autotuner/utils.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/scope_guard.h"
#include "tc/lang/canonicalize.h"

using namespace tc;
using namespace tc::aten;
using namespace tc::autotune;
using CudaOptionsCache =
    Autotuner<CudaBackend, GeneticSearch>::OptionsCacheType;

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
    CudaOptionsCache& optionsCache,
    const std::string& tc,
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs) {
  auto inputDLTensors = makeDLConstTensors(inputs);
  auto outputDLTensors = makeDLTensors(outputs);
  return optionsCache.getTopKOptions(
      lang::canonicalTc(tc),
      makeTensorInfoVector(extractRawPtrs(inputDLTensors)),
      makeTensorInfoVector(extractRawPtrs(outputDLTensors)),
      CudaGPUInfo::GPUInfo().getCudaDeviceStr(),
      FLAGS_tuner_gen_restore_number);
}

TEST(RestoreCandidates, NotATCid) {
  CudaOptionsCache optionsCache;
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};
  ASSERT_THROW(
      restoreCandidates(optionsCache, "bla", inputs, inputs),
      lang::ErrorReport);
}

static constexpr auto tc_ = R"(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
    output(m, k) +=! A(m, r_n) * B(r_n, k)
})";

TEST(RestoreCandidates, NoRuntimeRecorded) {
  CudaOptionsCache optionsCache;
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};
  auto options = CudaMappingOptions::makeMlpMappingOptions();
  auto pExecutor = compile<CudaBackend>(tc_, "matmul", inputs, options);
  std::vector<at::Tensor> outputs = prepareOutputs(tc_, "matmul", inputs);
  run(*pExecutor, inputs, outputs);

  FLAGS_tuner_gen_restore_number = 1;
  ASSERT_EQ(restoreCandidates(optionsCache, tc_, inputs, outputs).size(), 0u);
}

TEST(RestoreCandidates, Hit) {
  CudaOptionsCache optionsCache;
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({10, 16}),
                                 at::CUDA(at::kFloat).rand({16, 20})};
  auto options = CudaMappingOptions::makeMlpMappingOptions();
  auto pExecutor = compile<CudaBackend>(tc_, "matmul", inputs, options);
  std::vector<at::Tensor> outputs = prepareOutputs(tc_, "matmul", inputs);
  auto timings = profile(*pExecutor, inputs, outputs);

  auto inputDLTensors = makeDLConstTensors(inputs);
  auto outputDLTensors = makeDLTensors(outputs);
  optionsCache.recordRuntime(
      lang::canonicalTc(tc_),
      makeTensorInfoVector(extractRawPtrs(inputDLTensors)),
      makeTensorInfoVector(extractRawPtrs(outputDLTensors)),
      CudaGPUInfo::GPUInfo().getCudaDeviceStr(),
      options,
      timings.kernelRuntime);

  {
    options = CudaMappingOptions::makeNaiveMappingOptions();
    auto pExecutor = compile<CudaBackend>(tc_, "matmul", inputs, options);
    auto timings = profile(*pExecutor, inputs, outputs);
    optionsCache.recordRuntime(
        lang::canonicalTc(tc_),
        makeTensorInfoVector(extractRawPtrs(inputDLTensors)),
        makeTensorInfoVector(extractRawPtrs(outputDLTensors)),
        CudaGPUInfo::GPUInfo().getCudaDeviceStr(),
        options,
        timings.kernelRuntime);
  }

  FLAGS_tuner_gen_restore_number = 2;
  auto restored = restoreCandidates(optionsCache, tc_, inputs, outputs);
  ASSERT_EQ(restored.size(), 2u);

  FLAGS_tuner_gen_restore_number = 1;
  restored = restoreCandidates(optionsCache, tc_, inputs, outputs);
  ASSERT_EQ(restored.size(), 1u);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
