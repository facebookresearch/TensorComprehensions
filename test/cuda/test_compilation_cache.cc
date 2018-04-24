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
#include <future>

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"

#include "test_harness_aten_cuda.h"

class CudaCacheTest : public ::testing::Test {
 protected:
  void SetUp() {
    tc::CudaCache::enableCache();
    ASSERT_TRUE(tc::CudaCache::cacheEnabled());
    tc::CudaCache::getCache()->clear();
    ASSERT_EQ(tc::CudaCache::getCache()->size(), 0u);
    ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 0);
    ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 0);

    inputs.resize(3);
    for (auto& input : inputs) {
      input.ndim = 2;
      input.shape = new int64_t[2];
      input.shape[0] = 5;
      input.shape[1] = 6;
      input.strides = nullptr;
    }
    inputs[1].ndim = 0;
    inputs[2].ndim = 0;
  }

  void TearDown() {
    tc::CudaCache::disableCache();
    ASSERT_FALSE(tc::CudaCache::cacheEnabled());
    for (auto& input : inputs) {
      delete[] input.shape;
    }
  }
  std::vector<DLTensor> inputs;

  std::vector<const DLTensor*> InputPtrs() const {
    std::vector<const DLTensor*> ptrs;
    for (const auto& input : inputs) {
      ptrs.push_back(&input);
    }
    return ptrs;
  }
};

TEST_F(CudaCacheTest, EntrySameKeyDifferentValue) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel0",
      "ker000",
      {0, 0, 1},
      tc::Grid(std::vector<size_t>{1, 1, 1}),
      tc::Block(std::vector<size_t>{1, 1, 1}),
      options,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  ASSERT_THROW(
      tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
          "kernel0",
          "ker000",
          {0, 0, 1},
          tc::Grid(std::vector<size_t>{2, 1, 1}),
          tc::Block(std::vector<size_t>{1, 2, 1}),
          options,
          inputPtrs,
          outputPtrs,
          "source1",
          tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr())),
      tc::CacheEntrySameKeyDifferentValue);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 1u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);
}

TEST_F(CudaCacheTest, DifferentIDs) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel0",
      "ker000",
      {0, 0, 1},
      tc::Grid(std::vector<size_t>{1, 1, 1}),
      tc::Block(std::vector<size_t>{1, 1, 1}),
      options,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel1",
      "ker111",
      {1, 1, 0},
      tc::Grid(std::vector<size_t>{2, 1, 1}),
      tc::Block(std::vector<size_t>{1, 2, 1}),
      options,
      inputPtrs,
      outputPtrs,
      "source1",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  auto ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel0", options, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source0");
  ASSERT_EQ(ret->grid, tc::Grid({1, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 1, 1}));
  ASSERT_EQ(ret->specializedName, "ker000");
  {
    auto params = std::vector<int>{0, 0, 1};
    ASSERT_EQ(ret->parameters, params);
  }

  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel1", options, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source1");
  ASSERT_EQ(ret->grid, tc::Grid({2, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 2, 1}));
  ASSERT_EQ(ret->specializedName, "ker111");
  {
    auto params = std::vector<int>{1, 1, 0};
    ASSERT_EQ(ret->parameters, params);
  }

  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel2", options, inputPtrs, outputPtrs);
  ASSERT_FALSE(ret);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);
}

TEST_F(CudaCacheTest, DifferentOptions) {
  auto options0 = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options0,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  auto options1 = tc::CudaMappingOptions::makeMlpCudaMappingOptions();
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel",
      "",
      {},
      tc::Grid(std::vector<size_t>{{2, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 2, 1}}),
      options1,
      inputPtrs,
      outputPtrs,
      "source1",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  auto ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel", options0, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source0");
  ASSERT_EQ(ret->grid, tc::Grid({1, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 1, 1}));

  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel", options1, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source1");
  ASSERT_EQ(ret->grid, tc::Grid({2, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 2, 1}));

  auto options2 = tc::CudaMappingOptions::makeConvolutionCudaMappingOptions();
  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel", options2, inputPtrs, outputPtrs);
  ASSERT_FALSE(ret);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);
}

TEST_F(CudaCacheTest, DifferentInputs) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  auto s = inputs[0].shape[0];
  inputs[0].shape[0] = 42;
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel",
      "",
      {},
      tc::Grid(std::vector<size_t>{{2, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 2, 1}}),
      options,
      inputPtrs,
      outputPtrs,
      "source1",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  inputs[0].shape[0] = s;
  auto ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel", options, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source0");
  ASSERT_EQ(ret->grid, tc::Grid({1, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 1, 1}));

  inputs[0].shape[0] = 42;
  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel", options, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source1");
  ASSERT_EQ(ret->grid, tc::Grid({2, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 2, 1}));

  inputs[0].shape[0] = 44;
  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel", options, inputPtrs, outputPtrs);
  ASSERT_FALSE(ret);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);
}

TEST_F(CudaCacheTest, DoubleInsertion) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  EXPECT_THROW(
      tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
          "kernel",
          "",
          {},
          tc::Grid(std::vector<size_t>{{1, 1, 1}}),
          tc::Block(std::vector<size_t>{{1, 1, 1}}),
          options,
          inputPtrs,
          outputPtrs,
          "source1",
          tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr())),
      tc::CacheEntrySameKeyDifferentValue);

  EXPECT_THROW(
      tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
          "kernel",
          "",
          {},
          tc::Grid(std::vector<size_t>{{2, 1, 1}}),
          tc::Block(std::vector<size_t>{{1, 1, 1}}),
          options,
          inputPtrs,
          outputPtrs,
          "source0",
          tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr())),
      tc::CacheEntrySameKeyDifferentValue);

  EXPECT_THROW(
      tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
          "kernel",
          "",
          {},
          tc::Grid(std::vector<size_t>{{1, 1, 1}}),
          tc::Block(std::vector<size_t>{{1, 2, 1}}),
          options,
          inputPtrs,
          outputPtrs,
          "source0",
          tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr())),
      tc::CacheEntrySameKeyDifferentValue);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 1u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 4);
}

TEST_F(CudaCacheTest, Serialization) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel0",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel1",
      "",
      {},
      tc::Grid(std::vector<size_t>{{2, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 2, 1}}),
      options,
      inputPtrs,
      outputPtrs,
      "source1",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  auto buf = tc::CudaCache::getCache()->toProtobuf();

  tc::CudaCache::loadCacheFromProtobuf(buf);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 0);

  auto ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel0", options, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source0");
  ASSERT_EQ(ret->grid, tc::Grid({1, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 1, 1}));

  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel1", options, inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(ret->source, "source1");
  ASSERT_EQ(ret->grid, tc::Grid({2, 1, 1}));
  ASSERT_EQ(ret->block, tc::Block({1, 2, 1}));

  ret = tc::CudaCache::getCache()->retrieveKernel(
      "kernel2", options, inputPtrs, outputPtrs);
  ASSERT_FALSE(ret);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 0);
}

class OptionsCacheTest : public ::testing::Test {
 protected:
  void SetUp() {
    tc::OptionsCache::enableCache();
    ASSERT_TRUE(tc::OptionsCache::cacheEnabled());
    tc::OptionsCache::getCache()->clear();
    ASSERT_EQ(tc::OptionsCache::getCache()->size(), 0u);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 0);

    inputs.resize(3);
    for (auto& input : inputs) {
      input.ndim = 2;
      input.shape = new int64_t[2];
      input.shape[0] = 5;
      input.shape[1] = 6;
      input.strides = nullptr;
    }
    inputs[1].ndim = 0;
    inputs[2].ndim = 0;
  }

  void TearDown() {
    tc::OptionsCache::disableCache();
    ASSERT_FALSE(tc::OptionsCache::cacheEnabled());
    for (auto& input : inputs) {
      delete[] input.shape;
    }
  }
  std::vector<DLTensor> inputs;

  std::vector<const DLTensor*> InputPtrs() const {
    std::vector<const DLTensor*> ptrs;
    for (const auto& input : inputs) {
      ptrs.push_back(&input);
    }
    return ptrs;
  }
};

TEST_F(OptionsCacheTest, DifferentIDs) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options, inputPtrs, outputPtrs, std::chrono::microseconds(10));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options, inputPtrs, outputPtrs, std::chrono::microseconds(11));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel1", options, inputPtrs, outputPtrs, std::chrono::microseconds(1));

  auto ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel0", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0].options, options);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 2u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(10));
  ASSERT_EQ(ret[0].recordedRuntimes[1], std::chrono::microseconds(11));

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel1", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0].options, options);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 1u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(1));

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel2", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 0u);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 3);
}

TEST_F(OptionsCacheTest, DifferentOptions) {
  auto options0 = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto options1 = tc::CudaMappingOptions::makeMlpCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options0, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options1, inputPtrs, outputPtrs, std::chrono::microseconds(2));

  auto ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel", inputPtrs, outputPtrs);

  ASSERT_EQ(ret.size(), 2u);
  ASSERT_EQ(ret[0].options, options0);
  ASSERT_EQ(ret[1].options, options1);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 1u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(1));
  ASSERT_EQ(ret[1].recordedRuntimes.size(), 1u);
  ASSERT_EQ(ret[1].recordedRuntimes[0], std::chrono::microseconds(2));

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 1u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 1);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 1);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 2);
}

TEST_F(OptionsCacheTest, DifferentInputs) {
  auto options = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options, inputPtrs, outputPtrs, std::chrono::microseconds(2));

  auto s = inputs[0].shape[0];
  inputs[0].shape[0] = 42;

  auto options_ =
      tc::CudaMappingOptions::makeGroupConvolutionCudaMappingOptions();
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options_, inputPtrs, outputPtrs, std::chrono::microseconds(3));

  inputs[0].shape[0] = s;
  auto ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel", inputPtrs, outputPtrs);

  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0].options, options);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 2u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(1));
  ASSERT_EQ(ret[0].recordedRuntimes[1], std::chrono::microseconds(2));

  inputs[0].shape[0] = 42;
  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel", inputPtrs, outputPtrs);

  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0].options, options_);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 1u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(3));

  inputs[0].shape[0] = 43;
  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 0u);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 3);
}

TEST_F(OptionsCacheTest, RetrieveBest) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({3});

  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options0, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options1, inputPtrs, outputPtrs, std::chrono::microseconds(2));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options2, inputPtrs, outputPtrs, std::chrono::microseconds(3));

  auto ret = tc::OptionsCache::getCache()->retrieveBestOptions(
      "kernel", inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(*ret, options0);

  ret = tc::OptionsCache::getCache()->retrieveBestOptions(
      "kernelX", inputPtrs, outputPtrs);
  ASSERT_FALSE(ret);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 1u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 3u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 2);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 1);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 3);
}

TEST_F(OptionsCacheTest, RetrieveTopK) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({3});
  auto options3 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({4});
  auto options4 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({5});

  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options0, inputPtrs, outputPtrs, std::chrono::microseconds(3));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options1, inputPtrs, outputPtrs, std::chrono::microseconds(2));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options2, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options3, inputPtrs, outputPtrs, std::chrono::microseconds(4));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options4, inputPtrs, outputPtrs, std::chrono::microseconds(5));

  auto ret = tc::OptionsCache::getCache()->retrieveTopKOptions(
      "kernelX", inputPtrs, outputPtrs, 3);
  ASSERT_EQ(ret.size(), 0u);

  ret = tc::OptionsCache::getCache()->retrieveTopKOptions(
      "kernel", inputPtrs, outputPtrs, 3);
  ASSERT_EQ(ret.size(), 3u);
  ASSERT_EQ(ret[0], options2);
  ASSERT_EQ(ret[1], options1);
  ASSERT_EQ(ret[2], options0);

  ret = tc::OptionsCache::getCache()->retrieveTopKOptions(
      "kernel", inputPtrs, outputPtrs, 6);
  ASSERT_EQ(ret.size(), 5u);
  ASSERT_EQ(ret[0], options2);
  ASSERT_EQ(ret[1], options1);
  ASSERT_EQ(ret[2], options0);
  ASSERT_EQ(ret[3], options3);
  ASSERT_EQ(ret[4], options4);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 1u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 5u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 3);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 5);
}

TEST_F(OptionsCacheTest, KeepOnlyBestCandidates) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({3});
  auto options3 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({4});
  auto options4 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({5});

  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options0, inputPtrs, outputPtrs, std::chrono::microseconds(3));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options1, inputPtrs, outputPtrs, std::chrono::microseconds(2));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options2, inputPtrs, outputPtrs, std::chrono::microseconds(4));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel1", options2, inputPtrs, outputPtrs, std::chrono::microseconds(4));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel1", options3, inputPtrs, outputPtrs, std::chrono::microseconds(1));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel2", options4, inputPtrs, outputPtrs, std::chrono::microseconds(5));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options0, inputPtrs, outputPtrs, std::chrono::microseconds(2));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options1, inputPtrs, outputPtrs, std::chrono::microseconds(6));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options3, inputPtrs, outputPtrs, std::chrono::microseconds(5));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options4, inputPtrs, outputPtrs, std::chrono::microseconds(1));

  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 10u);
  tc::OptionsCache::getCache()->keepOnlyBestCandidates(2);

  auto ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel0", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 2u);
  ASSERT_EQ(ret[0].options, options1);
  ASSERT_EQ(ret[1].options, options0);

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel1", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 2u);
  ASSERT_EQ(ret[0].options, options3);
  ASSERT_EQ(ret[1].options, options2);

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel2", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0].options, options4);

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel3", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 2u);
  ASSERT_EQ(ret[0].options, options4);
  ASSERT_EQ(ret[1].options, options0);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 7u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 4);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 10);
}

TEST_F(OptionsCacheTest, RetrieveBestMedianTime) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({2});

  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options0, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options0, inputPtrs, outputPtrs, std::chrono::microseconds(9));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options0, inputPtrs, outputPtrs, std::chrono::microseconds(10));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options1, inputPtrs, outputPtrs, std::chrono::microseconds(8));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options1, inputPtrs, outputPtrs, std::chrono::microseconds(8));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel", options1, inputPtrs, outputPtrs, std::chrono::microseconds(8));

  auto ret = tc::OptionsCache::getCache()->retrieveBestOptions(
      "kernel", inputPtrs, outputPtrs);
  ASSERT_TRUE(ret);
  ASSERT_EQ(*ret, options1);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 1u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 1);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 1);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 6);
}

TEST_F(OptionsCacheTest, Serialization) {
  auto options0 = tc::CudaMappingOptions::makeNaiveCudaMappingOptions().tile(0);
  auto options1 = tc::CudaMappingOptions::makeNaiveCudaMappingOptions().tile(1);
  auto inputPtrs = InputPtrs();
  auto outputPtrs = InputPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0",
      options0,
      inputPtrs,
      outputPtrs,
      std::chrono::microseconds(10));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0",
      options1,
      inputPtrs,
      outputPtrs,
      std::chrono::microseconds(11));
  tc::CudaProfilingInfo pInfoOrig;
  pInfoOrig.runtime = std::chrono::microseconds(444);
  pInfoOrig.ipc = 1.23;
  pInfoOrig.globalLoadEfficiency = 6.546;
  pInfoOrig.globalStoreEfficiency = 7.123;
  pInfoOrig.sharedMemoryEfficiency = 7.111111;
  pInfoOrig.localMemoryOverhead = 888.222;
  pInfoOrig.achievedOccupancy = 11231231.14;
  pInfoOrig.warpExecutionEfficiency = 910293123918239.1029381;

  tc::OptionsCache::getCache()->recordProfilingInfo(
      "kernel0", options0, inputPtrs, outputPtrs, pInfoOrig);
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel1", options0, inputPtrs, outputPtrs, std::chrono::microseconds(1));

  auto buf = tc::OptionsCache::getCache()->toProtobuf();
  tc::OptionsCache::loadCacheFromProtobuf(buf);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 3u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 0);

  auto ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel0", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 2u);
  ASSERT_EQ(ret[0].options, options0);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 2u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(10));
  ASSERT_EQ(ret[0].recordedRuntimes[1], std::chrono::microseconds(444));

  ASSERT_EQ(ret[1].options, options1);
  ASSERT_EQ(ret[1].recordedRuntimes.size(), 1u);
  ASSERT_EQ(ret[1].recordedRuntimes[0], std::chrono::microseconds(11));

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel1", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0].options, options0);
  ASSERT_EQ(ret[0].recordedRuntimes.size(), 1u);
  ASSERT_EQ(ret[0].recordedRuntimes[0], std::chrono::microseconds(1));

  ret = tc::OptionsCache::getCache()->retrieveOptionsAndRuntimes(
      "kernel2", inputPtrs, outputPtrs);
  ASSERT_EQ(ret.size(), 0u);


  auto ret2 = tc::OptionsCache::getCache()->retrieveOptionsAndProfilingInfo(
      "kernel0", inputPtrs, outputPtrs);
  ASSERT_EQ(ret2.size(), 1u);
  ASSERT_EQ(ret2.front().options, options0);
  ASSERT_EQ(ret2.front().profilingInfo.size(), 1u);
  ASSERT_EQ(ret2.front().profilingInfo.front(), pInfoOrig);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 3u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 3);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 0);
}

TEST(
    CudaAndOptionsCacheInteraction,
    RemoveFromCudaCacheEntriesNotInOptionsCache) {
  tc::OptionsCache::enableCache();
  tc::OptionsCache::getCache()->clear();
  tc::CudaCache::enableCache();
  tc::CudaCache::getCache()->clear();

  auto options0 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({3});
  auto options3 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({4});
  auto options4 =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions().mapToBlocks({5});

  std::vector<DLTensor> inputs;
  inputs.resize(3);
  for (auto& input : inputs) {
    input.ndim = 2;
    input.shape = new int64_t[2];
    input.shape[0] = 5;
    input.shape[1] = 6;
    input.strides = nullptr;
  }
  inputs[1].ndim = 0;
  inputs[2].ndim = 0;

  tc::ScopeGuard g([&]() {
    for (auto& input : inputs) {
      delete[] input.shape;
    }
  });

  auto toPtrs = [&]() {
    std::vector<const DLTensor*> ptrs;
    for (const auto& input : inputs) {
      ptrs.push_back(&input);
    }
    return ptrs;
  };

  auto inputPtrs = toPtrs();
  auto outputPtrs = toPtrs();

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options0, inputPtrs, outputPtrs, std::chrono::microseconds(3));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel0",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options0,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options1, inputPtrs, outputPtrs, std::chrono::microseconds(2));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel0",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options1,
      inputPtrs,
      outputPtrs,
      "source1",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel0", options2, inputPtrs, outputPtrs, std::chrono::microseconds(4));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel0",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options2,
      inputPtrs,
      outputPtrs,
      "source2",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel1", options2, inputPtrs, outputPtrs, std::chrono::microseconds(4));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel1",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options2,
      inputPtrs,
      outputPtrs,
      "source2",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel1", options3, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel1",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options3,
      inputPtrs,
      outputPtrs,
      "source3",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel2", options4, inputPtrs, outputPtrs, std::chrono::microseconds(5));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel2",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options4,
      inputPtrs,
      outputPtrs,
      "source4",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options0, inputPtrs, outputPtrs, std::chrono::microseconds(2));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel3",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options0,
      inputPtrs,
      outputPtrs,
      "source0",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options1, inputPtrs, outputPtrs, std::chrono::microseconds(6));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel3",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options1,
      inputPtrs,
      outputPtrs,
      "source1",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options3, inputPtrs, outputPtrs, std::chrono::microseconds(5));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel3",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options3,
      inputPtrs,
      outputPtrs,
      "source3",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  tc::OptionsCache::getCache()->recordRuntime(
      "kernel3", options4, inputPtrs, outputPtrs, std::chrono::microseconds(1));
  tc::CudaCache::getCache()->cacheKernel(tc::CudaCachedEntry(
      "kernel3",
      "",
      {},
      tc::Grid(std::vector<size_t>{{1, 1, 1}}),
      tc::Block(std::vector<size_t>{{1, 1, 1}}),
      options4,
      inputPtrs,
      outputPtrs,
      "source4",
      tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));

  tc::OptionsCache::getCache()->keepOnlyBestCandidates(1);
  tc::removeFromCudaCacheEntriesNotInOptionsCache(
      *tc::CudaCache::getCache(), *tc::OptionsCache::getCache());

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 4u);

  ASSERT_FALSE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel0", options0, inputPtrs, outputPtrs));
  ASSERT_TRUE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel0", options1, inputPtrs, outputPtrs));
  ASSERT_FALSE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel0", options2, inputPtrs, outputPtrs));

  ASSERT_FALSE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel1", options2, inputPtrs, outputPtrs));
  ASSERT_TRUE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel1", options3, inputPtrs, outputPtrs));

  ASSERT_TRUE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel2", options4, inputPtrs, outputPtrs));

  ASSERT_FALSE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel3", options0, inputPtrs, outputPtrs));
  ASSERT_FALSE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel3", options1, inputPtrs, outputPtrs));
  ASSERT_FALSE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel3", options3, inputPtrs, outputPtrs));
  ASSERT_TRUE(tc::CudaCache::getCache()->retrieveKernel(
      "kernel3", options4, inputPtrs, outputPtrs));
}

/*
 *class FCReluTester {
 * public:
 *  FCReluTester(int B, int M, int N)
 *      : inputs_{at::CUDA(at::kFloat).rand({B, M}),
 *                at::CUDA(at::kFloat).rand({N, M}),
 *                at::CUDA(at::kFloat).rand({N})},
 *        M{M} {}
 *  void Run() {
 *    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
 *    atCompl.define(tc_);
 *    std::vector<at::Tensor> outputs_;
 *    atCompl.run(
 *        "fcrelu",
 *        inputs_,
 *        outputs_,
 *        tc::CudaMappingOptions::makeMlpCudaMappingOptions(), true);
 *    at::Tensor diff =
 *        outputs_[0].sub(inputs_[0].mm(inputs_[1]).add(inputs_[2]).clamp_min(0));
 *    checkRtol(diff, inputs_, M);
 *  }
 *
 * private:
 *  std::vector<at::Tensor> inputs_;
 *  int M;
 *  static constexpr auto tc_ = R"(
 *  def fcrelu(float(B,M) I, float(N,M) W1, float(N) B1) -> (O1) {
 *      O1(b, n) += I(b, m) * W1(n, m)
 *      O1(b, n) = O1(b, n) + B1(n)
 *      O1(b, n) = fmax(O1(b, n), 0)
 *    })";
 *};
 */

class MatMulTester {
 public:
  MatMulTester(int N, int M, int B)
      : inputs_{at::CUDA(at::kFloat).rand({N, M}),
                at::CUDA(at::kFloat).rand({M, B})},
        M{M} {}
  void Run(
      tc::CudaMappingOptions options =
          tc::CudaMappingOptions::makeMlpCudaMappingOptions()) {
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(tc_);
    std::vector<at::Tensor> outputs_;
    auto handle = atCompl.compile("matmul", inputs_, options);
    atCompl.run("matmul", inputs_, outputs_, handle, true);
    at::Tensor diff = outputs_[0].sub(inputs_[0].mm(inputs_[1]));
    checkRtol(diff, inputs_, M, 1e-6);
  }

 private:
  std::vector<at::Tensor> inputs_;
  int M;
  static constexpr auto tc_ = R"(
def matmul(float(M,N) A, float(N,K) B) -> (output) {
    output(m, k) +=! A(m, r_n) * B(r_n, k)
})";
};

class ConvolutionTester {
 public:
  ConvolutionTester(int N, int C, int H, int W, int O, int KH, int KW)
      : inputs_{at::CUDA(at::kFloat).rand({N, C, H, W}),
                at::CUDA(at::kFloat).rand({O, C, KH, KW}),
                at::CUDA(at::kFloat).rand({O})},
        C{C},
        KH{KH},
        KW{KW} {}
  void Run(
      tc::CudaMappingOptions options =
          tc::CudaMappingOptions::makeConvolutionCudaMappingOptions()) {
    tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
    atCompl.define(tc_);
    std::vector<at::Tensor> outputs_;
    auto handle = atCompl.compile("convolution", inputs_, options);
    atCompl.run("convolution", inputs_, outputs_, handle, true);

    at::Tensor expected = at::conv2d(inputs_[0], inputs_[1], inputs_[2]);
    at::Tensor diff = outputs_[1].sub(expected);
    checkRtol(diff, inputs_, C * KW * KH, 1e-6);
  }

 private:
  std::vector<at::Tensor> inputs_;
  int C;
  int KH;
  int KW;
  static constexpr auto tc_ = R"(
def convolution(float(N,C,H,W) I, float(O,C,KH,KW) W1, float(O) B)
-> (tmp, O1)
{
    tmp(n, o, h, w) +=!  I(n, r_c, h + r_kh, w + r_kw) * W1(o, r_c, r_kh, r_kw)
     O1(n, o, h, w)  = tmp(n, o, h, w) + B(o)
})";
};

class CompilationCacheTest : public ::testing::Test {
 protected:
  void SetUp() {
    tc::CudaCache::enableCache();
    ASSERT_TRUE(tc::CudaCache::cacheEnabled());
    tc::CudaCache::getCache()->clear();
    ASSERT_EQ(tc::CudaCache::getCache()->size(), 0u);
    ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 0);
    ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 0);

    tc::OptionsCache::enableCache();
    ASSERT_TRUE(tc::OptionsCache::cacheEnabled());
    tc::OptionsCache::getCache()->clear();
    ASSERT_EQ(tc::OptionsCache::getCache()->size(), 0u);
    ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 0u);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 0);

    tc::ManualCudaCache::enableCache();
    ASSERT_TRUE(tc::ManualCudaCache::cacheEnabled());
    tc::ManualCudaCache::getCache()->clear();
    ASSERT_EQ(tc::ManualCudaCache::getCache()->size(), 0u);
    ASSERT_EQ(tc::ManualCudaCache::getCache()->numberAttemptedRetrievals, 0);
    ASSERT_EQ(tc::ManualCudaCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::ManualCudaCache::getCache()->numberCacheAttemps, 0);

    // test0.Run();
    test1.Run();
    test2.Run();

    ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
    ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 2);
    ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);

    ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
    ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
    ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 2);
  }

  void TearDown() {
    tc::CudaCache::disableCache();
    ASSERT_FALSE(tc::CudaCache::cacheEnabled());
    tc::OptionsCache::disableCache();
    ASSERT_FALSE(tc::OptionsCache::cacheEnabled());
  }

  // FCReluTester test0{8, 16, 16};
  MatMulTester test1{8, 32, 16};
  ConvolutionTester test2{1, 1, 1, 2, 2, 1, 1};
};

TEST_F(CompilationCacheTest, ExpectQuerySuccess) {
  // FCReluTester test0{8, 16, 16};
  // test0.Run();

  MatMulTester test1{8, 32, 16};
  test1.Run();

  ConvolutionTester test2{1, 1, 1, 2, 2, 1, 1};
  test2.Run();

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 4);
}

TEST_F(CompilationCacheTest, ExpectQuerySuccessConcurrent) {
  /*
   *  auto fut0 = std::async(std::launch::async, []() {
   *    FCReluTester test0{8, 16, 16};
   *    test0.Run();
   *  });
   *
   */
  auto fut1 = std::async(std::launch::async, []() {
    MatMulTester test1{8, 32, 16};
    test1.Run();
  });

  auto fut2 = std::async(std::launch::async, []() {
    ConvolutionTester test2{1, 1, 1, 2, 2, 1, 1};
    test2.Run();
  });

  // fut0.get();
  fut1.get();
  fut2.get();

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 2);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 4);
}

TEST_F(CompilationCacheTest, ShapesNotPresentInCache) {
  // FCReluTester test0{10, 16, 16};
  // test0.Run();

  MatMulTester test1{12, 32, 16};
  test1.Run();

  ConvolutionTester test2{2, 1, 1, 2, 2, 1, 1};
  test2.Run();

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 4);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 4);
}
TEST_F(CompilationCacheTest, ShapesNotPresentInCacheConcurrent) {
  // auto fut0 = std::async(std::launch::async, []() {
  // FCReluTester test0{10, 16, 16};
  // test0.Run();
  //});

  auto fut1 = std::async(std::launch::async, []() {
    MatMulTester test1{12, 32, 16};
    test1.Run();
  });

  auto fut2 = std::async(std::launch::async, []() {
    ConvolutionTester test2{2, 1, 1, 2, 2, 1, 1};
    test2.Run();
  });

  // fut0.get();
  fut1.get();
  fut2.get();

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 4);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 4);
}

TEST_F(CompilationCacheTest, ModifyIslOptions) {
  // FCReluTester test0{8, 16, 16};
  // test0.ModifyParameters(
  //{tc::Tile{4}, tc::Block{128}, tc::Grid{1}, tc::Unroll{1}});
  // test0.Run();

  MatMulTester test1{8, 32, 16};
  auto options = tc::CudaMappingOptions::makeMlpCudaMappingOptions()
                     .tile(1, 1, 1)
                     .mapToThreads(2, 2, 2)
                     .mapToBlocks(1, 1, 1)
                     .unroll(1);
  test1.Run(options);

  ConvolutionTester test2{1, 1, 1, 2, 2, 1, 1};
  options = tc::CudaMappingOptions::makeConvolutionCudaMappingOptions()
                .tile(2, 2, 2)
                .mapToThreads(1, 1, 1)
                .mapToBlocks(1, 1)
                .unroll(1);
  test2.Run(options);

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 4);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 4);
}

TEST_F(CompilationCacheTest, ModifyIslOptionsConcurrent) {
  // auto fut0 = std::async(std::launch::async, []() {
  // FCReluTester test0{8, 16, 16};
  // test0.ModifyParameters(
  //{tc::Tile{4}, tc::Block{128}, tc::Grid{1}, tc::Unroll{1}});
  // test0.Run();
  //});

  auto fut1 = std::async(std::launch::async, []() {
    MatMulTester test1{8, 32, 16};
    auto options = tc::CudaMappingOptions::makeMlpCudaMappingOptions()
                       .tile(1, 1, 1)
                       .mapToThreads(2, 2, 2)
                       .mapToBlocks(1, 1, 1)
                       .unroll(1);
    test1.Run(options);
  });

  auto fut2 = std::async(std::launch::async, []() {
    ConvolutionTester test2{1, 1, 1, 2, 2, 1, 1};
    auto options = tc::CudaMappingOptions::makeConvolutionCudaMappingOptions()
                       .tile(2, 2, 2)
                       .mapToThreads(1, 1, 1)
                       .mapToBlocks(1, 1)
                       .unroll(1);
    test2.Run(options);
  });

  // fut0.get();
  fut1.get();
  fut2.get();

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 4u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 4);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 4);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 4u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 4);
}

TEST_F(CompilationCacheTest, Serialization) {
  {
    auto buf = tc::CudaCache::getCache()->toProtobuf();
    tc::CudaCache::loadCacheFromProtobuf(buf);
  }
  {
    auto buf = tc::OptionsCache::getCache()->toProtobuf();
    tc::OptionsCache::loadCacheFromProtobuf(buf);
  }

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 0);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 0);

  /*
   *FCReluTester test0{8, 16, 16};
   *test0.Run();
   */

  MatMulTester test1{8, 32, 16};
  test1.Run();

  ConvolutionTester test2{1, 1, 1, 2, 2, 1, 1};
  test2.Run();

  ASSERT_EQ(tc::CudaCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::CudaCache::getCache()->numberAttemptedRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberSuccessfulRetrievals, 2);
  ASSERT_EQ(tc::CudaCache::getCache()->numberCacheAttemps, 0);

  ASSERT_EQ(tc::OptionsCache::getCache()->size(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->totalSize(), 2u);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberAttemptedRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberSuccessfulRetrievals, 0);
  ASSERT_EQ(tc::OptionsCache::getCache()->numberCacheAttemps, 2);
}

TEST(CompilationCache, ManualInjection) {
  static constexpr auto tc = R"(
def add(float(N) A, float(N) B) -> (output) {
    output(n) = A(n) + B(n)
})";

  tc::ManualCudaCache::enableCache();
  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc);
  std::vector<at::Tensor> outputs;
  std::vector<at::Tensor> inputs{at::CUDA(at::kFloat).rand({100}),
                                 at::CUDA(at::kFloat).rand({100})};

  tc::CudaMappingOptions options =
      tc::CudaMappingOptions::makeNaiveCudaMappingOptions();

  auto tensorsPair = tc::toConstDlpackTensors(inputs);
  tc::ScopeGuard g([&]() { tc::deleteDlmTensors(tensorsPair.second); });
  std::string cudaSource = R"CUDA(
  extern "C"{
__global__ void add100(float* __restrict__ output, const float* __restrict__ A, const float* __restrict B)
{
    int t = threadIdx.x;
    output[t] = A[t] + B[t];
}
}
)CUDA";
  {
    std::vector<const DLTensor*> outputs{tensorsPair.first[0]};

    tc::ManualCudaCache::getCache()->cacheKernel(tc::ManualCudaCachedEntry(
        "add",
        "add100",
        {},
        tc::Grid(std::vector<size_t>{1, 1, 1}),
        tc::Block(std::vector<size_t>{100, 1, 1}),
        tensorsPair.first,
        outputs,
        cudaSource,
        tc::CudaGPUInfo::GPUInfo().GetCudaDeviceStr()));
  }

  auto handle = atCompl.compile("add", inputs, options);
  atCompl.run("add", inputs, outputs, handle, false);

  at::Tensor diff = outputs[0].sub(inputs[0].add(inputs[1]));
  checkRtol(diff, inputs);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
