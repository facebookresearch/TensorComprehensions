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
#include <memory>

#include <gtest/gtest.h>

#include "tc/aten/aten.h"

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/autotuner.h"
#include "tc/autotuner/genetic_search.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/utils/time.h"
#include "tc/lang/canonicalize.h"

#include "test_harness_aten_cuda.h"

using tc::autotune::OptionsCacheKey;
using CudaOptionsCache = tc::autotune::OptionsCache<tc::CudaBackend>;
struct CudaOptionsCacheForTesting : public CudaOptionsCache {
 public:
  // Unsafe equal_range not performed under lock, for testing only
  std::pair<MultiMapType::iterator, MultiMapType::iterator> equal_range(
      const OptionsCacheKey& key) {
    return store_.equal_range(key);
  }
  typename tc::CudaBackend::OptionsCacheProtoType toProtobuf() const {
    return CudaOptionsCache::toProtobuf();
  }
  void fromProtobuf(
      const typename tc::CudaBackend::OptionsCacheProtoType& proto) {
    return CudaOptionsCache::fromProtobuf(proto);
  }
};

std::string backendStr() {
  return tc::CudaGPUInfo::GPUInfo().getCudaDeviceStr();
}

class OptionsCacheTest : public ::testing::Test {
 protected:
  void SetUp() {
    optionsCache = std::shared_ptr<CudaOptionsCacheForTesting>(
        new CudaOptionsCacheForTesting);
    ASSERT_EQ(optionsCache->size(), 0u);
    ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberCacheAttempts, 0u);

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

    outputs.resize(3);
    for (auto& output : outputs) {
      output.ndim = 2;
      output.shape = new int64_t[2];
      output.shape[0] = 5;
      output.shape[1] = 6;
      output.strides = nullptr;
    }
    outputs[1].ndim = 0;
    outputs[2].ndim = 0;
  }

  void TearDown() {
    for (auto& input : inputs) {
      delete[] input.shape;
    }
    for (auto& output : outputs) {
      delete[] output.shape;
    }
  }
  std::vector<DLConstTensor> inputs;
  std::vector<DLTensor> outputs;
  std::shared_ptr<CudaOptionsCacheForTesting> optionsCache;

  std::vector<const DLConstTensor*> makeInputPtrs() const {
    std::vector<const DLConstTensor*> ptrs;
    for (const auto& input : inputs) {
      ptrs.push_back(&input);
    }
    return ptrs;
  }
  std::vector<const DLTensor*> makeOutputPtrs() const {
    std::vector<const DLTensor*> ptrs;
    for (const auto& output : outputs) {
      ptrs.push_back(&output);
    }
    return ptrs;
  }
};

TEST_F(OptionsCacheTest, DifferentIDs) {
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options,
      tc::Duration::fromMicroSeconds(10));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options,
      tc::Duration::fromMicroSeconds(11));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel1"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options,
      tc::Duration::fromMicroSeconds(1));

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel0"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    ASSERT_EQ(optionsCache->count(key), 1u);
    auto range = optionsCache->equal_range(key);
    auto it = range.first;
    ASSERT_EQ(it->second.mappingOptions, options);
    ASSERT_EQ(it->second.runtimes.size(), 2u);
    ASSERT_EQ(it->second.runtimes[0], tc::Duration::fromMicroSeconds(10));
    ASSERT_EQ(it->second.runtimes[1], tc::Duration::fromMicroSeconds(11));
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel1"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    auto it = range.first;
    ASSERT_EQ(optionsCache->count(key), 1u);
    ASSERT_EQ(it->second.mappingOptions, options);
    ASSERT_EQ(it->second.runtimes.size(), 1u);
    ASSERT_EQ(it->second.runtimes[0], tc::Duration::fromMicroSeconds(1));
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel2"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    ASSERT_EQ(optionsCache->count(key), 0u);
  }

  ASSERT_EQ(optionsCache->size(), 2u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 3u);
}

TEST_F(OptionsCacheTest, DifferentOptions) {
  auto options0 = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto options1 = tc::CudaMappingOptions::makeMlpMappingOptions();
  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(1));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(2));

  OptionsCacheKey key{lang::CanonicalTcString("kernel"),
                      tc::makeTensorInfoVector(inputPtrs),
                      tc::makeTensorInfoVector(outputPtrs),
                      backendStr()};
  ASSERT_EQ(optionsCache->count(key), 2u);

  auto range = optionsCache->equal_range(key);
  // unordered_multimap, no apriori knowledge of insertion order
  auto it1 = range.first;
  auto it2 = range.first;
  it2++;
  if (it2->second.mappingOptions == options0) {
    std::swap(it1, it2);
  }
  ASSERT_EQ(it1->second.mappingOptions, options0);
  ASSERT_EQ(it1->second.runtimes.size(), 1u);
  ASSERT_EQ(it1->second.runtimes[0], tc::Duration::fromMicroSeconds(1));

  ASSERT_EQ(it2->second.mappingOptions, options1);
  ASSERT_EQ(it2->second.runtimes.size(), 1u);
  ASSERT_EQ(it2->second.runtimes[0], tc::Duration::fromMicroSeconds(2));

  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 2u);
}

TEST_F(OptionsCacheTest, DifferentInputs) {
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions();
  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options,
      tc::Duration::fromMicroSeconds(1));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options,
      tc::Duration::fromMicroSeconds(2));

  auto s = inputs[0].shape[0];
  inputs[0].shape[0] = 42;

  auto options_ = tc::CudaMappingOptions::makeGroupConvolutionMappingOptions();
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options_,
      tc::Duration::fromMicroSeconds(3));

  {
    inputs[0].shape[0] = s;
    OptionsCacheKey key{lang::CanonicalTcString("kernel"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};

    auto range = optionsCache->equal_range(key);
    auto it = range.first;
    ASSERT_EQ(optionsCache->count(key), 1u);
    ASSERT_EQ(it->second.mappingOptions, options);
    ASSERT_EQ(it->second.runtimes.size(), 2u);
    ASSERT_EQ(it->second.runtimes[0], tc::Duration::fromMicroSeconds(1));
    ASSERT_EQ(it->second.runtimes[1], tc::Duration::fromMicroSeconds(2));
  }

  {
    inputs[0].shape[0] = 42;
    OptionsCacheKey key{lang::CanonicalTcString("kernel"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};

    auto range = optionsCache->equal_range(key);
    auto it = range.first;
    ASSERT_EQ(optionsCache->count(key), 1u);
    ASSERT_EQ(it->second.mappingOptions, options_);
    ASSERT_EQ(it->second.runtimes.size(), 1u);
    ASSERT_EQ(it->second.runtimes[0], tc::Duration::fromMicroSeconds(3));
  }

  {
    inputs[0].shape[0] = 43;
    OptionsCacheKey key{lang::CanonicalTcString("kernel"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    ASSERT_EQ(optionsCache->count(key), 0u);
    ASSERT_EQ(optionsCache->size(), 2u);
    ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberCacheAttempts, 3u);
  }
}

TEST_F(OptionsCacheTest, RetrieveBest) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({3});

  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(1));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(2));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options2,
      tc::Duration::fromMicroSeconds(3));

  auto ret = optionsCache->getTopKOptions(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      1);
  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0], options0);

  ret = optionsCache->getTopKOptions(
      lang::CanonicalTcString("kernelX"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      1);
  ASSERT_EQ(ret.size(), 0u);

  ASSERT_EQ(optionsCache->getKeys().size(), 1u);
  ASSERT_EQ(optionsCache->size(), 3u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 2u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 1u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 3u);
}

TEST_F(OptionsCacheTest, RetrieveTopK) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({3});
  auto options3 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({4});
  auto options4 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({5});

  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(3));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(2));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options2,
      tc::Duration::fromMicroSeconds(1));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options3,
      tc::Duration::fromMicroSeconds(4));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options4,
      tc::Duration::fromMicroSeconds(5));

  auto ret = optionsCache->getTopKOptions(
      lang::CanonicalTcString("kernelX"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      3);
  ASSERT_EQ(ret.size(), 0u);

  ret = optionsCache->getTopKOptions(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      3);
  ASSERT_EQ(ret.size(), 3u);
  ASSERT_EQ(ret[0], options2);
  ASSERT_EQ(ret[1], options1);
  ASSERT_EQ(ret[2], options0);

  ret = optionsCache->getTopKOptions(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      6);
  ASSERT_EQ(ret.size(), 5u);
  ASSERT_EQ(ret[0], options2);
  ASSERT_EQ(ret[1], options1);
  ASSERT_EQ(ret[2], options0);
  ASSERT_EQ(ret[3], options3);
  ASSERT_EQ(ret[4], options4);

  ASSERT_EQ(optionsCache->getKeys().size(), 1u);
  ASSERT_EQ(optionsCache->size(), 5u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 3u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 2u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 5u);
}

TEST_F(OptionsCacheTest, KeepOnlyBestCandidates) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({2});
  auto options2 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({3});
  auto options3 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({4});
  auto options4 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({5});

  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(3));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(2));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options2,
      tc::Duration::fromMicroSeconds(4));

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel1"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options2,
      tc::Duration::fromMicroSeconds(4));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel1"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options3,
      tc::Duration::fromMicroSeconds(1));

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel2"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options4,
      tc::Duration::fromMicroSeconds(5));

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel3"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(2));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel3"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(6));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel3"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options3,
      tc::Duration::fromMicroSeconds(5));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel3"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options4,
      tc::Duration::fromMicroSeconds(1));

  {
    ASSERT_EQ(optionsCache->size(), 10u);
    optionsCache->pruneKeepTopK(2);
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel0"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    auto it1 = range.first;
    auto it2 = range.first;
    it2++;
    if (it2->second.mappingOptions == options0) {
      std::swap(it1, it2);
    }
    ASSERT_EQ(optionsCache->count(key), 2u);
    ASSERT_EQ(it1->second.mappingOptions, options0);
    ASSERT_EQ(it2->second.mappingOptions, options1);
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel1"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    auto it1 = range.first;
    auto it2 = range.first;
    it2++;
    if (it2->second.mappingOptions == options2) {
      std::swap(it1, it2);
    }
    ASSERT_EQ(optionsCache->count(key), 2u);
    ASSERT_EQ(it1->second.mappingOptions, options2);
    ASSERT_EQ(it2->second.mappingOptions, options3);
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel2"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    auto it = range.first;
    ASSERT_EQ(optionsCache->count(key), 1u);
    ASSERT_EQ(it->second.mappingOptions, options4);
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel3"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    auto it1 = range.first;
    auto it2 = range.first;
    it2++;
    if (it2->second.mappingOptions == options0) {
      std::swap(it1, it2);
    }
    ASSERT_EQ(optionsCache->count(key), 2u);
    ASSERT_EQ(it1->second.mappingOptions, options0);
    ASSERT_EQ(it2->second.mappingOptions, options4);
  }

  ASSERT_EQ(optionsCache->getKeys().size(), 4u);
  ASSERT_EQ(optionsCache->size(), 7u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 10u);
}

TEST_F(OptionsCacheTest, RetrieveBestMedianTime) {
  auto options0 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({1});
  auto options1 =
      tc::CudaMappingOptions::makeNaiveMappingOptions().mapToBlocks({2});

  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(1));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(9));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(10));

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(8));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(8));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(8));

  auto ret = optionsCache->getTopKOptions(
      lang::CanonicalTcString("kernel"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      1);
  ASSERT_EQ(ret.size(), 1u);
  ASSERT_EQ(ret[0], options1);

  ASSERT_EQ(optionsCache->getKeys().size(), 1u);
  ASSERT_EQ(optionsCache->size(), 2u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 1u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 1u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 6u);
}

TEST_F(OptionsCacheTest, Serialization) {
  auto options0 = tc::CudaMappingOptions::makeNaiveMappingOptions().tile(0);
  auto options1 = tc::CudaMappingOptions::makeNaiveMappingOptions().tile(1);
  auto inputPtrs = makeInputPtrs();
  auto outputPtrs = makeOutputPtrs();

  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(10));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel0"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options1,
      tc::Duration::fromMicroSeconds(11));
  optionsCache->recordRuntime(
      lang::CanonicalTcString("kernel1"),
      tc::makeTensorInfoVector(inputPtrs),
      tc::makeTensorInfoVector(outputPtrs),
      backendStr(),
      options0,
      tc::Duration::fromMicroSeconds(1));

  auto buf = optionsCache->toProtobuf();
  optionsCache->clear();
  optionsCache->fromProtobuf(buf);

  ASSERT_EQ(optionsCache->getKeys().size(), 2u);
  ASSERT_EQ(optionsCache->size(), 3u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 0u);

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel0"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    // unordered_multimap, no a-priori knowledge of insertion order
    auto it1 = range.first;
    auto it2 = range.first;
    it2++;
    if (it2->second.mappingOptions == options0) {
      std::swap(it1, it2);
    }
    ASSERT_EQ(optionsCache->count(key), 2u);
    ASSERT_EQ(it1->second.mappingOptions, options0);
    ASSERT_EQ(it1->second.runtimes.size(), 1u);
    ASSERT_EQ(it1->second.runtimes[0].toMicroSeconds(), 10u);

    ASSERT_EQ(it2->second.mappingOptions, options1);
    ASSERT_EQ(it2->second.runtimes.size(), 1u);
    ASSERT_EQ(it2->second.runtimes[0].toMicroSeconds(), 11u);
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel1"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    auto range = optionsCache->equal_range(key);
    auto it = range.first;
    ASSERT_EQ(optionsCache->count(key), 1u);
    ASSERT_EQ(it->second.mappingOptions, options0);
    ASSERT_EQ(it->second.runtimes.size(), 1u);
    ASSERT_EQ(it->second.runtimes[0].toMicroSeconds(), 1u);
  }

  {
    OptionsCacheKey key{lang::CanonicalTcString("kernel2"),
                        tc::makeTensorInfoVector(inputPtrs),
                        tc::makeTensorInfoVector(outputPtrs),
                        backendStr()};
    ASSERT_EQ(optionsCache->count(key), 0u);
    ASSERT_EQ(optionsCache->getKeys().size(), 2u);
    ASSERT_EQ(optionsCache->size(), 3u);
    ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberCacheAttempts, 0u);
  }
}

class MatMulTester {
 public:
  MatMulTester(
      std::shared_ptr<CudaOptionsCacheForTesting> optionsCache,
      int N,
      int M,
      int B)
      : inputs_{at::CUDA(at::kFloat).rand({N, M}),
                at::CUDA(at::kFloat).rand({M, B})},
        M{M},
        optionsCache_{optionsCache} {}
  void Run(
      tc::CudaMappingOptions options =
          tc::CudaMappingOptions::makeMlpMappingOptions()) {
    auto pExecutor =
        tc::aten::compile<tc::CudaBackend>(tc_, "matmul", inputs_, options);
    auto outputs_ = tc::aten::prepareOutputs(tc_, "matmul", inputs_);
    auto timings = tc::aten::profile(*pExecutor, inputs_, outputs_);

    auto inputDLTensors = tc::aten::makeDLConstTensors(inputs_);
    auto outputDLTensors = tc::aten::makeDLTensors(outputs_);
    optionsCache_->recordRuntime(
        lang::canonicalTc(tc_),
        tc::makeTensorInfoVector(tc::extractRawPtrs(inputDLTensors)),
        tc::makeTensorInfoVector(tc::extractRawPtrs(outputDLTensors)),
        backendStr(),
        options,
        timings.kernelRuntime);

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
  std::shared_ptr<CudaOptionsCacheForTesting> optionsCache_;
};

class ConvolutionTester {
 public:
  ConvolutionTester(
      std::shared_ptr<CudaOptionsCacheForTesting> optionsCache,
      int N,
      int C,
      int H,
      int W,
      int O,
      int KH,
      int KW)
      : inputs_{at::CUDA(at::kFloat).rand({N, C, H, W}),
                at::CUDA(at::kFloat).rand({O, C, KH, KW}),
                at::CUDA(at::kFloat).rand({O})},
        C{C},
        KH{KH},
        KW{KW},
        optionsCache_{optionsCache} {}
  void Run(
      tc::CudaMappingOptions options =
          tc::CudaMappingOptions::makeConvolutionMappingOptions()) {
    auto pExecutor = tc::aten::compile<tc::CudaBackend>(
        tc_, "convolution", inputs_, options);
    auto outputs_ = tc::aten::prepareOutputs(tc_, "convolution", inputs_);
    auto timings = tc::aten::profile(*pExecutor, inputs_, outputs_);

    auto inputDLTensors = tc::aten::makeDLConstTensors(inputs_);
    auto outputDLTensors = tc::aten::makeDLTensors(outputs_);
    optionsCache_->recordRuntime(
        lang::canonicalTc(tc_),
        tc::makeTensorInfoVector(tc::extractRawPtrs(inputDLTensors)),
        tc::makeTensorInfoVector(tc::extractRawPtrs(outputDLTensors)),
        backendStr(),
        options,
        timings.kernelRuntime);

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
  std::shared_ptr<CudaOptionsCacheForTesting> optionsCache_;
};

class CompilationCacheTest : public ::testing::Test {
 public:
  CompilationCacheTest()
      : optionsCache{new CudaOptionsCacheForTesting()},
        test1{optionsCache, 8, 32, 16},
        test2{optionsCache, 1, 1, 1, 2, 2, 1, 1} {}

 protected:
  void SetUp() {
    optionsCache->clear();
    ASSERT_EQ(optionsCache->size(), 0u);
    ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberCacheAttempts, 0u);

    test1.Run();
    test2.Run();

    ASSERT_EQ(optionsCache->size(), 2u);
    ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberCacheAttempts, 2u);
  }

  std::shared_ptr<CudaOptionsCacheForTesting> optionsCache;
  MatMulTester test1;
  ConvolutionTester test2;
};

TEST_F(CompilationCacheTest, ExpectQuerySuccess) {
  MatMulTester test1{optionsCache, 8, 32, 16};
  test1.Run();

  ConvolutionTester test2{optionsCache, 1, 1, 1, 2, 2, 1, 1};
  test2.Run();

  ASSERT_EQ(optionsCache->size(), 2u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 4u);
}

TEST_F(CompilationCacheTest, ExpectQuerySuccessConcurrent) {
  auto fut1 = std::async(std::launch::async, [this]() {
    MatMulTester test1{this->optionsCache, 8, 32, 16};
    test1.Run();
  });

  auto fut2 = std::async(std::launch::async, [this]() {
    ConvolutionTester test2{this->optionsCache, 1, 1, 1, 2, 2, 1, 1};
    test2.Run();
  });

  fut1.get();
  fut2.get();

  ASSERT_EQ(optionsCache->size(), 2u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 4u);
}

TEST_F(CompilationCacheTest, ShapesNotPresentInCache) {
  MatMulTester test1{optionsCache, 12, 32, 16};
  test1.Run();

  ConvolutionTester test2{optionsCache, 2, 1, 1, 2, 2, 1, 1};
  test2.Run();

  ASSERT_EQ(optionsCache->size(), 4u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 4u);
}

TEST_F(CompilationCacheTest, ShapesNotPresentInCacheConcurrent) {
  auto fut1 = std::async(std::launch::async, [this]() {
    MatMulTester test1{this->optionsCache, 12, 32, 16};
    test1.Run();
  });

  auto fut2 = std::async(std::launch::async, [this]() {
    ConvolutionTester test2{this->optionsCache, 2, 1, 1, 2, 2, 1, 1};
    test2.Run();
  });

  fut1.get();
  fut2.get();

  ASSERT_EQ(optionsCache->size(), 4u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 4u);
}

TEST_F(CompilationCacheTest, ModifyIslOptions) {
  MatMulTester test1{optionsCache, 8, 32, 16};
  auto options = tc::CudaMappingOptions::makeMlpMappingOptions()
                     .tile(1, 1, 1)
                     .mapToThreads(2, 2, 2)
                     .mapToBlocks(1, 1, 1)
                     .unroll(1);
  test1.Run(options);

  ConvolutionTester test2{optionsCache, 1, 1, 1, 2, 2, 1, 1};
  options = tc::CudaMappingOptions::makeConvolutionMappingOptions()
                .tile(2, 2, 2)
                .mapToThreads(1, 1, 1)
                .mapToBlocks(1, 1)
                .unroll(1);
  test2.Run(options);

  ASSERT_EQ(optionsCache->getKeys().size(), 2u);
  ASSERT_EQ(optionsCache->size(), 4u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 4u);
}

TEST_F(CompilationCacheTest, ModifyIslOptionsConcurrent) {
  auto fut1 = std::async(std::launch::async, [this]() {
    MatMulTester test1{this->optionsCache, 8, 32, 16};
    auto options = tc::CudaMappingOptions::makeMlpMappingOptions()
                       .tile(1, 1, 1)
                       .mapToThreads(2, 2, 2)
                       .mapToBlocks(1, 1, 1)
                       .unroll(1);
    test1.Run(options);
  });

  auto fut2 = std::async(std::launch::async, [this]() {
    ConvolutionTester test2{this->optionsCache, 1, 1, 1, 2, 2, 1, 1};
    auto options = tc::CudaMappingOptions::makeConvolutionMappingOptions()
                       .tile(2, 2, 2)
                       .mapToThreads(1, 1, 1)
                       .mapToBlocks(1, 1)
                       .unroll(1);
    test2.Run(options);
  });

  fut1.get();
  fut2.get();

  ASSERT_EQ(optionsCache->getKeys().size(), 2u);
  ASSERT_EQ(optionsCache->size(), 4u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 4u);
}

TEST_F(CompilationCacheTest, Serialization) {
  {
    auto buf = optionsCache->toProtobuf();
    optionsCache->clear();
    optionsCache->fromProtobuf(buf);
    ASSERT_EQ(optionsCache->getKeys().size(), 2u);
    ASSERT_EQ(optionsCache->size(), 2u);
    ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
    ASSERT_EQ(optionsCache->numberCacheAttempts, 0u);
  }

  MatMulTester test1{optionsCache, 8, 32, 16};
  test1.Run();

  ConvolutionTester test2{optionsCache, 1, 1, 1, 2, 2, 1, 1};
  test2.Run();

  ASSERT_EQ(optionsCache->size(), 2u);
  ASSERT_EQ(optionsCache->numberAttemptedRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberSuccessfulRetrievals, 0u);
  ASSERT_EQ(optionsCache->numberCacheAttempts, 2u);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
